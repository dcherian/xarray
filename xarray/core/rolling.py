import functools
import itertools
import warnings
from typing import Any, Callable, Dict

import numpy as np

from . import dtypes, duck_array_ops, utils
from .arithmetic import CoarsenArithmetic
from .options import _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import either_dict_or_kwargs

try:
    import bottleneck
except ImportError:
    # use numpy methods instead
    bottleneck = None


_ROLLING_REDUCE_DOCSTRING_TEMPLATE = """\
Reduce this object's data windows by applying `{name}` along its dimension.

Parameters
----------
keep_attrs : bool, default: None
    If True, the attributes (``attrs``) will be copied from the original
    object to the new one. If False, the new object will be returned
    without attributes. If None uses the global default.
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
reduced : same type as caller
    New object with `{name}` applied along its rolling dimnension.
"""


class Rolling:
    """A object that implements the moving window pattern.

    See Also
    --------
    xarray.Dataset.groupby
    xarray.DataArray.groupby
    xarray.Dataset.rolling
    xarray.DataArray.rolling
    """

    __slots__ = ("obj", "window", "min_periods", "center", "pad", "dim", "keep_attrs")
    _attributes = ("window", "min_periods", "center", "pad", "dim", "keep_attrs")

    def __init__(
        self, obj, windows, min_periods=None, center=False, pad=True, keep_attrs=None
    ):
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool or mapping of hashable to bool, default: False
            Set the labels at the center of the window.
        pad : bool or mapping of hashable to bool, default: True
            Pad the sides of the rolling window with ``NaN``.  For different
            padding, see ``DataArray.pad`` or ``Dataset.pad``.

        Returns
        -------
        rolling : type of input argument
        """
        self.dim, self.window = [], []
        for d, w in windows.items():
            self.dim.append(d)
            if w <= 0:
                raise ValueError("window must be > 0")
            self.window.append(w)

        self.center = self._mapping_to_list(center, default=False)
        self.pad = self._mapping_to_list(pad, default=True)
        self.obj = obj

        # attributes
        if min_periods is not None and min_periods <= 0:
            raise ValueError("min_periods must be greater than zero or None")

        self.min_periods = np.prod(self.window) if min_periods is None else min_periods

        if keep_attrs is not None:
            warnings.warn(
                "Passing ``keep_attrs`` to ``rolling`` is deprecated and will raise an"
                " error in xarray 0.18. Please pass ``keep_attrs`` directly to the"
                " applied function. Note that keep_attrs is now True per default.",
                FutureWarning,
            )
        self.keep_attrs = keep_attrs

    def __repr__(self):
        """provide a nice str repr of our rolling object"""

        attrs = [
            "{k}->{v}{c}{p}".format(
                k=k, v=w, c="(center)" if c else "", p="(no pad)" if not p else ""
            )
            for k, w, c, p in zip(self.dim, self.window, self.center, self.pad)
        ]
        return "{klass} [{attrs}]".format(
            klass=self.__class__.__name__, attrs=",".join(attrs)
        )

    def __len__(self):
        return self.obj.sizes[self.dim]

    def _reduce_method(  # type: ignore[misc]
        name: str, fillna, rolling_agg_func: Callable = None
    ) -> Callable:
        """Constructs reduction methods built on a numpy reduction function (e.g. sum),
        a bottleneck reduction function (e.g. move_sum), or a Rolling reduction (_mean)."""
        if rolling_agg_func:
            array_agg_func = None
        else:
            array_agg_func = getattr(duck_array_ops, name)

        bottleneck_move_func = getattr(bottleneck, "move_" + name, None)

        def method(self, keep_attrs=None, **kwargs):

            keep_attrs = self._get_keep_attrs(keep_attrs)

            return self._numpy_or_bottleneck_reduce(
                array_agg_func,
                bottleneck_move_func,
                rolling_agg_func,
                keep_attrs=keep_attrs,
                fillna=fillna,
                **kwargs,
            )

        method.__name__ = name
        method.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name=name)
        return method

    def _mean(self, keep_attrs, **kwargs):
        result = self.sum(keep_attrs=False, **kwargs) / self.count(keep_attrs=False)
        if keep_attrs:
            result.attrs = self.obj.attrs
        return result

    _mean.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name="mean")

    argmax = _reduce_method("argmax", dtypes.NINF)
    argmin = _reduce_method("argmin", dtypes.INF)
    max = _reduce_method("max", dtypes.NINF)
    min = _reduce_method("min", dtypes.INF)
    prod = _reduce_method("prod", 1)
    sum = _reduce_method("sum", 0)
    mean = _reduce_method("mean", None, _mean)
    std = _reduce_method("std", None)
    var = _reduce_method("var", None)
    median = _reduce_method("median", None)

    def count(self, keep_attrs=None):
        keep_attrs = self._get_keep_attrs(keep_attrs)
        rolling_count = self._counts(keep_attrs=keep_attrs)
        enough_periods = rolling_count >= self.min_periods
        return rolling_count.where(enough_periods)

    count.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name="count")

    def _mapping_to_list(
        self, arg, default=None, allow_default=True, allow_allsame=True
    ):
        if utils.is_dict_like(arg):
            if allow_default:
                return [arg.get(d, default) for d in self.dim]
            for d in self.dim:
                if d not in arg:
                    raise KeyError(f"argument has no key {d}.")
            return [arg[d] for d in self.dim]
        elif allow_allsame:  # for single argument
            return [arg] * len(self.dim)
        elif len(self.dim) == 1:
            return [arg]
        else:
            raise ValueError(
                "Mapping argument is necessary for {}d-rolling.".format(len(self.dim))
            )

    def _get_keep_attrs(self, keep_attrs):

        if keep_attrs is None:
            # TODO: uncomment the next line and remove the others after the deprecation
            # keep_attrs = _get_keep_attrs(default=True)

            if self.keep_attrs is None:
                keep_attrs = _get_keep_attrs(default=True)
            else:
                keep_attrs = self.keep_attrs

        return keep_attrs

    def _get_output_coords(self, all_dims=False) -> Dict[str, Any]:
        """Get output coordinates, taking into account window size, window, centering, and padding.

        If any of the dimensions are not padded, the output size can be shorter than the input size
        along that dimension, so we need to shorten and properly label the corresponding coordinates.

        If `all_dims` is False, returns coordinates only for the dimension(s) used for the rolling
        window.  This is most useful if the coordinates will be used in a `da.sel()` call.

        If `all_dims` is True, returns all coordinates.  This is most useful for constructing a new
        DataArray or Dataset, where the data has already been constructed to be the correct size
        along each dimension.
        """

        coords = self.obj.coords
        dim = list(self.obj.coords) if all_dims else self.dim
        window = self.window
        center = self.center
        pad = self.pad

        # Dimensions which require offsets are those which are not padded, but the logic to determine
        # the offset is very similar to determining padding sizes.
        # So, we invert the `pad` flag(s), call `get_pads()`, and work from there.

        if pad is False:
            offset = [True]
        elif utils.is_list_like(pad):
            offset = [not p for p in pad]

        offset_pads = utils.get_pads(self.dim, window, center, offset)

        output_coords: Dict[str, Any] = {}

        def get_offsets(d: str):
            pad_start_offset, pad_end_offset = offset_pads[d]

            start_offset = None if not pad_start_offset else pad_start_offset
            end_offset = None if not pad_end_offset else -pad_end_offset

            return start_offset, end_offset

        for d in dim:
            obj_coords = coords[d]
            if d in self.dim:
                start_offset, end_offset = get_offsets(d)
                obj_coords = obj_coords.isel({d: slice(start_offset, end_offset)})
            else:
                for obj_d in obj_coords.dims:
                    if obj_d in self.dim:
                        start_offset, end_offset = get_offsets(obj_d)
                        obj_coords = obj_coords.isel(
                            {obj_d: slice(start_offset, end_offset)}
                        )

            output_coords[d] = obj_coords

        return output_coords


class DataArrayRolling(Rolling):
    __slots__ = ("window_labels",)

    def __init__(
        self, obj, windows, min_periods=None, center=False, pad=True, keep_attrs=None
    ):
        """
        Moving window object for DataArray.
        You should use DataArray.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool or mapping of hashable to bool, default: False
            Set the labels at the center of the window.
        pad : bool or mapping of hashable to bool, default: True
            Pad the sides of the rolling window with ``NaN``.  For different
            padding, see ``DataArray.pad``.

        Returns
        -------
        rolling : type of input argument

        See Also
        --------
        xarray.DataArray.rolling
        xarray.DataArray.groupby
        xarray.Dataset.rolling
        xarray.Dataset.groupby
        """
        super().__init__(
            obj,
            windows,
            min_periods=min_periods,
            center=center,
            pad=pad,
            keep_attrs=keep_attrs,
        )

        # TODO legacy attribute
        self.window_labels = self.obj[self.dim[0]]

    def __iter__(self):
        if len(self.dim) > 1:
            raise ValueError("__iter__ is only supported for 1d-rolling")
        dim = self.dim[0]
        center = self.center[0]
        pad = self.pad[0]
        window = self.window[0]
        center_offset = window // 2 if center else 0

        pads = utils.get_pads(self.dim, self.window, self.center, self.pad)
        start_pad, end_pad = pads[dim]

        # Select the proper subset of labels, based on whether or not to center and/or pad
        first_label_idx = 0 if pad else center_offset if center else window - 1
        last_label_idx = (
            len(self.obj[dim])
            if pad or not center
            else len(self.obj[dim]) - center_offset
        )

        labels = (
            self.obj[dim][slice(first_label_idx, last_label_idx)]
            if self.obj[dim].coords
            else np.arange(last_label_idx - first_label_idx)
        )

        padded_obj = self.obj.pad(pads, mode="constant", constant_values=dtypes.NA)

        if pad and not center:
            first_stop = 1
            last_stop = len(self.obj[dim])
        elif pad and center:
            first_stop = end_pad + 1
            last_stop = len(self.obj[dim]) + end_pad
        elif not pad:
            first_stop = window
            last_stop = len(self.obj[dim])

        # These are indicies into the padded array, so we need to add start_pad
        stops = np.arange(first_stop, last_stop + 1) + start_pad
        starts = stops - window

        for (label, start, stop) in zip(labels, starts, stops):
            window = padded_obj.isel({self.dim[0]: slice(start, stop)})

            counts = window.count(dim=self.dim[0])
            window = window.where(counts >= self.min_periods)

            yield (label, window)

    def construct(
        self,
        window_dim=None,
        stride=1,
        fill_value=dtypes.NA,
        keep_attrs=None,
        **window_dim_kwargs,
    ):
        """
        Convert this rolling object to xr.DataArray,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim : str or mapping, optional
            A mapping from dimension name to the new window dimension names.
        stride : int or mapping of int, default: 1
            Size of stride for the rolling window.
        fill_value : default: dtypes.NA
            Filling value to match the dimension size.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **window_dim_kwargs : {dim: new_name, ...}, optional
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        DataArray that is a view of the original array. The returned array is
        not writeable.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))

        >>> rolling = da.rolling(b=3)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])
        Dimensions without coordinates: a, b, window_dim

        >>> rolling = da.rolling(b=3, center=True)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.],
                [ 2.,  3., nan]],
        <BLANKLINE>
               [[nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.],
                [ 6.,  7., nan]]])
        Dimensions without coordinates: a, b, window_dim

        """

        return self._construct(
            self.obj,
            window_dim=window_dim,
            stride=stride,
            fill_value=fill_value,
            keep_attrs=keep_attrs,
            **window_dim_kwargs,
        )

    def _construct(
        self,
        obj,
        window_dim=None,
        stride=1,
        fill_value=dtypes.NA,
        keep_attrs=None,
        **window_dim_kwargs,
    ):
        from .dataarray import DataArray

        keep_attrs = self._get_keep_attrs(keep_attrs)

        if window_dim is None:
            if len(window_dim_kwargs) == 0:
                raise ValueError(
                    "Either window_dim or window_dim_kwargs need to be specified."
                )
            window_dim = {d: window_dim_kwargs[d] for d in self.dim}

        window_dim = self._mapping_to_list(
            window_dim, allow_default=False, allow_allsame=False
        )
        stride = self._mapping_to_list(stride, default=1)

        window = obj.variable.rolling_window(
            self.dim,
            self.window,
            window_dim,
            self.center,
            self.pad,
            fill_value=fill_value,
        )

        attrs = obj.attrs if keep_attrs else {}
        coords = self._get_output_coords(all_dims=True)

        result = DataArray(
            window,
            dims=obj.dims + tuple(window_dim),
            coords=coords,
            attrs=attrs,
            name=obj.name,
        )
        return result.isel(
            **{d: slice(None, None, s) for d, s in zip(self.dim, stride)}
        )

    def reduce(self, func, keep_attrs=None, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, **kwargs)` to return the result of collapsing an
            np.ndarray over an the rolling dimension.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))
        >>> rolling = da.rolling(b=3)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])
        Dimensions without coordinates: a, b, window_dim

        >>> rolling.reduce(np.sum)
        <xarray.DataArray (a: 2, b: 4)>
        array([[nan, nan,  3.,  6.],
               [nan, nan, 15., 18.]])
        Dimensions without coordinates: a, b

        >>> rolling = da.rolling(b=3, min_periods=1)
        >>> rolling.reduce(np.nansum)
        <xarray.DataArray (a: 2, b: 4)>
        array([[ 0.,  1.,  3.,  6.],
               [ 4.,  9., 15., 18.]])
        Dimensions without coordinates: a, b
        """

        keep_attrs = self._get_keep_attrs(keep_attrs)

        rolling_dim = {
            d: utils.get_temp_dimname(self.obj.dims, f"_rolling_dim_{d}")
            for d in self.dim
        }

        # save memory with reductions GH4325
        fillna = kwargs.pop("fillna", dtypes.NA)
        if fillna is not dtypes.NA:
            obj = self.obj.fillna(fillna)
        else:
            obj = self.obj
        windows = self._construct(
            obj, rolling_dim, keep_attrs=keep_attrs, fill_value=fillna
        )

        result = windows.reduce(
            func, dim=list(rolling_dim.values()), keep_attrs=keep_attrs, **kwargs
        )

        # Find valid windows based on count.
        counts = self._counts(keep_attrs=False)
        return result.where(counts >= self.min_periods)

    def _counts(self, keep_attrs):
        """Number of non-nan entries in each rolling window."""

        rolling_dim = {
            d: utils.get_temp_dimname(self.obj.dims, f"_rolling_dim_{d}")
            for d in self.dim
        }
        # We use False as the fill_value instead of np.nan, since boolean
        # array is faster to be reduced than object array.
        # The use of skipna==False is also faster since it does not need to
        # copy the strided array.
        output_dim_coords = self._get_output_coords()
        counts = (
            self.obj.notnull(keep_attrs=keep_attrs)
            .rolling(
                center={d: self.center[i] for i, d in enumerate(self.dim)},
                pad={p: self.pad[i] for i, p in enumerate(self.pad)},
                **{d: w for d, w in zip(self.dim, self.window)},
            )
            .construct(rolling_dim, fill_value=False, keep_attrs=keep_attrs)
            .sum(dim=list(rolling_dim.values()), skipna=False, keep_attrs=keep_attrs)
        ).sel(output_dim_coords)
        return counts

    def _bottleneck_reduce(self, func, keep_attrs, **kwargs):
        from .dataarray import DataArray

        # bottleneck doesn't allow min_count to be 0, although it should
        # work the same as if min_count = 1
        # Note bottleneck only works with 1d-rolling.
        if self.min_periods is not None and self.min_periods == 0:
            min_count = 1
        else:
            min_count = self.min_periods

        axis = self.obj.get_axis_num(self.dim[0])

        padded = self.obj.variable
        if self.center[0]:
            if is_duck_dask_array(padded.data):
                # workaround to make the padded chunk size larger than
                # self.window - 1
                shift = -(self.window[0] + 1) // 2
                offset = (self.window[0] - 1) // 2
                valid = (slice(None),) * axis + (
                    slice(offset, offset + self.obj.shape[axis]),
                )
            else:
                shift = (-self.window[0] // 2) + 1
                valid = (slice(None),) * axis + (slice(-shift, None),)
            padded = padded.pad({self.dim[0]: (0, -shift)}, mode="constant")

        if is_duck_dask_array(padded.data):
            raise AssertionError("should not be reachable")
        else:
            values = func(
                padded.data, window=self.window[0], min_count=min_count, axis=axis
            )

        if self.center[0]:
            values = values[valid]

        attrs = self.obj.attrs if keep_attrs else {}
        output_dim_coords = self._get_output_coords()

        return DataArray(values, self.obj.coords, attrs=attrs, name=self.obj.name).sel(
            output_dim_coords
        )

    def _numpy_or_bottleneck_reduce(
        self,
        array_agg_func,
        bottleneck_move_func,
        rolling_agg_func,
        keep_attrs,
        fillna,
        **kwargs,
    ):
        if "dim" in kwargs:
            warnings.warn(
                f"Reductions are applied along the rolling dimension(s) "
                f"'{self.dim}'. Passing the 'dim' kwarg to reduction "
                f"operations has no effect.",
                DeprecationWarning,
                stacklevel=3,
            )
            del kwargs["dim"]

        if (
            bottleneck_move_func is not None
            and not is_duck_dask_array(self.obj.data)
            and len(self.dim) == 1
        ):
            # TODO: renable bottleneck with dask after the issues
            # underlying https://github.com/pydata/xarray/issues/2940 are
            # fixed.
            return self._bottleneck_reduce(
                bottleneck_move_func, keep_attrs=keep_attrs, **kwargs
            )
        if rolling_agg_func:
            return rolling_agg_func(self, keep_attrs=self._get_keep_attrs(keep_attrs))
        if fillna is not None:
            if fillna is dtypes.INF:
                fillna = dtypes.get_pos_infinity(self.obj.dtype, max_for_int=True)
            elif fillna is dtypes.NINF:
                fillna = dtypes.get_neg_infinity(self.obj.dtype, min_for_int=True)
            kwargs.setdefault("skipna", False)
            kwargs.setdefault("fillna", fillna)

        return self.reduce(array_agg_func, keep_attrs=keep_attrs, **kwargs)


class DatasetRolling(Rolling):
    __slots__ = ("rollings",)

    def __init__(
        self, obj, windows, min_periods=None, center=False, pad=True, keep_attrs=None
    ):
        """
        Moving window object for Dataset.
        You should use Dataset.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : Dataset
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool or mapping of hashable to bool, default: False
            Set the labels at the center of the window.
        pad : bool or mapping of hashable to bool, default: True
            Pad the sides of the window with ``NaN``.  For different
            padding, see ``Dataset.pad``.

        Returns
        -------
        rolling : type of input argument

        See Also
        --------
        xarray.Dataset.rolling
        xarray.DataArray.rolling
        xarray.Dataset.groupby
        xarray.DataArray.groupby
        """
        super().__init__(obj, windows, min_periods, center, pad, keep_attrs)
        if any(d not in self.obj.dims for d in self.dim):
            raise KeyError(self.dim)
        # Keep each Rolling object as a dictionary
        self.rollings = {}
        for key, da in self.obj.data_vars.items():
            # keeps rollings only for the dataset depending on self.dim
            dims, center, pad = [], {}, {}
            for i, d in enumerate(self.dim):
                if d in da.dims:
                    dims.append(d)
                    center[d] = self.center[i]
                    pad[d] = self.pad[i]

            if dims:
                w = {d: windows[d] for d in dims}
                self.rollings[key] = DataArrayRolling(da, w, min_periods, center, pad)

    def _dataset_implementation(self, func, keep_attrs, **kwargs):
        from .dataset import Dataset

        keep_attrs = self._get_keep_attrs(keep_attrs)

        reduced = {}
        for key, da in self.obj.data_vars.items():
            if any(d in da.dims for d in self.dim):
                reduced[key] = func(self.rollings[key], keep_attrs=keep_attrs, **kwargs)
            else:
                reduced[key] = self.obj[key].copy()
                # we need to delete the attrs of the copied DataArray
                if not keep_attrs:
                    reduced[key].attrs = {}

        attrs = self.obj.attrs if keep_attrs else {}
        coords = self._get_output_coords(all_dims=True)

        return Dataset(reduced, coords=coords, attrs=attrs)

    def reduce(self, func, keep_attrs=None, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, **kwargs)` to return the result of collapsing an
            np.ndarray over an the rolling dimension.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data.
        """
        return self._dataset_implementation(
            functools.partial(DataArrayRolling.reduce, func=func),
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def _counts(self, keep_attrs):
        return self._dataset_implementation(
            DataArrayRolling._counts, keep_attrs=keep_attrs
        )

    def _numpy_or_bottleneck_reduce(
        self,
        array_agg_func,
        bottleneck_move_func,
        rolling_agg_func,
        keep_attrs,
        **kwargs,
    ):
        return self._dataset_implementation(
            functools.partial(
                DataArrayRolling._numpy_or_bottleneck_reduce,
                array_agg_func=array_agg_func,
                bottleneck_move_func=bottleneck_move_func,
                rolling_agg_func=rolling_agg_func,
            ),
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def construct(
        self,
        window_dim=None,
        stride=1,
        fill_value=dtypes.NA,
        keep_attrs=None,
        **window_dim_kwargs,
    ):
        """
        Convert this rolling object to xr.Dataset,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim : str or mapping, optional
            A mapping from dimension name to the new window dimension names.
            Just a string can be used for 1d-rolling.
        stride : int, optional
            size of stride for the rolling window.
        fill_value : Any, default: dtypes.NA
            Filling value to match the dimension size.
        **window_dim_kwargs : {dim: new_name, ...}, optional
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        Dataset with variables converted from rolling object.
        """

        from .dataset import Dataset

        keep_attrs = self._get_keep_attrs(keep_attrs)

        if window_dim is None:
            if len(window_dim_kwargs) == 0:
                raise ValueError(
                    "Either window_dim or window_dim_kwargs need to be specified."
                )
            window_dim = {d: window_dim_kwargs[d] for d in self.dim}

        window_dim = self._mapping_to_list(
            window_dim, allow_default=False, allow_allsame=False
        )
        stride = self._mapping_to_list(stride, default=1)

        dataset = {}
        for key, da in self.obj.data_vars.items():
            # keeps rollings only for the dataset depending on self.dim
            dims = [d for d in self.dim if d in da.dims]
            if dims:
                wi = {d: window_dim[i] for i, d in enumerate(self.dim) if d in da.dims}
                st = {d: stride[i] for i, d in enumerate(self.dim) if d in da.dims}

                dataset[key] = self.rollings[key].construct(
                    window_dim=wi,
                    fill_value=fill_value,
                    stride=st,
                    keep_attrs=keep_attrs,
                )
            else:
                dataset[key] = da.copy()

            # as the DataArrays can be copied we need to delete the attrs
            if not keep_attrs:
                dataset[key].attrs = {}

        attrs = self.obj.attrs if keep_attrs else {}

        coords = self._get_output_coords(all_dims=True)

        return Dataset(dataset, coords=coords, attrs=attrs).isel(
            **{d: slice(None, None, s) for d, s in zip(self.dim, stride)}
        )


class Coarsen(CoarsenArithmetic):
    """A object that implements the coarsen.

    See Also
    --------
    Dataset.coarsen
    DataArray.coarsen
    """

    __slots__ = (
        "obj",
        "boundary",
        "coord_func",
        "windows",
        "side",
        "trim_excess",
        "keep_attrs",
    )
    _attributes = ("windows", "side", "trim_excess")

    def __init__(self, obj, windows, boundary, side, coord_func, keep_attrs):
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        boundary : 'exact' | 'trim' | 'pad'
            If 'exact', a ValueError will be raised if dimension size is not a
            multiple of window size. If 'trim', the excess indexes are trimed.
            If 'pad', NA will be padded.
        side : 'left' or 'right' or mapping from dimension to 'left' or 'right'
        coord_func : mapping from coordinate name to func.

        Returns
        -------
        coarsen
        """
        self.obj = obj
        self.windows = windows
        self.side = side
        self.boundary = boundary

        if keep_attrs is not None:
            warnings.warn(
                "Passing ``keep_attrs`` to ``coarsen`` is deprecated and will raise an"
                " error in xarray 0.19. Please pass ``keep_attrs`` directly to the"
                " applied function, i.e. use ``ds.coarsen(...).mean(keep_attrs=False)``"
                " instead of ``ds.coarsen(..., keep_attrs=False).mean()``"
                " Note that keep_attrs is now True per default.",
                FutureWarning,
            )
        self.keep_attrs = keep_attrs

        absent_dims = [dim for dim in windows.keys() if dim not in self.obj.dims]
        if absent_dims:
            raise ValueError(
                f"Dimensions {absent_dims!r} not found in {self.obj.__class__.__name__}."
            )
        if not utils.is_dict_like(coord_func):
            coord_func = {d: coord_func for d in self.obj.dims}
        for c in self.obj.coords:
            if c not in coord_func:
                coord_func[c] = duck_array_ops.mean
        self.coord_func = coord_func

    def _get_keep_attrs(self, keep_attrs):

        if keep_attrs is None:
            # TODO: uncomment the next line and remove the others after the deprecation
            # keep_attrs = _get_keep_attrs(default=True)

            if self.keep_attrs is None:
                keep_attrs = _get_keep_attrs(default=True)
            else:
                keep_attrs = self.keep_attrs

        return keep_attrs

    def __repr__(self):
        """provide a nice str repr of our coarsen object"""

        attrs = [
            "{k}->{v}".format(k=k, v=getattr(self, k))
            for k in self._attributes
            if getattr(self, k, None) is not None
        ]
        return "{klass} [{attrs}]".format(
            klass=self.__class__.__name__, attrs=",".join(attrs)
        )

    def construct(
        self,
        window_dim=None,
        keep_attrs=None,
        **window_dim_kwargs,
    ):
        """
        Convert this Coarsen object to a DataArray or Dataset,
        where the coarsening dimension is split or reshaped to two
        new dimensions.

        Parameters
        ----------
        window_dim: mapping
            A mapping from existing dimension name to new dimension names.
            The size of the second dimension will be the length of the
            coarsening window.
        keep_attrs: bool, optional
            Preserve attributes if True
        **window_dim_kwargs : {dim: new_name, ...}
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        Dataset or DataArray with reshaped dimensions

        Examples
        --------
        >>> da = xr.DataArray(np.arange(24), dims="time")
        >>> da.coarsen(time=12).construct(time=("year", "month"))
        <xarray.DataArray (year: 2, month: 12)>
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
        Dimensions without coordinates: year, month

        See Also
        --------
        DataArrayRolling.construct
        DatasetRolling.construct
        """

        from .dataarray import DataArray
        from .dataset import Dataset

        window_dim = either_dict_or_kwargs(
            window_dim, window_dim_kwargs, "Coarsen.construct"
        )
        if not window_dim:
            raise ValueError(
                "Either window_dim or window_dim_kwargs need to be specified."
            )

        bad_new_dims = tuple(
            win
            for win, dims in window_dim.items()
            if len(dims) != 2 or isinstance(dims, str)
        )
        if bad_new_dims:
            raise ValueError(
                f"Please provide exactly two dimension names for the following coarsening dimensions: {bad_new_dims}"
            )

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        missing_dims = set(window_dim) - set(self.windows)
        if missing_dims:
            raise ValueError(
                f"'window_dim' must contain entries for all dimensions to coarsen. Missing {missing_dims}"
            )
        extra_windows = set(self.windows) - set(window_dim)
        if extra_windows:
            raise ValueError(
                f"'window_dim' includes dimensions that will not be coarsened: {extra_windows}"
            )

        reshaped = Dataset()
        if isinstance(self.obj, DataArray):
            obj = self.obj._to_temp_dataset()
        else:
            obj = self.obj

        reshaped.attrs = obj.attrs if keep_attrs else {}

        for key, var in obj.variables.items():
            reshaped_dims = tuple(
                itertools.chain(*[window_dim.get(dim, [dim]) for dim in list(var.dims)])
            )
            if reshaped_dims != var.dims:
                windows = {w: self.windows[w] for w in window_dim if w in var.dims}
                reshaped_var, _ = var.coarsen_reshape(windows, self.boundary, self.side)
                attrs = var.attrs if keep_attrs else {}
                reshaped[key] = (reshaped_dims, reshaped_var, attrs)
            else:
                reshaped[key] = var

        should_be_coords = set(window_dim) & set(self.obj.coords)
        result = reshaped.set_coords(should_be_coords)
        if isinstance(self.obj, DataArray):
            return self.obj._from_temp_dataset(result)
        else:
            return result


class DataArrayCoarsen(Coarsen):
    __slots__ = ()

    _reduce_extra_args_docstring = """"""

    @classmethod
    def _reduce_method(
        cls, func: Callable, include_skipna: bool = False, numeric_only: bool = False
    ):
        """
        Return a wrapped function for injecting reduction methods.
        see ops.inject_reduce_methods
        """
        kwargs: Dict[str, Any] = {}
        if include_skipna:
            kwargs["skipna"] = None

        def wrapped_func(self, keep_attrs: bool = None, **kwargs):
            from .dataarray import DataArray

            keep_attrs = self._get_keep_attrs(keep_attrs)

            reduced = self.obj.variable.coarsen(
                self.windows, func, self.boundary, self.side, keep_attrs, **kwargs
            )
            coords = {}
            for c, v in self.obj.coords.items():
                if c == self.obj.name:
                    coords[c] = reduced
                else:
                    if any(d in self.windows for d in v.dims):
                        coords[c] = v.variable.coarsen(
                            self.windows,
                            self.coord_func[c],
                            self.boundary,
                            self.side,
                            keep_attrs,
                            **kwargs,
                        )
                    else:
                        coords[c] = v
            return DataArray(
                reduced, dims=self.obj.dims, coords=coords, name=self.obj.name
            )

        return wrapped_func

    def reduce(self, func: Callable, keep_attrs: bool = None, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form `func(x, axis, **kwargs)`
            to return the result of collapsing an np.ndarray over the coarsening
            dimensions.  It must be possible to provide the `axis` argument
            with a tuple of integers.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))
        >>> coarsen = da.coarsen(b=2)
        >>> coarsen.reduce(np.sum)
        <xarray.DataArray (a: 2, b: 2)>
        array([[ 1,  5],
               [ 9, 13]])
        Dimensions without coordinates: a, b
        """
        wrapped_func = self._reduce_method(func)
        return wrapped_func(self, keep_attrs=keep_attrs, **kwargs)


class DatasetCoarsen(Coarsen):
    __slots__ = ()

    _reduce_extra_args_docstring = """"""

    @classmethod
    def _reduce_method(
        cls, func: Callable, include_skipna: bool = False, numeric_only: bool = False
    ):
        """
        Return a wrapped function for injecting reduction methods.
        see ops.inject_reduce_methods
        """
        kwargs: Dict[str, Any] = {}
        if include_skipna:
            kwargs["skipna"] = None

        def wrapped_func(self, keep_attrs: bool = None, **kwargs):
            from .dataset import Dataset

            keep_attrs = self._get_keep_attrs(keep_attrs)

            if keep_attrs:
                attrs = self.obj.attrs
            else:
                attrs = {}

            reduced = {}
            for key, da in self.obj.data_vars.items():
                reduced[key] = da.variable.coarsen(
                    self.windows,
                    func,
                    self.boundary,
                    self.side,
                    keep_attrs=keep_attrs,
                    **kwargs,
                )

            coords = {}
            for c, v in self.obj.coords.items():
                # variable.coarsen returns variables not containing the window dims
                # unchanged (maybe removes attrs)
                coords[c] = v.variable.coarsen(
                    self.windows,
                    self.coord_func[c],
                    self.boundary,
                    self.side,
                    keep_attrs=keep_attrs,
                    **kwargs,
                )

            return Dataset(reduced, coords=coords, attrs=attrs)

        return wrapped_func

    def reduce(self, func: Callable, keep_attrs=None, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form `func(x, axis, **kwargs)`
            to return the result of collapsing an np.ndarray over the coarsening
            dimensions.  It must be possible to provide the `axis` argument with
            a tuple of integers.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Arrays with summarized data.
        """
        wrapped_func = self._reduce_method(func)
        return wrapped_func(self, keep_attrs=keep_attrs, **kwargs)
