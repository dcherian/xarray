import itertools
import math
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import numpy as np

from xarray.core.nputils import inverse_permutation
from xarray.namedarray.utils import module_available


def reshape_blockwise(
    x: Any,
    shape: int | tuple[int, ...],
    chunks: tuple[tuple[int, ...], ...] | None = None,
):
    if module_available("dask", "2024.08.2"):
        from dask.array import reshape_blockwise

        return reshape_blockwise(x, shape=shape, chunks=chunks)
    else:
        return x.reshape(shape)


def sliding_window_view(
    x, window_shape, axis=None, *, automatic_rechunk=True, **kwargs
):
    # Backcompat for handling `automatic_rechunk`, delete when dask>=2024.11.0
    # Note that subok, writeable are unsupported by dask, so we ignore those in kwargs
    from dask.array.lib.stride_tricks import sliding_window_view

    if module_available("dask", "2024.11.0"):
        return sliding_window_view(
            x, window_shape=window_shape, axis=axis, automatic_rechunk=automatic_rechunk
        )
    else:
        # automatic_rechunk is not supported
        return sliding_window_view(x, window_shape=window_shape, axis=axis)


def interp_helper(
    func: Callable[[np.ndarray], ...],
    data,
    *,
    x: tuple[np.ndarray, ...],
    new_x: tuple[np.ndarray, ...],
    axis: tuple[int, ...],
    depth: int,
    out_chunks=None,
    blockwise_kwargs: Mapping[Any, Any],
    dtype=None,
    meta=None,
):
    """
    Helper function to apply an interpolator ``func`` to blocks of the data.

    ``func`` will only be applied to blocks that are necessary to construct the output.
    The blocks of interest are determined by comparing ``new_x`` to ``x``, both of which
    must be in-memory arrays.

    Parameters
    ----------
    func: Callable
        interpolation function that will be called on each block as
        ``func(data, *interleaved_x_new_x, **blockwise_kwargs)``
    data: dask.array.Array
        array to interpolate
    x : tuple[np.ndarray, ...]
        input array coordinates
    new_x : tuple[np.ndarray, ...]
        output array coordinates to interpolate to
    axis : tuple[int, ...]
        axes of `data` along which to interpolate
    depth: int TODO: make dict[int, int]
        The number of elements that each block should share with its neighbors
        If a tuple or dict then this can be different per axis.
    blockwise_kwargs:
        Arbitrary kwargs unpacked and passed to func

    Returns
    ------
    dask.array.Array
    """
    import toolz as tlz
    from dask.array import Array, from_array, reshape, reshape_blockwise
    from dask.array.core import slices_from_chunks
    from dask.array.overlap import overlap
    from dask.array.routines import take
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph

    def _take(array: np.ndarray, *, mask: np.ndarray, axis: int) -> np.ndarray:
        """Runs take with indices inferred from a boolean mask. Doing so preserves dimensionality
        compared to using `__getitem__`."""
        if array.ndim == 0:
            assert mask.all()
            return array

        squeezed = mask.squeeze()
        assert squeezed.ndim == 1
        (indices,) = np.nonzero(squeezed)
        return np.take(array, indices=indices, axis=axis)

    # Assumption: We are always interping from a 1D coordinate X
    # to potential nD coordinates new_X
    assert all(coord.ndim == 1 for coord in x)
    # desired coordinate locations must be of same dimensionality
    assert all(np.ndim(_) == np.ndim(new_x[0]) for _ in new_x[1:])
    # check that data are aligned.
    assert all(
        size == coord.size for size, coord in zip(data.shape[-len(x) :], x, strict=True)
    )

    chunksizes = tuple(data.chunks[ax] for ax in axis)

    if all(len(_) == 1 for _ in chunksizes):
        # no overlap needed if this is a blockwise op
        overlapper = lambda x, depth: x
        # TODO: more short-circuiting
    else:
        overlapper = partial(overlap, boundary=None)
    overlapped = overlapper(data, depth=dict.fromkeys(axis, depth))

    # chunk and overlap the coordinate arrays appropriately
    chunked_grid_coords = tuple(
        overlapper(from_array(coord, chunks=chunks), depth={0: depth})
        for ax, (coord, chunks) in enumerate(zip(x, chunksizes, strict=True))
    )

    # With advanced interpolation, we are taking a 1D x and interp-ing
    # to a nD new_x. The only way to do this is general is to ravel out the
    # destination coordinates, and then reshape back to the correct order
    # `out_shape` (calculated later) is the shape we reshape back to.
    flat_new_x = [_.ravel() if np.ndim(_) != 0 else _ for _ in new_x]
    # flat_new_x = [_.ravel() for _ in new_x]

    # these are the chunks that are needed to construct the output
    # TODO: what happens when a point overlaps with the grid?
    # TODO: this call to digitize, labels each output point with
    # input chunk. This is a potential extension point for interping from
    # source nD coordinates
    digitized = tuple(
        np.atleast_1d(np.digitize(desired, coord[list(np.cumsum(chunks))[:-1]]))
        for ax, desired, coord, chunks in zip(
            axis, flat_new_x, x, chunksizes, strict=True
        )
    )

    # TODO: what if len(x) is 1?
    is_orthogonal = not (
        np.broadcast_shapes(*(_.shape for _ in new_x)) == flat_new_x[0].shape
    )
    ndim = len(x)
    out_shape = data.shape[:-ndim]
    token = "interp-" + tokenize(data, *x, *new_x)
    blockwise_func = partial(func, **blockwise_kwargs)

    # now find all the blocks needed to construct the output
    if is_orthogonal:
        loop_dim_chunks = itertools.product(
            *(range(len(chunks)) for chunks in data.chunks[:-ndim])
        )
        unique_chunks = tuple(map(np.unique, digitized))
        needed_chunks = tuple(itertools.product(*unique_chunks))
        out_shape += np.broadcast_shapes(*(_.shape for _ in new_x))
        # We are sending the desired output coordinate locations to the appropriate
        # block of the input. After interpolation we must argsort back to the correct order
        # This `argsorter` is only needed for calculating the "inverse" argsort indices: `invert_argsorter`
        argsorter = tuple(np.argsort(_.squeeze()) for _ in flat_new_x)

        layer = {}
        for loop_dim_chunk_coord, (
            flat_out_chunk_coord,
            input_core_dim_chunk_coord,
        ) in itertools.product(loop_dim_chunks, enumerate(needed_chunks)):
            out_core_dim_chunk_coord = np.unravel_index(
                flat_out_chunk_coord, out_shape[-ndim:]
            )
            layer[token, *loop_dim_chunk_coord, *out_core_dim_chunk_coord] = (
                blockwise_func,
                # block to interpolate
                (overlapped.name, *loop_dim_chunk_coord, *input_core_dim_chunk_coord),
                # corresponding input coordinate block
                *(
                    (coord.name, chunk_coord)
                    for coord, chunk_coord in zip(
                        chunked_grid_coords, input_core_dim_chunk_coord, strict=True
                    )
                ),
                # output coordinates that can be constructed from this input block
                *(
                    # TODO: do we need to use the argsorter?
                    _take(out_coord, mask=out_chunk == current_chunk, axis=ax)
                    for (ax, out_chunk, out_coord, current_chunk) in zip(
                        range(ndim),
                        digitized,
                        flat_new_x,
                        input_core_dim_chunk_coord,
                        strict=True,
                    )
                ),
            )

        desired_chunks = tuple(
            tuple(
                len(v)
                for v in tlz.groupby(
                    lambda x: x,
                    digitized_.squeeze(
                        axis=tuple(a for a in range(digitized_.ndim) if a != ax)
                    ),
                ).values()
            )
            if np.ndim(new) != 0
            else 0
            for ax, (digitized_, new) in enumerate(zip(digitized, new_x, strict=True))
        )
        # desired_chunks = tuple(
        #     len(np.unique(_)) if np.ndim(new) != 0 else 0
        #     for _, new in zip(digitized, new_x, strict=True)
        # )

    else:
        needed_chunks = tuple(zip(*digitized, strict=True))
        out_shape += new_x[0].shape

        # maps a block index to indices of the desired points in that block
        grouped = tlz.groupby(
            key=lambda x: tuple(map(int, x[1])),
            seq=enumerate(needed_chunks),
        )
        blocks_to_idx = {
            block_id: tuple(_[0] for _ in vals) for block_id, vals in grouped.items()
        }
        desired_chunks = (tuple(len(a) for a in blocks_to_idx.values()),)
        needed_chunk_indexer = tuple(zip(*grouped.keys(), strict=True))
        # We are sending the desired output coordinate locations to the appropriate
        # block of the input. After interpolation we must argsort back to the correct order
        # This `argsorter` is only needed for calculating the "inverse" argsort indices: `invert_argsorter`
        argsorter = (np.concatenate(list(blocks_to_idx.values())),)

        keys = overlapped._key_array[..., *needed_chunk_indexer, :]
        layer = {}
        # number of blocks we will pull from the coords axis
        size = math.prod(keys.shape[:-1][-ndim:])
        for flatidx, idx in enumerate(np.ndindex(keys.shape[:-1])):
            loop_dim_chunk_coord = idx[:-1]
            out_core_dim_chunk_coord = flatidx % size
            input_block = tuple(keys[*idx, :])
            input_core_dim_chunk_coord = input_block[-ndim:]
            indices = blocks_to_idx[input_core_dim_chunk_coord]
            output_block = (token, *loop_dim_chunk_coord, out_core_dim_chunk_coord)
            layer[output_block] = (
                blockwise_func,
                # block to interpolate
                input_block,
                # corresponding input coordinate block
                *(
                    (coord.name, chunk_coord)
                    for coord, chunk_coord in zip(
                        chunked_grid_coords, input_core_dim_chunk_coord, strict=True
                    )
                ),
                # output coordinates that can be constructed from this input block
                *(np.take(out_coord, indices=indices, axis=0) for out_coord in new_x),
            )

    graph = HighLevelGraph.from_collections(
        token, layer, dependencies=[overlapped, *chunked_grid_coords]
    )
    result = Array(
        graph,
        token,
        chunks=data.chunks[:-ndim] + desired_chunks,
        dtype=np.float64,
        meta=data._meta,
    )
    invert_argsorter = tuple(inverse_permutation(_) for _ in argsorter)

    # sort the output points by block-id
    # so that every point in a single block occurs near each other
    # argsort back to the original order of points
    # if chunking along the interped output dimensions is desired, we add one element of cleverness
    if out_chunks is not None:
        slices = slices_from_chunks(data.chunks[-ndim:])
        # permute the indices so that we can reshape to the desired chunking in a blockwise fashion.
        take_indices = np.concatenate(
            [invert_argsorter.reshape(out_shape)[slc].ravel() for slc in slices]
        )
        return reshape_blockwise(
            take(result, indices=take_indices, axis=-1),
            shape=out_shape,
            chunks=out_chunks,
        )
    else:
        for ax, idxr in enumerate(invert_argsorter):
            result = take(result, idxr, axis=-1 - ax)
        return reshape(result, shape=out_shape)
