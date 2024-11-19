from collections.abc import Callable, Mapping
from functools import partial
from itertools import chain, zip_longest
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
    from dask.array import blockwise, from_array
    from dask.array.overlap import overlap
    from dask.array.routines import take

    from xarray.core.dask_array_ops import subset_to_blocks

    # Assumption: We are always interping from a 1D coordinate X
    # to potential nD coordinats new_X
    assert all(coord.ndim == 1 for coord in x)

    # With advanced interpolation, we are taking a 1D x and interp-ing
    # to a nD new_x. The only way to do this is general is to ravel out the
    # destination coordinates, and then reshape back to the correct order
    flat_new_x = [_.ravel() for _ in new_x]
    out_shape = data.shape[: -len(x)] + new_x[0].shape

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

    # these are the chunks that are needed to construct the output
    # TODO: what happens when a point overlaps with the grid?
    digitized = tuple(
        tuple(np.digitize(desired, coord[list(np.cumsum(chunks))[:-1]]))
        for ax, desired, coord, chunks in zip(
            axis, flat_new_x, x, chunksizes, strict=True
        )
    )
    needed_chunks = tuple(zip(*(digitized), strict=True))

    # maps a block index to indices of the desired points in that block
    grouped = tlz.groupby(
        key=lambda x: tuple(map(int, x[1])), seq=enumerate(needed_chunks)
    )
    blocks_to_idx = {
        block_id: tuple(_[0] for _ in vals) for block_id, vals in grouped.items()
    }
    desired_chunks = tuple(len(a) for a in blocks_to_idx.values())

    # subset to needed blocks only
    subset = subset_to_blocks(
        overlapped,
        coords=tuple(zip(*blocks_to_idx.keys(), strict=True)),
        new_chunks=desired_chunks,
        token="var",
    )

    grid_block_coords = []
    for ax, coord in enumerate(chunked_grid_coords):
        chunkids = [key[ax] for key in blocks_to_idx]
        grid_block_coords.append(
            subset_to_blocks(
                coord,
                coords=(chunkids,),
                new_chunks=tuple(chunksizes[ax][i] for i in chunkids),
                # TODO: more tokenize
                token=f"interp-ax-{ax}",
            )
        )
    # sort the output points by block-id
    # so that every point in a single block occurs near each other
    argsorter = np.concatenate(list(blocks_to_idx.values()))
    invert_argsorter = inverse_permutation(argsorter)

    # The axis indices here are meaningless. I'm faking them to build a purely blockwise graph.
    # each block is sent to _interp with the appropriate grid coordinates, and the output points to
    # interpolate to.
    ndim = len(x)
    new_axis = (data.ndim - ndim,)
    out_axis = tuple(range(data.ndim - ndim)) + new_axis
    result = blockwise(
        func,
        out_axis,
        subset,
        out_axis,
        *chain(
            # input grid coords
            *zip_longest(grid_block_coords, (new_axis,), fillvalue=new_axis),
            # output point coords
            *zip_longest(
                (
                    from_array(points_single_axis[argsorter], desired_chunks)
                    for points_single_axis in flat_new_x
                ),
                (new_axis,),
                fillvalue=new_axis,
            ),
        ),
        concatenate=False,
        # TODO: FIX THIS
        adjust_chunks={out_axis[-1]: desired_chunks},
        # This is important, it stops all broadcasting.
        align_arrays=False,
        **blockwise_kwargs,
    )

    from dask.array import reshape, reshape_blockwise, take
    from dask.array.core import slices_from_chunks

    ndim = len(x)
    slices = slices_from_chunks(data.chunks[-ndim:])
    # argsort back to the original order of points
    # if chunking along the interped output dimensions is desired, we add one element of cleverness
    if out_chunks is not None:
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
        # TODO: rechunk to a single block along axis=-1 first?
        return reshape(take(result, invert_argsorter, axis=-1), shape=out_shape)
