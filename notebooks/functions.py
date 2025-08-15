import xarray as xr
import numpy as np
import xbatcher
from xbatcher.loaders.torch import MapDataset, IterableDataset
import torch
from tqdm import tqdm

from typing import Literal

def _get_resample_factor(
    bgen: xbatcher.BatchGenerator,
    output_tensor_dim: dict[str, int],
    resample_dim: list[str]
):
    resample_factor = {}
    for dim in resample_dim:
        r = output_tensor_dim[dim] / bgen.input_dims[dim]
        is_int = (r == int(r))
        is_inv_int = (1/r == int(1/r)) if r != 0 else False
        assert is_int or is_inv_int, f"Resample ratio for dim '{dim}' must be an integer or its inverse."
        resample_factor[dim] = r

    return resample_factor


def _get_output_array_size(
    bgen: xbatcher.BatchGenerator,
    output_tensor_dim: dict[str, int],
    new_dim: list[str],
    core_dim: list[str],
    resample_dim: list[str]
):
    resample_factor = _get_resample_factor(bgen, output_tensor_dim, resample_dim)
    output_size = {}
    for key, size in output_tensor_dim.items():
        if key in new_dim:
            # This is a new axis, size is determined
            # by the tensor size.
            output_size[key] = output_tensor_dim[key]
        elif key in core_dim:
            # This is an old but unmodified axis, size is
            # determined by the source array
            if output_tensor_dim[key] != bgen.ds.sizes[key]:
                raise ValueError(
                    f"Axis {key} is a core dim, but the tensor size"
                    f"({output_tensor_dim[key]}) does not equal the "
                    f"source data array size ({bgen.ds.sizes[key]})."
                )
            output_size[key] = bgen.ds.sizes[key]
        elif key in resample_dim:
            # This is a resampled axis, determine the new size
            # by the resample factor.
            temp_output_size = bgen.ds.sizes[key] * resample_factor[key]
            assert temp_output_size.is_integer(), f"Resampling for dim '{key}' results in non-integer size."
            output_size[key] = int(temp_output_size)
        else:
            raise ValueError(f"Axis {key} must be specified in one of new_dim, core_dim, or resample_dim") 
    return output_size


def _resample_coordinate(
    coord: xr.DataArray,
    factor: float,
    mode: Literal["centers", "edges"]="edges"
) -> np.ndarray:
    '''
    Coarsen or densify a 1D array of xarray coordinates. ``factor > 1``
    densifies, and ``factor < 1`` coarsens.

    **It is assumed that entries in ``coord`` have constant step size**.
    '''
    assert len(coord.shape) == 1 and coord.shape[0] > 1

    new_n = coord.shape[0] * factor
    assert new_n.is_integer()

    old_step = coord.data[1] - coord.data[0]
    return np.linspace(
        coord.data[0], 
        coord.data[-1] + old_step, 
        num=int(coord.shape[0] * factor),
        endpoint=False
    )


def _get_output_array_coordinates(
    src_da: xr.DataArray,
    output_array_dim: list[str],
    resample_factor: dict[str, int],
    resample_mode: Literal["centers", "edges"]="edges"
) -> dict[str, np.ndarray]:
    output_coords = {}
    for dim in output_array_dim:
        if dim in src_da.coords and dim in resample_factor:
            # Source array has coordinate and it is changing
            output_coords[dim] = _resample_coordinate(src_da[dim], resample_factor[dim], resample_mode)
        elif dim in src_da.coords:
            # Source array has coordinate but it isn't changing size
            output_coords[dim] = src_da[dim].copy(deep=True).data
        else:
            # Source array doesn't have a coordinate on this dim or
            # this is a new dim, ignore
            continue
    return output_coords
    
    
def predict_on_array(
    dataset: MapDataset | IterableDataset,
    model: torch.nn.Module,
    output_tensor_dim: dict[str, int],
    new_dim: list[str],
    core_dim: list[str],
    resample_dim: list[str],
    resample_mode: Literal["centers", "edges"]="edges",
    batch_size: int=16,
    progress_bar: bool=True
) -> xr.DataArray:
    '''
    Generate predictions from a PyTorch model and reassemble predictions
    into a ``xr.DataArray``, accounting for changes in dimensions. This function
    is analagous to ``xr.apply_ufunc``, except we operate on array patches instead
    of array slices and the user function is replaced with a PyTorch module.

    ``model`` is allowed to drop axes, add axes, or resize axes compared to
    the input tensor. As in ``xr.apply_ufunc``, users must specify how axes
    change via ``new_dim`` and ``resample_dim``. At least one output
    axis must be an input dimension in the ``BatchGenerator`` used to
    generate array patches.

    

    Parameters
    ----------
    ``dataset`` (``MapDataset | IterableDataset``): A dataset that uses a
    ``BatchGenerator`` to produce examples.

    ``model`` (``torch.nn.Module``): A PyTorch module that returns a single
    output tensor.

    ``output_tensor_dim`` (``dict[str, int]``): A dictionary representing the
    names and sizes of output tensor dimensions.

    ``new_dim`` (``list[str]``): A list of axes in ``output_tensor_dim`` that
    are independent of the batch generator and source array. 
    These will be inserted as new dimensions in the output ``xr.DataArray``.

    ``core_dim`` (``list[str]``): A list of axes in ``output_tensor_dim`` that
    are unchanged from the source array and not used for windowing in ``dataset``.

    ``resample_dim`` (``list[str]``): A list of axes in ``output_tensor_dim`` that
    are used to generate patches in ``dataset``. Only these axes are allowed to change
    size.

    ``resample_mode`` (``"edges"|"centers"``): Whether to treat coordinates on the input
    array as pixel edges or centers.

    ``progress_bar`` (``bool``): Whether to show a progress bar while iterating 
    over ``dataset``.

    Notes
    -----
    The output array size is determined by the axes in ``output_tensor_dim`` according
    to the below table.

    | Axis          | Output size                                     |
    |---------------|-------------------------------------------------|
    | New axis      | Same as tensor size                             |
    | Core axis     | Same as source ``xr.DataArray`` size            |
    | Resample axis | Source array size * (tensor size / window size) |

    For example, consider a super-resolution workflow where ``dataset`` generates
    patches of size (x=10, y=10), ``model`` generates tensors of size (x=20, y=20),
    and the source ``xr.DataArray`` has size (x=512, y=512). From the above table,
    we see that the model increases the size of both window axes by a factor of 2. Therefore,
    the output data array will have size (x=1024, y=2024).

    Models may coarsen or densify tensors, but must do so by an integer factor.

    Overlaps are allowed, in which case the average of all output values is returned.
    '''
    s_new = set(new_dim)
    s_core = set(core_dim)
    s_resample = set(resample_dim)

    if s_new & s_core or s_new & s_resample or s_core & s_resample:
        raise ValueError("new_dim, core_dim, and resample_dim must be disjoint sets.")

    bgen = dataset.X_generator

    # Get resample factors
    resample_factor = _get_resample_factor(
        bgen,
        output_tensor_dim, 
        resample_dim
    )
    
    # Set up output array
    output_size = _get_output_array_size(
        bgen,
        output_tensor_dim,
        new_dim,
        core_dim,
        resample_dim
    )
            
    output_da = xr.DataArray(
        data=np.zeros(tuple(output_size.values())),
        dims=tuple(output_size.keys()),
    )
    output_n = xr.full_like(output_da, 0)
    
    # Prepare data laoder
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Iterate over each batch
    for i, batch in tqdm(enumerate(loader), total=len(loader), disable=not progress_bar):
        input_tensor = batch[0] if isinstance(batch, (list, tuple)) else batch
        out_batch = model(input_tensor).detach().numpy()

        # Iterate over each sample in the batch
        for ib in range(out_batch.shape[0]):
            # Get the slice object associated with this sample
            global_index = (i * batch_size) + ib
            old_indexer = bgen._batch_selectors.selectors[global_index][0]
            # Only index into axes that are resampled, rescaling the bounds
            # Perhaps use xbatcher _gen_slices here?
            new_indexer = {}
            for key in old_indexer:
                if key in resample_dim:
                    new_indexer[key] = slice(
                        int(old_indexer[key].start * resample_factor[key]),
                        int(old_indexer[key].stop * resample_factor[key])
                    )

            output_da.loc[new_indexer] += out_batch[ib, ...]
            output_n.loc[new_indexer] += 1

    # Calculate mean
    output_da = output_da / output_n

    # Assign coordinates
    # We wait to do this until the very end because slicing into the
    # output array is easier with integer indices. .loc[] would follow
    # the assigned coordinates if we did this earlier.
    output_da = output_da.assign_coords(
        _get_output_array_coordinates(
            dataset.X_generator.ds, 
            list(output_tensor_dim.keys()), 
            resample_factor, 
            resample_mode
        )
    )

    return output_da
