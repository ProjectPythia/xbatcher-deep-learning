import xarray as xr
import numpy as np
import torch
import xbatcher
import pytest
from xbatcher.loaders.torch import MapDataset

from functions import _get_output_array_size, _resample_coordinate
from functions import predict_on_array, _get_resample_factor
from dummy_models import Identity, MeanAlongDim, SubsetAlongAxis, ExpandAlongAxis, AddAxis

@pytest.fixture
def map_dataset_fixture() -> MapDataset:
    data = xr.DataArray(
        data=np.arange(20 * 10).reshape(20, 10).astype(np.float32),
        dims=("x", "y"),
        coords={"x": np.arange(20, dtype=float), "y": np.arange(10, dtype=float)},
    )
    bgen = xbatcher.BatchGenerator(data, input_dims=dict(x=10, y=5), input_overlap=dict(x=2, y=2))
    return MapDataset(bgen)

@pytest.mark.parametrize("factor, mode, expected", [
    (2.0, "edges", np.arange(0, 10, 0.5)),
    (0.5, "edges", np.arange(0, 10, 2.0)),
])
def test_resample_coordinate(factor, mode, expected):
    coord = xr.DataArray(np.arange(10, dtype=float), dims="x")
    resampled = _resample_coordinate(coord, factor, mode)
    np.testing.assert_allclose(resampled, expected)

@pytest.mark.parametrize(
    "model, output_tensor_dim, new_dim, core_dim, resample_dim, manual_transform",
    [
        # Case 1: Identity - No change
        (
            Identity(),
            {'x': 10, 'y': 5},
            [], [], ['x', 'y'],
            lambda da: da.data
        ),
        # Case 2: ExpandAlongAxis - Upsampling
        (
            ExpandAlongAxis(ax=1, n_repeats=2), # ax=1 is 'x'
            {'x': 20, 'y': 5},
            [], [], ['x', 'y'],
            lambda da: da.data.repeat(2, axis=0) # axis=0 in the 2D numpy array
        ),
        # Case 3: SubsetAlongAxis - Coarsening
        (
            SubsetAlongAxis(ax=1, n=5), # ax=1 is 'x'
            {'x': 5, 'y': 5},
            [], [], ['x', 'y'],
            lambda da: da.isel(x=slice(0, 5)).data
        ),
        # Case 4: MeanAlongDim - Dimension reduction
        (
            MeanAlongDim(ax=2), # ax=2 is 'y'
            {'x': 10},
            [], [], ['x'],
            lambda da: da.mean(dim='y').data
        ),
        # Case 5: AddAxis - Add a new dimension
        (
            AddAxis(ax=1), # Add new dim at axis 1
            {'channel': 1, 'x': 10, 'y': 5},
            ['channel'], [], ['x', 'y'],
            lambda da: np.expand_dims(da.data, axis=0)
        ),
    ]
)
def test_predict_on_array_all_models(
    map_dataset_fixture, model, output_tensor_dim, new_dim, core_dim, resample_dim, manual_transform
):
    """
    Tests reassembly, averaging, and coordinate assignment using a variety of models.
    """
    dataset = map_dataset_fixture
    bgen = dataset.X_generator
    resample_factor = _get_resample_factor(bgen, output_tensor_dim, resample_dim)

    # --- Run the function under test ---
    result_da = predict_on_array(
        dataset=dataset, model=model, output_tensor_dim=output_tensor_dim,
        new_dim=new_dim, core_dim=core_dim, resample_dim=resample_dim, batch_size=4,
        progress_bar=False
    )

    # --- Manually calculate the expected result ---
    expected_size = _get_output_array_size(bgen, output_tensor_dim, new_dim, core_dim, resample_dim)
    expected_sum = xr.DataArray(np.zeros(list(expected_size.values())), dims=list(expected_size.keys()))
    expected_count = xr.full_like(expected_sum, 0, dtype=int)

    for i in range(len(dataset)):
        batch_da = bgen[i]
        old_indexer = bgen._batch_selectors.selectors[i][0]
        new_indexer = {}
        for key in old_indexer:
            if key in resample_dim:
                new_indexer[key] = slice(int(old_indexer[key].start * resample_factor.get(key, 1)), int(old_indexer[key].stop * resample_factor.get(key, 1)))
            elif key in core_dim:
                new_indexer[key] = old_indexer[key]

        model_output_on_batch = manual_transform(batch_da)
        print(f"Batch {i}: {new_indexer} -> {model_output_on_batch.shape}")
        print(f"Expected sum shape: {expected_sum.loc[new_indexer].shape}")
        expected_sum.loc[new_indexer] += model_output_on_batch
        expected_count.loc[new_indexer] += 1

    expected_avg_data = expected_sum.data / expected_count.data

    # --- Assert correctness ---
    np.testing.assert_allclose(result_da.values, expected_avg_data, equal_nan=True)
