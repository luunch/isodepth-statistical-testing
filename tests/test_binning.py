import numpy as np

from data.schemas import DatasetBundle, TestConfig, run_config_from_mapping
from methods.binning import bin_dataset_to_pseudospots


def test_binning_sums_counts_into_square_pseudospots():
    dataset = DatasetBundle(
        S=np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [11.0, 1.0],
            ],
            dtype=np.float32,
        ),
        A=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        meta={
            "coordinate_um_per_unit": 1.0,
            "binning_preprocessing": {
                "normalize_total": False,
                "log1p": False,
                "standardize_expression": False,
            },
        },
    ).validate()
    config = TestConfig(
        method="binning",
        n_perms=1,
        bin_shape="square",
        bin_spot_distance_um=10.0,
        coordinate_um_per_unit=1.0,
        block_jitter=False,
    ).validate()

    binned, artifacts = bin_dataset_to_pseudospots(dataset, config)

    assert binned.n_cells == 2
    assert np.allclose(binned.S, np.array([[6.0, 6.0], [16.0, 6.0]], dtype=np.float32))
    assert np.allclose(binned.A, np.array([[4.0, 6.0], [5.0, 6.0]], dtype=np.float32))
    assert artifacts["binning_summary"]["original_n_cells"] == 3
    assert artifacts["binning_summary"]["n_pseudospots"] == 2
    assert np.array_equal(artifacts["binning_cell_counts"], np.array([2, 1]))


def test_binning_keeps_cell_types_as_separate_pseudospots():
    dataset = DatasetBundle(
        S=np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        ),
        A=np.array(
            [
                [1.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        ),
        meta={
            "coordinate_um_per_unit": 1.0,
            "cell_type_mode": "together",
            "cell_type_labels": np.array([0, 1], dtype=np.int64),
            "cell_type_names": ["a", "b"],
            "n_cell_types": 2,
            "binning_preprocessing": {
                "normalize_total": False,
                "log1p": False,
                "standardize_expression": False,
            },
        },
    ).validate()
    config = TestConfig(
        method="binning",
        n_perms=1,
        bin_shape="square",
        bin_spot_distance_um=10.0,
        coordinate_um_per_unit=1.0,
        block_jitter=False,
    ).validate()

    binned, _ = bin_dataset_to_pseudospots(dataset, config)

    assert binned.n_cells == 2
    assert np.array_equal(binned.meta["cell_type_labels"], np.array([0, 1]))
    assert np.allclose(binned.S, np.array([[6.0, 6.0], [6.0, 6.0]], dtype=np.float32))
    assert np.allclose(binned.A, dataset.A)


def test_binning_config_round_trips():
    run_config = run_config_from_mapping(
        {
            "data": {
                "source": "synthetic",
            },
            "test": {
                "method": "binning",
                "bin_shape": "hexagonal",
                "bin_spot_distance_um": 75,
            }
        }
    )

    assert run_config.test.method == "binning"
    assert run_config.test.bin_shape == "hexagon"
    assert run_config.test.bin_spot_distance_um == 75
