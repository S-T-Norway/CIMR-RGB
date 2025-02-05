import logging

import pytest
import numpy as np
from pyresample import geometry, kd_tree


from cimr_rgb.rgb_logging import RGBLogging
from cimr_rgb.regridder import ReGridder


# =============================================================================
# Dummy Logger and Configuration for testing
# =============================================================================


class DummyLogger:
    """
    A dummy logger that collects error messages.
    """

    def __init__(self):
        self.messages = []

    def error(self, msg, *args, **kwargs):
        self.messages.append(msg)

    def info(self, msg):
        pass

    def warning(self, msg):
        pass


class DummyConfig:
    """
    Dummy configuration object holding grid_type and logger.
    """

    def __init__(self, grid_type: str, logger=None):
        self.grid_type = grid_type
        self.logger = logger
        self.logpar_decorate = False


# =============================================================================
# Dummy kd_tree.get_neighbour_info functions for controlled testing.
# =============================================================================


def dummy_kd_tree_1d(source_geo_def, target_geo_def, neighbours, radius_of_influence):
    """
    Returns a 1D distance array (with one np.inf value) to test filtering.
    """
    valid_input_index = np.array([0, 1, 2])
    valid_output_index = np.array([0, 1, 2])
    index_array = np.array([10, 20, 30])
    distance_array = np.array([0.1, np.inf, 0.3])
    return valid_input_index, valid_output_index, index_array, distance_array


def dummy_kd_tree_2d(source_geo_def, target_geo_def, neighbours, radius_of_influence):
    """
    Returns a 2D distance array (with some rows completely np.inf) to test filtering.
    """
    valid_input_index = np.array([0, 1, 2, 3])
    valid_output_index = np.array([0, 1, 2, 3])
    index_array = np.array([[10, 11], [20, 21], [30, 31], [40, 41]])
    distance_array = np.array([
        [0.1, np.inf],  # valid row
        [np.inf, np.inf],  # row to be filtered out
        [0.2, 0.3],  # valid row
        [np.inf, np.inf],
    ])  # row to be filtered out
    return valid_input_index, valid_output_index, index_array, distance_array


def dummy_kd_tree_invalid_ndim(
    source_geo_def, target_geo_def, neighbours, radius_of_influence
):
    """
    Returns a 3D distance array (unsupported) to trigger an error.
    """
    valid_input_index = np.array([0, 1])
    valid_output_index = np.array([0, 1])
    index_array = np.array([10, 20])
    distance_array = np.zeros((2, 2, 2))  # 3D array
    return valid_input_index, valid_output_index, index_array, distance_array


# =============================================================================
# Parameterized test function using @pytest.mark.parametrize
# =============================================================================


@pytest.mark.parametrize(
    "test_case, grid_type, source_lon, source_lat, target_lon, target_lat, search_radius, neighbours, monkeypatch_func, expected_exception, expected_output",
    [
        # Valid 1D case:
        #   - source: 1D arrays (swath definition acceptable)
        #   - target: 2D arrays (grid definition requires 2D arrays)
        (
            "valid_1d",
            "L1C",
            np.array([0, 1, 2]),
            np.array([10, 11, 12]),
            np.array([[5, 6, 7]]),
            np.array([[15, 16, 17]]),
            100.0,
            2,
            dummy_kd_tree_1d,
            None,
            (
                np.array([0.1, 0.3]),
                np.array([10, 30]),
                np.array([0, 2]),
                np.array([0, 1, 2]),
            ),
        ),
        # Valid 2D case:
        #   - source: 1D arrays (swath definition)
        #   - target: 2D arrays (e.g. a 2×4 grid)
        (
            "valid_2d",
            "L1C",
            np.array([0, 1, 2, 3]),
            np.array([10, 11, 12, 13]),
            np.array([[5, 6, 7, 8], [5, 6, 7, 8]]),
            np.array([[15, 16, 17, 18], [15, 16, 17, 18]]),
            50.0,
            2,
            dummy_kd_tree_2d,
            None,
            (
                np.array([
                    [0.1, np.inf],
                    [0.2, 0.3],
                ]),  # reduced_distance: rows 0 and 2 kept from dummy_kd_tree_2d
                np.array([[10, 11], [30, 31]]),  # reduced_index
                np.array([0, 2]),  # original_indices
                np.array([0, 1, 2, 3]),
            ),
        ),
        # Error: Mismatched source coordinate arrays.
        (
            "mismatched_source",
            "L1C",
            np.array([0, 1, 2]),
            np.array([10, 11]),  # lengths 3 vs. 2
            np.array([[5, 6, 7]]),
            np.array([[15, 16, 17]]),
            100.0,
            2,
            None,
            "source_lon and source_lat must have the same length",
            None,
        ),
        # Error: Mismatched target coordinate arrays.
        (
            "mismatched_target",
            "L1C",
            np.array([0, 1, 2]),
            np.array([10, 11, 12]),
            np.array([[5, 6, 7]]),
            np.array([[15, 16, 17], [18, 19, 20]]),  # row counts differ: 1 vs. 2
            100.0,
            2,
            None,
            "target_lon and target_lat must have the same length",
            None,
        ),
        # Error: Negative search_radius.
        (
            "negative_search_radius",
            "L1C",
            np.array([0, 1]),
            np.array([10, 11]),
            np.array([[5, 6]]),
            np.array([[15, 16]]),
            -10.0,
            2,
            None,
            "search_radius must be non-negative",
            None,
        ),
        # Error: Invalid neighbours value.
        (
            "invalid_neighbours",
            "L1C",
            np.array([0, 1]),
            np.array([10, 11]),
            np.array([[5, 6]]),
            np.array([[15, 16]]),
            10.0,
            0,
            None,
            "neighbours must be a positive integer",
            None,
        ),
        # Error: Unsupported grid type.
        (
            "invalid_grid",
            "INVALID",
            np.array([0, 1]),
            np.array([10, 11]),
            np.array([[5, 6]]),
            np.array([[15, 16]]),
            10.0,
            2,
            None,
            "Unsupported grid_type",
            None,
        ),
        # Error: Unsupported distance array dimensions.
        (
            "invalid_distance_array_ndim",
            "L1C",
            np.array([0, 1]),
            np.array([10, 11]),
            np.array([[5, 6]]),
            np.array([[15, 16]]),
            10.0,
            1,
            dummy_kd_tree_invalid_ndim,
            "Unsupported dimensions for distance_array",
            None,
        ),
    ],
)
def test_get_neighbours_parametrized(
    monkeypatch,
    test_case,
    grid_type,
    source_lon,
    source_lat,
    target_lon,
    target_lat,
    search_radius,
    neighbours,
    monkeypatch_func,
    expected_exception,
    expected_output,
):
    """
    Parameterized unit test for ReGridder.get_neighbours covering both valid outputs and error conditions.

    Parameters
    ----------
    test_case : str
        Descriptive name of the test case.
    grid_type : str
        The grid type for the configuration.
    source_lon, source_lat : np.ndarray
        1D arrays for source coordinates.
    target_lon, target_lat : np.ndarray
        2D arrays for target coordinates (for grid definitions).
    search_radius : float
        The search radius for the neighbor search.
    neighbours : int
        The number of neighbors to search for.
    monkeypatch_func : function or None
        A dummy function to patch kd_tree.get_neighbour_info (if provided).
    expected_exception : str or None
        Expected exception substring (if an error is expected).
    expected_output : tuple or None
        Expected output tuple if no error is expected.
    """
    # Create a dummy logger and configuration, then instantiate ReGridder.
    dummy_logger = DummyLogger()
    config = DummyConfig(grid_type, logger=dummy_logger)
    instance = ReGridder(config)

    # Patch kd_tree.get_neighbour_info if a dummy function is provided.
    if monkeypatch_func is not None:
        monkeypatch.setattr(kd_tree, "get_neighbour_info", monkeypatch_func)

    if expected_exception is not None:
        with pytest.raises(ValueError, match=expected_exception):
            instance.get_neighbours(
                source_lon,
                source_lat,
                target_lon,
                target_lat,
                search_radius,
                neighbours,
            )
    else:
        result = instance.get_neighbours(
            source_lon, source_lat, target_lon, target_lat, search_radius, neighbours
        )
        exp_red_dist, exp_red_idx, exp_orig_idx, exp_valid_in = expected_output
        np.testing.assert_array_equal(result[0], exp_red_dist)
        np.testing.assert_array_equal(result[1], exp_red_idx)
        np.testing.assert_array_equal(result[2], exp_orig_idx)
        np.testing.assert_array_equal(result[3], exp_valid_in)
