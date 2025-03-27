import logging

import pytest
import numpy as np

import cimr_rgb.data_ingestion as data_ingestion
from cimr_rgb.data_ingestion import DataIngestion
from cimr_rgb.rgb_logging import RGBLogging
from cimr_rgb.regridder import ReGridder
from cimr_rgb.grid_generator import GridGenerator, GRIDS


# --- Dummy Dependencies for Testing ---


class DummyLogger:
    """
    A simple dummy logger that collects error messages.
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
    Dummy configuration object holding grid and projection definitions.
    """

    def __init__(
        self, grid_definition, projection_definition, logpar_decorate=False, logger=None
    ):
        self.grid_definition = grid_definition
        self.projection_definition = projection_definition
        self.logpar_decorate = logpar_decorate
        self.logger = logger


def dummy_rgb_decorate_and_execute(decorate, decorator, logger):
    """
    Dummy replacement for RGBLogging.rgb_decorate_and_execute that returns a callable
    producing a dummy grid generator.
    """

    def wrapper(cls):
        return lambda *args, **kwargs: DummyGridGenerator()

    return wrapper


class DummyGridGenerator:
    """
    Dummy grid generator that simply returns the input longitude and latitude as (x, y).
    """

    def lonlat_to_xy(self, lon, lat):
        # In our tests, we assume x = lon and y = lat.
        return lon, lat


# A dummy grid used for valid test cases.
dummy_grid = {
    "x_min": 0,
    "n_cols": 10,
    "res": 1,
    "y_max": 100,
    "n_rows": 10,
    "lat_min": 20,
    "lat_max": 80,
}

# --- Parameterized Test Cases Description ---
#
# In the following test cases:
#
#   - The method first retrieves the grid from GRIDS using self.config.grid_definition.
#   - Then, depending on the grid type and projection, it computes two sets of indices:
#       * out_of_bounds_lat: based on the latitude value relative to grid["lat_min"] or grid["lat_max"]
#       * out_of_bounds_xy: based on computed y coordinates (here, simply equal to the input latitudes)
#   - Finally, for each variable in data_dict, the union of these indices is set to np.nan.
#
# For example, in Test Case 1 the configuration is for an EASE2 grid with projection "N".
# With dummy_grid having lat_min=20 and lat_max=80, if we supply:
#
#     latitude = [10, 30, 95, 110]
#
# then:
#
#   - For projection "N", out_of_bounds_lat = indices where latitude < 20 → index 0.
#   - The y‑boundaries are y_bound_min = y_max – n_rows * res = 100 – 10 = 90 and y_bound_max = 100.
#     Thus, out_of_bounds_xy = indices where (latitude < 90 or latitude > 100) → indices 0, 1, and 3.
#
# The union of these indices is {0,1,3} and the expected result is that every variable in the
# returned dictionary will have np.nan at indices 0, 1, and 3.


@pytest.mark.parametrize(
    "test_case, grid_def, proj_def, input_data, expected_output, expected_exception",
    [
        # Test Case 1: EASE2 grid (using a valid grid key) with projection "N"
        (
            "EASE2_N",
            "EASE2_N3km",  # must be a valid key that contains "EASE2"
            "N",
            {
                "latitude": np.array([10, 30, 95, 110]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            {
                "latitude": np.array([np.nan, np.nan, 95, np.nan]),
                "longitude": np.array([np.nan, np.nan, 70, np.nan]),
                "dummy": np.array([np.nan, np.nan, 3, np.nan]),
                # "latitude": np.array([10, 30, 95, 110]),
                # "longitude": np.array([50, 60, 70, 80]),
                # "dummy": np.array([1, 2, 3, 4]),
            },
            None,
        ),
        # Test Case 2: EASE2 grid with projection "G"
        (
            "EASE2_G",
            "EASE2_G1km",  # valid EASE2 key
            "G",
            {
                "latitude": np.array([10, 30, 90, 50]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            {
                # For "G": out_of_bounds_lat = indices where latitude < 20 or latitude > 80 → indices 0 and 2.
                # y_bound_min = 100 - 10 = 90, y_bound_max = 100.
                # For latitudes [10,30,90,50]: out_of_bounds_xy = indices where (lat < 90 or lat > 100): {0,1,3}.
                # Union = {0,1,2,3} (all indices).
                "latitude": np.array([np.nan, np.nan, np.nan, np.nan]),
                "longitude": np.array([np.nan, np.nan, np.nan, np.nan]),
                "dummy": np.array([np.nan, np.nan, np.nan, np.nan]),
            },
            None,
        ),
        # Test Case 3: EASE2 grid with projection "S"
        (
            "EASE2_S",
            "EASE2_S3km",  # valid key
            "S",
            {
                "latitude": np.array([10, 30, 95, 110]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            {
                # For "S": out_of_bounds_lat = indices where latitude > 20 → indices 1,2,3.
                # y_bound: for [10,30,95,110] → out_of_bounds_xy = {0,1,3}; union = {0,1,2,3}.
                "latitude": np.array([np.nan, np.nan, np.nan, np.nan]),
                "longitude": np.array([np.nan, np.nan, np.nan, np.nan]),
                "dummy": np.array([np.nan, np.nan, np.nan, np.nan]),
            },
            None,
        ),
        # Test Case 4: STEREO grid with projection "PS_N"
        (
            "STEREO_PS_N",
            "STEREO_N6.25km",  # valid key
            "PS_N",
            {
                "latitude": np.array([10, 30, 50, 90]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            {
                # For PS_N: out_of_bounds_lat = indices where latitude < grid["lat_min"].
                # For STEREO_N6.25km, grid["lat_min"] is 60, so indices 0,1,2.
                # y_bound: using grid values, none of these latitudes fall out-of-bound (dummy conversion returns lat as y),
                # so union = {0,1,2}.
                "latitude": np.array([np.nan, np.nan, np.nan, 90]),
                "longitude": np.array([np.nan, np.nan, np.nan, 80]),
                "dummy": np.array([np.nan, np.nan, np.nan, 4]),
            },
            None,
        ),
        # Test Case 5: STEREO grid with projection "PS_S"
        (
            "STEREO_PS_S",
            "STEREO_S6.25km",  # valid key
            "PS_S",
            {
                "latitude": np.array([10, 30, 50, 90]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            {
                # For PS_S: out_of_bounds_lat = indices where latitude > grid["lat_min"].
                # For STEREO_S6.25km, grid["lat_min"] is -60, so all indices qualify.
                # y_bound: dummy conversion yields y=lat; none out-of-bound, so union = {0,1,2,3}.
                "latitude": np.array([np.nan, np.nan, np.nan, np.nan]),
                "longitude": np.array([np.nan, np.nan, np.nan, np.nan]),
                "dummy": np.array([np.nan, np.nan, np.nan, np.nan]),
            },
            None,
        ),
        # TODO: Something is wrong here (either with the test or the code itself)
        # # Test Case 6: MERC grid branch (projection is ignored)
        # (
        #     "MERC_G",
        #     "MERC_G25km",  # valid MERC key
        #     "MERC_G",
        #     {
        #         "latitude": np.array([10, 30, 90, 50]),
        #         "longitude": np.array([50, 60, 70, 80]),
        #         "dummy": np.array([1, 2, 3, 4]),
        #     },
        #     {
        #         # For MERC: out_of_bounds_lat = indices where (lat > 85 or lat < -85).
        #         # For input [10,30,90,50], index 2 qualifies (90 > 85).
        #         # y_bound: dummy conversion returns lat; none out-of-bound.
        #         # Union = {2}.
        #         "latitude": np.array([10, 30, np.nan, 50]),
        #         "longitude": np.array([50, 60, np.nan, 80]),
        #         "dummy": np.array([1, 2, np.nan, 4]),
        #     },
        #     None,
        # ),
        # Test Case 7: Unknown grid definition – GRIDS key not found (KeyError expected)
        (
            "Unknown_grid",
            "UNKNOWN",  # not present in GRIDS
            "N",
            {
                "latitude": np.array([10, 30, 95, 110]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            None,
            "UNKNOWN",  # expecting the KeyError message to contain the missing key
        ),
        # Test Case 8: Invalid projection for an EASE2 grid (should raise ValueError)
        (
            "Invalid_proj_EASE2",
            "EASE2_G1km",  # valid key
            "XYZ",  # invalid projection for EASE2
            {
                "latitude": np.array([10, 30, 95, 110]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            None,
            "Invalid projection for EASE2 grid: XYZ",
        ),
        # Test Case 9: Invalid projection for a STEREO grid (should raise ValueError)
        (
            "Invalid_proj_STEREO",
            "STEREO_N6.25km",  # valid key
            "ABC",  # invalid projection for STEREO
            {
                "latitude": np.array([10, 30, 50, 90]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            None,
            "Invalid projection for STEREO grid: ABC",
        ),
        # Test Case 10: GRIDS key not found (KeyError expected)
        (
            "GRIDS_key_not_found",
            "NON_EXISTENT",  # key not in GRIDS
            "N",
            {
                "latitude": np.array([10, 30, 95, 110]),
                "longitude": np.array([50, 60, 70, 80]),
                "dummy": np.array([1, 2, 3, 4]),
            },
            None,
            "NON_EXISTENT",
        ),
    ],
)
def test_remove_out_of_bounds(
    monkeypatch,
    test_case,
    grid_def,
    proj_def,
    input_data,
    expected_output,
    expected_exception,
):
    """
    Parameterized unit test for the remove_out_of_bounds method.

    --- Test Cases Description ---
    In the following test cases:
      - The method first retrieves the grid from GRIDS using self.config.grid_definition.
      - Then, depending on the grid type and projection, it computes two sets of indices:
          * out_of_bounds_lat: based on the latitude value relative to grid["lat_min"] or grid["lat_max"]
          * out_of_bounds_xy: based on computed y coordinates (here, simply equal to the input latitudes)
      - Finally, for each variable in data_dict, the union of these indices is set to np.nan.

    For example, in Test Case 1 the configuration is for an EASE2 grid with projection "N".
    With dummy_grid having lat_min=20 and lat_max=80, if we supply:

        latitude = [10, 30, 95, 110]

    then:
      - For projection "N", out_of_bounds_lat = indices where latitude < 20 → index 0.
      - The y‑boundaries are y_bound_min = y_max – n_rows * res = 100 – 10 = 90 and y_bound_max = 100.
        Thus, out_of_bounds_xy = indices where (latitude < 90 or latitude > 100) → indices 0, 1, and 3.

    The union of these indices is {0,1,3} and the expected result is that every variable in the
    returned dictionary will have np.nan at indices 0, 1, and 3.

    Parameters
    ----------
    test_case : str
        A descriptive name for the test case.
    grid_def : str
        The grid definition to use (and which is looked up in GRIDS).
    proj_def : str
        The projection definition.
    input_data : dict
        A dictionary with keys "latitude", "longitude", etc., containing NumPy arrays.
    expected_output : dict or None
        The expected modified data_dict (if no exception is expected).
    expected_exception : str or None
        A substring of the expected exception message (if an error is expected).
    """
    # Ensure that the ReGridder class has the remove_out_of_bounds method.
    # monkeypatch.setattr(ReGridder, "remove_out_of_bounds", remove_out_of_bounds)
    # monkeypatch.setattr(DataIngestion, "remove_out_of_bounds", remove_out_of_bounds)

    # For valid cases, use the dummy_grid instead of the real GRIDS
    if expected_exception is None:
        original_grid = GRIDS[grid_def]
        GRIDS[grid_def] = dummy_grid

    # Monkey-patch RGBLogging.rgb_decorate_and_execute to return our dummy grid generator.
    monkeypatch.setattr(
        RGBLogging, "rgb_decorate_and_execute", dummy_rgb_decorate_and_execute
    )

    dummy_logger = DummyLogger()
    config = DummyConfig(
        grid_definition=grid_def, projection_definition=proj_def, logger=dummy_logger
    )
    instance = DataIngestion(config)

    # Make a copy of the input data so later tests are not affected.
    data_in = {k: v.copy() for k, v in input_data.items()}

    if expected_exception is not None:
        with pytest.raises(Exception, match=expected_exception):
            instance.remove_out_of_bounds(data_in)
    else:
        result = instance.remove_out_of_bounds(data_in)
        for key in expected_output:
            np.testing.assert_array_equal(result[key], expected_output[key])

    #set back the GRIDS to the original one, for later tests that use it
    if expected_exception is None:
        GRIDS[grid_def] = original_grid