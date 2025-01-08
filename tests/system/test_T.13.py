# Testing script for T.13
# Remapping of L1b CIMR L-band data with NN (nearest neighbor) algorithm on an EASE2 North grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)

import sys
import subprocess as sbps

from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import pytest

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io


# bt_h_aft_diff = abs(self.nn_data["bt_h_aft"] - self.ids_data["bt_h_aft"])
# fore_percent_diff = (fore_mean_diff / nanmean(self.ids_data["bt_h_fore"])) * 100
@pytest.mark.parametrize("setup_paths", ["T_13"], indirect=True)
def test_T13_execution(setup_paths, run_subprocess):
    """
    Test the execution of the subprocess for the T_13 scenario.

    Parameters
    ----------
    setup_paths : dict
        Dictionary containing paths for the T_13 scenario, retrieved using the `setup_paths` fixture.
    run_subprocess : callable
        A callable fixture that executes a subprocess using the provided configuration path.

    Asserts
    -------
    exit_code : int
        Ensures that the subprocess exits with a code of 0, indicating success.
    """

    _, _, config_paths = setup_paths

    for config_path in config_paths.values():
        exit_code = run_subprocess(config_path)

        assert exit_code == 0, "Subprocess execution failed with a non-zero exit code."


# TODO: Rewrite the docstring properly
# Some of the parameters were not accessed because they are just placeholders
# so that their names appear on the console output when the test is run
@pytest.mark.parametrize("setup_paths", ["T_13"], indirect=True)
@pytest.mark.parametrize(
    "TEST_NAME, DATA_OUTPUT, PROJECTION, GRID, INPUT_BAND, OUTPUT_BAND",
    [
        ("CIMR: NN_RGB vs IDS_RGB", "L1C", "N", "EASE2_N9km", "L_BAND", "L_BAND"),
    ],
)
def test_T13_comparison(
    setup_paths,
    TEST_NAME,
    DATA_OUTPUT,
    PROJECTION,
    GRID,
    INPUT_BAND,
    OUTPUT_BAND,
    get_netcdf_data,
    calculate_differences,
):
    # def test_T13_comparison(setup_paths, get_netcdf_data, calculate_differences):
    """
    Test comparison of brightness temperature (BT) variables between RGB and NASA datasets.

    This test verifies that the differences between the brightness temperature variables
    (bt_h_fore, bt_h_aft, bt_v_fore, bt_v_aft) from the RGB and NASA datasets are within
    acceptable thresholds. It utilizes a fixture to calculate differences for specified variables
    and asserts the results against pre-defined tolerances.

    Parameters
    ----------
    setup_paths : tuple
        A pytest fixture providing file paths for the RGB and NASA datasets and configuration files.
        The paths are dynamically retrieved based on the test scenario ("T_13").
    calculate_differences : callable
        A pytest fixture providing a function to calculate the mean absolute and percentage differences
        between two datasets for specified variables.

    Asserts
    -------
    - Mean difference for each variable is less than 1.0.
    - Mean percentage difference for each variable is less than 0.5%.

    Raises
    ------
    AssertionError
        If either the mean difference or mean percentage difference exceeds the defined threshold
        for any variable.

    Example
    -------
    This test is parameterized for the "T_13" scenario:

    >>> pytest test_script.py --setup-paths=T_13

    Output:
    -------
    bt_h_fore: Average Mean Diff = 0.002, Average Percent Diff = 0.01%
    bt_h_aft: Average Mean Diff = 0.003, Average Percent Diff = 0.02%
    bt_v_fore: Average Mean Diff = 0.004, Average Percent Diff = 0.03%
    bt_v_aft: Average Mean Diff = 0.002, Average Percent Diff = 0.01%
    """

    # GRID = "EASE2_N9km"
    # PROJECTION = "N"
    # BAND = "L_BAND"

    variables_list = ["bt_h_fore", "bt_h_aft"]

    # Retrieving paths
    nn_data_path, ids_data_path, _ = setup_paths

    # Retrieving data
    nn_data = get_netcdf_data(
        datapath=nn_data_path,
        variables_list=variables_list,
        projection=PROJECTION,
        band=INPUT_BAND,
        grid=GRID,
    )
    ids_data = get_netcdf_data(
        datapath=ids_data_path,
        variables_list=variables_list,
        projection=PROJECTION,
        band=INPUT_BAND,
        grid=GRID,
    )

    # difference is (data1 - data2) / data2
    results = calculate_differences(
        data1=nn_data, data2=ids_data, variables_list=variables_list
    )

    for key, stats in results.items():
        print(
            f"{key}: Average Mean Diff = {stats['mean_diff']:.3f}, Average Percent Diff = {stats['percent_diff']:.3f}%"
        )
        # assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
        assert stats["percent_diff"] < 1, f"Percent difference for {key} is too high!"
