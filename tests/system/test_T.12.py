# Testing script for T.12
# Remapping of L1b SMAP data with IDS (inverse distance squared) algorithm on an EASE2 global grid
# The remmapped data are compatible with SMAP L1c data obtained by NASA, with an average relative difference of brightness temperature < 0.25%

import sys
import os
import pathlib as pb
import subprocess as sbps
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


from numpy import array, full, nan
# import matplotlib

# tkagg = matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

# plt.ion()

import pytest

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io


GRID = "EASE2_G36km"
PROJECTION = "G"


def get_netcdf_data(path):
    """
    Extract and grid data from a netCDF file.

    Parameters
    ----------
    path : str
        Path to the netCDF file.

    Returns
    -------
    dict
        A dictionary containing gridded brightness temperature variables.
    """

    import netCDF4 as nc

    gridded_vars = {}

    with nc.Dataset(path, "r") as f:
        data = f[f"{PROJECTION}"]
        measurement = data["Measurement"]
        l_band = measurement["L_BAND"]

        for bt in ["bt_h_fore", "bt_h_aft", "bt_v_fore", "bt_v_aft"]:
            if "fore" in bt:
                row = array(l_band["cell_row_fore"][:])
                col = array(l_band["cell_col_fore"][:])
            elif "aft" in bt:
                row = array(l_band["cell_row_aft"][:])
                col = array(l_band["cell_col_aft"][:])
            else:
                row = ""
                col = ""
            var = array(l_band[bt][:])
            grid = full((GRIDS[GRID]["n_rows"], GRIDS[GRID]["n_cols"]), nan)

            for count, sample in enumerate(var):
                if sample == 9.969209968386869e36:
                    continue
                grid[row[count], col[count]] = sample
            gridded_vars[bt] = grid

        return gridded_vars


def get_hdf5_data(path):
    """
    Extract and grid data from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        A dictionary containing gridded brightness temperature variables.
    """

    import h5py

    gridded_vars = {}

    with h5py.File(path, "r") as f:
        data = f["Global_Projection"]
        row = data["cell_row"][:]
        col = data["cell_column"][:]

        bts = {
            "bt_h_fore": data["cell_tb_h_fore"][:],
            "bt_h_aft": data["cell_tb_h_aft"][:],
            "bt_v_fore": data["cell_tb_v_fore"][:],
            "bt_v_aft": data["cell_tb_v_aft"][:],
        }

        for bt in bts:
            var = array(bts[bt])
            grid = full((GRIDS[GRID]["n_rows"], GRIDS[GRID]["n_cols"]), nan)

            for count, sample in enumerate(var):
                grid[row[count], col[count]] = sample
            gridded_vars[bt] = grid

        return gridded_vars


@pytest.mark.parametrize("setup_paths", ["T_12"], indirect=True)
def test_T12_execution(setup_paths, run_subprocess):
    """
    Test the execution of the subprocess for the T_12 scenario.

    Parameters
    ----------
    setup_paths : dict
        Dictionary containing paths for the T_12 scenario, retrieved using the `setup_paths` fixture.
    run_subprocess : callable
        A callable fixture that executes a subprocess using the provided configuration path.

    Asserts
    -------
    exit_code : int
        Ensures that the subprocess exits with a code of 0, indicating success.
    """

    _, _, config_paths = setup_paths

    for config_path in config_paths:
        exit_code = run_subprocess(config_path)

        assert exit_code == 0, "Subprocess execution failed with a non-zero exit code."


@pytest.mark.parametrize("setup_paths", ["T_12"], indirect=True)
def test_T12_comparison(setup_paths, calculate_differences):
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
        The paths are dynamically retrieved based on the test scenario ("T_12").
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
    This test is parameterized for the "T_12" scenario:

    >>> pytest test_script.py --setup-paths=T_12

    Output:
    -------
    bt_h_fore: Average Mean Diff = 0.002, Average Percent Diff = 0.01%
    bt_h_aft: Average Mean Diff = 0.003, Average Percent Diff = 0.02%
    bt_v_fore: Average Mean Diff = 0.004, Average Percent Diff = 0.03%
    bt_v_aft: Average Mean Diff = 0.002, Average Percent Diff = 0.01%
    """

    rgb_data_path, nasa_data_path, _ = setup_paths
    rgb_data = get_netcdf_data(path=rgb_data_path)
    nasa_data = get_hdf5_data(nasa_data_path)

    variables_list = ["bt_h_fore", "bt_h_aft", "bt_v_fore", "bt_v_aft"]

    results = calculate_differences(
        data1=rgb_data, data2=nasa_data, variables_list=variables_list
    )

    for key, stats in results.items():
        print(
            f"{key}: Average Mean Diff = {stats['mean_diff']:.3f}, Average Percent Diff = {stats['percent_diff']:.3f}%"
        )
        assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
        assert stats["percent_diff"] < 0.5, f"Percent difference for {key} is too high!"
