# Testing Script for T.14
# Remapping of AMSR2 data with DIB (drop-in-the-bucket) algorithm on an EASE2 South polar grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)


import sys
import os
import pathlib as pb
import subprocess as sbps

import pytest


from numpy import array, full, nan, nanmean, polyfit, isnan, isinf

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io


BANDS = "10_BAND", "23_BAND", "89a_BAND"
PROJECTION = "6_BAND_TARGET"


@pytest.mark.parametrize("setup_paths", ["T_31"], indirect=True)
def test_T31_execution(setup_paths, run_subprocess):
    """
    Test the execution of the subprocess for the T_31 scenario.

    Parameters
    ----------
    setup_paths : dict
        Dictionary containing paths for the T_31 scenario, retrieved using the `setup_paths` fixture.
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
@pytest.mark.parametrize("setup_paths", ["T_31"], indirect=True)
@pytest.mark.parametrize(
    "TEST_NAME, DATA_OUTPUT, PROJECTION, GRID, INPUT_BAND, OUTPUT_BAND",
    [
        (
            "AMSR2: DIB_RGB vs IDS_RGB",
            "L1R",
            "6_BAND_TARGET",
            "6_BAND_TARGET",
            "10_BAND",
            "6_BAND",
        ),
        (
            "AMSR2: DIB_RGB vs IDS_RGB",
            "L1R",
            "6_BAND_TARGET",
            "6_BAND_TARGET",
            "23_BAND",
            "6_BAND",
        ),
        (
            "AMSR2: DIB_RGB vs IDS_RGB",
            "L1R",
            "6_BAND_TARGET",
            "6_BAND_TARGET",
            "89a_BAND",
            "6_BAND",
        ),
    ],
)
def test_T31_comparison(
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
    variables_list = ["bt_h"]

    # Retrieving paths
    datapath1, datapath2, _ = setup_paths

    # Retrieving data
    data1 = get_netcdf_data(
        datapath=datapath1,
        variables_list=variables_list,
        projection=PROJECTION,
        band=INPUT_BAND,
        grid=GRID,
    )
    data2 = get_netcdf_data(
        datapath=datapath2,
        variables_list=variables_list,
        projection=PROJECTION,
        band=INPUT_BAND,
        grid=GRID,
    )

    # difference is (data1 - data2) / data2
    results = calculate_differences(
        data1=data1, data2=data2, variables_list=variables_list
    )

    for key, stats in results.items():
        print(
            f"{key}: Average Mean Diff = {stats['mean_diff']:.3f}, Average Percent Diff = {stats['percent_diff']:.3f}%"
        )
        # assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
        assert stats["percent_diff"] < 1, f"Percent difference for {key} is too high!"
