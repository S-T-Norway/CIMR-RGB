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


import numpy as np
from numpy import array, full, nan
# import matplotlib

# tkagg = matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# plt.ion()

import pytest

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io


GRID = "EASE2_G36km"
PROJECTION = "G"


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
@pytest.mark.parametrize(
    "TEST_NAME, DATA_OUTPUT, PROJECTION, GRID, INPUT_BAND, OUTPUT_BAND",
    [
        ("SMAP: IDS_RGB vs IDS_NASA", "L1C", "G", "EASE2_G36km", "L_BAND", "L_BAND"),
    ],
)
def test_T12_comparison(
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

    variables_list = ["bt_h_fore", "bt_h_aft", "bt_v_fore", "bt_v_aft"]

    rgb_data_path, nasa_data_path, _ = setup_paths

    rgb_data = get_netcdf_data(
        datapath=rgb_data_path,
        variables_list=variables_list,
        projection=PROJECTION,
        band=INPUT_BAND,
        grid=GRID,
    )

    nasa_data = get_hdf5_data(nasa_data_path)

    results = calculate_differences(
        data1=rgb_data, data2=nasa_data, variables_list=variables_list
    )

    # Plotting comparison
    # repo_root = grasp_io.find_repo_root()
    # img_path = repo_root.joinpath(
    #     "output/MS3_verification_tests/T_12/T_12_difference2.png"
    # )
    map_compare(data1=rgb_data, data2=nasa_data)
    scatter_compare(data1=rgb_data, data2=nasa_data)

    for key, stats in results.items():
        print(
            f"{key}: Average Mean Diff = {stats['mean_diff']:.3f}, Average Percent Diff = {stats['percent_diff']:.3f}%"
        )
        # assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
        assert stats["percent_diff"] < 0.25, (
            f"Percent difference for {key} is too high!"
        )


def map_compare_new(
    data1,
    data2,
    variables,
    filenames,
    difference,
    rows=2,
    cols=3,
    figsize=(20, 20),
):
    #
    #
    #

    cmap = "viridis"

    # bt_h plt
    fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)

    # The number of rows is equal to aft fore
    for variable in variables:
        for row in range(rows):
            # for col in range(cols):
            im00 = axs[row, 0].imshow(data1[variable][:, 550:], cmap=cmap)
            axs[row, 0].set_title(f"RGB Remap ({variable})")
            om01 = axs[row, 1].imshow(data2[variable][:, 550:], cmap=cmap)
            axs[row, 1].set_title(f"NASA Remap ({variable})")
            #
            im02 = axs[0, 2].imshow(
                difference[variable]["diff"][:, 550:],
                cmap=cmap,
            )
            axs[row, 2].set_title(f"Difference ({variable})")
            fig.colorbar(im02, ax=axs[0, 2])
    #
    axs[0, 2].text(
        50,
        50,
        r"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{RGB}}_i - \mathrm{{NASA}}_i |$",
        fontsize=14,
        color="black",
    )

    axs[0, 2].text(
        50,
        100,
        rf"$\mu_{{fore}} =  {difference['bt_h_fore']['mean_diff']:.2f} K, \ \text{{or}} \ {difference['bt_h_fore']['percent_diff']:.2f}\%$",
        fontsize=14,
        color="black",
    )

    axs[1, 2].text(
        50,
        50,
        r"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{RGB}}_i - \mathrm{{NASA}}_i |$",
        fontsize=14,
        color="black",
    )
    axs[1, 2].text(
        50,
        100,
        rf"$\mu_{{aft}} =  {difference['bt_h_aft']['mean_diff']:.2f} K, \ \text{{or}} \ {difference['bt_h_aft']['percent_diff']:.2f}\%$",
        fontsize=14,
        color="black",
    )

    # fore
    # im00 = axs[0, 0].imshow(data1["bt_h_fore"][:, 550:], cmap=cmap)
    # axs[0, 0].set_title("RGB Remap (bt_h_fore)")

    # om01 = axs[0, 1].imshow(data2["bt_h_fore"][:, 550:], cmap=cmap)
    # axs[0, 1].set_title("NASA Remap (bt_h_fore)")

    # bt_h_fore_diff = abs(data1["bt_h_fore"] - data2["bt_h_fore"])

    # print(difference["bt_h_fore"]["mean_diff"])
    # im02 = axs[0, 2].imshow(
    #     difference["bt_h_fore"]["diff"][:, 550:],
    #     cmap=cmap,
    # )
    # axs[0, 2].set_title("Difference (bt_h_fore)")

    # difference["bt_h_fore"]["percent_diff"]

    # ----------------------
    # bt_h
    # ----------------------

    # aft
    # im10 = axs[1, 0].imshow(data1["bt_h_aft"][:, 550:], cmap=cmap)
    # axs[1, 0].set_title("RGB Remap (bt_h_aft)")

    # im11 = axs[1, 1].imshow(data2["bt_h_aft"][:, 550:], cmap=cmap)
    # axs[1, 1].set_title("NASA Remap (bt_h_aft)")

    # im12 = axs[1, 2].imshow(difference["bt_h_aft"]["diff"][:, 550:], cmap=cmap)
    # axs[1, 2].set_title("Difference (bt_h_aft)")

    # #
    # fig.colorbar(im02, ax=axs[0, 2])
    # fig.colorbar(im12, ax=axs[1, 2])

    # Add Statistics
    # Calculate the average relative difference
    # fore_mean_diff = np.nanmean(bt_h_fore_diff)
    # aft_mean_diff = np.nanmean(bt_h_aft_diff)
    # print(f"Average relative difference for bt_h_fore: {fore_mean_diff}")
    # print(f"Average relative difference for bt_h_aft: {aft_mean_diff}")

    print(
        f"Average relative difference for bt_h_fore: {difference['bt_h_fore']['mean_diff']}"
    )
    print(
        f"Average relative difference for bt_h_aft: {difference['bt_h_aft']['percent_diff']}"
    )
    # Add statistics to the plot
    # axs[0,2].text(50,50, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
    # axs[0,2].text(50, 50, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
    axs[0, 2].text(
        50,
        50,
        r"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{RGB}}_i - \mathrm{{NASA}}_i |$",
        fontsize=14,
        color="black",
    )

    axs[0, 2].text(
        50,
        100,
        rf"$\mu_{{fore}} =  {difference['bt_h_fore']['mean_diff']:.2f} K, \ \text{{or}} \ {difference['bt_h_fore']['percent_diff']:.2f}\%$",
        fontsize=14,
        color="black",
    )

    axs[1, 2].text(
        50,
        50,
        r"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{RGB}}_i - \mathrm{{NASA}}_i |$",
        fontsize=14,
        color="black",
    )
    axs[1, 2].text(
        50,
        100,
        rf"$\mu_{{aft}} =  {difference['bt_h_aft']['mean_diff']:.2f} K, \ \text{{or}} \ {difference['bt_h_aft']['percent_diff']:.2f}\%$",
        fontsize=14,
        color="black",
    )
    # img_path = repo_root.joinpath(
    #     "output/MS3_verification_tests/T_12/T_12_difference1.png"
    # )  # ""
    # plt.savefig(img_path, dpi=300)
    # plt.show()
    plt.savefig(filenames[0], dpi=300)
    # plt.show()

    # ----------------------
    # bt_v
    # ----------------------

    #

    #

    #

    #


def map_compare(data1, data2):
    cmap = "viridis"
    # bt_h plt
    fig, axs = plt.subplots(2, 3, figsize=(20, 20), constrained_layout=True)
    im00 = axs[0, 0].imshow(data1["bt_h_fore"][:, 550:], cmap=cmap)
    axs[0, 0].set_title("RGB Remap (bt_h_fore)")
    om01 = axs[0, 1].imshow(data2["bt_h_fore"][:, 550:], cmap=cmap)
    axs[0, 1].set_title("NASA Remap (bt_h_fore)")
    bt_h_fore_diff = abs(data1["bt_h_fore"] - data2["bt_h_fore"])
    im02 = axs[0, 2].imshow(bt_h_fore_diff[:, 550:], cmap=cmap)
    axs[0, 2].set_title("Difference (bt_h_fore)")
    # aft
    im10 = axs[1, 0].imshow(data1["bt_h_aft"][:, 550:], cmap=cmap)
    axs[1, 0].set_title("RGB Remap (bt_h_aft)")
    im11 = axs[1, 1].imshow(data2["bt_h_aft"][:, 550:], cmap=cmap)
    axs[1, 1].set_title("NASA Remap (bt_h_aft)")
    bt_h_aft_diff = abs(data1["bt_h_aft"] - data2["bt_h_aft"])
    im12 = axs[1, 2].imshow(bt_h_aft_diff[:, 550:], cmap=cmap)
    axs[1, 2].set_title("Difference (bt_h_aft)")
    fig.colorbar(im02, ax=axs[0, 2])
    fig.colorbar(im12, ax=axs[1, 2])

    # Add Statistics
    # Calculate the average relative difference
    fore_mean_diff = np.nanmean(bt_h_fore_diff)
    aft_mean_diff = np.nanmean(bt_h_aft_diff)
    print(f"Average relative difference for bt_h_fore: {fore_mean_diff}")
    print(f"Average relative difference for bt_h_aft: {aft_mean_diff}")

    # Calculate percentage Differences
    fore_percent_diff = (fore_mean_diff / np.nanmean(data2["bt_h_fore"])) * 100
    aft_percent_diff = (aft_mean_diff / np.nanmean(data2["bt_h_aft"])) * 100
    print(f"Average percentage difference for bt_h_fore: {fore_percent_diff}")
    print(f"Average percentage difference for bt_h_aft: {aft_percent_diff}")

    # Add statistics to the plot
    # axs[0,2].text(50,50, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
    # axs[0,2].text(50, 50, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
    axs[0, 2].text(
        50,
        50,
        rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{RGB}}_i - \mathrm{{NASA}}_i |$",
        fontsize=14,
        color="black",
    )

    axs[0, 2].text(
        50,
        100,
        rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
        fontsize=14,
        color="black",
    )

    axs[1, 2].text(
        50,
        50,
        rf"$\mu_{{aft}} =  {aft_mean_diff:.2f} K, \ \text{{or}} \ {aft_percent_diff:.2f}\%$",
        fontsize=14,
        color="black",
    )
    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(
        "output/system/T_12/T_12_difference1.png"
    )  # ""
    plt.savefig(img_path, dpi=300)
    # plt.show()

    # bt_v plt
    fig, axs = plt.subplots(2, 3, figsize=(20, 20), constrained_layout=True)
    im00 = axs[0, 0].imshow(data1["bt_v_fore"][:, 550:], cmap=cmap)
    axs[0, 0].set_title("RGB Remap (bt_v_fore)")
    om01 = axs[0, 1].imshow(data2["bt_v_fore"][:, 550:], cmap=cmap)
    axs[0, 1].set_title("NASA Remap (bt_v_fore)")
    bt_v_fore_diff = abs(data1["bt_v_fore"] - data2["bt_v_fore"])
    im02 = axs[0, 2].imshow(bt_v_fore_diff[:, 550:], cmap=cmap)
    axs[0, 2].set_title("Difference (bt_v_fore)")
    # aft
    im10 = axs[1, 0].imshow(data1["bt_v_aft"][:, 550:], cmap=cmap)
    axs[1, 0].set_title("RGB Remap (bt_v_aft)")
    im11 = axs[1, 1].imshow(data2["bt_v_aft"][:, 550:], cmap=cmap)
    axs[1, 1].set_title("NASA Remap (bt_v_aft)")
    bt_v_aft_diff = abs(data1["bt_v_aft"] - data2["bt_v_aft"])
    im12 = axs[1, 2].imshow(bt_v_aft_diff[:, 550:], cmap=cmap)
    axs[1, 2].set_title("Difference (bt_v_aft)")
    fig.colorbar(im02, ax=axs[0, 2])
    fig.colorbar(im12, ax=axs[1, 2])

    # Add Statistics
    # Calculate the average relative difference
    fore_mean_diff = np.nanmean(bt_v_fore_diff)
    aft_mean_diff = np.nanmean(bt_h_aft_diff)
    print(f"Average relative difference for bt_v_fore: {fore_mean_diff}")
    print(f"Average relative difference for bt_v_aft: {aft_mean_diff}")

    # Calculate percentage Differences
    fore_percent_diff = (fore_mean_diff / np.nanmean(data2["bt_v_fore"])) * 100
    aft_percent_diff = (aft_mean_diff / np.nanmean(data2["bt_v_aft"])) * 100
    print(f"Average percentage difference for bt_v_fore: {fore_percent_diff}")
    print(f"Average percentage difference for bt_v_aft: {aft_percent_diff}")

    # Add statistics to the plot
    # axs[0,2].text(50,50, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
    # axs[0,2].text(50, 50, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
    axs[0, 2].text(
        50,
        50,
        rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{RGB}}_i - \mathrm{{NASA}}_i |$",
        fontsize=14,
        color="black",
    )

    axs[0, 2].text(
        50,
        100,
        rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
        fontsize=14,
        color="black",
    )

    axs[1, 2].text(
        50,
        50,
        rf"$\mu_{{aft}} =  {aft_mean_diff:.2f} K, \ \text{{or}} \ {aft_percent_diff:.2f}\%$",
        fontsize=14,
        color="black",
    )
    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(
        # "output/MS3_verification_tests/T_12/T_12_difference2.png"
        "output/system/T_12/T_12_difference2.png"
    )  # ""
    plt.savefig(img_path, dpi=300)
    # plt.show()


def scatter_stats(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    x = x[mask]
    y = y[mask]
    m, b = np.polyfit(x, y, 1)
    y_fit = m * x + b

    # Calculate R^2
    ss_res = sum((y - y_fit) ** 2)
    ss_tot = sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return x, y, m, b, y_fit, r_squared


def scatter_compare(data1, data2):
    x = data1["bt_h_fore"].flatten()
    y = data2["bt_h_fore"].flatten()
    x_h_fore, y_h_fore, m_h_fore, b_h_fore, y_fit_h_fore, r_squared = scatter_stats(
        x, y
    )

    x = data1["bt_h_aft"].flatten()
    y = data2["bt_h_aft"].flatten()
    x_h_aft, y_h_aft, m_h_aft, b_h_aft, y_fit_h_aft, r_squared = scatter_stats(x, y)

    x = data1["bt_v_fore"].flatten()
    y = data2["bt_v_fore"].flatten()
    x_v_fore, y_v_fore, m_v_fore, b_v_fore, y_fit_v_fore, r_squared = scatter_stats(
        x, y
    )

    x = data1["bt_v_aft"].flatten()
    y = data2["bt_v_aft"].flatten()

    x_v_aft, y_v_aft, m_v_aft, b_v_aft, y_fit_v_aft, r_squared = scatter_stats(x, y)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[0, 0].scatter(x_h_fore, y_h_fore)
    axs[0, 0].plot(x_h_fore, y_fit_h_fore, color="red")
    axs[0, 0].legend(title=f"$R^2 = {r_squared:.3f}$")
    axs[0, 0].set_title("bt_h_fore")
    axs[0, 0].set_xlabel("RGB BT [K]")
    axs[0, 0].set_ylabel("NASA BT [K]")

    axs[0, 1].scatter(x_h_aft, y_h_aft)
    axs[0, 1].plot(x_h_aft, y_fit_h_aft, color="red")
    axs[0, 1].legend(title=f"$R^2 = {r_squared:.3f}$")
    axs[0, 1].set_title("bt_h_aft")
    axs[0, 1].set_xlabel("RGB BT [K]")
    axs[0, 1].set_ylabel("NASA BT [K]")

    axs[1, 0].scatter(x_v_fore, y_v_fore)
    axs[1, 0].plot(x_v_fore, y_fit_v_fore, color="red")
    axs[1, 0].legend(title=f"$R^2 = {r_squared:.3f}$")
    axs[1, 0].set_title("bt_v_fore")
    axs[1, 0].set_xlabel("RGB BT [K]")
    axs[1, 0].set_ylabel("NASA BT [K]")

    axs[1, 1].scatter(x_v_aft, y_v_aft)
    axs[1, 1].plot(x_v_aft, y_fit_v_aft, color="red")
    axs[1, 1].legend(title=f"$R^2 = {r_squared:.3f}$")
    axs[1, 1].set_title("bt_v_aft")
    axs[1, 1].set_xlabel("RGB BT [K]")
    axs[1, 1].set_ylabel("NASA BT [K]")

    repo_root = grasp_io.find_repo_root()
    print(repo_root)
    img_path = repo_root.joinpath(
        "output/system/T_12/T_12_scatter.png"
    )  # ""
    plt.savefig(img_path, dpi=300)

    # plt.show()
