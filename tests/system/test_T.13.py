# Testing script for T.13
# Remapping of L1b CIMR L-band data with NN (nearest neighbor) algorithm on an EASE2 North grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)

import sys
import subprocess as sbps

from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import pytest
import matplotlib.pyplot as plt

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io


# bt_h_aft_diff = abs(data1["bt_h_aft"] - data2["bt_h_aft"])
# fore_percent_diff = (fore_mean_diff / nanmean(data2["bt_h_fore"])) * 100
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

    map_compare(data1=nn_data, data2=ids_data)
    scatter_compare(data1=nn_data, data2=ids_data)

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


def map_compare(data1, data2):
    cmap = "viridis"
    # bt_h plt
    fig, axs = plt.subplots(2, 3, figsize=(20, 20), constrained_layout=True)
    im00 = axs[0, 0].imshow(data1["bt_h_fore"][750:1250, 0:400], cmap=cmap)
    axs[0, 0].set_title("NN Remap (bt_h_fore)")
    im01 = axs[0, 1].imshow(data2["bt_h_fore"][750:1250, 0:400], cmap=cmap)
    axs[0, 1].set_title("IDS Remap (bt_h_fore)")
    bt_h_fore_diff = abs(data1["bt_h_fore"] - data2["bt_h_fore"])
    im02 = axs[0, 2].imshow(bt_h_fore_diff[750:1250, 0:400], cmap=cmap)
    axs[0, 2].set_title("Difference (bt_h_fore)")
    # aft
    im10 = axs[1, 0].imshow(data1["bt_h_aft"][750:1250, 0:400], cmap=cmap)
    axs[1, 0].set_title("NN Remap (bt_h_aft)")
    im11 = axs[1, 1].imshow(data2["bt_h_aft"][750:1250, 0:400], cmap=cmap)
    axs[1, 1].set_title("IDS Remap (bt_h_aft)")
    bt_h_aft_diff = abs(data1["bt_h_aft"] - data2["bt_h_aft"])
    im12 = axs[1, 2].imshow(bt_h_aft_diff[750:1250, 0:400], cmap=cmap)
    axs[1, 2].set_title("Difference (bt_h_aft)")
    fig.colorbar(im02, ax=axs[0])
    fig.colorbar(im12, ax=axs[1])

    # Add Statistics
    # Calculate the average relative difference
    fore_mean_diff = nanmean(bt_h_fore_diff)
    aft_mean_diff = nanmean(bt_h_aft_diff)
    print(f"Average relative difference for bt_h_fore: {fore_mean_diff}")
    print(f"Average relative difference for bt_h_aft: {aft_mean_diff}")

    # Calculate percentage Differences
    fore_percent_diff = (fore_mean_diff / nanmean(data2["bt_h_fore"])) * 100
    aft_percent_diff = (aft_mean_diff / nanmean(data2["bt_h_aft"])) * 100
    print(f"Average percentage difference for bt_h_fore: {fore_percent_diff}")
    print(f"Average percentage difference for bt_h_aft: {aft_percent_diff}")

    # Add statistics to the plot
    # axs[0,2].text(100,300, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
    # axs[0,2].text(100, 300, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
    axs[0, 2].text(
        100,
        420,
        rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{NN}}_i |$",
        fontsize=14,
        color="black",
    )

    axs[0, 2].text(
        100,
        470,
        rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
        fontsize=14,
        color="black",
    )

    axs[1, 2].text(
        100,
        420,
        rf"$\mu_{{aft}} =  {aft_mean_diff:.2f} K, \ \text{{or}} \ {aft_percent_diff:.2f}\%$",
        fontsize=14,
        color="black",
    )
    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(
        "output/MS3_verification_tests/T_13/T_13_difference1.png"
    )  # ""
    plt.savefig(img_path, dpi=300)
    # plt.show()


def scatter_stats(x, y):
    mask = ~isnan(x) & ~isnan(y) & ~isinf(x) & ~isinf(y)
    x = x[mask]
    y = y[mask]
    m, b = polyfit(x, y, 1)
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

    fig, axs = plt.subplots(1, 2, figsize=(20, 12))
    axs[0].scatter(x_h_fore, y_h_fore)
    axs[0].plot(x_h_fore, y_fit_h_fore, color="red")
    axs[0].legend(title=f"$R^2 = {r_squared:.3f}$")
    axs[0].set_title("bt_h_fore")
    axs[0].set_xlabel("NN BT [K]")
    axs[0].set_ylabel("IDS BT [K]")

    axs[1].scatter(x_h_aft, y_h_aft)
    axs[1].plot(x_h_aft, y_fit_h_aft, color="red")
    axs[1].legend(title=f"$R^2 = {r_squared:.3f}$")
    axs[1].set_title("bt_h_aft")
    axs[1].set_xlabel("NN BT [K]")
    axs[1].set_ylabel("IDS BT [K]")

    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(
        "output/MS3_verification_tests/T_13/T_13_scatter.png"
    )  # ""
    plt.savefig(img_path, dpi=300)
    # plt.show()
