# Testing Script for T.14
# Remapping of AMSR2 data with DIB (drop-in-the-bucket) algorithm on an EASE2 South polar grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)


import sys
import os
import pathlib as pb
import subprocess as sbps

import pytest
import matplotlib.pyplot as plt
from numpy import array, full, nan, nanmean, polyfit, isnan, isinf

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io


BANDS = "10_BAND", "23_BAND", "89a_BAND"
PROJECTION = "6_BAND_TARGET"

# GLobal dictionary to store test results for plotting purposes
test_results = {}


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
    # GLobal dictionary to store test results for plotting purposes
    global test_results

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

    # Plotting stuff
    imgname = f"T_31_difference_{INPUT_BAND}.png"
    map_compare(
        dib_data=data1,
        ids_data=data2,
        input_band=INPUT_BAND,
        output_band=OUTPUT_BAND,
        imgname=imgname,
    )

    # difference is (data1 - data2) / data2
    results = calculate_differences(
        data1=data1, data2=data2, variables_list=variables_list
    )

    # Save results to the global dictionary
    test_results[INPUT_BAND] = {
        "dib_data": data1,
        "ids_data": data2,
        "results": results,
    }

    for key, stats in results.items():
        print(
            f"{key}: Average Mean Diff = {stats['mean_diff']:.3f}, Average Percent Diff = {stats['percent_diff']:.3f}%"
        )
        # assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
        assert stats["percent_diff"] < 1, f"Percent difference for {key} is too high!"


def map_compare(dib_data, ids_data, input_band, output_band, imgname):
    bt_h_ids_10 = ids_data[f"bt_h"]
    # bt_h_ids_23 = ids_data["bt_h_23_BAND"]
    # bt_h_ids_89a = ids_data["bt_h_89a_BAND"]
    bt_h_dib_10 = dib_data[f"bt_h"]
    # bt_h_dib_23 = dib_data["bt_h_23_BAND"]
    # bt_h_dib_89a = dib_data["bt_h_89a_BAND"]

    cmap = "viridis"
    # bt_h plt
    # ---------------------- 10_BAND ----------------------
    fig, axs = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)
    fig.suptitle(f"{input_band} -- > {output_band}")
    plt.subplots_adjust(wspace=0.01)
    im00 = axs[0].imshow(bt_h_dib_10[:, :, 0], cmap=cmap)
    axs[0].set_title("DIB Remap (bt_h)")
    fig.colorbar(im00, ax=axs[0])

    im01 = axs[1].imshow(bt_h_ids_10[:, :, 0], cmap=cmap)
    axs[1].set_title("IDS Remap (bt_h)")
    fig.colorbar(im01, ax=axs[1])

    bt_h_diff = abs(bt_h_ids_10 - bt_h_dib_10)
    im02 = axs[2].imshow(bt_h_diff[:, :, 0], cmap=cmap)
    axs[2].set_title("Difference (bt_h)")

    fig.colorbar(im02, ax=axs[2])

    # plt.show()

    # Add Statistics
    # Calculate the average relative difference
    fore_mean_diff = nanmean(bt_h_diff)

    print(
        f"Average relative difference for bt_h for band {input_band}: {fore_mean_diff}"
    )

    # Calculate percentage Differences
    fore_percent_diff = (fore_mean_diff / nanmean(bt_h_ids_10)) * 100

    print(
        f"Average percentage difference for bt_h  for band {input_band}: {fore_percent_diff}"
    )

    # Add statistics to the plot
    # axs[0,2].text(100,300, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
    # axs[0,2].text(100, 300, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
    # fig.text(
    #     0.7,
    #     0.5,
    #     rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{DIB}}_i |$",
    #     fontsize=14,
    #     color="black",
    # )

    # fig.text(
    #     0.7,
    #     0.4,
    #     rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
    #     fontsize=14,
    #     color="black",
    # )

    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(
        f"output/system/T_31/{imgname}"  # T_14_difference1.png"
    )  # ""
    plt.savefig(img_path, dpi=300)


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


# def scatter_compare(dib_data, ids_data):
#     x_10 = dib_data["bt_h_10_BAND"].flatten()
#     y_10 = ids_data["bt_h_10_BAND"].flatten()
# x_h_10, y_h_10, m_h_10, b_h_10, y_fit_h_10, r_squared_10 = scatter_stats(x_10, y_10)

# x_23 = dib_data["bt_h_23_BAND"].flatten()
# y_23 = ids_data["bt_h_23_BAND"].flatten()
# x_h_23, y_h_23, m_h_23, b_h_23, y_fit_h_23, r_squared_23 = self.scatter_stats(
#     x_23, y_23
# )

# x_89a = dib_data["bt_h_89a_BAND"].flatten()
# y_89a = ids_data["bt_h_89a_BAND"].flatten()
# x_h_89a, y_h_89a, m_h_89a, b_h_89a, y_fit_h_89a, r_squared_89a = self.scatter_stats(
#     x_89a, y_89a
# )

# fig, axs = plt.subplots(1, 3, figsize=(20, 12))
# plt.suptitle("L1R AMSR2 Remap to the footprints of Band 6")
# axs[0].scatter(x_h_10, y_h_10)
# axs[0].plot(x_h_10, y_fit_h_10, color="red")
# axs[0].legend(title=f"$R^2 = {r_squared_10:.3f}$")
# axs[0].set_title("bt_h (Band 10)")
# axs[0].set_xlabel("DIB BT [K]")
# axs[0].set_ylabel("IDS BT [K]")

# axs[1].scatter(x_h_23, y_h_23)
# axs[1].plot(x_h_23, y_fit_h_23, color="red")
# axs[1].legend(title=f"$R^2 = {r_squared_23:.3f}$")
# axs[1].set_title("bt_h (Band 23)")
# axs[1].set_xlabel("DIB BT [K]")
# axs[1].set_ylabel("IDS BT [K]")

# axs[2].scatter(x_h_89a, y_h_89a)
# axs[2].plot(x_h_89a, y_fit_h_89a, color="red")
# axs[2].legend(title=f"$R^2 = {r_squared_89a:.3f}$")
# axs[2].set_title("bt_h (Band 89a)")
# axs[2].set_xlabel("DIB BT [K]")
# axs[2].set_ylabel("IDS BT [K]")

# repo_root = grasp_io.find_repo_root()
# img_path = repo_root.joinpath(
#     "output/MS3_verification_tests/T_31/T_31_scatter.png"
# )  # ""
# plt.savefig(img_path, dpi=300)
# plt.show()


def plot_all_results(results, imgname="T_31_scatter_combined.png"):
    fig, axs = plt.subplots(1, 3, figsize=(20, 12))
    plt.suptitle("L1R AMSR2 Remap")

    bands = ["7_BAND", "18_BAND", "89a_BAND"]
    for idx, band in enumerate(bands):
        if band in results:
            dib_data = results[band]["dib_data"]
            ids_data = results[band]["ids_data"]

            x = dib_data["bt_h"].flatten()
            y = ids_data["bt_h"].flatten()
            x_h, y_h, m_h, b_h, y_fit_h, r_squared = scatter_stats(x, y)

            axs[idx].scatter(x_h, y_h)
            axs[idx].plot(x_h, y_fit_h, color="red")
            axs[idx].legend(title=f"$R^2 = {r_squared:.3f}$")
            axs[idx].set_title(f"bt_h ({band})")
            axs[idx].set_xlabel("DIB BT [K]")
            axs[idx].set_ylabel("IDS BT [K]")

    # Save the combined plot
    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(f"output/system/T_31/{imgname}")
    plt.savefig(img_path, dpi=300)


def test_plot_results():
    """
    Dummy test to plot results into one plot
    """
    global test_results
    plot_all_results(test_results)
