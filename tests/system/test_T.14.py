import sys
import subprocess as sbps

from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import pytest
import matplotlib.pyplot as plt

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io

BANDS = "7_BAND", "18_BAND", "89b_BAND"
PROJECTION = "S"
GRID = "EASE2_S9km"

# GLobal dictionary to store test results for plotting purposes
test_results = {}


@pytest.mark.parametrize("setup_paths", ["T_14"], indirect=True)
def test_T14_execution(setup_paths, run_subprocess):
    """
    Test the execution of the subprocess for the T_14 scenario.

    Parameters
    ----------
    setup_paths : dict
        Dictionary containing paths for the T_14 scenario, retrieved using the `setup_paths` fixture.
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
@pytest.mark.parametrize("setup_paths", ["T_14"], indirect=True)
@pytest.mark.parametrize(
    "TEST_NAME, DATA_OUTPUT, PROJECTION, GRID, INPUT_BAND, OUTPUT_BAND",
    [
        ("AMSR2: DIB_RGB vs IDS_RGB", "L1C", "S", "EASE2_S9km", "7_BAND", "7_BAND"),
        ("AMSR2: DIB_RGB vs IDS_RGB", "L1C", "S", "EASE2_S9km", "18_BAND", "18_BAND"),
        ("AMSR2: DIB_RGB vs IDS_RGB", "L1C", "S", "EASE2_S9km", "89b_BAND", "89b_BAND"),
    ],
)
def test_T14_comparison(
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
    # BANDS = "7_BAND", "18_BAND", "89b_BAND"
    # PROJECTION = "S"
    # GRID = "EASE2_S9km"

    # GLobal dictionary to store test results for plotting purposes
    global test_results

    variables_list = ["bt_h"]

    # Retrieving paths
    nn_data_path, ids_data_path, _ = setup_paths

    # Retrieving data
    dib_data = get_netcdf_data(
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

    # Plotting stuff
    imgname = f"T_14_difference_{OUTPUT_BAND}.png"
    map_compare(dib_data=dib_data, ids_data=ids_data, band=OUTPUT_BAND, imgname=imgname)
    # imgname = f"T_14_scatter_{OUTPUT_BAND}.png"
    # scatter_compare(
    #     dib_data=dib_data, ids_data=ids_data, band=OUTPUT_BAND, imgname=imgname
    # )

    # difference is (data1 - data2) / data2
    results = calculate_differences(
        data1=dib_data, data2=ids_data, variables_list=variables_list
    )

    # Save results to the global dictionary
    test_results[OUTPUT_BAND] = {
        "dib_data": dib_data,
        "ids_data": ids_data,
        "results": results,
    }

    for key, stats in results.items():
        print(
            f"{key}: Average Mean Diff = {stats['mean_diff']:.3f}, Average Percent Diff = {stats['percent_diff']:.3f}%"
        )
        # assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
        assert stats["percent_diff"] < 1, f"Percent difference for {key} is too high!"


def map_compare(dib_data, ids_data, band, imgname):
    bt_h_ids_7 = ids_data["bt_h"]
    # bt_h_ids_18 = ids_data["18_BAND"]["bt_h"]
    # bt_h_ids_89b = ids_data["89b_BAND"]["bt_h"]
    bt_h_dib_7 = dib_data["bt_h"]
    # bt_h_dib_18 = dib_data["18_BAND"]["bt_h"]
    # bt_h_dib_89b = dib_data["89b_BAND"]["bt_h"]

    cmap = "viridis"
    # bt_h plt
    # ---------------------- 7_BAND ----------------------
    fig, axs = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)
    fig.suptitle(f"AMSR2: L1C, {band}")
    plt.subplots_adjust(wspace=0.01)
    im00 = axs[0].imshow(bt_h_dib_7, cmap=cmap)
    axs[0].set_title("DIB Remap (bt_h)")
    fig.colorbar(im00, ax=axs[0])

    im01 = axs[1].imshow(bt_h_ids_7, cmap=cmap)
    axs[1].set_title("IDS Remap (bt_h)")
    fig.colorbar(im01, ax=axs[1])

    bt_h_diff = abs(bt_h_ids_7 - bt_h_dib_7)
    im02 = axs[2].imshow(bt_h_diff, cmap=cmap)
    axs[2].set_title("Difference (bt_h)")
    fig.colorbar(im02, ax=axs[2])

    # plt.show()

    # Add Statistics
    # Calculate the average relative difference
    fore_mean_diff = nanmean(bt_h_diff)

    print(f"Average relative difference for bt_h for {band}: {fore_mean_diff}")

    # Calculate percentage Differences
    fore_percent_diff = (fore_mean_diff / nanmean(ids_data["bt_h"])) * 70

    print(f"Average percentage difference for bt_h for {band}: {fore_percent_diff}")

    # Add statistics to the plot
    # axs[0,2].text(70,300, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
    # axs[0,2].text(70, 300, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
    fig.text(
        0.7,
        0.65,
        rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{DIB}}_i |$",
        fontsize=14,
        color="black",
    )

    fig.text(
        0.7,
        0.6,
        rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
        fontsize=14,
        color="black",
    )
    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(
        f"output/MS3_verification_tests/T_14/{imgname}"  # T_14_difference1.png"
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


# def scatter_compare(ids_data, dib_data, band, imgname):
#     x_7 = dib_data["bt_h"].flatten()
#     y_7 = ids_data["bt_h"].flatten()
#     x_h_7, y_h_7, m_h_7, b_h_7, y_fit_h_7, r_squared_7 = scatter_stats(x_7, y_7)

#     fig, axs = plt.subplots(1, 3, figsize=(20, 12))
#     # plt.suptitle("L1R AMSR2 Remap to the footprints of Band 6")
#     plt.suptitle("L1C AMSR2 Remap")
#     axs[0].scatter(x_h_7, y_h_7)
#     axs[0].plot(x_h_7, y_fit_h_7, color="red")
#     axs[0].legend(title=f"$R^2 = {r_squared_7:.3f}$")
#     axs[0].set_title(f"bt_h ({band})")
#     axs[0].set_xlabel("DIB BT [K]")
#     axs[0].set_ylabel("IDS BT [K]")

#     repo_root = grasp_io.find_repo_root()
#     img_path = repo_root.joinpath(
#         f"output/MS3_verification_tests/T_14/{imgname}"  # T_14_difference1.png"
#     )  # ""
#     plt.savefig(img_path, dpi=300)
#     # plt.show()


def plot_all_results(results, imgname="T_14_scatter_combined.png"):
    fig, axs = plt.subplots(1, 3, figsize=(20, 12))
    plt.suptitle("L1C AMSR2 Remap")

    bands = ["7_BAND", "18_BAND", "89b_BAND"]
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
    img_path = repo_root.joinpath(f"output/MS3_verification_tests/T_14/{imgname}")
    plt.savefig(img_path, dpi=300)


def test_plot_results():
    """
    Dummy test to plot results into one plot
    """
    global test_results
    plot_all_results(test_results)
