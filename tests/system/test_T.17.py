# Testing script for T.17
# Remapping of L1b CIMR C-band data with LW (Landweber) algorithm on a Mercator global grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)
import sys
import subprocess as sbps

from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import pytest
import matplotlib.pyplot as plt

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io


@pytest.mark.parametrize("setup_paths", ["T_17"], indirect=True)
def test_T17_execution(setup_paths, run_subprocess):
    """
    Test the execution of the subprocess for the T_17 scenario.

    Parameters
    ----------
    setup_paths : dict
        Dictionary containing paths for the T_17 scenario, retrieved using the `setup_paths` fixture.
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
@pytest.mark.parametrize("setup_paths", ["T_17"], indirect=True)
@pytest.mark.parametrize(
    "TEST_NAME, DATA_OUTPUT, PROJECTION, GRID, INPUT_BAND, OUTPUT_BAND",
    [
        (
            "CIMR: LW_RGB vs IDS_RGB",
            "L1C",
            "G",
            "EASE2_G9km",
            "C_BAND",
            "C_BAND",
        ),
    ],
)
def test_T17_comparison(
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

    map_compare(lw_data=data1, ids_data=data2)
    scatter_compare(lw_data=data1, ids_data=data2)

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


# TODO: T17 is effectively broken now, need to return to it asap.
def map_compare(lw_data, ids_data):
    cmap = "viridis"
    # bt_h plt
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    im00 = axs[0].imshow(lw_data["bt_h"], cmap=cmap)
    axs[0].set_title("LW Remap (bt_h)")
    im01 = axs[1].imshow(ids_data["bt_h"], cmap=cmap)
    axs[1].set_title("IDS Remap (bt_h)")
    bt_h_fore_diff = abs(lw_data["bt_h"] - ids_data["bt_h"])
    im02 = axs[2].imshow(bt_h_fore_diff, cmap=cmap)
    axs[2].set_title("Difference (bt_h)")
    # aft

    fig.colorbar(im02, ax=axs[2])

    # Add Statistics
    # Calculate the average relative difference
    fore_mean_diff = nanmean(bt_h_fore_diff)

    print(f"Average relative difference for bt_h: {fore_mean_diff}")

    # Calculate percentage Differences
    fore_percent_diff = (fore_mean_diff / nanmean(ids_data["bt_h"])) * 100

    print(f"Average percentage difference for bt_h: {fore_percent_diff}")

    # Add statistics to the plot
    # axs[0,2].text(100,300, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
    # axs[0,2].text(100, 300, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
    axs[2].text(
        100,
        420,
        rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{LW}}_i |$",
        fontsize=14,
        color="black",
    )

    axs[2].text(
        100,
        470,
        rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
        fontsize=14,
        color="black",
    )

    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(
        "output/MS3_verification_tests/T_17/T_17_difference1.png"
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


def scatter_compare(lw_data, ids_data):
    x = lw_data["bt_h"].flatten()
    y = ids_data["bt_h"].flatten()
    x_h_fore, y_h_fore, m_h_fore, b_h_fore, y_fit_h_fore, r_squared = scatter_stats(
        x, y
    )

    # x = lw_data['bt_h_aft'].flatten()
    # y = ids_data['bt_h_aft'].flatten()
    # x_h_aft, y_h_aft, m_h_aft, b_h_aft, y_fit_h_aft, r_squared = self.scatter_stats(x, y)

    fig, axs = plt.subplots()
    axs.scatter(x_h_fore, y_h_fore)
    axs.plot(x_h_fore, y_fit_h_fore, color="red")
    axs.legend(title=f"$R^2 = {r_squared:.3f}$")
    axs.set_title("bt_h_fore")
    axs.set_xlabel("LW BT [K]")
    axs.set_ylabel("IDS BT [K]")

    # axs[1].scatter(x_h_aft, y_h_aft)
    # axs[1].plot(x_h_aft, y_fit_h_aft, color='red')
    # axs[1].legend(title=f"$R^2 = {r_squared:.3f}$")
    # axs[1].set_title('bt_h_aft')
    # axs[1].set_xlabel('LW BT [K]')
    # axs[1].set_ylabel('IDS BT [K]')

    repo_root = grasp_io.find_repo_root()
    img_path = repo_root.joinpath(
        "output/MS3_verification_tests/T_17/T_17_scatter.png"
    )  # ""
    plt.savefig(img_path, dpi=300)

    # plt.show()
