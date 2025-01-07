# Testing Script for T.14
# Remapping of AMSR2 data with DIB (drop-in-the-bucket) algorithm on an EASE2 South polar grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)


import sys
import os
import pathlib as pb
import subprocess as sbps

import matplotlib

tkagg = matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.ion()

# sys.path.append("/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb")
from numpy import array, full, nan, nanmean, polyfit, isnan, isinf

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io

repo_root = grasp_io.find_repo_root()
# Add RGB remapped IDS netCDF here
ids_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_14/AMSR2_L1C_IDS_9km_test.nc"
)  # ""
# Add the RGB remapped DIB netCDF here
dib_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_14/AMSR2_L1C_DIB_9km_test.nc"
)

BANDS = "7_BAND", "18_BAND", "89b_BAND"
PROJECTION = "S"
GRID = "EASE2_S9km"


class AMSR2_comparison:
    def __init__(self, ids_data_path, dib_data_path):
        self.ids_data = self.get_netcdf_data(ids_data_path)
        self.dib_data = self.get_netcdf_data(dib_data_path)

    @staticmethod
    def get_netcdf_data(path):
        import netCDF4 as nc

        data_dict = {}
        with nc.Dataset(path, "r") as f:
            data = f[f"{PROJECTION}"]
            measurement = data["Measurement"]
            for band in BANDS:
                gridded_vars = {}
                band_data = measurement[band]
                cell_row = band_data["cell_row"][:]
                cell_col = band_data["cell_col"][:]
                for bt in ["bt_h"]:
                    var = array(band_data[bt][:])
                    grid = full((GRIDS[GRID]["n_rows"], GRIDS[GRID]["n_cols"]), nan)

                    for count, sample in enumerate(var):
                        if sample == 9.969209968386869e36:
                            continue
                        grid[cell_row[count], cell_col[count]] = sample
                    gridded_vars[bt] = grid
                data_dict[band] = gridded_vars
            return data_dict

    def map_compare(self):
        bt_h_ids_7 = self.ids_data["7_BAND"]["bt_h"]
        bt_h_ids_18 = self.ids_data["18_BAND"]["bt_h"]
        bt_h_ids_89b = self.ids_data["89b_BAND"]["bt_h"]
        bt_h_dib_7 = self.dib_data["7_BAND"]["bt_h"]
        bt_h_dib_18 = self.dib_data["18_BAND"]["bt_h"]
        bt_h_dib_89b = self.dib_data["89b_BAND"]["bt_h"]

        cmap = "viridis"
        # bt_h plt
        # ---------------------- 7_BAND ----------------------
        fig, axs = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)
        fig.suptitle("7 -- > 6")
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
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_14/T_14_difference1.png"
        )  # ""
        plt.savefig(img_path, dpi=300)
        plt.show()

        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_h_diff)

        print(f"Average relative difference for bt_h for band 7: {fore_mean_diff}")

        # Calculate percentage Differences
        fore_percent_diff = (
            fore_mean_diff / nanmean(self.ids_data["7_BAND"]["bt_h"])
        ) * 70

        print(
            f"Average percentage difference for bt_h  for band 7: {fore_percent_diff}"
        )

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

        # # ---------------------- 18_BAND ----------------------
        fig, axs = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)
        fig.suptitle("18 -- > 6")
        im7 = axs[0].imshow(bt_h_dib_18, cmap=cmap)
        axs[0].set_title("DIB Remap (bt_h)")
        fig.colorbar(im7, ax=axs[0])

        im11 = axs[1].imshow(bt_h_ids_18, cmap=cmap)
        axs[1].set_title("IDS Remap (bt_h)")
        fig.colorbar(im11, ax=axs[1])

        bt_h_diff = abs(bt_h_ids_18 - bt_h_dib_18)
        im12 = axs[2].imshow(bt_h_diff, cmap=cmap)
        axs[2].set_title("Difference (bt_h)")
        fig.colorbar(im12, ax=axs[2])

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
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_14/T_14_difference2.png"
        )  # ""
        plt.savefig(img_path, dpi=300)

        plt.show()
        #
        # # ---------------------- 89b_BAND ----------------------
        fig, axs = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)
        fig.suptitle("89b -- > 6")
        im20 = axs[0].imshow(bt_h_dib_89b, cmap=cmap)
        axs[0].set_title("DIB Remap (bt_h)")
        fig.colorbar(im20, ax=axs[0])

        im21 = axs[1].imshow(bt_h_ids_89b, cmap=cmap)
        axs[1].set_title("IDS Remap (bt_h)")
        fig.colorbar(im21, ax=axs[1])

        bt_h_diff = abs(bt_h_ids_89b - bt_h_dib_89b)
        im22 = axs[2].imshow(bt_h_diff, cmap=cmap)
        axs[2].set_title("Difference (bt_h)")
        fig.colorbar(im22, ax=axs[2])

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
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_14/T_14_difference3.png"
        )  # ""
        plt.savefig(img_path, dpi=300)

        plt.show()

    @staticmethod
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

    def scatter_compare(self):
        x_7 = self.dib_data["7_BAND"]["bt_h"].flatten()
        y_7 = self.ids_data["7_BAND"]["bt_h"].flatten()
        x_h_7, y_h_7, m_h_7, b_h_7, y_fit_h_7, r_squared_7 = self.scatter_stats(
            x_7, y_7
        )

        x_18 = self.dib_data["18_BAND"]["bt_h"].flatten()
        y_18 = self.ids_data["18_BAND"]["bt_h"].flatten()
        x_h_18, y_h_18, m_h_18, b_h_18, y_fit_h_18, r_squared_18 = self.scatter_stats(
            x_18, y_18
        )

        x_89b = self.dib_data["89b_BAND"]["bt_h"].flatten()
        y_89b = self.ids_data["89b_BAND"]["bt_h"].flatten()
        x_h_89b, y_h_89b, m_h_89b, b_h_89b, y_fit_h_89b, r_squared_89b = (
            self.scatter_stats(x_89b, y_89b)
        )

        fig, axs = plt.subplots(1, 3, figsize=(20, 12))
        plt.suptitle("L1R AMSR2 Remap to the footprints of Band 6")
        axs[0].scatter(x_h_7, y_h_7)
        axs[0].plot(x_h_7, y_fit_h_7, color="red")
        axs[0].legend(title=f"$R^2 = {r_squared_7:.3f}$")
        axs[0].set_title("bt_h (Band 7)")
        axs[0].set_xlabel("DIB BT [K]")
        axs[0].set_ylabel("IDS BT [K]")

        axs[1].scatter(x_h_18, y_h_18)
        axs[1].plot(x_h_18, y_fit_h_18, color="red")
        axs[1].legend(title=f"$R^2 = {r_squared_18:.3f}$")
        axs[1].set_title("bt_h (Band 18)")
        axs[1].set_xlabel("DIB BT [K]")
        axs[1].set_ylabel("IDS BT [K]")

        axs[2].scatter(x_h_89b, y_h_89b)
        axs[2].plot(x_h_89b, y_fit_h_89b, color="red")
        axs[2].legend(title=f"$R^2 = {r_squared_89b:.3f}$")
        axs[2].set_title("bt_h (Band 89b)")
        axs[2].set_xlabel("DIB BT [K]")
        axs[2].set_ylabel("IDS BT [K]")
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_14/T_14_scatter.png"
        )  # ""
        plt.savefig(img_path, dpi=300)
        plt.show()


def run_python_subprocess(config_path):
    """
    Runs a Python script as a subprocess, displaying output in real-time.

    Parameters:
    ----------
    script_path : str
        Path to the Python script to be executed.
    *args : str
        Additional arguments to pass to the script.

    Returns:
    ----------
    int
        Exit code of the subprocess.
    """
    try:
        # Construct the command
        # command = [sys.executable, script_path] + list(args)
        command = [
            "python",
            "-m",
            "cimr_rgb",
            str(config_path),
        ]  # Adjust command if needed

        # Run the subprocess and allow real-time output
        result = sbps.run(
            command,
            stdout=sys.stdout,  # Redirect stdout to the parent process's stdout
            stderr=sys.stderr,  # Redirect stderr to the parent process's stderr
        )

        # Return the exit code
        return result.returncode
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        return -1


if __name__ == "__main__":
    # IDS
    config_path = grasp_io.find_repo_root().joinpath(
        "tests/MS3_verification_tests/T_14/T_14_IDS.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    # DIB
    config_path = grasp_io.find_repo_root().joinpath(
        "tests/MS3_verification_tests/T_14/T_14_DIB.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    # AMSR2_comparison(ids_data_path, dib_data_path)
    AMSR2_comparison(ids_data_path, dib_data_path).map_compare()
    AMSR2_comparison(ids_data_path, dib_data_path).scatter_compare()
