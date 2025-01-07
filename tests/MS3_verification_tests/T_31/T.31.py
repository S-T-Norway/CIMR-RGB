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

# Add RGB remapped IDS netCDF here
# ids_data_path = "/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/MS3_verification_tests/T_31/AMSR2_L1R_IDS_2024-12-08_18-05-15.nc"
# ids_data_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/MS3_verification_tests/CIMR_L1R_IDS_36km_2024-12-04_13-06-26_SR30.nc'
# ids_data_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/MS3_verification_tests/CIMR_L1R_IDS_36km_2024-12-04_13-09-31_SR100.nc'
# ids_data_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/MS3_verification_tests/CIMR_L1R_IDS_36km_2024-12-04_13-10-09_SR200.nc'
# Add the RGB remapped DIB netCDF here


# dib_data_path = "/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/MS3_verification_tests/T_31/AMSR2_L1R_DIB_2024-12-08_18-06-18.nc"

repo_root = grasp_io.find_repo_root()
# Add RGB remapped IDS netCDF here
ids_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_31/AMSR2_L1R_IDS_test.nc"
)  # ""
# Add the RGB remapped BG netCDF here
dib_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_31/AMSR2_L1R_DIB_test.nc"
)


BANDS = "10_BAND", "23_BAND", "89a_BAND"
PROJECTION = "6_BAND_TARGET"


class AMSR2_comparison:
    def __init__(self, ids_data_path, dib_data_path):
        self.ids_data = self.get_netcdf_data(ids_data_path)
        self.dib_data = self.get_netcdf_data(dib_data_path)

    @staticmethod
    def get_netcdf_data(path):
        import netCDF4 as nc

        gridded_vars = {}
        with nc.Dataset(path, "r") as f:
            data = f[f"{PROJECTION}"]
            for band in BANDS:
                measurement = data["Measurement"]
                channel = measurement[band]
                gridded_vars[f"bt_h_{band}"] = channel["bt_h"][:]
        return gridded_vars

    def map_compare(self):
        bt_h_ids_10 = self.ids_data["bt_h_10_BAND"]
        bt_h_ids_23 = self.ids_data["bt_h_23_BAND"]
        bt_h_ids_89a = self.ids_data["bt_h_89a_BAND"]
        bt_h_dib_10 = self.dib_data["bt_h_10_BAND"]
        bt_h_dib_23 = self.dib_data["bt_h_23_BAND"]
        bt_h_dib_89a = self.dib_data["bt_h_89a_BAND"]

        cmap = "viridis"
        # bt_h plt
        # ---------------------- 10_BAND ----------------------
        fig, axs = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)
        fig.suptitle("10 -- > 6")
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
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_31/T_31_difference1.png"
        )  # ""
        plt.savefig(img_path, dpi=300)
        plt.show()

        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_h_diff)

        print(f"Average relative difference for bt_h for band 10: {fore_mean_diff}")

        # Calculate percentage Differences
        fore_percent_diff = (
            fore_mean_diff / nanmean(self.ids_data["bt_h_10_BAND"])
        ) * 100

        print(
            f"Average percentage difference for bt_h  for band 10: {fore_percent_diff}"
        )

        # Add statistics to the plot
        # axs[0,2].text(100,300, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
        # axs[0,2].text(100, 300, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
        fig.text(
            0.7,
            0.5,
            rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{DIB}}_i |$",
            fontsize=14,
            color="black",
        )

        fig.text(
            0.7,
            0.4,
            rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
            fontsize=14,
            color="black",
        )

        # # ---------------------- 23_BAND ----------------------
        fig, axs = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)
        fig.suptitle("23 -- > 6")
        im10 = axs[0].imshow(bt_h_dib_23[:, :, 0], cmap=cmap)
        axs[0].set_title("DIB Remap (bt_h)")
        fig.colorbar(im10, ax=axs[0])

        im11 = axs[1].imshow(bt_h_ids_23[:, :, 0], cmap=cmap)
        axs[1].set_title("IDS Remap (bt_h)")
        fig.colorbar(im11, ax=axs[1])

        bt_h_diff = abs(bt_h_ids_23 - bt_h_dib_23)
        im12 = axs[2].imshow(bt_h_diff[:, :, 0], cmap=cmap)
        axs[2].set_title("Difference (bt_h)")
        fig.colorbar(im12, ax=axs[2])

        fig.text(
            0.7,
            0.5,
            rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{DIB}}_i |$",
            fontsize=14,
            color="black",
        )

        fig.text(
            0.7,
            0.4,
            rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
            fontsize=14,
            color="black",
        )
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_31/T_31_difference2.png"
        )  # ""
        plt.savefig(img_path, dpi=300)

        plt.show()
        #
        # # ---------------------- 89a_BAND ----------------------
        fig, axs = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)
        fig.suptitle("89a -- > 6")
        im20 = axs[0].imshow(bt_h_dib_89a[:, :, 0], cmap=cmap)
        axs[0].set_title("DIB Remap (bt_h)")
        fig.colorbar(im20, ax=axs[0])

        im21 = axs[1].imshow(bt_h_ids_89a[:, :, 0], cmap=cmap)
        axs[1].set_title("IDS Remap (bt_h)")
        fig.colorbar(im21, ax=axs[1])

        bt_h_diff = abs(bt_h_ids_89a - bt_h_dib_89a)
        im22 = axs[2].imshow(bt_h_diff[:, :, 0], cmap=cmap)
        axs[2].set_title("Difference (bt_h)")
        fig.colorbar(im22, ax=axs[2])

        fig.text(
            0.7,
            0.5,
            rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{DIB}}_i |$",
            fontsize=14,
            color="black",
        )

        fig.text(
            0.7,
            0.4,
            rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
            fontsize=14,
            color="black",
        )
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_31/T_31_difference3.png"
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
        x_10 = self.dib_data["bt_h_10_BAND"].flatten()
        y_10 = self.ids_data["bt_h_10_BAND"].flatten()
        x_h_10, y_h_10, m_h_10, b_h_10, y_fit_h_10, r_squared_10 = self.scatter_stats(
            x_10, y_10
        )

        x_23 = self.dib_data["bt_h_23_BAND"].flatten()
        y_23 = self.ids_data["bt_h_23_BAND"].flatten()
        x_h_23, y_h_23, m_h_23, b_h_23, y_fit_h_23, r_squared_23 = self.scatter_stats(
            x_23, y_23
        )

        x_89a = self.dib_data["bt_h_89a_BAND"].flatten()
        y_89a = self.ids_data["bt_h_89a_BAND"].flatten()
        x_h_89a, y_h_89a, m_h_89a, b_h_89a, y_fit_h_89a, r_squared_89a = (
            self.scatter_stats(x_89a, y_89a)
        )

        fig, axs = plt.subplots(1, 3, figsize=(20, 12))
        plt.suptitle("L1R AMSR2 Remap to the footprints of Band 6")
        axs[0].scatter(x_h_10, y_h_10)
        axs[0].plot(x_h_10, y_fit_h_10, color="red")
        axs[0].legend(title=f"$R^2 = {r_squared_10:.3f}$")
        axs[0].set_title("bt_h (Band 10)")
        axs[0].set_xlabel("DIB BT [K]")
        axs[0].set_ylabel("IDS BT [K]")

        axs[1].scatter(x_h_23, y_h_23)
        axs[1].plot(x_h_23, y_fit_h_23, color="red")
        axs[1].legend(title=f"$R^2 = {r_squared_23:.3f}$")
        axs[1].set_title("bt_h (Band 23)")
        axs[1].set_xlabel("DIB BT [K]")
        axs[1].set_ylabel("IDS BT [K]")

        axs[2].scatter(x_h_89a, y_h_89a)
        axs[2].plot(x_h_89a, y_fit_h_89a, color="red")
        axs[2].legend(title=f"$R^2 = {r_squared_89a:.3f}$")
        axs[2].set_title("bt_h (Band 89a)")
        axs[2].set_xlabel("DIB BT [K]")
        axs[2].set_ylabel("IDS BT [K]")
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_31/T_31_scatter.png"
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
        "tests/MS3_verification_tests/T_31/T_31_IDS.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    # DIB
    config_path = grasp_io.find_repo_root().joinpath(
        "tests/MS3_verification_tests/T_31/T_31_DIB.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    # AMSR2_comparison(ids_data_path, dib_data_path)
    AMSR2_comparison(ids_data_path, dib_data_path).map_compare()
    AMSR2_comparison(ids_data_path, dib_data_path).scatter_compare()
