# Testing Script for T.19
# Remapping of L1b CIMR L-band simulated data to L1r C-band data using BG algorithm (resolution upgrading)
# The remapped data are compatible with L1r data obtained by regridding with IDS algorithm (L1c -> L1r) on the same output grid (average relative difference of the brightness temperature < 1%)


import sys
import os
import pathlib as pb
import subprocess as sbps

import matplotlib

tkagg = matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.ion()

# sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb')
from numpy import array, full, nan, nanmean, polyfit, isnan, isinf

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io

# Add RGB remapped IDS netCDF here
# ids_data_path = ""
# Add the RGB remapped BG netCDF here
# bg_data_path = ""

repo_root = grasp_io.find_repo_root()
# Add RGB remapped IDS netCDF here
ids_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_19/CIMR_L1R_IDS_test.nc"
)  # ""
# Add the RGB remapped BG netCDF here
bg_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_19/CIMR_L1R_BG_test.nc"
)


BAND = "L_BAND"
PROJECTION = "C_BAND_TARGET"


class CIMR_comparison:
    def __init__(self, ids_data_path, bg_data_path):
        self.ids_data = self.get_netcdf_data(ids_data_path)
        self.bg_data = self.get_netcdf_data(bg_data_path)

    @staticmethod
    def get_netcdf_data(path):
        import netCDF4 as nc

        gridded_vars = {}
        with nc.Dataset(path, "r") as f:
            data = f[f"{PROJECTION}"]
            measurement = data["Measurement"]
            band = measurement[BAND]

            bt_h = band["bt_h"][:]
            gridded_vars["bt_h"] = bt_h
            return gridded_vars

    def map_compare(self):
        cmap = "viridis"
        # bt_h plt
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        im00 = axs[0].imshow(self.bg_data["bt_h"][:, :, 0], cmap=cmap)
        axs[0].set_title("BG Remap (bt_h)")
        im01 = axs[1].imshow(self.ids_data["bt_h"][:, :, 0], cmap=cmap)
        axs[1].set_title("IDS Remap (bt_h)")
        bt_h_diff = abs(self.bg_data["bt_h"] - self.ids_data["bt_h"])
        im02 = axs[2].imshow(bt_h_diff[:, :, 0], cmap=cmap)
        axs[2].set_title("Difference (bt_h)")

        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_h_diff)

        print(f"Average relative difference for bt_h: {fore_mean_diff}")

        # Calculate percentage Differences
        fore_percent_diff = (fore_mean_diff / nanmean(self.ids_data["bt_h"])) * 100

        print(f"Average percentage difference for bt_h: {fore_percent_diff}")

        # Add statistics to the plot
        # axs[0,2].text(100,300, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
        # axs[0,2].text(100, 300, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
        axs[2].text(
            50,
            200,
            rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{BG}}_i |$",
            fontsize=14,
            color="black",
        )

        axs[2].text(
            50,
            250,
            rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
            fontsize=14,
            color="black",
        )
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
        x = self.bg_data["bt_h"].flatten()
        y = self.ids_data["bt_h"].flatten()
        x_h_fore, y_h_fore, m_h_fore, b_h_fore, y_fit_h_fore, r_squared = (
            self.scatter_stats(x, y)
        )

        fig, axs = plt.subplots()
        axs.scatter(x_h_fore, y_h_fore)
        axs.plot(x_h_fore, y_fit_h_fore, color="red")
        axs.legend(title=f"$R^2 = {r_squared:.3f}$")
        axs.set_title("bt_h")
        axs.set_xlabel("BG BT [K]")
        axs.set_ylabel("IDS BT [K]")
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
        "tests/MS3_verification_tests/T_19/T_19_IDS.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    # DIB
    config_path = grasp_io.find_repo_root().joinpath(
        "tests/MS3_verification_tests/T_19/T_19_BG.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    # CIMR_comparison(ids_data_path, bg_data_path)
    CIMR_comparison(ids_data_path, bg_data_path).map_compare()
    CIMR_comparison(ids_data_path, bg_data_path).scatter_compare()
