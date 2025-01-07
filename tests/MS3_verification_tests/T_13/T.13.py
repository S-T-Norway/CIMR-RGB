# Testing script for T.13
# Remapping of L1b CIMR L-band data with NN (nearest neighbor) algorithm on an EASE2 North grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)

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

repo_root = grasp_io.find_repo_root()
# Add RGB remapped IDS netCDF here
ids_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_13/CIMR_L1C_IDS_9km_test.nc"
)  # ""
# Add the RGB remapped NN netCDF here
nn_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_13/CIMR_L1C_NN_9km_test.nc"
)

GRID = "EASE2_N9km"
PROJECTION = "N"
BAND = "L_BAND"


class SMAP_comparison:
    def __init__(self, ids_data_path, nn_data_path):
        self.ids_data = self.get_netcdf_data(ids_data_path)
        self.nn_data = self.get_netcdf_data(nn_data_path)

    @staticmethod
    def get_netcdf_data(path):
        import netCDF4 as nc

        gridded_vars = {}
        with nc.Dataset(path, "r") as f:
            data = f[f"{PROJECTION}"]
            measurement = data["Measurement"]
            band = measurement[BAND]

            for bt in ["bt_h_fore", "bt_h_aft"]:
                if "fore" in bt:
                    row = array(band["cell_row_fore"][:])
                    col = array(band["cell_col_fore"][:])
                elif "aft" in bt:
                    row = array(band["cell_row_aft"][:])
                    col = array(band["cell_col_aft"][:])
                else:
                    row = ""
                    col = ""
                var = array(band[bt][:])
                grid = full((GRIDS[GRID]["n_rows"], GRIDS[GRID]["n_cols"]), nan)

                for count, sample in enumerate(var):
                    if sample == 9.969209968386869e36:
                        continue
                    if sample == 0.0:
                        continue
                    grid[row[count], col[count]] = sample
                gridded_vars[bt] = grid
            return gridded_vars

    def map_compare(self):
        cmap = "viridis"
        # bt_h plt
        fig, axs = plt.subplots(2, 3, figsize=(20, 20), constrained_layout=True)
        im00 = axs[0, 0].imshow(self.nn_data["bt_h_fore"][750:1250, 0:400], cmap=cmap)
        axs[0, 0].set_title("NN Remap (bt_h_fore)")
        im01 = axs[0, 1].imshow(self.ids_data["bt_h_fore"][750:1250, 0:400], cmap=cmap)
        axs[0, 1].set_title("IDS Remap (bt_h_fore)")
        bt_h_fore_diff = abs(self.nn_data["bt_h_fore"] - self.ids_data["bt_h_fore"])
        im02 = axs[0, 2].imshow(bt_h_fore_diff[750:1250, 0:400], cmap=cmap)
        axs[0, 2].set_title("Difference (bt_h_fore)")
        # aft
        im10 = axs[1, 0].imshow(self.nn_data["bt_h_aft"][750:1250, 0:400], cmap=cmap)
        axs[1, 0].set_title("NN Remap (bt_h_aft)")
        im11 = axs[1, 1].imshow(self.ids_data["bt_h_aft"][750:1250, 0:400], cmap=cmap)
        axs[1, 1].set_title("IDS Remap (bt_h_aft)")
        bt_h_aft_diff = abs(self.nn_data["bt_h_aft"] - self.ids_data["bt_h_aft"])
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
        fore_percent_diff = (fore_mean_diff / nanmean(self.ids_data["bt_h_fore"])) * 100
        aft_percent_diff = (aft_mean_diff / nanmean(self.ids_data["bt_h_aft"])) * 100
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
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_13/T_13_difference1.png"
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
        x = self.nn_data["bt_h_fore"].flatten()
        y = self.ids_data["bt_h_fore"].flatten()
        x_h_fore, y_h_fore, m_h_fore, b_h_fore, y_fit_h_fore, r_squared = (
            self.scatter_stats(x, y)
        )

        x = self.nn_data["bt_h_aft"].flatten()
        y = self.ids_data["bt_h_aft"].flatten()
        x_h_aft, y_h_aft, m_h_aft, b_h_aft, y_fit_h_aft, r_squared = self.scatter_stats(
            x, y
        )

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
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_13/T_13_scatter.png"
        )  # ""
        plt.savefig(img_path, dpi=300)
        plt.show()

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
    # NN
    config_path = grasp_io.find_repo_root().joinpath(
        "tests/MS3_verification_tests/T_13/T_13_NN.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    # IDS
    config_path = grasp_io.find_repo_root().joinpath(
        "tests/MS3_verification_tests/T_13/T_13_IDS.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    SMAP_comparison(ids_data_path, nn_data_path).map_compare()
    SMAP_comparison(ids_data_path, nn_data_path).scatter_compare()
