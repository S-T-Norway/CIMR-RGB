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


from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import matplotlib

tkagg = matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.ion()

import pytest

# sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb')
from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io

# from tests.system.shared_confs import shared_confs
# from conftest import run_subprocess
# import conftest


# Add the project root directory to sys.path
# sys.path.insert(0, str(pb.Path(__file__).resolve().parent.parent.parent))
# from tests.system.shared_confs import run_subprocess


# repo_root = grasp_io.find_repo_root()
# # Add RGB remapped netCDF here
# rgb_data = repo_root.joinpath(
#     "output/MS3_verification_tests/T_12/SMAP_L1C_IDS_36km_test.nc"
# )  # ""
# print(rgb_data)
# # Add the NASA version you are comparing to here
# nasa_data = repo_root.joinpath(
#     "dpr/L1C/SMAP/NASA/SMAP_L1C_TB_47185_D_20231201T212059_R19240_002.h5"
# )  # ""

GRID = "EASE2_G36km"
PROJECTION = "G"


class SMAP_comparison:
    """
    Class for comparing remapped SMAP data from RGB and NASA sources.

    Parameters
    ----------
    rgb_data_path : str
        Path to the RGB remapped netCDF file.
    nasa_data_path : str
        Path to the NASA remapped HDF5 file.

    Attributes
    ----------
    rgb_data : dict
        Dictionary containing gridded variables from the RGB data.
    nasa_data : dict
        Dictionary containing gridded variables from the NASA data.
    """

    def __init__(self, rgb_data_path, nasa_data_path):
        self.rgb_data = self.get_netcdf_data(rgb_data_path)
        self.nasa_data = self.get_hdf5_data(nasa_data_path)

    @staticmethod
    def get_netcdf_data(path):
        """
        Extract and grid data from a netCDF file.

        Parameters
        ----------
        path : str
            Path to the netCDF file.

        Returns
        -------
        dict
            A dictionary containing gridded brightness temperature variables.
        """

        import netCDF4 as nc

        gridded_vars = {}
        with nc.Dataset(path, "r") as f:
            data = f[f"{PROJECTION}"]
            measurement = data["Measurement"]
            l_band = measurement["L_BAND"]

            for bt in ["bt_h_fore", "bt_h_aft", "bt_v_fore", "bt_v_aft"]:
                if "fore" in bt:
                    row = array(l_band["cell_row_fore"][:])
                    col = array(l_band["cell_col_fore"][:])
                elif "aft" in bt:
                    row = array(l_band["cell_row_aft"][:])
                    col = array(l_band["cell_col_aft"][:])
                else:
                    row = ""
                    col = ""
                var = array(l_band[bt][:])
                grid = full((GRIDS[GRID]["n_rows"], GRIDS[GRID]["n_cols"]), nan)

                for count, sample in enumerate(var):
                    if sample == 9.969209968386869e36:
                        continue
                    grid[row[count], col[count]] = sample
                gridded_vars[bt] = grid

            return gridded_vars

    @staticmethod
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

    def map_compare(self):
        cmap = "viridis"
        # bt_h plt
        fig, axs = plt.subplots(2, 3, figsize=(20, 20), constrained_layout=True)
        im00 = axs[0, 0].imshow(self.rgb_data["bt_h_fore"][:, 550:], cmap=cmap)
        axs[0, 0].set_title("RGB Remap (bt_h_fore)")
        om01 = axs[0, 1].imshow(self.nasa_data["bt_h_fore"][:, 550:], cmap=cmap)
        axs[0, 1].set_title("NASA Remap (bt_h_fore)")

        bt_h_fore_diff = abs(self.rgb_data["bt_h_fore"] - self.nasa_data["bt_h_fore"])

        im02 = axs[0, 2].imshow(bt_h_fore_diff[:, 550:], cmap=cmap)
        axs[0, 2].set_title("Difference (bt_h_fore)")
        # aft
        im10 = axs[1, 0].imshow(self.rgb_data["bt_h_aft"][:, 550:], cmap=cmap)
        axs[1, 0].set_title("RGB Remap (bt_h_aft)")
        im11 = axs[1, 1].imshow(self.nasa_data["bt_h_aft"][:, 550:], cmap=cmap)
        axs[1, 1].set_title("NASA Remap (bt_h_aft)")

        bt_h_aft_diff = abs(self.rgb_data["bt_h_aft"] - self.nasa_data["bt_h_aft"])

        im12 = axs[1, 2].imshow(bt_h_aft_diff[:, 550:], cmap=cmap)
        axs[1, 2].set_title("Difference (bt_h_aft)")
        fig.colorbar(im02, ax=axs[0, 2])
        fig.colorbar(im12, ax=axs[1, 2])

        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_h_fore_diff)
        aft_mean_diff = nanmean(bt_h_aft_diff)
        print(f"Average relative difference for bt_h_fore: {fore_mean_diff}")
        print(f"Average relative difference for bt_h_aft: {aft_mean_diff}")

        # Calculate percentage Differences
        fore_percent_diff = (
            fore_mean_diff / nanmean(self.nasa_data["bt_h_fore"])
        ) * 100
        aft_percent_diff = (aft_mean_diff / nanmean(self.nasa_data["bt_h_aft"])) * 100
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
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_12/T_12_difference1.png"
        )  # ""
        plt.savefig(img_path, dpi=300)
        plt.show()

        # bt_v plt
        fig, axs = plt.subplots(2, 3, figsize=(20, 20), constrained_layout=True)
        im00 = axs[0, 0].imshow(self.rgb_data["bt_v_fore"][:, 550:], cmap=cmap)
        axs[0, 0].set_title("RGB Remap (bt_v_fore)")
        om01 = axs[0, 1].imshow(self.nasa_data["bt_v_fore"][:, 550:], cmap=cmap)
        axs[0, 1].set_title("NASA Remap (bt_v_fore)")
        bt_v_fore_diff = abs(self.rgb_data["bt_v_fore"] - self.nasa_data["bt_v_fore"])
        im02 = axs[0, 2].imshow(bt_v_fore_diff[:, 550:], cmap=cmap)
        axs[0, 2].set_title("Difference (bt_v_fore)")
        # aft
        im10 = axs[1, 0].imshow(self.rgb_data["bt_v_aft"][:, 550:], cmap=cmap)
        axs[1, 0].set_title("RGB Remap (bt_v_aft)")
        im11 = axs[1, 1].imshow(self.nasa_data["bt_v_aft"][:, 550:], cmap=cmap)
        axs[1, 1].set_title("NASA Remap (bt_v_aft)")
        bt_v_aft_diff = abs(self.rgb_data["bt_v_aft"] - self.nasa_data["bt_v_aft"])
        im12 = axs[1, 2].imshow(bt_v_aft_diff[:, 550:], cmap=cmap)
        axs[1, 2].set_title("Difference (bt_v_aft)")
        fig.colorbar(im02, ax=axs[0, 2])
        fig.colorbar(im12, ax=axs[1, 2])

        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_v_fore_diff)
        aft_mean_diff = nanmean(bt_h_aft_diff)
        print(f"Average relative difference for bt_v_fore: {fore_mean_diff}")
        print(f"Average relative difference for bt_v_aft: {aft_mean_diff}")

        # Calculate percentage Differences
        fore_percent_diff = (
            fore_mean_diff / nanmean(self.nasa_data["bt_v_fore"])
        ) * 100
        aft_percent_diff = (aft_mean_diff / nanmean(self.nasa_data["bt_v_aft"])) * 100
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
        img_path = repo_root.joinpath(
            "output/MS3_verification_tests/T_12/T_12_difference2.png"
        )  # ""
        plt.savefig(img_path, dpi=300)
        plt.show()

    # @staticmethod
    # def scatter_stats(x, y):
    #     mask = ~isnan(x) & ~isnan(y) & ~isinf(x) & ~isinf(y)
    #     x = x[mask]
    #     y = y[mask]
    #     m, b = polyfit(x, y, 1)
    #     y_fit = m * x + b

    #     # Calculate R^2
    #     ss_res = sum((y - y_fit) ** 2)
    #     ss_tot = sum((y - y.mean()) ** 2)
    #     r_squared = 1 - (ss_res / ss_tot)

    #     return x, y, m, b, y_fit, r_squared

    # def scatter_compare(self):
    #     x = self.rgb_data["bt_h_fore"].flatten()
    #     y = self.nasa_data["bt_h_fore"].flatten()
    #     x_h_fore, y_h_fore, m_h_fore, b_h_fore, y_fit_h_fore, r_squared = (
    #         self.scatter_stats(x, y)
    #     )

    #     x = self.rgb_data["bt_h_aft"].flatten()
    #     y = self.nasa_data["bt_h_aft"].flatten()
    #     x_h_aft, y_h_aft, m_h_aft, b_h_aft, y_fit_h_aft, r_squared = self.scatter_stats(
    #         x, y
    #     )

    #     x = self.rgb_data["bt_v_fore"].flatten()
    #     y = self.nasa_data["bt_v_fore"].flatten()
    #     x_v_fore, y_v_fore, m_v_fore, b_v_fore, y_fit_v_fore, r_squared = (
    #         self.scatter_stats(x, y)
    #     )

    #     x = self.rgb_data["bt_v_aft"].flatten()
    #     y = self.nasa_data["bt_v_aft"].flatten()

    #     x_v_aft, y_v_aft, m_v_aft, b_v_aft, y_fit_v_aft, r_squared = self.scatter_stats(
    #         x, y
    #     )

    #     fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    #     axs[0, 0].scatter(x_h_fore, y_h_fore)
    #     axs[0, 0].plot(x_h_fore, y_fit_h_fore, color="red")
    #     axs[0, 0].legend(title=f"$R^2 = {r_squared:.3f}$")
    #     axs[0, 0].set_title("bt_h_fore")
    #     axs[0, 0].set_xlabel("RGB BT [K]")
    #     axs[0, 0].set_ylabel("NASA BT [K]")

    #     axs[0, 1].scatter(x_h_aft, y_h_aft)
    #     axs[0, 1].plot(x_h_aft, y_fit_h_aft, color="red")
    #     axs[0, 1].legend(title=f"$R^2 = {r_squared:.3f}$")
    #     axs[0, 1].set_title("bt_h_aft")
    #     axs[0, 1].set_xlabel("RGB BT [K]")
    #     axs[0, 1].set_ylabel("NASA BT [K]")

    #     axs[1, 0].scatter(x_v_fore, y_v_fore)
    #     axs[1, 0].plot(x_v_fore, y_fit_v_fore, color="red")
    #     axs[1, 0].legend(title=f"$R^2 = {r_squared:.3f}$")
    #     axs[1, 0].set_title("bt_v_fore")
    #     axs[1, 0].set_xlabel("RGB BT [K]")
    #     axs[1, 0].set_ylabel("NASA BT [K]")

    #     axs[1, 1].scatter(x_v_aft, y_v_aft)
    #     axs[1, 1].plot(x_v_aft, y_fit_v_aft, color="red")
    #     axs[1, 1].legend(title=f"$R^2 = {r_squared:.3f}$")
    #     axs[1, 1].set_title("bt_v_aft")
    #     axs[1, 1].set_xlabel("RGB BT [K]")
    #     axs[1, 1].set_ylabel("NASA BT [K]")

    #     img_path = repo_root.joinpath(
    #         "output/MS3_verification_tests/T_12/T_12_scatter.png"
    #     )  # ""
    #     plt.savefig(img_path, dpi=300)

    #     plt.show()

    def calculate_differences(self):
        """
        Calculate the differences between RGB and NASA data.

        Returns
        -------
        dict
            A dictionary with keys as variable names and values containing:
            - mean_diff : float
                Mean absolute difference between RGB and NASA data.
            - percent_diff : float
                Mean percentage difference relative to NASA data.
        """
        results = {}
        for key in ["bt_h_fore", "bt_h_aft", "bt_v_fore", "bt_v_aft"]:
            diff = abs(self.rgb_data[key] - self.nasa_data[key])
            mean_diff = nanmean(diff)
            percent_diff = (mean_diff / nanmean(self.nasa_data[key])) * 100
            results[key] = {
                "mean_diff": mean_diff,
                "percent_diff": percent_diff,
            }
        return results


# def run_python_subprocess(config_path):
#     """
#     Runs a Python script as a subprocess, displaying output in real-time.

#     Parameters:
#     ----------
#     script_path : str
#         Path to the Python script to be executed.
#     *args : str
#         Additional arguments to pass to the script.

#     Returns:
#     ----------
#     int
#         Exit code of the subprocess.
#     """
#     try:
#         # Construct the command
#         # command = [sys.executable, script_path] + list(args)
#         command = [
#             "python",
#             "-m",
#             "cimr_rgb",
#             str(config_path),
#         ]  # Adjust command if needed

#         # Run the subprocess and allow real-time output
#         result = sbps.run(
#             command,
#             stdout=sys.stdout,  # Redirect stdout to the parent process's stdout
#             stderr=sys.stderr,  # Redirect stderr to the parent process's stderr
#         )

#         # Return the exit code
#         return result.returncode
#     except Exception as e:
#         print(f"Error occurred: {e}", file=sys.stderr)
#         return -1


# Both fixture and this approach suppose to work
def run_python_subprocess(config_path):
    """
    Runs a Python subprocess to execute the CIMR-RGB module with real-time output.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    int
        Exit code of the subprocess.
    """
    try:
        command = ["python", "-m", "cimr_rgb", str(config_path)]
        # Run the subprocess with output streaming to the console
        result = sbps.run(
            command,
            stdout=sys.stdout,  # Redirect subprocess stdout to console
            stderr=sys.stderr,  # Redirect subprocess stderr to console
            text=True,
        )
        return result.returncode  # Return exit code
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        return -1


# @pytest.fixture
# def run_subprocess():
#     """
#     Pytest fixture to execute a subprocess with real-time output.

#     This fixture provides a callable function to run a subprocess,
#     streaming the output directly to the console during execution.

#     Returns
#     -------
#     callable
#         A function that takes a configuration file path as input and
#         runs the subprocess. The function returns the subprocess exit code.

#     Examples
#     --------
#     def test_example(run_python_subprocess):
#         exit_code = run_python_subprocess("path/to/config.xml")
#         assert exit_code == 0
#     """

#     def _run(config_path):
#         try:
#             command = ["python", "-m", "cimr_rgb", str(config_path)]
#             result = sbps.run(
#                 command,
#                 stdout=sys.stdout,  # Stream subprocess stdout live
#                 stderr=sys.stderr,  # Stream subprocess stderr live
#                 text=True,
#             )
#             return result.returncode
#         except Exception as e:
#             print(f"Error occurred: {e}", file=sys.stderr)
#             return -1

#     return _run


@pytest.fixture
def setup_paths():
    """
    Set up file paths for testing.

    Returns
    -------
    tuple
        A tuple containing:
        - rgb_data : str
            Path to the RGB remapped netCDF file.
        - nasa_data : str
            Path to the NASA remapped HDF5 file.
        - config_path : str
            Path to the XML configuration file.
    """

    repo_root = grasp_io.find_repo_root()
    rgb_data = repo_root.joinpath(
        "output/MS3_verification_tests/T_12/SMAP_L1C_IDS_36km_test.nc"
    )
    nasa_data = repo_root.joinpath(
        "dpr/L1C/SMAP/NASA/SMAP_L1C_TB_47185_D_20231201T212059_R19240_002.h5"
    )
    config_path = repo_root.joinpath("tests/MS3_verification_tests/T_12/T_12_IDS.xml")

    return rgb_data, nasa_data, config_path


def test_subprocess_execution(setup_paths, run_subprocess):
    """
    Test the execution of the subprocess.

    Parameters
    ----------
    setup_paths : tuple
        Tuple containing file paths for testing.

    Asserts
    -------
    Exit code of the subprocess is 0.
    """

    _, _, config_path = setup_paths
    # exit_code = run_python_subprocess(config_path)
    exit_code = run_subprocess(config_path)

    assert exit_code == 0, "Subprocess execution failed with a non-zero exit code."


def test_smap_comparison(setup_paths):
    rgb_data, nasa_data, _ = setup_paths
    comparison = SMAP_comparison(rgb_data, nasa_data)
    results = comparison.calculate_differences()

    for key, stats in results.items():
        print(
            f"{key}: Average Mean Diff = {stats['mean_diff']}, Average Percent Diff = {stats['percent_diff']}%"
        )
        assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
        assert stats["percent_diff"] < 0.5, f"Percent difference for {key} is too high!"


# if __name__ == "__main__":
#     # IDS
#     config_path = grasp_io.find_repo_root().joinpath(
#         "tests/MS3_verification_tests/T_12/T_12_IDS.xml"
#     )
#     exit_code = run_python_subprocess(config_path=config_path)
#     print(f"Subprocess exited with code: {exit_code}")

#     # CIMR_comparison(ids_data_path, rsir_data_path)
#     SMAP_comparison(rgb_data, nasa_data).map_compare()
#     SMAP_comparison(rgb_data, nasa_data).scatter_compare()
