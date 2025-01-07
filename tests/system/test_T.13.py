# Testing script for T.13
# Remapping of L1b CIMR L-band data with NN (nearest neighbor) algorithm on an EASE2 North grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)

import sys
import subprocess as sbps

from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import pytest

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io

GRID = "EASE2_N9km"
PROJECTION = "N"
BAND = "L_BAND"

# class SMAPComparison:
#     """
#     Class for comparing remapped CIMR L1b L-band data using NN and IDS algorithms.

#     Parameters
#     ----------
#     ids_data_path : str
#         Path to the IDS remapped netCDF file.
#     nn_data_path : str
#         Path to the NN remapped netCDF file.

#     Attributes
#     ----------
#     ids_data : dict
#         Dictionary containing gridded variables from the IDS data.
#     nn_data : dict
#         Dictionary containing gridded variables from the NN data.
#     """

#     def __init__(self, ids_data_path, nn_data_path):
#         self.ids_data = self.get_netcdf_data(ids_data_path)
#         self.nn_data = self.get_netcdf_data(nn_data_path)

#     @staticmethod
#     def get_netcdf_data(path):
#         """
#         Extract and grid data from a netCDF file.

#         Parameters
#         ----------
#         path : str
#             Path to the netCDF file.

#         Returns
#         -------
#         dict
#             A dictionary containing gridded brightness temperature variables.
#         """
#         import netCDF4 as nc

#         gridded_vars = {}
#         with nc.Dataset(path, "r") as f:
#             data = f["N"]
#             measurement = data["Measurement"]
#             band = measurement["L_BAND"]

#             for bt in ["bt_h_fore", "bt_h_aft"]:
#                 if "fore" in bt:
#                     row = array(band["cell_row_fore"][:])
#                     col = array(band["cell_col_fore"][:])
#                 elif "aft" in bt:
#                     row = array(band["cell_row_aft"][:])
#                     col = array(band["cell_col_aft"][:])
#                 var = array(band[bt][:])
#                 grid = full(
#                     (GRIDS["EASE2_N9km"]["n_rows"], GRIDS["EASE2_N9km"]["n_cols"]), nan
#                 )

#                 for count, sample in enumerate(var):
#                     if sample == 9.969209968386869e36 or sample == 0.0:
#                         continue
#                     grid[row[count], col[count]] = sample
#                 gridded_vars[bt] = grid
#         return gridded_vars

#     def calculate_differences(self):
#         """
#         Calculate the differences between NN and IDS data.

#         Returns
#         -------
#         dict
#             A dictionary with keys as variable names and values containing:
#             - mean_diff : float
#                 Mean absolute difference between NN and IDS data.
#             - percent_diff : float
#                 Mean percentage difference relative to IDS data.
#         """
#         results = {}
#         for key in ["bt_h_fore", "bt_h_aft"]:
#             diff = abs(self.nn_data[key] - self.ids_data[key])
#             mean_diff = nanmean(diff)
#             percent_diff = (mean_diff / nanmean(self.ids_data[key])) * 100
#             results[key] = {
#                 "mean_diff": mean_diff,
#                 "percent_diff": percent_diff,
#             }
#         return results


# def run_python_subprocess(config_path):
#     """
#     Executes a subprocess with real-time output.

#     Parameters
#     ----------
#     config_path : str
#         Path to the configuration file.

#     Returns
#     -------
#     int
#         Exit code of the subprocess.
#     """
#     try:
#         command = ["python", "-m", "cimr_rgb", str(config_path)]
#         result = sbps.run(
#             command,
#             stdout=sys.stdout,  # Stream subprocess stdout live
#             stderr=sys.stderr,  # Stream subprocess stderr live
#             text=True,
#         )
#         return result.returncode
#     except Exception as e:
#         print(f"Error occurred: {e}", file=sys.stderr)
#         return -1


# @pytest.fixture
# def setup_paths():
#     """
#     Set up file paths for testing.

#     Returns
#     -------
#     tuple
#         A tuple containing:
#         - ids_data : str
#             Path to the IDS remapped netCDF file.
#         - nn_data : str
#             Path to the NN remapped netCDF file.
#         - config_paths : dict
#             Dictionary containing configuration paths for NN and IDS tests.
#     """
#     repo_root = grasp_io.find_repo_root()
#     ids_data = repo_root.joinpath(
#         "output/MS3_verification_tests/T_13/CIMR_L1C_IDS_9km_test.nc"
#     )
#     nn_data = repo_root.joinpath(
#         "output/MS3_verification_tests/T_13/CIMR_L1C_NN_9km_test.nc"
#     )
#     config_paths = {
#         "NN": repo_root.joinpath("tests/system/configs/T_13_NN.xml"),
#         "IDS": repo_root.joinpath("tests/system/configs/T_13_IDS.xml"),
#     }
#     return ids_data, nn_data, config_paths


# def test_subprocess_execution(setup_paths):
#     """
#     Test the execution of the subprocess for NN and IDS.

#     Parameters
#     ----------
#     setup_paths : tuple
#         Tuple containing file paths for testing.

#     Asserts
#     -------
#     Exit code of the subprocess is 0 for both NN and IDS.
#     """
#     _, _, config_paths = setup_paths
#     for key, config_path in config_paths.items():
#         exit_code = run_python_subprocess(config_path)
#         assert exit_code == 0, (
#             f"Subprocess execution for {key} failed with a non-zero exit code."
#         )


# def test_smap_comparison(setup_paths):
#     """
#     Test the SMAP data comparison.

#     Parameters
#     ----------
#     setup_paths : tuple
#         Tuple containing file paths for testing.

#     Asserts
#     -------
#     Mean and percentage differences are below specified thresholds.
#     """
#     ids_data, nn_data, _ = setup_paths
#     comparison = SMAPComparison(ids_data, nn_data)
#     results = comparison.calculate_differences()

#     for key, stats in results.items():
#         print(
#             f"{key}: Mean Diff = {stats['mean_diff']}, Percent Diff = {stats['percent_diff']}%"
#         )
#         assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
#         assert stats["percent_diff"] < 1.0, f"Percent difference for {key} is too high!"


#
# bt_h_aft_diff = abs(self.nn_data["bt_h_aft"] - self.ids_data["bt_h_aft"])
# fore_percent_diff = (fore_mean_diff / nanmean(self.ids_data["bt_h_fore"])) * 100
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


# TODO: Make this one work
# @pytest.mark.parametrize("setup_paths", ["T_13"], indirect=True)
# def test_T13_comparison(setup_paths, calculate_differences):
#     """
#     Test comparison of brightness temperature (BT) variables between RGB and NASA datasets.

#     This test verifies that the differences between the brightness temperature variables
#     (bt_h_fore, bt_h_aft, bt_v_fore, bt_v_aft) from the RGB and NASA datasets are within
#     acceptable thresholds. It utilizes a fixture to calculate differences for specified variables
#     and asserts the results against pre-defined tolerances.

#     Parameters
#     ----------
#     setup_paths : tuple
#         A pytest fixture providing file paths for the RGB and NASA datasets and configuration files.
#         The paths are dynamically retrieved based on the test scenario ("T_13").
#     calculate_differences : callable
#         A pytest fixture providing a function to calculate the mean absolute and percentage differences
#         between two datasets for specified variables.

#     Asserts
#     -------
#     - Mean difference for each variable is less than 1.0.
#     - Mean percentage difference for each variable is less than 0.5%.

#     Raises
#     ------
#     AssertionError
#         If either the mean difference or mean percentage difference exceeds the defined threshold
#         for any variable.

#     Example
#     -------
#     This test is parameterized for the "T_13" scenario:

#     >>> pytest test_script.py --setup-paths=T_13

#     Output:
#     -------
#     bt_h_fore: Average Mean Diff = 0.002, Average Percent Diff = 0.01%
#     bt_h_aft: Average Mean Diff = 0.003, Average Percent Diff = 0.02%
#     bt_v_fore: Average Mean Diff = 0.004, Average Percent Diff = 0.03%
#     bt_v_aft: Average Mean Diff = 0.002, Average Percent Diff = 0.01%
#     """

#     rgb_data_path, nasa_data_path, _ = setup_paths
#     rgb_data = get_netcdf_data(path=rgb_data_path)
#     nasa_data = get_hdf5_data(nasa_data_path)

#     variables_list = ["bt_h_fore", "bt_h_aft", "bt_v_fore", "bt_v_aft"]

#     results = calculate_differences(
#         data1=rgb_data, data2=nasa_data, variables_list=variables_list
#     )

#     for key, stats in results.items():
#         print(
#             f"{key}: Average Mean Diff = {stats['mean_diff']:.3f}, Average Percent Diff = {stats['percent_diff']:.3f}%"
#         )
#         assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
#         assert stats["percent_diff"] < 0.5, f"Percent difference for {key} is too high!"
