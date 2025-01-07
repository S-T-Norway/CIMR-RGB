import sys
import subprocess as sbps

from numpy import array, nan, nanmean, polyfit, isnan, isinf
import pytest
import matplotlib.pyplot as plt

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io

BAND = "L_BAND"
PROJECTION = "C_BAND_TARGET"


# class CIMRComparison:
#     """
#     Class for comparing remapped CIMR L-band simulated data using IDS and BG algorithms.

#     Parameters
#     ----------
#     ids_data_path : str
#         Path to the IDS remapped netCDF file.
#     bg_data_path : str
#         Path to the BG remapped netCDF file.

#     Attributes
#     ----------
#     ids_data : dict
#         Dictionary containing gridded variables from the IDS data.
#     bg_data : dict
#         Dictionary containing gridded variables from the BG data.
#     """

#     def __init__(self, ids_data_path, bg_data_path):
#         self.ids_data = self.get_netcdf_data(ids_data_path)
#         self.bg_data = self.get_netcdf_data(bg_data_path)

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
#             data = f["C_BAND_TARGET"]
#             measurement = data["Measurement"]
#             band = measurement["L_BAND"]
#             gridded_vars["bt_h"] = array(band["bt_h"][:])
#         return gridded_vars

#     def calculate_differences(self):
#         """
#         Calculate the differences between BG and IDS data.

#         Returns
#         -------
#         dict
#             A dictionary with keys as variable names and values containing:
#             - mean_diff : float
#                 Mean absolute difference between BG and IDS data.
#             - percent_diff : float
#                 Mean percentage difference relative to IDS data.
#         """
#         results = {}
#         diff = abs(self.bg_data["bt_h"] - self.ids_data["bt_h"])
#         mean_diff = nanmean(diff)
#         percent_diff = (mean_diff / nanmean(self.ids_data["bt_h"])) * 100
#         results["bt_h"] = {
#             "mean_diff": mean_diff,
#             "percent_diff": percent_diff,
#         }
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
#         - bg_data : str
#             Path to the BG remapped netCDF file.
#         - config_paths : dict
#             Dictionary containing configuration paths for IDS and BG tests.
#     """
#     repo_root = grasp_io.find_repo_root()
#     ids_data = repo_root.joinpath(
#         "output/MS3_verification_tests/T_19/CIMR_L1R_IDS_test.nc"
#     )
#     bg_data = repo_root.joinpath(
#         "output/MS3_verification_tests/T_19/CIMR_L1R_BG_test.nc"
#     )
#     config_paths = {
#         "IDS": repo_root.joinpath("tests/MS3_verification_tests/T_19/T_19_IDS.xml"),
#         "BG": repo_root.joinpath("tests/MS3_verification_tests/T_19/T_19_BG.xml"),
#     }
#     return ids_data, bg_data, config_paths


# def test_subprocess_execution(setup_paths):
#     """
#     Test the execution of the subprocess for IDS and BG.

#     Parameters
#     ----------
#     setup_paths : tuple
#         Tuple containing file paths for testing.

#     Asserts
#     -------
#     Exit code of the subprocess is 0 for both IDS and BG.
#     """
#     _, _, config_paths = setup_paths
#     for key, config_path in config_paths.items():
#         exit_code = run_python_subprocess(config_path)
#         assert exit_code == 0, (
#             f"Subprocess execution for {key} failed with a non-zero exit code."
#         )


# def test_cimr_comparison(setup_paths):
#     """
#     Test the CIMR data comparison.

#     Parameters
#     ----------
#     setup_paths : tuple
#         Tuple containing file paths for testing.

#     Asserts
#     -------
#     Mean and percentage differences are below specified thresholds.
#     """
#     ids_data, bg_data, _ = setup_paths
#     comparison = CIMRComparison(ids_data, bg_data)
#     results = comparison.calculate_differences()

#     for key, stats in results.items():
#         print(
#             f"{key}: Mean Diff = {stats['mean_diff']}, Percent Diff = {stats['percent_diff']}%"
#         )
#         assert stats["mean_diff"] < 1.0, f"Mean difference for {key} is too high!"
#         assert stats["percent_diff"] < 1.0, f"Percent difference for {key} is too high!"


@pytest.mark.parametrize("setup_paths", ["T_19"], indirect=True)
def test_T19_execution(setup_paths, run_subprocess):
    """
    Test the execution of the subprocess for the T_19 scenario.

    Parameters
    ----------
    setup_paths : dict
        Dictionary containing paths for the T_19 scenario, retrieved using the `setup_paths` fixture.
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
