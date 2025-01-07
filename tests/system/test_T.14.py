import sys
import subprocess as sbps

from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import pytest

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io

BANDS = "7_BAND", "18_BAND", "89b_BAND"
PROJECTION = "S"
GRID = "EASE2_S9km"

# class AMSR2Comparison:
#     """
#     Class for comparing remapped AMSR2 L1b data using IDS and DIB algorithms.

#     Parameters
#     ----------
#     ids_data_path : str
#         Path to the IDS remapped netCDF file.
#     dib_data_path : str
#         Path to the DIB remapped netCDF file.

#     Attributes
#     ----------
#     ids_data : dict
#         Dictionary containing gridded variables from the IDS data.
#     dib_data : dict
#         Dictionary containing gridded variables from the DIB data.
#     """

#     def __init__(self, ids_data_path, dib_data_path):
#         self.ids_data = self.get_netcdf_data(ids_data_path)
#         self.dib_data = self.get_netcdf_data(dib_data_path)

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
#             A dictionary containing gridded brightness temperature variables for each band.
#         """
#         import netCDF4 as nc

#         data_dict = {}
#         with nc.Dataset(path, "r") as f:
#             data = f["S"]
#             measurement = data["Measurement"]
#             for band in ["7_BAND", "18_BAND", "89b_BAND"]:
#                 gridded_vars = {}
#                 band_data = measurement[band]
#                 cell_row = band_data["cell_row"][:]
#                 cell_col = band_data["cell_col"][:]
#                 for bt in ["bt_h"]:
#                     var = array(band_data[bt][:])
#                     grid = full(
#                         (GRIDS["EASE2_S9km"]["n_rows"], GRIDS["EASE2_S9km"]["n_cols"]),
#                         nan,
#                     )

#                     for count, sample in enumerate(var):
#                         if sample == 9.969209968386869e36:
#                             continue
#                         grid[cell_row[count], cell_col[count]] = sample
#                     gridded_vars[bt] = grid
#                 data_dict[band] = gridded_vars
#         return data_dict

#     def calculate_differences(self):
#         """
#         Calculate the differences between DIB and IDS data for each band.

#         Returns
#         -------
#         dict
#             A dictionary with keys as band names and values containing:
#             - mean_diff : float
#                 Mean absolute difference between DIB and IDS data.
#             - percent_diff : float
#                 Mean percentage difference relative to IDS data.
#         """
#         results = {}
#         for band in ["7_BAND", "18_BAND", "89b_BAND"]:
#             diff = abs(self.dib_data[band]["bt_h"] - self.ids_data[band]["bt_h"])
#             mean_diff = nanmean(diff)
#             percent_diff = (mean_diff / nanmean(self.ids_data[band]["bt_h"])) * 100
#             results[band] = {
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
#         - dib_data : str
#             Path to the DIB remapped netCDF file.
#         - config_paths : dict
#             Dictionary containing configuration paths for IDS and DIB tests.
#     """
#     repo_root = grasp_io.find_repo_root()
#     ids_data = repo_root.joinpath(
#         "output/MS3_verification_tests/T_14/AMSR2_L1C_IDS_9km_test.nc"
#     )
#     dib_data = repo_root.joinpath(
#         "output/MS3_verification_tests/T_14/AMSR2_L1C_DIB_9km_test.nc"
#     )
#     config_paths = {
#         "IDS": repo_root.joinpath("tests/MS3_verification_tests/T_14/T_14_IDS.xml"),
#         "DIB": repo_root.joinpath("tests/MS3_verification_tests/T_14/T_14_DIB.xml"),
#     }
#     return ids_data, dib_data, config_paths


# def test_subprocess_execution(setup_paths):
#     """
#     Test the execution of the subprocess for IDS and DIB.

#     Parameters
#     ----------
#     setup_paths : tuple
#         Tuple containing file paths for testing.

#     Asserts
#     -------
#     Exit code of the subprocess is 0 for both IDS and DIB.
#     """
#     _, _, config_paths = setup_paths
#     for key, config_path in config_paths.items():
#         exit_code = run_python_subprocess(config_path)
#         assert exit_code == 0, (
#             f"Subprocess execution for {key} failed with a non-zero exit code."
#         )


# def test_amsr2_comparison(setup_paths):
#     """
#     Test the AMSR2 data comparison.

#     Parameters
#     ----------
#     setup_paths : tuple
#         Tuple containing file paths for testing.

#     Asserts
#     -------
#     Mean and percentage differences are below specified thresholds.
#     """
#     ids_data, dib_data, _ = setup_paths
#     comparison = AMSR2Comparison(ids_data, dib_data)
#     results = comparison.calculate_differences()

#     for band, stats in results.items():
#         print(
#             f"{band}: Mean Diff = {stats['mean_diff']}, Percent Diff = {stats['percent_diff']}%"
#         )
#         assert stats["mean_diff"] < 1.0, f"Mean difference for {band} is too high!"
#         assert stats["percent_diff"] < 1.0, (
#             f"Percent difference for {band} is too high!"
#         )


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
