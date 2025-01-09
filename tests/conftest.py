# """
# Contains shared fixtures and other configurations across all tests
# """

# import pytest


# @pytest.fixture
# def valid_config_file():
#     """
#     Creates a valid configuration XML file for testing.
#     """
#     root = Element("config")
#
#     # OutputData
#     output_data = SubElement(root, "OutputData")
#     SubElement(output_data, "output_path").text = "./test_output"
#     SubElement(output_data, "save_to_disk").text = "True"

#     # LoggingParams
#     logging_params = SubElement(root, "LoggingParams")
#     SubElement(logging_params, "config_path").text = "./src/cimr_rgb/logger_config.json"
#     SubElement(logging_params, "decorate").text = "True"

#     # InputData
#     input_data = SubElement(root, "InputData")
#     SubElement(input_data, "type").text = "SMAP"
#     SubElement(input_data, "path").text = "./dpr/L1B/SMAP/SMAP_L1B_TB_47185_D_20231201T212120_R18290_001.h5"
#     SubElement(input_data, "antenna_patterns_path").text = "./dpr/antenna_patterns"
#     SubElement(input_data, "quality_control").text = "True"
#     SubElement(input_data, "target_band").text = "L"
#     SubElement(input_data, "split_fore_aft").text = "True"

#     # GridParams
#     grid_params = SubElement(root, "GridParams")
#     SubElement(grid_params, "grid_type").text = "L1C"
#     SubElement(grid_params, "grid_definition").text = "EASE2_G9km"
#     SubElement(grid_params, "projection_definition").text = "G"
#     # 800 900 3200 3260 south 1450 1460 1780 1800 north 650 660 1250 1260
#     SubElement(grid_params, "reduced_grid_inds").text = ""

#     # ReGridderParams
#     regridder_params = SubElement(root, "ReGridderParams")
#     SubElement(regridder_params, "regridding_algorithm").text = "IDS"
#     SubElement(regridder_params, "search_radius").text = "18"
#     SubElement(regridder_params, "max_neighbours").text = "20"
#     SubElement(regridder_params, "variables_to_regrid").text = "bt_h bt_v"
#     SubElement(regridder_params, "source_antenna_method").text = "gaussian"
#     SubElement(regridder_params, "target_antenna_method").text = "real"
#     SubElement(regridder_params, "polarisation_method").text = "scalar"
#     SubElement(regridder_params, "source_antenna_threshold").text = "0.5"
#     SubElement(regridder_params, "target_antenna_threshold").text = "0.5"
#     SubElement(regridder_params, "max_theta_antenna_patterns").text = "40."
#     SubElement(regridder_params, "MRF_grid_definition").text = "EASE2_G3km"
#     SubElement(regridder_params, "MRF_projection_definition").text = "G"
#     SubElement(regridder_params, "source_gaussian_params").text = "100000 100000"
#     SubElement(regridder_params, "target_gaussian_params").text = "100000 100000"
#     SubElement(regridder_params, "boresight_shift").text = "True"
#     # For rSIR and BG
#     SubElement(regridder_params, "rsir_iteration").text = "15"
#     SubElement(regridder_params, "bg_smoothing").text = "1"


#     #  <rsir_iteration>15</rsir_iteration>
#     #  <bg_smoothing>1</bg_smoothing>

#     # (Temporary) Config file will be created inside:
#     # valid_config_file = '/tmp/tmpcpc5g83y.xml'

#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
#     tree = ElementTree(root)
#     tree.write(temp_file.name)
#     temp_file.close()
#     yield temp_file.name
#     os.unlink(temp_file.name)


# @pytest.fixture
# def invalid_config_file():
#     """
#     Creates an invalid configuration XML file for testing.
#     """
#     root = Element("config")
#     # Missing essential sections and attributes for validation tests
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
#     tree = ElementTree(root)
#     tree.write(temp_file.name)
#     temp_file.close()
#     yield temp_file.name
#     os.unlink(temp_file.name)


# def test_valid_config_file(valid_config_file):
#     """
#     Test that a valid config file initializes the class without errors.
#     """

#     # Passing in the dynamically created config file via valid_config_file method
#     config = ConfigFile(valid_config_file)
#     assert config.output_path.name == "test_output"
#     # Checking the name for logger_config.json
#     # (i.e., it should be logger_config.json == logger_config.json)
#     assert config.logpar_config.name == "logger_config.json"
#     assert config.input_data_type == "SMAP"
#     assert config.grid_type == "L1C"
#     assert config.regridding_algorithm == "IDS"


"""
Contains shared fixtures and other configurations across all tests
"""

import sys
import pathlib as pb
import subprocess as sbps
import json

import pytest
import numpy as np

from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io


@pytest.fixture
def setup_paths(request):
    """
    Fixture to set up file paths for testing based on a specified scenario from a JSON configuration file.

    This fixture reads a JSON configuration file containing predefined scenarios and their associated file paths.
    It resolves the paths relative to the repository root, allowing dynamic selection of file paths based on the
    scenario passed via the `request.param` parameter. The resolved paths are returned as a tuple for use in tests.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture request object. The `param` attribute of the request specifies the scenario
        to be used for resolving file paths.

    Returns
    -------
    tuple
        A tuple containing:
        - datapath1 : pathlib.Path
            The resolved path to the first data file.
        - datapath2 : pathlib.Path
            The resolved path to the second data file.
        - config_paths : pathlib.Path or list or dict
            The resolved configuration file paths, which may be a single path, a list of paths,
            or a dictionary of paths depending on the scenario configuration.

    Raises
    ------
    ValueError
        If the specified scenario is not defined in the JSON configuration file.

    Notes
    -----
    - The JSON configuration file should be located at `tests/system/test_scenarios.json` relative to the repository root.
    - Each scenario in the JSON file must define the keys `datapath1`, `datapath2`, and `config_paths`.

    Example JSON Configuration
    ---------------------------
    {
        "T_12": {
            "datapath1": "output/MS3_verification_tests/T_12/SMAP_L1C_IDS_36km_test.nc",
            "datapath2": "dpr/L1C/SMAP/NASA/SMAP_L1C_TB_47185_D_20231201T212059_R19240_002.h5",
            "config_paths": ["tests/system/configs/T_12_IDS.xml"]
        },
        "T_13": {
            "datapath1": "output/MS3_verification_tests/T_13/CIMR_L1C_IDS_9km_test.nc",
            "datapath2": "output/MS3_verification_tests/T_13/CIMR_L1C_NN_9km_test.nc",
            "config_paths": {
                "NN": "tests/system/configs/T_13_NN.xml",
                "IDS": "tests/system/configs/T_13_IDS.xml"
            }
        }
    }

    Examples
    --------
    # Example usage in a test
    @pytest.mark.parametrize("setup_paths", ["T_12"], indirect=True)
    def test_example(setup_paths):
        datapath1, datapath2, config_paths = setup_paths
        assert datapath1.exists()
        assert datapath2.exists()
    """

    # Load the configuration file
    test_config_json = "test_scenarios.json"
    # Get the repository root
    repo_root = grasp_io.find_repo_root()
    test_config_path = repo_root.joinpath("tests", "system", f"{test_config_json}")
    with open(test_config_path, "r") as f:
        scenarios = json.load(f)

    # Retrieve the scenario from the test parameter
    scenario = getattr(request, "param", None)
    if scenario not in scenarios:
        raise ValueError(f"Scenario '{scenario}' not defined in {test_config_json}.")

    # Resolve paths relative to the repo root for a given dictionary
    resolved_paths_dict = {}
    for key, value in scenarios[scenario].items():
        if isinstance(value, dict):
            # Resolve paths for dictionary values
            resolved_paths_dict[key] = {k: repo_root / v for k, v in value.items()}
        elif isinstance(value, list):
            # Resolve paths for list values
            resolved_paths_dict[key] = [repo_root / v for v in value]
        else:
            # Resolve paths for single values
            resolved_paths_dict[key] = repo_root / value

    datapath1 = resolved_paths_dict["datapath1"]
    datapath2 = resolved_paths_dict["datapath2"]
    config_paths = resolved_paths_dict["config_paths"]

    return datapath1, datapath2, config_paths


@pytest.fixture
def run_subprocess():
    """
    Pytest fixture to execute a subprocess with real-time output.

    This fixture provides a callable function to run a subprocess,
    streaming the output directly to the console during execution.

    Returns
    -------
    callable
        A function that takes a configuration file path as input and
        runs the subprocess. The function returns the subprocess exit code.

    Examples
    --------
    def test_example(run_python_subprocess):
        exit_code = run_python_subprocess("path/to/config.xml")
        assert exit_code == 0
    """

    def _run(config_path):
        try:
            command = ["python", "-m", "cimr_rgb", str(config_path)]
            result = sbps.run(
                command,
                stdout=sys.stdout,  # Stream subprocess stdout live
                stderr=sys.stderr,  # Stream subprocess stderr live
                text=True,
            )
            return result.returncode
        except Exception as e:
            print(f"Error occurred: {e}", file=sys.stderr)
            return -1

    return _run


@pytest.fixture
def calculate_differences():
    """
    Fixture to calculate differences between two datasets.

    This fixture provides a callable function that computes the mean absolute
    difference and mean percentage difference for specified variables in two
    datasets. It is designed to work with datasets where variables are accessible
    as dictionary keys.

    Returns
    -------
    callable
        A function that takes the following parameters:
        - data1 : dict
            The first dataset, containing variables as keys and their corresponding
            data as values.
        - data2 : dict
            The second dataset, containing variables as keys and their corresponding
            data as values.
        - variables_list : list of str
            A list of variable names (keys) for which the differences are to be calculated.

        The function returns:
        -------
        dict
            A dictionary where each key corresponds to a variable name from `variables_list`,
            and each value is a dictionary containing:
            - mean_diff : float
                The mean absolute difference between `data1` and `data2` for the variable.
            - percent_diff : float
                The mean percentage difference relative to `data2` for the variable.

    Examples
    --------
    >>> diff_calculator = calculate_differences()
    >>> data1 = {"temperature": np.array([300, 305, 310]), "humidity": np.array([50, 55, 60])}
    >>> data2 = {"temperature": np.array([298, 307, 312]), "humidity": np.array([52, 53, 58])}
    >>> variables = ["temperature", "humidity"]
    >>> results = diff_calculator(data1, data2, variables)
    >>> print(results)
    {
        "temperature": {"mean_diff": 2.0, "percent_diff": 0.64},
        "humidity": {"mean_diff": 2.0, "percent_diff": 3.77}
    }
    """

    def _calculate(data1, data2, variables_list):
        results = {}

        for key in variables_list:
            diff = abs(data1[key] - data2[key])
            mean_diff = np.nanmean(diff)
            percent_diff = (mean_diff / np.nanmean(data2[key])) * 100
            results[key] = {
                "mean_diff": mean_diff,
                "percent_diff": percent_diff,
            }
        return results

    return _calculate


@pytest.fixture
def get_netcdf_data():
    def _get_data(datapath, variables_list, projection, band, grid):
        import netCDF4 as nc

        PROJECTION = projection
        BAND = band
        GRID = grid

        gridded_vars = {}

        with nc.Dataset(datapath, "r") as f:
            data = f[f"{PROJECTION}"]
            measurement = data["Measurement"]
            band_data = measurement[BAND]

            # Whether it is L1R or L1C (BAND_TARGET is inside L1R)
            if "BAND_TARGET" in PROJECTION:
                for bt in variables_list:
                    bt_h = band_data[bt][:]
                    gridded_vars[bt] = bt_h
            else:
                for bt in variables_list:
                    if "fore" in bt:
                        cell_row = np.array(band_data["cell_row_fore"][:])
                        cell_col = np.array(band_data["cell_col_fore"][:])
                    elif "aft" in bt:
                        cell_row = np.array(band_data["cell_row_aft"][:])
                        cell_col = np.array(band_data["cell_col_aft"][:])
                    else:
                        cell_row = np.array(band_data["cell_row"][:])
                        cell_col = np.array(band_data["cell_col"][:])

                    var = np.array(band_data[bt][:])
                    grid = np.full(
                        (GRIDS[GRID]["n_rows"], GRIDS[GRID]["n_cols"]), np.nan
                    )

                    # Here, different tests had different sample conditions, so
                    # need to check whether it is very relevant if I mix all these
                    # conditions here.
                    for count, sample in enumerate(var):
                        if sample == 9.969209968386869e36:
                            continue
                        if sample == 0.0:
                            print("a sample was zero")
                            continue
                        # This part is from LW (T17)
                        if cell_row[count] == -9223372036854775806:
                            continue
                        if cell_col[count] == -9223372036854775806:
                            continue

                        grid[int(cell_row[count]), int(cell_col[count])] = sample

                    gridded_vars[bt] = grid

            return gridded_vars

    return _get_data
