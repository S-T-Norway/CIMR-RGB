"""
Contains shared fixtures and other configurations across all tests 
"""

import pytest 


@pytest.fixture
def valid_config_file():
    """
    Creates a valid configuration XML file for testing.
    """
    root = Element("config")
    
    # OutputData
    output_data = SubElement(root, "OutputData")
    SubElement(output_data, "output_path").text = "./test_output"
    SubElement(output_data, "save_to_disk").text = "True"

    # LoggingParams
    logging_params = SubElement(root, "LoggingParams")
    SubElement(logging_params, "config_path").text = "./src/cimr_rgb/logger_config.json"
    SubElement(logging_params, "decorate").text = "True"

    # InputData
    input_data = SubElement(root, "InputData")
    SubElement(input_data, "type").text = "SMAP"
    SubElement(input_data, "path").text = "./dpr/L1B/SMAP/SMAP_L1B_TB_47185_D_20231201T212120_R18290_001.h5" 
    SubElement(input_data, "antenna_patterns_path").text = "./dpr/antenna_patterns"
    SubElement(input_data, "quality_control").text = "True"
    SubElement(input_data, "target_band").text = "L"
    SubElement(input_data, "split_fore_aft").text = "True"

    # GridParams
    grid_params = SubElement(root, "GridParams")
    SubElement(grid_params, "grid_type").text = "L1C"
    SubElement(grid_params, "grid_definition").text = "EASE2_G9km"
    SubElement(grid_params, "projection_definition").text = "G"
    # 800 900 3200 3260 south 1450 1460 1780 1800 north 650 660 1250 1260
    SubElement(grid_params, "reduced_grid_inds").text = ""

    # ReGridderParams
    regridder_params = SubElement(root, "ReGridderParams")
    SubElement(regridder_params, "regridding_algorithm").text = "IDS"
    SubElement(regridder_params, "search_radius").text = "18"
    SubElement(regridder_params, "max_neighbours").text = "20"
    SubElement(regridder_params, "variables_to_regrid").text = "bt_h bt_v"
    SubElement(regridder_params, "source_antenna_method").text = "gaussian"
    SubElement(regridder_params, "target_antenna_method").text = "real"
    SubElement(regridder_params, "polarisation_method").text = "scalar"
    SubElement(regridder_params, "source_antenna_threshold").text = "0.5"
    SubElement(regridder_params, "target_antenna_threshold").text = "0.5"
    SubElement(regridder_params, "max_theta_antenna_patterns").text = "40."
    SubElement(regridder_params, "MRF_grid_definition").text = "EASE2_G3km"
    SubElement(regridder_params, "MRF_projection_definition").text = "G"
    SubElement(regridder_params, "source_gaussian_params").text = "100000 100000"
    SubElement(regridder_params, "target_gaussian_params").text = "100000 100000"
    SubElement(regridder_params, "boresight_shift").text = "True"
    # For rSIR and BG  
    SubElement(regridder_params, "rsir_iteration").text = "15"
    SubElement(regridder_params, "bg_smoothing").text = "1"


    #  <rsir_iteration>15</rsir_iteration>
    #  <bg_smoothing>1</bg_smoothing>

    # (Temporary) Config file will be created inside: 
    # valid_config_file = '/tmp/tmpcpc5g83y.xml'

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    tree = ElementTree(root)
    tree.write(temp_file.name)
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name) 


@pytest.fixture
def invalid_config_file():
    """
    Creates an invalid configuration XML file for testing.
    """
    root = Element("config")
    # Missing essential sections and attributes for validation tests
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    tree = ElementTree(root)
    tree.write(temp_file.name)
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)


def test_valid_config_file(valid_config_file):
    """
    Test that a valid config file initializes the class without errors.
    """

    # Passing in the dynamically created config file via valid_config_file method 
    config = ConfigFile(valid_config_file)
    assert config.output_path.name == "test_output"
    # Checking the name for logger_config.json 
    # (i.e., it should be logger_config.json == logger_config.json)
    assert config.logpar_config.name == "logger_config.json"
    assert config.input_data_type == "SMAP"
    assert config.grid_type == "L1C"
    assert config.regridding_algorithm == "IDS" 






