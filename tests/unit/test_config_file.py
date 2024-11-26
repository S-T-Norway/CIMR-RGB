import os 
import pathlib as pb 
from xml.etree.ElementTree import Element, SubElement, ElementTree 
import tempfile 
import itertools as it  

import numpy as np 
import pytest 

from cimr_rgb.config_file import ConfigFile   
from cimr_rgb.grid_generator import GRIDS


# ---------------------------
# From geeksforgeeks 
# https://www.geeksforgeeks.org/pytest-tutorial-testing-python-application-using-pytest/
# 
# How to use Fixtures at Large Scale?

# When dealing with extensive test suites, it's crucial to manage fixtures efficiently:

#  Use "conftest.py": Place common fixtures in a "conftest.py" file at the root of your test directory, making them accessible across multiple test files.
#  Fixture Scoping: Apply proper scoping ('function', 'class', 'module', 'session') to optimize setup and teardown operations.
#  Fixture Parametrization: Reuse fixtures with different inputs using params in the fixture decorator.

# When to Avoid Fixtures?

# In Pytest, while knowing about the importance of creating fixtures it also crucial to know that where to avoid fixtures for the smooth testing.

#  For overly simple setups: where the setup is just one or two lines of code.
#  When they reduce test readability: if the fixture is used only once or its purpose is not clear.
#  If they introduce unnecessary dependencies between tests, reducing the ability to run tests in isolation.

# ---------------------------


#def test_read_config():
#
#    # We expecting an Assertion error: 
#    with pytest.raises(AssertionError): 
#        config_file = pb.Path("../configs/rgb_config.xml").resolve()  
#        result = config_file #ConfigFile.read_config(config_file) 
#        assert result == "/home/eva-v3/Desktop/ubuntu-22.04/cimr/configs/rgb_config.xml"  

# --------------------
# Units 
# --------------------

def test_validate_input_data_type():
    """
    Tests the `validate_input_data_type` method of the `ConfigFile` class.

    This test checks the method's behavior for both valid and invalid input data types
    by dynamically modifying the `<type>` element in a mock XML configuration.

    The test performs the following:
    
    1. Valid Inputs:
       - Defines a list of valid input data types: ['AMSR2', 'SMAP', 'CIMR'].
       - For each valid type, sets the text of the `<type>` element and asserts that
         the method returns the correct value without raising any errors.

    2. Invalid Inputs:
       - Defines a list of invalid input data types: ['AMSR25', 'SMap', 'cimr', 'bac45'].
       - For each invalid type, sets the text of the `<type>` element and asserts that
         the method raises a `ValueError`.

    The test ensures the `validate_input_data_type` method correctly identifies and validates
    input data types, handling errors as expected for invalid values.
    """

    # Creating a specific set of parameters to emulate input from configuration file  
    root = Element("config")
    input_data = SubElement(root, "InputData")
    type_element = SubElement(input_data, "type")

    # Expected to pass 
    valid_input = ['AMSR2', 'SMAP', 'CIMR']
    for name in valid_input: 

        type_element.text = name 

        assert ConfigFile.validate_input_data_type(
            config_object = root,
            input_data_type='InputData/type'
        ) == name 

    # Expected to fail 
    # Testing for invalid inputs --- the code should produce ValueErrors in all the cases 
    invalid_input = ['AMSR25', 'SMap', 'cimr', "bac45"]
    for name in invalid_input: 

        type_element.text = name 

        with pytest.raises(ValueError):  # Assuming invalid input raises ValueError
            ConfigFile.validate_input_data_type(
                config_object   = root,
                input_data_type ='InputData/type'
            )

# TODO: Change the code to dynamically infer the correct data path (now it is hardcoded)
def test_validate_input_data_path():

    # Creating a specific set of parameters to emulate input from configuration file  
    root = Element("config")
    input_data = SubElement(root, "InputData")
    type_element = SubElement(input_data, "path")

    
    type_element.text = pb.Path("./dpr/L1B/SMAP/SMAP_L1B_TB_47185_D_20231201T212120_R18290_001.h5").resolve() 

    assert ConfigFile.validate_input_data_path(
        config_object    = root,
        input_data_path  = 'InputData/path'
    ) == type_element.text 

    # TODO: This results in an error, needs investigation 
    type_element.text = pb.Path("").resolve()  
    with pytest.raises(FileNotFoundError):  # Assuming invalid input raises ValueError
        ConfigFile.validate_input_data_path(
            config_object    = root,
            input_data_path  = 'InputData/path'
        )


# TODO: I need to understand the logic better  
#def test_validate_input_antenna_patterns_path():
#    ... 
    #self.antenna_patterns_path = self.validate_input_antenna_patterns_path(
    #    config_object        =  config_object,
    #    antenna_patterns_path = 'InputData/antenna_patterns_path',
    #    input_data_type = self.input_data_type
    #)



def test_validate_grid_type():

    # Creating a specific set of parameters to emulate input from configuration file  
    root = Element("config")
    input_data  = SubElement(root, "InputData")
    input_type  = SubElement(input_data, "type")
    grid_params = SubElement(root, "GridParams")
    grid_type   = SubElement(grid_params, "grid_type")

    # Expected to pass 
    valid_input = {'AMSR2': ['L1R', 'L1C'], 'SMAP': ['L1C'], 'CIMR': ['L1R', 'L1C']} 
    # dtype -- data_type 
    # gtype -- grid_type
    for dtype, gtypes in valid_input.items(): 

        input_type.text = dtype   

        for gtype in gtypes:  

            grid_type.text  = gtype   
            
            assert ConfigFile.validate_grid_type(
               config_object = root ,
               grid_type = 'GridParams/grid_type',
               input_data_type = dtype
            ) == gtype 

    # Expected to fail 
    invalid_input = {'AMSR2': ['L1r', 'l1C'], 'SMAP': ['L1R'], 'CIMR': ['abc']} 
    for dtype, gtypes in invalid_input.items(): 

        input_type.text = dtype   

        for gtype in gtypes:  

            grid_type.text  = gtype   

            with pytest.raises(ValueError):  # Assuming invalid input raises ValueError
            
                ConfigFile.validate_grid_type(
                   config_object = root ,
                   grid_type = 'GridParams/grid_type',
                   input_data_type = dtype
                ) 


# TODO: 
# - This method needs to be checked, because there is a splitting for grid_type for CIMR but not for AMSR2 
# - Add the method inside of it as a standalone thing 
# - Fix error in the code due to which the method fails to crash (and so does 
#   not pass the target bands farther into pipeline) 
def test_validate_target_band():

    # Creating a specific set of parameters to emulate input from configuration file  
    root        = Element("config")
    input_data  = SubElement(root, "InputData")
    input_type  = SubElement(input_data, "type")
    target_band = SubElement(input_data, "target_band")
    grid_params = SubElement(root, "GridParams")
    grid_type   = SubElement(grid_params, "grid_type")

    # Function to generate all possible combinations as lists 
    def generate_all_combinations(bands):
        # Chain combinations of all possible lengths (1 to len(bands))
        all_combinations = it.chain.from_iterable(it.combinations(bands, r) for r in range(1, len(bands) + 1))
        # Convert each combination to a set and return as a list
        #return [set(comb) for comb in all_combinations]
        return [list(comb) for comb in all_combinations]

    # Expected to pass 
    cimr_bands  = ['L', 'C', 'X', 'KA', 'KU']
    # Generate the list of all available combinations for valid bands 
    cimr_bands_sets = generate_all_combinations(cimr_bands)
    amsr2_bands = ['6', '7', '10', '18', '23', '36', '89a', '89b'] 
    amsr2_bands_sets = generate_all_combinations(amsr2_bands)

    all_bands_sets = {'CIMR': cimr_bands_sets, 
                      'AMSR2': amsr2_bands_sets} 


    valid_input = {'AMSR2': ['L1R', 'L1C'], 
                   'CIMR':  ['L1R', 'L1C']} 

    for dtype, gtypes in valid_input.items(): 

        input_type.text = dtype 
        
        for gtype in gtypes: 

            grid_type.text = gtype  

            #print(dtype, gtype)
            
            for bands in all_bands_sets[dtype]: 
                # Emulating the input from a parameter file 
                target_band.text = " ".join(bands)
                #print(target_band, target_band.text)
                # This line generates a list of values back 
                #print(root.find("InputData/target_band").text.split()) 

                # We basically check for what we input vs what we received, while the code did not crush 
                assert ConfigFile.validate_target_band(
                    config_object    = root,
                    target_band      = 'InputData/target_band',
                    input_data_type  = dtype,
                    grid_type        = gtype  
                ) == bands  

    # Expected to fail 
    cimr_bands  = ['l', 'c', 'x1', 'kA', 'KU']
    # Generate the list of all available combinations for valid bands 
    cimr_bands_sets = generate_all_combinations(cimr_bands)
    amsr2_bands = ['12', '89', 'Ab'] 
    amsr2_bands_sets = generate_all_combinations(amsr2_bands)

    all_bands_sets = {'cIMr': cimr_bands_sets, 
                      'amsr2': amsr2_bands_sets} 


    invalid_input = {'amsr2': ['l1r', 'l1C'], 
                     'cIMr':  ['L1r', 'L1c']} 

    # This code passes through, although it really should not and it seems 
    # that validate_input_data_type does not check for correct data data_type 
    # (AMSR2, SMAP etc.). The same is valid for grid_type variable, since there 
    # is no check for it. 
    for dtype, gtypes in invalid_input.items(): 

        input_type.text = dtype 
        
        for gtype in gtypes: 

            grid_type.text = gtype  

            #print(dtype, gtype)
            
            for bands in all_bands_sets[dtype]: 
                # Emulating the input from a parameter file 
                target_band.text = " ".join(bands)

                with pytest.raises(ValueError):  # Assuming invalid input raises ValueError
                    ConfigFile.validate_target_band(
                        config_object    = root,
                        target_band      = 'InputData/target_band',
                        input_data_type  = dtype,
                        grid_type        = gtype  
                    )   
                
                

# TODO: write this one, should take only one parameter at a time?  
# def test_validate_source_band():

#     # Creating a specific set of parameters to emulate input from configuration file  
#     root        = Element("config")
#     input_data  = SubElement(root, "InputData")
#     input_type  = SubElement(input_data, "type")
#     source_band = SubElement(input_data, "source_band")
#     grid_params = SubElement(root, "GridParams")
#     grid_type   = SubElement(grid_params, "grid_type")

#     source_band = ConfigFile.validate_source_band(
#         config_object= root,
#         source_band  = 'InputData/source_band',
#         input_data_type = input_data_type
#     )






def test_validate_grid_definition():
    
    # Creating a specific set of parameters to emulate input from configuration file  
    root        = Element("config")
    grid_params = SubElement(root, "GridParams")
    grid_def    = SubElement(grid_params, "grid_definition")


    valid_input = ['EASE2_G9km', 'EASE2_N9km', 'EASE2_S9km',
                   'EASE2_G36km', 'EASE2_N36km', 'EASE2_S36km',
                   'STEREO_N25km', 'STEREO_S25km', 'STEREO_N6.25km',
                   'STEREO_N12.5km', 'STEREO_S6.25km', 'STEREO_S12.5km',
                   'STEREO_S25km', 'MERC_G25km', 'MERC_G12.5km', 'MERC_G6.25km']
    

    
    # Expected to pass  
    for name in valid_input: 
        grid_def.text = name 
        assert ConfigFile.validate_grid_definition(
            config_object    = root,
            grid_definition  = 'GridParams/grid_definition'
        ) == name  

    # Expected to fail 
    invalid_input = ['EASE', 'EASE2', 'eaSE2_S9km', "EASE2_S9", "MERC_G6"]

    for name in invalid_input: 
        grid_def.text = name 
        with pytest.raises(ValueError):  # Assuming invalid input raises ValueError
            ConfigFile.validate_grid_definition(
                config_object    = root,
                grid_definition  = 'GridParams/grid_definition'
            ) 
                



# TODO: Need to understand logic better 
def test_validate_projection_definition():

    # Creating a specific set of parameters to emulate input from configuration file  
    root        = Element("config")
    grid_params = SubElement(root, "GridParams")
    grid_def    = SubElement(grid_params, "grid_definition")
    proj_def    = SubElement(grid_params, "projection_definition")


    # Expected to pass  
    grid_def_valid_input = {"EASE2": 
                            ['EASE2_G9km', 'EASE2_N9km', 'EASE2_S9km',
                             'EASE2_G36km', 'EASE2_N36km', 'EASE2_S36km'],
                            "STEREO": 
                            ['STEREO_N25km', 'STEREO_S25km', 'STEREO_N6.25km',
                             'STEREO_N12.5km', 'STEREO_S6.25km', 'STEREO_S12.5km',
                             'STEREO_S25km'], 
                            "MERC": 
                            ['MERC_G25km', 'MERC_G12.5km', 'MERC_G6.25km']}

    proj_def_valid_input = {"EASE2": ["G", "N", "S"],  "STEREO": ["PS_N", "PS_S"], "MERC": ["MERC_G"]}

    for def_range, def_names in grid_def_valid_input.items(): 

        for def_name in def_names: 
            grid_def.text = def_name  
            #print(grid_def.text)

            for proj_name in proj_def_valid_input[def_range]: 

                #for proj_name in proj_names: 
                proj_def.text = proj_name 

                assert ConfigFile.validate_projection_definition(
                    config_object    = root,
                    grid_definition  = def_name,
                    projection_definition  = 'GridParams/projection_definition'
                ) == proj_name # in proj_def_valid_input #["G", "N", "PS_N", "PS_S", "MERC_G"] 

    # Expected to fail 
    # [Note]: grid_definition is not checked in this method and perhaps it should be? 
    #grid_def_invalid_input = {"EASE2": 
    #                        ['EASE2_G9km', 'EASE2_N9km', 'EASE2_S9km',
    #                         'EASE2_G36km', 'EASE2_N36km', 'EASE2_S36km'],
    #                        "STEREO": 
    #                        ['STEREO_N25km', 'STEREO_S25km', 'STEREO_N6.25km',
    #                         'STEREO_N12.5km', 'STEREO_S6.25km', 'STEREO_S12.5km',
    #                         'STEREO_S25km'], 
    #                        "MERC": 
    #                        ['MERC_G25km', 'MERC_G12.5km', 'MERC_G6.25km']}

    # [Note]: The empty string does not fail in this case 
    proj_def_invalid_input = {"EASE2": ["g", "n", " "],  "STEREO": ["pS_N", "Ps_s"], "MERC": ["abcd_G", ""]}
    for def_range, def_names in grid_def_valid_input.items(): 

        for def_name in def_names: 
            grid_def.text = def_name  

            for proj_name in proj_def_invalid_input[def_range]: 

                #for proj_name in proj_names: 
                proj_def.text = proj_name 

                #print(grid_def.text, proj_def.text)

                with pytest.raises(ValueError):  # Assuming invalid input raises ValueError
                    ConfigFile.validate_projection_definition(
                        config_object    = root,
                        grid_definition  = def_name,
                        projection_definition  = 'GridParams/projection_definition'
                    )


def test_validate_regridding_algorithm():
    # Creating a specific set of parameters to emulate input from configuration file  
    root        = Element("config")
    regridder_params = SubElement(root, "ReGridderParams")
    regridding_algo = SubElement(regridder_params, "regridding_algorithm")

    # Expected to pass 
    valid_input = ['NN', 'DIB', 'IDS', 'BG', 'RSIR', 'LW', 'CG']

    for algo in valid_input: 
        regridding_algo.text = algo 
        assert ConfigFile.validate_regridding_algorithm(
            config_object          = root,
            regridding_algorithm   = 'ReGridderParams/regridding_algorithm'
        ) == algo 

    # Expected to fail 
    invalid_input = ['nn', 'DiB', 'ids', '', 'rSIR', 'lW', 'Cg']
    for algo in invalid_input: 
        regridding_algo.text = algo 
        with pytest.raises(ValueError):  # Assuming invalid input raises ValueError
            ConfigFile.validate_regridding_algorithm(
                config_object          = root,
                regridding_algorithm   = 'ReGridderParams/regridding_algorithm'
            ) 


 # def test_validate_split_fore_aft():

 #     root        = Element("config")
 #     input_data  = SubElement(root, "InputData")
 #     input_type  = SubElement(input_data, "type")
 #     split_data  = SubElement(input_data, "split_fore_aft")
#     #regridder_params = SubElement(root, "ReGridderParams")
#     #regridding_algo = SubElement(regridder_params, "regridding_algorithm")
#     #proj_def    = SubElement(regridder_params, "projection_definition")
#          
#     # Expected to pass 
#     # AMSR2 is False by default, so the empty string or any other input is valid  
#     valid_input = {"CIMR": ['True', 'False'], "AMSR2": ["fAlse", "true", ""], "SMAP": ['False', 'True']}

#     for instrument, bool_vals in valid_input.items(): 

#         for def_name in def_names: 
#             grid_def.text = def_name  

#             for proj_name in proj_def_valid_input[def_range]: 

#                 #for proj_name in proj_names: 
#                 proj_def.text = proj_name 

#             assert ConfigFile.validate_split_fore_aft(
#                 config_object   = root,
#                 split_fore_aft  = 'InputData/split_fore_aft',
#                 input_data_type = input_type.text 
#             )


    #    if input_data_type == 'AMSR2':
    #        return False
    #    valid_input = ['True', 'False']
    #    if config_object.find(split_fore_aft).text in valid_input:
    #        if config_object.find(split_fore_aft).text == 'True':
    #            return True
    #        else:
    #            return False
    #    raise ValueError(
    #        f"Invalid split fore aft. Check Configuration File."
    #        f" Valid split fore aft are: {valid_input}"
    #    )



# @pytest.mark.parametrize(
#     "split_value, input_data_type, expected",
#     [
#         ('True', 'SomeType', True),    # Valid True case
#         ('False', 'SomeType', False), # Valid False case
#         ('True', 'AMSR2', False),     # Special case for AMSR2
#         ('InvalidValue', 'SomeType', pytest.raises(ValueError)), # Invalid case
#     ],
# )
# def test_validate_split_fore_aft(split_value, input_data_type, expected):
#     # Arrange
#     config_object = Element("config")
#     input_data = SubElement(config_object, "InputData")
#     split_data = SubElement(input_data, "split_fore_aft")
#     split_data.text = split_value

#     # Act & Assert
#     if isinstance(expected, pytest.raises):
#         with expected:  # Handle exceptions for invalid cases
#             ConfigFile.validate_split_fore_aft(
#                 config_object=config_object,
#                 split_fore_aft='InputData/split_fore_aft',
#                 input_data_type=input_data_type,
#             )
#     else:
#         result = ConfigFile.validate_split_fore_aft(
#             config_object=config_object,
#             split_fore_aft='InputData/split_fore_aft',
#             input_data_type=input_data_type,
#         )
#         assert result == expected 



# TODO: Rewrite the original method because it seems to pass even 
#       when neither of acceptable instruments are called 
@pytest.mark.parametrize(
    "split_value, input_data_type, expected, expect_exception",
    [
        # Expected to pass 
        ('True', 'SMAP', True, False),   # Valid True case
        ('False', 'CIMR', False, False), # Valid False case
        ('True', 'AMSR2', False, False),    # Special case for AMSR2
        ('False', 'AMSR2', False, False),    # Special case for AMSR2
        ('', 'AMSR2', False, False),    # Special case for AMSR2
        # Expected to fail 
        ('InvalidValue', 'InputData', None, True), # Invalid case
        ('False', 'SomeType', None, True), # Valid False case
    ],
)
def test_validate_split_fore_aft(split_value, input_data_type, expected, expect_exception):
    # Arrange
    config_object = Element("config")
    input_data = SubElement(config_object, "InputData")
    split_data = SubElement(input_data, "split_fore_aft")
    split_data.text = split_value

    # Expected to fail 
    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError, match="Invalid split fore aft. Check Configuration File."):
            ConfigFile.validate_split_fore_aft(
                config_object=config_object,
                split_fore_aft='InputData/split_fore_aft',
                input_data_type=input_data_type,
            )
    # Expected to pass 
    else:
        # Act: No exception expected
        result = ConfigFile.validate_split_fore_aft(
            config_object=config_object,
            split_fore_aft='InputData/split_fore_aft',
            input_data_type=input_data_type,
        )
        # Assert: Compare result to expected value
        assert result == expected



    #<ReGridderParams>
    #  <regridding_algorithm>IDS</regridding_algorithm>
    #  <search_radius>18</search_radius>
    #  <max_neighbours>20</max_neighbours>
    #  <variables_to_regrid></variables_to_regrid>
    #  <source_antenna_method>gaussian</source_antenna_method>
    #  <target_antenna_method>real</target_antenna_method>
    #  <polarisation_method>scalar</polarisation_method>
    #  <source_antenna_threshold>0.5</source_antenna_threshold>
    #  <target_antenna_threshold>0.5</target_antenna_threshold>
    #  <MRF_grid_definition>EASE2_G3km</MRF_grid_definition>
    #  <MRF_projection_definition>G</MRF_projection_definition>
    #  <source_gaussian_params>100000 100000</source_gaussian_params>
    #  <target_gaussian_params>100000 100000</target_gaussian_params>
    #  <boresight_shift>True</boresight_shift>
    #  <rsir_iteration>15</rsir_iteration>
    #  <bg_smoothing>1</bg_smoothing>
    #</ReGridderParams>


@pytest.mark.parametrize(
    "save_to_disk_value, expected, expect_exception",
    [
        ('True', True, False),   # Valid True case
        ('False', False, False), # Valid False case
        ('InvalidValue', None, True), # Invalid case
        ('', None, True),  # Empty string, invalid case
    ],
)
def test_validate_save_to_disk(save_to_disk_value, expected, expect_exception):
    # Arrange
    config_object = Element("config")
    output_data = SubElement(config_object, "OutputData")
    save_to_disk = SubElement(output_data, "save_to_disk")
    save_to_disk.text = save_to_disk_value

    # Pytest need to match error message exactly in this case for it to work 
    error_message = ("Invalid `save_to_disk`. Check Configuration File. ")# + \
                    # "`save_to_disk` must be either True or False")

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError, match=error_message): 
        #"Invalid `save_to_disk`. Check Configuration File. save_to_disk must be either True or False"
            ConfigFile.validate_save_to_disk(
                config_object=config_object,
                save_to_disk='OutputData/save_to_disk',
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_save_to_disk(
            config_object=config_object,
            save_to_disk='OutputData/save_to_disk',
        )
        # Assert: Compare result to expected value
        assert result == expected 



# @pytest.mark.parametrize(
#     "search_radius_value, grid_definition, grid_type, expected, expect_exception",
#     [
#         # Valid Cases
#         ('5.0', 'EASE2_G1km', 'L1C', 5000.0, False),  # Defined search_radius (5.0 km)
#         # (None, 'EASE2_G1km', 'L1C', 'CIMR', np.sqrt(2 * (1000.9 / 2) ** 2), False),  # L1C with grid_definition
#         # (None, 'EASE2_G3km', 'L1C', 'SMAP', np.sqrt(2 * (3002.69 / 2) ** 2), False),  # L1C with SMAP
#         # (None, 'EASE2_N3km', 'L1C', 'AMSR2', np.sqrt(2 * (3000 / 2) ** 2), False),  # L1C with AMSR2
#         # (None, 'EASE2_G1km', 'L1R', 'CIMR', 73000 / 2, False),  # L1R with CIMR
#         # (None, 'EASE2_G1km', 'L1R', 'AMSR2', 62000 / 2, False),  # L1R with AMSR2

#         # # Invalid Cases
#         # (None, 'EASE2_G1km', 'L1C', 'INVALID', None, True),  # Invalid input_data_type
#         # (None, 'EASE2_G1km', 'L1C', 'cimr', None, True),  # Case-sensitive check (lowercase)
#         # (None, 'EASE2_G1km', 'L1C', 'Smap', None, True),  # Case-sensitive check (mixed case)
#     ],
# )
# def test_validate_search_radius(search_radius_value, grid_definition, grid_type, input_data_type, expected, expect_exception):
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     search_radius = SubElement(regridder_params, "search_radius")
#     if search_radius_value is not None:
#         search_radius.text = search_radius_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match=f"Invalid `input_data_type`: {input_data_type}"):
#             ConfigFile.validate_search_radius(
#                 config_object=config_object,
#                 search_radius='ReGridderParams/search_radius',
#                 grid_definition=grid_definition,
#                 grid_type=grid_type,
#                 #input_data_type=input_data_type,
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_search_radius(
#             config_object=config_object,
#             search_radius='ReGridderParams/search_radius',
#             grid_definition=grid_definition,
#             grid_type=grid_type,
#             #input_data_type=input_data_type,
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# TODO: Fix SMAP 
@pytest.mark.parametrize(
    "search_radius_value, grid_definition, grid_type, input_data_type, expected, expect_exception",
    [
        # Valid Cases
        ('5.0', 'EASE2_G1km', 'L1C', 'CIMR', 5000.0, False),  # Valid numeric value
        (None, 'EASE2_G1km', 'L1C', 'CIMR', np.sqrt(2 * (GRIDS['EASE2_G1km']['res'] / 2) ** 2), False),
        ("", 'EASE2_G36km', 'L1R', 'CIMR', 73000 / 2, False), 
        (" ", 'EASE2_G1km', 'L1R', 'AMSR2', 62000 / 2, False), 
        (" ", 'STEREO_N25km', 'L1C', 'CIMR', np.sqrt(2 * (GRIDS['STEREO_N25km']['res'] / 2) ** 2), False), 
        ("18", 'MERC_G12.km', 'L1C', 'SMAP', 18000, False), 

        # Invalid Cases
        ("18", 'MERC_G12.km', 'L1R', 'SMAP', 18000, True), # TODO: SMAP does not have L1R  
        ('abc', 'EASE2_G1km', 'L1C', 'CIMR', None, True),  # Non-numeric string
        ('123abc', 'EASE2_G1km', 'L1C', 'CIMR', None, True),  # Partially numeric
    ],
)
def test_validate_search_radius_numeric_check(search_radius_value, grid_definition, grid_type, input_data_type, expected, expect_exception):
    # Arrange
    config_object = Element("config")
    regridder_params = SubElement(config_object, "ReGridderParams")
    search_radius = SubElement(regridder_params, "search_radius")
    if search_radius_value is not None:
        search_radius.text = search_radius_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError, match="Invalid `search_radius`"):
            ConfigFile.validate_search_radius(
                config_object=config_object,
                search_radius='ReGridderParams/search_radius',
                grid_definition=grid_definition,
                grid_type=grid_type,
               input_data_type = input_data_type,
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_search_radius(
            config_object=config_object,
            search_radius='ReGridderParams/search_radius',
            grid_definition=grid_definition,
            grid_type=grid_type,
           input_data_type=input_data_type,
        )
        # Assert: Compare result to expected value
        assert result == expected




# # TODO: Write the test for this method  
# def test_get_scan_geometry():
#     ... 


@pytest.mark.parametrize(
    "input_data_type, variables_to_regrid_value, expected, expect_exception",
    [
        # Valid Cases for SMAP
        ('SMAP', "bt_h bt_v longitude latitude", ['bt_h', 'bt_v', 'longitude', 'latitude'], False),
        ('SMAP', None, ['bt_h', 'bt_v', 'bt_3', 'bt_4', 'processing_scan_angle', 'longitude', 'latitude'], False),  # Default vars
        ('SMAP', "bt_h bt_v invalid_var", None, True),  # Invalid variable in SMAP

        # Valid Cases for AMSR2
        ('AMSR2', "bt_h bt_v longitude latitude", ['bt_h', 'bt_v', 'longitude', 'latitude'], False),
        ('AMSR2', None, ['bt_h', 'bt_v'], False),  # No default_vars for AMSR2, so None should be an error
        ('AMSR2', "invalid_var", None, True),  # Invalid variable in AMSR2

        # Valid Cases for CIMR
        ('CIMR', "bt_h bt_v longitude latitude oza", ['bt_h', 'bt_v', 'longitude', 'latitude', 'oza'], False),
        ('CIMR', None, ['bt_h', 'bt_v', 'bt_3', 'bt_4', 'processing_scan_angle', 'longitude', 'latitude'], False),  # Default vars
        ('CIMR', "invalid_var", None, True),  # Invalid variable in CIMR

        # Invalid input_data_type
        ('INVALID_TYPE', None, None, True),  # Unsupported input_data_type
    ],
)
def test_validate_variables_to_regrid(input_data_type, variables_to_regrid_value, 
                                      expected, expect_exception):
    # Arrange
    config_object = Element("config")
    regridder_params = SubElement(config_object, "ReGridderParams")
    variables_to_regrid = SubElement(regridder_params, "variables_to_regrid")
    if variables_to_regrid_value is not None:
        variables_to_regrid.text = variables_to_regrid_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):
            ConfigFile.validate_variables_to_regrid(
                config_object=config_object,
                input_data_type=input_data_type,
                variables_to_regrid='ReGridderParams/variables_to_regrid',
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_variables_to_regrid(
            config_object=config_object,
            input_data_type=input_data_type,
            variables_to_regrid='ReGridderParams/variables_to_regrid',
        )
        # Assert: Compare result to expected value
        assert result == expected 


# TODO: The test passes. but I think more needs to be done for checking the configuration 
@pytest.mark.parametrize(
    "regridding_algorithm, max_neighbours_value, expected, expect_exception",
    [
        # Case: Regridding algorithm is NN
        ('NN', None, 1, False),  # Always return 1 regardless of max_neighbours

        # Case: Valid max_neighbours value
        ('Other', "500", 500, False),  # Valid integer string
        ('Other', "1000", 1000, False),  # Valid integer string

        # Case: max_neighbours not defined
        ('Other', None, 1000, False),  # Defaults to 1000

        # Case: Invalid max_neighbours value
        ('Other', "invalid", None, True),  # Non-integer string
        ('Other', "", None, True),  # Empty string
        ('Other', " ", None, True),  # Space-only string
    ],
)
def test_validate_max_neighbours(regridding_algorithm, max_neighbours_value, expected, expect_exception):
    # Arrange
    config_object = Element("config")
    regridder_params = SubElement(config_object, "ReGridderParams")
    max_neighbours = SubElement(regridder_params, "max_neighbours")
    if max_neighbours_value is not None:
        max_neighbours.text = max_neighbours_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):
            ConfigFile.validate_max_neighbours(
                config_object=config_object,
                max_neighbours='ReGridderParams/max_neighbours',
                regridding_algorithm=regridding_algorithm,
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_max_neighbours(
            config_object=config_object,
            max_neighbours='ReGridderParams/max_neighbours',
            regridding_algorithm=regridding_algorithm,
        )
        # Assert: Compare result to expected value
        assert result == expected 



# TODO: Check the docstring 
@pytest.mark.parametrize(
    "source_antenna_method_value, expected, expect_exception",
    [
        # Valid cases
        ('gaussian', 'gaussian', False),
        ('real', 'real', False),
        ('gaussian_projected', 'gaussian_projected', False),

        # Default case
        (None, 'real', False),  # Defaults to 'real' if value is None
        ('', 'real', False),  # Empty string
        (' ', 'real', False),  # Space-only string

        # Invalid cases
        ('invalid_value', None, True),  # Invalid string
        ('GAUSSIAN', None, True),  # Invalid string
        ('Gaussian', None, True),  # Invalid string
    ],
)
def test_validate_source_antenna_method(source_antenna_method_value, expected, expect_exception):
    """
    Test the `validate_source_antenna_method` method of the ConfigFile class.

    This test covers the following scenarios:
    - Valid `source_antenna_method` values (e.g., 'gaussian', 'real', 'gaussian_projected').
    - Missing `source_antenna_method` (defaults to 'real').
    - Invalid `source_antenna_method` values (e.g., 'invalid_value', empty strings).

    Parameters:
    - source_antenna_method_value: The value of the source_antenna_method in the configuration file.
    - expected: The expected output or None if an exception is expected.
    - expect_exception: Boolean indicating if a ValueError is expected.

    Valid cases should return the correct value.
    Missing or invalid cases should raise a ValueError or default to 'real' if appropriate.
    """

    # Arrange
    config_object = Element("config")
    regridder_params = SubElement(config_object, "ReGridderParams")
    source_antenna = SubElement(regridder_params, "source_antenna_method")
    if source_antenna_method_value is not None:
        source_antenna.text = source_antenna_method_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError, match="Invalid antenna method. Check Configuration File."):
            ConfigFile.validate_source_antenna_method(
                config_object=config_object,
                source_antenna_method='ReGridderParams/source_antenna_method',
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_source_antenna_method(
            config_object=config_object,
            source_antenna_method='ReGridderParams/source_antenna_method',
        )
        # Assert: Compare result to expected value
        assert result == expected





# TODO: Check the docstring 
@pytest.mark.parametrize(
    "target_antenna_method_value, expected, expect_exception",
    [
        # Valid cases
        ('gaussian', 'gaussian', False),
        ('real', 'real', False),
        ('gaussian_projected', 'gaussian_projected', False),

        # Default case
        (None, 'real', False),  # Defaults to 'real' if value is None
        ('', 'real', False),  # Empty string
        (' ', 'real', False),  # Space-only string

        # Invalid cases
        ('invalid_value', None, True),  # Invalid string
        ('GAUSSIAN', None, True),  # Invalid string
        ('Gaussian', None, True),  # Invalid string
    ],
)
def test_validate_target_antenna_method(target_antenna_method_value, expected, expect_exception):
    """
    Test the `validate_target_antenna_method` method of the ConfigFile class.

    This test validates:
    - Correct behavior for valid `target_antenna_method` values (e.g., 'gaussian', 'real', 'gaussian_projected').
    - Default value ('real') when `target_antenna_method` is missing or None.
    - Error handling for invalid `target_antenna_method` values.

    Parameters:
    - target_antenna_method_value: The value of the target_antenna_method in the configuration file.
    - expected: The expected output or None if an exception is expected.
    - expect_exception: Boolean indicating if a ValueError is expected.

    Valid cases return the appropriate value.
    Missing values default to 'real'.
    Invalid values raise a ValueError.
    """

    # Arrange
    config_object = Element("config")
    regridder_params = SubElement(config_object, "ReGridderParams")
    target_antenna = SubElement(regridder_params, "target_antenna_method")
    if target_antenna_method_value is not None:
        target_antenna.text = target_antenna_method_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError, match="Invalid antenna method. Check Configuration File."):
            ConfigFile.validate_target_antenna_method(
                config_object=config_object,
                target_antenna_method='ReGridderParams/target_antenna_method',
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_target_antenna_method(
            config_object=config_object,
            target_antenna_method='ReGridderParams/target_antenna_method',
        )
        # Assert: Compare result to expected value
        assert result == expected 



# TODO: Check the docstring 
@pytest.mark.parametrize(
    "source_antenna_threshold_value, expected, expect_exception",
    [
        # Valid cases
        ('9.5', 9.5, False),  # Float value
        ('10', 10.0, False),  # Integer value as string
        ('0', 0.0, False),  # Zero value
        ('-5', -5.0, False),  # Negative value

        # Default case
        (None, None, False),  # Returns None if value is not provided
        ('', None, False),  # Empty string
        (' ', None, False),  # Space-only string

        # Invalid cases
        ('invalid', None, True),  # Non-numeric string
        ('a9.5', 9.5, True),  # Typo in Float value
    ],
)
def test_validate_source_antenna_threshold(source_antenna_threshold_value, expected, expect_exception):
    """
    Test the `validate_source_antenna_threshold` method of the ConfigFile class.

    This test validates:
    - Correct behavior for valid numeric `source_antenna_threshold` values (float and integer).
    - Default behavior (returns None) when `source_antenna_threshold` is missing or None.
    - Error handling for invalid `source_antenna_threshold` values (non-numeric).

    Parameters:
    - source_antenna_threshold_value: The value of the source_antenna_threshold in the configuration file.
    - expected: The expected output or None if an exception is expected.
    - expect_exception: Boolean indicating if a ValueError is expected.

    Valid cases return the appropriate numeric value.
    Missing values return None.
    Invalid values raise a ValueError.
    """

    # Arrange
    config_object = Element("config")
    regridder_params = SubElement(config_object, "ReGridderParams")
    source_antenna_threshold = SubElement(regridder_params, "source_antenna_threshold")

    if source_antenna_threshold_value is not None:
        source_antenna_threshold.text = source_antenna_threshold_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError, match="Invalid antenna threshold"): #. Check Configuration File."):
            ConfigFile.validate_source_antenna_threshold(
                config_object=config_object,
                source_antenna_threshold='ReGridderParams/source_antenna_threshold',
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_source_antenna_threshold(
            config_object=config_object,
            source_antenna_threshold='ReGridderParams/source_antenna_threshold',
        )
        # Assert: Compare result to expected value
        assert result == expected 



# TODO: Check the docstring 
@pytest.mark.parametrize(
    "target_antenna_threshold_value, expected, expect_exception",
    [
        # Valid cases
        ('9.5', 9.5, False),  # Float value
        ('10', 10.0, False),  # Integer value as string
        ('0', 0.0, False),  # Zero value
        ('-5', -5.0, False),  # Negative value

        # Default case
        (None, 9.0, False),  # Defaults to 9.0 if value is None
        ('', 9.0, False),  # Empty string
        (' ', 9.0, False),  # Space-only string

        # Invalid cases
        ('invalid', None, True),  # Non-numeric string
        ('a9.5', 9.5, True),  # Typo in Float value
    ],
)
def test_validate_target_antenna_threshold(target_antenna_threshold_value, expected, expect_exception):
    """
    Test the `validate_target_antenna_threshold` method of the ConfigFile class.

    This test validates:
    - Correct behavior for valid numeric `target_antenna_threshold` values (float and integer).
    - Default behavior (returns 9.0) when `target_antenna_threshold` is missing or None.
    - Error handling for invalid `target_antenna_threshold` values (non-numeric).

    Parameters:
    - target_antenna_threshold_value: The value of the target_antenna_threshold in the configuration file.
    - expected: The expected output or None if an exception is expected.
    - expect_exception: Boolean indicating if a ValueError is expected.

    Valid cases return the appropriate numeric value.
    Missing values return the default of 9.0.
    Invalid values raise a ValueError.
    """
    # Arrange
    config_object = Element("config")
    regridder_params = SubElement(config_object, "ReGridderParams")
    target_antenna_threshold = SubElement(regridder_params, "target_antenna_threshold")
    if target_antenna_threshold_value is not None:
        target_antenna_threshold.text = target_antenna_threshold_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError, match="Invalid antenna threshold:"): #. Check Configuration File."):
            ConfigFile.validate_target_antenna_threshold(
                config_object=config_object,
                target_antenna_threshold='ReGridderParams/target_antenna_threshold',
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_target_antenna_threshold(
            config_object=config_object,
            target_antenna_threshold='ReGridderParams/target_antenna_threshold',
        )
        # Assert: Compare result to expected value
        assert result == expected 




# def test_validate_polarisation_method():
#     ... 








# def test_validate_boresight_shift():
#     ... 



# def test_validate_reduced_grid_inds():
#     ... 



# def test_validate_source_gaussian_params( ):
#     ... 



# def test_validate_target_gaussian_params():
#     ... 


# def test_validate_rsir_iteration():
#     ...


# def test_validate_max_number_iteration():
#     ... 



# def test_validate_relative_tolerance():
#     ...


# def test_validate_regularization_parameter():
#     ... 



# def test_validate_MRF_grid_definition():
#     ... 


# def test_validate_MRF_projection_definition():
#     ... 


# def test_validate_bg_smoothing():
#     ... 


# def test_validate_quality_control(): 
#     ... 


#def validate_quality_control(config_object, quality_control, input_data_type):
#    if input_data_type == 'AMSR2':
#        return False
#    elif input_data_type == 'CIMR':
#        return False
#    else:
#        valid_input = ['True', 'False']
#        if config_object.find(quality_control).text in valid_input:
#            if config_object.find(quality_control).text == 'True':
#                return True
#            else:
#                return False
#        raise ValueError(
#            f"Invalid split fore aft. Check Configuration File."
#            f" Valid split fore aft are: {valid_input}"
#        )











    #self.dpr_path        = path.join(path.dirname(getcwd()), 'dpr')

    #self.quality_control = self.validate_quality_control(
    #    config_object=config_object,
    #    quality_control='InputData/quality_control',
    #    input_data_type = self.input_data_type
    #)



    
    





