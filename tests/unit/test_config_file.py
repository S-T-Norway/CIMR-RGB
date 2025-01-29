import pathlib as pb

# from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
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


# def test_read_config():
#
#    # We expecting an Assertion error:
#    with pytest.raises(AssertionError):
#        config_file = pb.Path("../configs/rgb_config.xml").resolve()
#        result = config_file #ConfigFile.read_config(config_file)
#        assert result == "/home/eva-v3/Desktop/ubuntu-22.04/cimr/configs/rgb_config.xml"

# --------------------
# Units
# --------------------


@pytest.mark.parametrize(
    "input_value, expected_output, expect_exception",
    [
        # Expected to Pass
        ("AMSR2", "AMSR2", False),
        (" smap", "SMAP", False),
        ("CimR ", "CIMR", False),
        # Expected to Fail
        ("AMSR24", None, True),
        ("random string", None, True),
        (None, None, True),
    ],
)
def test_validate_input_data_type(input_value, expected_output, expect_exception):
    r"""
    Pytest unit test for the `validate_input_data_type` method.

    This test function validates the `input_data_type` parameter by checking its correctness
    and ensuring it aligns with expected values from the XML configuration. The test uses
    `pytest.mark.parametrize` to cover multiple test cases efficiently.

    Parameters
    ----------
    input_value : str or None
        The value extracted from the XML configuration. If None, it simulates a missing tag.
    expected_output : str or None
        The expected validated input data type if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Raises
    ------
    ValueError
        If `input_value` is not in the list of valid input data types.
    AttributeError
        If the required XML tag is missing or incorrectly specified.

    Notes
    -----
    - Uses `pytest.raises` to assert expected exceptions.
    - Employs parameterized testing for efficiency.
    """

    config_xml = (
        f"""<config><InputData><type>{input_value}</type></InputData></config>"""
    )
    config_object = ET.ElementTree(ET.fromstring(config_xml)).getroot()

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises((ValueError, AttributeError)):
            ConfigFile.validate_input_data_type(
                config_object=config_object, input_data_type="InputData/type"
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_input_data_type(
            config_object=config_object, input_data_type="InputData/type"
        )
        assert result == expected_output


# TODO: This code needs to be rethought or not implemented
# TODO: Change the code to dynamically infer the correct data path (now it is hardcoded)
# def test_validate_input_data_path():
#     # Creating a specific set of parameters to emulate input from configuration file
#     root = ET.Element("config")
#     input_data = ET.SubElement(root, "InputData")
#     type_element = ET.SubElement(input_data, "path")

#     type_element.text = pb.Path(
#         "./dpr/L1B/SMAP/SMAP_L1B_TB_47185_D_20231201T212120_R18290_001.h5"
#     ).resolve()

#     assert (
#         ConfigFile.validate_input_data_path(
#             config_object=root, input_data_path="InputData/path"
#         )
#         == type_element.text
#     )

#     # TODO: This results in an error, needs investigation
#     type_element.text = pb.Path("").resolve()
#     with pytest.raises(FileNotFoundError):  # Assuming invalid input raises ValueError
#         ConfigFile.validate_input_data_path(
#             config_object=root, input_data_path="InputData/path"
#         )


# # TODO: I need to understand the logic better
# # def test_validate_input_antenna_patterns_path():
# #    ...
# # self.antenna_patterns_path = self.validate_input_antenna_patterns_path(
# #    config_object        =  config_object,
# #    antenna_patterns_path = 'InputData/antenna_patterns_path',
# #    input_data_type = self.input_data_type
# # )


@pytest.mark.parametrize(
    "grid_type_value, input_data_type_value, expected_output, expect_exception",
    [
        # Expected to Pass
        ("L1R", "AMSR2", "L1R", False),
        ("L1C", "SMAP", "L1C", False),
        ("l1C", "CIMR", "L1C", False),
        # Expected to Fail
        ("L1R", "SMAP", None, True),
        ("random string", "CIMR", None, True),
        ("", None, None, True),
        (None, "SMAP", None, True),
    ],
)
def test_validate_grid_type(
    grid_type_value,
    input_data_type_value,
    expected_output,
    expect_exception,
):
    r"""
    Pytest unit test for the `validate_grid_type` method.

    This test function validates the `grid_type` parameter in the XML configuration
    by checking its correctness against expected values and ensuring it aligns
    with the associated input data type. The test utilizes `pytest.mark.parametrize`
    to evaluate multiple cases efficiently.

    Parameters
    ----------
    grid_type_value : str or None
        The value assigned to the grid type tag in the XML.
    input_data_type_value : str or None
        The associated input data type used for validation.
    expected_output : str or None
        The expected validated grid type if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Raises
    ------
    ValueError
        If `grid_type_value` is not valid for the given `input_data_type_value`.
    AttributeError
        If the required XML tag is missing or incorrectly specified.

    Notes
    -----
    - Uses `pytest.raises` to assert expected exceptions.
    - Employs parameterized testing to cover multiple cases.
    """

    config_xml = f"""<config><GridParams><grid_type>{grid_type_value}</grid_type></GridParams></config>"""
    config_object = ET.ElementTree(ET.fromstring(config_xml)).getroot()

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises((ValueError, AttributeError)):
            ConfigFile.validate_grid_type(
                config_object=config_object,
                grid_type="GridParams/grid_type",
                input_data_type=input_data_type_value,
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_grid_type(
            config_object=config_object,
            grid_type="GridParams/grid_type",
            input_data_type=input_data_type_value,
        )
        assert result == expected_output


# # TODO:
# # - This method needs to be checked, because there is a splitting for grid_type for CIMR but not for AMSR2
# # - Add the method inside of it as a standalone thing
# # - Fix error in the code due to which the method fails to crash (and so does
# #   not pass the target bands farther into pipeline)
# def test_validate_target_band():
#     # Creating a specific set of parameters to emulate input from configuration file
#     root = Element("config")
#     input_data = SubElement(root, "InputData")
#     input_type = SubElement(input_data, "type")
#     target_band = SubElement(input_data, "target_band")
#     grid_params = SubElement(root, "GridParams")
#     grid_type = SubElement(grid_params, "grid_type")

#     # Function to generate all possible combinations as lists
#     def generate_all_combinations(bands):
#         # Chain combinations of all possible lengths (1 to len(bands))
#         all_combinations = it.chain.from_iterable(
#             it.combinations(bands, r) for r in range(1, len(bands) + 1)
#         )
#         # Convert each combination to a set and return as a list
#         # return [set(comb) for comb in all_combinations]
#         return [list(comb) for comb in all_combinations]

#     # Expected to pass
#     cimr_bands = ["L", "C", "X", "KA", "KU"]
#     # Generate the list of all available combinations for valid bands
#     cimr_bands_sets = generate_all_combinations(cimr_bands)
#     amsr2_bands = ["6", "7", "10", "18", "23", "36", "89a", "89b"]
#     amsr2_bands_sets = generate_all_combinations(amsr2_bands)

#     all_bands_sets = {"CIMR": cimr_bands_sets, "AMSR2": amsr2_bands_sets}

#     valid_input = {"AMSR2": ["L1R", "L1C"], "CIMR": ["L1R", "L1C"]}

#     for dtype, gtypes in valid_input.items():
#         input_type.text = dtype

#         for gtype in gtypes:
#             grid_type.text = gtype

#             # print(dtype, gtype)

#             for bands in all_bands_sets[dtype]:
#                 # Emulating the input from a parameter file
#                 target_band.text = " ".join(bands)
#                 # print(target_band, target_band.text)
#                 # This line generates a list of values back
#                 # print(root.find("InputData/target_band").text.split())

#                 # We basically check for what we input vs what we received, while the code did not crush
#                 assert (
#                     ConfigFile.validate_target_band(
#                         config_object=root,
#                         target_band="InputData/target_band",
#                         input_data_type=dtype,
#                         grid_type=gtype,
#                     )
#                     == bands
#                 )

#     # Expected to fail
#     cimr_bands = ["l", "c", "x1", "kA", "KU"]
#     # Generate the list of all available combinations for valid bands
#     cimr_bands_sets = generate_all_combinations(cimr_bands)
#     amsr2_bands = ["12", "89", "Ab"]
#     amsr2_bands_sets = generate_all_combinations(amsr2_bands)

#     all_bands_sets = {"cIMr": cimr_bands_sets, "amsr2": amsr2_bands_sets}

#     invalid_input = {"amsr2": ["l1r", "l1C"], "cIMr": ["L1r", "L1c"]}

#     # This code passes through, although it really should not and it seems
#     # that validate_input_data_type does not check for correct data data_type
#     # (AMSR2, SMAP etc.). The same is valid for grid_type variable, since there
#     # is no check for it.
#     for dtype, gtypes in invalid_input.items():
#         input_type.text = dtype

#         for gtype in gtypes:
#             grid_type.text = gtype

#             # print(dtype, gtype)

#             for bands in all_bands_sets[dtype]:
#                 # Emulating the input from a parameter file
#                 target_band.text = " ".join(bands)

#                 with pytest.raises(
#                     ValueError
#                 ):  # Assuming invalid input raises ValueError
#                     ConfigFile.validate_target_band(
#                         config_object=root,
#                         target_band="InputData/target_band",
#                         input_data_type=dtype,
#                         grid_type=gtype,
#                     )


# # TODO: write this one, should take only one parameter at a time?
# # def test_validate_source_band():

# #     # Creating a specific set of parameters to emulate input from configuration file
# #     root        = Element("config")
# #     input_data  = SubElement(root, "InputData")
# #     input_type  = SubElement(input_data, "type")
# #     source_band = SubElement(input_data, "source_band")
# #     grid_params = SubElement(root, "GridParams")
# #     grid_type   = SubElement(grid_params, "grid_type")

# #     source_band = ConfigFile.validate_source_band(
# #         config_object= root,
# #         source_band  = 'InputData/source_band',
# #         input_data_type = input_data_type
# #     )


@pytest.mark.parametrize(
    "grid_definition_value, expected_output, expect_exception",
    [
        # Expected to Pass
        ("EASE2_G9km", "EASE2_G9km", False),
        (" MERC_G25km ", "MERC_G25km", False),
        # # Expected to Fail
        ("stereo_N25km", "STEREO_N25km", True),
        ("EASE2_G", "EASE2_G9km", True),
        ("", "", True),
        (["EASE2_G9km", "EASE2_N36km"], "EASE2_G9km", True),
        (None, None, True),
    ],
)
def test_validate_grid_definition(
    grid_definition_value,
    expected_output,
    expect_exception,
):
    r"""
    Pytest unit test for the `validate_grid_definition` method.

    This test function checks the validation of `grid_definition` values in the XML
    configuration. It ensures that valid grid definitions pass validation and invalid ones
    raise appropriate exceptions. The test uses `pytest.mark.parametrize` to cover
    multiple test cases efficiently.

    Parameters
    ----------
    grid_definition_value : str or list or None
        The value extracted from the XML configuration for grid definition.
    expected_output : str or None
        The expected validated grid definition if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected during validation.

    Raises
    ------
    ValueError
        If `grid_definition_value` is not in the list of valid grid definitions.
    AttributeError
        If the required XML tag is missing or incorrectly specified.

    Notes
    -----
    - Uses `pytest.raises` to assert expected exceptions.
    - Employs parameterized testing for efficiency.
    - Tests both valid and invalid grid definitions, including whitespace-trimmed cases.
    """

    config_xml = f"""<config><GridParams><grid_definition>{grid_definition_value}</grid_definition></GridParams></config>"""
    config_object = ET.ElementTree(ET.fromstring(config_xml)).getroot()

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises((ValueError, AttributeError)):
            ConfigFile.validate_grid_definition(
                config_object=config_object,
                grid_definition="GridParams/grid_definition",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_grid_definition(
            config_object=config_object,
            grid_definition="GridParams/grid_definition",
        )
        assert result == expected_output


@pytest.mark.parametrize(
    "grid_definition_value, projection_definition_value, expected_output, expect_exception",
    [
        # Expected to Pass
        ("EASE2_G9km", " G ", "G", False),
        ("MERC_G25km", "merc_G", "MERC_G", False),
        (None, "MERC_G", None, False),
        # # Expected to Fail
        ("MERC_G25km", "random string", "MERC_G", True),
        ("STEREO_N25km", " ", None, True),
        ("EASE2_N36km", None, None, True),
    ],
)
def test_validate_projection_definition(
    grid_definition_value,
    projection_definition_value,
    expected_output,
    expect_exception,
):
    r"""
    Pytest unit test for the `validate_projection_definition` method.

    This test function checks the validation of `projection_definition` values
    based on the selected `grid_definition`. It ensures that valid projections pass
    validation and invalid ones raise appropriate exceptions. The test uses
    `pytest.mark.parametrize` to cover multiple test cases efficiently.

    Parameters
    ----------
    grid_definition_value : str or None
        The grid definition used to determine valid projection definitions.
    projection_definition_value : str or None
        The projection definition extracted from the XML configuration.
    expected_output : str or None
        The expected validated projection definition if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected during validation.

    Raises
    ------
    ValueError
        If `projection_definition_value` is not valid for the given `grid_definition_value`.
    AttributeError
        If the required XML tag is missing or incorrectly specified.

    Notes
    -----
    - Uses `pytest.raises` to assert expected exceptions.
    - Employs parameterized testing for efficiency.
    - Tests valid and invalid projection definitions, including whitespace-trimmed cases.
    """

    config_xml = f"""<config><GridParams><projection_definition>{projection_definition_value}</projection_definition></GridParams></config>"""
    config_object = ET.ElementTree(ET.fromstring(config_xml)).getroot()

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises((ValueError, AttributeError)):
            ConfigFile.validate_projection_definition(
                config_object=config_object,
                grid_definition=grid_definition_value,
                projection_definition="GridParams/projection_definition",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_projection_definition(
            config_object=config_object,
            grid_definition=grid_definition_value,
            projection_definition="GridParams/projection_definition",
        )
        assert result == expected_output


@pytest.mark.parametrize(
    "regridding_algorithm_value, expected_output, expect_exception",
    [
        # Expected to Pass
        (" NN ", "NN", False),
        ("Dib", "DIB", False),
        # # Expected to Fail
        (["BG"], None, True),
        ("IDS LW", None, True),
        ("IDS13", None, True),
        ("random string", None, True),
        (" ", None, True),
    ],
)
def test_validate_regridding_algorithm(
    regridding_algorithm_value,
    expected_output,
    expect_exception,
):
    r"""
    Pytest unit test for the `validate_regridding_algorithm` method.

    This test function checks the validation of `regridding_algorithm` values
    in the XML configuration. It ensures that valid regridding algorithms pass
    validation and invalid ones raise appropriate exceptions. The test uses
    `pytest.mark.parametrize` to cover multiple test cases efficiently.

    Parameters
    ----------
    regridding_algorithm_value : str or list
        The regridding algorithm extracted from the XML configuration.
    expected_output : str or None
        The expected validated regridding algorithm if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected during validation.

    Raises
    ------
    ValueError
        If `regridding_algorithm_value` is not in the list of valid regridding algorithms.
    AttributeError
        If the required XML tag is missing or incorrectly specified.
    """

    config_xml = f"""<config><ReGridderParams><regridding_algorithm>{regridding_algorithm_value}</regridding_algorithm></ReGridderParams></config>"""
    config_object = ET.ElementTree(ET.fromstring(config_xml)).getroot()

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises((ValueError, AttributeError)):
            ConfigFile.validate_regridding_algorithm(
                config_object=config_object,
                regridding_algorithm="ReGridderParams/regridding_algorithm",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_regridding_algorithm(
            config_object=config_object,
            regridding_algorithm="ReGridderParams/regridding_algorithm",
        )
        assert result == expected_output


# # def test_validate_split_fore_aft():

# #     root        = Element("config")
# #     input_data  = SubElement(root, "InputData")
# #     input_type  = SubElement(input_data, "type")
# #     split_data  = SubElement(input_data, "split_fore_aft")
# #     #regridder_params = SubElement(root, "ReGridderParams")
# #     #regridding_algo = SubElement(regridder_params, "regridding_algorithm")
# #     #proj_def    = SubElement(regridder_params, "projection_definition")
# #
# #     # Expected to pass
# #     # AMSR2 is False by default, so the empty string or any other input is valid
# #     valid_input = {"CIMR": ['True', 'False'], "AMSR2": ["fAlse", "true", ""], "SMAP": ['False', 'True']}

# #     for instrument, bool_vals in valid_input.items():

# #         for def_name in def_names:
# #             grid_def.text = def_name

# #             for proj_name in proj_def_valid_input[def_range]:

# #                 #for proj_name in proj_names:
# #                 proj_def.text = proj_name

# #             assert ConfigFile.validate_split_fore_aft(
# #                 config_object   = root,
# #                 split_fore_aft  = 'InputData/split_fore_aft',
# #                 input_data_type = input_type.text
# #             )


# #    if input_data_type == 'AMSR2':
# #        return False
# #    valid_input = ['True', 'False']
# #    if config_object.find(split_fore_aft).text in valid_input:
# #        if config_object.find(split_fore_aft).text == 'True':
# #            return True
# #        else:
# #            return False
# #    raise ValueError(
# #        f"Invalid split fore aft. Check Configuration File."
# #        f" Valid split fore aft are: {valid_input}"
# #    )


# # @pytest.mark.parametrize(
# #     "split_value, input_data_type, expected",
# #     [
# #         ('True', 'SomeType', True),    # Valid True case
# #         ('False', 'SomeType', False), # Valid False case
# #         ('True', 'AMSR2', False),     # Special case for AMSR2
# #         ('InvalidValue', 'SomeType', pytest.raises(ValueError)), # Invalid case
# #     ],
# # )
# # def test_validate_split_fore_aft(split_value, input_data_type, expected):
# #     # Arrange
# #     config_object = Element("config")
# #     input_data = SubElement(config_object, "InputData")
# #     split_data = SubElement(input_data, "split_fore_aft")
# #     split_data.text = split_value

# #     # Act & Assert
# #     if isinstance(expected, pytest.raises):
# #         with expected:  # Handle exceptions for invalid cases
# #             ConfigFile.validate_split_fore_aft(
# #                 config_object=config_object,
# #                 split_fore_aft='InputData/split_fore_aft',
# #                 input_data_type=input_data_type,
# #             )
# #     else:
# #         result = ConfigFile.validate_split_fore_aft(
# #             config_object=config_object,
# #             split_fore_aft='InputData/split_fore_aft',
# #             input_data_type=input_data_type,
# #         )
# #         assert result == expected


# # TODO: Rewrite the original method because it seems to pass even
# #       when neither of acceptable instruments are called
# @pytest.mark.parametrize(
#     "split_value, input_data_type, expected, expect_exception",
#     [
#         # Expected to pass
#         ("True", "SMAP", True, False),  # Valid True case
#         ("False", "CIMR", False, False),  # Valid False case
#         ("True", "AMSR2", False, False),  # Special case for AMSR2
#         ("False", "AMSR2", False, False),  # Special case for AMSR2
#         ("", "AMSR2", False, False),  # Special case for AMSR2
#         # Expected to fail
#         ("InvalidValue", "InputData", None, True),  # Invalid case
#         ("False", "SomeType", None, True),  # Valid False case
#     ],
# )
# def test_validate_split_fore_aft(
#     split_value, input_data_type, expected, expect_exception
# ):
#     # Arrange
#     config_object = Element("config")
#     input_data = SubElement(config_object, "InputData")
#     split_data = SubElement(input_data, "split_fore_aft")
#     split_data.text = split_value

#     # Expected to fail
#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(
#             ValueError, match="Invalid split fore aft. Check Configuration File."
#         ):
#             ConfigFile.validate_split_fore_aft(
#                 config_object=config_object,
#                 split_fore_aft="InputData/split_fore_aft",
#                 input_data_type=input_data_type,
#             )
#     # Expected to pass
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_split_fore_aft(
#             config_object=config_object,
#             split_fore_aft="InputData/split_fore_aft",
#             input_data_type=input_data_type,
#         )
#         # Assert: Compare result to expected value
#         assert result == expected

#     # <ReGridderParams>
#     #  <regridding_algorithm>IDS</regridding_algorithm>
#     #  <search_radius>18</search_radius>
#     #  <max_neighbours>20</max_neighbours>
#     #  <variables_to_regrid></variables_to_regrid>
#     #  <source_antenna_method>gaussian</source_antenna_method>
#     #  <target_antenna_method>real</target_antenna_method>
#     #  <polarisation_method>scalar</polarisation_method>
#     #  <source_antenna_threshold>0.5</source_antenna_threshold>
#     #  <target_antenna_threshold>0.5</target_antenna_threshold>
#     #  <MRF_grid_definition>EASE2_G3km</MRF_grid_definition>
#     #  <MRF_projection_definition>G</MRF_projection_definition>
#     #  <source_gaussian_params>100000 100000</source_gaussian_params>
#     #  <target_gaussian_params>100000 100000</target_gaussian_params>
#     #  <boresight_shift>True</boresight_shift>
#     #  <rsir_iteration>15</rsir_iteration>
#     #  <bg_smoothing>1</bg_smoothing>
#     # </ReGridderParams>


# @pytest.mark.parametrize(
#     "save_to_disk_value, expected, expect_exception",
#     [
#         ("True", True, False),  # Valid True case
#         ("False", False, False),  # Valid False case
#         ("InvalidValue", None, True),  # Invalid case
#         ("", None, True),  # Empty string, invalid case
#     ],
# )
# def test_validate_save_to_disk(save_to_disk_value, expected, expect_exception):
#     # Arrange
#     config_object = Element("config")
#     output_data = SubElement(config_object, "OutputData")
#     save_to_disk = SubElement(output_data, "save_to_disk")
#     save_to_disk.text = save_to_disk_value

#     # Pytest need to match error message exactly in this case for it to work
#     error_message = "Invalid `save_to_disk`. Check Configuration File. "  # + \
#     # "`save_to_disk` must be either True or False")

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match=error_message):
#             # "Invalid `save_to_disk`. Check Configuration File. save_to_disk must be either True or False"
#             ConfigFile.validate_save_to_disk(
#                 config_object=config_object,
#                 save_to_disk="OutputData/save_to_disk",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_save_to_disk(
#             config_object=config_object,
#             save_to_disk="OutputData/save_to_disk",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # @pytest.mark.parametrize(
# #     "search_radius_value, grid_definition, grid_type, expected, expect_exception",
# #     [
# #         # Valid Cases
# #         ('5.0', 'EASE2_G1km', 'L1C', 5000.0, False),  # Defined search_radius (5.0 km)
# #         # (None, 'EASE2_G1km', 'L1C', 'CIMR', np.sqrt(2 * (1000.9 / 2) ** 2), False),  # L1C with grid_definition
# #         # (None, 'EASE2_G3km', 'L1C', 'SMAP', np.sqrt(2 * (3002.69 / 2) ** 2), False),  # L1C with SMAP
# #         # (None, 'EASE2_N3km', 'L1C', 'AMSR2', np.sqrt(2 * (3000 / 2) ** 2), False),  # L1C with AMSR2
# #         # (None, 'EASE2_G1km', 'L1R', 'CIMR', 73000 / 2, False),  # L1R with CIMR
# #         # (None, 'EASE2_G1km', 'L1R', 'AMSR2', 62000 / 2, False),  # L1R with AMSR2

# #         # # Invalid Cases
# #         # (None, 'EASE2_G1km', 'L1C', 'INVALID', None, True),  # Invalid input_data_type
# #         # (None, 'EASE2_G1km', 'L1C', 'cimr', None, True),  # Case-sensitive check (lowercase)
# #         # (None, 'EASE2_G1km', 'L1C', 'Smap', None, True),  # Case-sensitive check (mixed case)
# #     ],
# # )
# # def test_validate_search_radius(search_radius_value, grid_definition, grid_type, input_data_type, expected, expect_exception):
# #     # Arrange
# #     config_object = Element("config")
# #     regridder_params = SubElement(config_object, "ReGridderParams")
# #     search_radius = SubElement(regridder_params, "search_radius")
# #     if search_radius_value is not None:
# #         search_radius.text = search_radius_value

# #     if expect_exception:
# #         # Act & Assert: Expect an exception
# #         with pytest.raises(ValueError, match=f"Invalid `input_data_type`: {input_data_type}"):
# #             ConfigFile.validate_search_radius(
# #                 config_object=config_object,
# #                 search_radius='ReGridderParams/search_radius',
# #                 grid_definition=grid_definition,
# #                 grid_type=grid_type,
# #                 #input_data_type=input_data_type,
# #             )
# #     else:
# #         # Act: No exception expected
# #         result = ConfigFile.validate_search_radius(
# #             config_object=config_object,
# #             search_radius='ReGridderParams/search_radius',
# #             grid_definition=grid_definition,
# #             grid_type=grid_type,
# #             #input_data_type=input_data_type,
# #         )
# #         # Assert: Compare result to expected value
# #         assert result == expected


# # TODO: Fix SMAP
# @pytest.mark.parametrize(
#     "search_radius_value, grid_definition, grid_type, input_data_type, expected, expect_exception",
#     [
#         # Valid Cases
#         ("5.0", "EASE2_G1km", "L1C", "CIMR", 5000.0, False),  # Valid numeric value
#         (
#             None,
#             "EASE2_G1km",
#             "L1C",
#             "CIMR",
#             np.sqrt(2 * (GRIDS["EASE2_G1km"]["res"] / 2) ** 2),
#             False,
#         ),
#         ("", "EASE2_G36km", "L1R", "CIMR", 73000 / 2, False),
#         (" ", "EASE2_G1km", "L1R", "AMSR2", 62000 / 2, False),
#         (
#             " ",
#             "STEREO_N25km",
#             "L1C",
#             "CIMR",
#             np.sqrt(2 * (GRIDS["STEREO_N25km"]["res"] / 2) ** 2),
#             False,
#         ),
#         ("18", "MERC_G12.km", "L1C", "SMAP", 18000, False),
#         # Invalid Cases
#         (
#             "18",
#             "MERC_G12.km",
#             "L1R",
#             "SMAP",
#             18000,
#             True,
#         ),  # TODO: SMAP does not have L1R
#         ("abc", "EASE2_G1km", "L1C", "CIMR", None, True),  # Non-numeric string
#         ("123abc", "EASE2_G1km", "L1C", "CIMR", None, True),  # Partially numeric
#     ],
# )
# def test_validate_search_radius_numeric_check(
#     search_radius_value,
#     grid_definition,
#     grid_type,
#     input_data_type,
#     expected,
#     expect_exception,
# ):
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     search_radius = SubElement(regridder_params, "search_radius")
#     if search_radius_value is not None:
#         search_radius.text = search_radius_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match="Invalid `search_radius`"):
#             ConfigFile.validate_search_radius(
#                 config_object=config_object,
#                 search_radius="ReGridderParams/search_radius",
#                 grid_definition=grid_definition,
#                 grid_type=grid_type,
#                 input_data_type=input_data_type,
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_search_radius(
#             config_object=config_object,
#             search_radius="ReGridderParams/search_radius",
#             grid_definition=grid_definition,
#             grid_type=grid_type,
#             input_data_type=input_data_type,
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # # TODO: Write the test for this method
# # def test_get_scan_geometry():
# #     ...


# @pytest.mark.parametrize(
#     "input_data_type, variables_to_regrid_value, expected, expect_exception",
#     [
#         # Valid Cases for SMAP
#         (
#             "SMAP",
#             "bt_h bt_v longitude latitude",
#             ["bt_h", "bt_v", "longitude", "latitude"],
#             False,
#         ),
#         (
#             "SMAP",
#             None,
#             [
#                 "bt_h",
#                 "bt_v",
#                 "bt_3",
#                 "bt_4",
#                 "processing_scan_angle",
#                 "longitude",
#                 "latitude",
#             ],
#             False,
#         ),  # Default vars
#         ("SMAP", "bt_h bt_v invalid_var", None, True),  # Invalid variable in SMAP
#         # Valid Cases for AMSR2
#         (
#             "AMSR2",
#             "bt_h bt_v longitude latitude",
#             ["bt_h", "bt_v", "longitude", "latitude"],
#             False,
#         ),
#         (
#             "AMSR2",
#             None,
#             ["bt_h", "bt_v"],
#             False,
#         ),  # No default_vars for AMSR2, so None should be an error
#         ("AMSR2", "invalid_var", None, True),  # Invalid variable in AMSR2
#         # Valid Cases for CIMR
#         (
#             "CIMR",
#             "bt_h bt_v longitude latitude oza",
#             ["bt_h", "bt_v", "longitude", "latitude", "oza"],
#             False,
#         ),
#         (
#             "CIMR",
#             None,
#             [
#                 "bt_h",
#                 "bt_v",
#                 "bt_3",
#                 "bt_4",
#                 "processing_scan_angle",
#                 "longitude",
#                 "latitude",
#             ],
#             False,
#         ),  # Default vars
#         ("CIMR", "invalid_var", None, True),  # Invalid variable in CIMR
#         # Invalid input_data_type
#         ("INVALID_TYPE", None, None, True),  # Unsupported input_data_type
#     ],
# )
# def test_validate_variables_to_regrid(
#     input_data_type, variables_to_regrid_value, expected, expect_exception
# ):
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     variables_to_regrid = SubElement(regridder_params, "variables_to_regrid")
#     if variables_to_regrid_value is not None:
#         variables_to_regrid.text = variables_to_regrid_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError):
#             ConfigFile.validate_variables_to_regrid(
#                 config_object=config_object,
#                 input_data_type=input_data_type,
#                 variables_to_regrid="ReGridderParams/variables_to_regrid",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_variables_to_regrid(
#             config_object=config_object,
#             input_data_type=input_data_type,
#             variables_to_regrid="ReGridderParams/variables_to_regrid",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: The test passes. but I think more needs to be done for checking the configuration
# @pytest.mark.parametrize(
#     "regridding_algorithm, max_neighbours_value, expected, expect_exception",
#     [
#         # Case: Regridding algorithm is NN
#         ("NN", None, 1, False),  # Always return 1 regardless of max_neighbours
#         # Case: Valid max_neighbours value
#         ("Other", "500", 500, False),  # Valid integer string
#         ("Other", "1000", 1000, False),  # Valid integer string
#         # Case: max_neighbours not defined
#         ("Other", None, 1000, False),  # Defaults to 1000
#         # Case: Invalid max_neighbours value
#         ("Other", "invalid", None, True),  # Non-integer string
#         ("Other", "", None, True),  # Empty string
#         ("Other", " ", None, True),  # Space-only string
#     ],
# )
# def test_validate_max_neighbours(
#     regridding_algorithm, max_neighbours_value, expected, expect_exception
# ):
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     max_neighbours = SubElement(regridder_params, "max_neighbours")
#     if max_neighbours_value is not None:
#         max_neighbours.text = max_neighbours_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError):
#             ConfigFile.validate_max_neighbours(
#                 config_object=config_object,
#                 max_neighbours="ReGridderParams/max_neighbours",
#                 regridding_algorithm=regridding_algorithm,
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_max_neighbours(
#             config_object=config_object,
#             max_neighbours="ReGridderParams/max_neighbours",
#             regridding_algorithm=regridding_algorithm,
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "source_antenna_method_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("gaussian", "gaussian", False),
#         ("real", "real", False),
#         ("gaussian_projected", "gaussian_projected", False),
#         # Default case
#         (None, "real", False),  # Defaults to 'real' if value is None
#         ("", "real", False),  # Empty string
#         (" ", "real", False),  # Space-only string
#         # Invalid cases
#         ("invalid_value", None, True),  # Invalid string
#         ("GAUSSIAN", None, True),  # Invalid string
#         ("Gaussian", None, True),  # Invalid string
#     ],
# )
# def test_validate_source_antenna_method(
#     source_antenna_method_value, expected, expect_exception
# ):
#     """
#     Test the `validate_source_antenna_method` method of the ConfigFile class.

#     This test covers the following scenarios:
#     - Valid `source_antenna_method` values (e.g., 'gaussian', 'real', 'gaussian_projected').
#     - Missing `source_antenna_method` (defaults to 'real').
#     - Invalid `source_antenna_method` values (e.g., 'invalid_value', empty strings).

#     Parameters:
#     - source_antenna_method_value: The value of the source_antenna_method in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases should return the correct value.
#     Missing or invalid cases should raise a ValueError or default to 'real' if appropriate.
#     """

#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     source_antenna = SubElement(regridder_params, "source_antenna_method")
#     if source_antenna_method_value is not None:
#         source_antenna.text = source_antenna_method_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(
#             ValueError, match="Invalid antenna method. Check Configuration File."
#         ):
#             ConfigFile.validate_source_antenna_method(
#                 config_object=config_object,
#                 source_antenna_method="ReGridderParams/source_antenna_method",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_source_antenna_method(
#             config_object=config_object,
#             source_antenna_method="ReGridderParams/source_antenna_method",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "target_antenna_method_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("gaussian", "gaussian", False),
#         ("real", "real", False),
#         ("gaussian_projected", "gaussian_projected", False),
#         # Default case
#         (None, "real", False),  # Defaults to 'real' if value is None
#         ("", "real", False),  # Empty string
#         (" ", "real", False),  # Space-only string
#         # Invalid cases
#         ("invalid_value", None, True),  # Invalid string
#         ("GAUSSIAN", None, True),  # Invalid string
#         ("Gaussian", None, True),  # Invalid string
#     ],
# )
# def test_validate_target_antenna_method(
#     target_antenna_method_value, expected, expect_exception
# ):
#     """
#     Test the `validate_target_antenna_method` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid `target_antenna_method` values (e.g., 'gaussian', 'real', 'gaussian_projected').
#     - Default value ('real') when `target_antenna_method` is missing or None.
#     - Error handling for invalid `target_antenna_method` values.

#     Parameters:
#     - target_antenna_method_value: The value of the target_antenna_method in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the appropriate value.
#     Missing values default to 'real'.
#     Invalid values raise a ValueError.
#     """

#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     target_antenna = SubElement(regridder_params, "target_antenna_method")
#     if target_antenna_method_value is not None:
#         target_antenna.text = target_antenna_method_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(
#             ValueError, match="Invalid antenna method. Check Configuration File."
#         ):
#             ConfigFile.validate_target_antenna_method(
#                 config_object=config_object,
#                 target_antenna_method="ReGridderParams/target_antenna_method",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_target_antenna_method(
#             config_object=config_object,
#             target_antenna_method="ReGridderParams/target_antenna_method",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "source_antenna_threshold_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("9.5", 9.5, False),  # Float value
#         ("10", 10.0, False),  # Integer value as string
#         ("0", 0.0, False),  # Zero value
#         ("-5", -5.0, False),  # Negative value
#         # Default case
#         (None, None, False),  # Returns None if value is not provided
#         ("", None, False),  # Empty string
#         (" ", None, False),  # Space-only string
#         # Invalid cases
#         ("invalid", None, True),  # Non-numeric string
#         ("a9.5", 9.5, True),  # Typo in Float value
#     ],
# )
# def test_validate_source_antenna_threshold(
#     source_antenna_threshold_value, expected, expect_exception
# ):
#     """
#     Test the `validate_source_antenna_threshold` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid numeric `source_antenna_threshold` values (float and integer).
#     - Default behavior (returns None) when `source_antenna_threshold` is missing or None.
#     - Error handling for invalid `source_antenna_threshold` values (non-numeric).

#     Parameters:
#     - source_antenna_threshold_value: The value of the source_antenna_threshold in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the appropriate numeric value.
#     Missing values return None.
#     Invalid values raise a ValueError.
#     """

#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     source_antenna_threshold = SubElement(regridder_params, "source_antenna_threshold")

#     if source_antenna_threshold_value is not None:
#         source_antenna_threshold.text = source_antenna_threshold_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(
#             ValueError, match="Invalid antenna threshold"
#         ):  # . Check Configuration File."):
#             ConfigFile.validate_source_antenna_threshold(
#                 config_object=config_object,
#                 source_antenna_threshold="ReGridderParams/source_antenna_threshold",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_source_antenna_threshold(
#             config_object=config_object,
#             source_antenna_threshold="ReGridderParams/source_antenna_threshold",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "target_antenna_threshold_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("9.5", 9.5, False),  # Float value
#         ("10", 10.0, False),  # Integer value as string
#         ("0", 0.0, False),  # Zero value
#         ("-5", -5.0, False),  # Negative value
#         # Default case
#         (None, 9.0, False),  # Defaults to 9.0 if value is None
#         ("", 9.0, False),  # Empty string
#         (" ", 9.0, False),  # Space-only string
#         # Invalid cases
#         ("invalid", None, True),  # Non-numeric string
#         ("a9.5", 9.5, True),  # Typo in Float value
#     ],
# )
# def test_validate_target_antenna_threshold(
#     target_antenna_threshold_value, expected, expect_exception
# ):
#     """
#     Test the `validate_target_antenna_threshold` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid numeric `target_antenna_threshold` values (float and integer).
#     - Default behavior (returns 9.0) when `target_antenna_threshold` is missing or None.
#     - Error handling for invalid `target_antenna_threshold` values (non-numeric).

#     Parameters:
#     - target_antenna_threshold_value: The value of the target_antenna_threshold in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the appropriate numeric value.
#     Missing values return the default of 9.0.
#     Invalid values raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     target_antenna_threshold = SubElement(regridder_params, "target_antenna_threshold")
#     if target_antenna_threshold_value is not None:
#         target_antenna_threshold.text = target_antenna_threshold_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(
#             ValueError, match="Invalid antenna threshold:"
#         ):  # . Check Configuration File."):
#             ConfigFile.validate_target_antenna_threshold(
#                 config_object=config_object,
#                 target_antenna_threshold="ReGridderParams/target_antenna_threshold",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_target_antenna_threshold(
#             config_object=config_object,
#             target_antenna_threshold="ReGridderParams/target_antenna_threshold",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "polarisation_method_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("scalar", "scalar", False),
#         ("mueller", "mueller", False),
#         # Default case
#         (None, "scalar", False),  # Defaults to 'scalar' if value is None
#         ("", "scalar", False),  # Empty string
#         (" ", "scalar", False),  # Space-only string
#         # Invalid cases
#         ("invalid", None, True),  # Invalid string
#     ],
# )
# def test_validate_polarisation_method(
#     polarisation_method_value, expected, expect_exception
# ):
#     """
#     Test the `validate_polarisation_method` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid `polarisation_method` values (e.g., 'scalar', 'mueller').
#     - Default behavior (returns 'scalar') when `polarisation_method` is missing or None.
#     - Error handling for invalid `polarisation_method` values (non-valid strings).

#     Parameters:
#     - polarisation_method_value: The value of the polarisation_method in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the appropriate value.
#     Missing values default to 'scalar'.
#     Invalid values raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     polarisation = SubElement(regridder_params, "polarisation_method")
#     if polarisation_method_value is not None:
#         polarisation.text = polarisation_method_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(
#             ValueError, match="Invalid polarisation method:"
#         ):  # . Check Configuration File."):
#             ConfigFile.validate_polarisation_method(
#                 config_object=config_object,
#                 polarisation_method="ReGridderParams/polarisation_method",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_polarisation_method(
#             config_object=config_object,
#             polarisation_method="ReGridderParams/polarisation_method",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "input_data_type, boresight_shift_value, expected, expect_exception",
#     [
#         # Case: Input data type is not SMAP
#         ("AMSR2", None, False, False),  # Always return False for non-SMAP
#         ("CIMR", "True", False, False),  # Always return False for non-SMAP
#         # Case: Input data type is SMAP
#         ("SMAP", "True", True, False),  # Valid True case
#         ("SMAP", "False", False, False),  # Valid False case
#         ("SMAP", None, False, False),  # Defaults to False if value is None
#         ("SMAP", "", False, False),  # Empty string
#         ("SMAP", " ", False, False),  # Space-only string
#         # Invalid cases
#         ("SMAP", "invalid_value", None, True),  # Invalid string
#     ],
# )
# def test_validate_boresight_shift(
#     input_data_type, boresight_shift_value, expected, expect_exception
# ):
#     """
#     Test the `validate_boresight_shift` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid `boresight_shift` values (e.g., 'True', 'False').
#     - Default behavior (returns False) when `boresight_shift` is missing or None.
#     - Always returns False for non-SMAP input_data_type.
#     - Error handling for invalid `boresight_shift` values.

#     Parameters:
#     - input_data_type: The type of input data (e.g., 'SMAP', 'AMSR2', 'CIMR').
#     - boresight_shift_value: The value of the boresight_shift in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the appropriate Boolean value.
#     Missing values default to False.
#     Invalid values raise a ValueError.
#     """

#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     boresight_shift = SubElement(regridder_params, "boresight_shift")
#     if boresight_shift_value is not None:
#         boresight_shift.text = boresight_shift_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(
#             ValueError, match="Invalid boresight shift:"
#         ):  # . Check Configuration File."):
#             ConfigFile.validate_boresight_shift(
#                 config_object=config_object,
#                 boresight_shift="ReGridderParams/boresight_shift",
#                 input_data_type=input_data_type,
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_boresight_shift(
#             config_object=config_object,
#             boresight_shift="ReGridderParams/boresight_shift",
#             input_data_type=input_data_type,
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "reduced_grid_inds_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("0 10 0 10", [0, 10, 0, 10], False),
#         ("5 15 3 12", [5, 15, 3, 12], False),
#         ("", None, False),  # Empty string
#         # Missing case
#         (None, None, False),
#         # Invalid cases
#         ("-1 10 0 10", None, True),  # Negative grid_row_min
#         ("0 -10 0 10", None, True),  # Negative grid_row_max
#         ("0 10 -5 10", None, True),  # Negative grid_col_min
#         ("0 10 0 -10", None, True),  # Negative grid_col_max
#         ("10 5 0 10", None, True),  # grid_row_min > grid_row_max
#         ("0 10 12 5", None, True),  # grid_col_min > grid_col_max
#         ("0 10", None, True),  # Insufficient values
#         ("invalid input here", None, True),  # Non-numeric values
#     ],
# )
# def test_validate_reduced_grid_inds(
#     reduced_grid_inds_value, expected, expect_exception
# ):
#     """
#     Test the `validate_reduced_grid_inds` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid grid indices.
#     - Default behavior (returns None) when `reduced_grid_inds` is missing.
#     - Error handling for invalid grid indices (negative values, out-of-order ranges, etc.).

#     Parameters:
#     - reduced_grid_inds_value: The value of reduced_grid_inds in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the parsed grid indices as a list of integers.
#     Missing values return None.
#     Invalid values raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     grid_params = SubElement(config_object, "GridParams")
#     reduced_grid_inds = SubElement(grid_params, "reduced_grid_inds")
#     if reduced_grid_inds_value is not None:
#         reduced_grid_inds.text = reduced_grid_inds_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match="Invalid `reduced_grid_inds`:"):
#             ConfigFile.validate_reduced_grid_inds(
#                 config_object=config_object,
#                 reduced_grid_inds="GridParams/reduced_grid_inds",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_reduced_grid_inds(
#             config_object=config_object,
#             reduced_grid_inds="GridParams/reduced_grid_inds",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "source_gaussian_params_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("1.5 2.5", [1.5, 2.5], False),  # Two valid floats
#         ("0 10", [0.0, 10.0], False),  # Integer values as strings
#         ("-5.5 3.2", [-5.5, 3.2], False),  # Negative and positive floats
#         # Invalid cases
#         (None, None, True),  # Missing value
#         ("1.5", None, True),  # Only one parameter
#         ("1.5 2.5 3.5", None, True),  # Too many parameters
#         ("invalid 2.5", None, True),  # Non-numeric parameter
#         ("1.5 invalid", None, True),  # Non-numeric parameter
#         ("", None, True),  # Empty string
#         (" ", None, True),  # Space-only string
#     ],
# )
# def test_validate_source_gaussian_params(
#     source_gaussian_params_value, expected, expect_exception
# ):
#     """
#     Test the `validate_source_gaussian_params` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid source Gaussian parameters (two numbers).
#     - Error handling for invalid or missing parameters.

#     Parameters:
#     - source_gaussian_params_value: The value of source_gaussian_params in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return a list of two floats.
#     Invalid cases raise a ValueError.
#     """

#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     source_gaussian_params = SubElement(regridder_params, "source_gaussian_params")
#     if source_gaussian_params_value is not None:
#         source_gaussian_params.text = source_gaussian_params_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(
#             ValueError, match="Invalid|Missing "
#         ):  # source Gaussian parameters"):#|Missing source Gaussian parameters|Invalid parameter"):
#             ConfigFile.validate_source_gaussian_params(
#                 config_object=config_object,
#                 source_gaussian_params="ReGridderParams/source_gaussian_params",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_source_gaussian_params(
#             config_object=config_object,
#             source_gaussian_params="ReGridderParams/source_gaussian_params",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "target_gaussian_params_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("1.5 2.5", [1.5, 2.5], False),  # Two valid floats
#         ("0 10", [0.0, 10.0], False),  # Integer values as strings
#         ("-5.5 3.2", [-5.5, 3.2], False),  # Negative and positive floats
#         # Invalid cases
#         (None, None, True),  # Missing value
#         ("1.5", None, True),  # Only one parameter
#         ("1.5 2.5 3.5", None, True),  # Too many parameters
#         ("invalid 2.5", None, True),  # Non-numeric parameter
#         ("1.5 invalid", None, True),  # Non-numeric parameter
#         ("", None, True),  # Empty string
#         (" ", None, True),  # Space-only string
#     ],
# )
# def test_validate_target_gaussian_params(
#     target_gaussian_params_value, expected, expect_exception
# ):
#     """
#     Test the `validate_target_gaussian_params` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid target Gaussian parameters (two numbers).
#     - Error handling for invalid or missing parameters.

#     Parameters:
#     - target_gaussian_params_value: The value of target_gaussian_params in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return a list of two floats.
#     Invalid cases raise a ValueError.
#     """

#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     target_gaussian_params = SubElement(regridder_params, "target_gaussian_params")
#     if target_gaussian_params_value is not None:
#         target_gaussian_params.text = target_gaussian_params_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match="Invalid|Missing"):
#             ConfigFile.validate_target_gaussian_params(
#                 config_object=config_object,
#                 target_gaussian_params="ReGridderParams/target_gaussian_params",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_target_gaussian_params(
#             config_object=config_object,
#             target_gaussian_params="ReGridderParams/target_gaussian_params",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "rsir_iteration_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("10", 10, False),  # Valid integer
#         ("0", 0, False),  # Edge case: Zero iterations
#         ("1", 1, False),  # Edge case: Minimum positive iterations
#         # Invalid cases
#         (None, None, True),  # Missing value
#         ("", None, True),  # Empty string
#         ("invalid", None, True),  # Non-numeric string
#         ("1.5", None, True),  # Float value
#         ("-5", None, True),  # Negative integer (assuming it's invalid in this context)
#     ],
# )
# def test_validate_rsir_iteration(rsir_iteration_value, expected, expect_exception):
#     """
#     Test the `validate_rsir_iteration` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid RSIR iteration counts.
#     - Error handling for invalid or missing RSIR iteration counts.

#     Parameters:
#     - rsir_iteration_value: The value of rsir_iteration in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the integer value of the iteration count.
#     Invalid cases raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     rsir_iteration = SubElement(regridder_params, "rsir_iteration")
#     if rsir_iteration_value is not None:
#         rsir_iteration.text = rsir_iteration_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match="Invalid|Missing|rSIR"):
#             ConfigFile.validate_rsir_iteration(
#                 config_object=config_object,
#                 rsir_iteration="ReGridderParams/rsir_iteration",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_rsir_iteration(
#             config_object=config_object,
#             rsir_iteration="ReGridderParams/rsir_iteration",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "max_number_iteration_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("10", 10, False),  # Valid integer
#         ("0", 0, False),  # Edge case: Zero iterations
#         ("1", 1, False),  # Edge case: Minimum positive iterations
#         # Invalid cases
#         (None, None, True),  # Missing value
#         ("", None, True),  # Empty string
#         ("invalid", None, True),  # Non-numeric string
#         ("1.5", None, True),  # Float value
#         ("-5", None, True),  # Negative integer
#     ],
# )
# def test_validate_max_number_iteration(
#     max_number_iteration_value, expected, expect_exception
# ):
#     """
#     Test the `validate_max_number_iteration` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid maximum number of iteration counts.
#     - Error handling for invalid or missing maximum iteration counts.

#     Parameters:
#     - max_number_iteration_value: The value of max_number_iteration in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the integer value of the iteration count.
#     Invalid cases raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     max_number_iteration = SubElement(regridder_params, "max_number_iteration")
#     if max_number_iteration_value is not None:
#         max_number_iteration.text = max_number_iteration_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match="Invalid|Missing|Maximum"):
#             ConfigFile.validate_max_number_iteration(
#                 config_object=config_object,
#                 max_number_iteration="ReGridderParams/max_number_iteration",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_max_number_iteration(
#             config_object=config_object,
#             max_number_iteration="ReGridderParams/max_number_iteration",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "relative_tolerance_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("0.01", 0.01, False),  # Valid positive float
#         ("0", 0.0, False),  # Edge case: Zero tolerance
#         ("1.5", 1.5, False),  # Valid larger float value
#         # Invalid cases
#         (None, None, True),  # Missing value
#         ("", None, True),  # Empty string
#         ("invalid", None, True),  # Non-numeric string
#         ("-0.01", None, True),  # Negative float
#         ("-1", None, True),  # Negative integer
#     ],
# )
# def test_validate_relative_tolerance(
#     relative_tolerance_value, expected, expect_exception
# ):
#     """
#     Test the `validate_relative_tolerance` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid relative tolerance values.
#     - Error handling for invalid or missing tolerance values.

#     Parameters:
#     - relative_tolerance_value: The value of relative_tolerance in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the float value of the tolerance.
#     Invalid cases raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     relative_tolerance = SubElement(regridder_params, "relative_tolerance")
#     if relative_tolerance_value is not None:
#         relative_tolerance.text = relative_tolerance_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match="Invalid|Missing|Relative"):
#             ConfigFile.validate_relative_tolerance(
#                 config_object=config_object,
#                 relative_tolerance="ReGridderParams/relative_tolerance",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_relative_tolerance(
#             config_object=config_object,
#             relative_tolerance="ReGridderParams/relative_tolerance",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO:
# # - Check the docstring
# # - Check the tests here (figure out what kind of vals the reg param can get)
# @pytest.mark.parametrize(
#     "regularization_parameter_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("0.1", 0.1, False),  # Positive float
#         ("0", 0.0, False),  # Zero value
#         ("10", 10.0, False),  # Positive integer
#         ("-0.1", -0.1, False),  # Negative float
#         ("-5", -5.0, False),  # Negative integer
#         # Invalid cases
#         (None, None, True),  # Missing value
#         ("", None, True),  # Empty string
#         ("invalid", None, True),  # Non-numeric string
#     ],
# )
# def test_validate_regularization_parameter(
#     regularization_parameter_value, expected, expect_exception
# ):
#     """
#     Test the `validate_regularization_parameter` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid regularization parameter values.
#     - Error handling for invalid or missing values.

#     Parameters:
#     - regularization_parameter_value: The value of regularization_parameter in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the float value of the regularization parameter.
#     Invalid cases raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     regularization_parameter = SubElement(regridder_params, "regularization_parameter")
#     if regularization_parameter_value is not None:
#         regularization_parameter.text = regularization_parameter_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match="Invalid|Missing"):
#             ConfigFile.validate_regularization_parameter(
#                 config_object=config_object,
#                 regularization_parameter="ReGridderParams/regularization_parameter",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_regularization_parameter(
#             config_object=config_object,
#             regularization_parameter="ReGridderParams/regularization_parameter",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "MRF_grid_definition_value, expected, expect_exception",
#     [
#         # Valid cases
#         ("EASE2_G3km", "EASE2_G3km", False),
#         ("EASE2_G1km", "EASE2_G1km", False),
#         ("EASE2_S36km", "EASE2_S36km", False),
#         # Invalid cases
#         (None, None, True),  # Missing value
#         ("", None, True),  # Empty string
#         ("invalid_grid", None, True),  # Invalid grid definition
#         ("EASE2_G3km ", None, True),  # Trailing space (not an exact match)
#     ],
# )
# def test_validate_MRF_grid_definition(
#     MRF_grid_definition_value, expected, expect_exception
# ):
#     """
#     Test the `validate_MRF_grid_definition` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid grid definitions.
#     - Error handling for invalid or missing grid definitions.

#     Parameters:
#     - MRF_grid_definition_value: The value of the MRF grid definition in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.

#     Valid cases return the valid grid definition.
#     Invalid cases raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     MRF_grid_definition = SubElement(regridder_params, "MRF_grid_definition")

#     if MRF_grid_definition_value is not None:
#         MRF_grid_definition.text = MRF_grid_definition_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match="Invalid|Missing"):
#             ConfigFile.validate_MRF_grid_definition(
#                 config_object=config_object,
#                 MRF_grid_definition="ReGridderParams/MRF_grid_definition",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_MRF_grid_definition(
#             config_object=config_object,
#             MRF_grid_definition="ReGridderParams/MRF_grid_definition",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# @pytest.mark.parametrize(
#     "MRF_projection_definition_value, expected, expect_exception, match_message",
#     [
#         # Valid cases
#         ("G", "G", False, None),  # Global projection
#         ("N", "N", False, None),  # Northern projection
#         ("S", "S", False, None),  # Southern projection
#         (" G ", "G", False, None),  # Trailing/leading whitespace
#         # Invalid cases
#         (
#             None,
#             None,
#             True,
#             "Missing or blank MRF projection definition in the configuration file",
#         ),  # Missing value
#         (
#             "",
#             None,
#             True,
#             "Missing or blank MRF projection definition in the configuration file",
#         ),  # Empty string
#         (
#             "invalid",
#             None,
#             True,
#             "Invalid Projection Definition",
#         ),  # Invalid projection definition
#     ],
# )
# def test_validate_MRF_projection_definition(
#     MRF_projection_definition_value, expected, expect_exception, match_message
# ):
#     """
#     Test the `validate_MRF_projection_definition` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid projection definitions.
#     - Error handling for invalid or missing projection definitions.

#     Parameters:
#     - MRF_projection_definition_value: The value of the MRF projection definition in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.
#     - match_message: Substring to match in the raised exception message for invalid cases.

#     Valid cases return the valid projection definition.
#     Invalid cases raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     MRF_projection_definition = SubElement(
#         regridder_params, "MRF_projection_definition"
#     )
#     if MRF_projection_definition_value is not None:
#         MRF_projection_definition.text = MRF_projection_definition_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match=match_message):
#             ConfigFile.validate_MRF_projection_definition(
#                 config_object=config_object,
#                 MRF_projection_definition="ReGridderParams/MRF_projection_definition",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_MRF_projection_definition(
#             config_object=config_object,
#             MRF_projection_definition="ReGridderParams/MRF_projection_definition",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "bg_smoothing_value, expected, expect_exception, match_message",
#     [
#         # Valid cases
#         ("1.5", 1.5, False, None),  # Positive float
#         ("0", 0.0, False, None),  # Zero
#         ("-2.3", -2.3, False, None),  # Negative float
#         (None, 0.0, False, None),  # Missing value, should default to 0
#         (" 3.7 ", 3.7, False, None),  # Float with leading/trailing whitespace
#         # Invalid cases
#         ("invalid", None, True, "Invalid `bg_smoothing` value"),  # Non-numeric string
#         ("", None, True, "Invalid `bg_smoothing` value"),  # Empty string
#         (" ", None, True, "Invalid `bg_smoothing` value"),  # Space-only string
#     ],
# )
# def test_validate_bg_smoothing(
#     bg_smoothing_value, expected, expect_exception, match_message
# ):
#     """
#     Test the `validate_bg_smoothing` method of the ConfigFile class.

#     This test validates:
#     - Correct behavior for valid bg_smoothing values.
#     - Default behavior for missing values (defaults to 0).
#     - Error handling for invalid or non-numeric values.

#     Parameters:
#     - bg_smoothing_value: The value of bg_smoothing in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.
#     - match_message: Substring to match in the raised exception message for invalid cases.

#     Valid cases return the float value of bg_smoothing.
#     Invalid cases raise a ValueError.
#     """

#     # Arrange
#     config_object = Element("config")
#     regridder_params = SubElement(config_object, "ReGridderParams")
#     bg_smoothing = SubElement(regridder_params, "bg_smoothing")
#     if bg_smoothing_value is not None:
#         bg_smoothing.text = bg_smoothing_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match=match_message):
#             ConfigFile.validate_bg_smoothing(
#                 config_object=config_object,
#                 bg_smoothing="ReGridderParams/bg_smoothing",
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_bg_smoothing(
#             config_object=config_object,
#             bg_smoothing="ReGridderParams/bg_smoothing",
#         )
#         # Assert: Compare result to expected value
#         assert result == expected


# # TODO: Check the docstring
# @pytest.mark.parametrize(
#     "input_data_type, quality_control_value, expected, expect_exception, match_message",
#     [
#         # Input data types that always return False
#         ("AMSR2", "True", False, False, None),  # AMSR2 ignores quality control setting
#         ("CIMR", "False", False, False, None),  # CIMR ignores quality control setting
#         # Valid cases for other input data types
#         ("Other", "True", True, False, None),  # Valid True setting
#         ("Other", "False", False, False, None),  # Valid False setting
#         # Invalid cases for other input data types
#         (
#             "Other",
#             "Invalid",
#             None,
#             True,
#             "Invalid `quality_control` value: Invalid",
#         ),  # Invalid quality control value
#         (
#             "Other",
#             None,
#             None,
#             True,
#             "Invalid `quality_control` value: None",
#         ),  # Missing quality control value
#     ],
# )
# def test_validate_quality_control(
#     input_data_type, quality_control_value, expected, expect_exception, match_message
# ):
#     """
#     Test the `validate_quality_control` method of the ConfigFile class.

#     This test validates:
#     - Correct handling of 'AMSR2' and 'CIMR' input data types (always return False).
#     - Validation of 'True' and 'False' values for other input data types.
#     - Error handling for invalid or missing quality control values.

#     Parameters:
#     - input_data_type: The type of input data (e.g., 'AMSR2', 'CIMR', or 'Other').
#     - quality_control_value: The value of quality_control in the configuration file.
#     - expected: The expected output or None if an exception is expected.
#     - expect_exception: Boolean indicating if a ValueError is expected.
#     - match_message: Substring to match in the raised exception message for invalid cases.

#     Valid cases return the correct boolean value for quality control.
#     Invalid cases raise a ValueError.
#     """
#     # Arrange
#     config_object = Element("config")
#     input_data = SubElement(config_object, "InputData")
#     quality_control = SubElement(input_data, "quality_control")
#     if quality_control_value is not None:
#         quality_control.text = quality_control_value

#     if expect_exception:
#         # Act & Assert: Expect an exception
#         with pytest.raises(ValueError, match=match_message):
#             ConfigFile.validate_quality_control(
#                 config_object=config_object,
#                 quality_control="InputData/quality_control",
#                 input_data_type=input_data_type,
#             )
#     else:
#         # Act: No exception expected
#         result = ConfigFile.validate_quality_control(
#             config_object=config_object,
#             quality_control="InputData/quality_control",
#             input_data_type=input_data_type,
#         )
#         # Assert: Compare result to expected value
#         assert result == expected
