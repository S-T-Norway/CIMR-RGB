import pathlib as pb
import xml.etree.ElementTree as ET
import itertools as it

import numpy as np
import pytest
from unittest.mock import MagicMock

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


@pytest.mark.parametrize(
    "input_data_type_value, split_fore_aft_value, expected_output, expect_exception",
    [
        # Expected to Pass
        ("SMAP", "True", True, False),
        ("CIMR", " false ", False, False),
        ("AMSR2", "TRUE", False, False),
        ("AMSR2", " ", False, False),
        ("AMSR2", "true false ", False, False),
        ("AMSR2", "random string ", False, False),
        ("Another Satellite", "True", True, False),
        # Expected to Fail
        ("SMAP", ["True"], "True", True),
        ("SMAP", ["True", "False"], "True", True),
        ("SMAP", " ", None, True),
        ("SMAP", None, None, True),
        ("CIMR", "True False", "True", True),
        ("CIMR", " ", None, True),
        ("CIMR", None, None, True),
    ],
)
def test_validate_split_fore_aft(
    input_data_type_value, split_fore_aft_value, expected_output, expect_exception
):
    r"""
    Pytest unit test for the `validate_split_fore_aft` method.

    This test function checks the validation of the `split_fore_aft` parameter
    in the XML configuration. It ensures that valid values are correctly parsed
    and invalid values raise appropriate exceptions. The test uses
    `pytest.mark.parametrize` to cover multiple test cases efficiently.

    Parameters
    ----------
    input_data_type_value : str
        The input data type used to determine valid values for `split_fore_aft`.
    split_fore_aft_value : str or list or None
        The value extracted from the XML configuration for `split_fore_aft`.
    expected_output : bool or None
        The expected validated `split_fore_aft` value if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected during validation.

    Raises
    ------
    ValueError
        If `split_fore_aft_value` contains an invalid value or multiple values.
    AttributeError
        If the required XML tag is missing or incorrectly specified.

    Notes
    -----
    - Uses `pytest.raises` to assert expected exceptions.
    - Employs parameterized testing for efficiency.
    - Tests various valid and invalid cases, including trimming whitespace and handling lists
    """

    config_xml = f"""<config><InputData><split_fore_aft>{split_fore_aft_value}</split_fore_aft></InputData></config>"""
    config_object = ET.ElementTree(ET.fromstring(config_xml)).getroot()

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises((ValueError, AttributeError)):
            ConfigFile.validate_split_fore_aft(
                config_object=config_object,
                split_fore_aft="InputData/split_fore_aft",
                input_data_type=input_data_type_value,
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_split_fore_aft(
            config_object=config_object,
            split_fore_aft="InputData/split_fore_aft",
            input_data_type=input_data_type_value,
        )
        assert result == expected_output


@pytest.mark.parametrize(
    "save_to_disk_value, expected_output, expect_exception",
    [
        # Expected to Pass
        (" True", True, False),  # Valid True case
        ("faLse", False, False),  # Valid False case
        # Expected to Fail
        ("InvalidValue", None, True),  # Invalid case
        ("", None, True),  # Empty string, invalid case
        (None, None, True),  # Empty string, invalid case
    ],
)
def test_validate_save_to_disk(save_to_disk_value, expected_output, expect_exception):
    """
    Pytest unit test for the `validate_save_to_disk` method.

    This test function validates the `save_to_disk` parameter in the XML configuration.
    It ensures that valid values are properly parsed as booleans, while invalid values
    raise appropriate exceptions. The test uses `pytest.mark.parametrize` to efficiently
    check multiple cases.

    Parameters
    ----------
    save_to_disk_value : str or None
        The value extracted from the XML configuration for `save_to_disk`.
    expected_output : bool or None
        The expected validated `save_to_disk` value if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected during validation.

    Raises
    ------
    ValueError
        If `save_to_disk_value` is not `TRUE` or `FALSE`.
    AttributeError
        If the required XML tag is missing or incorrectly specified.

    Notes
    -----
    - Uses `pytest.raises` to assert expected exceptions.
    - Employs parameterized testing to cover multiple test cases efficiently.
    - Tests valid and invalid cases, including case variations and missing values.
    """

    config_xml = f"""<config><OutputData><save_to_disk>{save_to_disk_value}</save_to_disk></OutputData></config>"""
    config_object = ET.ElementTree(ET.fromstring(config_xml)).getroot()

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises((ValueError, AttributeError)):
            ConfigFile.validate_save_to_disk(
                config_object=config_object,
                save_to_disk="OutputData/save_to_disk",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_save_to_disk(
            config_object=config_object,
            save_to_disk="OutputData/save_to_disk",
        )
        assert result == expected_output


@pytest.mark.parametrize(
    "search_radius_value, grid_type, input_data_type, expected, expect_exception",
    [
        # Valid Cases
        ("5.0", "L1C", "CIMR", 5000.0, False),  # Defined search_radius (5.0 km)
        (None, "L1C", "AMSR2", None, False),  # L1C with grid_definition
        (None, "L1R", "AMSR2", 62000 / 2, False),  # L1C with grid_definition
        (None, "L1R", "CIMR", 73000 / 2, False),  # L1C with grid_definition
        # Invalid Cases
        (
            "random string",
            "L1C",
            "CIMR",
            None,
            True,
        ),
        (None, "Other Grid", "AMSR2", None, True),
        (None, "L1R", "Other Data Type", None, True),
        (" ", "Other Grid", "AMSR2", None, True),
        ("", "Other Grid", "Other Data Type", None, True),
    ],
)
def test_validate_search_radius(
    search_radius_value,
    grid_type,
    input_data_type,
    expected,
    expect_exception,
):
    """
    Pytest unit test for the `validate_search_radius` method.

    This test function validates the `search_radius` parameter from the XML configuration.
    It ensures that valid numerical values are properly converted to meters, while invalid
    values raise appropriate exceptions. The test uses `pytest.mark.parametrize` to efficiently
    check multiple cases.

    Parameters
    ----------
    search_radius_value : str or None
        The value extracted from the XML configuration for `search_radius`.
    grid_type : str
        The grid type associated with the search radius.
    input_data_type : str
        The input data type influencing the search radius validation.
    expected : float or None
        The expected validated `search_radius` value in meters if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected during validation.

    Raises
    ------
    ValueError
        If `search_radius_value` is not a valid numerical value.

    Notes
    -----
    - Uses `pytest.raises` to assert expected exceptions.
    - Employs parameterized testing to cover multiple test cases efficiently.
    - Tests valid and invalid cases, including empty values and incorrect types.
    """

    config_xml = f"""<config><ReGridderParams><search_radius>{search_radius_value}</search_radius></ReGridderParams></config>"""
    config_object = ET.ElementTree(ET.fromstring(config_xml)).getroot()

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):
            ConfigFile.validate_search_radius(
                config_object=config_object,
                search_radius="ReGridderParams/search_radius",
                grid_type=grid_type,
                input_data_type=input_data_type,
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_search_radius(
            config_object=config_object,
            search_radius="ReGridderParams/search_radius",
            grid_type=grid_type,
            input_data_type=input_data_type,
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "input_data_type_value, band_to_remap, expected, expect_exception",
    [
        # Valid Cases
        ("SMAP", None, (779, 241), False),
        ("CIMR", "L", (74, 691), False),
        ("CIMR", "C", (74, 2747 * 4), False),
        ("CIMR", "X", (74, 2807 * 4), False),
        ("CIMR", "KA", (74, 10395 * 8), False),
        ("CIMR", "KU", (74, 7692 * 8), False),
        # Invalid Cases
        ("UNKNOWN", None, None, True),  # Invalid input_data_type
    ],
)
def test_get_scan_geometry(
    input_data_type_value, band_to_remap, expected, expect_exception
):
    """
    Pytest unit test for the `get_scan_geometry` method.

    This test verifies the correct retrieval of scan geometry for different
    sensor types and frequency bands. It checks both valid and invalid cases.

    Parameters
    ----------
    input_data_type_value : str
        The input data type being tested (e.g., `SMAP`, `CIMR`).
    band_to_remap : str or None
        The frequency band for `CIMR` input data types.
    expected : tuple or None
        The expected result (num_scans, num_earth_samples) if valid, otherwise None.
    expect_exception : bool
        Whether a `ValueError` exception is expected.

    Raises
    ------
    ValueError
        If `input_data_type` is invalid.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Mocks the config object using `MagicMock`.
    """

    # Mock the config object
    config_mock = MagicMock()
    config_mock.input_data_type = input_data_type_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):
            ConfigFile.get_scan_geometry(
                config=config_mock, band_to_remap=band_to_remap
            )
    else:
        # Act: No exception expected
        result = ConfigFile.get_scan_geometry(
            config=config_mock, band_to_remap=band_to_remap
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "input_data_type, variables_to_regrid_value, expected, expect_exception",
    [
        # Valid Cases for SMAP
        (
            "SMAP",
            "bt_h bt_v longitude latitude",
            ["bt_h", "bt_v", "longitude", "latitude"],
            False,
        ),
        (
            "SMAP",
            None,
            [
                "bt_h",
                "bt_v",
                "bt_3",
                "bt_4",
                "processing_scan_angle",
                "longitude",
                "latitude",
                "faraday_rot_angle",
                "nedt_h",
                "nedt_v",
                "nedt_3",
                "nedt_4",
                "regridding_n_samples",
                "regridding_l1b_orphans",
                "acq_time_utc",
                "azimuth",
            ],
            False,
        ),  # Default vars
        ("SMAP", "bt_h bt_v invalid_var", None, True),  # Invalid variable in SMAP
        # Valid Cases for AMSR2
        (
            "AMSR2",
            "bt_h bt_v longitude latitude",
            ["bt_h", "bt_v", "longitude", "latitude"],
            False,
        ),
        (
            "AMSR2",
            None,
            [
                "bt_h",
                "bt_v",
                "longitude",
                "latitude",
                "regridding_n_samples",
                "x_position",
                "y_position",
                "z_position",
                "x_velocity",
                "y_velocity",
                "z_velocity",
                "azimuth",
                "solar_azimuth",
                "acq_time_utc",
            ],
            False,
        ),
        ("AMSR2", "invalid_var", None, True),  # Invalid variable in AMSR2
        # Valid Cases for CIMR
        (
            "CIMR",
            "bt_h bt_v longitude latitude oza",
            ["bt_h", "bt_v", "longitude", "latitude", "oza"],
            False,
        ),
        (
            "CIMR",
            None,
            [
                "bt_h",
                "bt_v",
                "bt_3",
                "bt_4",
                "processing_scan_angle",
                "longitude",
                "latitude",
                "nedt_h",
                "nedt_v",
                "nedt_3",
                "nedt_4",
                "regridding_n_samples",
                "regridding_l1b_orphans",
                "acq_time_utc",
                "azimuth",
                "oza",
            ],
            False,
        ),  # Default vars
        ("CIMR", "invalid_var", None, True),  # Invalid variable in CIMR
        # Invalid input_data_type
        ("INVALID_TYPE", None, None, True),  # Unsupported input_data_type
    ],
)
def test_validate_variables_to_regrid(
    input_data_type, variables_to_regrid_value, expected, expect_exception
):
    r"""
    Pytest unit test for the `validate_variables_to_regrid` method.

    This test function validates the behavior of `validate_variables_to_regrid`, ensuring
    that valid variables are correctly parsed, and invalid values raise appropriate exceptions.

    Parameters
    ----------
    input_data_type : str
        The input data type being tested.
    variables_to_regrid_value : str or None
        The value assigned to the `variables_to_regrid` XML tag.
    expected : list of str or None
        The expected validated output list of variables if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    variables_to_regrid = ET.SubElement(regridder_params, "variables_to_regrid")
    if variables_to_regrid_value is not None:
        variables_to_regrid.text = variables_to_regrid_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):
            ConfigFile.validate_variables_to_regrid(
                config_object=config_object,
                input_data_type=input_data_type,
                variables_to_regrid="ReGridderParams/variables_to_regrid",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_variables_to_regrid(
            config_object=config_object,
            input_data_type=input_data_type,
            variables_to_regrid="ReGridderParams/variables_to_regrid",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "regridding_algorithm, max_neighbours_value, expected, expect_exception",
    [
        # Case: Regridding algorithm is NN
        ("NN", None, 1, False),  # Always return 1 regardless of max_neighbours
        # Case: Valid max_neighbours value
        ("Other", "500", 500, False),  # Valid integer string
        ("Other", "1000", 1000, False),  # Valid integer string
        # Case: max_neighbours not defined
        ("Other", None, 1000, False),  # Defaults to 1000
        # Case: Invalid max_neighbours value
        ("Other", "invalid", None, True),  # Non-integer string
        ("Other", "", None, True),  # Empty string
        ("Other", " ", None, True),  # Space-only string
    ],
)
def test_validate_max_neighbours(
    regridding_algorithm, max_neighbours_value, expected, expect_exception
):
    r"""
    Pytest unit test for the `validate_max_neighbours` method.

    This test function validates the `max_neighbours` parameter, ensuring that
    valid values are correctly parsed and invalid values raise appropriate exceptions.

    Parameters
    ----------
    regridding_algorithm : str
        The regridding algorithm being tested (`NN` always returns `1`).
    max_neighbours_value : str or None
        The value assigned to the `max_neighbours` XML tag.
    expected : int or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Tests cases where `max_neighbours` is valid, missing, or invalid.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    max_neighbours = ET.SubElement(regridder_params, "max_neighbours")
    if max_neighbours_value is not None:
        max_neighbours.text = max_neighbours_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):
            ConfigFile.validate_max_neighbours(
                config_object=config_object,
                max_neighbours="ReGridderParams/max_neighbours",
                regridding_algorithm=regridding_algorithm,
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_max_neighbours(
            config_object=config_object,
            max_neighbours="ReGridderParams/max_neighbours",
            regridding_algorithm=regridding_algorithm,
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "source_antenna_method_value, expected, expect_exception",
    [
        # Valid cases
        ("gaussian", "gaussian", False),
        ("instrument", "instrument", False),
        ("gaussian_projected", "gaussian_projected", False),
        ("GAUSSIAN", "gaussian", False),  # Invalid string
        ("Gaussian", "gaussian", False),  # Invalid string
        # Default case
        (None, "instrument", False),  # Defaults to 'real' if value is None
        ("", "instrument", False),  # Empty string
        (" ", "instrument", False),  # Space-only string
        # Invalid cases
        ("invalid_value", None, True),  # Invalid string
    ],
)
def test_validate_source_antenna_method(
    source_antenna_method_value, expected, expect_exception
):
    r"""
    Pytest unit test for the `validate_source_antenna_method` method.

    This test function validates the behavior of `validate_source_antenna_method`, ensuring
    that valid values are correctly parsed, and invalid values raise appropriate exceptions.

    The function is designed to be case insensitive, meaning input values will be converted to lowercase.

    Test Scenarios:
    --------------
    - Valid `source_antenna_method` values (e.g., `'gaussian'`, `'instrument'`, `'gaussian_projected'`).
    - Case-insensitive inputs should be correctly mapped to their lowercase equivalents.
    - Missing `source_antenna_method` values should default to `'instrument'`.
    - Invalid values should raise a `ValueError`.

    Parameters
    ----------
    source_antenna_method_value : str or None
        The value assigned to the `source_antenna_method` XML tag.
    expected : str or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures empty or missing values default to `'instrument'`.
    - Tests case insensitivity by checking uppercase and mixed-case inputs.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    source_antenna = ET.SubElement(regridder_params, "source_antenna_method")
    if source_antenna_method_value is not None:
        source_antenna.text = source_antenna_method_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(
            ValueError  # , match="Invalid antenna method. Check Configuration File."
        ):
            ConfigFile.validate_source_antenna_method(
                config_object=config_object,
                source_antenna_method="ReGridderParams/source_antenna_method",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_source_antenna_method(
            config_object=config_object,
            source_antenna_method="ReGridderParams/source_antenna_method",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "target_antenna_method_value, expected, expect_exception",
    [
        # Valid cases
        ("gaussian", "gaussian", False),
        ("instrument", "instrument", False),
        ("gaussian_projected", "gaussian_projected", False),
        ("GAUSSIAN", "gaussian", False),  # Invalid string
        ("Gaussian", "gaussian", False),  # Invalid string
        # Default case
        (None, "instrument", False),  # Defaults to 'real' if value is None
        ("", "instrument", False),  # Empty string
        (" ", "instrument", False),  # Space-only string
        # Invalid cases
        ("invalid_value", None, True),  # Invalid string
    ],
)
def test_validate_target_antenna_method(
    target_antenna_method_value, expected, expect_exception
):
    r"""
    Pytest unit test for the `validate_target_antenna_method` method.

    This test ensures that `validate_target_antenna_method` correctly handles
    valid, default, and invalid cases, while treating input values case-insensitively.

    Test Scenarios
    --------------
    - Valid `target_antenna_method` values (e.g., `'gaussian'`, `'instrument'`, `'gaussian_projected'`).
    - Case-insensitive inputs should be correctly mapped to their lowercase equivalents.
    - Missing or empty values should default to `'instrument'`.
    - Invalid values should raise a `ValueError`.

    Parameters
    ----------
    target_antenna_method_value : str or None
        The value assigned to the `target_antenna_method` XML tag.
    expected : str or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures empty or missing values default to `'instrument'`.
    - Tests case insensitivity by checking uppercase and mixed-case inputs.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    target_antenna = ET.SubElement(regridder_params, "target_antenna_method")
    if target_antenna_method_value is not None:
        target_antenna.text = target_antenna_method_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(
            ValueError  # , match="Invalid antenna method. Check Configuration File."
        ):
            ConfigFile.validate_target_antenna_method(
                config_object=config_object,
                target_antenna_method="ReGridderParams/target_antenna_method",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_target_antenna_method(
            config_object=config_object,
            target_antenna_method="ReGridderParams/target_antenna_method",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "source_antenna_threshold_value, expected, expect_exception",
    [
        # Valid cases
        ("9.5", 9.5, False),  # Float value
        ("10", 10.0, False),  # Integer value as string
        ("0", 0.0, False),  # Zero value
        ("-5", -5.0, False),  # Negative value
        # Default case
        (None, None, False),  # Returns None if value is not provided
        ("", None, False),  # Empty string
        (" ", None, False),  # Space-only string
        # Invalid cases
        ("invalid", None, True),  # Non-numeric string
        ("a9.5", 9.5, True),  # Typo in Float value
    ],
)
def test_validate_source_antenna_threshold(
    source_antenna_threshold_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_source_antenna_threshold` method.

    This test ensures that `validate_source_antenna_threshold` correctly handles
    valid, default, and invalid cases, while treating input values as numeric.

    Test Scenarios
    --------------
    - Valid `source_antenna_threshold` values (e.g., positive/negative floats and integers).
    - Missing or empty values should return `None`.
    - Invalid values (non-numeric) should raise a `ValueError`.

    Parameters
    ----------
    source_antenna_threshold_value : str or None
        The value assigned to the `source_antenna_threshold` XML tag.
    expected : float or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures empty or missing values default to `None`.
    - Tests valid float and integer conversions.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    source_antenna_threshold = ET.SubElement(
        regridder_params, "source_antenna_threshold"
    )

    if source_antenna_threshold_value is not None:
        source_antenna_threshold.text = source_antenna_threshold_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(
            ValueError, match="Invalid antenna threshold"
        ):  # . Check Configuration File."):
            ConfigFile.validate_source_antenna_threshold(
                config_object=config_object,
                source_antenna_threshold="ReGridderParams/source_antenna_threshold",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_source_antenna_threshold(
            config_object=config_object,
            source_antenna_threshold="ReGridderParams/source_antenna_threshold",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "target_antenna_threshold_value, expected, expect_exception",
    [
        # Valid cases
        ("9.5", 9.5, False),  # Float value
        ("10", 10.0, False),  # Integer value as string
        ("0", 0.0, False),  # Zero value
        ("-5", -5.0, False),  # Negative value
        # Default case
        (None, 9.0, False),  # Defaults to 9.0 if value is None
        ("", 9.0, False),  # Empty string
        (" ", 9.0, False),  # Space-only string
        # Invalid cases
        ("invalid", None, True),  # Non-numeric string
        ("a9.5", 9.5, True),  # Typo in Float value
    ],
)
def test_validate_target_antenna_threshold(
    target_antenna_threshold_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_target_antenna_threshold` method.

    This test ensures that `validate_target_antenna_threshold` correctly handles
    valid, default, and invalid cases, while treating input values as numeric.

    Test Scenarios
    --------------
    - Valid `target_antenna_threshold` values (e.g., positive/negative floats and integers).
    - Missing or empty values should return the default of `9.0`.
    - Invalid values (non-numeric) should raise a `ValueError`.

    Parameters
    ----------
    target_antenna_threshold_value : str or None
        The value assigned to the `target_antenna_threshold` XML tag.
    expected : float or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures empty or missing values default to `9.0`.
    - Tests valid float and integer conversions.

    Examples
    --------
    >>> test_validate_target_antenna_threshold("9.5", 9.5, False)
    >>> test_validate_target_antenna_threshold(None, 9.0, False)
    >>> test_validate_target_antenna_threshold("invalid", None, True)
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    target_antenna_threshold = ET.SubElement(
        regridder_params, "target_antenna_threshold"
    )
    if target_antenna_threshold_value is not None:
        target_antenna_threshold.text = target_antenna_threshold_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(
            ValueError, match="Invalid antenna threshold:"
        ):  # . Check Configuration File."):
            ConfigFile.validate_target_antenna_threshold(
                config_object=config_object,
                target_antenna_threshold="ReGridderParams/target_antenna_threshold",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_target_antenna_threshold(
            config_object=config_object,
            target_antenna_threshold="ReGridderParams/target_antenna_threshold",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "polarisation_method_value, expected, expect_exception",
    [
        # Valid cases
        ("scalar", "scalar", False),
        ("mueller", "mueller", False),
        # Default case
        (None, "scalar", False),  # Defaults to 'scalar' if value is None
        ("", "scalar", False),  # Empty string
        (" ", "scalar", False),  # Space-only string
        # Invalid cases
        ("invalid", None, True),  # Invalid string
    ],
)
def test_validate_polarisation_method(
    polarisation_method_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_polarisation_method` method.

    This test ensures that `validate_polarisation_method` correctly handles
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `polarisation_method` values (e.g., `'scalar'`, `'mueller'`).
    - Missing or empty values should return the default of `'scalar'`.
    - Invalid values (non-valid strings) should raise a `ValueError`.

    Parameters
    ----------
    polarisation_method_value : str or None
        The value assigned to the `polarisation_method` XML tag.
    expected : str or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures empty or missing values default to `'scalar'`.
    - Tests valid string inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    polarisation = ET.SubElement(regridder_params, "polarisation_method")
    if polarisation_method_value is not None:
        polarisation.text = polarisation_method_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(
            ValueError, match="Invalid polarisation method:"
        ):  # . Check Configuration File."):
            ConfigFile.validate_polarisation_method(
                config_object=config_object,
                polarisation_method="ReGridderParams/polarisation_method",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_polarisation_method(
            config_object=config_object,
            polarisation_method="ReGridderParams/polarisation_method",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "input_data_type, boresight_shift_value, expected, expect_exception",
    [
        # Case: Input data type is not SMAP
        ("AMSR2", None, False, False),  # Always return False for non-SMAP
        ("CIMR", "true", False, False),  # Always return False for non-SMAP
        # Case: Input data type is SMAP
        ("SMAP", "True", True, False),  # Valid True case
        ("SMAP", "FALSE", False, False),  # Valid False case
        ("SMAP", None, False, False),  # Defaults to False if value is None
        ("SMAP", "", False, False),  # Empty string
        ("SMAP", " ", False, False),  # Space-only string
        # Invalid cases
        ("SMAP", "invalid_value", None, True),  # Invalid string
    ],
)
def test_validate_boresight_shift(
    input_data_type, boresight_shift_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_boresight_shift` method.

    This test ensures that `validate_boresight_shift` correctly processes
    valid, default, and invalid cases, while treating input values case insensitively.

    Test Scenarios
    --------------
    - Valid `boresight_shift` values (e.g., `'True'`, `'False'`).
    - Missing or empty values should return the default of `False`.
    - Non-SMAP input data types should always return `False`.
    - Invalid values should raise a `ValueError`.

    Parameters
    ----------
    input_data_type : str
        The type of input data (e.g., `'SMAP'`, `'AMSR2'`, `'CIMR'`).
    boresight_shift_value : str or None
        The value assigned to the `boresight_shift` XML tag.
    expected : bool or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures empty or missing values default to `False`.
    - Tests case insensitivity by verifying mixed-case inputs.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    boresight_shift = ET.SubElement(regridder_params, "boresight_shift")
    if boresight_shift_value is not None:
        boresight_shift.text = boresight_shift_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(
            ValueError  # , match="Invalid boresight shift:"
        ):  # . Check Configuration File."):
            ConfigFile.validate_boresight_shift(
                config_object=config_object,
                boresight_shift="ReGridderParams/boresight_shift",
                input_data_type=input_data_type,
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_boresight_shift(
            config_object=config_object,
            boresight_shift="ReGridderParams/boresight_shift",
            input_data_type=input_data_type,
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "reduced_grid_inds_value, expected, expect_exception",
    [
        # Valid cases
        ("0 10 0 10", [0, 10, 0, 10], False),
        ("5 15 3 12", [5, 15, 3, 12], False),
        ("", None, False),  # Empty string
        # Missing case
        (None, None, False),
        # Invalid cases
        ("-1 10 0 10", None, True),  # Negative grid_row_min
        ("0 -10 0 10", None, True),  # Negative grid_row_max
        ("0 10 -5 10", None, True),  # Negative grid_col_min
        ("0 10 0 -10", None, True),  # Negative grid_col_max
        ("10 5 0 10", None, True),  # grid_row_min > grid_row_max
        ("0 10 12 5", None, True),  # grid_col_min > grid_col_max
        ("0 10", None, True),  # Insufficient values
        ("invalid input here", None, True),  # Non-numeric values
    ],
)
def test_validate_reduced_grid_inds(
    reduced_grid_inds_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_reduced_grid_inds` method.

    This test ensures that `validate_reduced_grid_inds` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `reduced_grid_inds` values should return a list of four integers.
    - Missing or empty values should return `None`.
    - Negative values should raise a `ValueError`.
    - Row/column minimums exceeding maximums should raise a `ValueError`.
    - Non-numeric values should raise a `ValueError`.

    Parameters
    ----------
    reduced_grid_inds_value : str or None
        The value assigned to the `reduced_grid_inds` XML tag.
    expected : list of int or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures empty or missing values default to `None`.
    - Tests valid integer inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    grid_params = ET.SubElement(config_object, "GridParams")
    reduced_grid_inds = ET.SubElement(grid_params, "reduced_grid_inds")
    if reduced_grid_inds_value is not None:
        reduced_grid_inds.text = reduced_grid_inds_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):  # , match="Invalid `reduced_grid_inds`:"):
            ConfigFile.validate_reduced_grid_inds(
                config_object=config_object,
                reduced_grid_inds="GridParams/reduced_grid_inds",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_reduced_grid_inds(
            config_object=config_object,
            reduced_grid_inds="GridParams/reduced_grid_inds",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "source_gaussian_params_value, expected, expect_exception",
    [
        # Valid cases
        ("1.5 2.5", [1.5, 2.5], False),  # Two valid floats
        ("0 10", [0.0, 10.0], False),  # Integer values as strings
        ("-5.5 3.2", [-5.5, 3.2], False),  # Negative and positive floats
        # Invalid cases
        (None, None, True),  # Missing value
        ("1.5", None, True),  # Only one parameter
        ("1.5 2.5 3.5", None, True),  # Too many parameters
        ("invalid 2.5", None, True),  # Non-numeric parameter
        ("1.5 invalid", None, True),  # Non-numeric parameter
        ("", None, True),  # Empty string
        (" ", None, True),  # Space-only string
    ],
)
def test_validate_source_gaussian_params(
    source_gaussian_params_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_source_gaussian_params` method.

    This test ensures that `validate_source_gaussian_params` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `source_gaussian_params` values should return a list of two float values.
    - Missing or empty values should raise a `ValueError`.
    - Inputs with incorrect numbers of parameters should raise a `ValueError`.
    - Non-numeric values should raise a `ValueError`.

    Parameters
    ----------
    source_gaussian_params_value : str or None
        The value assigned to the `source_gaussian_params` XML tag.
    expected : list of float or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only two valid numeric values are accepted.
    - Tests valid float inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    source_gaussian_params = ET.SubElement(regridder_params, "source_gaussian_params")
    if source_gaussian_params_value is not None:
        source_gaussian_params.text = source_gaussian_params_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(
            ValueError  # , match="Invalid|Missing "
        ):  # source Gaussian parameters"):#|Missing source Gaussian parameters|Invalid parameter"):
            ConfigFile.validate_source_gaussian_params(
                config_object=config_object,
                source_gaussian_params="ReGridderParams/source_gaussian_params",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_source_gaussian_params(
            config_object=config_object,
            source_gaussian_params="ReGridderParams/source_gaussian_params",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "target_gaussian_params_value, expected, expect_exception",
    [
        # Valid cases
        ("1.5 2.5", [1.5, 2.5], False),  # Two valid floats
        ("0 10", [0.0, 10.0], False),  # Integer values as strings
        ("-5.5 3.2", [-5.5, 3.2], False),  # Negative and positive floats
        # Invalid cases
        (None, None, True),  # Missing value
        ("1.5", None, True),  # Only one parameter
        ("1.5 2.5 3.5", None, True),  # Too many parameters
        ("invalid 2.5", None, True),  # Non-numeric parameter
        ("1.5 invalid", None, True),  # Non-numeric parameter
        ("", None, True),  # Empty string
        (" ", None, True),  # Space-only string
    ],
)
def test_validate_target_gaussian_params(
    target_gaussian_params_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_target_gaussian_params` method.

    This test ensures that `validate_target_gaussian_params` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `target_gaussian_params` values should return a list of two float values.
    - Missing or empty values should raise a `ValueError`.
    - Inputs with incorrect numbers of parameters should raise a `ValueError`.
    - Non-numeric values should raise a `ValueError`.

    Parameters
    ----------
    target_gaussian_params_value : str or None
        The value assigned to the `target_gaussian_params` XML tag.
    expected : list of float or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only two valid numeric values are accepted.
    - Tests valid float inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    target_gaussian_params = ET.SubElement(regridder_params, "target_gaussian_params")
    if target_gaussian_params_value is not None:
        target_gaussian_params.text = target_gaussian_params_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):  # , match="Invalid|Missing"):
            ConfigFile.validate_target_gaussian_params(
                config_object=config_object,
                target_gaussian_params="ReGridderParams/target_gaussian_params",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_target_gaussian_params(
            config_object=config_object,
            target_gaussian_params="ReGridderParams/target_gaussian_params",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "rsir_iteration_value, expected, expect_exception",
    [
        # Valid cases
        ("10", 10, False),  # Valid integer
        ("0", 0, False),  # Edge case: Zero iterations
        ("1", 1, False),  # Edge case: Minimum positive iterations
        # Invalid cases
        (None, None, True),  # Missing value
        ("", None, True),  # Empty string
        ("invalid", None, True),  # Non-numeric string
        ("1.5", None, True),  # Float value
        ("-5", None, True),  # Negative integer (assuming it's invalid in this context)
    ],
)
def test_validate_rsir_iteration(rsir_iteration_value, expected, expect_exception):
    """
    Pytest unit test for the `validate_rsir_iteration` method.

    This test ensures that `validate_rsir_iteration` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `rsir_iteration` values should return a non-negative integer.
    - Missing or empty values should raise a `ValueError`.
    - Negative or non-numeric values should raise a `ValueError`.

    Parameters
    ----------
    rsir_iteration_value : str or None
        The value assigned to the `rsir_iteration` XML tag.
    expected : int or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only non-negative integers are accepted.
    - Tests valid integer inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    rsir_iteration = ET.SubElement(regridder_params, "rsir_iteration")
    if rsir_iteration_value is not None:
        rsir_iteration.text = rsir_iteration_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):  # , match="Invalid|Missing|rSIR"):
            ConfigFile.validate_rsir_iteration(
                config_object=config_object,
                rsir_iteration="ReGridderParams/rsir_iteration",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_rsir_iteration(
            config_object=config_object,
            rsir_iteration="ReGridderParams/rsir_iteration",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "max_number_iteration_value, expected, expect_exception",
    [
        # Valid cases
        ("10", 10, False),  # Valid integer
        ("0", 0, False),  # Edge case: Zero iterations
        ("1", 1, False),  # Edge case: Minimum positive iterations
        # Invalid cases
        (None, None, True),  # Missing value
        ("", None, True),  # Empty string
        ("invalid", None, True),  # Non-numeric string
        ("1.5", None, True),  # Float value
        ("-5", None, True),  # Negative integer
    ],
)
def test_validate_max_number_iteration(
    max_number_iteration_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_max_number_iteration` method.

    This test ensures that `validate_max_number_iteration` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `max_number_iteration` values should return a non-negative integer.
    - Missing or empty values should raise a `ValueError`.
    - Negative or non-numeric values should raise a `ValueError`.

    Parameters
    ----------
    max_number_iteration_value : str or None
        The value assigned to the `max_number_iteration` XML tag.
    expected : int or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only non-negative integers are accepted.
    - Tests valid integer inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    max_number_iteration = ET.SubElement(regridder_params, "max_number_iteration")
    if max_number_iteration_value is not None:
        max_number_iteration.text = max_number_iteration_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):  # , match="Invalid|Missing|Maximum"):
            ConfigFile.validate_max_number_iteration(
                config_object=config_object,
                max_number_iteration="ReGridderParams/max_number_iteration",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_max_number_iteration(
            config_object=config_object,
            max_number_iteration="ReGridderParams/max_number_iteration",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "relative_tolerance_value, expected, expect_exception",
    [
        # Valid cases
        ("0.01", 0.01, False),  # Valid positive float
        ("0", 0.0, False),  # Edge case: Zero tolerance
        ("1.5", 1.5, False),  # Valid larger float value
        # Invalid cases
        (None, None, True),  # Missing value
        ("", None, True),  # Empty string
        ("invalid", None, True),  # Non-numeric string
        ("-0.01", None, True),  # Negative float
        ("-1", None, True),  # Negative integer
    ],
)
def test_validate_relative_tolerance(
    relative_tolerance_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_relative_tolerance` method.

    This test ensures that `validate_relative_tolerance` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `relative_tolerance` values should return a non-negative float.
    - Missing or empty values should raise a `ValueError`.
    - Negative or non-numeric values should raise a `ValueError`.

    Parameters
    ----------
    relative_tolerance_value : str or None
        The value assigned to the `relative_tolerance` XML tag.
    expected : float or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only non-negative floats are accepted.
    - Tests valid float inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    relative_tolerance = ET.SubElement(regridder_params, "relative_tolerance")
    if relative_tolerance_value is not None:
        relative_tolerance.text = relative_tolerance_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):  # , match="Invalid|Missing|Relative"):
            ConfigFile.validate_relative_tolerance(
                config_object=config_object,
                relative_tolerance="ReGridderParams/relative_tolerance",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_relative_tolerance(
            config_object=config_object,
            relative_tolerance="ReGridderParams/relative_tolerance",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "regularization_parameter_value, expected, expect_exception",
    [
        # Valid cases
        ("0.1", 0.1, False),  # Positive float
        ("0", 0.0, False),  # Zero value
        ("10", 10.0, False),  # Positive integer
        ("-0.1", -0.1, False),  # Negative float
        ("-5", -5.0, False),  # Negative integer
        # Invalid cases
        (None, None, True),  # Missing value
        ("", None, True),  # Empty string
        ("invalid", None, True),  # Non-numeric string
    ],
)
def test_validate_regularization_parameter(
    regularization_parameter_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_regularization_parameter` method.

    This test ensures that `validate_regularization_parameter` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `regularization_parameter` values should return a float.
    - Missing or empty values should raise a `ValueError`.
    - Non-numeric values should raise a `ValueError`.

    Parameters
    ----------
    regularization_parameter_value : str or None
        The value assigned to the `regularization_parameter` XML tag.
    expected : float or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only valid float values are accepted.
    - Tests valid float inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    regularization_parameter = ET.SubElement(
        regridder_params, "regularization_parameter"
    )
    if regularization_parameter_value is not None:
        regularization_parameter.text = regularization_parameter_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):  # , match="Invalid|Missing"):
            ConfigFile.validate_regularization_parameter(
                config_object=config_object,
                regularization_parameter="ReGridderParams/regularization_parameter",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_regularization_parameter(
            config_object=config_object,
            regularization_parameter="ReGridderParams/regularization_parameter",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "MRF_grid_definition_value, expected, expect_exception",
    [
        # Valid cases
        ("EASE2_G3km", "EASE2_G3km", False),
        ("EASE2_G1km", "EASE2_G1km", False),
        (" EASE2_S36km", "EASE2_S36km", False),
        ("EASE2_G3km ", "EASE2_G3km", False),  # Trailing space (not an exact match)
        # Invalid cases
        (None, None, True),  # Missing value
        ("", None, True),  # Empty string
        ("invalid_grid", None, True),  # Invalid grid definition
    ],
)
def test_validate_MRF_grid_definition(
    MRF_grid_definition_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_MRF_grid_definition` method.

    This test ensures that `validate_MRF_grid_definition` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `MRF_grid_definition` values should return a recognized grid definition.
    - Missing or empty values should raise a `ValueError`.
    - Invalid values should raise a `ValueError`.

    Parameters
    ----------
    MRF_grid_definition_value : str or None
        The value assigned to the `MRF_grid_definition` XML tag.
    expected : str or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only predefined grid definitions are accepted.
    - Tests valid inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    MRF_grid_definition = ET.SubElement(regridder_params, "MRF_grid_definition")

    if MRF_grid_definition_value is not None:
        MRF_grid_definition.text = MRF_grid_definition_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):  # , match="Invalid|Missing"):
            ConfigFile.validate_MRF_grid_definition(
                config_object=config_object,
                MRF_grid_definition="ReGridderParams/MRF_grid_definition",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_MRF_grid_definition(
            config_object=config_object,
            MRF_grid_definition="ReGridderParams/MRF_grid_definition",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "MRF_projection_definition_value, expected, expect_exception, match_message",
    [
        # Valid cases
        ("G", "G", False, None),  # Global projection
        ("N", "N", False, None),  # Northern projection
        ("S", "S", False, None),  # Southern projection
        (" G ", "G", False, None),  # Trailing/leading whitespace
        # Invalid cases
        (
            None,
            None,
            True,
            "Missing or blank MRF projection definition in the configuration file",
        ),  # Missing value
        (
            "",
            None,
            True,
            "Missing or blank MRF projection definition in the configuration file",
        ),  # Empty string
        (
            "invalid",
            None,
            True,
            "Invalid Projection Definition",
        ),  # Invalid projection definition
    ],
)
def test_validate_MRF_projection_definition(
    MRF_projection_definition_value, expected, expect_exception, match_message
):
    """
    Pytest unit test for the `validate_MRF_projection_definition` method.

    This test ensures that `validate_MRF_projection_definition` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `MRF_projection_definition` values should return a recognized projection ('G', 'N', or 'S').
    - Missing or empty values should raise a `ValueError`.
    - Invalid values should raise a `ValueError`.

    Parameters
    ----------
    MRF_projection_definition_value : str or None
        The value assigned to the `MRF_projection_definition` XML tag.
    expected : str or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.
    match_message : str
        The expected exception message for invalid cases.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only predefined projection definitions are accepted.
    - Tests valid inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    MRF_projection_definition = ET.SubElement(
        regridder_params, "MRF_projection_definition"
    )
    if MRF_projection_definition_value is not None:
        MRF_projection_definition.text = MRF_projection_definition_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError, match=match_message):
            ConfigFile.validate_MRF_projection_definition(
                config_object=config_object,
                MRF_projection_definition="ReGridderParams/MRF_projection_definition",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_MRF_projection_definition(
            config_object=config_object,
            MRF_projection_definition="ReGridderParams/MRF_projection_definition",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "bg_smoothing_value, expected, expect_exception",
    [
        # Valid cases
        ("1.5", 1.5, False),  # Positive float
        ("0", 0.0, False),  # Zero
        ("-2.3", -2.3, False),  # Negative float
        (None, 0.0, False),  # Missing value, should default to 0
        (" 3.7 ", 3.7, False),  # Float with leading/trailing whitespace
        ("", 0.0, False),  # Empty string
        (" ", 0.0, False),  # Space-only string
        # Invalid cases
        ("invalid", None, True),  # Non-numeric string
    ],
)
def test_validate_bg_smoothing(
    bg_smoothing_value,
    expected,
    expect_exception,
):
    """
    Pytest unit test for the `validate_bg_smoothing` method.

    This test ensures that `validate_bg_smoothing` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - Valid `bg_smoothing` values should return a float.
    - Missing values default to `0.0`.
    - Invalid values should raise a `ValueError`.

    Parameters
    ----------
    bg_smoothing_value : str or None
        The value assigned to the `bg_smoothing` XML tag.
    expected : float or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures only valid floats are accepted.
    - Tests valid inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    regridder_params = ET.SubElement(config_object, "ReGridderParams")
    bg_smoothing = ET.SubElement(regridder_params, "bg_smoothing")
    if bg_smoothing_value is not None:
        bg_smoothing.text = bg_smoothing_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):  # , match=match_message):
            ConfigFile.validate_bg_smoothing(
                config_object=config_object,
                bg_smoothing="ReGridderParams/bg_smoothing",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_bg_smoothing(
            config_object=config_object,
            bg_smoothing="ReGridderParams/bg_smoothing",
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "input_data_type, quality_control_value, expected, expect_exception",
    [
        # Input data types that always return False
        ("AMSR2", "True", False, False),  # AMSR2 ignores quality control setting
        ("CIMR", "False", False, False),  # CIMR ignores quality control setting
        # Valid cases for other input data types
        ("Other", "True", True, False),  # Valid True setting
        ("Other", "False", False, False),  # Valid False setting
        ("Other", "false", False, False),
        # Invalid cases for other input data types
        ("Other", "Invalid", None, True),  # Invalid quality control value
        ("Other", None, None, True),
    ],
)
def test_validate_quality_control(
    input_data_type, quality_control_value, expected, expect_exception
):
    """
    Pytest unit test for the `validate_quality_control` method.

    This test ensures that `validate_quality_control` correctly processes
    valid, default, and invalid cases.

    Test Scenarios
    --------------
    - `AMSR2` and `CIMR` always return `False`.
    - Valid values ('True' and 'False', case insensitive) return the corresponding boolean.
    - Invalid or missing values raise a `ValueError`.

    Parameters
    ----------
    input_data_type : str
        The type of input data (e.g., 'AMSR2', 'CIMR', or another type).
    quality_control_value : str or None
        The value assigned to the `quality_control` XML tag.
    expected : bool or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures `AMSR2` and `CIMR` always return `False`.
    - Tests valid inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    input_data = ET.SubElement(config_object, "InputData")
    quality_control = ET.SubElement(input_data, "quality_control")

    if quality_control_value is not None:
        quality_control.text = quality_control_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):
            ConfigFile.validate_quality_control(
                config_object=config_object,
                quality_control="InputData/quality_control",
                input_data_type=input_data_type,
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_quality_control(
            config_object=config_object,
            quality_control="InputData/quality_control",
            input_data_type=input_data_type,
        )
        # Assert: Compare result to expected value
        assert result == expected


@pytest.mark.parametrize(
    "antenna_pattern_uncertainty_value, expected, expect_exception",
    [
        # Valid cases
        ("1.5", 1.5, False),  # Positive float
        ("0", 0.0, False),  # Zero
        ("-2.3", -2.3, False),  # Negative float
        (None, 0.0, False),  # Missing value, should default to 0.0
        (" 3.7 ", 3.7, False),  # Float with leading/trailing whitespace
        ("", 0.0, False),  # Empty string should return default 0.0
        (" ", 0.0, False),  # Space-only string should return default 0.0
        # Invalid cases
        ("invalid", None, True),  # Non-numeric string
    ],
)
def test_validate_antenna_pattern_uncertainty(
    antenna_pattern_uncertainty_value, expected, expect_exception
):
    """
    Test the `validate_antenna_pattern_uncertainty` method of the ConfigFile class.

    This test validates:
    - Correct behavior for valid `antenna_pattern_uncertainty` values.
    - Default behavior for missing values (defaults to 0.0).
    - Error handling for invalid or non-numeric values.

    Parameters
    ----------
    antenna_pattern_uncertainty_value : str or None
        The value of `antenna_pattern_uncertainty` in the configuration file.
    expected : float or None
        The expected validated output if valid, otherwise None.
    expect_exception : bool
        Indicates whether an exception is expected.

    Notes
    -----
    - Uses `pytest.raises` to verify exceptions.
    - Ensures default return value is `0.0` for missing entries.
    - Tests valid float inputs and invalid cases.
    """

    # Arrange
    config_object = ET.Element("config")
    uncertainty_params = ET.SubElement(config_object, "Uncertainty")
    antenna_pattern_uncertainty = ET.SubElement(
        uncertainty_params, "antenna_pattern_uncertainty"
    )

    if antenna_pattern_uncertainty_value is not None:
        antenna_pattern_uncertainty.text = antenna_pattern_uncertainty_value

    if expect_exception:
        # Act & Assert: Expect an exception
        with pytest.raises(ValueError):
            ConfigFile.validate_antenna_pattern_uncertainty(
                config_object=config_object,
                antenna_pattern_uncertainty="Uncertainty/antenna_pattern_uncertainty",
            )
    else:
        # Act: No exception expected
        result = ConfigFile.validate_antenna_pattern_uncertainty(
            config_object=config_object,
            antenna_pattern_uncertainty="Uncertainty/antenna_pattern_uncertainty",
        )
        # Assert: Compare result to expected value
        assert result == expected
