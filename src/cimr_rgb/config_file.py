"""
This script reads the configuration file and validates the chosen parameters.
"""

import re
import sys
import pathlib as pb
import logging
import typing
import importlib.resources as pkg_resources
import json
import datetime
from os import path, getcwd
from xml.etree.ElementTree import ParseError, parse
from operator import truediv

from numpy import sqrt

from cimr_rgb import configs  # Import the package where your JSON is located
from cimr_rgb.grid_generator import GRIDS
from cimr_rgb.rgb_logging import RGBLogging
import cimr_grasp.grasp_io as grasp_io


class ConfigFile:
    """
    This class reads the configuration file and validates the chosen parameters.

    Attributes:
    -----------
    config_file_path: str
        Path to the configuration file

    Methods:
    -----------
    read_config(config_file_path)
        Reads the configuration file and returns the root element

    validate_{configuration_variable}
        Validates the configuration variable and returns the value if valid

    """

    def __init__(self, config_file_path: pb.Path | str):
        """
        Initializes the ConfigFile class and stores validated
        config parameters as class attributes.

        Parameters
        ----------
        config_file_path: Path or str
        """

        # converting the relative path into absolute one if needed
        if not pb.Path(config_file_path).is_absolute():
            config_file_path = pb.Path(config_file_path).resolve()

        # print(f"My name {config_file_path.stem}")

        config_object, tree = self.read_config(config_file_path)

        self.output_path = self.validate_output_directory_path(
            config_object, output_path="OutputData/output_path", logger=None
        )

        # TODO: Get the time stamp which will be propagated to create
        #       name for log files, config files and data products
        # self.timestamp = config_object.find("OutputData/timestamp").text
        self.suffix = config_object.find("OutputData/suffix").text
        self.timestamp_fmt = config_object.find("OutputData/timestamp_fmt").text
        # if self.timestamp is None or self.timestamp.strip() == "":
        # Getting the current time stamp to propagate into the software
        self.timestamp = datetime.datetime.now()

        # Format the date and time as "YYYY-MM-DD_HH-MM-SS"
        self.timestamp = self.timestamp.strftime(self.timestamp_fmt)
        # config_object.find("OutputData/timestamp").text = timestamp_elem

        # l1c_utc_time = datetime.datetime.now()

        # Format the date and time as "YYYY-MM-DD_HH-MM-SS"
        # self.timestamp = l1c_utc_time.strftime(self.timestamp_fmt)

        # TODO: Get the time stamp which will be propagated to create
        #       name for log files, config files and data products
        # self.timestamp = config_object.find("OutputData/timestamp").text
        # self.timestamp_fmt = config_object.find("OutputData/timestamp_fmt").text

        # if self.timestamp is None or self.timestamp.strip() == "":
        #     # Getting the current time stamp to propagate into the software
        #     self.timestamp = datetime.datetime.now()

        #     # Format the date and time as "YYYY-MM-DD_HH-MM-SS"
        #     self.timestamp = self.timestamp.strftime(self.timestamp_fmt)
        #     # config_object.find("OutputData/timestamp").text = timestamp_elem

        # TODO: Put this into its own validation method?
        # -----------
        # Path to logger_config.json
        # -----------
        # Creating output directory for logs (to save log files there)
        logdir = pb.Path(self.output_path).joinpath("logs")
        # Create the directory structure if necessary
        grasp_io.rec_create_dir(path=logdir)

        # Getting the parh to config file in JSON format
        self.logpar_config = config_object.find("LoggingParams/config_path").text

        # If the paramter was left empty, we open the default configuration file from cimr-rgb package
        if self.logpar_config is None or self.logpar_config.strip() == "":
            # Loading default CIMR RGB Log Config file as dictionary
            # (the one which is installed as part of the package)
            with pkg_resources.open_text(configs, "cimr_rgb_logger_config.json") as f:
                self.logpar_config = json.load(f)

        elif str(self.logpar_config).lower() == "none":
            # Configuring the logger object by adding the NullHandler as per
            # python's official documentation. In this way we do not interfere with
            # library users' logging functionality.
            self.logger_name = __name__
            self.logger = logging.getLogger(self.logger_name)
            self.logger.addHandler(logging.NullHandler())

            self.logpar_config = None

        else:
            self.logpar_config = pb.Path(self.logpar_config).resolve()

        # Initialising RGB logging (can be an empty object)
        rgb_logging = RGBLogging(
            logdir=logdir,
            log_config=self.logpar_config,
            filename=pb.Path(config_file_path).stem,
            # name_suffix  = self.file_time_signature
        )

        if (
            str(self.logpar_config).lower() != "none"
        ):  # or self.logpar_config.strip() == "":
            self.logger_name = config_object.find("LoggingParams/logger_name").text
            self.logger = rgb_logging.get_logger(self.logger_name)

        # TODO: Put this into its own validation method
        # Whether to use RGB decorator
        self.logpar_decorate = (
            config_object.find("LoggingParams/decorate").text
        ).lower()
        if self.logpar_decorate == "true":
            self.logpar_decorate = True
        elif self.logpar_decorate == "false":
            self.logpar_decorate = False
        else:
            raise ValueError(
                "Invalid value for `decorate` encountered. \nThe `decorate` parameter can either be `True` or `False`."
            )

        rgb_logging.setup_global_exception_handler(logger=self.logger)
        # -----------
        # OutputData metadata
        self.product_version = self.validate_output_data_metadata(
            config_object.find("OutputData/version")
        )
        self.creator_email = self.validate_email(
            config=config_object, email="OutputData/creator_email"
        )
        self.creator_url = self.validate_output_data_metadata(
            config_object.find("OutputData/creator_url")
        )
        self.creator_institution = self.validate_output_data_metadata(
            config_object.find("OutputData/creator_institution")
        )
        self.creator_name = self.validate_output_data_metadata(
            config_object.find("OutputData/creator_name")
        )

        # -----------

        self.input_data_type = self.validate_input_data_type(
            config_object=config_object, input_data_type="InputData/type"
        )

        self.input_data_path = self.validate_input_data_path(
            config_object=config_object, input_data_path="InputData/path"
        )

        self.antenna_patterns_path = self.validate_input_antenna_patterns_path(
            config_object=config_object,
            antenna_patterns_path="InputData/antenna_patterns_path",
            input_data_type=self.input_data_type,
        )

        # self.dpr_path = path.join(path.dirname(getcwd()), "dpr")

        self.quality_control = self.validate_quality_control(
            config_object=config_object,
            quality_control="InputData/quality_control",
            input_data_type=self.input_data_type,
        )

        self.grid_type = self.validate_grid_type(
            config_object=config_object,
            grid_type="GridParams/grid_type",
            input_data_type=self.input_data_type,
        )

        self.target_band = self.validate_target_band(
            config_object=config_object,
            target_band="InputData/target_band",
            input_data_type=self.input_data_type,
            grid_type=self.grid_type,
        )

        self.source_band = self.validate_source_band(
            config_object=config_object,
            source_band="InputData/source_band",
            input_data_type=self.input_data_type,
        )

        if self.grid_type == "L1C":
            # source and target band must be exactly the same
            if self.target_band != self.source_band:
                raise ValueError(
                    "Error: Source and Target bands must be the same for L1C data"
                )

        if self.grid_type == "L1C":
            self.grid_definition = self.validate_grid_definition(
                config_object=config_object,
                # grid_type        = self.grid_type,
                grid_definition="GridParams/grid_definition",
            )

            self.projection_definition = self.validate_projection_definition(
                config_object=config_object,
                grid_definition=self.grid_definition,
                projection_definition="GridParams/projection_definition",
            )
        else:
            self.grid_definition = None
            self.projection_definition = None

        self.regridding_algorithm = self.validate_regridding_algorithm(
            config_object=config_object,
            regridding_algorithm="ReGridderParams/regridding_algorithm",
        )

        self.split_fore_aft = self.validate_split_fore_aft(
            config_object=config_object,
            split_fore_aft="InputData/split_fore_aft",
            input_data_type=self.input_data_type,
        )

        self.save_to_disk = self.validate_save_to_disk(
            config_object=config_object, save_to_disk="OutputData/save_to_disk"
        )

        self.search_radius = self.validate_search_radius(
            config_object=config_object,
            search_radius="ReGridderParams/search_radius",
            # grid_definition=self.grid_definition,
            grid_type=self.grid_type,
            input_data_type=self.input_data_type,
        )

        self.max_neighbours = self.validate_max_neighbours(
            config_object=config_object,
            max_neighbours="ReGridderParams/max_neighbours",
            regridding_algorithm=self.regridding_algorithm,
        )

        self.boresight_shift = self.validate_boresight_shift(
            config_object=config_object,
            boresight_shift="ReGridderParams/boresight_shift",
            input_data_type=self.input_data_type,
        )

        if self.grid_type == "L1R":
            try:
                if self.target_band == self.source_band:
                    raise ValueError(
                        "Error: Source and Target bands cannot be the same"
                    )
            except ValueError as e:
                # print(f"{e}")
                self.logger.error(f"{e}")
                sys.exit(1)

        self.reduced_grid_inds = self.validate_reduced_grid_inds(
            config_object=config_object,
            reduced_grid_inds="GridParams/reduced_grid_inds",
        )

        # SMAP specific Parameters
        if self.input_data_type == "SMAP":
            self.aft_angle_min = 90
            self.aft_angle_max = 270
            self.scan_geometry = {"L": (779, 241)}
            self.variable_key_map = {
                "bt_h": "tb_h",
                "bt_v": "tb_v",
                "bt_3": "tb_3",
                "bt_4": "tb_4",
                "processing_scan_angle": "antenna_scan_angle",
                "longitude": "tb_lon",
                "latitude": "tb_lat",
                "x_position": "x_pos",
                "y_position": "y_pos",
                "z_position": "z_pos",
                "x_velocity": "x_vel",
                "y_velocity": "y_vel",
                "z_velocity": "z_vel",
                "sub_satellite_lon": "sc_nadir_lon",
                "sub_satellite_lat": "sc_nadir_lat",
                "altitude": "sc_geodetic_alt_ellipsoid",
                "faraday_rot_angle": "faraday_rotation_angle",
                "nedt_h": "nedt_h",
                "nedt_v": "nedt_v",
                "nedt_3": "nedt_3",
                "nedt_4": "nedt_4",
                "regridding_n_samples": "regridding_n_samples",
                "regridding_l1b_orphans": "regridding_l1b_orphans",
                "acq_time_utc": "antenna_scan_time_utc",
                "azimuth": "antenna_earth_azimuth",
                "scan_quality_flag": "antenna_scan_qual_flag",
                "data_quality_h": "tb_qual_flag_h",
                "data_quality_v": "tb_qual_flag_v",
                "data_quality_3": "tb_qual_flag_3",
                "data_quality_4": "tb_qual_flag_4",
            }

        # AMSR2 specific parameters
        if self.input_data_type == "AMSR2":
            # self.LMT = 2200
            self.key_mappings = {
                # 'input': (co_reg_key, bt_key)
                "6": (0, "Brightness Temperature (6.9GHz,"),
                "7": (1, "Brightness Temperature (7.3GHz,"),
                "10": (2, "Brightness Temperature (10.7GHz,"),
                "18": (3, "Brightness Temperature (18.7GHz,"),
                "23": (4, "Brightness Temperature (23.8GHz,"),
                "36": (5, "Brightness Temperature (36.5GHz,"),
                "89a": (None, "Brightness Temperature (89.0GHz-A,"),
                "89b": (None, "Brightness Temperature (89.0GHz-B,"),
            }

            self.variable_key_map = {
                "x_position": "Navigation Data",
                "y_position": "Navigation Data",
                "z_position": "Navigation Data",
                "x_velocity": "Navigation Data",
                "y_velocity": "Navigation Data",
                "z_velocity": "Navigation Data",
                "azimuth": "Earth Azimuth",
                "solar_azimuth": "Sun Azimuth",
                "acq_time_utc": "Scan Time",
            }

            self.scan_geometry = {
                "6": (1974, 243),
                "7": (1974, 243),
                "10": (1974, 243),
                "18": (1974, 243),
                "23": (1974, 243),
                "36": (1974, 243),
                "89a": (1974, 486),
                "89b": (1974, 486),
            }

            self.num_horns = {
                "89a": 1,
                "89b": 1,
                "6": 1,
                "7": 1,
                "10": 1,
                "18": 1,
                "23": 1,
                "36": 1,
            }

        # CIMR Specific Parameters
        if self.input_data_type == "CIMR":
            self.variable_key_map = {
                "longitude": "lon",
                "latitude": "lat",
                "bt_h": "brightness_temperature_h",
                "bt_v": "brightness_temperature_v",
                "bt_3": "brightness_temperature_t3",
                "bt_4": "brightness_temperature_t4",
                "processing_scan_angle": "scan_angle",
                "x_position": "satellite_position",
                "y_position": "satellite_position",
                "z_position": "satellite_position",
                "x_velocity": "satellite_velocity",
                "y_velocity": "satellite_velocity",
                "z_velocity": "satellite_velocity",
                "sub_satellite_lon": "sub_satellite_lon",
                "sub_satellite_lat": "sub_satellite_lat",
                "attitude": "SatelliteBody2EarthCenteredInertialFrame",
                "nedt_h": "nedt_h",
                "nedt_v": "nedt_v",
                "nedt_3": "nedt_3",
                "nedt_4": "nedt_4",
                "regridding_n_samples": "regridding_n_samples",
                "regridding_l1b_orphans": "regridding_l1b_orphans",
                "acq_time_utc": "utc_time",
                "azimuth": "earth_azimuth",
                "oza": "OZA",
            }
            self.aft_angle_min = 180
            self.aft_angle_max = 360
            self.num_horns = {
                "L": 1,
                "C": 4,
                "X": 4,
                "KA": 8,
                "K": 8,
            }
            self.scan_angle_feed_offsets = {}
            self.u0 = {}
            self.v0 = {}
            # Scan Geometry Hard Coding for now
            self.scan_geometry = {
                "L": (74, 691),
                "C": (74, 2747 * 4),
                "X": (74, 2807 * 4),
                "KA": (74, 10395 * 8),
                "K": (74, 7692 * 8),
            }
            self.cimr_nedt = self.validate_cimr_nedt(config_object=config_object)

        self.variables_to_regrid = self.validate_variables_to_regrid(
            config_object=config_object,
            input_data_type=self.input_data_type,
            variables_to_regrid="ReGridderParams/variables_to_regrid",
        )

        if self.regridding_algorithm in ["BG", "RSIR", "LW", "CG"]:
            self.source_antenna_method = self.validate_source_antenna_method(
                config_object=config_object,
                source_antenna_method="ReGridderParams/source_antenna_method",
            )

            if self.source_antenna_method in ["gaussian", "gaussian_projected"]:
                self.source_gaussian_params = self.validate_source_gaussian_params(
                    config_object=config_object,
                    source_gaussian_params="ReGridderParams/source_gaussian_params",
                )
            else:
                self.source_gaussian_params = None

            self.target_antenna_method = self.validate_target_antenna_method(
                config_object=config_object,
                target_antenna_method="ReGridderParams/target_antenna_method",
            )

            if self.target_antenna_method in ["gaussian", "gaussian_projected"]:
                self.target_gaussian_params = self.validate_target_gaussian_params(
                    config_object=config_object,
                    target_gaussian_params="ReGridderParams/target_gaussian_params",
                )
            else:
                self.target_gaussian_params = None

            self.source_antenna_threshold = self.validate_source_antenna_threshold(
                config_object=config_object,
                source_antenna_threshold="ReGridderParams/source_antenna_threshold",
            )

            self.target_antenna_threshold = self.validate_target_antenna_threshold(
                config_object=config_object,
                target_antenna_threshold="ReGridderParams/target_antenna_threshold",
            )

            self.max_theta_antenna_patterns = self.validate_max_theta_antenna_patterns(
                config_object=config_object,
                max_theta_antenna_patterns="ReGridderParams/max_theta_antenna_patterns",
            )

            self.polarisation_method = self.validate_polarisation_method(
                config_object=config_object,
                polarisation_method="ReGridderParams/polarisation_method",
            )

            self.MRF_grid_definition = self.validate_MRF_grid_definition(
                config_object=config_object,
                MRF_grid_definition="ReGridderParams/MRF_grid_definition",
            )

            self.MRF_projection_definition = self.validate_MRF_projection_definition(
                config_object=config_object,
                MRF_projection_definition="ReGridderParams/MRF_projection_definition",
            )

            self.antenna_pattern_uncertainty = self.validate_antenna_pattern_uncertainty(
                config_object=config_object,
                antenna_pattern_uncertainty="ReGridderParams/antenna_pattern_uncertainty",
            )

            if self.input_data_type == "SMAP":
                self.antenna_tilt_angle = 144.54

            elif self.input_data_type == "CIMR":
                self.antenna_tilt_angle = 46.886  # update to read from file

        if self.regridding_algorithm == "RSIR":
            self.rsir_iteration = self.validate_rsir_iteration(
                config_object=config_object,
                rsir_iteration="ReGridderParams/rsir_iteration",
            )

        if self.regridding_algorithm == "BG":
            self.bg_smoothing = self.validate_bg_smoothing(
                config_object=config_object, bg_smoothing="ReGridderParams/bg_smoothing"
            )

        if self.regridding_algorithm in ["LW", "CG"]:
            self.max_iterations = self.validate_max_iterations(
                config_object=config_object,
                max_iterations="ReGridderParams/max_iterations",
            )
            self.relative_tolerance = self.validate_relative_tolerance(
                config_object=config_object,
                relative_tolerance="ReGridderParams/relative_tolerance",
            )
            self.regularisation_parameter = self.validate_regularisation_parameter(
                config_object=config_object,
                regularisation_parameter="ReGridderParams/regularisation_parameter",
            )
            self.max_chunk_size = self.validate_max_chunk_size(
                config_object=config_object,
                max_chunk_size="ReGridderParams/max_chunk_size",
            )
            self.chunk_buffer = self.validate_chunk_buffer(
                config_object, chunk_buffer="ReGridderParams/chunk_buffer"
            )

    @staticmethod
    def read_config(config_file_path):
        """
        Reads the configuration file and returns the root element
        Parameters
        ----------
        config_file_path: str

        Returns
        -------
        root: xml.etree.ElementTree.Element
            Root element of the configuration file
        """

        try:
            tree = parse(config_file_path)
            root = tree.getroot()
        except ParseError as e:
            raise ValueError(f"Error parsing the configuration file: {e}") from e
        return root, tree

    @staticmethod
    def validate_email(
        config,
        email: str = "OutputData/creator_email",
        logger: typing.Optional[logging.Logger] = None,
    ):
        """
        Validates if the provided string is a valid email address.

        Args:
            email (str): The email address to validate.

        Raises:
            ValueError: If the email is not in a valid format.
        """

        email = config.find(email).text

        # Regular expression for validating an email
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        # Validate email format
        if not re.match(email_regex, email):
            error_message = (
                "Email should have the following format: username@domain.com"
            )
            if logger:
                logger.error(f"Invalid email provided: {email} - {error_message}")
            else:
                print(f"Invalid email provided: {email} - {error_message}")
            raise ValueError(error_message)
        else:
            return email

    @staticmethod
    def validate_output_data_metadata(parameter):
        if parameter.text.strip() == "" or parameter is None:
            parameter = ""
        else:
            parameter = parameter.text.strip()
        return parameter

    @staticmethod
    def validate_output_directory_path(
        config_object,
        output_path: str = "OutputData/output_path",
        logger: typing.Optional[logging.Logger] = None,
    ) -> pb.Path:
        """
        Validates, resolves, and creates the output directory path specified in the configuration object.

        This method:
        1. Retrieves the `output_path` element from the provided `config_object` (XML root).
        2. Resolves the path to handle relative paths, environment variables (`$VAR`),
           and home directory symbols (`~`) using the `resolve_config_path` method.
        3. Recursively creates the directory structure if it does not already exist using `rec_create_dir`.

        Parameters:
        -----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.

        output_path : str, optional
            The XML path to the `output_path` element (default is `"OutputData/output_path"`).

        logger : Optional[logging.Logger], optional
            A custom logging object. If `None`, no logging will be performed.

        Returns:
        --------
        pb.Path
            The fully resolved and validated output directory path as a `Path` object.

        Raises:
        -------
        ValueError:
            If the `output_path` element is missing or empty in the configuration object.

        FileNotFoundError:
            If resolving the path results in a directory that cannot be accessed or created.

        PermissionError:
            If there are insufficient permissions to create or access the directory.

        Exception:
            Any unexpected errors during directory validation or creation.

        Notes:
        ------
        - Logs messages at DEBUG level if directories are created or already exist.
        - Uses helper methods `resolve_config_path` and `rec_create_dir` for path resolution and creation.

        Example:
        --------
        Assuming an XML configuration object contains the following:
        ```xml
        <OutputData>
            <output_path>~/my_project/output</output_path>
        </OutputData>
        ```
        Call:
        ```python
        resolved_path = ConfigFile.validate_output_directory_path(config_object)
        print(resolved_path)  # Output: '/home/user/my_project/output'
        ```
        """

        # outputdir       = pb.Path(config_object.find(output_path).text)
        # outputdir       = grasp_io.resolve_config_path(
        #    path_string = outputdir
        # )
        ## recursively creating (nested) output directories
        # grasp_io.rec_create_dir(outputdir)

        try:
            # Initialize the empty logger if None is provided
            if logger is None:
                logger = logging.getLogger(__name__)
                logger.addHandler(logging.NullHandler())

            # Retrieve the output path text from the configuration object
            outputdir_element = config_object.find(output_path)
            if outputdir_element is None or outputdir_element.text.strip() == "":
                raise ValueError(
                    f"Missing or empty `output_path` element in configuration file: {output_path}"
                )

            outputdir = pb.Path(outputdir_element.text)

            # Resolve the path using the helper method
            outputdir = grasp_io.resolve_config_path(path_string=outputdir)

            if logger:
                logger.debug(f"Resolved output directory path: {outputdir}")

            # Create the directory structure if necessary
            grasp_io.rec_create_dir(path=outputdir)

            return outputdir

        except ValueError as ve:
            if logger:
                logger.error(f"ValueError: {ve}")
            else:
                print(f"ValueError: {ve}")
            raise
        except FileNotFoundError as fnf:
            if logger:
                logger.error(
                    f"FileNotFoundError: Unable to access path `{output_path}` - {fnf}"
                )
            else:
                print(
                    f"FileNotFoundError: Unable to access path `{output_path}` - {fnf}"
                )
            raise
        except PermissionError as pe:
            if logger:
                logger.error(
                    f"PermissionError: Insufficient permissions for path `{output_path}` - {pe}"
                )
            else:
                print(
                    f"PermissionError: Insufficient permissions for path `{output_path}` - {pe}"
                )
            raise
        except Exception as e:
            if logger:
                logger.error(
                    f"An unexpected error occurred while validating the output directory: {e}"
                )
            else:
                print(
                    f"An unexpected error occurred while validating the output directory: {e}"
                )
            raise

    @staticmethod
    def validate_input_data_type(config_object, input_data_type):
        r"""
        Validate the `input_data_type` parameter by checking its existence in the XML
        configuration and ensuring it matches a predefined set of valid values.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            The root element of the XML configuration file.
        input_data_type : str
            The XML tag name corresponding to the input data type.

        Returns
        -------
        str
            The validated input data type in uppercase format.

        Raises
        ------
        AttributeError
            If the XML tag `input_data_type` is missing or incorrectly specified.
        ValueError
            If the extracted `input_data_type` is not in the predefined valid set.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = <config><InputData><type>AMSR2</type></InputData></config>
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_input_data_type(config_object, "inputType")
        'AMSR2'

        >>> xml_data = <config><InputData><type>INVALID</type></InputData></config>
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_input_data_type(config_object, "inputType")
        Traceback (most recent call last):
            ...
        ValueError: Invalid input data type. Valid input data types are: ['AMSR2', 'SMAP', 'CIMR'].

        >>> xml_data = <config><inputData>AMSR2</inputData></config>
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_input_data_type(config_object, "inputType")
        Traceback (most recent call last):
            ...
        AttributeError: Missing or incorrect XML tag: '<inputData>' not found in the configuration file.
        """

        valid_input = ["AMSR2", "SMAP", "CIMR"]

        # Checking if `<InputData><type>` parameter is present in the config file
        try:
            input_data_type = str(
                config_object.find(input_data_type).text.strip()
            ).upper()
        except AttributeError:
            raise AttributeError(
                f"Missing or incorrect XML tag: '{input_data_type}' not found in the configuration file."
            )

        if input_data_type in valid_input:
            return input_data_type

        raise ValueError(
            f"Invalid input data type. Valid input data types are: {valid_input}."
        )

    # TODO: Add proper docstring in scipy/numpy format
    @staticmethod
    def validate_input_data_path(config_object, input_data_path):
        """
        Validates the input data path and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        input_data_path: str
            Path to the input data type in the configuration file

        Returns
        -------
        str
            Validated input data path
        """

        valid_extensions = [".h5", ".hdf5", ".nc"]

        input_data_path = (config_object.find(input_data_path).text).strip()
        input_data_path = grasp_io.resolve_config_path(path_string=input_data_path)

        if input_data_path.exists():
            if input_data_path.suffix in valid_extensions:
                return input_data_path
            else:
                raise ValueError(
                    f"File\n {input_data_path} is of invalid type. Valid file types are: {valid_extensions}."
                )
        else:
            raise FileNotFoundError(
                f"File\n {input_data_path}\n not found. Check file location."
            )

    # TODO: Add proper docstring in scipy/numpy format
    @staticmethod
    def validate_input_antenna_patterns_path(
        config_object, antenna_patterns_path, input_data_type
    ):
        """
        Validates the input data path and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        input_data_path: str
            Path to the input data type in the configuration file

        Returns
        -------
        str
            Validated input data path
        """

        antenna_patterns_path = pb.Path(
            config_object.find(antenna_patterns_path).text.strip()
        )
        antenna_patterns_path = grasp_io.resolve_config_path(
            path_string=antenna_patterns_path
        )

        if antenna_patterns_path.exists():
            if input_data_type == "SMAP":
                try:
                    relative_path = "SMAP/RadiometerAntPattern_170830_v011.h5"
                    antenna_patterns_path = path.join(
                        antenna_patterns_path, relative_path
                    )
                except AttributeError as e:
                    raise ValueError(
                        "Error: SMAP Antenna Pattern not found in dpr"
                    ) from e
            elif input_data_type == "CIMR":
                try:
                    relative_path = "CIMR"
                    antenna_patterns_path = path.join(
                        antenna_patterns_path, relative_path
                    )
                except AttributeError as e:
                    raise ValueError(
                        f"Error: CIMR Antenna Pattern folder not found in {antenna_patterns_path}"
                    ) from e

            return antenna_patterns_path
        else:
            raise FileNotFoundError(
                f"File\n {antenna_patterns_path}\n not found. Check file location."
            )

    @staticmethod
    def validate_grid_type(config_object, grid_type, input_data_type):
        """
        Validates the grid type and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        grid_type: str
            User selected Grid Type in the configuration file
        input_data_type: str
            User selected Input Data Type in the configuration file

        Returns
        -------
        str
            Validated grid type
        """

        r"""
        Validate the grid type parameter from the XML configuration.

        This method verifies if the grid type exists in the XML configuration 
        and ensures it matches a predefined set of acceptable grid types 
        based on the selected input data type.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            The root element of the XML configuration file.
        grid_type : str
            The XML tag path corresponding to the grid type.
        input_data_type : str
            The user-selected input data type that determines the valid grid types.

        Returns
        -------
        str
            The validated grid type if it is found and matches one of the 
            predefined valid values for the given `input_data_type`.

        Raises
        ------
        AttributeError
            If the required XML tag for `grid_type` is missing or incorrectly 
            specified in the configuration file.
        ValueError
            If the extracted grid type is not in the list of valid grid types.

        Notes
        -----
        - The function searches for the `<GridParams><grid_type>` tag in the 
          XML file and validates its value against predefined grid types.
        - Valid grid types depend on the input data type:
          
          - If `input_data_type` is `SMAP`: `['L1C']`
          - Otherwise: `['L1C', 'L1R']`
        - Grid type values are case-insensitive and converted to uppercase.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><GridParams><grid_type>L1C</grid_type></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_grid_type(config_object, "GridParams/grid_type", "SMAP")
        'L1C'

        >>> xml_data = '''<config><GridParams><grid_type>INVALID</grid_type></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_grid_type(config_object, "GridParams/grid_type", "CIMR")
        Traceback (most recent call last):
            ...
        ValueError: Invalid Grid Type. Check Configuration File. Valid grid types are: ['L1C', 'L1R'] for GridParams/grid_type data.

        >>> xml_data = '''<config><GridParams><otherTag>L1C</otherTag></GridParams></config>''' 
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_grid_type(config_object, "GridParams/grid_type", "SMAP")
        Traceback (most recent call last):
            ...
        AttributeError: Missing or incorrect XML tag: 'GridParams/grid_type' not found in the configuration file.
        """

        if input_data_type == "SMAP":
            valid_input = ["L1C"]
        else:
            valid_input = ["L1C", "L1R"]

        # Checking if `<GridParams><grid_type>` parameter is present in the config file
        try:
            grid_type = str(config_object.find(grid_type).text.strip()).upper()
        except AttributeError:
            raise AttributeError(
                f"Missing or incorrect XML tag: '{grid_type}' not found in the configuration file."
            )

        if grid_type in valid_input:
            return grid_type

        raise ValueError(
            f"Invalid Grid Type. Check Configuration File. Valid grid types are:"
            f" {valid_input} for GridParams/grid_type data."
        )

    # TODO:
    @staticmethod
    def validate_target_band(config_object, target_band, input_data_type, grid_type):
        """
        Validates the target band and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        target_band: str
            User selected Target Band in the configuration file
        input_data_type: str
            User selected Input Data Type in the configuration file
        grid_type: str
            User selected Grid Type in the configuration file

        Returns
        -------
        str
            Validated target band
        """

        # TODO: Add validate_input_data_type here? Otherwise this method just passes through

        if input_data_type == "AMSR2":
            valid_input = ["6", "7", "10", "18", "23", "36", "89a", "89b", "All"]
            config_input = config_object.find(target_band).text.split()
            if config_input == ["All"]:
                return ["6", "7", "10", "18", "23", "36", "89a", "89b"]
            else:
                for i in config_input:
                    if i not in valid_input:
                        raise ValueError(
                            f"Invalid target bands for AMSR2 L1C remap."
                            f" Valid target bands are: {valid_input} or any combination of individual bands."
                        )
                return config_object.find(target_band).text.split()

        elif input_data_type == "CIMR":
            if grid_type == "L1C":
                valid_input = ["L", "C", "X", "KA", "K", "All"]
            elif grid_type == "L1R":
                valid_input = ["L", "C", "X", "KA", "K"]
            config_input = config_object.find(target_band).text.split()
            if config_input == ["All"]:
                return ["L", "C", "X", "KA", "K"]
            else:
                for i in config_input:
                    if i not in valid_input:
                        raise ValueError(
                            f"Invalid target bands for CIMR L1C remap."
                            f" Valid target bands are: {valid_input} or any combination of individual bands."
                        )
                return config_object.find(target_band).text.split()

        if input_data_type == "SMAP":
            valid_input = ["L"]
            if config_object.find(target_band).text in valid_input:
                return config_object.find(target_band).text.split()
            raise ValueError(
                f"Invalid target band for SMAP L1C remap. Valid target band is: {valid_input}"
            )

    # TODO:
    @staticmethod
    def validate_source_band(config_object, source_band, input_data_type):
        """
        Validates the source band and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        source_band: str
            User selected Source Band in the configuration file
        input_data_type: str
            User selected Input Data Type in the configuration file

        Returns
        -------
        str
            Validated source band
        """
        if input_data_type == "AMSR2":
            valid_input = ["6", "7", "10", "18", "23", "36", "89a", "89b"]

        if input_data_type == "SMAP":
            valid_input = ["L"]

        if input_data_type == "CIMR":
            valid_input = ["L", "C", "X", "KA", "K"]

        try:
            value = config_object.find(source_band).text.split()
        except AttributeError as e:
            raise ValueError(
                "Error: Source Band not found in configuration file. Check configuration file."
            ) from e

        if all(item in valid_input for item in value):
            return value
        else:
            raise ValueError(
                f"Invalid Source Band, check configuration file. "
                f"Valid source bands are: {valid_input}."
            )

    @staticmethod
    def validate_grid_definition(config_object, grid_definition):
        r"""
        Validate the grid definition parameter from the XML configuration.

        This method checks if the provided grid definition exists in the XML
        configuration and validates it against a predefined set of acceptable
        grid definitions.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            The root element of the XML configuration file.
        grid_definition : str
            The XML tag path corresponding to the grid definition.

        Returns
        -------
        str
            The validated grid definition if it is found and matches one of the
            predefined valid values.

        Raises
        ------
        AttributeError
            If the required XML tag for `grid_definition` is missing or incorrectly
            specified in the configuration file.
        ValueError
            If the extracted grid definition is not in the list of valid grid definitions.

        Notes
        -----
        - Grid definitions are case-sensitive.
        - The function searches for the `<GridParams><grid_definition>` tag in the
          XML file and validates its value against predefined grid types.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><GridParams><grid_definition>EASE2_G9km</grid_definition></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_grid_definition(config_object, "GridParams/grid_definition")
        'EASE2_G9km'

        >>> xml_data = '''<config><GridParams><grid_definition>INVALID_GRID</grid_definition></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_grid_definition(config_object, "GridParams/grid_definition")
        Traceback (most recent call last):
            ...
        ValueError: Invalid Grid Definition, check configuration file. Valid grid definitions are:
            ['EASE2_G9km', 'EASE2_N9km', 'EASE2_S9km', 'EASE2_G36km', 'EASE2_N36km', 'EASE2_S36km',
             'STEREO_N25km', 'STEREO_S25km', 'STEREO_N6.25km', 'STEREO_N12.5km', 'STEREO_S6.25km',
             'STEREO_S12.5km', 'STEREO_S25km', 'MERC_G25km', 'MERC_G12.5km', 'MERC_G6.25km']

        >>> xml_data = '''<config><GridParams><otherTag>EASE2_G9km</otherTag></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_grid_definition(config_object, "GridParams/grid_definition")
        Traceback (most recent call last):
            ...
        AttributeError: Missing or incorrect XML tag: 'GridParams/grid_definition' not found in the configuration file.
        """

        valid_input = [
            "EASE2_G1km",
            "EASE2_G3km",
            "EASE2_G9km",
            "EASE2_N1km",
            "EASE2_N3km",
            "EASE2_N9km",
            "EASE2_S9km",
            "EASE2_G36km",
            "EASE2_N36km",
            "EASE2_S36km",
            "STEREO_N25km",
            "STEREO_S25km",
            "STEREO_N6.25km",
            "STEREO_N12.5km",
            "STEREO_S6.25km",
            "STEREO_S12.5km",
            "STEREO_S25km",
            "MERC_G25km",
            "MERC_G12.5km",
            "MERC_G6.25km",
        ]

        # Checking if `<GridParams><grid_definition>` parameter is present in the config file
        #
        # Note: grid_definition is case sensitive
        try:
            grid_definition = config_object.find(grid_definition).text.strip()
        except AttributeError:
            raise AttributeError(
                f"Missing or incorrect XML tag: '{grid_definition}' not found in the configuration file."
            )

        if grid_definition in valid_input:
            return grid_definition

        raise ValueError(
            f"Invalid Grid Definition ({grid_definition}), check configuration file. "
            f"Valid grid definitions are: {valid_input}"
        )

    @staticmethod
    def validate_projection_definition(
        config_object, grid_definition, projection_definition
    ):
        r"""
        Validate the projection definition parameter from the XML configuration.

        This method verifies if the projection definition exists in the XML
        configuration and ensures it matches the expected projection type based on
        the selected grid definition.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            The root element of the XML configuration file.
        grid_definition : str
            The user-selected grid definition that determines the expected projection types.
        projection_definition : str
            The XML tag path corresponding to the projection definition.

        Returns
        -------
        str or None
            The validated projection definition if it is found and matches one of the
            predefined valid values for the given `grid_definition`, or `None` if
            `grid_definition` is not provided.

        Raises
        ------
        AttributeError
            If the required XML tag for `projection_definition` is missing or incorrectly
            specified in the configuration file.
        ValueError
            If the extracted projection definition is not valid for the given `grid_definition`.

        Notes
        -----
        - The valid projection definitions depend on the `grid_definition`:

          - `EASE2_*` grids: `["G", "N", "S"]`
          - `STEREO_*` grids: `["PS_N", "PS_S"]`
          - `MERC_*` grids: `["MERC_G"]`

        - The function searches for the `<GridParams><projection_definition>` tag in the
          XML file and validates its value accordingly.
        - Projection definitions are case-insensitive and converted to uppercase.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><GridParams><projection_definition>G</projection_definition></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_projection_definition(config_object, "EASE2_G9km", "GridParams/projection_definition")
        'G'

        >>> xml_data = '''<config><GridParams><projection_definition>INVALID</projection_definition></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_projection_definition(config_object, "EASE2_G9km", "GridParams/projection_definition")
        Traceback (most recent call last):
            ...
        ValueError: Grid Definition `EASE2_G9km` received invalid projection definition: `INVALID`; check configuration file.
                    Valid projection definitions are: `['G', 'N', 'S']`

        >>> xml_data = '''<config><GridParams><otherTag>PS_N</otherTag></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_projection_definition(config_object, "STEREO_N25km", "GridParams/projection_definition")
        Traceback (most recent call last):
            ...
        AttributeError: Missing or incorrect XML tag: 'GridParams/projection_definition' not found in the configuration file.

        >>> ConfigFile.validate_projection_definition(config_object, None, "GridParams/projection_definition")
        None
        """

        if grid_definition:
            if "EASE2" in grid_definition:
                valid_input = ["G", "N", "S"]

            elif "STEREO" in grid_definition:
                valid_input = ["PS_N", "PS_S"]

            elif "MERC" in grid_definition:
                valid_input = ["MERC_G"]

            else:
                valid_input = []

            # Checking if `<GridParams><projection_definition>` parameter is present in the config file
            try:
                proj_val = str(
                    config_object.find(projection_definition).text.strip()
                ).upper()
            except AttributeError:
                raise AttributeError(
                    f"Missing or incorrect XML tag: '{projection_definition}' not found in the configuration file."
                )

            if proj_val in valid_input:
                return proj_val

            raise ValueError(
                f"Grid Definiton `{grid_definition}` received invalid projection definition: `{proj_val}`; "
                f"check configuration file."
                f" Valid projection definitions are: `{valid_input}`"
            )
        else:
            return None

    @staticmethod
    def validate_regridding_algorithm(config_object, regridding_algorithm):
        r"""
        Validate the regridding algorithm parameter from the XML configuration.

        This method verifies if the regridding algorithm exists in the XML configuration
        and ensures it matches a predefined set of acceptable regridding algorithms.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            The root element of the XML configuration file.
        regridding_algorithm : str
            The XML tag path corresponding to the regridding algorithm.

        Returns
        -------
        str
            The validated regridding algorithm if it is found and matches one of the
            predefined valid values.

        Raises
        ------
        AttributeError
            If the required XML tag for `regridding_algorithm` is missing or incorrectly
            specified in the configuration file.
        ValueError
            If the extracted regridding algorithm is not in the list of valid algorithms.

        Notes
        -----
        - The function searches for the `<ReGridderParams><regridding_algorithm>` tag in the
          XML file and validates its value against predefined regridding methods.
        - Valid regridding algorithms include:
          `['NN', 'DIB', 'IDS', 'BG', 'RSIR', 'LW', 'CG']`
        - Algorithm names are case-insensitive and converted to uppercase.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><regridding_algorithm>NN</regridding_algorithm></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_regridding_algorithm(config_object, "ReGridderParams/regridding_algorithm")
        'NN'

        >>> xml_data = '''<config><ReGridderParams><regridding_algorithm>INVALID</regridding_algorithm></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_regridding_algorithm(config_object, "ReGridderParams/regridding_algorithm")
        Traceback (most recent call last):
            ...
        ValueError: Invalid regridding algorithm. Check Configuration File. Valid regridding algorithms are: ['NN', 'DIB', 'IDS', 'BG', 'RSIR', 'LW', 'CG']

        >>> xml_data = '''<config><ReGridderParams><otherTag>NN</otherTag></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_regridding_algorithm(config_object, "ReGridderParams/regridding_algorithm")
        Traceback (most recent call last):
            ...
        AttributeError: Missing or incorrect XML tag: 'ReGridderParams/regridding_algorithm' not found in the configuration file.
        """

        valid_input = ["NN", "DIB", "IDS", "BG", "RSIR", "LW", "CG"]

        try:
            regridding_algorithm = str(
                config_object.find(regridding_algorithm).text.strip()
            ).upper()
        except AttributeError:
            raise AttributeError(
                f"Missing or incorrect XML tag: '{regridding_algorithm}' not found in the configuration file."
            )

        if regridding_algorithm in valid_input:
            return regridding_algorithm

        raise ValueError(
            f"Invalid regridding algorithm. Check Configuration File."
            f" Valid regridding algorithms are: {valid_input}"
        )

    @staticmethod
    def validate_split_fore_aft(config_object, split_fore_aft, input_data_type):
        r"""
        Validate the `split_fore_aft` parameter from the XML configuration.

        This method verifies if the `split_fore_aft` parameter exists in the XML
        configuration and ensures it matches a predefined set of acceptable values.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            The root element of the XML configuration file.
        split_fore_aft : str
            The XML tag path corresponding to the split fore/aft parameter.
        input_data_type : str
            The input data type used to determine valid values for `split_fore_aft`.

        Returns
        -------
        bool
            The validated `split_fore_aft` value as a boolean.

        Raises
        ------
        AttributeError
            If the required XML tag for `split_fore_aft` is missing or incorrectly
            specified in the configuration file.
        ValueError
            If the extracted `split_fore_aft` value is not in the list of valid options.

        Notes
        -----
        - If `input_data_type` is `AMSR2`, the function always returns `False`.
        - The function searches for the `<InputData><split_fore_aft>` tag in the
          XML file and validates its value against `TRUE` or `FALSE`.
        - The values are case-insensitive and converted to uppercase.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><InputData><split_fore_aft>True</split_fore_aft></InputData></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_split_fore_aft(config_object, "InputData/split_fore_aft", "SMAP")
        True

        >>> xml_data = '''<config><InputData><split_fore_aft>false</split_fore_aft></InputData></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_split_fore_aft(config_object, "InputData/split_fore_aft", "CIMR")
        False

        >>> xml_data = '''<config><InputData><otherTag>True</otherTag></InputData></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_split_fore_aft(config_object, "InputData/split_fore_aft", "CIMR")
        Traceback (most recent call last):
            ...
        AttributeError: Missing or incorrect XML tag: 'InputData/split_fore_aft' not found in the configuration file.

        >>> ConfigFile.validate_split_fore_aft(config_object, "InputData/split_fore_aft", "AMSR2")
        False
        """

        if input_data_type == "AMSR2":
            return False

        valid_input = ["TRUE", "FALSE"]

        try:
            split_fore_aft = str(
                config_object.find(split_fore_aft).text.strip()
            ).upper()
        except AttributeError:
            raise AttributeError(
                f"Missing or incorrect XML tag: '{split_fore_aft}' not found in the configuration file."
            )

        # if config_object.find(split_fore_aft).text in valid_input:
        if split_fore_aft in valid_input:
            if split_fore_aft == "TRUE":
                return True
            else:
                return False

        raise ValueError(
            f"Invalid split fore aft. Check Configuration File."
            f" Valid split fore aft are: {valid_input}"
        )

    @staticmethod
    def validate_save_to_disk(config_object, save_to_disk):
        r"""
        Validate the `save_to_disk` parameter from the XML configuration.

        This method verifies if the `save_to_disk` parameter exists in the XML
        configuration and ensures it matches a predefined set of acceptable values.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            The root element of the XML configuration file.
        save_to_disk : str
            The XML tag path corresponding to the save to disk parameter.

        Returns
        -------
        bool
            The validated `save_to_disk` value as a boolean.

        Raises
        ------
        AttributeError
            If the required XML tag for `save_to_disk` is missing or incorrectly
            specified in the configuration file.
        ValueError
            If the extracted `save_to_disk` value is not in the list of valid options.

        Notes
        -----
        - The function searches for the `<OutputParams><save_to_disk>` tag in the
          XML file and validates its value against `TRUE` or `FALSE`.
        - The values are case-insensitive and converted to uppercase.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><OutputParams><save_to_disk>True</save_to_disk></OutputParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_save_to_disk(config_object, "OutputParams/save_to_disk")
        True

        >>> xml_data = '''<config><OutputParams><save_to_disk>false</save_to_disk></OutputParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_save_to_disk(config_object, "OutputParams/save_to_disk")
        False

        >>> xml_data = '''<config><OutputParams><otherTag>True</otherTag></OutputParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_save_to_disk(config_object, "OutputParams/save_to_disk")
        Traceback (most recent call last):
            ...
        AttributeError: Missing or incorrect XML tag: 'OutputParams/save_to_disk' not found in the configuration file.
        """

        valid_input = ["TRUE", "FALSE"]

        try:
            value = str(config_object.find(save_to_disk).text.strip()).upper()
        except AttributeError:
            raise AttributeError(
                f"Missing or incorrect XML tag: '{save_to_disk}' not found in the configuration file."
            )

        # if value is not True and value is not False:
        if value in valid_input:
            if value == "TRUE":
                value = True
                return value
            else:
                value = False
                return value
        else:
            raise ValueError(
                "Invalid `save_to_disk`. Check Configuration File."
                " `save_to_disk` must be either True or False"
            )

    @staticmethod
    def validate_search_radius(
        config_object,
        search_radius,
        grid_type,
        input_data_type,
    ):
        r"""
        Validates the search radius (optional) and returns the value if valid.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the configuration file.
        search_radius : str
            Path to the search radius in the configuration file.
        grid_type : str
            The type of grid selected.
        input_data_type : str
            The input data type used for determining default values.

        Returns
        -------
        float or None
            The validated search radius in meters if specified, otherwise a default value based on grid type.

        Raises
        ------
        ValueError
            If `search_radius` contains a non-numeric value.
        ValueError
            If `grid_type` or `input_data_type` is invalid and no default search radius can be determined.

        Notes
        -----
        - Converts valid numerical values from kilometers to meters.
        - Uses predefined defaults based on `grid_type` and `input_data_type` when `search_radius` is None.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><search_radius>5.0</search_radius></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_search_radius(config_object, "ReGridderParams/search_radius", "L1C", "CIMR")
        5000.0
        """

        # Making this to accomodate cases of: None, "", and " "
        value = str(config_object.find(search_radius).text)

        if value == "None" or value == "" or value.strip() == "":
            value = None
        else:
            # Ensure the value is numeric
            try:
                value = float(value) * 1000
            except ValueError:
                raise ValueError(
                    f"Invalid `search_radius`: {value}. Must be a numeric value."
                )

        if value is None:
            if grid_type == "L1C":
                value = None
            elif grid_type == "L1R":
                if input_data_type == "CIMR":
                    return (
                        73000 / 2
                    )  # Largets CIMR footprint radius, maybe needs tailoring
                elif input_data_type == "AMSR2":
                    return (
                        62000 / 2
                    )  # Largest AMSR2 footprint radius, maybe needs tailoring
                else:
                    raise ValueError(f"Invalid `input_data_type`: {input_data_type}")
            else:
                raise ValueError(f"Invalid `grid_type`: {grid_type}")

        return value

    @staticmethod
    def get_scan_geometry(config, band_to_remap=None):
        r"""
        Retrieve the scan geometry based on the input data type and band to remap.

        This function determines the number of scans and earth samples per scan
        based on the sensor type and frequency band.

        Parameters
        ----------
        config : object
            A configuration object that must have an `input_data_type` attribute.
        band_to_remap : str, optional
            The frequency band to remap (applicable only for `CIMR` data).

        Returns
        -------
        tuple
            A tuple containing:
            - num_scans (int): Number of scans per swath.
            - num_earth_samples (int): Number of earth samples per scan.

        Raises
        ------
        ValueError
            If an invalid `input_data_type` is provided.

        Notes
        -----
        - For `SMAP`, the scan geometry is fixed at (779, 241).
        - For `CIMR`, the scan geometry depends on the selected frequency band.

        Examples
        --------
        >>> class Config:
        ...     input_data_type = "SMAP"
        >>> config = Config()
        >>> get_scan_geometry(config)
        (779, 241)

        >>> class Config:
        ...     input_data_type = "CIMR"
        >>> config = Config()
        >>> get_scan_geometry(config, band_to_remap="L")
        (74, 691)
        """

        if config.input_data_type == "SMAP":
            num_scans = 779
            num_earth_samples = 241
        elif config.input_data_type == "CIMR":
            if band_to_remap == "L":
                num_scans = 74
                num_earth_samples = 691
            elif band_to_remap == "C":
                num_scans = 74
                num_earth_samples = 2747 * 4
            elif band_to_remap == "X":
                num_scans = 74
                num_earth_samples = 2807 * 4
            elif band_to_remap == "KA":
                num_scans = 74
                num_earth_samples = 10395 * 8
            elif band_to_remap == "K":
                num_scans = 74
                num_earth_samples = 7692 * 8
            else:
                raise ValueError(f"Invalid band_to_remap: {band_to_remap}")
        else:
            raise ValueError(f"Invalid `input_data_type`: {config.input_data_type}")

        return num_scans, num_earth_samples

    # @staticmethod
    # def validate_variables_to_regrid(
    #     config_object, input_data_type, variables_to_regrid
    # ):
    #     value = config_object.find(variables_to_regrid).text

    #     if input_data_type == "SMAP":
    #         valid_input = [
    #             "bt_h",
    #             "bt_v",
    #             "bt_3",
    #             "bt_4",
    #             "processing_scan_angle",
    #             "longitude",
    #             "latitude",
    #             "faraday_rot_angle",
    #             "nedt_h",
    #             "nedt_v",
    #             "nedt_3",
    #             "nedt_4",
    #             "regridding_n_samples",
    #             "regridding_l1b_orphans",
    #             "acq_time_utc",
    #             "azimuth",
    #         ]

    #         default_vars = [
    #             "bt_h",
    #             "bt_v",
    #             "bt_3",
    #             "bt_4",
    #             "processing_scan_angle",
    #             "longitude",
    #             "latitude",
    #             "faraday_rot_angle",
    #             "nedt_h",
    #             "nedt_v",
    #             "nedt_3",
    #             "nedt_4",
    #             "regridding_n_samples",
    #             "regridding_l1b_orphans",
    #             "acq_time_utc",
    #             "azimuth",
    #         ]

    #     elif input_data_type == "AMSR2":
    #         valid_input = [
    #             "bt_h",
    #             "bt_v",
    #             "longitude",
    #             "latitude",
    #             "regridding_n_samples",
    #             "x_position",
    #             "y_position",
    #             "z_position",
    #             "x_velocity",
    #             "y_velocity",
    #             "z_velocity",
    #             "azimuth",
    #             "solar_azimuth",
    #             "acq_time_utc",
    #         ]

    #         default_vars = [
    #             "bt_h",
    #             "bt_v",
    #             "longitude",
    #             "latitude",
    #             "regridding_n_samples",
    #             "x_position",
    #             "y_position",
    #             "z_position",
    #             "x_velocity",
    #             "y_velocity",
    #             "z_velocity",
    #             "azimuth",
    #             "solar_azimuth",
    #             "acq_time_utc",
    #         ]

    #     elif input_data_type == "CIMR":
    #         valid_input = [
    #             "bt_h",
    #             "bt_v",
    #             "bt_3",
    #             "bt_4",
    #             "processing_scan_angle",
    #             "longitude",
    #             "latitude",
    #             "nedt_h",
    #             "nedt_v",
    #             "nedt_3",
    #             "nedt_4",
    #             "regridding_n_samples",
    #             "regridding_l1b_orphans",
    #             "acq_time_utc",
    #             "azimuth",
    #             "oza",
    #         ]

    #         default_vars = [
    #             "bt_h",
    #             "bt_v",
    #             "bt_3",
    #             "bt_4",
    #             "processing_scan_angle",
    #             "longitude",
    #             "latitude",
    #             "nedt_h",
    #             "nedt_v",
    #             "nedt_3",
    #             "nedt_4",
    #             "regridding_n_samples",
    #             "regridding_l1b_orphans",
    #             "acq_time_utc",
    #             "azimuth",
    #             "oza",
    #         ]
    #     else:
    #         raise ValueError(f"Invalid `input_data_type`: {input_data_type}")

    #     if value is not None:
    #         for variable in value.split():
    #             if variable not in valid_input:
    #                 raise ValueError(
    #                     f"Invalid variable_to_regrid. Check Configuration File."
    #                     f" Valid variables_to_regrid: {valid_input}"
    #                 )
    #         return value.split()
    #     else:
    #         # Return default variables
    #         return default_vars

    @staticmethod
    def validate_variables_to_regrid(
        config_object, input_data_type, variables_to_regrid
    ):
        """
        Validates the `variables_to_regrid` parameter and returns the list of variables
        if valid. If `variables_to_regrid` is missing, returns the default variable list.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        input_data_type : str
            The input data type (e.g., `'SMAP'`, `'AMSR2'`, `'CIMR'`).
        variables_to_regrid : str
            The XML path to the `variables_to_regrid` parameter.

        Returns
        -------
        list of str
            A list of validated variables to regrid.

        Raises
        ------
        ValueError
            If `input_data_type` is invalid.
            If any variable in `variables_to_regrid` is not in the list of valid inputs.

        Notes
        -----
        - If `variables_to_regrid` is not found or empty, the function returns the default variable list.
        - The function ensures that all provided variables are in the pre-defined list of valid variables.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><variables_to_regrid>bt_h bt_v</variables_to_regrid></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_variables_to_regrid(config_object, "SMAP", "ReGridderParams/variables_to_regrid")
        ['bt_h', 'bt_v']
        """

        # Retrieve the XML element and check if it exists
        element = config_object.find(variables_to_regrid)
        value = element.text.strip() if element is not None and element.text else None

        # Define valid and default variable lists based on input data type
        valid_input, default_vars = None, None

        if input_data_type == "SMAP":
            valid_input = [
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
            ]
            default_vars = valid_input[:]

        elif input_data_type == "AMSR2":
            valid_input = [
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
            ]
            default_vars = valid_input[:]

        elif input_data_type == "CIMR":
            valid_input = [
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
            ]
            default_vars = valid_input[:]

        else:
            raise ValueError(f"Invalid `input_data_type`: {input_data_type}")

        # If value is missing, return the default variables
        if value is None:
            return default_vars

        # Validate each variable in the provided list
        variables = value.split()
        for variable in variables:
            if variable not in valid_input:
                raise ValueError(
                    f"Invalid variable_to_regrid: `{variable}`. Check Configuration File.\n"
                    f"Valid variables_to_regrid: {valid_input}"
                )

        return variables

    @staticmethod
    def validate_max_neighbours(config_object, max_neighbours, regridding_algorithm):
        r"""
        Validates the `max_neighbours` parameter and returns the value if valid.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        max_neighbours : str
            The XML path to the `max_neighbours` parameter.
        regridding_algorithm : str
            The regridding algorithm used.

        Returns
        -------
        int
            The validated `max_neighbours` value.

        Notes
        -----
        - If `regridding_algorithm` is `NN`, the function always returns `1`.
        - If `max_neighbours` is missing, it defaults to `1000`.
        - The function ensures `max_neighbours` is a valid integer.

        Raises
        ------
        ValueError
            If `max_neighbours` is not a valid integer value.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><max_neighbours>500</max_neighbours></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_max_neighbours(config_object, "ReGridderParams/max_neighbours", "Other")
        500
        """

        # The default values here can be tweeked for input data type and Band
        if regridding_algorithm == "NN":
            return 1
        else:
            value = config_object.find(max_neighbours).text

            if value is not None:
                value = int(value)
            else:
                value = 1000

            return value

    @staticmethod
    def validate_source_antenna_method(config_object, source_antenna_method):
        r"""
        Validates the `source_antenna_method` parameter and ensures case insensitivity.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        source_antenna_method : str
            The XML path to the `source_antenna_method` parameter.

        Returns
        -------
        str
            The validated `source_antenna_method` value.

        Notes
        -----
        - The function ensures case insensitivity by converting input values to lowercase.
        - If `source_antenna_method` is missing or empty, it defaults to `instrument`.

        Raises
        ------
        ValueError
            If `source_antenna_method` is not in the list of valid methods.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><Antenna><source_antenna_method>GAUSSIAN</source_antenna_method></Antenna></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_source_antenna_method(config_object, "Antenna/source_antenna_method")
        'gaussian'
        """

        valid_input = ["gaussian", "instrument", "gaussian_projected"]

        value = config_object.find(source_antenna_method).text

        if value is None or value.strip() == "":
            return "instrument"

        value = value.strip().lower()

        if value in valid_input:
            return value
        else:
            raise ValueError(
                f"Invalid antenna method. Check Configuration File. "
                f"Valid antenna methods are: {valid_input}"
            )

    @staticmethod
    def validate_target_antenna_method(config_object, target_antenna_method):
        r"""
        Validates the `target_antenna_method` parameter and ensures case insensitivity.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        target_antenna_method : str
            The XML path to the `target_antenna_method` parameter.

        Returns
        -------
        str
            The validated `target_antenna_method` value.

        Notes
        -----
        - The function ensures case insensitivity by converting input values to lowercase.
        - If `target_antenna_method` is missing or empty, it defaults to `instrument`.

        Raises
        ------
        ValueError
            If `target_antenna_method` is not in the list of valid methods.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><Antenna><target_antenna_method>GAUSSIAN</target_antenna_method></Antenna></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_target_antenna_method(config_object, "Antenna/target_antenna_method")
        'gaussian'
        """
        valid_input = ["gaussian", "instrument", "gaussian_projected"]

        value = config_object.find(target_antenna_method).text

        if value is None or value.strip() == "":
            return "instrument"

        value = value.strip().lower()

        if value in valid_input:
            return value
        else:
            raise ValueError(
                f"Invalid antenna method. Check Configuration File. "
                f"Valid antenna methods are: {valid_input}"
            )

    @staticmethod
    def validate_source_antenna_threshold(config_object, source_antenna_threshold):
        """
        Validates the `source_antenna_threshold` parameter and ensures it is a valid float or integer.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        source_antenna_threshold : str
            The XML path to the `source_antenna_threshold` parameter.

        Returns
        -------
        float or None
            The validated `source_antenna_threshold` value as a float.
            Returns `None` if the value is missing or empty.

        Raises
        ------
        ValueError
            If `source_antenna_threshold` is not a valid numeric value.

        Notes
        -----
        - If `source_antenna_threshold` is missing or empty, it defaults to `None`.
        - If provided, the value must be convertible to a float.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><source_antenna_threshold>9.5</source_antenna_threshold></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_source_antenna_threshold(config_object, "ReGridderParams/source_antenna_threshold")
        9.5
        """
        value = config_object.find(source_antenna_threshold).text

        if value is None or value.strip() == "":
            # We should have a default set of values for each Antenna Pattern
            # For now, I will just choose 9dB
            return None

        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"Invalid antenna threshold: {value}. Check Configuration File."
                f" Antenna threshold must be a float or integer"
            )

    @staticmethod
    def validate_target_antenna_threshold(config_object, target_antenna_threshold):
        """
        Validates the `target_antenna_threshold` parameter and ensures it is a valid float or integer.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        target_antenna_threshold : str
            The XML path to the `target_antenna_threshold` parameter.

        Returns
        -------
        float
            The validated `target_antenna_threshold` value as a float.
            Returns `9.0` if the value is missing or empty.

        Raises
        ------
        ValueError
            If `target_antenna_threshold` is not a valid numeric value.

        Notes
        -----
        - If `target_antenna_threshold` is missing or empty, it defaults to `9.0`.
        - If provided, the value must be convertible to a float.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><target_antenna_threshold>9.5</target_antenna_threshold></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_target_antenna_threshold(config_object, "ReGridderParams/target_antenna_threshold")
        9.5
        """

        value = config_object.find(target_antenna_threshold).text

        if value is None or value.strip() == "":
            # We should have a default set of values for each Antenna Pattern
            # For now, I will just choose 9dB
            return 9.0

        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"Invalid antenna threshold: {value}. Check Configuration File."
                f" Antenna threshold must be a float or integer"
            )

    @staticmethod
    def validate_max_theta_antenna_patterns(config_object, max_theta_antenna_patterns):
        """
        Validates the `max_theta_antenna_patterns` parameter and ensures it is a valid float or integer.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        max_theta_antenna_patterns : str
            The XML path to the `max_theta_antenna_patterns` parameter.

        Returns
        -------
        float or None
            The validated `max_theta_antenna_patterns` value as a float.
            Returns `None` if the value is missing or empty.

        Raises
        ------
        ValueError
            If `max_theta_antenna_patterns` is not a valid numeric value.

        Notes
        -----
        - If `max_theta_antenna_patterns` is missing or empty, it defaults to `None`.
        - If provided, the value must be convertible to a float.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><max_theta_antenna_patterns>40.0</max_theta_antenna_patterns></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_max_theta_antenna_patterns(config_object, "ReGridderParams/max_theta_antenna_patterns")
        40.0
        """

        value = config_object.find(max_theta_antenna_patterns).text

        if value is None or value.strip() == "":
            # We should have a default set of values for each Antenna Pattern
            # For now, I will just choose 40.
            return None

        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"Invalid max theta for antenna patterns: {value}. Check Configuration File."
                f"Max theta for antenna patterns must be a float or integer"
            )

    @staticmethod
    def validate_polarisation_method(config_object, polarisation_method):
        """
        Validates the `polarisation_method` parameter and ensures it is a valid string input.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        polarisation_method : str
            The XML path to the `polarisation_method` parameter.

        Returns
        -------
        str
            The validated `polarisation_method` value.
            Defaults to `'scalar'` if the value is missing or empty.

        Raises
        ------
        ValueError
            If `polarisation_method` is not a recognized valid method.

        Notes
        -----
        - If `polarisation_method` is missing or empty, it defaults to `'scalar'`.
        - Recognized valid values include `'scalar'` and `'mueller'`.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><polarisation_method>mueller</polarisation_method></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_polarisation_method(config_object, "ReGridderParams/polarisation_method")
        'mueller'
        """

        valid_input = ["scalar", "mueller"]

        value = config_object.find(polarisation_method).text

        if value is None or value.strip() == "":
            return "scalar"

        if value not in valid_input:
            raise ValueError(
                f"Invalid polarisation method: `{value}`. Check Configuration File."
                f" Valid polarisation methods are: `{valid_input}`."
            )
        else:
            return value

    @staticmethod
    def validate_boresight_shift(config_object, boresight_shift, input_data_type):
        """
        Validates the `boresight_shift` parameter and ensures it is case insensitive.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        boresight_shift : str
            The XML path to the `boresight_shift` parameter.
        input_data_type : str
            The type of input data being processed.

        Returns
        -------
        bool
            `True` if boresight shift is enabled, `False` otherwise.

        Raises
        ------
        ValueError
            If `boresight_shift` is not a recognized boolean value.

        Notes
        -----
        - The function ensures case insensitivity by converting input values to lowercase.
        - If `boresight_shift` is missing or empty, it defaults to `False`.
        - Recognized valid values include `'true'` and `'false'`.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><boresight_shift>TRUE</boresight_shift></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_boresight_shift(config_object, "ReGridderParams/boresight_shift", "SMAP")
        True
        """

        if input_data_type != "SMAP":
            return False

        value = config_object.find(boresight_shift).text
        valid_input = ["true", "false"]

        if value is None or value.strip() == "":
            return False

        value = value.strip().lower()

        if value in valid_input:
            # returns either True or False
            return value == "true"
            # if value == "true":
            #     return True
            # else:
            #     return False
        else:
            raise ValueError(
                f"Invalid boresight shift: `{value}`. Check Configuration File. "
                f"Valid boresight shift values are: `{valid_input}`."
            )

    # TODO:
    # - Add a proper validation to check if the indices actually fall
    #   within the grid that the user wants to check.
    # - Also need to add L1r
    @staticmethod
    def validate_reduced_grid_inds(config_object, reduced_grid_inds):
        """
        Validates the `reduced_grid_inds` parameter and ensures it follows the expected format.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        reduced_grid_inds : str
            The XML path to the `reduced_grid_inds` parameter.

        Returns
        -------
        list of int or None
            A list of four integers `[row_min, row_max, col_min, col_max]` if valid.
            Returns `None` if the value is missing or empty.

        Raises
        ------
        ValueError
            If `reduced_grid_inds` does not contain exactly four valid integers.
            If any grid indices are negative.
            If `row_min` > `row_max` or `col_min` > `col_max`.

        Notes
        -----
        - The expected format is four space-separated integers.
        - If `reduced_grid_inds` is missing or empty, it defaults to `None`.
        - Grid indices must be non-negative integers, and row/column minimums must not exceed maximums.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><GridParams><reduced_grid_inds>0 10 0 10</reduced_grid_inds></GridParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_reduced_grid_inds(config_object, "GridParams/reduced_grid_inds")
        [0, 10, 0, 10]
        """

        value = config_object.find(reduced_grid_inds).text

        if value is None or value.strip() == "":
            return None
        else:
            value = value.split()

        try:
            if len(value) != 4:
                raise ValueError(
                    "Invalid reduced_grid_inds format. Expected 4 integers (row_min, row_max, col_min, col_max)."
                )

            grid_row_min = int(value[0])
            grid_row_max = int(value[1])
            grid_col_min = int(value[2])
            grid_col_max = int(value[3])

            if (
                grid_row_min < 0
                or grid_row_max < 0
                or grid_col_min < 0
                or grid_col_max < 0
            ):
                raise ValueError("Grid indices must be non-negative integers.")

            if grid_row_min > grid_row_max:
                raise ValueError("grid_row_min cannot be greater than grid_row_max.")

            if grid_col_min > grid_col_max:
                raise ValueError("grid_col_min cannot be greater than grid_col_max.")

            return [grid_row_min, grid_row_max, grid_col_min, grid_col_max]

        except ValueError:
            raise ValueError(
                f"Invalid `reduced_grid_inds`: {value}. Ensure it contains 4 valid integers "
                "(row_min, row_max, col_min, col_max)."
            )

    @staticmethod
    def validate_source_gaussian_params(config_object, source_gaussian_params):
        """
        Validates the `source_gaussian_params` parameter and ensures it consists of exactly two numeric values.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        source_gaussian_params : str
            The XML path to the `source_gaussian_params` parameter.

        Returns
        -------
        list of float
            A list containing two float values representing the source Gaussian parameters.

        Raises
        ------
        ValueError
            If `source_gaussian_params` is missing, empty, or does not contain exactly two valid numbers.

        Notes
        -----
        - The function ensures that only two numeric values are provided.
        - If `source_gaussian_params` is missing or empty, an exception is raised.
        - Non-numeric values will also trigger a `ValueError`.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><source_gaussian_params>1.5 2.5</source_gaussian_params></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_source_gaussian_params(config_object, "ReGridderParams/source_gaussian_params")
        [1.5, 2.5]
        """
        element = config_object.find(source_gaussian_params)
        if element is None or element.text is None or element.text.strip() == "":
            raise ValueError(
                "Missing source Gaussian parameters in the configuration file."
            )

        params = element.text.split()

        if len(params) != 2:
            raise ValueError(
                "Invalid source gaussian parameters. Check Configuration File. "
                "There should be exactly 2 parameters for the source gaussian."
            )

        try:
            float_params = [float(param) for param in params]
        except ValueError:
            raise ValueError(
                "Invalid parameter: All parameters must be valid numbers (int or float)."
            )

        return float_params

    @staticmethod
    def validate_target_gaussian_params(config_object, target_gaussian_params):
        """
        Validates the `target_gaussian_params` parameter and ensures it consists of exactly two numeric values.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        target_gaussian_params : str
            The XML path to the `target_gaussian_params` parameter.

        Returns
        -------
        list of float
            A list containing two float values representing the target Gaussian parameters.

        Raises
        ------
        ValueError
            If `target_gaussian_params` is missing, empty, or does not contain exactly two valid numbers.

        Notes
        -----
        - The function ensures that only two numeric values are provided.
        - If `target_gaussian_params` is missing or empty, an exception is raised.
        - Non-numeric values will also trigger a `ValueError`.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><target_gaussian_params>1.5 2.5</target_gaussian_params></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_target_gaussian_params(config_object, "ReGridderParams/target_gaussian_params")
        [1.5, 2.5]
        """

        element = config_object.find(target_gaussian_params)
        if element is None or element.text is None or element.text.strip() == "":
            raise ValueError(
                "Missing target Gaussian parameters in the configuration file."
            )

        params = element.text.split()

        if len(params) != 2:
            raise ValueError(
                "Invalid target gaussian parameters. Check Configuration File. "
                "There should be exactly 2 parameters for the target gaussian."
            )

        try:
            float_params = [float(param) for param in params]
        except ValueError:
            raise ValueError(
                "Invalid parameter: All parameters must be valid numbers (int or float)."
            )

        return float_params

    @staticmethod
    def validate_rsir_iteration(config_object, rsir_iteration):
        """
        Validates the rSIR iteration count.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        rsir_iteration : str
            The XML path to the `rsir_iteration` parameter.

        Returns
        -------
        int
            A non-negative integer representing the RSIR iteration count.

        Raises
        ------
        ValueError
            If the value is missing, not a valid integer, or negative.

        Notes
        -----
        - The function ensures the iteration count is a non-negative integer.
        - If `rsir_iteration` is missing or empty, an exception is raised.
        - Float or non-numeric values will also trigger a `ValueError`.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><rsir_iteration>10</rsir_iteration></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_rsir_iteration(config_object, "ReGridderParams/rsir_iteration")
        10
        """
        element = config_object.find(rsir_iteration)
        if element is None or element.text is None or element.text.strip() == "":
            raise ValueError("Missing rSIR iteration value in the configuration file.")

        try:
            iteration = int(element.text.strip())
        except ValueError:
            raise ValueError("Invalid rSIR iteration value. It must be an integer.")

        if iteration < 0:
            raise ValueError("rSIR iteration value must be a non-negative integer.")

        return iteration

    @staticmethod
    def validate_max_iterations(config_object, max_iterations):
        """
        Validates the maximum number of iterations.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        max_iterations : str
            The XML path to the `max_number_iteration` parameter.

        Returns
        -------
        int
            A non-negative integer representing the maximum number of iterations.

        Raises
        ------
        ValueError
            If the value is missing, not a valid integer, or negative.

        Notes
        -----
        - The function ensures the iteration count is a non-negative integer.
        - If `max_iterations` is missing or empty, an exception is raised.
        - Float or non-numeric values will also trigger a `ValueError`.


        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><max_iterations>10</max_iterations></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_iterations(config_object, "ReGridderParams/max_iterations")
        10
        """
        element = config_object.find(max_iterations)
        if element is None or element.text is None or element.text.strip() == "":
            raise ValueError(
                "Missing maximum number of iteration value in the configuration file."
            )

        try:
            max_iterations = int(element.text.strip())
        except ValueError:
            raise ValueError(
                "Invalid maximum number of iteration value. It must be a non-negative integer."
            )

        if max_iterations < 0:
            raise ValueError(
                "Maximum number of iterations must be a non-negative integer."
            )

        return max_iterations

    @staticmethod
    def validate_relative_tolerance(config_object, relative_tolerance):
        """
        Validates the relative tolerance value.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        relative_tolerance : str
            The XML path to the `relative_tolerance` parameter.

        Returns
        -------
        float
            A non-negative float representing the relative tolerance.

        Raises
        ------
        ValueError
            If the value is missing, not a valid float, or negative.

        Notes
        -----
        - The function ensures the tolerance value is a non-negative float.
        - If `relative_tolerance` is missing or empty, an exception is raised.
        - Non-numeric or negative values will also trigger a `ValueError`.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><relative_tolerance>0.01</relative_tolerance></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_relative_tolerance(config_object, "ReGridderParams/relative_tolerance")
        0.01
        """

        element = config_object.find(relative_tolerance)
        if element is None or element.text is None or element.text.strip() == "":
            raise ValueError(
                "Missing relative tolerance value in the configuration file."
            )

        try:
            tolerance = float(element.text.strip())
        except ValueError:
            raise ValueError(
                "Invalid relative tolerance value. It must be a non-negative float."
            )

        if tolerance < 0:
            raise ValueError("Relative tolerance must be a non-negative float.")

        return tolerance

    @staticmethod
    def validate_regularisation_parameter(config_object, regularisation_parameter):
        """
        Validates the regularisation parameter value.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        regularization_parameter : str
            The XML path to the `regularization_parameter` parameter.

        Returns
        -------
        float
            A float representing the regularization parameter.

        Raises
        ------
        ValueError
            If the value is missing or not a valid float.

        Notes
        -----
        - The function ensures the parameter is a valid float.
        - If `regularisation_parameter` is missing or empty, an exception is raised.
        - Non-numeric values will also trigger a `ValueError`.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><regularisation_parameter>0.1</regularisation_parameter></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_regularisation_parameter(config_object, "ReGridderParams/regularisation_parameter")
        0.1
        """

        element = config_object.find(regularisation_parameter)
        if element is None or element.text is None or element.text.strip() == "":
            raise ValueError(
                "Missing regularisation parameter value in the configuration file."
            )

        try:
            return float(element.text.strip())
        except ValueError:
            raise ValueError(
                f"Invalid regularisation parameter: {element.text}. It must be a valid float."
            )

    @staticmethod
    def validate_MRF_grid_definition(config_object, MRF_grid_definition):
        """
        Validates the MRF grid definition parameter from the configuration file.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        MRF_grid_definition : str
            The XML path to the `MRF_grid_definition` parameter.

        Returns
        -------
        str
            A valid MRF grid definition.

        Raises
        ------
        ValueError
            If the value is missing or not a valid grid definition.

        Notes
        -----
        - Ensures the provided `MRF_grid_definition` value is among the valid definitions.
        - If `MRF_grid_definition` is missing or contains an invalid value, a `ValueError` is raised.
        - Checks for an exact match without trailing spaces.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><MRF_grid_definition>EASE2_G3km</MRF_grid_definition></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_MRF_grid_definition(config_object, "ReGridderParams/MRF_grid_definition")
        'EASE2_G3km'
        """

        element = config_object.find(MRF_grid_definition)
        if element is None or element.text is None or element.text.strip() == "":
            raise ValueError(
                "Missing MRF grid definition value in the configuration file."
            )

        value = element.text.strip()
        valid_input = [
            "EASE2_G3km",
            "EASE2_G1km",
            "EASE2_G9km",
            "EASE2_N1km",
            "EASE2_N9km",
            "EASE2_S9km",
            "EASE2_G36km",
            "EASE2_N36km",
            "EASE2_S36km",
            "STEREO_N25km",
            "STEREO_S25km",
            "EASE2_N3km",
            "EASE2_S3km",
        ]

        if value in valid_input:
            return value

        raise ValueError(
            f"Invalid Grid Definition, check configuration file. "
            f"Valid grid definitions are: {valid_input}"
        )

    @staticmethod
    def validate_MRF_projection_definition(config_object, MRF_projection_definition):
        """
        Validates the MRF projection definition parameter from the configuration file.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        MRF_projection_definition : str
            The XML path to the `MRF_projection_definition` parameter.

        Returns
        -------
        str
            A valid projection definition ('G', 'N', or 'S').

        Raises
        ------
        ValueError
            If the value is missing, blank, or not in the list of valid projection definitions.

        Notes
        -----
        - Ensures the provided `MRF_projection_definition` value is among the valid definitions ('G', 'N', 'S').
        - If `MRF_projection_definition` is missing or contains an invalid value, a `ValueError` is raised.
        - Leading/trailing spaces are stripped before validation.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><MRF_projection_definition>G</MRF_projection_definition></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_MRF_projection_definition(config_object, "ReGridderParams/MRF_projection_definition")
        'G'
        """

        valid_input = ["G", "N", "S"]
        element = config_object.find(MRF_projection_definition)
        if element is None or element.text is None or element.text.strip() == "":
            raise ValueError(
                f"Missing or blank MRF projection definition in the configuration file. "
                f"Ensure a valid projection is specified, i.e.: {valid_input}."
            )

        value = element.text.strip()

        if value in valid_input:
            return value

        raise ValueError(
            f"Invalid Projection Definition, check configuration file. "
            f"Valid projection definitions are: {valid_input}"
        )

    @staticmethod
    def validate_bg_smoothing(config_object, bg_smoothing):
        """
        Validates the `bg_smoothing` parameter from the configuration file.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        bg_smoothing : str
            The XML path to the `bg_smoothing` parameter.

        Returns
        -------
        float
            A valid `bg_smoothing` value. Defaults to `0.0` if missing.

        Raises
        ------
        ValueError
            If the value is not a valid float.

        Notes
        -----
        - If the value is missing, it defaults to `0.0`.
        - Strips leading/trailing spaces before parsing.
        - Raises `ValueError` for invalid numeric input.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><ReGridderParams><bg_smoothing>1.5</bg_smoothing></ReGridderParams></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_bg_smoothing(config_object, "ReGridderParams/bg_smoothing")
        1.5
        """

        element = config_object.find(bg_smoothing)
        if element is None or element.text is None or element.text.strip() == "":
            return 0.0

        try:
            return float(element.text.strip())
        except ValueError as e:
            raise ValueError(
                "Invalid `bg_smoothing` value. It must be a valid float."
            ) from e

    @staticmethod
    def validate_quality_control(config_object, quality_control, input_data_type):
        """
        Validates the `quality_control` parameter based on the input data type and configuration file.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            Root element of the XML configuration file.
        quality_control : str
            The XML path to the `quality_control` parameter.
        input_data_type : str
            The type of input data (e.g., 'AMSR2', 'CIMR', or another type).

        Returns
        -------
        bool
            `False` for 'AMSR2' and 'CIMR'.
            `True` or `False` based on the configuration file for other input data types.

        Raises
        ------
        ValueError
            If the value is not 'True' or 'False' (case insensitive) for input data types other than 'AMSR2' and 'CIMR'.

        Notes
        -----
        - If `input_data_type` is 'AMSR2' or 'CIMR', this method always returns `False`.
        - For other data types, the method validates the value from the configuration file.
        - If the value is missing or invalid, a `ValueError` is raised.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = '''<config><InputData><quality_control>true</quality_control></InputData></config>'''
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_quality_control(config_object, "InputData/quality_control", "Other")
        True
        """

        if input_data_type in ["AMSR2", "CIMR"]:
            return False

        valid_input = ["true", "false"]

        element = config_object.find(quality_control)

        if element is None or element.text is None:
            raise ValueError(
                "Invalid `quality_control` value: None. Check Configuration File."
            )

        value = element.text.strip().lower()
        if value in valid_input:
            return value == "true"

        raise ValueError(
            f"Invalid `quality_control` value: {value}. Check Configuration File. "
            f"Valid inputs for `quality_control` parameter are: {valid_input}"
        )

    @staticmethod
    def validate_antenna_pattern_uncertainty(
        config_object, antenna_pattern_uncertainty
    ):
        """
        Validates the `antenna_pattern_uncertainty` parameter from the configuration file.

        Parameters
        ----------
        config_object : xml.etree.ElementTree.Element
            The root XML element containing the configuration.
        antenna_pattern_uncertainty : str
            The XML path to the `antenna_pattern_uncertainty` parameter.

        Returns
        -------
        float
            A float representing the `antenna_pattern_uncertainty` value.
            Defaults to `0.0` if the value is missing.

        Raises
        ------
        ValueError
            If the value is not a valid float.

        Examples
        --------
        >>> import xml.etree.ElementTree as ET
        >>> xml_data = "<config><Uncertainty><antenna_pattern_uncertainty>1.5</antenna_pattern_uncertainty></Uncertainty></config>"
        >>> config_object = ET.ElementTree(ET.fromstring(xml_data)).getroot()
        >>> ConfigFile.validate_antenna_pattern_uncertainty(config_object, "Uncertainty/antenna_pattern_uncertainty")
        1.5
        """

        element = config_object.find(antenna_pattern_uncertainty)

        if element is None or element.text is None or element.text.strip() == "":
            return 0.0  # Default to 0.0

        try:
            return float(
                element.text.strip()
            )  # Convert to float after stripping whitespace
        except ValueError:
            raise ValueError(
                "Invalid `antenna_pattern_uncertainty` value. It must be a valid float."
            )  # from e

    @staticmethod
    def validate_cimr_nedt(config_object):
        cimr_nedt = {"L": 0.3, "C": 0.2, "X": 0.3, "KA": 0.4, "K": 0.7}

        for nedt in ["L", "C", "X", "K", "KA"]:
            value = config_object.find(f"ReGridderParams/cimr_{nedt}_nedt").text
            if value is None or value.strip() == "":
                continue
            try:
                value = float(value)
                if value < 0:
                    raise ValueError(f"{nedt} must be a non-negative float.")
                cimr_nedt[nedt] = value
            except ValueError as e:
                raise ValueError(
                    f"Invalid {nedt} value. It must be a non-negative float."
                ) from e
        return cimr_nedt

    @staticmethod
    def validate_max_chunk_size(config_object, max_chunk_size):
        value = config_object.find(max_chunk_size).text
        if value is not None:
            try:
                value = int(value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid max_chunk_size value: {value}. It must be a valid integer > 0."
                ) from e
        else:
            raise ValueError("Missing max_chunk_size value in the configuration file.")
        return value

    @staticmethod
    def validate_chunk_buffer(config_object, chunk_buffer):
        value = config_object.find(chunk_buffer).text
        if value is not None:
            try:
                value = float(value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid chunk_buffer value: {value}. It must be a valid float > 0."
                ) from e
        else:
            raise ValueError("Missing chunk_buffer value in the configuration file.")
        return value
