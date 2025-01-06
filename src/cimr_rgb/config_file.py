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
        self.timestamp = config_object.find("OutputData/timestamp").text
        self.timestamp_fmt = config_object.find("OutputData/timestamp_fmt").text
        if self.timestamp is None or self.timestamp.strip() == "":
            # Getting the current time stamp to propagate into the software
            self.timestamp = datetime.datetime.now()

            # Format the date and time as "YYYY-MM-DD_HH-MM-SS"
            self.timestamp = self.timestamp.strftime(self.timestamp_fmt)
            # config_object.find("OutputData/timestamp").text = timestamp_elem

        # l1c_utc_time = datetime.datetime.now()

        # Format the date and time as "YYYY-MM-DD_HH-MM-SS"
        # self.timestamp = l1c_utc_time.strftime(self.timestamp_fmt)

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
                f"Invalid value for `decorate` encountered. \nThe `decorate` parameter can either be `True` or `False`."
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
            grid_definition=self.grid_definition,
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
            self.LMT = 2200
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
                "KU": 8,
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
                "KU": (74, 7692 * 8),
            }
            self.nedt = {"L": 0.3, "C": 0.2, "X": 0.3, "KA": 0.4, "KU": 0.7}

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
            self.max_number_iteration = self.validate_max_number_iteration(
                config_object=config_object,
                max_number_iteration="ReGridderParams/max_number_iteration",
            )
            self.relative_tolerance = self.validate_relative_tolerance(
                config_object=config_object,
                relative_tolerance="ReGridderParams/relative_tolerance",
            )
            self.regularization_parameter = self.validate_regularization_parameter(
                config_object=config_object,
                regularization_parameter="ReGridderParams/regularization_parameter",
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
        """
        Validates the input data type and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        input_data_type: str
            Path to the input data type in the configuration file

        Returns
        -------
        str
            Validated input data type
        """

        valid_input = ["AMSR2", "SMAP", "CIMR"]

        if config_object.find(input_data_type).text in valid_input:
            return config_object.find(input_data_type).text

        raise ValueError(
            f"Invalid input data type. Valid input data types are: {valid_input}"
        )

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

        input_data_path = (config_object.find(input_data_path).text).strip()
        input_data_path = grasp_io.resolve_config_path(path_string=input_data_path)
        print(input_data_path)

        if input_data_path.exists():
            return input_data_path
        else:
            raise FileNotFoundError(
                f"File\n {input_data_path}\n not found. Check file location."
            )

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
        )  # .resolve()
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
                        f"Error: SMAP Antenna Pattern not found in dpr"
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

        if input_data_type == "SMAP":
            valid_input = ["L1C"]
        else:
            valid_input = ["L1C", "L1R"]

        if config_object.find(grid_type).text in valid_input:
            return config_object.find(grid_type).text

        raise ValueError(
            f"Invalid Grid Type. Check Configuration File. Valid grid types are:"
            f" {valid_input} for {input_data_type} data."
        )

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
                valid_input = ["L", "C", "X", "KA", "KU", "All"]
            elif grid_type == "L1R":
                valid_input = ["L", "C", "X", "KA", "KU"]
            config_input = config_object.find(target_band).text.split()
            if config_input == ["All"]:
                return ["L", "C", "X", "KA", "KU"]
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
            valid_input = ["L", "C", "X", "KA", "KU"]

        if all(
            item in valid_input
            for item in config_object.find(source_band).text.split()
        ):
            return config_object.find(source_band).text.split()
        else:
            raise ValueError(
                f"Invalid Source Band, check configuration file. "
                f"Valid source bands are: {valid_input}."
            )

    @staticmethod
    def validate_grid_definition(config_object, grid_definition):
        """
        Validates the grid definition and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        grid_type: str
            User selected Grid Type in the configuration file
        grid_definition: str
            Path to the grid definition in the configuration file

        Returns
        -------
        str
            Validated grid definition
        """

        valid_input = [
            "EASE2_G9km",
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

        if config_object.find(grid_definition).text in valid_input:
            return config_object.find(grid_definition).text
        raise ValueError(
            f"Invalid Grid Definition, check configuration file. "
            f"Valid grid definitions are: {valid_input}"
        )

    # TODO: Check the docstring, seems to have an incorrect description
    @staticmethod
    def validate_projection_definition(
        config_object, grid_definition, projection_definition
    ):
        """
        Validates the projection definition and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        grid_definition: str
            User selected Grid Definition in the configuration file
        projection_definition: str
            Path to the projection definition in the configuration file

        Returns
        -------
        str
            Validated projection definition
        """

        if grid_definition:
            if "EASE2" in grid_definition:
                valid_input = ["G", "N", "S"]

            elif "STEREO" in grid_definition:
                valid_input = ["PS_N", "PS_S"]

            elif "MERC" in grid_definition:
                valid_input = ["MERC_G"]

            proj_val = config_object.find(projection_definition).text
            if proj_val in valid_input:
                return proj_val  # config_object.find(projection_definition).text
            raise ValueError(
                f"Grid Definiton `{grid_definition}` received invalid projection definition: `{proj_val}`; "
                f"check configuration file."
                f" Valid projection definitions are: `{valid_input}`"
            )
        else:
            return None

    @staticmethod
    def validate_regridding_algorithm(config_object, regridding_algorithm):
        """
        Validates the regridding algorithm and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        regridding_algorithm: str
            Path to the regridding algorithm in the configuration file

        Returns
        -------
        str
            Validated regridding algorithm
        """
        valid_input = ["NN", "DIB", "IDS", "BG", "RSIR", "LW", "CG"]
        if config_object.find(regridding_algorithm).text in valid_input:
            return config_object.find(regridding_algorithm).text
        raise ValueError(
            f"Invalid regridding algorithm. Check Configuration File."
            f" Valid regridding algorithms are: {valid_input}"
        )

    @staticmethod
    def validate_split_fore_aft(config_object, split_fore_aft, input_data_type):
        """
        Validates the split fore aft and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        split_fore_aft: str
            Path to the split fore aft in the configuration file

        Returns
        -------
        str
            Validated split fore aft
        """
        if input_data_type == "AMSR2":
            return False
        valid_input = ["True", "False"]
        if config_object.find(split_fore_aft).text in valid_input:
            if config_object.find(split_fore_aft).text == "True":
                return True
            else:
                return False
        raise ValueError(
            f"Invalid split fore aft. Check Configuration File."
            f" Valid split fore aft are: {valid_input}"
        )

    @staticmethod
    def validate_save_to_disk(config_object, save_to_disk):
        # value = bool(config_object.find(save_to_disk).text)
        value = config_object.find(save_to_disk).text

        # if value is not True and value is not False:
        if value not in ["True", "False"]:
            raise ValueError(
                f"Invalid `save_to_disk`. Check Configuration File."
                f" `save_to_disk` must be either True or False"
            )
        elif value == "True":
            value = True
        elif value == "False":
            value = False

        return value

    @staticmethod
    def validate_search_radius(
        config_object, search_radius, grid_definition, grid_type, input_data_type
    ):
        """
        Validates the search radius and returns the value if valid

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        search_radius: str
            Path to the search radius in the configuration file

        Returns
        -------
        int
            Validated search radius
        """
        value = config_object.find(search_radius).text

        if value is None or value.strip() == "":
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

        return value

    @staticmethod
    def get_scan_geometry(config, band_to_remap=None):
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
            elif band_to_remap == "KU":
                num_scans = 74
                num_earth_samples = 7692 * 8
        # else:
        #     raise ValueError(f"Invalid `input_data_type`: {input_data_type}")

        return num_scans, num_earth_samples

    @staticmethod
    def validate_variables_to_regrid(
        config_object, input_data_type, variables_to_regrid
    ):
        value = config_object.find(variables_to_regrid).text

        if input_data_type == 'SMAP':
            valid_input = ['bt_h', 'bt_v', 'bt_3', 'bt_4',
                         'processing_scan_angle', 'longitude', 'latitude', 'faraday_rot_angle', 'nedt_h',
                           'nedt_v', 'nedt_3', 'nedt_4', 'regridding_n_samples', 'regridding_l1b_orphans',
                           'acq_time_utc', 'azimuth']

            default_vars = ['bt_h', 'bt_v', 'bt_3', 'bt_4',
                         'processing_scan_angle', 'longitude', 'latitude', 'faraday_rot_angle', 'nedt_h',
                           'nedt_v', 'nedt_3', 'nedt_4', 'regridding_n_samples', 'regridding_l1b_orphans',
                           'acq_time_utc', 'azimuth']

        elif input_data_type == 'AMSR2':

            valid_input = ['bt_h', 'bt_v', 'longitude', 'latitude', 'regridding_n_samples',
                           'x_position', 'y_position', 'z_position', 'x_velocity',
                           'y_velocity', 'z_velocity', 'azimuth', 'solar_azimuth', 'acq_time_utc']

            default_vars = ['bt_h', 'bt_v', 'longitude', 'latitude', 'regridding_n_samples',
                           'x_position', 'y_position', 'z_position', 'x_velocity',
                           'y_velocity', 'z_velocity', 'azimuth', 'solar_azimuth', 'acq_time_utc']

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

            default_vars = [
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
        else:
            raise ValueError(f"Invalid `input_data_type`: {input_data_type}")

        if value is not None:
            for variable in value.split():
                if variable not in valid_input:
                    raise ValueError(
                        f"Invalid variable_to_regrid. Check Configuration File."
                        f" Valid variables_to_regrid: {valid_input}"
                    )
            return value.split()
        else:
            # Return default variables
            return default_vars

    @staticmethod
    def validate_max_neighbours(config_object, max_neighbours, regridding_algorithm):
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
        valid_input = ["gaussian", "real", "gaussian_projected"]

        value = config_object.find(source_antenna_method).text

        if value is None or value.strip() == "":
            return "real"
        elif value in valid_input:
            return value
        else:
            raise ValueError(
                f"Invalid antenna method. Check Configuration File."
                f" Valid antenna methods are: {valid_input}"
            )

    @staticmethod
    def validate_target_antenna_method(config_object, target_antenna_method):
        valid_input = ["gaussian", "real", "gaussian_projected"]

        value = config_object.find(target_antenna_method).text

        if value is None or value.strip() == "":
            return "real"
        elif value in valid_input:
            return value
        else:
            raise ValueError(
                f"Invalid antenna method. Check Configuration File."
                f" Valid antenna methods are: {valid_input}"
            )

    @staticmethod
    def validate_source_antenna_threshold(config_object, source_antenna_threshold):
        value = config_object.find(source_antenna_threshold).text

        if value is None or value.strip() == "":
            # We should have a default set of values for each Antenna Pattern
            # For now, I will just choose 9dB
            return None

        try:
            return float(value)

        except:
            raise ValueError(
                f"Invalid antenna threshold: {value}. Check Configuration File."
                f" Antenna threshold must be a float or integer"
            )

    @staticmethod
    def validate_target_antenna_threshold(config_object, target_antenna_threshold):
        value = config_object.find(target_antenna_threshold).text

        if value is None or value.strip() == "":
            # We should have a default set of values for each Antenna Pattern
            # For now, I will just choose 9dB
            return 9.0

        try:
            return float(value)
        except:
            raise ValueError(
                f"Invalid antenna threshold: {value}. Check Configuration File."
                f" Antenna threshold must be a float or integer"
            )

    @staticmethod
    def validate_max_theta_antenna_patterns(config_object, max_theta_antenna_patterns):
        value = config_object.find(max_theta_antenna_patterns).text

        if value is None or value.strip() == "":
            # We should have a default set of values for each Antenna Pattern
            # For now, I will just choose 40.
            return None

        try:
            return float(value)

        except:
            raise ValueError(
                f"Invalid max theta for antenna patterns: {value}. Check Configuration File."
                f"Max theta for antenna patterns must be a float or integer"
            )

    @staticmethod
    def validate_polarisation_method(config_object, polarisation_method):
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
        if input_data_type != "SMAP":
            return False

        value = config_object.find(boresight_shift).text
        valid_input = ["True", "False"]

        if value is None or value.strip() == "":
            return False

        elif value not in valid_input:
            raise ValueError(
                f"Invalid boresight shift: `{value}`. Check Configuration File."
                f" Valid boresight shift are: `{valid_input}`."
            )
        else:
            if value == "True":
                return True
            else:
                return False

    # TODO:
    # - Add a proper validation to check if the indices actually fall
    #   within the grid that the user wants to check.
    # - Also need to add L1r
    @staticmethod
    def validate_reduced_grid_inds(config_object, reduced_grid_inds):
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
        # We should add default values.
        value = config_object.find(source_gaussian_params).text

        # TODO: Do we need this check here?
        if value is None or value.strip() == "":
            raise ValueError(
                "Missing source Gaussian parameters in the configuration file."
            )

        params = value.split()

        # Check we only have 2 params
        if len(params) != 2:
            raise ValueError(
                f"Invalid source gaussian parameters. Check Configuration File."
                f" There should be 2 parameters for the source gaussian"
            )

        try:
            float_params = [float(param) for param in params]

        except ValueError as e:
            raise ValueError(
                "Invalid parameter: All parameters must be valid numbers (int or float)."
            ) from e

        return float_params

    # TODO: Addd default values
    @staticmethod
    def validate_target_gaussian_params(config_object, target_gaussian_params):
        # We should add default values.
        value = config_object.find(target_gaussian_params).text

        if value is None or value.strip() == "":
            raise ValueError(
                "Missing target Gaussian parameters in the configuration file."
            )

        params = value.split()

        # Check we only have 2 params
        if len(params) != 2:
            raise ValueError(
                f"Invalid target gaussian parameters. Check Configuration File."
                f" There should be 2 parameters for the target gaussian"
            )

        try:
            float_params = [float(param) for param in params]
        except ValueError as e:
            raise ValueError(
                "Invalid parameter: All parameters must be valid numbers (int or float)."
            ) from e

        return float_params

    @staticmethod
    def validate_rsir_iteration(config_object, rsir_iteration):
        """
        Validates rSIR iteration count.

        Parameters:
        - config_object: XML configuration object.
        - rsir_iteration: Path to the rsir_iteration parameter in the configuration.

        Returns:
        - An integer representing the RSIR iteration count.

        Raises:
        - ValueError: If the value is missing, not a valid integer, or negative.
        """

        try:
            value = config_object.find(rsir_iteration).text

            if value is None or value.strip() == "":
                raise ValueError(
                    "Missing rSIR iteration value in the configuration file."
                )

            iteration = int(value)

            if iteration < 0:
                raise ValueError("rSIR iteration value must be a non-negative integer.")

            return iteration

        except ValueError as e:
            raise ValueError(
                "Invalid rSIR iteration value. It must be an integer."
            ) from e

    @staticmethod
    def validate_max_number_iteration(config_object, max_number_iteration):
        """
        Validates the maximum number of iterations.

        Parameters:
        - config_object: XML configuration object.
        - max_number_iteration: Path to the max_number_iteration parameter in the configuration.

        Returns:
        - An integer representing the maximum number of iterations.

        Raises:
        - ValueError: If the value is missing, not a valid integer, or negative.
        """

        try:
            value = config_object.find(max_number_iteration).text
            if value is None:
                raise ValueError(
                    "Missing maximum number of iteration value in the configuration file."
                )

            max_iterations = int(value)
            if max_iterations < 0:
                raise ValueError(
                    "Maximum number of iterations must be a non-negative integer."
                )

            return max_iterations
        except ValueError as e:
            raise ValueError(
                "Invalid maximum number of iteration value. It must be a non-negative integer."
            ) from e

    @staticmethod
    def validate_relative_tolerance(config_object, relative_tolerance):
        """
        Validates the relative tolerance value.

        Parameters:
        - config_object: XML configuration object.
        - relative_tolerance: Path to the relative_tolerance parameter in the configuration.

        Returns:
        - A float representing the relative tolerance.

        Raises:
        - ValueError: If the value is missing, not a valid float, or negative.
        """

        try:
            value = config_object.find(relative_tolerance).text
            if value is None:
                raise ValueError(
                    "Missing relative tolerance value in the configuration file."
                )

            tolerance = float(value)
            if tolerance < 0:
                raise ValueError("Relative tolerance must be a non-negative float.")

            return tolerance
        except ValueError as e:
            raise ValueError(
                "Invalid relative tolerance value. It must be a non-negative float."
            ) from e

    @staticmethod
    def validate_regularization_parameter(config_object, regularization_parameter):
        """
        Validates the regularization parameter value.

        Parameters:
        - config_object: XML configuration object.
        - regularization_parameter: Path to the regularization_parameter in the configuration.

        Returns:
        - A float representing the regularization parameter.

        Raises:
        - ValueError: If the value is missing or not a valid float.
        """

        try:
            value = config_object.find(regularization_parameter).text

            if value is None:
                raise ValueError(
                    "Missing regularization parameter value in the configuration file."
                )

            return float(value)

        except ValueError as e:
            value = config_object.find(regularization_parameter).text
            raise ValueError(
                f"Invalid regularization parameter: {value}. It must be a valid float."
            ) from e

    # TODO: Figure out whether we need this try except statement
    @staticmethod
    def validate_MRF_grid_definition(config_object, MRF_grid_definition):
        """
        Validates the MRF grid definition parameter from the configuration file.

        Parameters:
        - config_object: XML configuration object.
        - MRF_grid_definition: Path to the MRF grid definition parameter in the configuration.

        Returns:
        - A string representing a valid grid definition.

        Raises:
        - ValueError: If the value is missing or not a valid grid definition.
        """

        try:
            value = config_object.find(MRF_grid_definition).text

            valid_input = [
                "EASE2_G3km",
                "EASE2_G1km",
                "EASE2_G9km",
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

            if value in valid_input and value.strip() != "":
                return value

            raise ValueError(
                f"Invalid Grid Definition, check configuration file. "
                f"Valid grid definitions are: {valid_input}"
            )

        except AttributeError:
            raise ValueError(
                "Invalid XML structure. Ensure the MRF grid definition is correctly specified."
            )

    @staticmethod
    def validate_MRF_projection_definition(config_object, MRF_projection_definition):
        """
        Validates the MRF projection definition parameter from the configuration file.

        Parameters:
        - config_object: XML configuration object.
          The root XML element containing the configuration.
        - MRF_projection_definition: Path to the MRF projection definition parameter in the configuration.

        Returns:
        - A string representing a valid projection definition ('G', 'N', or 'S').

        Raises:
        - ValueError: If the value is missing, blank, or not in the list of valid projection definitions.
          - Raises "Missing or blank MRF projection definition in the configuration file." if the value is None or empty.
          - Raises "Invalid Projection Definition" if the value is not one of the valid definitions ('G', 'N', 'S').
        """

        value = config_object.find(MRF_projection_definition).text

        valid_input = ["G", "N", "S"]

        if value is None or value.strip() == "":
            raise ValueError(
                "Missing or blank MRF projection definition in the configuration file. "
                f"Ensure a valid projection is specified, i.e.: {valid_input}."
            )

        value = value.strip()

        if value in valid_input:
            return value

        raise ValueError(
            f"Invalid Projection Definition, check configuration file."
            f" Valid projection definitions are: {valid_input}"
        )

    @staticmethod
    def validate_bg_smoothing(config_object, bg_smoothing):
        """
        Validates the bg_smoothing parameter from the configuration file.

        Parameters:
        - config_object: XML configuration object.
          The root XML element containing the configuration.
        - bg_smoothing: str value for bg_smoothing parameter (to be converted into float).

        Returns:
        - A float representing the bg_smoothing value. Defaults to 0 if the value is missing.

        Raises:
        - ValueError: If the value is not a valid float.
        """

        try:
            value = config_object.find(bg_smoothing).text

            if value is not None:
                value = float(value)
            else:
                value = 0

            return value

        except ValueError as e:
            raise ValueError(
                "Invalid `bg_smoothing` value. It must be a valid float."
            ) from e

    @staticmethod
    def validate_quality_control(config_object, quality_control, input_data_type):
        """
        Validates the `quality_control` parameter based on the input data type and configuration file.

        Parameters:
        - config_object: XML configuration object.
          The root XML element containing the configuration.
        - quality_control: value for `quality_control` parameter in the configuration file.
        - input_data_type: The type of input data (i.e., 'AMSR2', 'CIMR', or SMAP).

        Returns:
        - A boolean value (`True` or `False`) representing the `quality_control` setting.
          - Always returns `False` for 'AMSR2' and 'CIMR' input data types.
          - For other input data types, the method validates the value from the configuration file.

        Raises:
        - ValueError: If the value in the configuration file is not 'True' or 'False'.
        """

        if input_data_type == "AMSR2":
            return False

        elif input_data_type == "CIMR":
            return False

        else:
            valid_input = ["True", "False"]

            value = config_object.find(quality_control).text

            if value in valid_input:
                if value == "True":
                    return True

                else:
                    return False

            raise ValueError(
                f"Invalid `quality_control` value: {value}. Check Configuration File."
                f" Valid inputs for `quality_control` parameter are: {valid_input}"
            )
