"""
This script reads the configuration file and validates the chosen parameters.
"""
from operator import truediv
from os import path, getcwd
import sys
from xml.etree.ElementTree import ParseError, parse
from grid_generator import GRIDS
from numpy import sqrt


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
    def __init__(self, config_file_path):
        """
        Initializes the ConfigFile class and stores validated
        config parameters as class attributes.

        Parameters
        ----------
        config_file_path: str
        """
        config_object = self.read_config(config_file_path)

        self.input_data_type = self.validate_input_data_type(
            config_object=config_object,
            input_data_type='inputData/type'
        )

        self.input_data_path = self.validate_input_data_path(
            config_object=config_object,
            input_data_path='inputData/path'
        )

        self.dpr_path = path.join(path.dirname(getcwd()), 'dpr')

        self.grid_type = self.validate_grid_type(
            config_object=config_object,
            grid_type='GridParams/gridType',
            input_data_type=self.input_data_type
        )

        self.target_band = self.validate_target_band(
            config_object=config_object,
            target_band='inputData/targetBand',
            input_data_type=self.input_data_type,
            grid_type=self.grid_type
        )


        if self.grid_type == "L1C":
            self.source_band=[]
        else:
            self.source_band = self.validate_source_band(
                config_object=config_object,
                source_band='inputData/sourceBand',
                input_data_type=self.input_data_type
            )

        self.grid_definition = self.validate_grid_definition(
            config_object=config_object,
            grid_type=self.grid_type,
            grid_definition='GridParams/gridDefinition'
        )

        self.projection_definition = self.validate_projection_definition(
            config_object=config_object,
            grid_definition=self.grid_definition,
            projection_definition='GridParams/projectionDefinition'
        )

        self.regridding_algorithm = self.validate_regridding_algorithm(
            config_object=config_object,
            regridding_algorithm='ReGridderParams/regriddingAlgorithm'
        )

        self.split_fore_aft = self.validate_split_fore_aft(
            config_object=config_object,
            split_fore_aft='inputData/splitForeAft',
            input_data_type=self.input_data_type
        )

        self.save_to_disk = self.validate_save_to_disk(
            config_object=config_object,
            save_to_disk='outputData/saveTodisk'
        )

        self.search_radius = self.validate_search_radius(
            config_object=config_object,
            search_radius='ReGridderParams/searchRadius',
            grid_definition=self.grid_definition
        )

        self.max_neighbours = self.validate_max_neighbours(
            config_object = config_object,
            max_neighbours = 'ReGridderParams/max_neighbours',
            regridding_algorithm = self.regridding_algorithm)

        if self.grid_type == "L1R":
            try:
                if self.target_band == self.source_band:
                    raise ValueError("Error: Source and Target bands cannot be the same")
            except ValueError as e:
                print(f"{e}")
                sys.exit(1)

        # SMAP specific Parameters
        if self.input_data_type == "SMAP":
            self.aft_angle_min = 90
            self.aft_angle_max = 270
            self.variable_key_map = {
                'bt_h': 'tb_h',
                'bt_v': 'tb_v',
                'bt_3': 'tb_3',
                'bt_4': 'tb_4',
                'processing_scan_angle': 'antenna_scan_angle',
                'longitude': 'tb_lon',
                'latitude': 'tb_lat',
                'x_position': 'x_pos',
                'y_position': 'y_pos',
                'z_position': 'z_pos',
                'x_velocity': 'x_vel',
                'y_velocity': 'y_vel',
                'z_velocity': 'z_vel',
                'sub_satellite_lon': 'sc_nadir_lon',
                'sub_satellite_lat': 'sc_nadir_lat'
            }

        # AMSR2 specific parameters
        if self.input_data_type == "AMSR2":
            self.LMT = 2200
            self.key_mappings = {
                # 'input': (co_reg_key, bt_key)
                '6': (0, 'Brightness Temperature (6.9GHz,'),
                '7': (1, 'Brightness Temperature (7.3GHz,'),
                '10': (2, 'Brightness Temperature (10.7GHz,'),
                '18': (3, 'Brightness Temperature (18.7GHz,'),
                '23': (4, 'Brightness Temperature (23.8GHz,'),
                '36': (5, 'Brightness Temperature (36.5GHz,'),
                '89a': (None, 'Brightness Temperature (89.0GHz-A,'),
                '89b': (None, 'Brightness Temperature (89.0GHz-B,')
            }
            self.kernel_size = config_object.find('ReGridderParams/kernelSize').text

        # CIMR Specific Parameters
        if self.input_data_type == "CIMR":
            self.variable_key_map = {
                'longitude': 'lon',
                'latitude': 'lat',
                'bt_h': 'brightness_temperature_h',
                'bt_v': 'brightness_temperature_v',
                'bt_3': 'brightness_temperature_t3',
                'bt_4': 'brightness_temperature_t4',
                'processing_scan_angle': 'scan_angle',
                'x_position': 'satellite_position',
                'y_position': 'satellite_position',
                'z_position': 'satellite_position',
                'x_velocity': 'satellite_velocity',
                'y_velocity': 'satellite_velocity',
                'z_velocity': 'satellite_velocity',
                'sub_satellite_lon': 'sub_satellite_lon',
                'sub_satellite_lat': 'sub_satellite_lat',
                'attitude': 'SatelliteBody2EarthCenteredInertialFrame'
            }
            self.aft_angle_min = 180
            self.aft_angle_max = 360
            self.num_horns = {
                'L': 1,
                'C': 4,
                'X': 4,
                'KA': 8,
                'KU': 8,
            }

        self.variables_to_regrid = self.validate_variables_to_regrid(
            config_object = config_object,
            input_data_type = self.input_data_type,
            variables_to_regrid = 'ReGridderParams/variables_to_regrid'
        )

        if self.regridding_algorithm == 'BG' or self.regridding_algorithm=='RSIR':

            self.antenna_method = self.validate_antenna_method(
                config_object = config_object,
                antenna_method = 'ReGridderParams/antenna_method'
            )

            self.antenna_threshold = self.validate_antenna_threshold(
                config_object=config_object,
                antenna_threshold = 'ReGridderParams/antenna_threshold'
            )

            self.polarisation_method = self.validate_polarisation_method(
                config_object=config_object,
                polarisation_method='ReGridderParams/polarisation_method'
            )


            if self.input_data_type == 'SMAP':
                # Antenna Pattern Path
                # Try and load a file, raise an error if its not there
                try:
                    relative_path = '../dpr/antenna_patterns/SMAP/RadiometerAntPattern_170830_v011.h5'
                    self.antenna_pattern_path = path.normpath(path.join(getcwd(), relative_path))
                except AttributeError as e:
                    raise ValueError(f"Error: SMAP Antenna Pattern not found in dpr") from e

                self.antenna_tilt_angle = 144.54

            elif self.input_data_type == 'CIMR':
                relative_path = '../dpr/antenna_patterns/CIMR'
                self.antenna_pattern_path = path.normpath(path.join(getcwd(), relative_path))

                self.antenna_tilt_angle = 46.886 # update to read from file
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
        return root

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
        valid_input = ['AMSR2', 'SMAP', 'CIMR']
        if config_object.find(input_data_type).text in valid_input:
            return config_object.find(input_data_type).text
        raise ValueError(f"Invalid input data type. Valid input data types are: {valid_input}")

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
        return config_object.find(input_data_path).text

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
            valid_input = ['L1C']
        else:
            valid_input = ['L1C', 'L1R']

        if config_object.find(grid_type).text in valid_input:
            return config_object.find(grid_type).text
        raise ValueError(f"Invalid Grid Type. Check Configuration File. Valid grid types are:"
                         f" {valid_input} for {input_data_type} data.")

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

        if input_data_type == "AMSR2":
            valid_input = ['6', '7', '10', '18', '23', '36', '89a', '89b', 'All']
            config_input = config_object.find(target_band).text.split()
            if config_input == ['All']:
                return ['6', '7', '10', '18', '23', '36', '89a', '89b']
            else:
                for i in config_input:
                    if i not in valid_input:
                        raise ValueError(
                            f"Invalid target bands for AMSR2 L1C remap."
                            f" Valid target bands are: {valid_input} or any combination of individual bands.")
                return config_object.find(target_band).text.split()

        if input_data_type == "CIMR":
            if grid_type == "L1C":
                valid_input = ['L', 'C', 'X', 'KA', 'KU', 'All']
                config_input = config_object.find(target_band).text.split()
                if config_input == ['All']:
                    return ['L', 'C', 'X', 'KA', 'KU']
                else:
                    for i in config_input:
                        if i not in valid_input:
                            raise ValueError(
                                f"Invalid target bands for CIMR L1C remap."
                                f" Valid target bands are: {valid_input} or any combination of individual bands.")
                    return config_object.find(target_band).text.split()

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
            valid_input = ['6', '7', '10', '18', '23', '36', '89a', '89b']
            if config_object.find(source_band).text in valid_input:
                return config_object.find(source_band).text
            raise ValueError(
                f"Invalid Target Band, check configuration file. "
                f"Valid target bands are: {valid_input}"
            )

        if input_data_type == "SMAP":
            pass

        if input_data_type == "CIMR":
            pass

    @staticmethod
    def validate_grid_definition(config_object, grid_type, grid_definition):
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
        if grid_type == 'L1R':
            return None

        valid_input = ['EASE2_G9km', 'EASE2_N9km', 'EASE2_S9km',
                       'EASE2_G36km', 'EASE2_N36km', 'EASE2_S36km']
        if config_object.find(grid_definition).text in valid_input:
            return config_object.find(grid_definition).text
        raise ValueError(
            f"Invalid Grid Definition, check configuration file. "
            f"Valid grid definitions are: {valid_input}"
        )

    @staticmethod
    def validate_projection_definition(config_object, grid_definition, projection_definition):
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
            if 'EASE2' in grid_definition:
                valid_input = ['G', 'N', 'S']
                if config_object.find(projection_definition).text in valid_input:
                    return config_object.find(projection_definition).text
                raise ValueError(
                    f"Invalid Projection Definition, check configuration file."
                    f" Valid projection definitions are: {valid_input}"
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
        valid_input = ['NN', 'DIB', 'IDS', 'BG', 'RSIR']
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
        if input_data_type == 'AMSR2':
            return False
        valid_input = ['True', 'False']
        if config_object.find(split_fore_aft).text in valid_input:
            if config_object.find(split_fore_aft).text == 'True':
                return True
            else:
                return False
        raise ValueError(
            f"Invalid split fore aft. Check Configuration File."
            f" Valid split fore aft are: {valid_input}"
        )

    @staticmethod
    def validate_save_to_disk(config_object, save_to_disk):
        value = bool(config_object.find(save_to_disk).text)
        if value is not True and value is not False:
            raise ValueError(
                f"Invalid saveToDisk. Check Configuration File."
                f" SaveToDisk must be either True or False")
        return value

    @staticmethod
    def validate_search_radius(config_object, search_radius, grid_definition):
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
        if value is not None:
            value = float(value)*1000
        else:
            value = sqrt(2 * (GRIDS[grid_definition]['res'] / 2) ** 2)
        return value

    @staticmethod
    def get_scan_geometry(config, band_to_remap=None):
        if config.input_data_type == 'SMAP':
            num_scans = 779
            num_earth_samples = 241
        elif config.input_data_type == 'CIMR':
            if band_to_remap == 'L':
                num_scans = 74
                num_earth_samples = 691
            elif band_to_remap == 'C':
                num_scans = 74
                num_earth_samples = 2747*4
            elif band_to_remap == 'X':
                num_scans = 74
                num_earth_samples = 2807*4
            elif band_to_remap == 'KA':
                num_scans = 74
                num_earth_samples = 10395*8
            elif band_to_remap == 'KU':
                num_scans = 74
                num_earth_samples = 7692*8
        return num_scans, num_earth_samples

    @staticmethod
    def validate_variables_to_regrid(config_object, input_data_type, variables_to_regrid):
        value = config_object.find(variables_to_regrid).text
        if input_data_type == 'SMAP':
            valid_input = ['bt_h', 'bt_v', 'bt_3', 'bt_4',
                         'processing_scan_angle', 'longitude', 'latitude']
            default_vars = ['bt_h', 'bt_v', 'bt_3', 'bt_4',
                            'processing_scan_angle', 'longitude', 'latitude']

        elif input_data_type == 'AMSR2':
            valid_input = ['bt_h, bt_v, longitude, latitude']

        elif input_data_type == 'CIMR':
            valid_input = ['bt_h', 'bt_v', 'bt_3', 'bt_4',
                           'processing_scan_angle', 'longitude', 'latitude']
            default_vars = ['bt_h', 'bt_v', 'bt_3', 'bt_4',
                            'processing_scan_angle', 'longitude', 'latitude']

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
        if regridding_algorithm == 'NN':
            return 1
        else:
            value = config_object.find(max_neighbours).text
            if value is not None:
                value = int(value)
            else:
                value = 1000
            return value

    @staticmethod
    def validate_antenna_method(config_object, antenna_method):
        valid_input = ['gaussian', 'real']
        value = config_object.find(antenna_method)
        if value is None:
            return 'real'
        elif value.text in valid_input:
            return value.text
        else:
            raise ValueError(
                f"Invalid antenna method. Check Configuration File."
                f" Valid antenna methods are: {valid_input}"
            )

    @staticmethod
    def validate_antenna_threshold(config_object, antenna_threshold):
        if config_object.find(antenna_threshold).text is None:
            # We should have a default set of values for each Antenna Pattern
            # For now, I will just choose 9dB
            return 9.

        try:
            return float(config_object.find(antenna_threshold).text)
        except:
            raise ValueError(
                f"Invalid antenna threshold. Check Configuration File."
                f" Antenna threshold must be a float or integer"
            )

    @staticmethod
    def validate_polarisation_method(config_object, polarisation_method):
        valid_input = ['scalar', 'mueller']
        if config_object.find(polarisation_method).text is None:
            return 'scalar'
        if config_object.find(polarisation_method).text not in valid_input:
            raise ValueError(
                f"Invalid polarisation method. Check Configuration File."
                f" Valid polarisation methods are: {valid_input}"
            )
        else:
            return config_object.find(polarisation_method).text







