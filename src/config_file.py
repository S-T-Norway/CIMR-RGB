"""
This script reads the configuration file and validates the chosen parameters.
"""
from os import path, getcwd
import sys
from xml.etree.ElementTree import ParseError, parse


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
            split_fore_aft='inputData/splitForeAft'
        )

        if self.grid_type == "L1r":
            try:
                if self.target_band == self.source_band:
                    raise ValueError("Error: Source and Target bands cannot be the same")
            except ValueError as e:
                print(f"{e}")
                sys.exit(1)

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
            valid_input = ['L1c']
        else:
            valid_input = ['L1c', 'L1r']

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
        if grid_type == 'L1c':
            return None

        if input_data_type == "AMSR2":
            valid_input = ['6', '7', '10', '18', '23', '36', '89a', '89b']
            if config_object.find(target_band).text in valid_input:
                return config_object.find(target_band).text
            raise ValueError(
                f"Invalid Target Band, check configuration file. "
                f"Valid target bands are: {valid_input}"
            )

        if input_data_type == "SMAP":
            pass

        if input_data_type == "CIMR":
            pass

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
        if grid_type == 'L1r':
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
        valid_input = ['NN', 'DIB', 'IDS']
        if config_object.find(regridding_algorithm).text in valid_input:
            return config_object.find(regridding_algorithm).text
        raise ValueError(
            f"Invalid regridding algorithm. Check Configuration File."
            f" Valid regridding algorithms are: {valid_input}"
            )

    @staticmethod
    def validate_split_fore_aft(config_object, split_fore_aft):
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
        valid_input = ['True', 'False']
        if config_object.find(split_fore_aft).text in valid_input:
            return config_object.find(split_fore_aft).text
        raise ValueError(
            f"Invalid split fore aft. Check Configuration File."
            f" Valid split fore aft are: {valid_input}"
        )
