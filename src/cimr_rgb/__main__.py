"""
This  module is the entry point for the RGB.
It is responsible for calling the other functions in the script
It is also responsible for handling the command line arguments
It is also responsible for handling the exceptions that are raised
It is also responsible for printing the output to the console
It is also responsible for returning the output to the caller
"""
import os
import pathlib as pb 
import pickle
import argparse 

from numpy import full, nan, array 
#import simplekml

# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# -----------------

from .config_file       import ConfigFile
from .data_ingestion    import DataIngestion
from .grid_generator    import GridGenerator, GRIDS
from .regridder         import ReGridder
from .rgb_logging       import RGBLogging 
from .product_generator import ProductGenerator 


def get_rgb_configuration(parser: argparse.ArgumentParser, 
                          file_to_write: pb.Path = pb.Path(pb.Path(__file__).parents[1]).joinpath("output/logs/rgb_config.xml") 
                          ) -> ConfigFile: 
    """
    Parses command line arguments, substitutes the default ones enabled inside
    XML configuration file, saves modified configuration file, opens modified
    configuration file as ConfigFile object and returns this object with
    modified validated parameters. 

    Parameters
    ----------
    parser:  argparse.ArgumentParser 
        The argument parser object used to handle command-line arguments.

    Command-Line Arguments
    ----------------------
    - '-t', '--input-data-type'        : (str) Type of input data 
        Description: "Value for inputData/type parameter." 
    - '-p', '--input-data-path'        : (str) Path to input data 
        Description: "Value for inputData/path parameter." 
    - '-T', '--target-band'            : (str) Target band of the inout data 
        Description: "Value for inputData/target parameter." 
    - '-S', '--source-band'            : (str) Source band of the input data 
        Description: "Value for inputData/source parameter." 
    - '-f', '--split-fore-aft'         : (str) Fore/Aft split for the input data 
        Description: "Value for inputData/splitForeAft parameter." 
    - '-s', '--save-to-disk'           : (str) Save output data to disk 
        Description: "Value for outputData/saveTodisk parameter." 
    - '-o', '--output-path'            : (str) Output pat for the data 
        Description: "Value for outputData/outputPath parameter." 
    - '-g', '--grid-type'              : (str) Type of grid parameters 
        Description: "Value for GridParams/gridType parameter." 
    - '-d', '--grid-definition'        : (str) Definition of grid parameters 
        Description: "Value for GridParams/gridDefinition parameter." 
    - '-P', '--projection-definition'  : (str) Projection definiton of grid parameters 
        Description: "Value for GridParams/projectionDefinition parameter." 
    - '-a', '--regridding-algorithm'   : (str) Algorithm for regridding parameters
        Description: "Value for ReGridderParams/regriddingAlgorithm parameter." 
    - '-c', '--logging-params-config'  : (str) Path to the RGB logging configuration
        Description: "Value for LoggingParams/configPath parameter." 
    - '-D', '--logging-params-decorate': (str) Whether to use RGB decorator
        Description: "Value for LoggingParams/decorate parameter." 

    Returns
    -------
    rgb_config: ConfigFile 
        Modified XML configuration if any of the commandline arguments were provided. 
    """

    # converting the relative path into absolute one if needed 
    if not pb.Path(file_to_write).is_absolute(): 
        config_file_path = pb.Path(config_file_path).resolve() 

    # TODO: Change this to root dir or something of the repo (the way it is done in CIMR GRASP )
    outputdir = pb.Path("../output").resolve()
    if not pb.Path(outputdir).exists(): 
        pb.Path(outputdir).mkdir() 

    logsdir  = pb.Path(outputdir).joinpath("logs")
    if not pb.Path(logsdir).exists(): 
        pb.Path(logsdir).mkdir() 

    # The command line parameters take the following form: 
    # config_params = {'name': ['p1', 'parameter1', 'type', 'description']} 
    valid_config_params = {
        'inputData/type'     : 
            ['t', 'input-data-type', str, "Value for inputData/type parameter."], 
        'inputData/path'     : 
            ['p', 'input-data-path', str, "Value for inputData/path parameter."], 
        'inputData/targetBand': 
            ['T', 'target-band',  str, "Value for inputData/target parameter."], 
        'inputData/sourceBand': 
            ['S', 'source-band',  str, "Value for inputData/source parameter."], 
        'inputData/splitForeAft': 
            ['f', 'split-fore-aft',  str, "Value for inputData/splitForeAft parameter."], 
        'outputData/saveTodisk': 
            ['s', 'save-to-disk',  str, "Value for outputData/saveTodisk parameter."], 
        'outputData/outputPath': 
            ['o', 'output-path',  str, "Value for outputData/outputPath parameter."], 
        'GridParams/gridType': 
            ['g', 'grid-type', str, "Value for GridParams/gridType parameter."], 
        'GridParams/gridDefinition': 
            ['d', 'grid-definition', str, "Value for GridParams/gridDefinition parameter."], 
        'GridParams/projectionDefinition': 
            ['P', 'projection-definition', str, "Value for GridParams/projectionDefinition parameter."], 
        'ReGridderParams/regriddingAlgorithm': 
            ['a', 'regridding-algorithm', str, "Value for ReGridderParams/regriddingAlgorithm parameter."], 
        # TODO: Create validation methods for loggers  
        'LoggingParams/configPath': 
            ['c', 'logging-params-config', str, "Value for LoggingParams/configPath parameter."], 
        'LoggingParams/decorate': 
            ['D', 'logging-params-decorate', str, "Value for LoggingParams/decorate parameter."], 
    }


    for params in valid_config_params.values(): 
        parser.add_argument(f'-{params[0]}', f'--{params[1]}', type = params[2], help=f"{params[3]}")

    args = parser.parse_args() 


    # Returning the configuration file path (from the command line). If it is
    # not given, then the dafult one will be used.  
    rgb_config_path = pb.Path(args.config_file).resolve()  

    modified_pars = {} 

    root, tree = ConfigFile.read_config(rgb_config_path)

    # Looping through valid parameters and assign its value if parameter is not
    # empty. Else, use default value provided via parameter file. 
    for key, value in valid_config_params.items():
        
        arg_name  = value[1].replace('-', '_')
        # Getting attributes from the commandline that will correspond to the
        # (modified) values of the parameter file 
        arg_value = getattr(args, arg_name)   

        element = root.find(key)

        if (arg_value and element) is not None: #or str(arg_value).strip() != '': 
            element.text       = arg_value 
            modified_pars[key] = arg_value  
        elif element is None: 
            raise ValueError(f"Key '{key}' not found in the XML configuration.")

        
    # Write the updated XML back to a new file
    tree.write(file_to_write, encoding="utf-8", xml_declaration=True)

    rgb_config = ConfigFile(file_to_write)

    logger     = rgb_config.logger 

    for key, value in modified_pars.items(): 
        logger.info(f"Parameter: `{key}` received commandline value: `{value}`") 

    logger.info("---------")

    logger.info(f"CIMR RGB Configuration")

    logger.info("---------")
    
    logger.info(f"Input File:              {rgb_config.input_data_path}")   
    logger.info(f"Input Data Type:         {rgb_config.input_data_type}") 
    logger.info(f"Split into Fore and Aft: {rgb_config.split_fore_aft}") 
    logger.info(f"Save to Disk:            {rgb_config.save_to_disk}") 
    logger.info(f"Regridding Algorithm:    {rgb_config.regridding_algorithm}") 
    logger.info(f"Source Band:             {rgb_config.source_band}")       
    logger.info(f"Target Band:             {rgb_config.target_band}") 
    logger.info(f"Grid Type:               {rgb_config.grid_type}") 
    logger.info(f"Grid Definition:         {rgb_config.grid_definition}") 
    logger.info(f"Projection Definition:   {rgb_config.projection_definition}")

    logger.info("---------")

    return rgb_config  


def main(): 
    # This is the main function that is called when the script is run
    # It is the entry point of the script

    # Setting the default value of the configuration parameter 
    rgb_config_path   = pb.Path('..', 'config-maks.xml').resolve() 

    # TODO: Implementing the parsing of commandline arguments 
    parser = argparse.ArgumentParser(description = "Update XML configuration parameters.")
    # Will use the default value of config_file if none is provided via command line: 
    # https://docs.python.org/3/library/argparse.html#nargs 
    parser.add_argument('config_file', type = str, help = "Path to the XML parameter file.", 
                    nargs="?", default = rgb_config_path)

    rgb_config        = get_rgb_configuration(parser = parser)

    # Ingest and Extract L1B Data
    timed_obj         = RGBLogging.rgb_decorated(
            decorate  = rgb_config.logpar_decorate, 
            decorator = RGBLogging.track_perf, 
            logger    = rgb_config.logger 
            )(DataIngestion) 

    ingestion_object  = timed_obj(rgb_config)

    timed_func        = RGBLogging.rgb_decorated(
            decorate  = rgb_config.logpar_decorate, 
            decorator = RGBLogging.track_perf, 
            logger    = rgb_config.logger
            )(ingestion_object.ingest_data)

    data_dict         = timed_func()  

    # Regrid Data
    regridder         = ReGridder(rgb_config)

    if rgb_config.input_data_type == 'SMAP':

        timed_func        = RGBLogging.rgb_decorated(
                decorate  = rgb_config.logpar_decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = rgb_config.logger
                )(regridder.regrid_l1c)

        data_dict_out     = timed_func(data_dict)
        
    if rgb_config.input_data_type == 'AMSR2':

        data_dict_out = regridder.regrid_l1c(data_dict)

    if rgb_config.input_data_type == 'CIMR':

        data_dict_out = regridder.regrid_l1c(data_dict)

    ProductGenerator(rgb_config).generate_l1c_product() 

    


if __name__ == '__main__':


    main() 

    # Intermediate results check
    # Put in the variables you want from the data_dict_out in data_dict.
    # grid_shape = GRIDS[config.grid_definition]['n_rows'], GRIDS[config.grid_definition]['n_cols']
    # # # create nan array with shape of grid_shape
    # grid = full(grid_shape, nan)
    # variable = data_dict_out['L']['bt_h_fore']
    # cell_row = data_dict_out['L']['cell_row_fore']
    # cell_col = data_dict_out['L']['cell_col_fore']
    # for i in range(len(cell_row)):
    #     grid[cell_row[i], cell_col[i]] = variable[i]
    # plt.imshow(grid)
    # Load equivalent L1c Data here if you want to compare

    # Intermediate results check
    # Put in the variables you want from the data_dict_out in data_dict.
    # grid_shape = GRIDS[config.grid_definition]['n_rows'], GRIDS[config.grid_definition]['n_cols']
    # # # create nan array with shape of grid_shape
    # grid = full(grid_shape, nan)
    # variable = data_dict_out['89a']['bt_h']
    # cell_row = data_dict_out['89a']['cell_row']
    # cell_col = data_dict_out['89a']['cell_col']
    # for i in range(len(cell_row)):
    #     grid[cell_row[i], cell_col[i]] = variable[i]
    # plt.imshow(grid)
    # Load equivalent L1c Data here if you want to compare


    # Temp save solution
    # save_path = ''
    # # Save pickle dictionary to save path
    # with open(save_path, 'wb') as f:
    #     pickle.dump(data_dict_out, f)

    # If the prodct is L1r we build the array as follows
    # if config.grid_type == 'L1R':
    #     # Build an array of nans
    #     variable = 'bt_h_fore'
    #     grid_shape = config.num_target_scans, config.num_target_samples
    #     grid = full(grid_shape, nan)
    #     scan_number = data_dict_out['C']['cell_row_fore']
    #     sample_number = data_dict_out['C']['cell_col_fore']
    #     for count, sample in enumerate(data_dict_out['C'][variable]):
    #         grid[int(scan_number[count]), int(sample_number[count])] = sample
    #     test = 0












