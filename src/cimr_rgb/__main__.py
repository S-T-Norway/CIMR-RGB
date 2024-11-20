"""
This  module is the entry point for the RGB.
It is responsible for calling the other functions in the script
It is also responsible for handling the command line arguments
It is also responsible for handling the exceptions that are raised
It is also responsible for printing the output to the console
It is also responsible for returning the output to the caller
"""
import os
import sys
#sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "tests"))
import pathlib as pb 
import pickle
import argparse 

from numpy import full, nan, array

# ---- Testing ----
#import matplotlib
#tkagg = matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
# -----------------

from cimr_rgb.config_file       import ConfigFile
from cimr_rgb.data_ingestion    import DataIngestion
from cimr_rgb.grid_generator    import GridGenerator, GRIDS
from cimr_rgb.regridder         import ReGridder
from cimr_rgb.rgb_logging       import RGBLogging
from cimr_rgb.product_generator import ProductGenerator

# Maksym: I assume this comes frome tests directory 
#from inspect_SMAP_l1c import compare_smap_l1c


# TODO: Change description for configuration parameters 
def get_rgb_configuration(parser: argparse.ArgumentParser, 
                          #config_file: pb.Path #= pb.Path(pb.Path(__file__).parents[1]).joinpath("output/logs/rgb_config.xml") 
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
    #if not pb.Path(config_file).is_absolute(): 
    #    config_file = pb.Path(config_file).resolve() 

    #print(config_file.name)
    #print(config_file)
    #exit() 


    # The command line parameters take the following form: 
    # config_params = {'name': ['p1', 'parameter1', 'type', 'description']} 
    valid_config_params = {
        # Input data parameters 
        'InputData/type'     : 
            ['t', 'input-data-type', str, "Value for InputData/type parameter."], 
        'InputData/path'     : 
            ['p', 'input-data-path', str, "Value for InputData/path parameter."], 
        'InputData/antenna_patterns_path': 
            ['app', 'antenna-patterns-path',  str, "Value for InputData/antenna_patterns_path parameter."], 
        'InputData/split_fore_aft': 
            ['sfa', 'split-fore-aft',  str, "Value for InputData/split_fore_aft parameter."], 
        'InputData/source_band': 
            ['sb', 'source-band',  str, "Value for InputData/source_band parameter."], 
        'InputData/target_band': 
            ['tb', 'target-band',  str, "Value for InputData/target_band parameter."], 
        'InputData/quality_control': 
            ['qc', 'quality-control',  str, "Value for InputData/quality_control parameter."], 
        # TODO: Add LMT parameter 
        # Grid parameters 
        'GridParams/grid_type': 
            ['gt', 'grid-type', str, "Value for GridParams/grid_type parameter."], 
        'GridParams/grid_definition': 
            ['gd', 'grid-definition', str, "Value for GridParams/grid_definition parameter."], 
        'GridParams/projection_definition': 
            ['pd', 'projection-definition', str, "Value for GridParams/projection_definition parameter."], 
        'GridParams/reduced_grid_inds': 
            ['rg', 'reduced-grid-inds', str, "Value for GridParams/reduced_grid_inds parameter."], 
        # Regridder parameters 
        'ReGridderParams/regridding_algorithm': 
            ['ra', 'regridding-algorithm', str, "Value for ReGridderParams/regridding_algorithm parameter."], 
        'ReGridderParams/search_radius': 
            ['sr', 'search-radius', str, "Value for ReGridderParams/search_radius parameter."], 
        'ReGridderParams/max_neighbours': 
            ['mn', 'max-neighbours', str, "Value for ReGridderParams/max_neighbours parameter."], 
        'ReGridderParams/variables_to_regrid': 
            ['vtr', 'variables-to-regrid', str, "Value for ReGridderParams/variables_to_regrid parameter."], 
        'ReGridderParams/source_antenna_method': 
            ['sam', 'source-antenna-method', str, "Value for ReGridderParams/source_antenna_method parameter."], 
        'ReGridderParams/target_antenna_method': 
            ['tam', 'target-antenna-method', str, "Value for ReGridderParams/target_antenna_method parameter."], 
        'ReGridderParams/polarisation_method': 
            ['pm', 'polarisation-method', str, "Value for ReGridderParams/polarisation_method parameter."], 
        'ReGridderParams/source_antenna_threshold': 
            ['sat', 'source-antenna-threshold', str, "Value for ReGridderParams/source_antenna_threshold parameter."], 
        'ReGridderParams/target_antenna_threshold': 
            ['tat', 'target-antenna-threshold', str, "Value for ReGridderParams/target_antenna_threshold parameter."], 
        'ReGridderParams/MRF_grid_definition': 
            ['mat', 'mrf-grid-definition', str, "Value for ReGridderParams/MRF_grid_definition parameter."], 
        'ReGridderParams/MRF_projection_definition': 
            ['mpt', 'mrf-projection-definition', str, "Value for ReGridderParams/MRF_projection_definition parameter."], 
        'ReGridderParams/source_gaussian_params': 
            ['sgp', 'source-gaussian-params', str, "Value for ReGridderParams/source_gaussian_params parameter."], 
        'ReGridderParams/target_gaussian_params': 
            ['tgp', 'target-gaussian-params', str, "Value for ReGridderParams/target_gaussian_params parameter."], 
        'ReGridderParams/boresight_shift': 
            ['bs', 'boresight-shift', str, "Value for ReGridderParams/boresight_shift parameter."], 
        'ReGridderParams/rsir_iteration': 
            ['rsir', 'rsir-iteration', str, "Value for ReGridderParams/rsir_iteration parameter."], 
        'ReGridderParams/bg_smoothing': 
            ['bgs', 'bg-smoothing', str, "Value for ReGridderParams/bg_smoothing parameter."], 
        # Output data parameters 
        'OutputData/save_to_disk': 
            ['std', 'save-to-disk',  str, "Value for OutputData/save_to_disk parameter."], 
        'OutputData/output_path': 
            ['op', 'output-path',  str, "Value for OutputData/output_path parameter."], 
        # Logging params 
        'LoggingParams/config_path': 
            ['cp', 'logging-params-config', str, "Value for LoggingParams/config_path parameter."], 
        'LoggingParams/decorate': 
            ['d', 'logging-params-decorate', str, "Value for LoggingParams/decorate parameter."], 
    }


    for params in valid_config_params.values(): 
        parser.add_argument(f'-{params[0]}', f'--{params[1]}', type = params[2], help=f"{params[3]}")

    args = parser.parse_args() 


    # Returning the configuration file path (from the command line). If it is
    # not given, then the default one will be used.
    rgb_config_path = pb.Path(args.config_file).resolve()  

    modified_pars = {} 

    root, tree = ConfigFile.read_config(rgb_config_path)

    # TODO: LoggingParams may not work properly here. Check it 
    # 
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

    # Creating output directory based on the parameter provided via cmd or xml
    # file. Once this directory created, we also create `logs` folder to store
    # logs of the run. 
    outputdir = pb.Path(root.find("OutputData/output_path").text).resolve()
    if not pb.Path(outputdir).exists(): 
        pb.Path(outputdir).mkdir() 


    # Appending the name of configuration file to the output directory path  
    file_to_write = outputdir.joinpath(rgb_config_path.name) 
        
    # Write the updated XML back to a new file
    tree.write(file_to_write, encoding="utf-8", xml_declaration=True)

    # Open modified file and check all the parameters 
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

    # TODO: Default value should be taken from the installed package probably? 
    #       Do we even need a default value? 
    #       Need to add the confug as part of installable files in the MANIFEST
    #       file, otherwise it won't work 
    # 
    # Setting the default value of the configuration parameter 
    rgb_config_path   = pb.Path("configs", 'rgb_config.xml').resolve()
    # If running from IDE, comment out above and use the line below
    # rgb_config_path = pb.Path('/home/beywood/ST/CIMR_RGB/CIMR-RGB/configs', 'rgb_config.xml').resolve()

    parser = argparse.ArgumentParser(description = "Update XML configuration parameters.")
    # Will use the default value of config_file if none is provided via command line: 
    # https://docs.python.org/3/library/argparse.html#nargs 
    parser.add_argument('config_file', type = str, help = "Path to the XML parameter file.", 
                    nargs="?", default = rgb_config_path)

    rgb_config        = get_rgb_configuration(parser = parser)#, config_file = rgb_config_path)

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
        timed_func        = RGBLogging.rgb_decorated(
                decorate  = rgb_config.logpar_decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = rgb_config.logger
                )(regridder.regrid_l1c)
        data_dict_out     = timed_func(data_dict)

    # Generate L1C product according to CDL 
    ProductGenerator(rgb_config).generate_l1c_product(data_dict = data_dict_out) 

    


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
    

    # Maksym: This part was in main.py in devel, leave it here for now 
    
    ## If the prodct is L1r we build the array as follows
    #if config.grid_type == 'L1R':
    #    # Build an array of nans
    #    variable = 'bt_h'
    #    grid_shape = config.num_target_scans, config.num_target_samples
    #    grid = full(grid_shape, nan)
    #    scan_number = data_dict_out['L']['cell_row']
    #    sample_number = data_dict_out['L']['cell_col']
    #    for count, sample in enumerate(data_dict_out['L'][variable]):
    #        grid[int(scan_number[count]), int(sample_number[count])] = sample
    #    plt.imshow(grid)

    ##     test = 0












