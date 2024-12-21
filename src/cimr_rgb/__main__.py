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

# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "tests"))
import pathlib as pb
import pickle
import argparse
import datetime
import time


from numpy import full, nan, array
import psutil
import numpy as np
import scipy as sp
import pyresample
import h5py
import netCDF4 as nc
import tqdm
import pyproj
import shapely
import matplotlib
import cartopy

# ---- Testing ----
# import matplotlib
# tkagg = matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.ion()
# sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests')
# from inspect_SMAP_l1c import compare_smap_l1c
# -----------------

from cimr_rgb.config_file import ConfigFile
from cimr_rgb.data_ingestion import DataIngestion
from cimr_rgb.grid_generator import GridGenerator, GRIDS
from cimr_rgb.regridder import ReGridder
from cimr_rgb.rgb_logging import RGBLogging
from cimr_rgb.product_generator import ProductGenerator
import cimr_grasp.grasp_io as grasp_io

# Maksym: I assume this comes frome tests directory
# from inspect_SMAP_l1c import compare_smap_l1c


# TODO: Change description for configuration parameters
def get_rgb_configuration(
    parser: argparse.ArgumentParser,
    # config_file: pb.Path #= pb.Path(pb.Path(__file__).parents[1]).joinpath("output/logs/rgb_config.xml")
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

    # The command line parameters take the following form:
    # config_params = {'name': ['p1', 'parameter1', 'type', 'description']}
    valid_config_params = {
        # Input data parameters
        "InputData/type": [
            "t",
            "input-data-type",
            str,
            "Value for InputData/type parameter.",
        ],
        "InputData/path": [
            "p",
            "input-data-path",
            str,
            "Value for InputData/path parameter.",
        ],
        "InputData/antenna_patterns_path": [
            "app",
            "antenna-patterns-path",
            str,
            "Value for InputData/antenna_patterns_path parameter.",
        ],
        "InputData/split_fore_aft": [
            "sfa",
            "split-fore-aft",
            str,
            "Value for InputData/split_fore_aft parameter.",
        ],
        "InputData/source_band": [
            "sb",
            "source-band",
            str,
            "Value for InputData/source_band parameter.",
        ],
        "InputData/target_band": [
            "tb",
            "target-band",
            str,
            "Value for InputData/target_band parameter.",
        ],
        "InputData/quality_control": [
            "qc",
            "quality-control",
            str,
            "Value for InputData/quality_control parameter.",
        ],
        # TODO: Add LMT parameter
        # Grid parameters
        "GridParams/grid_type": [
            "gt",
            "grid-type",
            str,
            "Value for GridParams/grid_type parameter.",
        ],
        "GridParams/grid_definition": [
            "gd",
            "grid-definition",
            str,
            "Value for GridParams/grid_definition parameter.",
        ],
        "GridParams/projection_definition": [
            "pd",
            "projection-definition",
            str,
            "Value for GridParams/projection_definition parameter.",
        ],
        "GridParams/reduced_grid_inds": [
            "rg",
            "reduced-grid-inds",
            str,
            "Value for GridParams/reduced_grid_inds parameter.",
        ],
        # Regridder parameters
        "ReGridderParams/regridding_algorithm": [
            "ra",
            "regridding-algorithm",
            str,
            "Value for ReGridderParams/regridding_algorithm parameter.",
        ],
        "ReGridderParams/search_radius": [
            "sr",
            "search-radius",
            str,
            "Value for ReGridderParams/search_radius parameter.",
        ],
        "ReGridderParams/max_neighbours": [
            "mn",
            "max-neighbours",
            str,
            "Value for ReGridderParams/max_neighbours parameter.",
        ],
        "ReGridderParams/variables_to_regrid": [
            "vtr",
            "variables-to-regrid",
            str,
            "Value for ReGridderParams/variables_to_regrid parameter.",
        ],
        "ReGridderParams/source_antenna_method": [
            "sam",
            "source-antenna-method",
            str,
            "Value for ReGridderParams/source_antenna_method parameter.",
        ],
        "ReGridderParams/target_antenna_method": [
            "tam",
            "target-antenna-method",
            str,
            "Value for ReGridderParams/target_antenna_method parameter.",
        ],
        "ReGridderParams/polarisation_method": [
            "pm",
            "polarisation-method",
            str,
            "Value for ReGridderParams/polarisation_method parameter.",
        ],
        "ReGridderParams/source_antenna_threshold": [
            "sat",
            "source-antenna-threshold",
            str,
            "Value for ReGridderParams/source_antenna_threshold parameter.",
        ],
        "ReGridderParams/target_antenna_threshold": [
            "tat",
            "target-antenna-threshold",
            str,
            "Value for ReGridderParams/target_antenna_threshold parameter.",
        ],
        "ReGridderParams/max_theta_antenna_patterns": [
            "mtap",
            "max-theta-antenna-patterns",
            str,
            "Value for ReGridderParams/max_theta_antenna_patterns.",
        ],
        "ReGridderParams/MRF_grid_definition": [
            "mat",
            "mrf-grid-definition",
            str,
            "Value for ReGridderParams/MRF_grid_definition parameter.",
        ],
        "ReGridderParams/MRF_projection_definition": [
            "mpt",
            "mrf-projection-definition",
            str,
            "Value for ReGridderParams/MRF_projection_definition parameter.",
        ],
        "ReGridderParams/source_gaussian_params": [
            "sgp",
            "source-gaussian-params",
            str,
            "Value for ReGridderParams/source_gaussian_params parameter.",
        ],
        "ReGridderParams/target_gaussian_params": [
            "tgp",
            "target-gaussian-params",
            str,
            "Value for ReGridderParams/target_gaussian_params parameter.",
        ],
        "ReGridderParams/boresight_shift": [
            "bs",
            "boresight-shift",
            str,
            "Value for ReGridderParams/boresight_shift parameter.",
        ],
        "ReGridderParams/rsir_iteration": [
            "rsir",
            "rsir-iteration",
            str,
            "Value for ReGridderParams/rsir_iteration parameter.",
        ],
        "ReGridderParams/bg_smoothing": [
            "bgs",
            "bg-smoothing",
            str,
            "Value for ReGridderParams/bg_smoothing parameter.",
        ],
        # Output data parameters
        "OutputData/save_to_disk": [
            "std",
            "save-to-disk",
            str,
            "Value for OutputData/save_to_disk parameter.",
        ],
        "OutputData/output_path": [
            "op",
            "output-path",
            str,
            "Value for OutputData/output_path parameter.",
        ],
        "OutputData/version": [
            "v",
            "version",
            str,
            "Value for OutputData/version parameter.",
        ],
        "OutputData/creator_name": [
            "cn",
            "creator-name",
            str,
            "Value for OutputData/creator_name parameter.",
        ],
        "OutputData/creator_email": [
            "ce",
            "creator-email",
            str,
            "Value for OutputData/creator_email parameter.",
        ],
        "OutputData/creator_url": [
            "cu",
            "creator-url",
            str,
            "Value for OutputData/creator_url parameter.",
        ],
        "OutputData/creator_institution": [
            "ci",
            "creator-institution",
            str,
            "Value for OutputData/creator_institution parameter.",
        ],
        "OutputData/timestamp_fmt": [
            "tf",
            "timestamp_fmt",
            str,
            "Value for OutputData/timestamp_fmt parameter.",
        ],
        # Logging params
        "LoggingParams/config_path": [
            "cp",
            "logging-params-config",
            str,
            "Value for LoggingParams/config_path parameter.",
        ],
        "LoggingParams/decorate": [
            "d",
            "logging-params-decorate",
            str,
            "Value for LoggingParams/decorate parameter.",
        ],
        "LoggingParams/logger_name": [
            "ln",
            "logger_name",
            str,
            "Value for LoggingParams/logger_name parameter.",
        ],
    }

    for params in valid_config_params.values():
        parser.add_argument(
            f"-{params[0]}", f"--{params[1]}", type=params[2], help=f"{params[3]}"
        )

    args = parser.parse_args()

    # Returning the configuration file path (from the command line). If it is
    # not given, then the default one will be used.
    rgb_config_path = pb.Path(args.config_file).resolve()

    modified_pars = {}

    root, tree = ConfigFile.read_config(rgb_config_path)

    # Looping through valid parameters and assign its value if parameter is not
    # empty. Else, use default value provided via parameter file.
    for key, value in valid_config_params.items():
        arg_name = value[1].replace("-", "_")
        # Getting attributes from the commandline that will correspond to the
        # (modified) values of the parameter file
        arg_value = getattr(args, arg_name)

        element = root.find(key)

        if (arg_value and element) is not None:  # or str(arg_value).strip() != '':
            element.text = arg_value
            modified_pars[key] = arg_value
        elif element is None:
            raise ValueError(f"Key '{key}' not found in the XML configuration.")

    # Creating output directory based on the parameter provided via cmd or xml
    # file. Once this directory created, we also create `logs` folder to store
    # logs of the run.
    # outputdir = pb.Path(root.find("OutputData/output_path").text).resolve()
    # outputdir = pb.Path(root.find("OutputData/output_path").text)
    # outputdir       = grasp_io.resolve_config_path(
    #     path_string = outputdir
    # )
    # #outputdir  = grasp_io.resolve_config_path(outputdir)
    # #if not pb.Path(outputdir).exists():
    # #    pb.Path(outputdir).mkdir()
    # grasp_io.rec_create_dir(outputdir)

    outputdir = ConfigFile.validate_output_directory_path(
        config_object=root, output_path="OutputData/output_path", logger=None
    )

    # Creating output directory for logs (to save log files there)
    logdir = pb.Path(outputdir).joinpath("logs")
    # Create the directory structure if necessary
    grasp_io.rec_create_dir(path=logdir)

    # print(rgb_config_path.stem)
    # print(root.find('OutputData/timestamp').text)

    # TODO: Put this into config_file.py for validation
    # timestamp_elem = root.find("OutputData/timestamp").text
    timestamp_elem = root.find("OutputData/suffix").text
    timestamp_fmt_elem = root.find("OutputData/timestamp_fmt").text
    if timestamp_elem is None or timestamp_elem.strip() == "":
        # Getting the current time stamp to propagate into the software
        timestamp_elem = datetime.datetime.now()

        # Format the date and time as "YYYY-MM-DD_HH-MM-SS"
        timestamp_elem = timestamp_elem.strftime(timestamp_fmt_elem)
    root.find("OutputData/suffix").text = timestamp_elem

    # Appending the name of configuration file to the output directory path
    xml_config_file_name = rgb_config_path.stem + f"_{timestamp_elem}" + ".xml"
    file_to_write = logdir.joinpath(xml_config_file_name)

    # Write the updated XML back to a new file
    tree.write(file_to_write, encoding="utf-8", xml_declaration=True)

    # Open modified file and check all the parameters
    rgb_config = ConfigFile(file_to_write)

    logger = rgb_config.logger

    logger.info("=====================")

    logger.info(f"CIMR RGB Configuration")

    logger.info("=====================")

    if modified_pars:
        for key, value in modified_pars.items():
            logger.info(f"Parameter: `{key}` received commandline value: `{value}`")
    else:
        logger.info("No command line arguments were provided.")

    logger.info("---------------------")

    logger.info(f"Input File:                 {rgb_config.input_data_path}")
    logger.info(f"Input Data Type:            {rgb_config.input_data_type}")
    logger.info(f"Antenna Patterns Path:      {rgb_config.antenna_patterns_path}")
    logger.info(f"Quality Control:            {rgb_config.quality_control}")
    logger.info(f"Split into Fore and Aft:    {rgb_config.split_fore_aft}")
    logger.info(f"Save to Disk:               {rgb_config.save_to_disk}")
    logger.info(f"Output Path:                {rgb_config.output_path}")
    logger.info(f"Product Version:            {rgb_config.product_version}")
    logger.info(f"Creator Name:               {rgb_config.creator_name}")
    logger.info(f"Creator Email:              {rgb_config.creator_email}")
    logger.info(f"Creator URL:                {rgb_config.creator_url}")
    logger.info(f"Creator Institution:        {rgb_config.creator_institution}")
    logger.info(f"Timestamp Format:           {rgb_config.timestamp_fmt}")
    logger.info(f"Regridding Algorithm:       {rgb_config.regridding_algorithm}")
    logger.info(f"Source Band:                {rgb_config.source_band}")
    logger.info(f"Target Band:                {rgb_config.target_band}")
    logger.info(f"Grid Type:                  {rgb_config.grid_type}")
    logger.info(f"Grid Definition:            {rgb_config.grid_definition}")
    logger.info(f"Projection Definition:      {rgb_config.projection_definition}")
    logger.info(f"Reduced Grid Indices:       {rgb_config.reduced_grid_inds}")
    logger.info(f"Regridding Algorithm:       {rgb_config.regridding_algorithm}")
    logger.info(f"Search Radius:              {rgb_config.search_radius}")
    logger.info(f"Max Neighbour:              {rgb_config.max_neighbours}")
    logger.info(f"Variables to Regrid:        {rgb_config.variables_to_regrid}")
    logger.info(f"Boresight Shift:            {rgb_config.boresight_shift}")
    if rgb_config.regridding_algorithm in ["BG", "RSIR", "LW", "CG"]:
        logger.info(f"Source Antenna Method:      {rgb_config.source_antenna_method}")
        logger.info(f"Target Antenna Method:      {rgb_config.target_antenna_method}")
        logger.info(f"Polarisation Method:        {rgb_config.polarisation_method}")
        logger.info(
            f"Source Antenna Threshold:   {rgb_config.source_antenna_threshold}"
        )
        logger.info(
            f"Target Antenna Threshold:   {rgb_config.target_antenna_threshold}"
        )
        logger.info(
            f"Max Theta Antenna Patterns: {rgb_config.max_theta_antenna_patterns}"
        )
        logger.info(f"MRF Grid Definition:        {rgb_config.MRF_grid_definition}")
        logger.info(
            f"MRF Grid Definition:        {rgb_config.MRF_projection_definition}"
        )
        logger.info(f"Source Gaussian Params:     {rgb_config.source_gaussian_params}")
        logger.info(f"Target Gaussian Params:     {rgb_config.target_gaussian_params}")
    else:
        logger.info(
            f"Source Antenna Method:      N/A"
        )  # {rgb_config.source_antenna_method}")
        logger.info(
            f"Target Antenna Method:      N/A"
        )  # {rgb_config.target_antenna_method}")
        logger.info(
            f"Polarisation Method:        N/A"
        )  # {rgb_config.polarisation_method}")
        logger.info(
            f"Source Antenna Threshold:   N/A"
        )  # {rgb_config.source_antenna_threshold}")
        logger.info(
            f"Target Antenna Threshold:   N/A"
        )  # {rgb_config.target_antenna_threshold}")
        logger.info(
            f"Max Theta Antenna Patterns: N/A"
        )  # {rgb_config.max_theta_antenna_patterns}")
        logger.info(
            f"MRF Grid Definition:        N/A"
        )  # {rgb_config.MRF_grid_definition}")
        logger.info(
            f"MRF Grid Definition:        N/A"
        )  # {rgb_config.MRF_projection_definition}")
        logger.info(
            f"Source Gaussian Params:     N/A"
        )  # {rgb_config.source_gaussian_params}")
        logger.info(
            f"Target Gaussian Params:     N/A"
        )  # {rgb_config.target_gaussian_params}")
    if rgb_config.regridding_algorithm in ["RSIR"]:
        logger.info(f"rSIR Iteration:             {rgb_config.rsir_iteration}")
    else:
        logger.info(f"rSIR Iteration:             N/A")  # {rgb_config.rsir_iteration}")
    if rgb_config.regridding_algorithm in ["BG"]:
        logger.info(f"BG Smoothing:               {rgb_config.bg_smoothing}")
    else:
        logger.info(f"BG Smoothing:               N/A")  # {rgb_config.bg_smoothing}")
    logger.info(f"Logger Name:                {rgb_config.logger_name}")
    # Can give an entire dict as output
    # logger.info(f"{rgb_config.logpar_config}")
    logger.info(f"Use RGB Decorator:          {rgb_config.logpar_decorate}")

    logger.info("---------------------")

    logger.info("The following libraries were used:")
    logger.info(f"netCDF4:                    v{nc.__version__}")
    logger.info(f"numpy:                      v{np.__version__}")
    logger.info(f"pyresample:                 v{pyresample.__version__}")
    logger.info(f"scipy:                      v{sp.__version__}")
    logger.info(f"h5py:                       v{h5py.__version__}")
    logger.info(f"pyproj:                     v{pyproj.__version__}")
    logger.info(f"tqdm:                       v{tqdm.__version__}")
    logger.info(f"psutil:                     v{psutil.__version__}")
    logger.info(f"cartopy:                    v{cartopy.__version__}")
    logger.info(f"shapely:                    v{shapely.__version__}")
    logger.info(f"matplotlib:                 v{matplotlib.__version__}")

    logger.info("---------------------")

    return rgb_config


def main():
    # This is the main function that is called when the script is run
    # It is the entry point of the script

    # TODO: Default value should be taken from the installed package probably?
    #       Do we even need a default value?
    #       Need to add the confug as part of installable files in the MANIFEST
    #       file, otherwise it won't work
    #

    # Get the current process
    process = psutil.Process()

    # Record the start time and resource usage
    start_time = time.perf_counter()

    cpu_time_start = process.cpu_times()
    cpu_usage_start = process.cpu_percent(interval=None)

    # Memory size in bytes / 1024**2 = memory size in MB
    memory_usage_start = process.memory_info().rss / 1024**2

    # Setting the default value of the configuration parameter
    rgb_config_path = pb.Path("configs", "rgb_config.xml").resolve()
    # If running from IDE, comment out above and use the line below
    # rgb_config_path = pb.Path('/home/beywood/ST/CIMR_RGB/CIMR-RGB/configs', 'rgb_config.xml').resolve()

    parser = argparse.ArgumentParser(description="Update XML configuration parameters.")
    # Will use the default value of config_file if none is provided via command line:
    # https://docs.python.org/3/library/argparse.html#nargs
    parser.add_argument(
        "config_file", type=str, help="Path to the XML parameter file.", nargs="?"
    )  # , default = rgb_config_path)

    rgb_config = get_rgb_configuration(
        parser=parser
    )  # , config_file = rgb_config_path)

    # Ingest and Extract L1B Data
    timed_obj = RGBLogging.rgb_decorate_and_execute(
        decorate=rgb_config.logpar_decorate,
        decorator=RGBLogging.track_perf,
        logger=rgb_config.logger,
    )(DataIngestion)

    ingestion_object = timed_obj(rgb_config)

    timed_func = RGBLogging.rgb_decorate_and_execute(
        decorate=rgb_config.logpar_decorate,
        decorator=RGBLogging.track_perf,
        logger=rgb_config.logger,
    )(ingestion_object.ingest_data)

    data_dict = timed_func()

    # Regrid Data
    timed_func = RGBLogging.rgb_decorate_and_execute(
        decorate=rgb_config.logpar_decorate,
        decorator=RGBLogging.track_perf,
        logger=rgb_config.logger,
    )(ReGridder)

    # regridder         = ReGridder(rgb_config)
    regridder = timed_func(rgb_config)

    # if rgb_config.input_data_type == 'SMAP':

    #     timed_func        = RGBLogging.rgb_decorate_and_execute(
    #             decorate  = rgb_config.logpar_decorate,
    #             decorator = RGBLogging.track_perf,
    #             logger    = rgb_config.logger
    #             )(regridder.regrid_data)

    #     data_dict_out     = timed_func(data_dict)
    #
    # if rgb_config.input_data_type == 'AMSR2':

    #     data_dict_out = regridder.regrid_data(data_dict)

    # if rgb_config.input_data_type == 'CIMR':

    #     #data_dict_out = regridder.regrid_data(data_dict)
    #     timed_func        = RGBLogging.rgb_decorate_and_execute(
    #             decorate  = rgb_config.logpar_decorate,
    #             decorator = RGBLogging.track_perf,
    #             logger    = rgb_config.logger
    #             )(regridder.regrid_data)
    #     data_dict_out     = timed_func(data_dict)

    timed_func = RGBLogging.rgb_decorate_and_execute(
        decorate=rgb_config.logpar_decorate,
        decorator=RGBLogging.track_perf,
        logger=rgb_config.logger,
    )(regridder.regrid_data)

    data_dict_out = timed_func(data_dict)

    # Generate L1R/C product according to CDL
    timed_func = RGBLogging.rgb_decorate_and_execute(
        decorate=rgb_config.logpar_decorate,
        decorator=RGBLogging.track_perf,
        logger=rgb_config.logger,
        # )(ProductGenerator(rgb_config).generate_product)
    )(ProductGenerator(rgb_config).generate_new_product)
    # regridder         = ReGridder(rgb_config)
    regridder = timed_func(data_dict=data_dict_out)

    # Record the end time and resource usage
    end_time = time.perf_counter()

    cpu_time_end = process.cpu_times()
    cpu_usage_end = process.cpu_percent(interval=None)

    # Memory size in bytes / 1024**2 = memory size in MB
    memory_usage_end = process.memory_info().rss / 1024**2

    # Calculate metrics
    elapsed_time = end_time - start_time

    user_cpu_time = cpu_time_end.user - cpu_time_start.user
    system_cpu_time = cpu_time_end.system - cpu_time_start.system
    total_cpu_time = user_cpu_time + system_cpu_time

    memory_usage_change = memory_usage_end - memory_usage_start

    # Log performance metrics
    rgb_config.logger.info(f"`{main.__name__}` -- Executed in: {elapsed_time:.2f}s")
    rgb_config.logger.info(
        f"`{main.__name__}` -- CPU User Time (Change): {user_cpu_time:.2f}s"
    )
    rgb_config.logger.info(
        f"`{main.__name__}` -- CPU System Time: {system_cpu_time:.2f}s"
    )
    rgb_config.logger.info(
        f"`{main.__name__}` -- CPU Total Time: {total_cpu_time:.2f}s"
    )
    rgb_config.logger.info(
        f"`{main.__name__}` -- Process-Specific CPU Usage (Before): {cpu_usage_start:.2f}%"
    )
    rgb_config.logger.info(
        f"`{main.__name__}` -- Process-Specific CPU Usage (After): {cpu_usage_end:.2f}%"
    )
    rgb_config.logger.info(
        f"`{main.__name__}` -- Memory Usage Change: {memory_usage_change:.6f} MB"
    )

    rgb_config.logger.info("=====================")

    # Intermediate results check
    # Put in the variables you want from the data_dict_out in data_dict.
    # ProductGenerator(rgb_config).generate_l1c_product(data_dict = data_dict_out)
    # l1c_path = "/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/L1C/SMAP/NASA/SMAP_L1C_TB_47185_D_20231201T212059_R19240_002.h5"
    # plot = compare_smap_l1c(rgb_config, l1c_path).plot_diff(data_dict_out, 'bt_h_fore')

    # # Intermediate results check
    # # Put in the variables you want from the data_dict_out in data_dict.
    # grid_shape = GRIDS[rgb_config.grid_definition]['n_rows'], GRIDS[rgb_config.grid_definition]['n_cols']
    # # # # create nan array with shape of grid_shape
    # grid = full(grid_shape, nan)
    # variable = data_dict_out['L']['bt_h_fore']
    # cell_row = data_dict_out['L']['cell_row_fore']
    # cell_col = data_dict_out['L']['cell_col_fore']
    # for i in range(len(cell_row)):
    #     grid[cell_row[i], cell_col[i]] = variable[i]
    # plt.imshow(grid)


if __name__ == "__main__":
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
    # if config.grid_type == 'L1R':
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
