"""
This  module is the entry point for the RGB.
It is responsible for calling the other functions in the script
It is also responsible for handling the command line arguments
It is also responsible for handling the exceptions that are raised
It is also responsible for printing the output to the console
It is also responsible for returning the output to the caller
"""

# import os
# import sys

import pathlib as pb

# import pickle
import argparse
import datetime
import time


# from numpy import full, nan, array
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

from cimr_rgb.config_file import ConfigFile
from cimr_rgb.data_ingestion import DataIngestion

# from cimr_rgb.grid_generator import GridGenerator, GRIDS
from cimr_rgb.regridder import ReGridder
from cimr_rgb.rgb_logging import RGBLogging
from cimr_rgb.product_generator import ProductGenerator
import cimr_grasp.grasp_io as grasp_io


def get_rgb_configuration(
    parser: argparse.ArgumentParser,
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
    - `-t`, `--input-data-type` : str
        Type of input data.
    - `-p`, `--input-data-path` : str
        Path to input data.
    - `-app`, `--antenna-patterns-path` : str
        Path to antenna patterns.
    - `-sfa`, `--split-fore-aft` : str
        Fore/Aft split for the input data.
    - `-sb`, `--source-band` : str
        Source band of the input data.
    - `-tb`, `--target-band` : str
        Target band of the input data.
    - `-qc`, `--quality-control` : str
        Enable quality control.
    - `-gt`, `--grid-type` : str
        Type of grid parameters.
    - `-gd`, `--grid-definition` : str
        Definition of grid parameters.
    - `-pd`, `--projection-definition` : str
        Projection definition of grid parameters.
    - `-rg`, `--reduced-grid-inds` : str
        Reduced grid indices.
    - `-ra`, `--regridding-algorithm` : str
        Algorithm for regridding parameters.
    - `-sr`, `--search-radius` : str
        Search radius for regridding.
    - `-mn`, `--max-neighbours` : str
        Maximum number of neighbours.
    - `-vtr`, `--variables-to-regrid` : str
        Variables to regrid.
    - `-sam`, `--source-antenna-method` : str
        Source antenna method.
    - `-tam`, `--target-antenna-method` : str
        Target antenna method.
    - `-pm`, `--polarisation-method` : str
        Polarisation method.
    - `-sat`, `--source-antenna-threshold` : str
        Source antenna threshold.
    - `-tat`, `--target-antenna-threshold` : str
        Target antenna threshold.
    - `-mtap`, `--max-theta-antenna-patterns` : str
        Maximum theta antenna patterns.
    - `-mat`, `--mrf-grid-definition` : str
        MRF grid definition.
    - `-mpt`, `--mrf-projection-definition` : str
        MRF projection definition.
    - `-sgp`, `--source-gaussian-params` : str
        Source Gaussian parameters.
    - `-tgp`, `--target-gaussian-params` : str
        Target Gaussian parameters.
    - `-bs`, `--boresight-shift` : str
        Enable boresight shift.
    - `-rsir`, `--rsir-iteration` : str
        RSIR iteration count.
    - `-bgs`, `--bg-smoothing` : str
        Background smoothing level.
    - `-aps`, `--antenna-patterns-uncertainty` : str
        Antenna patterns uncertainty.
    - `-nedtl`, `--cimr-l-nedt` : float
        CIMR L-band NEDT.
    - `-nedtc`, `--cimr-c-nedt` : float
        CIMR C-band NEDT.
    - `-nedtx`, `--cimr-x-nedt` : float
        CIMR X-band NEDT.
    - `-nedtk`, `--cimr-k-nedt` : float
        CIMR K-band NEDT.
    - `-nedtka`, `--cimr-ka-nedt` : float
        CIMR Ka-band NEDT.
    - `-std`, `--save-to-disk` : bool
        Save output data to disk.
    - `-op`, `--output-path` : str
        Output path for data.
    - `-v`, `--version` : str
        Version of the configuration.
    - `-cn`, `--creator-name` : str
        Name of the creator.
    - `-ce`, `--creator-email` : str
        Email of the creator.
    - `-cu`, `--creator-url` : str
        URL of the creator.
    - `-ci`, `--creator-institution` : str
        Institution of the creator.
    - `-sf`, `--suffix` : str
        Suffix for output files.
    - `-tf`, `--timestamp-fmt` : str
        Timestamp format for output.
    - `-cp`, `--logging-params-config` : str
        Path to the logging configuration.
    - `-d`, `--logging-params-decorate` : str
        Whether to use logging decoration.
    - `-ln`, `--logger-name` : str
        Name of the logger.

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
        "ReGridderParams/antenna_pattern_uncertainty": [
            "aps",
            "antenna-patterns-uncertainty",
            str,
            "Value for ReGridderParams/antenna_pattern_uncertainty parameter.",
        ],
        "ReGridderParams/cimr_L_nedt": [
            "nedtl",
            "cimr-l-nedt",
            str,
            "Value for ReGridderParams/cimr_L_nedt parameter.",
        ],
        "ReGridderParams/cimr_C_nedt": [
            "nedtc",
            "cimr-c-nedt",
            str,
            "Value for ReGridderParams/cimr_C_nedt parameter.",
        ],
        "ReGridderParams/cimr_X_nedt": [
            "nedtx",
            "cimr-x-nedt",
            str,
            "Value for ReGridderParams/cimr_X_nedt parameter.",
        ],
        "ReGridderParams/cimr_K_nedt": [
            "nedtk",
            "cimr-k-nedt",
            str,
            "Value for ReGridderParams/cimr_K_nedt parameter.",
        ],
        "ReGridderParams/cimr_KA_nedt": [
            "nedtka",
            "cimr-ka-nedt",
            str,
            "Value for ReGridderParams/cimr_KA_nedt parameter.",
        ],
        "ReGridderParams/regularisation_parameter": [
            "rp",
            "regularisation-parameter",
            str,
            "Value for ReGridderParams/regularisation_parameter parameter.",
        ],
        "ReGridderParams/max_iterations": [
            "mi",
            "max-iterations",
            str,
            "Value for ReGridderParams/max_iterations parameter.",
        ],
        "ReGridderParams/relative_tolerance": [
            "rt",
            "relative-tolerance",
            str,
            "Value for ReGridderParams/relative_tolerance parameter.",
        ],
        "ReGridderParams/max_chunk_size": [
            "mcs",
            "max-chunk-size",
            str,
            "Value for ReGridderParams/max_chunk_size parameter.",
        ],
        "ReGridderParams/chunk_buffer": [
            "cb",
            "chunk-buffer",
            str,
            "Value for ReGridderParams/chunk_buffer parameter.",
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
        "OutputData/suffix": [
            "sf",
            "suffix",
            str,
            "Value for OutputData/suffix parameter.",
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

    outputdir = ConfigFile.validate_output_directory_path(
        config_object=root, output_path="OutputData/output_path", logger=None
    )

    # Creating output directory for logs (to save log files there)
    logdir = pb.Path(outputdir).joinpath("logs")
    # Create the directory structure if necessary
    grasp_io.rec_create_dir(path=logdir)

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

    logger.info("CIMR RGB Configuration")

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
            "Source Antenna Method:      N/A"
        )  # {rgb_config.source_antenna_method}")
        logger.info(
            "Target Antenna Method:      N/A"
        )  # {rgb_config.target_antenna_method}")
        logger.info(
            "Polarisation Method:        N/A"
        )  # {rgb_config.polarisation_method}")
        logger.info(
            "Source Antenna Threshold:   N/A"
        )  # {rgb_config.source_antenna_threshold}")
        logger.info(
            "Target Antenna Threshold:   N/A"
        )  # {rgb_config.target_antenna_threshold}")
        logger.info(
            "Max Theta Antenna Patterns: N/A"
        )  # {rgb_config.max_theta_antenna_patterns}")
        logger.info(
            "MRF Grid Definition:        N/A"
        )  # {rgb_config.MRF_grid_definition}")
        logger.info(
            "MRF Grid Definition:        N/A"
        )  # {rgb_config.MRF_projection_definition}")
        logger.info(
            "Source Gaussian Params:     N/A"
        )  # {rgb_config.source_gaussian_params}")
        logger.info(
            "Target Gaussian Params:     N/A"
        )  # {rgb_config.target_gaussian_params}")
    if rgb_config.regridding_algorithm in ["RSIR"]:
        logger.info(f"rSIR Iteration:             {rgb_config.rsir_iteration}")
    else:
        logger.info("rSIR Iteration:             N/A")  # {rgb_config.rsir_iteration}")
    if rgb_config.regridding_algorithm in ["BG"]:
        logger.info(f"BG Smoothing:               {rgb_config.bg_smoothing}")
    else:
        logger.info("BG Smoothing:               N/A")  # {rgb_config.bg_smoothing}")
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

    # Get the current process
    process = psutil.Process()

    # Record the start time and resource usage
    start_time = time.perf_counter()

    cpu_time_start = process.cpu_times()
    cpu_usage_start = process.cpu_percent(interval=None)

    # Memory size in bytes / 1024**2 = memory size in MB
    memory_usage_start = process.memory_info().rss / 1024**2

    # Setting the default value of the configuration parameter
    # rgb_config_path = pb.Path("configs", "rgb_config.xml").resolve()
    # If running from IDE, comment out above and use the line below
    # rgb_config_path = pb.Path('/home/beywood/ST/CIMR_RGB/CIMR-RGB/configs', 'rgb_config.xml').resolve()

    parser = argparse.ArgumentParser(description="Update XML configuration parameters.")
    # Will use the default value of config_file if none is provided via command line:
    # https://docs.python.org/3/library/argparse.html#nargs
    parser.add_argument(
        "config_file", type=str, help="Path to the XML parameter file.", nargs="?"
    )

    rgb_config = get_rgb_configuration(parser=parser)

    RGBLogging.setup_global_exception_handler(logger=rgb_config.logger)

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


if __name__ == "__main__":
    main()
