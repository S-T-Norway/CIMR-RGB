"""
This module is the entry point for the RGB.
It is responsible for calling the other functions in the script
It is also responsible for handling the command line arguments
It is also responsible for handling the exceptions that are raised
It is also responsible for printing the output to the console
It is also responsible for returning the output to the caller
"""
import os
import pickle

from data_ingestion import DataIngestion
from grid_generator import GridGenerator
from regridder import ReGridder

# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # This is the main function that is called when the script is run
    # It is the entry point of the script


    # Ingest and Extract L1b Data
    ingestion_object = DataIngestion(os.path.join(os.getcwd(), '..', 'config.xml'))
    config = ingestion_object.config
    data_dict = ingestion_object.ingest_data()

    # test = ids_kernel_weights(config, data_dict)

    # Regrid Data
    if config.input_data_type == 'SMAP':
        data_dict_out = ReGridder(config).regrid_l1c(data_dict)


        # Save the output, functionality to be added to config file.
        granule_name = (config.input_data_path.split('/')[-1].split('.')[0]).replace('L1B', 'L1C')
        with open(
                os.path.join(config.dpr_path,
                             f'/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/L1cProducts/SMAP/RGB/{granule_name}_{config.grid_definition}.pkl'), 'wb') as f:
            pickle.dump(data_dict_out, f)

    if config.input_data_type == 'AMSR2':
        data_dict_out = ReGridder(config).regrid_l1r(data_dict)
        # Save dictionary with pickle
        # change to 1r
        granule_name = (config.input_data_path.split('/')[-1].split('.')[0]).replace('GB', 'GR')
        with open(
                os.path.join(config.dpr_path,
                             f'/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/L1rProducts/AMSR2/RGB/{granule_name}_{config.source_band}_{config.target_band}.pkl'), 'wb') as f:
            pickle.dump(data_dict_out, f)

    if config.input_data_type == 'CIMR':
        # maybe this loop should go in the regridder
        data_dict_out = {}
        for band in data_dict:
            data_dict_out = ReGridder(config).regrid_l1c(data_dict[band])

        test = 0


