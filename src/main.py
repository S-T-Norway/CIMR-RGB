"""
This module is the entry point for the RGB.
It is responsible for calling the other functions in the script
It is also responsible for handling the command line arguments
It is also responsible for handling the exceptions that are raised
It is also responsible for printing the output to the console
It is also responsible for returning the output to the caller
"""
import os

from data_ingestion import DataIngestion
from grid_generator import GridGenerator
from regridder import ReGridder




if __name__ == '__main__':
    # This is the main function that is called when the script is run
    # It is the entry point of the script


    # Ingest and Extract L1b Data
    ingestion_object = DataIngestion(os.path.join(os.getcwd(), '..', 'config.xml'))
    config = ingestion_object.config
    data_dict = ingestion_object.ingest_data()

    # Create appropriate grid
    grid_object = (GridGenerator(config)
                   .generate_swath_grid(target_lons=data_dict['lons_target'],
                                        target_lats=data_dict['lats_target']))
    # Regrid Data
    if config.input_data_type == 'SMAP':
        data_dict_out = ReGridder(config).regrid_data(data_dict)
    # elif config.input_data_type == 'AMSR2':
    #     data_dict_out = ReGridder(config).regrid_amsr2_data(data_dict)

