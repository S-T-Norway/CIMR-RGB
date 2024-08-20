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




    # Ingest and Extract L1B Data
    ingestion_object = DataIngestion(os.path.join(os.getcwd(), '..', 'config.xml'))
    config = ingestion_object.config
    data_dict = ingestion_object.ingest_data()

    # test = ids_kernel_weights(config, data_dict)

    # Regrid Data
    if config.input_data_type == 'SMAP':
        data_dict_out = ReGridder(config).regrid_l1c(data_dict)

    if config.input_data_type == 'AMSR2':
        data_dict_out = ReGridder(config).regrid_l1r(data_dict)

    if config.input_data_type == 'CIMR':
        data_dict_out = {}
        for band in data_dict:
            data_dict_out[band] = ReGridder(config_object=config, band_to_remap=band).regrid_l1c(data_dict[band])



    if config.save_to_disk == True:
        if config.grid_type == 'L1C':
            grid_string = config.grid_definition
        elif config.grid_type == 'L1R':
            grid_string = f'{config.source_band}_{config.target_band}'
        if config.input_data_type == 'SMAP':
            replace_in = 'L1B'
            replace_out = config.grid_type
        elif config.input_data_type == 'AMSR2':
            replace_in = 'GBT'
            if config.grid_type == 'L1C':
                replace_out = 'GCT'
            elif config.grid_type == 'L1R':
                replace_out = 'GRT'
                grid_string = grid_string + f'_{str(config.kernel_size)}'
        elif config.input_data_type == 'CIMR':
            replace_in = 'l1b'
            replace_out = config.grid_type

        save_path = os.path.join(
           config.dpr_path,
           config.grid_type,
           config.input_data_type,
           'RGB',
           config.input_data_path.split('/')[-1].split('.')[0].replace(replace_in, replace_out) +
            '_' + grid_string + '.pkl')
        with open(save_path, 'wb') as f:
             pickle.dump(data_dict_out, f)
        print(f"Data saved to:"
              f"{save_path}")



