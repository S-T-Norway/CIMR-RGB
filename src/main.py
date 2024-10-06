"""
This module is the entry point for the RGB.
It is responsible for calling the other functions in the script
It is also responsible for handling the command line arguments
It is also responsible for handling the exceptions that are raised
It is also responsible for printing the output to the console
It is also responsible for returning the output to the caller
"""
import os
impoer pathlib as pb 
import pickle
from numpy import full, nan

from data_ingestion import DataIngestion
from grid_generator import GridGenerator, GRIDS
from regridder_v2 import ReGridder

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

    # Regrid Data
    if config.input_data_type == 'SMAP':
        data_dict_out = ReGridder(config).regrid_l1c(data_dict)

    if config.input_data_type == 'AMSR2':
        data_dict_out = ReGridder(config).regrid_l1r(data_dict)

    if config.input_data_type == 'CIMR':
            data_dict_out= ReGridder(config).regrid_l1c(data_dict)


    # Intermediate results check
    # Put in the variables you want from the data_dict_out in data_dict.
    # grid_shape = GRIDS[config.grid_definition]['n_rows'], GRIDS[config.grid_definition]['n_cols']
    # # create nan array with shape of grid_shape
    # grid = full(grid_shape, nan)
    # variable = data_dict_out['bt_h_fore']
    # cell_row = data_dict_out['cell_row_fore']
    # cell_col = data_dict_out['cell_col_fore']
    # for i in range(len(cell_row)):
    #     grid[cell_row[i], cell_col[i]] = variable[i]

    # Load equivalent L1c Data here if you want to compare



    # Temp save solution
    # save_path = ''
    # # Save pickle dictionary to save path
    # with open(save_path, 'wb') as f:
    #     pickle.dump(data_dict_out, f)







