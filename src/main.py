"""
This module is the entry point for the RGB.
It is responsible for calling the other functions in the script
It is also responsible for handling the command line arguments
It is also responsible for handling the exceptions that are raised
It is also responsible for printing the output to the console
It is also responsible for returning the output to the caller
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "tests"))
import pickle
from numpy import full, nan, array

from data_ingestion import DataIngestion
from grid_generator import GridGenerator, GRIDS
from regridder import ReGridder
from config_file import ConfigFile
from inspect_SMAP_l1c import compare_smap_l1c


# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import simplekml

if __name__ == '__main__':
    # This is the main function that is called when the script is run
    # It is the entry point of the script

    # Ingest and Extract L1B Data
    config = ConfigFile(os.path.join(os.getcwd(), '..', 'config.xml'))
    data_dict = DataIngestion(config).ingest_data()

    # Regrid Data
    if config.input_data_type == 'SMAP':
            data_dict_out = ReGridder(config).regrid_l1c(data_dict)

    if config.input_data_type == 'AMSR2':
            data_dict_out = ReGridder(config).regrid_l1c(data_dict)

    if config.input_data_type == 'CIMR':
            data_dict_out = ReGridder(config).regrid_l1c(data_dict)

#     l1c_path = '../dpr/L1C/SMAP/NASA_BG/SMAP_L1C_TB_E_47185_D_20231201T212059_R19240_002.h5'
#     compare_smap_l1c(config, l1c_path).plot_diff(data_dict_out, 'bt_h_fore')

    # Intermediate results check
    # Put in the variables you want from the data_dict_out in data_dict.
    # if config.grid_type == 'L1C':
    #     grid_shape = GRIDS[config.grid_definition]['n_rows'], GRIDS[config.grid_definition]['n_cols']
    #     # # create nan array with shape of grid_shape
    #     grid = full(grid_shape, nan)
    #     variable = data_dict_out['L']['bt_h']
    #     cell_row = data_dict_out['L']['cell_row']
    #     cell_col = data_dict_out['L']['cell_col']
    #     for i in range(len(cell_row)):
    #         grid[cell_row[i], cell_col[i]] = variable[i]
    #     plt.imshow(grid)
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
    #     variable = 'bt_h'
    #     grid_shape = config.num_target_scans, config.num_target_samples
    #     grid = full(grid_shape, nan)
    #     scan_number = data_dict_out['C']['cell_row']
    #     sample_number = data_dict_out['C']['cell_col']
    #     for count, sample in enumerate(data_dict_out['C'][variable]):
    #         grid[int(scan_number[count]), int(sample_number[count])] = sample
    #     plt.imshow(grid)

    #     test = 0












