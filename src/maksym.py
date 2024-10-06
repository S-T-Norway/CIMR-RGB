import sys 
import pathlib as pb 
import json
import os
import pickle

from numpy import full, nan, array

# Getting custom modules
# Add the root of your project to PYTHONPATH
rootpath = pb.Path('.').resolve().parents[0]
syspath = str(rootpath.joinpath("src", "cimr-grasp"))
sys.path.append(syspath) 
syspath = str(rootpath.joinpath("src"))
sys.path.append(syspath) 

from data_ingestion import DataIngestion
from grid_generator import GridGenerator, GRIDS
from regridder_v2 import ReGridder
from config_file import ConfigFile
# Importing custom made modules  
from cimr_grasp import grasp_io as io  

# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import simplekml


if __name__ == '__main__':

    # Initiate config object and validate parameters.
    #config = ConfigFile(os.path.join(os.getcwd(), '..', 'config.xml'))
    # Initiate config object and validate parameters.
    rootpath = io.find_repo_root()
    configpath = rootpath.joinpath('config-maks.xml')
    config = ConfigFile(configpath) 

    # You can already run GridGenerator with the config object. You can generate
    # an x, y grid and then convert those coordinates to lon, lat. The type of
    # grid that will be generated will of course depend on what you have put in the
    # config file. I would try this first using an EASE grid. For example, put in the config
    # gridDefinition='EASE_G36' and projectionDefinition = 'G'.

    #x_grid, y_grid = GridGenerator(config).generate_grid_xy()
    #print(x_grid.shape) 
    #x, y = GridGenerator(config).lonlat_to_xy([3, 45, 78], [1, 3, 9])
    #exit() 

    # Convert the grid to lon, lat
    #lon_grid, lat_grid = GridGenerator(config).xy_to_lonlat(x_grid, y_grid)
    #print(lon_grid)

    # Now you can play around with some SMAP data. The l1b data is given in lon, lat (epsg:4326)
    # and you may want to convert to x, y. First run data_ingestion to get the data dict
    
    # ingest_data -> ingest_smap -> read_hdf5, (clean_data -> remove_out_of_bounds -> lonlat_to_xy)
    data_dict = DataIngestion(config).ingest_data()
    print(data_dict['L'].keys())

    # Convert coordinates to x, y
    # Depending on whether you have split_fore_aft = True or False, you will have to change the name
    # of the input lon, lat.
    lon_sample = data_dict['L']['longitude_fore']
    lat_sample = data_dict['L']['latitude_fore']
    x_sample, y_sample = GridGenerator(config).lonlat_to_xy(lon_sample, lat_sample)
    #print(lon_sample.shape)
    #exit() 

    # For the functions xy_to_lonlat and lonlat_to_xy, you can also just use a random single data
    # point to validate its working correctly for example:
    x_test = 0
    y_test = 0
    lon_test, lat_test = GridGenerator(config).xy_to_lonlat(x_test, y_test)

    # Now if you want to regrid the data to the grid you have generated, you can call the regridder
    # function. Note: the regridder function calls the grid generator within it. I would start with
    # NN.

    data_dict_out = ReGridder(config).regrid_l1c(data_dict)
    exit() 

    # Data_dict out is the same format as data_dict, with the added variables cell_row, cell_col
    # which provide the location of each data point in the ease array. I think it is highly likely
    # that this won't work straight away with the new grid, but we can deal with it when you get there.

    # Here you can save any of the variables you want to inspect.
    save_path = '/run/media/eva-v3/seagate_small/Maks/CurrProjects/SaT/CIMR-RGB/code/cimr-126/output/proj_test'
    with open(os.path.join(save_path, 'data_dict_out.pkl'), 'wb') as f:
        pickle.dump(data_dict_out, f)

    # The following code plots the regridded data onto an EASE grid.
    # Get shape from GRIDS dictionary
    grid_shape = GRIDS[config.grid_definition]['n_rows'], GRIDS[config.grid_definition]['n_cols']
    # Create nan array with shape of grid_shape
    grid = full(grid_shape, nan)
    # Extract the variable you want plot
    variable = data_dict_out['L']['bt_h_fore']
    # Extract the locations on the grid
    cell_row = data_dict_out['L']['cell_row_fore']
    cell_col = data_dict_out['L']['cell_col_fore']
    # Put the data in the grid
    for i in range(len(cell_row)):
        grid[cell_row[i], cell_col[i]] = variable[i]
    # Plot Grid
    plt.imshow(grid)
    plt.show() 


