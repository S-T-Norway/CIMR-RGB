from netCDF4 import Dataset
from pyresample import kd_tree, geometry
import pickle

import sys, os
from numpy import array, where, nan, arange
import numpy as np
from cimr_rgb.grid_generator import GridGenerator
from cimr_rgb.config_file import ConfigFile
from scipy.stats import binned_statistic_2d
import cimr_grasp.grasp_io as grasp_io


####### parameters

bands = ['L_BAND', 'C_BAND', 'X_BAND', 'KA_BAND', 'K_BAND']
INCIDENCE_ANGLE  = 55


####### helper functions

def get_data(truth_data, INCIDENCE_ANGLE):
    with Dataset(truth_data, 'r') as data:
        BTs = {}
        if 'Latitude' in data.variables:
            latitude = array(data.variables['Latitude'][:,:])
            longitude = array(data.variables['Longitude'][:,:])
            BTs['L_BAND'] = array(data.variables['L_band_H'][:,:])
            BTs['C_BAND'] = array(data.variables['C_band_H'][:,:])
            BTs['X_BAND'] = array(data.variables['X_band_H'][:,:])
            BTs['K_BAND'] = array(data.variables['Ku_band_H'][:,:])
            BTs['KA_BAND'] = array(data.variables['Ka_band_H'][:,:])

        elif 'latitude' in data.variables:
            latitude = array(data.variables['latitude'][0,:,:])
            longitude = array(data.variables['longitude'][0,:,:])   
            longitude -= 360   
            incidence_angle = array(data.variables['incidence_angle'][:])
            try:
                incidence_index = where(incidence_angle == INCIDENCE_ANGLE)[0][0]
            except:
                raise ValueError(f'Incidence Angle {INCIDENCE_ANGLE} not found in the data, '
                                 f'available angles are {incidence_angle}')
            # Only using Hpol
            BTs['L_BAND'] = array(data.variables['toa_tbs_L_Hpo'][0,incidence_index,:,:])
            BTs['C_BAND'] = array(data.variables['toa_tbs_C_Hpo'][0,incidence_index,:,:])
            BTs['X_BAND'] = array(data.variables['toa_tbs_X_Hpo'][0,incidence_index,:,:])
            BTs['K_BAND'] = array(data.variables['toa_tbs_Ku_Hpo'][0,incidence_index,:,:])

    return latitude, longitude, BTs


def fit_to_ease(source_lons, source_lats, BTs, ease_grid, bands):


    """
    Resample the origin test card data onto an EASE grid

    Params:
        source_lons (1D array of floats): longitudes of data in the test card
        source_lats (1D array of floats): latitudes of data in the test card
        target_lons (2D array of floats): longitudes of cell centers in an EASE grid
        target_lats (2D array of floats): longitudes of cell centers in an EASE grid
        BTs (1D array of floats): values of the brightness temperature in the test card
        bands (list of strings): list of the bands, such as "L_BAND", "C_BAND", etc. 
    """

    repo_root = grasp_io.find_repo_root()
    config_path = repo_root.joinpath("tests/Performance_Assesment/performance_assessment_base.xml")
    config_object = ConfigFile(config_path)   
    grid = GridGenerator(config_object, ease_grid[6], ease_grid)
    source_x, source_y = grid.lonlat_to_xy(source_lons, source_lats)
    x, y = grid.generate_grid_xy()
    x = x.ravel()
    y = y.ravel()[::-1]
    x_edges = x - grid.resolution/2.
    y_edges = y - grid.resolution/2.
    x_edges = np.append(x_edges, x[-1] + grid.resolution/2.)
    y_edges = np.append(y_edges, y[-1] + grid.resolution/2.)
    edges_lons, edges_lats = grid.xy_to_lonlat(x_edges, y_edges)    #THIS WON'T WORK FOR POLAR SCENES, SINCE LON AND LAT WILL NOT BE MONOTONICAL

    EASE_BTs = {}

    for band in bands:
        EASE_BTs[band], _, _, _ =  binned_statistic_2d(source_x.flatten(), source_y.flatten(), BTs[band].flatten(), 
                                                       statistic='mean', bins=[x_edges, y_edges])
        #transpose needed because binned_statistic_2d is from scipy, so row and columns are inverted
        #first axis is inverted because latitudes are given in decreasing order in the ease grid
        EASE_BTs[band] = EASE_BTs[band].T[::-1, :]
    return EASE_BTs


def save_scene_on_ease(test_scene_path, target_ease_grids, output_id):
    target_ease_grids = np.atleast_1d(target_ease_grids)
    output_dir = os.path.dirname(test_scene_path)
    latitude, longitude, BTs = get_data(test_scene_path, INCIDENCE_ANGLE)
    for target_ease_grid in target_ease_grids:
        print(f"Interpolating {output_id} on a {target_ease_grid}")
        EASE_BTs = fit_to_ease(
            source_lons=longitude,
            source_lats=latitude,
            BTs=BTs,
            ease_grid = target_ease_grid,
            bands=bands    
            )
        np.savez_compressed(f"{os.path.join(output_dir, output_id)}_{target_ease_grid}", **EASE_BTs)
        print('.. done')
    

####### main 

if __name__ == '__main__':

    repo_root = grasp_io.find_repo_root()

    test_scene_central_america = repo_root.joinpath("dpr/Test_cards/SCEPS_central_america/cimr_sceps_toa_card_central_america_20161217_v2p0_aa_000.nc")
    test_scene_polar = repo_root.joinpath("dpr/Test_cards/SCEPS_polar_scene/cimr_sceps_toa_card_devalgo_polarscene_1_20161217_v2p0_aa_000.nc")
    test_scene_1 = repo_root.joinpath("dpr/Test_cards/SCEPS_test_scene_1/test_scene_1_compressed_lowres.nc")
    test_scene_2 = repo_root.joinpath("dpr/Test_cards/SCEPS_test_scene_2/test_scene_2_compressed_lowres.nc")
    test_scene_3 = repo_root.joinpath("dpr/Test_cards/SCEPS_test_scene_3/DEVALGO_testcard_soil_moisture.nc")

    # target EASE grids
    target_G_ease_grids = ["EASE2_G3km", "EASE2_G9km", "EASE2_G36km"]
    target_N_ease_grids = ["EASE2_N3km", "EASE2_N9km", "EASE2_N36km"]

    # save_scene_on_ease(test_scene_central_america, target_G_ease_grids, 'SCEPS_central_america')
    # save_scene_on_ease(test_scene_polar, target_N_ease_grids, 'SCEPS_polar_scene')
    # save_scene_on_ease(test_scene_1, target_N_ease_grids, 'SCEPS_test_scene_1')
    save_scene_on_ease(test_scene_2, target_N_ease_grids, 'SCEPS_test_scene_2')
    save_scene_on_ease(test_scene_3, target_G_ease_grids, 'SCEPS_test_scene_3')

    



