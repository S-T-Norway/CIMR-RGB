"""
This script is used to superimpose truth data from the cimr_sceps_toa_card_central_america_20161217_v2p0_aa_000.nc
onto EASE grids of different sizes for each CIMR band. This allows us to compare the EASE RGB output to truth and
calculate performance metrics.
"""

from netCDF4 import Dataset
from numpy import array, where, nan, arange
from pyresample import kd_tree, geometry
import pickle

import sys
sys.path.append('/cimr_rgb')
from cimr_rgb.grid_generator import GridGenerator
from cimr_rgb.config_file import ConfigFile

import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

# Configuration
config_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/quick_config.xml'
central_america_truth_scene = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_central_america/cimr_sceps_toa_card_central_america_20161217_v2p0_aa_000.nc'
sceps_3_truth_scene = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_3/DEVALGO_testcard_soil_moisture.nc'
output_folder = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_3/EASE_grids'
output_file_tag = 'SCEPS_3'
output_grid_resolutions = [3, 9, 36]
bands = ['L_BAND', 'C_BAND', 'X_BAND', 'KA_BAND', 'K_BAND']
INCIDENCE_ANGLE  = None


def get_data(truth_data, INCIDENCE_ANGLE=None):

    if INCIDENCE_ANGLE:
        # This if will be for the central america truth scene, which has incidence angles (and I presume for the polar scene too)
        with Dataset(truth_data, 'r') as data:
            latitude = array(data.variables['latitude'][0,:,:])
            longitude = array(data.variables['longitude'][0,:,:])
            if longitude.max() > 180 or longitude.min() < -180:
                longitude[longitude > 180] = longitude[longitude > 180] - 360
            incidence_angle = array(data.variables['incidence_angle'][:])
            try:
                incidence_index = where(incidence_angle == INCIDENCE_ANGLE)[0][0]
            except:
                raise ValueError(f'Incidence Angle {INCIDENCE_ANGLE} not found in the data, '
                                 f'available angles are {incidence_angle}')
            BTs = {}
            # Only using Hpol
            BTs['L_BAND'] = array(data.variables['toa_tbs_L_Hpo'][0,incidence_index,:,:])
            BTs['C_BAND'] = array(data.variables['toa_tbs_C_Hpo'][0,incidence_index,:,:])
            BTs['X_BAND'] = array(data.variables['toa_tbs_X_Hpo'][0,incidence_index,:,:])
            BTs['K_BAND'] = array(data.variables['toa_tbs_Ku_Hpo'][0,incidence_index,:,:])
            BTs['KA_BAND'] = array(data.variables['toa_tbs_Ka_Hpo'][0,incidence_index,:,:])

    else:
        # This works for the SCEPS_3 scene
        with Dataset(truth_data, 'r') as data:
            latitude = array(data.variables['Latitude'][:, :])
            longitude = array(data.variables['Longitude'][:, :])
            BTs = {}

            # Using only Hpol
            BTs['L_BAND'] = array(data.variables['L_band_H'][:, :])
            BTs['C_BAND'] = array(data.variables['C_band_H'][:, :])
            BTs['X_BAND'] = array(data.variables['X_band_H'][:, :])
            BTs['K_BAND'] = array(data.variables['Ku_band_H'][:, :])
            BTs['KA_BAND'] = array(data.variables['Ka_band_H'][:, :])




    return latitude, longitude, BTs

def fit_to_ease(source_lons, source_lats, target_lons, target_lats, BTs, bands, search_radius):

    source_def = geometry.SwathDefinition(lons=source_lons.flatten(), lats=source_lats.flatten())
    target_def = geometry.GridDefinition(lons=target_lons, lats=target_lats)

    EASE_BTs = {}
    for band in bands:
        regridded_bt = kd_tree.resample_nearest(
            source_geo_def = source_def,
            target_geo_def = target_def,
            data = BTs[band].flatten(),
            radius_of_influence = search_radius,
            fill_value= nan
        )
        EASE_BTs[band] = regridded_bt

    return EASE_BTs

def resample_ease_grids(data, x, y, block_size=3):
    n_y, n_x = data.shape

    # 1) Ensure each dimension is divisible by block_size (clip or pad if needed).
    n_y_clipped = (n_y // block_size) * block_size
    n_x_clipped = (n_x // block_size) * block_size

    data_clipped = data[:n_y_clipped, :n_x_clipped]

    # 2) Reshape so we can do block-wise means:
    #    - We want to turn (n_y_clipped, n_x_clipped) into
    #      (n_y_clipped/block_size, block_size, n_x_clipped/block_size, block_size)
    n_y_new = n_y_clipped // block_size
    n_x_new = n_x_clipped // block_size

    data_reshaped = data_clipped.reshape(
        n_y_new, block_size, n_x_new, block_size
    )

    # 3) Compute mean over the block axes:
    data_coarse = data_reshaped.mean(axis=(1, 3))

    # 4) Define the new coordinate arrays by taking the middle coordinate in each block.
    #    The middle coordinate index for a block of size block_size is offset = block_size // 2
    offset = block_size // 2

    # For rows (y direction), pick y[offset + i * block_size]
    y_coarse_indices = offset + arange(n_y_new) * block_size
    y_coarse = y[y_coarse_indices]

    # For columns (x direction), similarly
    x_coarse_indices = offset + arange(n_x_new) * block_size
    x_coarse = x[x_coarse_indices]

    return data_coarse, x_coarse, y_coarse

if __name__ == '__main__':
    # Open Data
    latitude, longitude, BTs = get_data(sceps_3_truth_scene, INCIDENCE_ANGLE)

    # Get target
    # Use any config file, just needs it for initiation
    config_object = ConfigFile(config_path)
    target_lons, target_lats = GridGenerator(config_object, 'G', 'EASE2_G1km').generate_grid_lonlat()

    # Put the 1km truth onto an EASE grid.
    EASE_BTs = fit_to_ease(
        source_lons=longitude,
        source_lats=latitude,
        target_lons=target_lons,
        target_lats=target_lats,
        BTs=BTs,
        bands=bands,
        search_radius=1000
    )

    # Downscale Data
    # Generate the 1km EASE x, y coordinates
    x_1km, y_1km = GridGenerator(config_object, 'G', 'EASE2_G1km').generate_grid_xy()
    for resolution in output_grid_resolutions:
        output_dict = {}
        for band in EASE_BTs:
            image, x, y = resample_ease_grids(EASE_BTs[band], x_1km, y_1km, resolution)
            image_out  = {}
            image_out['bt_h'] = image
            image_out['x'] = x
            image_out['y'] = y
            output_dict[band] = image_out

        # Output folder
        output_file_name = f"{output_folder}/{output_file_tag}_EASE2_G{resolution}km.pkl"

        # use pickle to save dictionary
        with open(output_file_name, 'wb') as f:
            pickle.dump(output_dict, f)

        # Finally also save the 1km grids
        output_dict_1km = {}
        for band in EASE_BTs:
            image_out = {}
            image_out['bt'] = EASE_BTs[band]
            image_out['x'] = x_1km
            image_out['y'] = output_dict_1km
            output_dict_1km[band] = image_out
        output_file_name = f"{output_folder}/{output_file_tag}_EASE2_G{1}km.pkl"
        with open(output_file_name, 'wb') as f:
            pickle.dump(output_dict_1km, f)
