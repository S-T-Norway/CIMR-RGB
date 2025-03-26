import sys
from cimr_rgb.config_file import ConfigFile
from cimr_rgb.data_ingestion import DataIngestion
from cimr_rgb.regridder import ReGridder
from cimr_rgb.grid_generator import GRIDS
import numpy as np


import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

config_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/quick_config.xml'
config_object = ConfigFile(config_file)
data_dict = DataIngestion(config_object).ingest_data()

data_dict_out = ReGridder(config_object).regrid_data(data_dict)


# Rebuild output grid
grid_dims = (GRIDS[config_object.grid_definition]['n_rows'],
             GRIDS[config_object.grid_definition]['n_cols'])
target_band = config_object.target_band[0]
output_grid = np.full(grid_dims, np.nan)



variable_to_plot = 'bt_h'
if config_object.split_fore_aft == 'True':
    for count, sample in enumerate(data_dict_out[target_band][f'{variable_to_plot}_fore']):
        row = data_dict_out[target_band]['cell_row_fore'][count]
        col = data_dict_out[target_band]['cell_col_fore'][count]
        output_grid[row, col] = data_dict_out[target_band][f'{variable_to_plot}_fore'][count]
else:
    for count, sample in enumerate(data_dict_out[target_band][f'{variable_to_plot}']):
        row = data_dict_out[target_band]['cell_row'][count]
        col = data_dict_out[target_band]['cell_col'][count]
        output_grid[row, col] = data_dict_out[target_band][f'{variable_to_plot}'][count]


# plt.imshow(output_grid, cmap='viridis')


# Open file in npz format
# km_36 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_central_america/SCEPS_central_america_EASE2_G36km.npz'
# km_9 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_central_america/SCEPS_central_america_EASE2_G9km.npz'
# km_3 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_central_america/SCEPS_central_america_EASE2_G3km.npz'

km_36 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_test_scene_3/SCEPS_test_scene_3_EASE2_G36km.npz'
km_9 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_test_scene_3/SCEPS_test_scene_3_EASE2_G9km.npz'
km_3 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards/SCEPS_test_scene_3/SCEPS_test_scene_3_EASE2_G3km.npz'

data_36 = np.load(km_36)
data_9 = np.load(km_9)
data_3 = np.load(km_3)

# # Define the slice indices 119 167 210 249
row_slice = slice(328, 483)
col_slice = slice(1852, 2007)

# Determine common color limits for the first two images
data1 = output_grid[row_slice, col_slice]
BAND = 'L_BAND'
data2 = data_9[BAND][row_slice, col_slice]
# If there is a negative value in data 2, turn it into nan
# if band == 'KA_BAND':
# data2[data2 < 0] = np.nan
# data1[data2 < 0] = np.nan



vmin = min(np.nanmin(data1), np.nanmin(data2))
vmax = max(np.nanmax(data1), np.nanmax(data2))

# Compute the difference image
diff = data1 - data2

# Create the subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot the first image with the common scale
im0 = axs[0].imshow(data1, vmin=vmin, vmax=vmax, extent=(col_slice.start, col_slice.stop, row_slice.stop, row_slice.start))
axs[0].set_ylabel("EASE Row [-]", fontsize=14)  # Add y-axis label for the first plot
axs[0].set_title("RGB", fontsize=14)
cbar0 = fig.colorbar(im0, ax=axs[0])
cbar0.ax.set_title("BT [K]")

# Plot the second image with the common scale
im1 = axs[1].imshow(data2, vmin=vmin, vmax=vmax, extent=(col_slice.start, col_slice.stop, row_slice.stop, row_slice.start))
axs[1].set_xlabel("EASE Column [-]", fontsize = 14)  # Add x-axis label for the second plot
axs[1].set_title("Reference Scene", fontsize=14)
cbar1 = fig.colorbar(im1, ax=axs[1])
cbar1.ax.set_title("BT [K]")

# Plot the difference image (using its own scale)
im2 = axs[2].imshow(diff, extent=(col_slice.start, col_slice.stop, row_slice.stop, row_slice.start))
axs[2].set_title("Difference", fontsize=14)
cbar2 = fig.colorbar(im2, ax=axs[2])
cbar2.ax.set_title("Error [K]")

# Add an overall title for the figure
fig.suptitle("RSIR Optimal Parameters reduced grid L Band EASE2 9km", fontsize=16)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
