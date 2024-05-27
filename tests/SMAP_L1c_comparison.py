# This script is to compare the SMAP L1c RGB remap, with the NASA remap for the equivalent granule.

import os
import h5py
from numpy import array, full, nan, nanmax, nanmean, nanmin, where, arange
import pickle

import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

GRIDS = {'EASE2_G9km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                        'res': 9008.05, 'n_cols': 3856, 'n_rows': 1624},
         'EASE2_N9km': {'epsg': 6931, 'x_min': -9000000.0, 'y_max': 9000000.0,
                        'res': 9000.0, 'n_cols': 2000, 'n_rows': 2000},
         'EASE2_S9km': {'epsg': 6932, 'x_min': -9000000.0, 'y_max': 9000000.0,
                        'res': 9000.0, 'n_cols': 2000, 'n_rows': 2000},
         'EASE2_G36km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                         'res': 36032.22, 'n_cols': 964, 'n_rows': 406},
         'EASE2_N36km': {'epsg': 6931, 'x_min': -9000000.0, 'y_max': 9000000.0,
                         'res': 36000.0, 'n_cols': 500, 'n_rows': 500},
         'EASE2_S36km': {'epsg': 6932, 'x_min': -9000000.0, 'y_max': 9000000.0,
                         'res': 36000.0, 'n_cols': 500, 'n_rows': 500}
         }
class SMAPComparison:
    def __init__(self, root_path):
        self.root_path = root_path
        self.RGB_dir = os.path.join(root_path, 'dpr/L1cProducts/SMAP/RGB')
        self.NASA_dir = os.path.join(root_path, 'dpr/L1cProducts/SMAP/NASA/')

    def check_existence(self):
        pass
    def get_RGB_data(self, rgb_granule_name):
        rgb_data = os.path.join(self.RGB_dir, rgb_granule_name + '.pkl')
        # Open pickled dictionary
        with open(rgb_data, 'rb') as f:
            rgb_data = pickle.load(f)
        return rgb_data
    def get_NASA_data(self, rgb_granule_name):
        nasa_granule_name = rgb_granule_name[0:46] + '.h5'
        nasa_dict = {}
        grid = GRIDS[f"{rgb_granule_name[47:]}"]
        if 'G' in rgb_granule_name:
            projection = 'Global_Projection'
        elif 'N' in rgb_granule_name:
            projection = 'North_Polar_Projection'
        elif 'S' in rgb_granule_name:
            projection = 'South_Polar_Projection'
        # Open hdf5 file
        with h5py.File(os.path.join(self.NASA_dir, nasa_granule_name), 'r') as f:
            # Get the data
            data = f[projection]
            nasa_dict['bt_h_target_fore'] = array(data['cell_tb_h_fore'])
            nasa_dict['bt_h_target_aft'] = array(data['cell_tb_h_aft'])
            nasa_dict['cell_row'] = array(data['cell_row'])
            nasa_dict['cell_col'] = array(data['cell_column'])
            nasa_dict['cell_frequency_fore'] = array(data['cell_number_measurements_h_fore'])
            nasa_dict['cell_frequency_aft'] = array(data['cell_number_measurements_h_aft'])

        rows, cols = grid['n_rows'], grid['n_cols']
        nasa_grid_dict = {}
        for variable in nasa_dict:
            temp_grid = full((rows, cols), nan)
            for count, measurement in enumerate(nasa_dict[variable]):
                temp_grid[nasa_dict['cell_row'][count], nasa_dict['cell_col'][count]] = measurement
            nasa_grid_dict[variable] = temp_grid

        return nasa_grid_dict

    def compare_single_granule(self, rgb_granule_name):
        # Get the RGB data
        nasa = self.get_NASA_data(rgb_granule_name=rgb_granule_name)
        rgb = self.get_RGB_data(rgb_granule_name)

        variable = 'bt_h_target_fore'  # Assuming this is just an example variable name

        # Calculate the difference
        # diff = (nasa[variable] - rgb[variable])
        diff = abs(nasa[variable] - rgb[variable])  # For example purposes
        print(f"Max Diff = {nanmax(diff)}")
        print(f"Mean Diff = {nanmean(diff)}")
        print(where(diff == nanmax(diff)))
        print(f"RGB at max diff = {rgb[variable][where(diff == nanmax(diff))]}")
        print(f"NASA at max diff = {nasa[variable][where(diff == nanmax(diff))]}")



        vmin, vmax = nanmin(diff), nanmax(diff)
        # Create subplots with GridSpec for precise control
        fig = plt.figure(figsize=(21, 7))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0)

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
        cax = fig.add_subplot(gs[0, 3])

        # Plot data
        cmap='plasma'
        im1 = ax1.imshow(rgb[variable][:, 0:450], aspect='auto', cmap = cmap)
        ax1.set_title('RGB Remap', fontsize = 14)

        im2 = ax2.imshow(nasa[variable][:, 0:450], aspect='auto', cmap = cmap)
        ax2.set_title('NASA Remap', fontsize = 14)

        im3 = ax3.imshow(diff[:, 0:450], aspect='auto',cmap=cmap)
        ax3.set_title('Difference', fontsize = 14)

        # Add color bar to the dedicated axis
        cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
        cbar.set_label('Difference [K]', fontsize = 14)

        # Remove y-axis labels from the second and third subplots
        ax2.tick_params(left=False, labelleft=False)
        ax3.tick_params(left=False, labelleft=False)

        # Labels
        ax1.set_ylabel('EASE2 Grid Rows [-]', fontsize = 14)
        ax2.set_xlabel('EASE2 Grid Columns [-]', fontsize = 14)



        plt.tight_layout()
        plt.show()

        fig.savefig('/home/beywood/ST/CIMR_RGB/WP200/SMAP_L1C_TB_47224_D_20231204T132117_R18290_001_frequency.png', dpi=300, bbox_inches='tight')


        # plt.figure()
        # plt.imshow(diff)

    def compare_all_granules(self):
        pass


# The name of the file you want to compare
root_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB'
# rgb_granule_name = 'SMAP_L1C_TB_47172_D_20231201T000114_R18290_001_EASE2_G36km'
rgb_granule_name = 'SMAP_L1C_TB_47224_D_20231204T132117_R18290_001_EASE2_G36km'

SMAPComparison(root_path).compare_single_granule(rgb_granule_name)

