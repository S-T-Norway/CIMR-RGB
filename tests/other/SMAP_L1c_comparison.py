# This script is to compare the SMAP L1C RGB remap, with the NASA remap for the equivalent granule.

import os
import h5py
from numpy import array, full, nan, nanmax, nanmean, nanmin, where, arange
import pickle

import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

root_path = os.path.join(os.getcwd(), '..')

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
        self.RGB_dir = os.path.join(root_path, 'dpr/L1C/SMAP/RGB')
        self.NASA_dir = os.path.join(root_path, 'dpr/L1C/SMAP/NASA/')

    def check_existence(self):
        pass
    def get_RGB_data(self, rgb_granule_name):
        rgb_data = os.path.join(self.RGB_dir, rgb_granule_name + '.pkl')
        # Open pickled dictionary
        with open(rgb_data, 'rb') as f:
            rgb_data = pickle.load(f)
        return rgb_data
    def get_NASA_data(self, rgb_granule_name):
        if "9km" in rgb_granule_name:
            self.NASA_dir = self.NASA_dir + 'Enhanced'
            for file in os.listdir(self.NASA_dir):
                if rgb_granule_name[12:28] in file:
                    nasa_granule_name = file
                    grid = GRIDS[f"{rgb_granule_name[46:]}"]
        else:
            nasa_granule_name = rgb_granule_name[0:46] + '.h5'
            grid = GRIDS[f"{rgb_granule_name[47:]}"]
        nasa_dict = {}

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
        im1 = ax1.imshow(rgb[variable][:, :], aspect='auto', cmap = cmap, interpolation='nearest')
        ax1.set_title('RGB Remap', fontsize = 14)

        im2 = ax2.imshow(nasa[variable][:, :], aspect='auto', cmap = cmap)
        ax2.set_title('NASA Remap', fontsize = 14)

        im3 = ax3.imshow(diff[:, :], aspect='auto',cmap=cmap, interpolation = 'nearest')
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
rgb_granule_name = 'SMAP_L1C_TB_47185_D_20231201T212120_R18290_001EASE2_G9km'
# rgb_granule_name = 'SMAP_L1C_TB_47185_D_20231201T212120_R18290_001EASE2_G9km'


# Make the plot
SMAPComparison(root_path).compare_single_granule(rgb_granule_name)


## Old testing code from somewhere else
# sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src')
# from grid_generator import GridGenerator
#
# # NASA Level 1c file
# enhanced_granule_path= '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/L1C/SMAP/NASA_BG/SMAP_L1C_TB_E_47185_D_20231201T212059_R19240_002.h5'
# # Open hdf file using with
# with h5py.File(enhanced_granule_path, 'r') as f:
#     # Get the dataset
#     data = f['Global_Projection']
#     tb_v = data['cell_tb_v_fore'][:]
#     col = data['cell_column'][:]
#     row = data['cell_row'][:]
#     cell_number_measurements_v_fore=data['cell_number_measurements_v_fore'][:]
#     cell_number_measurements_v_aft=data['cell_number_measurements_v_aft'][:]
#     cell_number_measurements_h_fore = data['cell_number_measurements_h_fore'][:]
#     cell_number_measurements_h_aft = data['cell_number_measurements_h_aft'][:]
#
#
# ease_shape = (1624,3856)
#
# grid_fore_frequency_nasa = np.full(ease_shape, np.nan)
# grid_bt_v_nasa = np.full(ease_shape, np.nan)
# for count, i in enumerate(tb_v):
#     # print(count)
#     grid_fore_frequency_nasa[row[count], col[count]] = cell_number_measurements_v_fore[count]
#     grid_bt_v_nasa[row[count], col[count]] = i
#
# flat_grid = grid_bt_v_nasa.flatten()
#
#
# # # Open results dictionary
# # with open('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/BG_temp_folder/tuesday_cimr_first_attempt', 'rb') as file:
# #     bg_dict = pickle.load(file)
# #
# # bg_dict_fore = bg_dict['fore']
# # grid_fore_frequency_nasa_st = np.full(ease_shape, np.nan)
# # grid_bt_v_nasa_st = np.full(ease_shape, np.nan)
# # for count, i in enumerate(bg_dict_fore):
# #     row, col = np.unravel_index(i, ease_shape)
# #     grid_bt_v_nasa_st[row, col] = bg_dict_fore[i]
# #     # grid_fore_frequency_nasa_st[row, col] = fore_frequency_dict[i]
#
# # phillipines_inds = [830:850, 3200:3220]
# # desert_inds =[1125:1145, 3180:3200]
# # ocean_inds = [520:540, 3290:3310]
# # island_7000 = [800:900, 3190:3260]
# # north_pole_large = [200:300, 3310, 3380]
#
# # fig, axs = plt.subplots(1, 3)
# # axs[0].imshow(grid_bt_v_nasa[520:540, 3290:3310], cmap='viridis')
# # axs[0].set_title('NASA')
# # axs[1].imshow(grid_bt_v_nasa_st[520:540, 3290:3310], cmap='viridis')
# # axs[1].set_title('ST')
# # diff = (grid_bt_v_nasa - grid_bt_v_nasa_st)
# # axs[2].imshow(diff[520:540, 3290:3310], cmap='viridis')
# # axs[2].set_title('Diff')
# # # fig.suptitle('bt_v')
# # plt.tight_layout()
#
# def comp_plot():
#     # Create subplots with GridSpec for precise control
#     fig = plt.figure(figsize=(21, 7))
#     gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
#
#     # Create subplots
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
#     ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
#     cax = fig.add_subplot(gs[0, 3])
#
#     # Plot data
#     cmap='viridis'
#     im1 = ax1.imshow(grid_bt_v_nasa[1125:1145, 3180:3200], aspect='auto', cmap = cmap, interpolation='nearest')
#     ax1.set_title('RGB Remap', fontsize = 19)
#
#     im2 = ax2.imshow(grid_bt_v_nasa_st[1125:1145, 3180:3200], aspect='auto', cmap = cmap)
#     ax2.set_title('NASA Remap', fontsize = 19)
#
#     im3 = ax3.imshow(diff[1125:1145, 3180:3200], aspect='auto',cmap=cmap, interpolation = 'nearest')
#     ax3.set_title('Difference', fontsize = 19)
#
#     # Add color bar to the dedicated axis
#     cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
#     cbar.set_label('Difference [K]', fontsize = 19)
#
#     # Remove y-axis labels from the second and third subplots
#     ax2.tick_params(left=False, labelleft=False)
#     ax3.tick_params(left=False, labelleft=False)
#
#     # Labels
#     ax1.set_ylabel('EASE2 Grid Rows [-]', fontsize = 19)
#     ax2.set_xlabel('EASE2 Grid Columns [-]', fontsize = 19)
#
#     plt.tight_layout()
#     plt.show()
#
# # comp_plot()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Plot a colour bar on axs[2]
# # fig.colorbar(axs[2].imshow(diff[830:850, 3200:3220], cmap='viridis'), ax=axs[2])
#
# # fig, axs = plt.subplots(1, 3)
# # axs[0].imshow(grid_fore_frequency_nasa[830:850, 3200:3220], cmap='viridis')
# # axs[0].set_title('NASA')
# # axs[1].imshow(grid_fore_frequency_nasa_st[830:850, 3200:3220], cmap='viridis')
# # axs[1].set_title('ST')
# # diff = abs(grid_fore_frequency_nasa - grid_fore_frequency_nasa_st)
# # axs[2].imshow(diff[830:850, 3200:3220], cmap='viridis')
# # axs[2].set_title('Diff')
# # # fig.suptitle('Number of measurements')
# # plt.tight_layout()
#
#
# # fig, axs = plt.subplots()
# # axs.imshow(grid_bt_v_nasa, cmap='viridis')
# # axs.set_title('NASA Enhanced (BG) L1c', fontsize=14)
# # axs.set_ylabel('EASE2 Global 9km Grid Rows [-]', fontsize=14)
# # axs.set_xlabel('EASE2 Global 9km Grid Columns [-]', fontsize=14)
#
# # Open results dictionary
# # with open('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/BG_temp_folder/tuesday_cimr_first_attempt', 'rb') as file:
# #     bg_dict = pickle.load(file)
# #
# # bg_dict_fore = bg_dict['fore']
# # grid = np.full(ease_shape, np.nan)
# # for count, i in enumerate(bg_dict_fore):
# #     row, col = np.unravel_index(i, ease_shape)
# #     grid[row, col] = bg_dict_fore[i]
#
# smap_file = '/CIMR-RGB/output/Results/IDS/SMAP/SMAP_9km_no_radius.pkl'
# with open(smap_file, 'rb') as file:
#     smap_bg_dict = pickle.load(file)
# bt_h_target_fore = smap_bg_dict['bt_h_target_fore']
# bt_h_target_fore[np.isnan(bt_h_target_fore)] = 0
# plt.imshow(bt_h_target_fore[:,2500:])
#
# # grid_smap = np.full(ease_shape, np.nan)
# # for count, i in enumerate(smap_bg_dict):
# #     row, col = np.unravel_index(i, ease_shape)
# #
# #     grid_smap[row, col] = smap_bg_dict[i]
