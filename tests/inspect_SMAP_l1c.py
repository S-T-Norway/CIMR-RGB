import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))
import h5py
import numpy as np
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from grid_generator import GRIDS
plt.ion()


def plot_to_grid(var, row, col, grid):
    grid_shape = grid['n_rows'], grid['n_cols']
    # # create nan array with shape of grid_shape
    grid = np.full(grid_shape, np.nan)
    for i in range(len(row)):
        grid[row[i], col[i]] = var[i]
    return grid

variable_key_map = {
                'bt_h': 'tb_h',
                'bt_v': 'tb_v',
                'bt_3': 'tb_3',
                'bt_4': 'tb_4',
                'processing_scan_angle': 'antenna_scan_angle',
                'longitude': 'tb_lon',
                'latitude': 'tb_lat',
                'x_position': 'x_pos',
                'y_position': 'y_pos',
                'z_position': 'z_pos',
                'x_velocity': 'x_vel',
                'y_velocity': 'y_vel',
                'z_velocity': 'z_vel',
                'sub_satellite_lon': 'sc_nadir_lon',
                'sub_satellite_lat': 'sc_nadir_lat',
                'altitude': 'sc_geodetic_alt_ellipsoid'
            }


class compare_smap_l1c:

    def __init__(self, config, l1c_path):
        self.config = config
        self.l1c_path = l1c_path
        if self.config.projection_definition == 'G':
            self.projection = 'Global_Projection'
        elif self.config.projection_definition == 'N':
            self.projection = 'North_Polar_Projection'
        elif self.config.projection_definition == 'S':
            self.projection = 'South_Polar_Projection'

    def get_l1c_data(self, l1c_path, variable):
        with h5py.File(l1c_path, 'r') as f:
            data = f[self.projection]
            var = data[variable][:]
            col = data['cell_column'][:]
            row = data['cell_row'][:]
        return var, row, col

    def plot_diff(self, data_dict, variable):
        data_dict_rgb = data_dict['L']
        if 'bt' in variable:
            # SMAP naming convention is tb
            variable_smap = variable.replace('bt', 'cell_tb')
        if 'scan_angle' in variable:
            print(variable)
            variable_smap = variable.replace('processing', 'cell_antenna')
            print(variable_smap)

        var_smap, row_smap, col_smap = self.get_l1c_data(self.l1c_path, variable_smap)
        var_rgb = data_dict_rgb[variable]
        if 'fore' in variable:
            row_rgb = data_dict_rgb['cell_row_fore']
            col_rgb = data_dict_rgb['cell_col_fore']
        elif 'aft' in variable:
            row_rgb = data_dict_rgb['cell_row_aft']
            col_rgb = data_dict_rgb['cell_col_aft']

        grid_smap = plot_to_grid(var_smap, row_smap, col_smap, GRIDS[self.config.grid_definition])
        grid_rgb = plot_to_grid(var_rgb, row_rgb, col_rgb, GRIDS[self.config.grid_definition])

        diff = abs(grid_smap - grid_rgb)

        if self.config.reduced_grid_inds:
            row_min = self.config.reduced_grid_inds[0]
            row_max = self.config.reduced_grid_inds[1]
            col_min = self.config.reduced_grid_inds[2]
            col_max = self.config.reduced_grid_inds[3]
        else:
            row_min, row_max, col_min, col_max = 0, GRIDS[self.config.grid_definition]['n_rows'], 0, GRIDS[self.config.grid_definition]['n_cols']

        # Create a subplot with 3 plots and a colour bar for the third plot
        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        cmap = 'viridis'
        im1 = axs[0].imshow(grid_rgb[row_min:row_max,col_min:col_max], cmap=cmap)
        axs[0].set_title('RGB Remap')
        im2 = axs[1].imshow(grid_smap[row_min:row_max,col_min:col_max], cmap=cmap)
        axs[1].set_title('NASA Remap')
        im3 = axs[2].imshow(diff[row_min:row_max,col_min:col_max], cmap=cmap)
        axs[2].set_title('Difference')
        fig.colorbar(im3, ax=axs[2])
        import time
        plt.show()
        time.sleep(5)
        print(matplotlib.get_backend())
        print(f"average_error = {np.nanmean(abs(diff))}")
        ED = (diff / grid_smap) * 100
        print(f"average_percentage_error = {np.nanmean(ED)}")
        plt.figure()
        plt.imshow(diff)
        plt.show()


        # Use the plot below to make a "report" plot
        # vmin, vmax = np.nanmin(diff), np.nanmax(diff)
        # # Create subplots with GridSpec for precise control
        # fig = plt.figure(figsize=(21, 7))
        # gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0)
        #
        # # Create subplots
        # ax1 = fig.add_subplot(gs[0, 0])
        # ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        # ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
        # cax = fig.add_subplot(gs[0, 3])
        #
        # # Plot data
        # cmap = 'plasma'
        # im1 = ax1.imshow(grid_rgb, aspect='auto', cmap=cmap, interpolation='nearest')
        # ax1.set_title('RGB Remap', fontsize=14)
        #
        # im2 = ax2.imshow(grid_smap, aspect='auto', cmap=cmap)
        # ax2.set_title('NASA Remap', fontsize=14)
        #
        # im3 = ax3.imshow(diff, aspect='auto', cmap=cmap, interpolation='nearest')
        # ax3.set_title('Difference', fontsize=14)
        #
        # # Add color bar to the dedicated axis
        # cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
        # cbar.set_label('Difference [K]', fontsize=14)
        #
        # # Remove y-axis labels from the second and third subplots
        # ax2.tick_params(left=False, labelleft=False)
        # ax3.tick_params(left=False, labelleft=False)
        #
        # # Labels
        # ax1.set_ylabel('EASE2 Grid Rows [-]', fontsize=14)
        # ax2.set_xlabel('EASE2 Grid Columns [-]', fontsize=14)
        #
        # plt.tight_layout()
        # plt.show()

    def plot_scatter(self):
        pass

