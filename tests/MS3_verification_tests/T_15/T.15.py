# Testing script for T.15
# Remapping of L1b SMAP data with rSIR algorithm on a Stereographic North polar grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)

import sys
import os

import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb')
from grid_generator import GRIDS
from numpy import array, full, nan, nanmean, polyfit, isnan, isinf


# Add RGB remapped IDS netCDF here
ids_data_path = ''
# Add the RGB remapped RSIR netCDF here
rsir_data_path = ''

GRID = 'STEREO_N25km'
PROJECTION = 'PS_N'

class SMAP_comparison:

    def __init__(self, ids_data_path, rsir_data_path):
        self.ids_data = self.get_netcdf_data(ids_data_path)
        self.rsir_data = self.get_netcdf_data(rsir_data_path)

    @staticmethod
    def get_netcdf_data(path):
        import netCDF4 as nc
        gridded_vars = {}
        with nc.Dataset(path, 'r') as f:
            data = f[f"{PROJECTION}"]
            measurement = data['Measurement']
            l_band = measurement['L_BAND']

            for bt in ['bt_h_fore', 'bt_h_aft']:
                if 'fore' in bt:
                    row = array(l_band['cell_row_fore'][:])
                    col = array(l_band['cell_col_fore'][:])
                elif 'aft' in bt:
                    row = array(l_band['cell_row_aft'][:])
                    col = array(l_band['cell_col_aft'][:])
                else:
                    row = ''
                    col = ''
                var = array(l_band[bt][:])
                grid = full((GRIDS[GRID]['n_rows'], GRIDS[GRID]['n_cols']), nan)

                for count, sample in enumerate(var):
                    if sample == 9.969209968386869e36:
                        continue
                    if sample == 0.:
                        continue
                    grid[row[count], col[count]] = sample
                gridded_vars[bt] = grid
            return gridded_vars

    def map_compare(self):

        cmap = 'viridis'
        # bt_h plt
        fig, axs = plt.subplots(2, 3, constrained_layout=True)
        im00 = axs[0,0].imshow(self.rsir_data['bt_h_fore'][:,:], cmap=cmap)
        axs[0,0].set_title('RSIR Remap (bt_h_fore)')
        im01 = axs[0,1].imshow(self.ids_data['bt_h_fore'][:,:], cmap=cmap)
        axs[0,1].set_title('IDS Remap (bt_h_fore)')
        bt_h_fore_diff = abs(self.rsir_data['bt_h_fore'] - self.ids_data['bt_h_fore'])
        im02 = axs[0,2].imshow(bt_h_fore_diff[:,:], cmap=cmap)
        axs[0,2].set_title('Difference (bt_h_fore)')
        # aft
        im10 = axs[1,0].imshow(self.rsir_data['bt_h_aft'][:,:], cmap=cmap)
        axs[1,0].set_title('RSIR Remap (bt_h_aft)')
        im11 = axs[1,1].imshow(self.ids_data['bt_h_aft'][:,:], cmap=cmap)
        axs[1,1].set_title('IDS Remap (bt_h_aft)')
        bt_h_aft_diff = abs(self.rsir_data['bt_h_aft'] - self.ids_data['bt_h_aft'])
        im12 = axs[1,2].imshow(bt_h_aft_diff[:,:], cmap=cmap)
        axs[1,2].set_title('Difference (bt_h_aft)')
        fig.colorbar(im02, ax=axs[0])
        fig.colorbar(im12, ax=axs[1])

        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_h_fore_diff)
        aft_mean_diff = nanmean(bt_h_aft_diff)
        print(f"Average relative difference for bt_h_fore: {fore_mean_diff}")
        print(f"Average relative difference for bt_h_aft: {aft_mean_diff}")

        # Calculate percentage Differences
        fore_percent_diff = (fore_mean_diff / nanmean(self.ids_data['bt_h_fore'])) * 100
        aft_percent_diff = (aft_mean_diff / nanmean(self.ids_data['bt_h_aft'])) * 100
        print(f"Average percentage difference for bt_h_fore: {fore_percent_diff}")
        print(f"Average percentage difference for bt_h_aft: {aft_percent_diff}")

        # Add statistics to the plot
        # axs[0,2].text(100,300, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
        # axs[0,2].text(100, 300, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
        axs[0,2].text(50, 300,
                fr"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{RSIR}}_i |$",
                fontsize=14, color="black")

        axs[0,2].text(50, 350,
                fr"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
                fontsize=14, color="black")

        axs[1,2].text(50, 300,
                       fr"$\mu_{{aft}} =  {aft_mean_diff:.2f} K, \ \text{{or}} \ {aft_percent_diff:.2f}\%$",
                       fontsize=14, color="black")
        plt.show()

    @staticmethod
    def scatter_stats(x, y):
        mask = ~isnan(x) & ~isnan(y) & ~isinf(x) & ~isinf(y)
        x = x[mask]
        y = y[mask]
        m, b = polyfit(x, y, 1)
        y_fit = m * x + b

        # Calculate R^2
        ss_res = sum((y - y_fit) ** 2)
        ss_tot = sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return x, y, m, b, y_fit, r_squared


    def scatter_compare(self):
        x = self.rsir_data['bt_h_fore'].flatten()
        y = self.ids_data['bt_h_fore'].flatten()
        x_h_fore, y_h_fore, m_h_fore, b_h_fore, y_fit_h_fore, r_squared = self.scatter_stats(x, y)

        x = self.rsir_data['bt_h_aft'].flatten()
        y = self.ids_data['bt_h_aft'].flatten()
        x_h_aft, y_h_aft, m_h_aft, b_h_aft, y_fit_h_aft, r_squared = self.scatter_stats(x, y)


        fig, axs = plt.subplots(1,2)
        axs[0].scatter(x_h_fore, y_h_fore)
        axs[0].plot(x_h_fore, y_fit_h_fore, color='red')
        axs[0].legend(title=f"$R^2 = {r_squared:.3f}$")
        axs[0].set_title('bt_h_fore')
        axs[0].set_xlabel('RSIR BT [K]')
        axs[0].set_ylabel('IDS BT [K]')

        axs[1].scatter(x_h_aft, y_h_aft)
        axs[1].plot(x_h_aft, y_fit_h_aft, color='red')
        axs[1].legend(title=f"$R^2 = {r_squared:.3f}$")
        axs[1].set_title('bt_h_aft')
        axs[1].set_xlabel('RSIR BT [K]')
        axs[1].set_ylabel('IDS BT [K]')

        plt.show()

SMAP_comparison(ids_data_path, rsir_data_path).map_compare()
SMAP_comparison(ids_data_path, rsir_data_path).scatter_compare()