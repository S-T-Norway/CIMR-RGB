import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import pickle
import numpy as np
GRIDS = {
         'EASE2_G3km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                        'res': 3002.69, 'n_cols': 11568, 'n_rows': 4872, 'lat_min': -86, 'lat_max': 86},
         'EASE2_G9km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                        'res': 9008.05, 'n_cols': 3856, 'n_rows': 1624, 'lat_min': -86, 'lat_max': 86},
         'EASE2_G36km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                         'res': 36032.22, 'n_cols': 964, 'n_rows': 406, 'lat_min': -86, 'lat_max': 86}
         }

rsir_path = '/home/beywood/Desktop/MS4/optimal_params_data_dicts/rsir_l_36km_optimal_params.pkl'
bg_path = '/home/beywood/Desktop/MS4/optimal_params_data_dicts/bg_l_36km_optimal_params.pkl'
ids_path =  '/home/beywood/Desktop/MS4/optimal_params_data_dicts/ids_l_36km_optimal_params.pkl'

# Open dictionaries with pickle
# ----- Loading (Unpickling) -----
with open(rsir_path, 'rb') as f:
    rsir_dict = pickle.load(f)

with open(ids_path, 'rb') as f:
    ids_dict = pickle.load(f)

with open(bg_path, 'rb') as f:
    bg_dict = pickle.load(f)


# Put the data on an ease grid
def fit_to_ease(bt_h, cell_row, cell_col, ease_grid):
    output_grid = np.full((ease_grid['n_rows'], ease_grid['n_cols']), np.nan)
    for count, sample in enumerate(bt_h):
        row = cell_row[count]
        col = cell_col[count]
        output_grid[row, col] = bt_h[count]
    return output_grid


ids_grid = fit_to_ease(ids_dict['L']['bt_h'], ids_dict['L']['cell_row'], ids_dict['L']['cell_col'], GRIDS['EASE2_G36km'])
rsir_grid = fit_to_ease(rsir_dict['L']['bt_h'], rsir_dict['L']['cell_row'], rsir_dict['L']['cell_col'], GRIDS['EASE2_G36km'])
bg_grid = fit_to_ease(bg_dict['L']['bt_h'], bg_dict['L']['cell_row'], bg_dict['L']['cell_col'], GRIDS['EASE2_G36km'])

row_slice = slice(120, 168)
col_slice = slice(211, 250)

vmin = min(ids_grid[row_slice, col_slice].min(), rsir_grid[row_slice, col_slice].min(), bg_grid[row_slice, col_slice].min())
vmax = max(ids_grid[row_slice, col_slice].max(), rsir_grid[row_slice, col_slice].max(), bg_grid[row_slice, col_slice].max())

fig, axs = plt.subplots(1, 3, constrained_layout=True)  # Adjust figure size
im0=axs[0].imshow(ids_grid[row_slice, col_slice], cmap='viridis', extent=(col_slice.start, col_slice.stop, row_slice.stop, row_slice.start),vmin=vmin, vmax=vmax)
axs[0].set_title('IDS', fontsize = 16)
axs[0].set_ylabel('EASE Row [-]', fontsize = 16)
# colorbar


im1=axs[1].imshow(rsir_grid[row_slice, col_slice], cmap='viridis', extent=(col_slice.start, col_slice.stop, row_slice.stop, row_slice.start), vmin=vmin, vmax=vmax)
axs[1].set_title('RSIR', fontsize = 16)
# xlabel
axs[1].set_xlabel('EASE Column [-]', fontsize = 16)
# colorbar


im2 = axs[2].imshow(bg_grid[row_slice, col_slice], cmap='viridis', extent=(col_slice.start, col_slice.stop, row_slice.stop, row_slice.start), vmin=vmin, vmax=vmax)
axs[2].set_title('BG', fontsize = 16)
# colorbar


cbar = fig.colorbar(im0, ax=axs, orientation='vertical', label='BT_H [K]')
# Set tick-label font size

# Set the colorbar's label font size
cbar.set_label('BT_H [K]', fontsize=16)


plt.suptitle('Optimal Regrid Comparison L_BAND BT_H (EASE2_G36km)', fontsize=16)
