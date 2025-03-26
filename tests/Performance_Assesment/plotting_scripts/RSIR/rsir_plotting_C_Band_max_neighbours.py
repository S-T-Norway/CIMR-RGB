import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

results_max_neighbours = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/RSIR/RSIR_C_9km_max_neighbours.csv'

# Read CSV File
df_max_neighbours = pd.read_csv(results_max_neighbours)
df_max_neighbours.head()
BAND = 'C'
band = df_max_neighbours[df_max_neighbours['band'] == BAND]

df_iter_1 = band[band['rsir_iteration'] == 1]
df_iter_25 = band[band['rsir_iteration'] == 25]
df_iter_50 = band[band['rsir_iteration'] == 50]
df_mrf_1km = band[band['MRF_grid_resolution'] == '1km']


df_iter_1 = df_iter_1.sort_values(by='max_neighbours', ascending=True)
df_iter_25 = df_iter_25.sort_values(by='max_neighbours', ascending=True)
df_iter_50 = df_iter_50.sort_values(by='max_neighbours', ascending=True)
df_mrf_1km = df_mrf_1km.sort_values(by='max_neighbours', ascending=True)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

# Primary y-axis: Plot mean absolute error
ax1.plot(df_iter_1['max_neighbours'], df_iter_1['mean_absolute_error'], label = 'rsir_iteration=1, MRF=3km')
ax1.plot(df_iter_25['max_neighbours'], df_iter_25['mean_absolute_error'], label = 'rsir_iteration=25, MRF=3km')
ax1.plot(df_iter_50['max_neighbours'], df_iter_50['mean_absolute_error'], label = 'rsir_iteration=50, MRF=3km')
ax1.plot(df_mrf_1km['max_neighbours'], df_mrf_1km['mean_absolute_error'], linestyle='--', label = 'rsir_iteration=25, MRF=1km')
# ax1.plot(smoothing_0_5['max_neighbours'], smoothing_0_5['mean_absolute_error'], label='bg_smoothing=0.5')
# ax1.plot(smoothing_1['max_neighbours'], smoothing_1['mean_absolute_error'], linestyle='--', label='bg_smoothing=1')

# Set labels and legend
ax1.set_xlabel('Max_neighbours [-]', fontsize=16)
ax1.set_ylabel('Mean Absolute Error [K]', fontsize=16)
ax1.tick_params(axis='y')
ax1.legend(fontsize=14, loc='upper right')  # Increased legend size

# Create secondary y-axis (without adding another line)
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Absolute Percentage Error [%]', fontsize=16)

# Adjust secondary y-axis scale to match MAPE values
mape_min, mape_max = df_max_neighbours['mean_absolute_percentage_error'].min(), df_max_neighbours['mean_absolute_percentage_error'].max()
ax2.set_ylim(mape_min, mape_max)

# Set tick colors
ax2.tick_params(axis='y')

plt.title('RSIR C BAND max_neighbours vs MAE & MAPE (EASE2_G9km)', fontsize=16)
plt.show()