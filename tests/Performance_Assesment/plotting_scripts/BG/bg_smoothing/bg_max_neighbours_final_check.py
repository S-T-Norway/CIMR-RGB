import matplotlib
import pandas as pd
from tests.Performance_Assesment import metrics as metrics
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

results_file_1 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/BG/bg_smoothing/BG_36km_bg_smoothing_extensive.csv'

# Read CSV File

df = pd.read_csv(results_file_1)
df.head()

# Plot MAE against max_neightbours
BAND = 'L'
band = df[df['band'] == BAND]


# Select a set based on the value of a column
df_1 = band[band['MRF_grid_resolution'] == '1km']
df_3 = band[band['MRF_grid_resolution'] == '3km']
df_9 = band[band['MRF_grid_resolution'] == '9km']


df_1_sorted = df_1.sort_values(by='bg_smoothing', ascending=True)
df_3_sorted = df_3.sort_values(by='bg_smoothing', ascending=True)
df_9_sorted = df_9.sort_values(by='bg_smoothing', ascending=True)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

# Primary y-axis: Plot mean absolute error
ax1.plot(df_1_sorted['bg_smoothing'], df_1_sorted['mean_absolute_error'], label = 'MRF_grid_resolution=1km')
ax1.plot(df_3_sorted['bg_smoothing'], df_3_sorted['mean_absolute_error'], label = 'MRF_grid_resolution=3km')
ax1.plot(df_9_sorted['bg_smoothing'], df_9_sorted['mean_absolute_error'], label = 'MRF_grid_resolution=9km')


# ax1.plot(smoothing_0_5['max_neighbours'], smoothing_0_5['mean_absolute_error'], label='bg_smoothing=0.5')
# ax1.plot(smoothing_1['max_neighbours'], smoothing_1['mean_absolute_error'], linestyle='--', label='bg_smoothing=1')

# Set labels and legend
ax1.set_xlabel('bg_smoothing [-]', fontsize=16)
ax1.set_ylabel('Mean Absolute Error [K]', fontsize=16)
ax1.tick_params(axis='y')
ax1.legend(fontsize=14, loc='upper right')  # Increased legend size

# Create secondary y-axis (without adding another line)
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Absolute Percentage Error [%]', fontsize=16)

# Adjust secondary y-axis scale to match MAPE values
mape_min, mape_max = band['mean_absolute_percentage_error'].min(), band['mean_absolute_percentage_error'].max()
ax2.set_ylim(mape_min, mape_max)

# Set tick colors
ax2.tick_params(axis='y')

plt.title('BG L BAND bg_smoothing vs MAE & MAPE (EASE2_G36km)', fontsize=16)
plt.show()


