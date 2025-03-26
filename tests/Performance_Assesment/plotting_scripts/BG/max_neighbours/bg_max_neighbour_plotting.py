import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

# results_file = '/tests/Performance_Assesment/plotting_scripts/BG/max_neighbours/ka_max_neighbour.csv'
results_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/BG/max_neighbours/BG_36km_max_neighbours_MRF_3km_gamma_0.0001.csv'
# results_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/BG/max_neighbours/surely_bg_can_be_better_l_band_max_neighbours_BG.csv'




# Read CSV File
df = pd.read_csv(results_file)

# Filter Data
smoothing_0001 = df[df['bg_smoothing'] == 0.0001]
smoothing_0_5 = df[df['bg_smoothing'] == 0.5]
smoothing_1 = df[df['bg_smoothing'] == 1]
smoothing_001 = df[df['bg_smoothing'] == 0.001]
smoothing_002 = df[df['bg_smoothing'] == 0.002]


# sort data
smoothing_0 = smoothing_0001.sort_values(by='max_neighbours', ascending=True)
smoothing_0_5 = smoothing_0_5.sort_values(by='max_neighbours', ascending=True)
smoothing_1 = smoothing_1.sort_values(by='max_neighbours', ascending=True)
smoothing_001 = smoothing_001.sort_values(by='max_neighbours', ascending=True)
smoothing_002 = smoothing_002.sort_values(by='max_neighbours', ascending=True)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

# Primary y-axis: Plot mean absolute error
ax1.plot(smoothing_0001['max_neighbours'], smoothing_0001['mean_absolute_error'], label='bg_smoothing = 0.0001')
ax1.plot(smoothing_0_5['max_neighbours'], smoothing_0_5['mean_absolute_error'], label='bg_smoothing=0.5')
ax1.plot(smoothing_1['max_neighbours'], smoothing_1['mean_absolute_error'], linestyle='--', label='bg_smoothing=1')
ax1.plot(smoothing_001['max_neighbours'], smoothing_001['mean_absolute_error'], linestyle='--', label='bg_smoothing=0.001')
ax1.plot(smoothing_002['max_neighbours'], smoothing_002['mean_absolute_error'], linestyle='--', label='bg_smoothing=0.002')

# Set labels and legend
ax1.set_xlabel('max_neighbours', fontsize=16)
ax1.set_ylabel('Mean Absolute Error [K]', fontsize=16)
ax1.tick_params(axis='y')
ax1.legend(fontsize=14, loc='upper right')  # Increased legend size

# Create secondary y-axis (without adding another line)
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Absolute Percentage Error [%]', fontsize=16)

# Adjust secondary y-axis scale to match MAPE values
mape_min, mape_max = df['mean_absolute_percentage_error'].min(), df['mean_absolute_percentage_error'].max()
ax2.set_ylim(mape_min, mape_max)

# Set tick colors
ax2.tick_params(axis='y')

plt.title('BG L BAND max_neighbour vs MAE & MAPE (EASE2_G36km)', fontsize=16)
plt.show()
