import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

# results_file = '/tests/Performance_Assesment/plotting_scripts/BG/max_neighbours/ka_max_neighbour.csv'
results_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/BG/enhanced_L_9km/bg_smoothing_enhanced_reduced.csv'



# Read CSV File
df = pd.read_csv(results_file)

# Filter Data
smoothing_0 = df[df['bg_smoothing'] == 0.001]
smoothing_0_5 = df[df['bg_smoothing'] == 0.5]
smoothing_1 = df[df['bg_smoothing'] == 1]

# sort data
smoothing_0 = smoothing_0.sort_values(by='max_neighbours', ascending=True)
smoothing_0_5 = smoothing_0_5.sort_values(by='max_neighbours', ascending=True)
smoothing_1 = smoothing_1.sort_values(by='max_neighbours', ascending=True)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

# Primary y-axis: Plot mean absolute error
ax1.plot(smoothing_0['max_neighbours'], smoothing_0['mean_absolute_error'], label='bg_smoothing = 0.001')
ax1.plot(smoothing_0_5['max_neighbours'], smoothing_0_5['mean_absolute_error'], label='bg_smoothing=0.5')
ax1.plot(smoothing_1['max_neighbours'], smoothing_1['mean_absolute_error'], label='bg_smoothing=1')

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
