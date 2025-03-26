import matplotlib
import pandas as pd


from tests.Performance_Assesment import metrics as metrics
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

results_file_1 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/BG/enhanced_L_9km/enhanced_l_band_bg_smoothing.csv'
# results_file_1 = '/tests/Performance_Assesment/plotting_scripts/BG/enhanced_L_9km/enhanced_l_bg_smoothing_3_neigbours.csv'
results_file_2 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/BG/enhanced_L_9km/enhanced_l_band_bg_smoothing_2_neighbours.csv'
# Read CSV File
results_file_3 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/BG/enhanced_L_9km/bg_smoothing_enhanced_reduced.csv'

df = pd.read_csv(results_file_1)
df.head()

df_2= pd.read_csv(results_file_2)

df_3 = pd.read_csv(results_file_3)
df_3.head()

# Plot MAE against max_neightbours
BAND = 'L'
band = df[df['band'] == BAND]
band_2 = df_2[df_2['band'] == BAND]
band_3 = df_3[df_3['band'] == BAND]


# Select a set based on the value of a column
df_1 = band[band['MRF_grid_resolution'] == '1km']
df_3 = band[band['MRF_grid_resolution'] == '3km']
df_9 = band[band['MRF_grid_resolution'] == '9km']

df_2_1  = band_2[band_2['MRF_grid_resolution'] == '1km']
df_2_3  = band_2[band_2['MRF_grid_resolution'] == '3km']
df_2_9  = band_2[band_2['MRF_grid_resolution'] == '9km']


df_1_sorted = df_1.sort_values(by='bg_smoothing', ascending=True)
df_3_sorted = df_3.sort_values(by='bg_smoothing', ascending=True)
df_9_sorted = df_9.sort_values(by='bg_smoothing', ascending=True)

df_2_1_sorted = df_2_1.sort_values(by='bg_smoothing', ascending=True)
df_2_3_sorted = df_2_3.sort_values(by='bg_smoothing', ascending=True)
df_2_9_sorted = df_2_9.sort_values(by='bg_smoothing', ascending=True)

df_reduced = band_3.sort_values(by='bg_smoothing', ascending=True)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

# Primary y-axis: Plot mean absolute error
ax1.plot(df_1_sorted['bg_smoothing'], df_1_sorted['mean_absolute_error'], label = 'MRF_grid=1km, max_neighbours = 3')
ax1.plot(df_3_sorted['bg_smoothing'], df_3_sorted['mean_absolute_error'], label = 'MRF_grid=3km, max_neighbours = 3')
ax1.plot(df_9_sorted['bg_smoothing'], df_9_sorted['mean_absolute_error'], label = 'MRF_grid=9km, max_neighbours = 3')

ax1.plot(df_2_1_sorted['bg_smoothing'], df_2_1_sorted['mean_absolute_error'],linestyle='--', label = 'MRF_grid=1km, max_neighbours=2')
ax1.plot(df_2_3_sorted['bg_smoothing'], df_2_3_sorted['mean_absolute_error'],linestyle='--', label = 'MRF_grid=3km, max_neighbours=2')
ax1.plot(df_2_9_sorted['bg_smoothing'], df_2_9_sorted['mean_absolute_error'],linestyle='--', label = 'MRF_grid=9km, max_neighbours=2')

ax1.scatter(df_reduced['bg_smoothing'], df_reduced['mean_absolute_error'], label = 'MRF_grid=9km, max_neighbours=55, reduced')
ax1.plot(df_reduced['bg_smoothing'], df_reduced['mean_absolute_error'],linestyle='--', label = 'MRF_grid=9km, max_neighbours=55, reduced')


# Set labels and legend
ax1.set_xlabel('bg_smoothing [-]', fontsize=16)
ax1.set_ylabel('Mean Absolute Error [K]', fontsize=16)
ax1.tick_params(axis='y')
ax1.legend(fontsize=14, loc='lower right')  # Increased legend size

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


