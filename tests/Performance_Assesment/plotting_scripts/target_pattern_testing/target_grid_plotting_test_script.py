import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

# results_file = '/tests/Performance_Assesment/plotting_scripts/BG/max_neighbours/ka_max_neighbour.csv'
results_file_test_1 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_1.csv'
results_file_test_2 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_2.csv'
results_file_test_3 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_3.csv'
results_file_test_4 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_4.csv'
results_file_test_5 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_5.csv'
# results_file_test_6 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/results/changing_target_pattern_test_6.csv'
results_file_test_7 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_7.csv'
results_file_test_8 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_8.csv'
# add 9, 10, 11, 12, 13
results_file_test_9 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_9.csv'
results_file_test_10 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_10.csv'
results_file_test_11 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_11.csv'
results_file_test_12 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_12.csv'
results_file_test_13 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/changing_target_pattern_test_13.csv'
results_file_14 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/testing_target_pattern_test_divide_10.csv'
results_file_15 = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/plotting_scripts/target_pattern_testing/test_non_overlap_inds.csv'


# Read CSV File
df_1 = pd.read_csv(results_file_test_1)
df_2 = pd.read_csv(results_file_test_2)
df_3 = pd.read_csv(results_file_test_3)
df_4 = pd.read_csv(results_file_test_4)
df_5 = pd.read_csv(results_file_test_5)
# df_6 = pd.read_csv(results_file_test_6)
df_7 = pd.read_csv(results_file_test_7)
df_8 = pd.read_csv(results_file_test_8)
df_9 = pd.read_csv(results_file_test_9)
df_10 = pd.read_csv(results_file_test_10)
df_11 = pd.read_csv(results_file_test_11)
df_12 = pd.read_csv(results_file_test_12)
df_13 = pd.read_csv(results_file_test_13)
df_14 = pd.read_csv(results_file_14)
df_15 = pd.read_csv(results_file_15)

# Filter Data
smoothing_0 = df_1[df_1['bg_smoothing'] == 0.04]
smoothing_0_5 = df_2[df_2['bg_smoothing'] == 0.04]
smoothing_1 = df_3[df_3['bg_smoothing'] == 0.04]
smoothing_4 = df_4[df_4['bg_smoothing'] == 0.04]
smoothing_5 = df_5[df_5['bg_smoothing'] == 0.04]
# smoothing_6 = df_6[df_6['bg_smoothing'] == 0.04]
smoothing_7 = df_7[df_7['bg_smoothing'] == 0.04]
smoothing_8 = df_8[df_8['bg_smoothing'] == 0.04]
smoothing_9 = df_9[df_9['bg_smoothing'] == 0.04]
smoothing_10 = df_10[df_10['bg_smoothing'] == 0.04]
smoothing_11 = df_11[df_11['bg_smoothing'] == 0.04]
smoothing_12 = df_12[df_12['bg_smoothing'] == 0.04]
smoothing_13 = df_13[df_13['bg_smoothing'] == 0.04]
smoothing_14 = df_14[df_14['bg_smoothing'] == 0.04]
smoothing_15 = df_15[df_15['bg_smoothing'] == 0.04]

# sort data
smoothing_0 = smoothing_0.sort_values(by='max_neighbours', ascending=True)
smoothing_0_5 = smoothing_0_5.sort_values(by='max_neighbours', ascending=True)
smoothing_1 = smoothing_1.sort_values(by='max_neighbours', ascending=True)
smoothing_4 = smoothing_4.sort_values(by='max_neighbours', ascending=True)
smoothing_5 = smoothing_5.sort_values(by='max_neighbours', ascending=True)
# smoothing_6 = smoothing_6.sort_values(by='max_neighbours', ascending=True)
smoothing_7 = smoothing_7.sort_values(by='max_neighbours', ascending=True)
smoothing_8 = smoothing_8.sort_values(by='max_neighbours', ascending=True)
smoothing_9 = smoothing_9.sort_values(by='max_neighbours', ascending=True)
smoothing_10 = smoothing_10.sort_values(by='max_neighbours', ascending=True)
smoothing_11 = smoothing_11.sort_values(by='max_neighbours', ascending=True)
smoothing_12 = smoothing_12.sort_values(by='max_neighbours', ascending=True)
smoothing_13 = smoothing_13.sort_values(by='max_neighbours', ascending=True)
smoothing_14  = smoothing_14.sort_values(by='max_neighbours', ascending=True)
smoothing_15 = smoothing_15.sort_values(by='max_neighbours', ascending=True)

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)

# Primary y-axis: Plot mean absolute error
ax1.plot(smoothing_0['max_neighbours'], smoothing_0['mean_absolute_error'],'--', label='original target method sigma = 36km')
ax1.plot(smoothing_0_5['max_neighbours'], smoothing_0_5['mean_absolute_error'],'--', label='original method / 2 sigma = 18km')
ax1.plot(smoothing_1['max_neighbours'], smoothing_1['mean_absolute_error'], linestyle='--', label='3db size sigma = 43x73km')
ax1.plot(smoothing_4['max_neighbours'], smoothing_4['mean_absolute_error'], linestyle='--', label='3db size sigma = 3km')
ax1.plot(smoothing_5['max_neighbours'], smoothing_5['mean_absolute_error'], linestyle='--', label='3db size sigma = 1km')
# ax1.plot(smoothing_6['max_neighbours'], smoothing_6['mean_absolute_error'], linestyle='--', label='3db size sigma = 1m')
ax1.plot(smoothing_7['max_neighbours'], smoothing_7['mean_absolute_error'], linestyle='--', label='3db size sigma = -3dB + target_cell_size')
ax1.plot(smoothing_8['max_neighbours'], smoothing_8['mean_absolute_error'], linestyle='-', label='3db size sigma = -3dB + -3db cell size')
ax1.plot(smoothing_9['max_neighbours'], smoothing_9['mean_absolute_error'], label='new implementation 43x79')
ax1.plot(smoothing_10['max_neighbours'], smoothing_10['mean_absolute_error'], label='new implementation 18x18')
ax1.plot(smoothing_11['max_neighbours'], smoothing_11['mean_absolute_error'], label='new implementation 9x9')
ax1.plot(smoothing_12['max_neighbours'], smoothing_12['mean_absolute_error'], label='new_implementation 3x3')
ax1.plot(smoothing_13['max_neighbours'], smoothing_13['mean_absolute_error'], label='new imeplemenation 1x1')
ax1.plot(smoothing_14['max_neighbours'], smoothing_14['mean_absolute_error'], label='divide target by 10, original implementation')
ax1.plot(smoothing_15['max_neighbours'], smoothing_15['mean_absolute_error'], label='non-overlapping indices')

# Set labels and legend
ax1.set_xlabel('max_neighbours', fontsize=16)
ax1.set_ylabel('Mean Absolute Error [K]', fontsize=16)
ax1.tick_params(axis='y')
ax1.legend(fontsize=14, loc='upper left')  # Increased legend size

# # Create secondary y-axis (without adding another line)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Mean Absolute Percentage Error [%]', fontsize=16)
#
# # Adjust secondary y-axis scale to match MAPE values
# mape_min, mape_max = df['mean_absolute_percentage_error'].min(), df['mean_absolute_percentage_error'].max()
# ax2.set_ylim(mape_min, mape_max)
#
# # Set tick colors
# ax2.tick_params(axis='y')

plt.title('Range of L1C Target Pattern Implementations (EASE2_G36km) (MRF = 3km)', fontsize=16)
plt.show()
