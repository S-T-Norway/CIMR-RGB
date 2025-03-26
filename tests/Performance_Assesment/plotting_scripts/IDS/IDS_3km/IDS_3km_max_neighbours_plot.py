import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

results_file = 'IDS_3km_max_neighbours.csv'



# Read CSV File
import pandas as pd
df = pd.read_csv(results_file)
df.head()

# Plot MAE against max_neightbours
# plt.figure()
L_BAND = df[df['band'] == 'L']
C_BAND = df[df['band'] == 'C']
X_BAND = df[df['band'] == 'X']
K_BAND = df[df['band'] == 'K']
K_BAND = K_BAND.sort_values(by='max_neighbours', ascending=True)

KA_BAND = df[df['band'] == 'KA']
KA_BAND = KA_BAND.sort_values(by='max_neighbours', ascending=True)

# plt.plot(L_BAND['max_neighbours'], L_BAND['mean_absolute_percentage_error'])
# plt.plot(C_BAND['max_neighbours'], C_BAND['mean_absolute_percentage_error'])
# plt.plot(K_BAND['max_neighbours'], K_BAND['mean_absolute_percentage_error'])


# plt.figure()
# plt.plot(L_BAND['bg_smoothing'], L_BAND['mean_absolute_percentage_error'])

fig, axs = plt.subplots(1, 5, figsize=(18, 5), constrained_layout=True)  # Adjust figure size
lw = 3
fontsize = 14
band_colour= ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for i, (band, band_data) in enumerate(zip(
        [L_BAND, C_BAND, X_BAND, K_BAND, KA_BAND],
        ['L BAND', 'C BAND', 'X BAND', 'K BAND', 'KA BAND'])):

    # Primary y-axis: Plot the error metric
    axs[i].plot(band['max_neighbours'], band['mean_absolute_error'], lw=lw, color=band_colour[i])
    axs[i].set_title(band_data, fontsize=16)

    # Create a second y-axis but WITHOUT plotting another line
    ax2 = axs[i].twinx()
    if i == 4:
        ax2.set_ylabel('Mean Absolute Percentage Error [%]', fontsize=16)

    # Adjust the scale of the second y-axis based on MAPE values
    ymin, ymax = axs[i].get_ylim()  # Get limits from the primary y-axis
    mape_min, mape_max = band['mean_absolute_percentage_error'].min(), band['mean_absolute_percentage_error'].max()
    ax2.set_ylim(mape_min, mape_max)  # Set the limits for the secondary y-axis



    if i == 0:
        axs[i].set_ylabel('Mean Absolute Error [K]', fontsize=16)

axs[2].set_xlabel('max_neighbours [-]', fontsize=14)
plt.suptitle('IDS max_neighbour vs MAE & MAPE (EASE2_G3km)', fontsize=16)
plt.show()