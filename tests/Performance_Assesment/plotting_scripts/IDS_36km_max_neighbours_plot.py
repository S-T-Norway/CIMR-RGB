import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

results_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/results/results.csv'



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


fig, axs = plt.subplots(1, 5)
lw=3
fontsize =14
axs[0].plot(L_BAND['max_neighbours'], L_BAND['mean_absolute_percentage_error'], lw=lw)
axs[0].set_title('L BAND', fontsize=fontsize)
axs[0].set_ylabel('Mean Absolute Percentage Error [%]', fontsize = 14)
axs[1].plot(C_BAND['max_neighbours'], C_BAND['mean_absolute_percentage_error'], lw=lw)
axs[1].set_title('C BAND', fontsize=fontsize)
axs[2].plot(X_BAND['max_neighbours'], X_BAND['mean_absolute_percentage_error'], lw=lw)
axs[2].set_title('X BAND', fontsize=fontsize)
axs[3].plot(K_BAND['max_neighbours'], K_BAND['mean_absolute_percentage_error'], lw=lw)
axs[3].set_title('K BAND', fontsize=fontsize)
axs[2].set_xlabel('max_neighbours [-]', fontsize = 14)
axs[4].plot(KA_BAND['max_neighbours'], KA_BAND['mean_absolute_percentage_error'], lw=lw)
axs[4].set_title('KA BAND', fontsize=fontsize)
plt.suptitle('IDS max_neighbour vs MAPE (EASE2_G36km)', fontsize=16)
plt.show()
