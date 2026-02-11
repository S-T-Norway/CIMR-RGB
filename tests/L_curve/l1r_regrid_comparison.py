# --- Imports & plotting config ---
import matplotlib
matplotlib.use("TkAgg")  # comment out if not using interactive windows
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import UnivariateSpline

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})


def load_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

bg_data = load_pickle('/home/beywood/ST/CIMR_RGB/Reporting/EUMETSAT proposal/data/c2l_3km_bg_0_45_125_279_300_neigh.pickle')
ids_data = load_pickle('/home/beywood/ST/CIMR_RGB/Reporting/EUMETSAT proposal/data/c2l_3km_ids_0_45_125_279_300_neigh.pickle')
rsir_data = load_pickle('/home/beywood/ST/CIMR_RGB/Reporting/EUMETSAT proposal/data/c2l_3km_rsir_0_45_125_279_500_neigh.pickle')

bt_ids = ids_data['C']['bt_h'][0: 45, 125:279]
bt_bg =  bg_data['C']['bt_h'][0: 45, 125:279]
bt_rsir = rsir_data['C']['bt_h'][0: 45, 125:279]

vmin = min(np.nanmin(bt_ids), np.nanmin(bt_bg), np.nanmin(bt_rsir))  # add np.nanmin(bt_rsir) when ready
vmax = max(np.nanmax(bt_ids), np.nanmax(bt_bg), np.nanmin(bt_rsir))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)

im1 = ax1.imshow(bt_ids, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
ax1.set_title("C-->L regrid, max_neighbours=300, regridding_algorithm=IDS")
ax1.set_xlabel("Earth Sample Number [-]")
ax1.set_ylabel("CIMR Scan Number [-]")

im2 = ax2.imshow(bt_bg, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
ax2.set_title("C-->L regrid, max_neighbours=300, regridding_algorithm=BG")
ax2.set_xlabel("Earth Sample Number [-]")
ax2.set_ylabel("CIMR Scan Number [-]")

im3 = ax3.imshow(bt_rsir, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax) # Update with RSIR
ax3.set_title("C-->L regrid, max_neighbours=500, regridding_algorithm=RSIR")
ax3.set_xlabel("Earth Sample Number [-]")
ax3.set_ylabel("CIMR Scan Number [-]")

# --- Shared colorbar on the right ---
cbar = fig.colorbar(im3, ax=[ax1, ax2, ax3], location="right", fraction=0.05, pad=0.04)
cbar.set_label("C Band BT (H-Pol) [K]")
plt.show()

# plt.figure()
# plt.imshow(bt_ids - bt_bg, origin="upper", cmap="bwr")
# plt.colorbar(label="Difference in Horizontally Polarised Brightness Temperature [K]")
# plt.title("Difference: Inverse Distance Squared - Backus-Gilbert")
# plt.xlabel("Earth Sample Number [-]")
# plt.ylabel("CIMR Scan Number [-]")

ids_bg = (bt_ids - bt_bg)
ids_rsir = (bt_ids - bt_rsir)
rsir_bg = (bt_rsir - bt_bg)

vmin_diff = -90# min(np.nanmin(ids_bg), np.nanmin(ids_rsir), np.nanmin(rsir_bg))
vmax_diff =90 #max(np.nanmax(ids_bg), np.nanmax(ids_rsir), np.nanmax(rsir_bg))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
im1 = ax1.imshow(ids_bg, origin="upper", cmap="bwr", vmin=vmin_diff, vmax=vmax_diff)
ax1.set_title("Difference: IDS - BG")
ax1.set_xlabel("Earth Sample Number [-]")
ax1.set_ylabel("CIMR Scan Number [-]")
im2 = ax2.imshow(ids_rsir, origin="upper", cmap="bwr", vmin=vmin_diff, vmax=vmax_diff)
ax2.set_title("Difference: IDS - RSIR")
ax2.set_xlabel("Earth Sample Number [-]")
ax2.set_ylabel("CIMR Scan Number [-]")
im3 = ax3.imshow(rsir_bg, origin="upper", cmap="bwr",
                    vmin=vmin_diff, vmax=vmax_diff)  # Update with RSIR
ax3.set_title("Difference: RSIR - BG")
ax3.set_xlabel("Earth Sample Number [-]")
ax3.set_ylabel("CIMR Scan Number [-]")
# --- Shared colorbar on the right ---
cbar = fig.colorbar(im3, ax=[ax1, ax2, ax3
], location="right", fraction=0.05, pad=0.04)
cbar.set_label("C Band BTs [K]")
plt.show()
