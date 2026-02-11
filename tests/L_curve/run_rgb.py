import csv

from cimr_rgb.config_file import ConfigFile
from cimr_rgb.data_ingestion import DataIngestion
from cimr_rgb.regridder import ReGridder
from cimr_rgb import utils
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import os
import xml.etree.ElementTree as ET

# Set to the config file in the L2PAD_Workshop folder
config_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/L_curve/L_curve_config.xml'

# Run the RGB
config_object = ConfigFile(config_file)
data_dict = DataIngestion(config_object).ingest_data()
data_dict_out = ReGridder(config_object).regrid_data(data_dict)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

# Example image (replace with your own data)
img = data_dict_out['C']['bt_h'][0:45, 125:279]

fig, ax = plt.subplots()
im = ax.imshow(img, cmap='viridis', origin='upper')

# Coordinates for rectangle (top-left corner)
x, y = 125, 0     # top-left corner
width, height =154 ,45

# # Add the rectangle
# rect = patches.Rectangle(
#     (x, y), width, height,
#     linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
# )
# ax.add_patch(rect)

# Axis labels
ax.set_xlabel("Earth Sample Number [-]")
ax.set_ylabel("Scan Number [-]")

# Subtitle
ax.set_title(f"C-->L regrid, max_neighbours={config_object.max_neighbours}, regridding_algorithm=IDS")

# Add colorbar underneath
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.1)
cbar.set_label("C Band BT [K]")

plt.tight_layout()
plt.show()
