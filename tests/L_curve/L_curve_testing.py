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

def set_text(root, path, value):
    """Set text of a single element found by XPath-like path."""
    el = root.find(path)
    if el is None:
        raise ValueError(f"Tag not found: {path}")
    # Booleans should be "True"/"False" as strings in your XML
    if isinstance(value, bool):
        el.text = "True" if value else "False"
    elif isinstance(value, (list, tuple)):
        el.text = " ".join(map(str, value))  # e.g. "100000 100000"
    else:
        el.text = str(value)

def update_config(input_xml, output_xml, updates):
    """
    updates: dict of { 'XPath-like path from root' : value }
    """
    tree = ET.parse(input_xml)
    root = tree.getroot()
    for path, val in updates.items():
        set_text(root, path, val)

    tree.write(output_xml, encoding="utf-8", xml_declaration=True)

# Set to the config file in the L2PAD_Workshop folder
config_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/L_curve/L_curve_config.xml'

max_neighbours = [700]
# bg_smoothing = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
bg_smoothing = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.021, 0.022, 0.023, 0.024, 0.025]
# bg_smoothing = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.021, 0.022, 0.023, 0.024, 0.025]

with open('l_curve_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['source_band', 'target_band','scan_direction', 'max_neighbours', 'bg_smoothing', 'fit_error', 'noise_error'])

updates = {"./ReGridderParams/max_neighbours": 10000}
update_config(config_file, config_file, updates)

for max_neighbours in max_neighbours:
    for bg_smooth in bg_smoothing:
        updates = {
            "./ReGridderParams/max_neighbours": max_neighbours,
            "./ReGridderParams/bg_smoothing": bg_smooth
        }
        update_config(config_file, config_file, updates)
        print(f"Running with max_neighbours={max_neighbours}, bg_smoothing={bg_smooth}")

        # Run the RGB
        config_object = ConfigFile(config_file)
        data_dict = DataIngestion(config_object).ingest_data()
        data_dict_out = ReGridder(config_object).regrid_data(data_dict)















