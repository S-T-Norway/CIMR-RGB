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

config_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/L_curve/L_curve_config.xml'

# Run the RGB
config_object = ConfigFile(config_file)
data_dict = DataIngestion(config_object).ingest_data()
data_dict_out = ReGridder(config_object).regrid_data(data_dict)