"""
This module contains the DataIngestion class, which is responsible for ingesting
and processing various types of satellite data. It supports various AMSR2, SMAP
and simulated CIMR data. Providing functionalities to read, clean, and convert
data into a standardized format for further analysis or processing in the pipeline.

The DataIngestion class utilizes configuration files to manage the ingestion
process.
"""

import re
from numpy import array, sqrt, cos, pi, sin, zeros, arctan2, arccos, nan
import numpy as np
import h5py as h5

from config_file import ConfigFile
from utils import remove_overlap

# SMAP Constants
SMAP_FILL_FLOAT_32 = -9999.0

# AMSR2 Constants
AM2_DEF_SNUM_HI = 486
AM2_DEF_SNUM_LOW = 243
MV = -9999.0


class DataIngestion:
    """
    This class is responsible for ingesting the data from the user specified path
    and converting it to a common format for use further in the pipeline.

    Attributes:
    -----------
    config_file_path : str
        The path to the configuration file that contains the user specified parameters.

    Methods
    -------
    amsr2_coreg_extraction(coreg_parameters):
        Extracts the coregistration parameters from the AMSR2 data. Used for extraction of lon, lats
        of the lower frequency channels from the higher frequency channels.

    read_hdf5():
        Reads the HDF5 file and extracts the data and metadata.

    extract_smap_qc(qc_dict):
        Applied NaNs to qc_dict extracted in read_hdf5() function.

    apply_smap_qc(qc_dict, data_dict):
        Applies the quality control values to the SMAP data for each polarisation.

    clean_data(data_dict):
        Cleans the data by replacing fill values with NaNs.

    amsr2_latlon_conversion(coreg_a, coreg_b, lats_hi, lons_hi):
        Obtains the latitudes and longitudes of the lower frequency channels from the
        higher frequency channels.

    ingest_amsr2():
        Ingests AMSR2 data from the user specified path
        and converts to a common format for use further in the pipeline.

    ingest_smap():
        Ingests SMAP data from the user specified path
        and converts to a common format for use further in the pipeline.

    ingest_data():
        Initiates ingestion of user specified data type.
    """

    def __init__(self, config_file_path):
        """
        Initializes the DataIngestion class with the user specified configuration file.

        Parameters
        ----------
        config_file_path: str
            The path to the configuration file that contains the user specified parameters.
        """
        self.config = ConfigFile(config_file_path)

    @staticmethod
    def amsr2_coreg_extraction(coreg_parameters):
        """
        Extracts the co-registration parameters from the AMSR2 data.
        Used for extraction of lon, lats of the lower frequency channels
        from the higher frequency channels.

        Parameters
        ----------
        coreg_parameters: str
            The co-registration parameters extracted from the AMSR2 metadata.

        Returns
        -------
        params_floats: list[float]
            List of extracted correg parameters.
        """
        params_strings = re.findall(r'--\d+\.\d+|-\d+\.\d+|\d+\.\d+', coreg_parameters)
        params_floats = []
        for param in params_strings:
            if param.startswith('--'):
                # Registration Parameter is negative
                params_floats.append(-float(param[2:]))
            else:
                params_floats.append(float(param[1:]))
        return params_floats

    def read_hdf5(self):
        """
        Reads the HDF5 file and extracts the data and metadata.

        Returns
        -------
        data_dict: dict
            Dictionary containing the data extracted from the HDF5 file.
        """

        with h5.File(self.config.input_data_path, 'r') as data:

            data_dict = {}
            # Not using qc_dict at the moment need to add functionality for "qc remap".
            qc_dict = {}

            if self.config.input_data_type == "AMSR2":

                # Extract Metadata
                coreg_a = self.amsr2_coreg_extraction(data.attrs['CoRegistrationParameterA1'][0])
                coreg_b = self.amsr2_coreg_extraction(data.attrs['CoRegistrationParameterA2'][0])
                overlap = int(data.attrs['OverlapScans'][0])
                num_scans = int(data.attrs['NumberOfScans'][0])
                self.config.num_scans = num_scans
                self.config.AM2_DEF_SNUM_HI = AM2_DEF_SNUM_HI
                self.config.AM2_DEF_SNUM_LOW = AM2_DEF_SNUM_LOW

                if not num_scans <= self.config.LMT:
                    raise ValueError(
                        f"Number of scans ({num_scans} exceeds limit of {self.config.LMT}."
                    )

                # Extract Data used for all re-grids
                lats_89a = remove_overlap(
                    array=data['Latitude of Observation Point for 89A'][:],
                    overlap=overlap
                )

                lons_89a = remove_overlap(
                    array=data['Longitude of Observation Point for 89A'][:],
                    overlap=overlap
                )

                # Extract 89b data if required
                if self.config.target_band == '89b' or self.config.source_band == '89b':
                    lats_89b = remove_overlap(
                        array=data['Latitude of Observation Point for 89B'][:],
                        overlap=overlap
                    )

                    lons_89b = remove_overlap(
                        array=data['Longitude of Observation Point for 89B'][:],
                        overlap=overlap
                    )
                else:
                    lats_89b = None
                    lons_89b = None

                # Extract BTs of Source Band
                key = self.config.key_mappings[self.config.source_band][1]
                bt_h_source, bt_v_source = data[key + 'H)'], data[key + 'V)']
                bt_h_source_scale = bt_h_source.attrs['SCALE FACTOR']
                bt_v_source_scale = bt_v_source.attrs['SCALE FACTOR']

                data_dict['bt_v_source'] = remove_overlap(bt_v_source, overlap) * bt_v_source_scale
                data_dict['bt_h_source'] = remove_overlap(bt_h_source, overlap) * bt_h_source_scale

                return data_dict, coreg_a, coreg_b, lats_89a, lons_89a, lats_89b, lons_89b

            if self.config.input_data_type == "SMAP":
                bt_data = data['Brightness_Temperature']

                # Uncomment all the variables you want to remap.
                data_dict['bt_h_target'] = bt_data['tb_h'][:]
                # data_dict['bt_v_target'] = bt_data['tb_v'][:]
                # data_dict['bt_3_target'] = bt_data['tb_3'][:]
                # data_dict['bt_4_target'] = bt_data['tb_4'][:]
                data_dict['lats_target'] = bt_data['tb_lat'][:]
                data_dict['lons_target'] = bt_data['tb_lon'][:]
                data_dict["antenna_scan_angle"] = bt_data["antenna_scan_angle"][:]
                # data_dict['bt_h_target_nedt'] = bt_data['nedt_h'][:]
                # data_dict['bt_v_target_nedt'] = bt_data['nedt_v'][:]
                # data_dict['bt_3_target_nedt'] = bt_data['nedt_3'][:]
                # data_dict['bt_4_target_nedt'] = bt_data['nedt_4'][:]
                # qc_dict['bt_h_qc'] = bt_data['tb_qual_flag_h'][:]
                # qc_dict['bt_v_qc'] = bt_data['tb_qual_flag_v'][:]
                # qc_dict['bt_3_qc'] = bt_data['tb_qual_flag_3'][:]
                # qc_dict['bt_4_qc'] = bt_data['tb_qual_flag_4'][:]
                # data_dict['boresight_angle'] = bt_data['earth_boresight_incidence'][:]
                # data_dict['sol_specular_theta'] = bt_data['solar_specular_theta'][:]
                # data_dict['sol_specular_phi'] = bt_data['solar_specular_phi'][:]

                for variable in data_dict:
                    data_dict[variable] = data_dict[variable].flatten('C')
                return data_dict

            if self.config.input_data_type == "CIMR":
                return data_dict

    @staticmethod
    def extract_smap_qc(qc_dict):
        """
        Applied NaNs to qc_dict extracted in read_hdf5() function.

        Parameters
        ----------
        qc_dict: dict
            Dictionary containing the quality control values extracted
            from the SMAP data for different polarisations.

        Returns
        -------
        qc_dict: dict
            Dictionary containing the quality control values extracted
            from the SMAP data for different polarisations
            with NaNs applied.

        """
        for qc in qc_dict:
            qc_dict[qc] = np.where(qc_dict[qc] == 0, 1, np.nan)
        return qc_dict

    @staticmethod
    def apply_smap_qc(qc_dict, data_dict):
        """
        Applies the quality control values to the SMAP data for each polarisation.

        Parameters
        ----------
        qc_dict: dict
            Dictionary containing the quality control values extracted
            from the SMAP data for different polarisations.
        data_dict: dict
            Dictionary containing the data extracted from the SMAP data.

        Returns
        -------
        data_dict: dict
            Dictionary containing the data extracted from the SMAP
            data with quality control values applied.

        """
        data_dict['bt_h_target'] = data_dict['bt_h_target'] * qc_dict['bt_h_qc']
        data_dict['bt_v_target'] = data_dict['bt_v_target'] * qc_dict['bt_v_qc']
        data_dict['bt_3_target'] = data_dict['bt_3_target'] * qc_dict['bt_3_qc']
        data_dict['bt_4_target'] = data_dict['bt_4_target'] * qc_dict['bt_4_qc']
        return data_dict

    def clean_data(self, data_dict):
        """
        Applied NaNs to fill values in data extracted from the HDF5 file.

        Parameters
        ----------
        data_dict: dict
            Dictionary containing the data extracted from the HDF5 file.

        Returns
        -------
        data_dict: dict
            Dictionary containing the data extracted from the HDF5 file with NaNs applied.

        """
        if self.config.input_data_type == "SMAP":
            nan_map = np.zeros(next(iter(data_dict.values())).shape, dtype=bool)

            for variable in data_dict:
                if data_dict[variable].dtype == 'float32':
                    data_dict[variable][data_dict[variable] == SMAP_FILL_FLOAT_32] = nan
                nan_map |= np.isnan(data_dict[variable])

            for variable in data_dict:
                data_dict[variable] = np.where(nan_map, np.nan, data_dict[variable])

        return data_dict

    def amsr2_latlon_conversion(self, coreg_a, coreg_b, lons_hi, lats_hi):
        """
        Obtains the latitudes and longitudes of the lower frequency channels from the
        higher frequency (89a) channel.

        Parameters:
        -----------
        coreg_a: float
            Coregistration parameter A.
        coreg_b: float
            Coregistration parameter B.
        lats_hi: numpy.ndarray
            Latitude values of the higher frequency channel.
        lons_hi: numpy.ndarray
            Longitude values of the higher frequency channel.

        Returns:
        --------
        lats_lo: numpy.ndarray
            Latitude values of the lower frequency channel.
        lons_lo: numpy.ndarray
            Longitude values of the lower frequency channel.
        """
        rad = pi / 180.0
        deg = 180.0 / pi

        # Initiate lower frequency channel lat/lon arrays
        lats_lo = zeros((self.config.num_scans, AM2_DEF_SNUM_LOW))
        lons_lo = zeros((self.config.num_scans, AM2_DEF_SNUM_LOW))

        for scan in range(self.config.num_scans):
            for sample in range(AM2_DEF_SNUM_LOW):
                lat_1 = lats_hi[scan, sample * 2 + 0]
                lat_2 = lats_hi[scan, sample * 2 + 1]
                lon_1 = lons_hi[scan, sample * 2 + 0]
                lon_2 = lons_hi[scan, sample * 2 + 1]

                # Check if lat/lons are within valid range
                if (lat_1 < -90.0 or lat_1 > 90.0 or
                        lat_2 < -90.0 or lat_2 > 90.0 or
                        lon_1 < -180.0 or lon_1 > 180.0 or
                        lon_2 < -180.0 or lon_2 > 180.0):
                    print(f"amsr2latlon conversion: out of range warning:"
                          f"Latitude and/or longitude are"
                          f"out of range on Scan = {scan} and Sample = {sample}")
                    lats_lo[scan, sample] = MV
                    lons_lo[scan, sample] = MV
                    continue

                # Conversion Calculation. Taken from AMSR2 Sample C programs.
                p1 = array([cos(lon_1 * rad) * cos(lat_1 * rad),
                            sin(lon_1 * rad) * cos(lat_1 * rad),
                            sin(lat_1 * rad)])

                p2 = array([cos(lon_2 * rad) * cos(lat_2 * rad),
                            sin(lon_2 * rad) * cos(lat_2 * rad),
                            sin(lat_2 * rad)])
                temp = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
                theta = arccos(temp)
                ex = p1
                temp = sqrt(p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2]) * sqrt(
                    p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2]) * sin(theta)
                ez = array([p1[1] * p2[2] - p1[2] * p2[1],
                            p1[2] * p2[0] - p1[0] * p2[2],
                            p1[0] * p2[1] - p1[1] * p2[0]]) / temp
                ey = array([ez[1] * ex[2] - ez[2] * ex[1],
                            ez[2] * ex[0] - ez[0] * ex[2],
                            ez[0] * ex[1] - ez[1] * ex[0]])
                j = cos(coreg_b * theta)
                k = cos(coreg_a * theta)
                l = sin(coreg_a * theta)
                m = sin(coreg_b * theta)
                pt = array([j * (k * ex[0] + l * ey[0]) + m * ez[0],
                            j * (k * ex[1] + l * ey[1]) + m * ez[1],
                            j * (k * ex[2] + l * ey[2]) + m * ez[2]])
                temp = sqrt(pt[0] * pt[0] + pt[1] * pt[1])

                lons_lo[scan, sample] = arctan2(pt[1], pt[0]) * deg
                lats_lo[scan, sample] = arctan2(pt[2], temp) * deg

        return lats_lo, lons_lo

    def ingest_amsr2(self):
        """
        Ingests AMSR2 data from the user specified path
        and converts to a common format for use further in the pipeline.

        Returns
        -------
        data_dict: dict
            Dictionary containing the data extracted from the HDF5 file.
        """

        data_dict, coreg_a, coreg_b, lats_89a, lons_89a, lats_89b, lons_89b = self.read_hdf5()

        if self.config.grid_type == "L1r":
            required_locations = ['source', 'target']
        else:
            required_locations = ['source']

        for band in required_locations:
            if getattr(self.config, f"{band}_band") == '89a':
                data_dict[f"lats_{band}"] = lats_89a
                data_dict[f"lons_{band}"] = lons_89a
            elif getattr(self.config, f"{band}_band") == '89b':
                data_dict[f"lats_{band}"] = lats_89b
                data_dict[f"lons_{band}"] = lons_89b
            else:
                # Extract BTs of Target Band using conversion algorithm from 89a channel.
                coreg_index = self.config.key_mappings[getattr(self.config, f"{band}_band")][0]
                lats, lons = self.amsr2_latlon_conversion(
                    coreg_a=coreg_a[coreg_index],
                    coreg_b=coreg_b[coreg_index],
                    lons_hi=lons_89a,
                    lats_hi=lats_89a
                )

                data_dict[f"lats_{band}"] = lats
                data_dict[f"lons_{band}"] = lons

        for variable in data_dict:
            data_dict[variable] = data_dict[variable].flatten('C')
        return data_dict

    def ingest_smap(self):
        """
        Ingests AMSR2 data from the user specified path
        and converts to a common format for use further in the pipeline.

        Returns
        -------
        data_dict: dict
            Dictionary containing the data extracted from the HDF5 file.
        """
        data_dict = self.read_hdf5()
        # qc_dict = self.extract_smap_qc(qc_dict)
        data_dict = self.clean_data(data_dict)
        # data_dict = self.apply_smap_qc(qc_dict, data_dict)
        return data_dict

    def ingest_data(self):
        """
        Initiates ingestion of user specified data type.

        Returns
        -------
        data_dict: dict
            Dictionary containing pre-processed data extracted from the HDF5 file.
        """
        if self.config.input_data_type == "AMSR2":
            return self.ingest_amsr2()
        if self.config.input_data_type == "SMAP":
            return self.ingest_smap()
