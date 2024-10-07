"""
This module contains the DataIngestion class, which is responsible for ingesting
and processing various types of satellite data. It supports various AMSR2, SMAP
and simulated CIMR data. Providing functionalities to read, clean, and convert
data into a standardized format for further analysis or processing in the pipeline.

The DataIngestion class utilizes configuration files to manage the ingestion
process.
"""

import re

from numpy import (array, sqrt, cos, pi, sin, zeros, arctan2, arccos, nan, tile, repeat, arange,
                   isnan, delete, where, concatenate, full, newaxis, float32, asarray, any)

from rgb_logging    import RGBLogging 
from config_file    import ConfigFile
from utils          import remove_overlap
from grid_generator import GRIDS, GridGenerator


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

    def __init__(self, config_object):
        """
        Initializes the DataIngestion class with the user specified configuration file.

        Parameters
        ----------
        config_file_path: str
            The path to the configuration file that contains the user specified parameters.
        """
        self.config = config_object
        self.logger = config_object.logger 


    def remove_out_of_bounds(self, data_dict):

        grid = GRIDS[self.config.grid_definition]

        x_bound_min = grid['x_min'] - 0.5 * grid['res']
        x_bound_max = grid['x_min'] + grid['n_cols']*grid['res'] + 0.5 * grid['res']
        y_bound_max = grid['y_max'] + 0.5 * grid['res']
        y_bound_min = grid['y_max'] - grid['n_rows']*grid['res'] - 0.5 * grid['res']


        source_x, source_y = GridGenerator(self.config).lonlat_to_xy(
            lon = data_dict['longitude'],
            lat = data_dict['latitude']
        )

        out_of_bound_inds = where((source_y < y_bound_min) | (source_y > y_bound_max))

        for variable in data_dict:
            data_dict[variable][out_of_bound_inds] = nan

        return data_dict


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

        import h5py as h5

        data_dict = {}

        with h5.File(self.config.input_data_path, 'r') as data:

            # Not using qc_dict at the moment need to add functionality for "qc remap".
            qc_dict = {}

            if self.config.input_data_type == "AMSR2":

                # Extract Metadata
                coreg_a   = self.amsr2_coreg_extraction(data.attrs['CoRegistrationParameterA1'][0])
                coreg_b   = self.amsr2_coreg_extraction(data.attrs['CoRegistrationParameterA2'][0])
                overlap   = int(data.attrs['OverlapScans'][0])
                num_scans = int(data.attrs['NumberOfScans'][0])
                self.config.num_scans = num_scans
                self.config.AM2_DEF_SNUM_HI = AM2_DEF_SNUM_HI
                self.config.AM2_DEF_SNUM_LOW = AM2_DEF_SNUM_LOW

                if not num_scans <= self.config.LMT:
                    raise ValueError(
                        f"Number of scans ({num_scans} exceeds limit of {self.config.LMT}."
                    )

                # Extract Data used for all re-grids
                # lats and lons of 89A are used to extrac the lats and lons for all other bands
                lats_89a    = remove_overlap(
                    array   = data['Latitude of Observation Point for 89A'][:],
                    overlap = overlap
                )

                lons_89a = remove_overlap(
                    array=data['Longitude of Observation Point for 89A'][:],
                    overlap=overlap
                )

                # Extract 89b data if required
                if '89b' in self.config.target_band or '89b' in self.config.source_band:
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

                # Extract BTs of all relevant bands
                bands_to_open = set(self.config.target_band + self.config.source_band)

                for band in bands_to_open:
                    variable_dict = {}
                    key = self.config.key_mappings[band][1]
                    bt_h, bt_v = data[key + 'H)'], data[key + 'V)']
                    bt_h_scale= bt_h.attrs['SCALE FACTOR']
                    bt_v_scale = bt_v.attrs['SCALE FACTOR']
                    variable_dict['bt_v'] = remove_overlap(bt_v, overlap) * bt_v_scale
                    variable_dict['bt_h'] = remove_overlap(bt_h, overlap) * bt_h_scale
                    data_dict[band] = variable_dict

                return data_dict, coreg_a, coreg_b, lats_89a, lons_89a, lats_89b, lons_89b

            if self.config.input_data_type == "SMAP":
                variable_dict = {}
                bt_data = data['Brightness_Temperature']
                spacecraft_data = data['Spacecraft_Data']

                # Extract variables
                required_variables = ['longitude', 'latitude', 'processing_scan_angle',
                                      'x_position', 'y_position', 'z_position',
                                      'sub_satellite_lat', 'sub_satellite_lon', 'x_velocity',
                                      'y_velocity', 'z_velocity', 'altitude']

                variables_to_open = set(required_variables + self.config.variables_to_regrid)

                for variable in variables_to_open:
                    variable_key = self.config.variable_key_map[variable]
                    if variable_key in bt_data:
                        variable_dict[variable] = bt_data[variable_key][:]
                    elif variable_key in spacecraft_data:
                        if variable == 'altitude':
                            # Only need the maximum altitude for ap_max_radius calculation
                            self.config.max_altitude = spacecraft_data[variable_key][:].max()
                            continue
                        variable_dict[variable] = spacecraft_data[variable_key][:]

                # Create a map between scan number, earth sample number and feed_horn number
                num_scans, num_samples = variable_dict['longitude'].shape
                variable_dict['scan_number'] = float32(repeat(arange(num_scans), num_samples).reshape(num_scans, num_samples))
                variable_dict['sample_number'] = float32(tile(arange(num_samples), (num_scans, 1)))
                variable_dict['feed_horn_number'] = float32(zeros((num_scans, num_samples)))

                # Add attitude variable

                # Turn the 1D (once per scan) variables into 2D variables
                for variable in variable_dict:
                    if len(variable_dict[variable].shape) == 1:
                        variable_dict[variable] = tile(variable_dict[variable], (num_samples, 1)).T

                # Remove out of bounds
                # 
                # [Note]: This method calls in the `generate_grid` method under the hood 
                variable_dict = self.remove_out_of_bounds(variable_dict)

                # Split Fore/Aft and Flatten
                if self.config.split_fore_aft == True:
                    variable_dict_out = {}
                    mask_dict = {'aft': (variable_dict['processing_scan_angle'] >= self.config.aft_angle_min) & (
                            variable_dict['processing_scan_angle'] <= self.config.aft_angle_max)}
                    mask_dict['fore'] = ~mask_dict['aft']

                    for variable in variable_dict:
                        for scan_direction in ['fore', 'aft']:
                            mask = mask_dict[scan_direction].flatten('C')
                            variable_dict_out[f"{variable}_{scan_direction}"] = variable_dict[variable].flatten('C')[mask]
                    variable_dict = variable_dict_out
                    del variable_dict_out
                    data_dict['L'] = variable_dict
                else:
                    for variable in variable_dict:
                        variable_dict[variable] = variable_dict[variable].flatten('C')
                    data_dict['L'] = variable_dict

                return data_dict

    @property
    def read_netcdf(self):
        """

        :return:
        """

        from netCDF4 import Dataset

        data_dict = {}
        if self.config.grid_type == "L1C":
            for band in self.config.target_band:
                with Dataset(self.config.input_data_path, 'r') as data:
                    band_data = data[band + '_BAND']
                    variable_dict = {}

                    # Extract Feed offsets and u, v to add to config
                    scan_angle_feeds_offsets = band_data['scan_angle_feeds_offsets'][:]
                    self.config.scan_angle_feed_offsets[band]=scan_angle_feeds_offsets
                    self.config.u0[band], self.config.v0[band] = getattr(band_data, 'uo'), getattr(band_data, 'vo')

                    # Extract variables (This can be tweaked to remove variables for Non-AP algorithms)
                    required_variables = ['longitude', 'latitude', 'processing_scan_angle',
                                      'x_position', 'y_position', 'z_position',
                                      'sub_satellite_lat', 'sub_satellite_lon', 'x_velocity',
                                      'y_velocity', 'z_velocity', 'attitude']

                    variables_to_open = set(required_variables + self.config.variables_to_regrid)

                    for variable in variables_to_open:
                        variable_key = self.config.variable_key_map[variable]
                        data = band_data[variable_key][:]
                        if variable == 'x_position':
                            variable_dict[variable] = data[:,:,0]
                        elif variable == 'y_position':
                            variable_dict[variable] = data[:,:,1]
                        elif variable == 'z_position':
                            variable_dict[variable] = data[:,:,2]
                        elif variable == 'x_velocity':
                            variable_dict[variable] = data[:,:,0]
                        elif variable == 'y_velocity':
                            variable_dict[variable] = data[:,:,1]
                        elif variable == 'z_velocity':
                            variable_dict[variable] = data[:,:,2]
                        else:
                            variable_dict[variable] = data

                    # Calculate max altitude for ap_radius calculation (same for all bands)
                    if not hasattr(self.config, 'max_altitude'):
                        altitude = sqrt(variable_dict['x_position']**2 + variable_dict['y_position']**2 + variable_dict['z_position']**2)
                        altitude -= 6371000
                        self.config.max_altitude = altitude.max()

                    # Create map between scan number and earth sample number
                    num_feed_horns = self.config.num_horns[band]
                    num_scans, num_samples = variable_dict['processing_scan_angle'].shape

                    # Combine Feed horns and Flatten
                    variable_dict = self.combine_cimr_feeds(variable_dict, num_feed_horns)
                    variable_dict['scan_number'] = float32(repeat(arange(num_scans)[:, newaxis], num_samples * num_feed_horns,axis=1).flatten('C'))
                    single_row = tile(arange(num_samples), num_feed_horns)
                    variable_dict['sample_number'] = float32(tile(single_row, (num_scans, 1)).flatten('C'))

                    # Remove out of bounds here
                    variable_dict = self.remove_out_of_bounds(variable_dict)

                    # Split Fore/Aft
                    if self.config.split_fore_aft == True:
                        mask_dict = {'aft': (variable_dict['processing_scan_angle'] >= self.config.aft_angle_min) & (
                                variable_dict['processing_scan_angle'] <= self.config.aft_angle_max)}
                        mask_dict['fore'] = ~mask_dict['aft']
                        variable_dict_out = {}
                        for variable in variable_dict:
                            for scan_direction in ['fore', 'aft']:
                                mask = mask_dict[scan_direction].flatten('C')
                                variable_dict_out[f"{variable}_{scan_direction}"] = variable_dict[variable][mask]

                        variable_dict = variable_dict_out
                        del variable_dict_out
                    data_dict[band] = variable_dict

        if self.config.grid_type == 'L1R':
            # Need to concatenate the target and source bands.
            # Actually I don't think that you need a separate statement here, we can just create a
            # "set" from target and source bands of all band data that needs to be opened.
            # then again, we only need the lats and lons of the source data.
            pass

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
            qc_dict[qc] = where(qc_dict[qc] == 0, 1, nan)
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
        for band in data_dict:
            variable_dict = data_dict[band]
            if self.config.split_fore_aft:
                # Get fore and aft array shapes
                fore_shape, aft_shape = None, None
                for variable in variable_dict:
                    if fore_shape is None and 'fore' in variable:
                        fore_shape = variable_dict[variable].shape
                    if aft_shape is None and 'aft' in variable:
                        aft_shape = variable_dict[variable].shape

                    if fore_shape is not None and aft_shape is not None:
                        break

                for scan_direction in ['fore', 'aft']:
                    if scan_direction == 'fore':
                        nan_map = zeros(fore_shape, dtype=bool)
                    else:
                        nan_map = zeros(aft_shape, dtype=bool)
                    for variable in variable_dict:
                        if scan_direction not in variable:
                            continue
                        else:
                            if self.config.input_data_type == 'SMAP':
                                # Include SMAP specific fill values
                                if variable_dict[variable].dtype == 'float32':
                                    variable_dict[variable][variable_dict[variable] == SMAP_FILL_FLOAT_32] = nan
                            elif self.config.input_data_type == 'CIMR':
                                # Include CIMR specific fil values
                                # Currently non nans as simulated data
                                pass
                            if variable_dict[variable].ndim == 1:
                                nan_map |= isnan(variable_dict[variable])
                            # Deadling with the attitude array
                            elif variable_dict[variable].ndim == 3:
                                # Needs testing when nans in cimr data
                                nan_map |= any(isnan(variable_dict[variable]), axis=(1, 2))

                    for variable in variable_dict:
                        if scan_direction not in variable:
                            continue
                        else:
                            variable_dict[variable] = delete(variable_dict[variable], where(nan_map)[0], axis=0)
                    data_dict[band] = variable_dict
            else:
                nan_map = zeros(next(iter(variable_dict.values())).shape, dtype=bool).flatten('C')
                for variable in variable_dict:
                    if self.config.input_data_type == 'SMAP':
                        if variable_dict[variable].dtype == 'float32':
                            variable_dict[variable][variable_dict[variable] == SMAP_FILL_FLOAT_32] = nan
                    elif self.config.input_data_type == 'CIMR':
                        # Include CIMR specific fil values
                        # Currently non nans as simulated data
                        pass
                    elif self.config.input_data_type == 'AMSR2':
                        # Include AMSR2 specific fill values
                        pass
                    nan_map |= isnan(variable_dict[variable])
                for variable in variable_dict:
                    variable_dict[variable] = delete(variable_dict[variable], where(nan_map)[0], axis=0)
                data_dict[band] = variable_dict
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
                    #print(f"amsr2latlon conversion: out of range warning:"
                    #      f"Latitude and/or longitude are"
                    #      f"out of range on Scan = {scan} and Sample = {sample}")
                    self.logger.warning(f"amsr2latlon conversion: out of range warning:"
                          f"Latitude and/or longitude are"
                          f"out of range on Scan = {scan} and Sample = {sample}")
                    lats_lo[scan, sample] = MV
                    lons_lo[scan, sample] = MV
                    continue

                # Conversion Calculation. Taken from AMSR2 Sample C programs.
                p1    = array([cos(lon_1 * rad) * cos(lat_1 * rad),
                               sin(lon_1 * rad) * cos(lat_1 * rad),
                               sin(lat_1 * rad)])

                p2    = array([cos(lon_2 * rad) * cos(lat_2 * rad),
                               sin(lon_2 * rad) * cos(lat_2 * rad),
                               sin(lat_2 * rad)])
                temp  = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
                theta = arccos(temp)
                ex    = p1
                temp  = sqrt(p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2]) * sqrt(
                     p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2]) * sin(theta)
                ez    = array([p1[1] * p2[2] - p1[2] * p2[1],
                               p1[2] * p2[0] - p1[0] * p2[2],
                               p1[0] * p2[1] - p1[1] * p2[0]]) / temp
                ey    = array([ez[1] * ex[2] - ez[2] * ex[1],
                               ez[2] * ex[0] - ez[0] * ex[2],
                               ez[0] * ex[1] - ez[1] * ex[0]])
                j     = cos(coreg_b * theta)
                k     = cos(coreg_a * theta)
                l     = sin(coreg_a * theta)
                m     = sin(coreg_b * theta)
                pt    = array([j * (k * ex[0] + l * ey[0]) + m * ez[0],
                               j * (k * ex[1] + l * ey[1]) + m * ez[1],
                               j * (k * ex[2] + l * ey[2]) + m * ez[2]])
                temp  = sqrt(pt[0] * pt[0] + pt[1] * pt[1])

                lons_lo[scan, sample] = arctan2(pt[1], pt[0]) * deg
                lats_lo[scan, sample] = arctan2(pt[2], temp) * deg

        return lats_lo, lons_lo


    def combine_cimr_feeds(self, variable_dict, num_feed_horns):
        """
        :param data_dict:
        :return:
        """

        # new variables: feed_horn_number, sample_number, scan_number
        for variable in variable_dict:
            data = variable_dict[variable]

            if len(data.shape) == 3:
                if variable != 'attitude':
                    try:
                        feed_horn_number
                    except NameError:
                        # Create Feed Horn number map
                        single_row = concatenate([full(data.shape[1], dim) for dim in range(data.shape[2])])
                        feed_horn_number = tile(single_row, (data.shape[0], 1))

                    data_out = zeros((data.shape[0], data.shape[1] * num_feed_horns))
                    for scan in range(data.shape[0]):
                            data_out[scan, :] = data[scan, :, :].flatten('F')

                    variable_dict[variable] = data_out.flatten('C')
                else:
                    # Need to preserve the 3rd dimension (matrix) for attitude data
                    data_out = repeat(data, num_feed_horns, axis=1)
                    variable_dict[variable] = asarray(data_out.reshape(data_out.shape[0]*data_out.shape[1], 1, data_out.shape[2]))
                    test =0

            elif len(data.shape) == 2:
                    data_out = zeros((data.shape[0], data.shape[1] * num_feed_horns))
                    for scan in range(data.shape[0]):
                        data_out[scan, :] = tile(data[scan, :], num_feed_horns)
                    variable_dict[variable] = data_out.flatten('C')

        variable_dict['feed_horn_number'] = float32(feed_horn_number.flatten('C'))

        return variable_dict


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

        # if self.config.grid_type == "L1R":
        #     required_locations = ['source', 'target']
        # else:
        #     required_locations = ['source']

        for band in data_dict:
            if band == '89a':
                data_dict[band]['longitude'] = lons_89a
                data_dict[band]['latitude'] = lats_89a
            elif band == '89b':
                data_dict[band]['longitude'] = lons_89b
                data_dict[band]['latitude'] = lats_89b
            else:
                # Extract BTs of Target Band using conversion algorithm from 89a channel.
                coreg_index = self.config.key_mappings[band][0]
                lats, lons = self.amsr2_latlon_conversion(
                    coreg_a=coreg_a[coreg_index],
                    coreg_b=coreg_b[coreg_index],
                    lons_hi=lons_89a,
                    lats_hi=lats_89a
                )

                data_dict[band]['longitude'] = lons
                data_dict[band]['latitude'] = lats

        # Create map between scan number and earth sample number
        for band in data_dict:
            num_scans, num_samples = data_dict[band]['longitude'].shape
            data_dict[band]['scan_number'] = float32(repeat(arange(num_scans), num_samples).reshape(num_scans, num_samples))
            data_dict[band]['sample_number'] = float32(tile(arange(num_samples), (num_scans, 1)))

        # Remove out of bounds inds
        for band in data_dict:
            data_dict[band] = self.remove_out_of_bounds(data_dict[band])

        # Flatten data
        for band in data_dict:
            for variable in data_dict[band]:
                data_dict[band][variable] = data_dict[band][variable].flatten('C')

        # Clean Data
        data_dict = self.clean_data(data_dict)

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

        self.logger.info("read_hdf5")

        # Retrieving Data 
        tracked_func  = RGBLogging.rgb_decorated(
            decorate  = self.config.logpar_decorate, 
            decorator = RGBLogging.track_perf, 
            logger    = self.logger 
            )(self.read_hdf5) 

        data_dict = tracked_func() 

        #data_dict = self.read_hdf5()

        # Cleaning Data
        tracked_func  = RGBLogging.rgb_decorated(
            decorate  = self.config.logpar_decorate, 
            decorator = RGBLogging.track_perf, 
            logger    = self.logger 
            )(self.clean_data) 
        data_dict = tracked_func(data_dict) 
        #data_dict = self.clean_data(data_dict)

        # qc_dict = self.extract_smap_qc(qc_dict)
        # data_dict = self.apply_smap_qc(qc_dict, data_dict)

        return data_dict


    def ingest_cimr(self):
        """

        :return:
        """
        # Open netcdf file
        data_dict = self.read_netcdf
        data_dict = self.clean_data(data_dict)

        # Apply QC as and when
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

        if self.config.input_data_type == "CIMR":
            return self.ingest_cimr()
