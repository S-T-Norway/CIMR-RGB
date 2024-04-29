"""
This module contains the ReGridder class which is responsible regridding input data to a target
grid. A variety of regridding algorithms are possible, the choice of which is determined from
the configuration file.

The output of the ReGridder class is a dictionary containing the regridded data.
"""
from numpy import (sin, cos, deg2rad, unique, full, isnan, where, clip, nansum, nanmin, nan, zeros,
                   count_nonzero, arccos, nanmean)

from grid_generator import GridGenerator
from utils import interleave_bits, deinterleave_bits

MAP_EQUATORIAL_RADIUS = 6378137.0

class ReGridder:
    """
    This class is responsible for regridding input data to a target grid.

    Attributes:
    -----------
    config_object: xml.etree.ElementTree.Element
        Root of the configuration file

    Methods:
    --------
    initiate_grid(source_lon=None, source_lat=None, target_lon=None, target_lat=None):
        Making use of the grid generation module, this function creates the target grid for the
        data to be regridded to.

    assign_to_pixels(source_x, source_y, target_x, target_y):
        Assigns each input Earth sample (from source_x and source_y) to a pixel in the target
         grid.

    get_frequency(pixel_map, measurement_variable, x_shape, y_shape):
        Calculates the frequency of Earth samples in each pixel of the target grid.

    get_ids_weights(pixel_map, target_x, target_y, source_x, source_y):
        Calculates the weights for the Inverse Distance Squared (IDS) regridding algorithm.

    ids_interpolation(measurements, weights):
        Performs the Inverse Distance Squared (IDS) interpolation for a single grid cell.


    """

    def __init__(self, config_object):
        self.config = config_object
        self.regridding_algorithm = self.config.regridding_algorithm
        self.calculate_frequency = False

    def initiate_grid(self, source_lon=None, source_lat=None, target_lon=None, target_lat=None):  #
        """
        This function creates the target grid for the data to be regridded to.

        Parameters
        ----------
        source_lon: (np.ndarray of float)
            Longitudes of the source channel
        source_lat: (np.ndarray of float)
            Latitudes of the source channel
        target_lon: (np.ndarray of float)
            Longitudes of the target channel
        target_lat: (np.ndarray of float)
            Latitudes of the target channel

        Returns
        -------
        source_x: (np.ndarray of float)
            X-coordinates of the input Earth sample measurements in user defined grid/ projection
        source_y: (np.ndarray of float)
            Y-coordinates of the input Earth sample measurements in user defined grid/ projection
        target_x: (np.ndarray of float)
            X-coordinates of the center of a grid cell in the target grid defined from a given grid
            definition.
        target_y: (np.ndarray of float)
            Y-coordinates of the center of a grid cell in the target grid defined from a given grid
            definition.
        """
        if self.config.grid_type == 'L1c':
            source_x, source_y = GridGenerator(self.config).lonlat_to_xy(
                lon=target_lon,
                lat=target_lat
            )
            target_x, target_y, res = GridGenerator(self.config).generate_grid_xy(
                return_resolution=True)
            y_bound_min = nanmin(target_y) - res / 2
            y_bound_max = nanmin(target_y.max() + res / 2)
            out_of_bounds_inds = where((source_y < y_bound_min) | (source_y > y_bound_max))
            source_y[out_of_bounds_inds] = nan
            source_x[out_of_bounds_inds] = nan
            return source_x, source_y, target_x, target_y

        if self.config.grid_type == 'L1r':
            return source_lon, source_lat, target_lon, target_lat

    @staticmethod
    def assign_to_pixels(source_x, source_y, target_x, target_y):
        """
        This function assigns each input Earth sample (from source_x and source_y) to a pixel in
        the target grid.

        Parameters
        ----------
        source_x: (np.ndarray of float)
            X-coordinates of the input Earth sample measurements in user defined grid/ projection
        source_y: (np.ndarray of float)
            Y-coordinates of the input Earth sample measurements in user defined grid/ projection
        target_x: (np.ndarray of float)
            X-coordinates of the center of a grid cell in the target grid defined from a given grid
        target_y: (np.ndarray of float)
            Y-coordinates of the center of a grid cell in the target grid defined from a given grid

        Returns
        -------
        pixel_map: (np.ndarray of float)
            An array the same length as source_x and source_y, where each element is assigned a
            unique integer that corresponds to it assigned pixel in the target grid.

            The process by which the coordinates are assigned a unique integer (and vice versa)
            is called bit interleaving/de-interleaving. Functions for this purpose can be found
            in the utils module.
        """

        pixel_map = zeros(source_x.shape)
        for count, source_measurement in enumerate(source_x):
            if isnan(source_measurement):
                pixel_map[count] = nan
                continue
            x_distances = abs(target_x - source_measurement)
            y_distances = abs(target_y - source_y[count])
            x_index = where(x_distances == nanmin(x_distances))[0][0]
            y_index = where(y_distances == nanmin(y_distances))[0][0]
            pixel_map[count] = interleave_bits(int(x_index), int(y_index))
        return pixel_map

    @staticmethod
    def get_frequency(pixel_map, measurement_variable, x_shape, y_shape):
        """
        This function calculates the frequency of Earth samples in each pixel of the target grid.
        Parameters
        ----------
        pixel_map: (np.ndarray of float)
            An array the same length as source_x and source_y, where each element is assigned a
            unique integer that corresponds to it assigned pixel in the target grid.

            The process by which the coordinates are assigned a unique integer (and vice versa)
            is called bit interleaving/de-interleaving. Functions for this purpose can be found
            in the utils module.
        measurement_variable: (np.ndarray of float)
            The variable for which the cell sample frequency is desired.
        x_shape: (int)
            Shape of the output grid for which cell sample frequency is being calculated.
        y_shape: (int)
            Shape of the output grid for which cell sample frequency is being calculated.

        Returns
        -------
        frequency_out: (np.ndarray of float)
            An array of shape (y_shape, x_shape) where each element represents the number of
            Earth samples measured for the output grid.
        """
        frequency_out = full((y_shape, x_shape), nan)
        unique_key = unique(pixel_map)
        for unique_target_pixel in unique_key:
            if isnan(unique_target_pixel):
                continue
            x_target_ind, y_target_ind = deinterleave_bits(int(unique_target_pixel))
            frequency = where(pixel_map == unique_target_pixel)
            measurements = measurement_variable[frequency]
            frequency_out[y_target_ind, x_target_ind] = count_nonzero(~isnan(measurements))
        return frequency_out

    def get_ids_weights(self, pixel_map, target_x, target_y, source_x, source_y):
        """
        This function calculated the weights for the Inverse Distance Squared (IDS) regridding
        algorithm. The algorithm is applied for each grid cell on the target grid. hence there
        will be N weights per grid cell, where N is equal to the number of valid Earth samples
        landing in that grid cell.

        Parameters
        ----------
        pixel_map: (np.ndarray of float)
            An array the same length as source_x and source_y, where each element is assigned a
            unique integer that corresponds to it assigned pixel in the target grid.

            The process by which the coordinates are assigned a unique integer (and vice versa)
            is called bit interleaving/de-interleaving. Functions for this purpose can be found
            in the utils module.
        target_x: (np.ndarray of float)
            X-coordinates of the center of a grid cell in the target grid defined from a given grid.
        target_y: (np.ndarray of float)
            Y-coordinates of the center of a grid cell in the target grid defined from a given grid.
        source_x: (np.ndarray of float)
            X-coordinates of the input Earth sample measurements in user defined grid/ projection.
        source_y: (np.ndarray of float)
            Y-coordinates of the input Earth sample measurements in user defined grid/ projection.

        Returns
        -------
        IDS_weights: (dict)
            A dictionary where the keys are the unique integers assigned to each grid cell in the
            target grid (see description of pixel map variable, and the values are ndarrays of the
            weights for each Earth sample in that grid cell.
        """
        ids_weights = {}
        unique_key = unique(pixel_map)
        lam_target, phi_target = GridGenerator(self.config).xy_to_lonlat(
            x=target_x,
            y=target_y
        )
        lam_source, phi_source = GridGenerator(self.config).xy_to_lonlat(
            x=source_x,
            y=source_y
        )

        for unique_target_pixel in unique_key:
            if isnan(unique_target_pixel):
                continue
            frequency = where(pixel_map == unique_target_pixel)[0]
            x_target_ind, y_target_ind = deinterleave_bits(int(unique_target_pixel))
            target_lam, target_phi = lam_target[x_target_ind], phi_target[y_target_ind]
            source_lam, source_phi = lam_source[frequency], phi_source[frequency]
            rad_phi_source, rad_phi_target = deg2rad(source_phi), deg2rad(target_phi)
            rad_lam_diff = deg2rad(source_lam - target_lam)
            inner_calc = sin(rad_phi_source) * sin(rad_phi_target) + \
                         cos(rad_phi_source) * cos(rad_phi_target) * cos(rad_lam_diff)
            inner_calc = clip(inner_calc, -1, 1)
            epsilon = 1e-8
            inner_calc[inner_calc == -1] += epsilon  # For values clipped to -1, increase slightly
            inner_calc[inner_calc == 1] -= epsilon
            alphas = 1 / (MAP_EQUATORIAL_RADIUS * arccos(inner_calc)) ** 2
            ids_weights[unique_target_pixel] = alphas
        return ids_weights

    @staticmethod
    def ids_interpolation(measurements, weights):
        """
        This function performs the Inverse Distance Squared (IDS) interpolation for a single grid
        cell.

        Parameters
        ----------
        measurements: (np.ndarray of float)
            The measurements for a single grid cell.

        weights: (np.ndarray of float)
            The weights for each measurement in the grid cell.

        Returns
        -------
        measure_out: (float)
            The interpolated measurement for the grid cell.
        """
        measure_out = 0
        for count, measurement in enumerate(measurements):
            if isnan(weights[count]):
                continue
            measure_out += weights[count] * measurement
        measure_out = measure_out / nansum(weights)
        return measure_out

    def regrid_data(self, data_dict):
        """
        This function regrids the input variables present in data_dict to the target grid. The
        regridding algorithm used is determined by the configuration file. For all regridding
        methods, this function also output the cell frequency, also split into fore and aft.

        Parameters
        ----------
        data_dict: (dict)
            A dictionary containing the input data to be regridded. The keys are the variable names
            and the values are the data arrays.

        Returns
        -------
        data_dict_out: (dict)
            A dictionary containing the regridded data. The keys are the variable names with the
            scan direction appended to them, and the values are the regridded data arrays.
        """
        data_dict_out = {}
        symetric_diff = None
        mask_dict = {'aft': (data_dict['antenna_scan_angle'] >= 90) & (
                data_dict['antenna_scan_angle'] <= 270)}
        mask_dict['fore'] = ~mask_dict['aft']

        source_x, source_y, target_x, target_y = self.initiate_grid(
            target_lon=data_dict['lon_target'],
            target_lat=data_dict['lat_target']
        )

        pixel_map = self.assign_to_pixels(
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            target_y=target_y
        )

        if self.regridding_algorithm == 'IDS':
            # Make below a function
            ids_weights = {}
            for scan_direction in ['fore', 'aft']:
                mask = mask_dict[scan_direction]
                source_x_mask = where(mask, source_x, nan)
                source_y_mask = where(mask, source_y, nan)
                ids_weights[scan_direction] = self.get_ids_weights(
                    pixel_map=pixel_map,
                    target_x=target_x,
                    target_y=target_y,
                    source_x=source_x_mask,
                    source_y=source_y_mask
                )

        for var in data_dict:
            for scan_direction in ['fore', 'aft']:
                measurement_out = full((target_y.shape[0], target_x.shape[0]), nan)
                mask = mask_dict[scan_direction]

                if self.regridding_algorithm == 'IDS':
                    if 'bt' in var and 'nedt' not in var:
                        try:
                            data_dict_out['cell_frequency_' + scan_direction]

                        except KeyError:
                            data_dict_out['cell_frequency_' + scan_direction] = self.get_frequency(
                                pixel_map=pixel_map,
                                measurement_variable=data_dict[var] * mask,
                                x_shape=target_x.shape[0],
                                y_shape=target_y.shape[0])

                    weights = ids_weights[scan_direction]
                    for weight in weights:
                        if all(isnan(weights[weight])):
                            continue
                        measurement_inds = where(pixel_map == weight)[0]
                        measurements = data_dict[var][measurement_inds]
                        remapped_measurement = self.ids_interpolation(measurements, weights[weight])
                        x_out, y_out = deinterleave_bits(int(weight))
                        measurement_out[y_out, x_out] = remapped_measurement

                elif self.regridding_algorithm == 'NN':
                    if 'bt' in var and 'nedt' not in var:
                        try:
                            data_dict_out['cell_frequency_' + scan_direction]

                        except KeyError:
                            data_dict_out['cell_frequency_' + scan_direction] = self.get_frequency(
                                pixel_map=pixel_map,
                                measurement_variable=data_dict[var] * mask,
                                x_shape=target_x.shape[0],
                                y_shape=target_y.shape[0])

                    nn_distance = 'GreatCircle'
                    # NN should have Cartesian and NN options for regridding
                    unique_key = unique(pixel_map)
                    if nn_distance == 'GreatCircle':
                        lam_target, phi_target = GridGenerator(self.config).xy_to_lonlat(
                            x=target_x,
                            y=target_y
                        )
                        lam_source, phi_source = GridGenerator(self.config).xy_to_lonlat(
                            x=source_x,
                            y=source_y
                        )

                    for unique_target_pixel in unique_key:
                        if isnan(unique_target_pixel):
                            continue
                        measurement_inds = where(pixel_map == unique_target_pixel)[0]
                        measurements = data_dict[var][measurement_inds]
                        x_target_ind, y_target_ind = deinterleave_bits(int(unique_target_pixel))

                        if nn_distance == 'GreatCircle':
                            x_distances = abs(source_x[measurement_inds] - target_x[x_target_ind])
                            y_distances = abs(source_y[measurement_inds] - target_y[y_target_ind])
                            distances = (x_distances ** 2 + y_distances ** 2) ** 0.5
                            remap_ind = where(distances == nanmin(distances))[0][0]
                            remapped_measurement = measurements[remap_ind]
                            measurement_out[y_target_ind, x_target_ind] = remapped_measurement
                        if nn_distance == 'Euclidean':
                            target_lam, target_phi = lam_target[x_target_ind], phi_target[y_target_ind]
                            source_lam, source_phi = lam_source[measurement_inds], phi_source[measurement_inds]
                            rad_phi_source, rad_phi_target = deg2rad(source_phi), deg2rad(target_phi)
                            rad_lam_diff = deg2rad(source_lam - target_lam)
                            inner_calc = sin(rad_phi_source) * sin(rad_phi_target) + \
                                         cos(rad_phi_source) * cos(rad_phi_target) * cos(rad_lam_diff)
                            inner_calc = clip(inner_calc, -1, 1)
                            epsilon = 1e-8
                            inner_calc[inner_calc == -1] += epsilon  # For values clipped to -1, increase slightly
                            inner_calc[inner_calc == 1] -= epsilon
                            distances = arccos(inner_calc)
                            remap_ind = where(distances == nanmin(distances))[0][0]
                            remapped_measurement = measurements[remap_ind]
                            measurement_out[y_target_ind, x_target_ind] = remapped_measurement

                elif self.regridding_algorithm == 'DIB':
                    if 'bt' in var and 'nedt' not in var:
                        try:
                            data_dict_out['cell_frequency_' + scan_direction]

                        except KeyError:
                            data_dict_out['cell_frequency_' + scan_direction] = self.get_frequency(
                                pixel_map=pixel_map,
                                measurement_variable=data_dict[var] * mask,
                                x_shape=target_x.shape[0],
                                y_shape=target_y.shape[0])

                    unique_key= unique(pixel_map)
                    for unique_target_pixel in unique_key:
                        if isnan(unique_target_pixel):
                            continue
                        measurement_inds = where(pixel_map == unique_target_pixel)[0]
                        measurements = data_dict[var][measurement_inds]
                        x_target_ind, y_target_ind = deinterleave_bits(int(unique_target_pixel))
                        remapped_measurement = nanmean(measurements)
                        measurement_out[y_target_ind, x_target_ind] = remapped_measurement

                data_dict_out[var + '_' + scan_direction] = measurement_out

            if symetric_diff is None:
                fore_nan = isnan(data_dict_out[var + '_fore'])
                aft_nan = isnan(data_dict_out[var + '_aft'])
                unique_fore = (~fore_nan) & aft_nan
                unique_aft = fore_nan & (~aft_nan)
                symetric_diff = ~(unique_fore | unique_aft)

        for var in data_dict_out:
            data_dict_out[var] = where(symetric_diff, data_dict_out[var], nan)

        return data_dict_out

    def regrid_amsr2_data(self, data_dict):
        """
        TBD
        Parameters
        ----------
        data_dict

        Returns
        -------

        """
        source_lon, source_lat, target_lon, target_lat = self.initiate_grid(
            source_lon=data_dict['lons_source_band'],
            source_lat=data_dict['lats_source_band'],
            target_lon=data_dict['lon_target_band'],
            target_lat=data_dict['lat_target_band']
        )
        return source_lon, source_lat, target_lon, target_lat
