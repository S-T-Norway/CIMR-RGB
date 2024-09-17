"""
This module contains the ReGridder class which is responsible regridding input data to a target
grid. A variety of regridding algorithms are possible, the choice of which is determined from
the configuration file.

The output of the ReGridder class is a dictionary containing the regridded data.
"""
import numpy as np
from numpy import (sin, cos, deg2rad, unique, full, isnan, where, clip, nansum, nanmin, nan, zeros,
                   count_nonzero, arccos, nanmean, pad, array, float32, delete, column_stack, meshgrid, ndarray,
                   ravel_multi_index)
from scipy.spatial import KDTree
from os import path
import pickle
import sys
from tqdm import tqdm
from grid_generator import GridGenerator
from config_file import ConfigFile

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

    regrid_smap_data(data_dict):
        Todo

    regrid_amsr2_data(data_dict):
        Todo
    """

    def __init__(self, config_object, band_to_remap=None):
        self.config = config_object
        self.regridding_algorithm = self.config.regridding_algorithm
        self.num_scans, self.num_earth_samples = ConfigFile.get_scan_geometry(self.config, band_to_remap)

    def initiate_grid(self, data_dict):
        """
        FUNCTION HAS BEEN EDITED, NEED TO CHANGE DESCRIPTION
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
        out_of_bounds_inds: (np.ndarray of int)
            Indices of source_x and source_y that are out of bounds of the target grid.
        """
        if self.config.grid_type == 'L1C':
            target_lon = data_dict['lons_target']
            target_lat = data_dict['lats_target']

            source_x, source_y = GridGenerator(self.config).lonlat_to_xy(
                lon=target_lon,
                lat=target_lat
            )

            target_x, target_y, res = GridGenerator(self.config).generate_grid_xy(
                return_resolution=True
            )

            y_bound_min = nanmin(target_y) - res / 2
            y_bound_max = nanmin(target_y.max() + res / 2)
            out_of_bounds_inds = where((source_y < y_bound_min) | (source_y > y_bound_max))
            source_y = np.delete(source_y, out_of_bounds_inds)
            source_x = np.delete(source_x, out_of_bounds_inds)

            # Remove out of bounds variables from data_dict
            for variable in data_dict:
                if variable in getattr(self.config, 'variables_1d', []):
                    continue
                else:
                    data_dict[variable] = delete(data_dict[variable], out_of_bounds_inds, axis=0)

            return source_x, source_y, target_x, target_y, data_dict

    @staticmethod
    def assign_to_pixels(source_x, source_y, target_x, target_y, search_radius=None):
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

        if search_radius is None:
            pixel_map = zeros(source_x.shape)
            for count, source_measurement in enumerate(source_x):
                if isnan(source_measurement):
                    pixel_map[count] = nan
                    continue
                x_distances = abs(target_x - source_measurement)
                y_distances = abs(target_y - source_y[count])
                x_index = where(x_distances == nanmin(x_distances))[0][0]
                y_index = where(y_distances == nanmin(y_distances))[0][0]
                pixel_map[count] = ravel_multi_index((y_index, x_index), (1624, 3856))
        else:
            target_x, target_y = meshgrid(target_x, target_y)
            target_points = column_stack((target_x.flatten(), target_y.flatten()))
            source_points = column_stack((source_x, source_y))
            tree = KDTree(target_points)
            pixel_map = tree.query_ball_point(x=source_points, r=search_radius*1000)
            pixel_map = [list(arr) for arr in pixel_map]
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

        if isinstance(pixel_map, ndarray) is False:
            max_length = max(len(sublist) for sublist in pixel_map)
            padded_pixel_map = [sublist + [0] * (max_length - len(sublist)) for sublist in pixel_map]
            pixel_map = array(padded_pixel_map)

        unique_key = unique(pixel_map)
        for unique_target_pixel in tqdm(unique_key):
            if isnan(unique_target_pixel):
                continue
            y_target_ind, x_target_ind = np.unravel_index(int(unique_target_pixel), (1624, 3856))
            frequency = where(pixel_map == unique_target_pixel)[0]
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
        if isinstance(pixel_map, ndarray):
            unique_key = unique(pixel_map)
        else:
            flattened_list = [item for sublist in pixel_map for item in sublist]
            unique_key = unique(flattened_list)
            max_length = max(len(sublist) for sublist in pixel_map)
            padded_pixel_map = [sublist + [0] * (max_length - len(sublist)) for sublist in pixel_map]
            pixel_map = array(padded_pixel_map)

        lam_target, phi_target = GridGenerator(self.config).xy_to_lonlat(
            x=target_x,
            y=target_y
        )
        lam_source, phi_source = GridGenerator(self.config).xy_to_lonlat(
            x=source_x,
            y=source_y
        )

        frequency_out = full((target_y.shape[0], target_x.shape[0]), nan)
        for count, unique_target_pixel in enumerate(tqdm(unique_key)):
            frequency = where(pixel_map == unique_target_pixel)[0]
            source_lam, source_phi = lam_source[frequency], phi_source[frequency]

            non_nans = ~isnan(source_lam)
            source_lam = source_lam[non_nans]
            source_phi = source_phi[non_nans]
            frequency = frequency[non_nans]

            if source_lam.size == 0:
                continue
            y_target_ind, x_target_ind = np.unravel_index(int(unique_target_pixel), (
            1624, 3856))  # Need to base this shape parameter on the chosen grid.

            if len(frequency) == 1:
                ids_weights[unique_target_pixel] = 1
                frequency_out[y_target_ind, x_target_ind ] = 1
                continue


            target_lam, target_phi = lam_target[x_target_ind], phi_target[y_target_ind]
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
            # This is the cell_frequency variable
            frequency_out[y_target_ind, x_target_ind ] = len(alphas)
        return ids_weights, frequency_out

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
        if len(measurements) == 1:
            measure_out = measurements[0]
        else:
            measure_out = 0
            for count, measurement in enumerate(measurements): # Can skip this line if there is only one weight
                measure_out += weights[count] * measurement
            measure_out = measure_out / nansum(weights)
        return measure_out

    def regrid_l1c(self, data_dict):
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
        # --------------------------------------------------------------------------------
        if self.config.input_data_type == 'AMSR2':
            print('Functionality not yet included/tested for AMSR2')
        # --------------------------------------------------------------------------------

        data_dict_out = {}
        symetric_diff = None

        source_x, source_y, target_x, target_y, data_dict = self.initiate_grid(
            data_dict = data_dict
        )

        pixel_map = self.assign_to_pixels(
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            target_y=target_y,
            search_radius=self.config.search_radius
        )

        mask_dict = {'aft': (data_dict['antenna_scan_angle'] >= self.config.aft_angle_min) & (
                data_dict['antenna_scan_angle'] <= self.config.aft_angle_max)}
        mask_dict['fore'] = ~mask_dict['aft']

        if self.regridding_algorithm == 'IDS':

            ids_weights = {}

            for scan_direction in ['fore', 'aft']:
                mask = mask_dict[scan_direction]
                source_x_mask = where(mask, source_x, nan)
                source_y_mask = where(mask, source_y, nan)
                print(f"Calculating IDS Weights {scan_direction}")
                ids_weights[scan_direction], frequency_out = self.get_ids_weights(
                    pixel_map=pixel_map,
                    target_x=target_x,
                    target_y=target_y,
                    source_x=source_x_mask,
                    source_y=source_y_mask
                )
                data_dict_out['cell_frequency_' + scan_direction] = frequency_out
                print(f"Calculated IDS Weights {scan_direction}")
                break


        for var in data_dict:
            if hasattr(self.config, 'variables_1d'):
                # Do something if variables_1d exists
                if var in self.config.variables_1d:
                    continue
            else:
                if var in ['x_pos', 'y_pos', 'z_pos', 'sc_nadir_lon', 'sc_nadir_lat']:
                    continue
            if var == 'scan_number':
                continue
            for scan_direction in ['fore', 'aft']:
                print(f"Regridding {var} {scan_direction}")
                measurement_out = full((target_y.shape[0], target_x.shape[0]), nan)
                mask = mask_dict[scan_direction]

                if self.regridding_algorithm == 'IDS':

                    if isinstance(pixel_map, ndarray) is False: # I do this a few times, maybe I only need to do it once
                        max_length = max(len(sublist) for sublist in pixel_map)
                        padded_pixel_map = [sublist + [0] * (max_length - len(sublist)) for sublist in pixel_map]
                        pixel_map = array(padded_pixel_map)

                    weights = ids_weights[scan_direction]
                    print(f"Applying interpolation weights to {var}_{scan_direction}")
                    for weight in tqdm(weights):
                        measurement_inds = where(pixel_map == weight)[0]
                        measurements = where(mask, data_dict[var], nan)[measurement_inds]
                        measurements = measurements[~np.isnan(measurements)]
                        remapped_measurement = self.ids_interpolation(measurements, weights[weight])
                        y_out, x_out = np.unravel_index(int(weight), (1624, 3856))
                        measurement_out[y_out, x_out] = remapped_measurement
                    print(f"{var}_{scan_direction} re-gridded")



                elif self.regridding_algorithm == 'NN':
                    if 'bt' in var and 'nedt' not in var:
                        try:
                            data_dict_out['cell_frequency_' + scan_direction]

                        except KeyError:
                            data_dict_out['cell_frequency_' + scan_direction] = self.get_frequency(
                                pixel_map=pixel_map,
                                measurement_variable=where(mask, data_dict[var], nan),
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

                        if nn_distance == 'Euclidian':
                            x_distances = abs(source_x[measurement_inds] - target_x[x_target_ind])
                            y_distances = abs(source_y[measurement_inds] - target_y[y_target_ind])
                            distances = (x_distances ** 2 + y_distances ** 2) ** 0.5
                            remap_ind = where(distances == nanmin(distances))[0][0]
                            remapped_measurement = measurements[remap_ind]
                            measurement_out[y_target_ind, x_target_ind] = remapped_measurement
                        if nn_distance == 'GreatCircle':
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
                                measurement_variable=where(mask, data_dict[var], nan),
                                x_shape=target_x.shape[0],
                                y_shape=target_y.shape[0])

                    unique_key = unique(pixel_map)
                    for unique_target_pixel in unique_key:
                        if isnan(unique_target_pixel):
                            continue
                        measurement_inds = where(pixel_map == unique_target_pixel)[0]
                        measurements = data_dict[var][measurement_inds]
                        x_target_ind, y_target_ind = deinterleave_bits(int(unique_target_pixel))
                        remapped_measurement = nanmean(measurements)
                        measurement_out[y_target_ind, x_target_ind] = remapped_measurement

                data_dict_out[var + '_' + scan_direction] = measurement_out
                break




        #     if symetric_diff is None:
        #         fore_nan = isnan(data_dict_out[var + '_fore'])
        #         aft_nan = isnan(data_dict_out[var + '_aft'])
        #         unique_fore = (~fore_nan) & aft_nan
        #         unique_aft = fore_nan & (~aft_nan)
        #         symetric_diff = ~(unique_fore | unique_aft)
        #
        # for var in data_dict_out:
        #     data_dict_out[var] = where(symetric_diff, data_dict_out[var], nan)

        return data_dict_out

    def pre_calc_check(self, data_dict):
        weights_path = f"source_{self.config.source_band}_target_{self.config.target_band}_kernel_{self.config.kernel_size}.pickle"
        if path.isfile(path.join(self.config.dpr_path, 'pre_calc_coefficients', self.config.input_data_type, weights_path)):
            # Open pickle dictionary
            print(f"Loading pre-calculated weights from {weights_path}")
            with open(path.join(self.config.dpr_path,'pre_calc_coefficients', self.config.input_data_type, weights_path), 'rb') as handle:
                weights = pickle.load(handle)
        else:
            weights = self.get_amsr2_weights(data_dict)  # This needs updating to include CIMR/BG Weights
            with open(
                    path.join(self.config.dpr_path, 'pre_calc_coefficients', self.config.input_data_type, weights_path),
                    'wb') as handle:
                pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return weights

    def get_amsr2_weights(self, data_dict):
        num_scans = self.config.num_scans
        kernel_size = int(self.config.kernel_size)
        kernel_radius = int((kernel_size - 1) / 2)
        if '89' in self.config.target_band:
            earth_samples = self.config.AM2_DEF_SNUM_HI
        else:
            earth_samples = self.config.AM2_DEF_SNUM_LOW

        lons_source = deg2rad(data_dict['lons_source'])
        lats_source = deg2rad(data_dict['lats_source'])
        lons_target = deg2rad(data_dict['lons_target'])
        lats_target = deg2rad(data_dict['lats_target'])

        # Weights Calculation Algorithm
        weights = full(
            (((num_scans + (2 * kernel_radius)) * (earth_samples + (2 * kernel_radius))), kernel_size, kernel_size),
            fill_value=nan, dtype=float32)
        count = 0
        for row in range(lons_target.shape[0]):
            for col in range(lons_target.shape[1]):
                if isnan(lons_target[row, col]):
                    count += 1
                    continue
                row_window = slice(row - kernel_radius, row + kernel_radius + 1)
                col_window = slice(col - kernel_radius, col + kernel_radius + 1)

                inner_calc = (cos(lons_source[row_window, col_window] - lons_target[row, col]) *
                              cos(lats_target[row, col]) *
                              cos(lats_source[row_window, col_window]) +
                              sin(lats_source[row_window, col_window]) *
                              sin(lats_target[row, col]))

                distances = arccos(inner_calc)
                weights[count, :, :] = distances
                count += 1
        return weights

    def amsr2_ids_interpolation(self, measurements, weights):
        measure_out = nansum(weights* measurements)/nansum(weights)
        return measure_out

    def regrid_l1r(self, data_dict):
        """
        TBD
        Parameters
        ----------
        data_dict

        Returns
        -------

        """
        #--------------------------------------------------------------------------------
        if self.config.input_data_type == 'SMAP' or self.config.input_data_type == 'CIMR':
            print('Functionality not yet included/tested for SMAP or CIMR L1R regrid')
        # --------------------------------------------------------------------------------

        data_dict_out = {}

        kernel_size = int(self.config.kernel_size)
        kernel_radius = int((kernel_size - 1) / 2)

        for var in data_dict:
            earth_samples = int(len(data_dict[var]) / self.config.num_scans)
            data_dict[var] = data_dict[var].reshape(self.config.num_scans, earth_samples)
            pad_width = int((kernel_size - 1) / 2)
            data_dict[var] = pad(
                array=data_dict[var],
                pad_width=(pad_width,),
                mode='constant',
                constant_values=nan
            )

        # Check for pre-calculated weights, otherwise calculate them.
        weights = self.pre_calc_check(data_dict)

        # Apply the weights for the regrid convolution
        count = 0
        for var in data_dict:
            measurement_in = data_dict[var]
            measurements_out = full(measurement_in.shape, fill_value=nan, dtype=float32)

            for row in range(measurement_in.shape[0]):
                for col in range(measurement_in.shape[1]):
                    if isnan(measurement_in[row, col]):
                        count += 1
                        continue

                    row_window = slice(row - kernel_radius, row + kernel_radius + 1)
                    col_window = slice(col - kernel_radius, col + kernel_radius + 1)
                    measurements = measurement_in[row_window, col_window]
                    kernel_weights = weights[count]
                    measure_out = self.amsr2_ids_interpolation(measurements, kernel_weights)
                    measurements_out[row, col] = measure_out
                    count += 1

            data_dict_out[var] = measurements_out[kernel_radius:-kernel_radius, kernel_radius:-kernel_radius]
            break
        return data_dict_out



