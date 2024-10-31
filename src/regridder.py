import numpy as np
from numpy import meshgrid, isinf, where, full, nan, unravel_index, all, sqrt

from nn import NNInterp
from ids import IDSInterp
from dib import DIBInterp
from bg import BGInterp
from rsir import rSIRInterp
from grid_generator import GridGenerator, GRIDS
from pyresample import kd_tree, geometry

import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ReGridder:
    def __init__(self, config):
        self.config = config
        self.algos = {
            'NN': lambda band=None: NNInterp(self.config),
            'DIB': lambda band=None: DIBInterp(self.config),
            'IDS': lambda band=None: IDSInterp(self.config),
            'BG': lambda band: BGInterp(self.config, band),
            'RSIR': lambda band: rSIRInterp(self.config, band)
        }

    def get_algorithm(self, algorithm_name, band=None):
        algo = self.algos.get(algorithm_name)
        return algo(band)

    # Check if you need the full data dict once the function is complete
    def get_grid(self, data_dict=None):
        # Get target grid
        if self.config.grid_type == 'L1C':
            target_x, target_y, res = GridGenerator(self.config,
                                                    projection_definition=self.config.projection_definition,
                                                    grid_definition=self.config.grid_definition).generate_grid_xy(
                return_resolution=True
            )
            x_shape = len(target_x)
            y_shape = len(target_y)
            target_x, target_y = meshgrid(target_x, target_y)
            target_x = target_x.flatten()
            target_y = target_y.flatten()

            target_lon, target_lat = GridGenerator(self.config,
                                                   projection_definition=self.config.projection_definition,
                                                   grid_definition=self.config.grid_definition
                                                   ).xy_to_lonlat(
                x=target_x,
                y=target_y
            )

            # Rehshape to original matrix
            target_lon = target_lon.reshape(y_shape, x_shape)
            target_lat = target_lat.reshape(y_shape, x_shape)
            target_grid = [target_lon, target_lat]

        elif self.config.grid_type == 'L1R':
            # For the generation of a target grid in L1r, we just needs the lats and lons of
            # the target band.
            target_band = self.config.target_band
            target_lon = data_dict[target_band[0]]['longitude']
            target_lat = data_dict[target_band[0]]['latitude']
            target_grid = [target_lon, target_lat]

        return target_grid

    def get_neighbours(self, source_lon, source_lat, target_lon, target_lat, search_radius, neighbours):

        source_def = geometry.SwathDefinition(
            lons=source_lon,
            lats=source_lat
        )

        if self.config.grid_type == 'L1C':
            target_def = geometry.GridDefinition(
                lons=target_lon,
                lats=target_lat)

        elif self.config.grid_type == 'L1R':
            target_def = geometry.SwathDefinition(
                lons=target_lon,
                lats=target_lat
            )

        valid_input_index, valid_output_index, index_array, distance_array = kd_tree.get_neighbour_info(
            source_geo_def=source_def,
            target_geo_def=target_def,
            neighbours=neighbours,
            radius_of_influence=search_radius
        )

        if distance_array.ndim == 1:
            inf_mask = ~isinf(distance_array)
            reduced_distance = distance_array[inf_mask]
            reduced_index = index_array[inf_mask]
            # Original indices is the original 1D location of the point in the EASE grid
            original_indices = where(inf_mask)[0]

        elif distance_array.ndim == 2:
            inf_mask = ~all(np.isinf(distance_array), axis=1)
            reduced_distance = distance_array[inf_mask]
            reduced_index = index_array[inf_mask]
            original_indices = where(inf_mask)[0]

        # Remember that valid_output_index could be necessary for something in the future
        return reduced_distance, reduced_index, original_indices, valid_input_index

    # We need data dict or just pass specific variables?
    def get_l1c_samples(self, variable_dict, target_grid): # also need to change the name of this for l1r

        samples_dict = {}

        search_radius = self.config.search_radius
        neighbours = self.config.max_neighbours

        samples_dict = {}
        if self.config.split_fore_aft:

            for scan_direction in ['fore', 'aft']:
                target_lon = target_grid[0]
                target_lat = target_grid[1]
                source_lon = variable_dict[f"longitude_{scan_direction}"]
                source_lat = variable_dict[f"latitude_{scan_direction}"]

                reduced_distance, reduced_index, original_indices, valid_input_index = self.get_neighbours(
                    source_lon=source_lon,
                    source_lat=source_lat,
                    target_lon=target_lon,
                    target_lat=target_lat,
                    search_radius=search_radius,
                    neighbours=neighbours
                )
                samples_dict_temp = {}
                samples_dict_temp['distances'] = reduced_distance
                samples_dict_temp['indexes'] = reduced_index
                samples_dict_temp['grid_1d_index'] = original_indices
                samples_dict_temp['valid_input_index'] = valid_input_index
                samples_dict[scan_direction] = samples_dict_temp

        else:
            # Dont split fore/aft
            target_lon = target_grid[0]
            target_lat = target_grid[1]
            source_lon = variable_dict["longitude"]
            source_lat = variable_dict["latitude"]

            reduced_distance, reduced_index, original_indices, valid_input_index = self.get_neighbours(
                source_lon=source_lon,
                source_lat=source_lat,
                target_lon=target_lon,
                target_lat=target_lat,
                search_radius=search_radius,
                neighbours=neighbours
            )

            samples_dict = {}
            samples_dict['distances'] = reduced_distance
            samples_dict['indexes'] = reduced_index
            samples_dict['grid_1d_index'] = original_indices
            samples_dict['valid_input_index'] = valid_input_index

        return samples_dict

    def sample_selection(self, variable_dict, target_grid=None):

        samples_dict = self.get_l1c_samples(variable_dict, target_grid)

        for variable in variable_dict:
            if 'fore' in variable:
                variable_dict[variable] = variable_dict[variable][samples_dict['fore']['valid_input_index']]
            elif 'aft' in variable:
                variable_dict[variable] = variable_dict[variable][samples_dict['aft']['valid_input_index']]
            else:
                variable_dict[variable] = variable_dict[variable][samples_dict['valid_input_index']]

        # Remove valid_input_index as we don't use it again, and reduce_grid_inds won't
        # work if its inside (this can be changed if it turns out we need it)
        if self.config.split_fore_aft:
            del samples_dict['fore']['valid_input_index']
            del samples_dict['aft']['valid_input_index']
        else:
            del samples_dict['valid_input_index']

        if self.config.reduced_grid_inds:
            if self.config.split_fore_aft:
                samples_dict['fore'] = self.reduce_grid_inds(samples_dict['fore'])
                samples_dict['aft'] = self.reduce_grid_inds(samples_dict['aft'])
            else:
                samples_dict = self.reduce_grid_inds(samples_dict)

        return samples_dict, variable_dict

    def reduce_grid_inds(self, samples_dict):

        if self.config.grid_type == 'L1C':
            grid_shape = GRIDS[self.config.grid_definition]['n_rows'], GRIDS[self.config.grid_definition]['n_cols']
        elif self.config.grid_type == 'L1R':
            grid_shape = self.config.scan_geometry[self.config.target_band[0]]

        rows_out, cols_out = unravel_index(samples_dict['grid_1d_index'], grid_shape)
        row_min = self.config.reduced_grid_inds[0]
        row_max = self.config.reduced_grid_inds[1]
        col_min = self.config.reduced_grid_inds[2]
        col_max = self.config.reduced_grid_inds[3]
        indices = where((rows_out >= row_min) & (rows_out <= row_max) &
                        (cols_out >= col_min) & (cols_out <= col_max))[0]
        filtered_samples_dict = {key: value[indices] for key, value in samples_dict.items()}

        return filtered_samples_dict

    def create_output_grid_inds(self, grid_1d_index):
        # Can be unified
        # Also am I making a 3D output variables for number of horns ?
        if self.config.grid_type == 'L1C':
            grid_shape = GRIDS[self.config.grid_definition]['n_rows'], GRIDS[self.config.grid_definition]['n_cols']
            # Do we not need scan and earth sample number here? becuase grid_1D index is referenced to the
            # non cleaned up input samples.
            row, col = unravel_index(grid_1d_index, grid_shape)
        elif self.config.grid_type == 'L1R':
            # Now we need to build the grid from the original size of the swath
            # from the input product, where do we have this shape saved?
            grid_shape = self.config.num_target_scans, self.config.num_target_samples
            # Im not sure grid_1D index is valid because it is referring to the shape once things have been removed
            # as of yet I dont think anything is being removed, but if I were to for example apply quality control
            # the size of the input array would change.
            row, col = unravel_index(grid_1d_index, grid_shape)

        return row, col

    def regrid_l1c(self, data_dict): # Need to change the name of this.

        # Get target grid
        if self.config.grid_type == 'L1C':
            target_grid = self.get_grid()
            target_dict = None
        elif self.config.grid_type == 'L1R':
            target_grid = self.get_grid(data_dict)
            target_dict = data_dict[self.config.target_band[0]]

        data_dict_out = {}
        for band in data_dict:
            if band not in self.config.source_band and self.config.grid_type == 'L1R':
                continue
            print(f"Regridding band: {band}")

            variable_dict = data_dict[band]
            samples_dict, variable_dict = self.sample_selection(variable_dict, target_grid)

            if self.config.split_fore_aft:
                variable_dict_out = {}
                for scan_direction in ['fore', 'aft']:
                    print(scan_direction)

                    # Create an argument dictionary for interp_variable_dict, not all the algorithms need
                    # all the arguments.
                    args = {
                        'samples_dict': samples_dict[scan_direction],
                        'variable_dict': variable_dict,
                        'target_dict': target_dict,
                        'target_grid': target_grid,
                        'scan_direction': scan_direction,
                        'band': band
                    }

                    # Regrid Variables
                    algorithm = self.get_algorithm(algorithm_name=self.config.regridding_algorithm,
                                                   band=band)
                    variable_dict_out[scan_direction] = algorithm.interp_variable_dict(**args)

                    # Add cell_row and cell_col indexes
                    cell_row, cell_col = self.create_output_grid_inds(grid_1d_index=samples_dict[scan_direction]['grid_1d_index'])
                    variable_dict_out[scan_direction][f'cell_row_{scan_direction}'] = cell_row
                    variable_dict_out[scan_direction][f'cell_col_{scan_direction}'] = cell_col

                # Combine fore and aft variables into single dictionary
                combined_dict = {**variable_dict_out['fore'], **variable_dict_out['aft']}
                variable_dict_out = combined_dict
            else:
                # Don't split fore/aft

                # Create an argument dictionary for interp_variable_dict, not all the algorithms need
                # all the arguments.
                args = {
                    'samples_dict': samples_dict,
                    'variable_dict': variable_dict,
                    'target_dict': data_dict[self.config.target_band[0]],
                    'target_grid': target_grid,
                    'scan_direction': None,
                    'band': band
                }

                # Regrid Variables
                algorithm = self.get_algorithm(algorithm_name=self.config.regridding_algorithm,
                                               band=band)
                variable_dict_out = algorithm.interp_variable_dict(**args)

                # Add cell_row and cell_col indexes
                cell_row, cell_col = self.create_output_grid_inds(samples_dict['grid_1d_index'])
                variable_dict_out['cell_row'] = cell_row
                variable_dict_out['cell_col'] = cell_col

            data_dict_out[band] = variable_dict_out
            print(f"Finished regridding band: {band}")

        return data_dict_out








