from numpy.core.multiarray import unravel_index

from grid_generator import GridGenerator, GRIDS
from ap_processing import AntennaPattern
from  numpy import where, nan, take, full, all, sum, zeros, identity, dot, nansum
from numpy.linalg import inv
from tqdm import tqdm

# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class BGInterp:
    def __init__(self, config):
        self.config = config

    def get_antenna_patterns(self, band, longitude, latitude, x_pos, y_pos, z_pos, x_vel, y_vel,
                             z_vel, processing_scan_angle, feed_horn_number, attitude, target_lon, target_lat):


        # Initiate Source Pattern
        source_ap = AntennaPattern(config=self.config,
                                   band=band)


        # Make integration grid
        int_dom_lons, int_dom_lats = source_ap.make_integration_grid(
            longitude=longitude,
            latitude=latitude
        )

        # Project source patterns to grid
        source_ant_patterns = []
        for sample in range(longitude.shape[0]):
            sample_pattern=source_ap.antenna_pattern_to_earth(
                int_dom_lons=int_dom_lons,
                int_dom_lats=int_dom_lats,
                x_pos=x_pos[sample],
                y_pos=y_pos[sample],
                z_pos=z_pos[sample],
                x_vel=x_vel[sample],
                y_vel=y_vel[sample],
                z_vel=z_vel[sample],
                processing_scan_angle=processing_scan_angle[sample],
                feed_horn_number=feed_horn_number[sample],
                attitude=attitude[sample],
                lon_l1b = longitude[sample],
                lat_l1b = latitude[sample]
            )
            sample_pattern /= sum(sample_pattern)
            source_ant_patterns.append(sample_pattern)

        # Project target patterns to grid
        if self.config.grid_type == 'L1R':
            target_ap = AntennaPattern(config=self.config,
                                       band=self.config.target_band)

            # At this point, we also need the source data.



        else:
            target_ant_pattern = source_ap.target_gaussian(int_dom_lons,
                                                int_dom_lats,
                                                target_lon,
                                                target_lat)
            target_ant_pattern /= sum(target_ant_pattern)

        # Checking the shift factor, there shouldnt be an array where they are all zero
        for count, pattern in enumerate(source_ant_patterns):
            if all(pattern == 0):
                print(f"Antenna pattern not succesfully projected to grid! - DEBUG!")

        return source_ant_patterns, target_ant_pattern

    def BG(self, band, samples_dict, target_grid, longitude,
           latitude, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, processing_scan_angle,
           feed_horn_number, attitude):

        indexes = samples_dict['indexes']
        fill_value = len(longitude)
        weights = full((indexes.shape[0], indexes.shape[1]), nan)

        for target_cell in tqdm(range(indexes.shape[0])):

            # Getting the target lon, lat
            if self.config.grid_type == 'L1C':
                # I get two different results here for target_lon, target_lat. They should be the same
                # I suspect that there may be an issue with the conversion functions in the grid generator.
                # THIS SHOULD BE LOOKED INTO AND VERIFIED
                # MAYBE THE UNRAVEL INDEX SHOULD ACTUALLY BE DONE USING SCAN AND SAMPLE NUMBERS
                target_row, target_col = unravel_index(samples_dict['grid_1d_index'][target_cell], (GRIDS[self.config.grid_definition]['n_rows'], GRIDS[self.config.grid_definition]['n_cols']))
                target_lon, target_lat = GridGenerator(self.config).rowcol_to_lonlat(target_row, target_col)

                # target_lon_test, target_lat_test = (target_grid[0].flatten('C')[samples_dict['grid_1d_index'][target_cell]],
                #                                     target_grid[1].flatten('C')[samples_dict['grid_1d_index'][target_cell]])

            elif self.config.grid_type == 'L1R':
                # Scan Geometry, is it still unravel index? I think so
                target_lon, target_lat = (target_grid[0][samples_dict['grid_1d_index'][target_cell]],
                                          target_grid[1][samples_dict['grid_1d_index'][target_cell]])


            # Get Antenna Patterns
            samples = indexes[target_cell, :]
            input_samples = samples[samples != fill_value]
            # print(f"input_samples: {input_samples}")
            source_ant_patterns, target_ant_pattern = self.get_antenna_patterns(
                band=band,
                longitude=longitude[input_samples],
                latitude=latitude[input_samples],
                x_pos=x_pos[input_samples],
                y_pos=y_pos[input_samples],
                z_pos=z_pos[input_samples],
                x_vel=x_vel[input_samples],
                y_vel=y_vel[input_samples],
                z_vel=z_vel[input_samples],
                processing_scan_angle=processing_scan_angle[input_samples],
                feed_horn_number=feed_horn_number[input_samples],
                attitude=attitude[input_samples],
                target_lon = target_lon,
                target_lat = target_lat
            )

            # BG algorithm
            num_input_samples = len(input_samples)
            g = zeros((num_input_samples, num_input_samples))
            v = zeros(num_input_samples)
            u = zeros(num_input_samples)

            for i in range(num_input_samples):
                u[i] = sum(source_ant_patterns[i])
                v[i] = sum(source_ant_patterns[i] * target_ant_pattern)
                for j in range(num_input_samples):
                    g[i, j] = sum(source_ant_patterns[i] * source_ant_patterns[j])

            k = 0. # Regularisation Factor
            g = g + k*identity(num_input_samples)
            ginv = inv(g)

            # Weights
            a = ginv@(v + (1 - u.T @ (ginv @ v)) / (u.T @ (ginv @ u)) * u)
            weights[target_cell, :len(input_samples)] = a

        return weights

    def get_weights(self, band, samples_dict, variable_dict, target_grid, scan_direction):

        if scan_direction:
            scan_direction = f"_{scan_direction}"
        else:
            scan_direction = ""

        if f"attitude{scan_direction}" in variable_dict:
            attitude = variable_dict[f"attitude{scan_direction}"]
        else:
            attitude = full(variable_dict[f"longitude{scan_direction}"].shape, None)

        weights = self.BG(
            band=band,
            samples_dict=samples_dict,
            target_grid=target_grid,
            longitude=variable_dict[f"longitude{scan_direction}"],
            latitude=variable_dict[f"latitude{scan_direction}"],
            x_pos=variable_dict[f'x_position{scan_direction}'],
            y_pos=variable_dict[f'y_position{scan_direction}'],
            z_pos=variable_dict[f'z_position{scan_direction}'],
            x_vel=variable_dict[f'x_velocity{scan_direction}'],
            y_vel=variable_dict[f'y_velocity{scan_direction}'],
            z_vel=variable_dict[f'z_velocity{scan_direction}'],
            processing_scan_angle=variable_dict[f'processing_scan_angle{scan_direction}'],
            feed_horn_number=variable_dict[f'feed_horn_number{scan_direction}'],
            attitude=attitude
        )

        return weights

    def interp_variable_dict(self, samples_dict, variable_dict, target_grid, scan_direction=None, band=None, **args):

        # This gets opened twice on split fore/aft, maybe changes this to be more efficient. could add to config
        # ap = AntennaPattern(
        #     config=self.config,
        #     band=band
        # )

        weights = self.get_weights(
            band=band,
            samples_dict=samples_dict,
            variable_dict=variable_dict,
            target_grid=target_grid,
            scan_direction=scan_direction
        )

        variable_dict_out = {}
        for variable in variable_dict:
            # Check if you want to regrid this variable
            if variable.removesuffix(f"_{scan_direction}") not in self.config.variables_to_regrid:
                continue

            # Apply weights to the variable you want to regrid
            fill_value = len(variable_dict[variable])

            # Combining BG with samples
            indexes = samples_dict['indexes']
            mask = (indexes == fill_value)
            vars = full(indexes.shape, nan)
            vars[~mask] = variable_dict[variable][indexes[~mask]]
            vars_out = nansum(weights*vars, axis=1)
            variable_dict_out[variable] = vars_out

        return variable_dict_out














