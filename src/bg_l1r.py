from grid_generator import GridGenerator, GRIDS
from ap_processing import AntennaPattern
from  numpy import where, nan, take, full, all, sum, zeros, identity, dot, nansum, unravel_index
from numpy.linalg import inv
from tqdm import tqdm

# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class BGInterp:
    def __init__(self, config):
        self.config = config


    def get_antenna_patterns(self, band, variable_dict, target_dict, target_lon, target_lat, source_inds, target_inds):

        # Initiate Source Pattern
        source_ap = AntennaPattern(config=self.config,
                                   band=band,
                                   antenna_method = self.config.source_antenna_method,
                                   polarisation_method = self.config.polarisation_method,
                                   antenna_threshold=self.config.source_antenna_threshold,
                                   gaussian_params=self.config.source_gaussian_params)

        # Make integration grid
        int_dom_lons, int_dom_lats = source_ap.make_integration_grid(
            longitude=variable_dict['longitude'][source_inds],
            latitude=variable_dict['latitude'][source_inds]
        )

        # Project source patterns to grid
        source_ant_patterns = []
        for sample in source_inds: # CHECK SAMPLE
            sample_pattern=source_ap.antenna_pattern_to_earth(
                int_dom_lons=int_dom_lons,
                int_dom_lats=int_dom_lats,
                x_pos=variable_dict['x_position'][sample],
                y_pos=variable_dict['y_position'][sample],
                z_pos=variable_dict['z_position'][sample],
                x_vel=variable_dict['x_velocity'][sample],
                y_vel=variable_dict['y_velocity'][sample],
                z_vel=variable_dict['z_velocity'][sample],
                processing_scan_angle=variable_dict['processing_scan_angle'][sample],
                feed_horn_number=variable_dict['feed_horn_number'][sample],
                attitude=variable_dict['attitude'][sample],
                lon_l1b = variable_dict['longitude'][sample],
                lat_l1b = variable_dict['latitude'][sample]
            )
            sample_pattern /= sum(sample_pattern)
            source_ant_patterns.append(sample_pattern)

        # Get target patterns
        target_ant_pattern = None
        if self.config.grid_type == 'L1R':

            target_ap = AntennaPattern(config=self.config,
                                       band=self.config.target_band[0],
                                       antenna_method=self.config.target_antenna_method,
                                       polarisation_method=self.config.polarisation_method,
                                       antenna_threshold=self.config.target_antenna_threshold,
                                       gaussian_params=self.config.target_gaussian_params)

            target_ant_pattern = target_ap.antenna_pattern_to_earth(
                int_dom_lons=int_dom_lons,
                int_dom_lats=int_dom_lats,
                x_pos=target_dict['x_position'][target_inds],
                y_pos=target_dict['y_position'][target_inds],
                z_pos=target_dict['z_position'][target_inds],
                x_vel=target_dict['x_velocity'][target_inds],
                y_vel=target_dict['y_velocity'][target_inds],
                z_vel=target_dict['z_velocity'][target_inds],
                processing_scan_angle=target_dict['processing_scan_angle'][target_inds],
                feed_horn_number=target_dict['feed_horn_number'][target_inds],
                attitude=target_dict['attitude'][target_inds],
                lon_l1b=target_lon,
                lat_l1b=target_lat
            )
            target_ant_pattern /= sum(target_ant_pattern)

        elif self.config.grid_type == 'L1C':
            pass
            # What to do here.

        return source_ant_patterns, target_ant_pattern

    def BG(self, band, samples_dict, variable_dict, target_dict, target_grid):

        indexes = samples_dict['indexes']
        fill_value = len(variable_dict['longitude'])
        weights = full((indexes.shape[0], indexes.shape[1]), nan)

        for target_cell in tqdm(range(indexes.shape[0])):

            # Getting the target lon, lat
            if self.config.grid_type == 'L1C':
                target_lon, target_lat = (target_grid[0].flatten('C')[samples_dict['grid_1d_index'][target_cell]],
                                                    target_grid[1].flatten('C')[samples_dict['grid_1d_index'][target_cell]])

            elif self.config.grid_type == 'L1R':
                target_lon, target_lat = (target_grid[0][samples_dict['grid_1d_index'][target_cell]],
                                          target_grid[1][samples_dict['grid_1d_index'][target_cell]])

            # Get Antenna Patterns
            samples = indexes[target_cell, :]
            input_samples = samples[samples != fill_value]
            # print(f"input_samples: {input_samples}")
            source_ant_patterns, target_ant_pattern = self.get_antenna_patterns(
                band=band,
                variable_dict=variable_dict,
                target_dict=target_dict,
                target_lon = target_lon,
                target_lat = target_lat,
                source_inds=input_samples,
                target_inds=samples_dict['grid_1d_index'][target_cell]
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

    def get_weights(self, band, samples_dict, variable_dict, target_dict, target_grid, scan_direction):

        # Preparing variable_dict
        if scan_direction:
            scan_direction = f"_{scan_direction}"
        else:
            scan_direction = ""

        # Adding a empty attitude variable for SMAP
        if f"attitude{scan_direction}" not in variable_dict:
            variable_dict[f'attitude{scan_direction}'] = full(variable_dict[f"longitude{scan_direction}"].shape, None)

        variable_dict={key.removesuffix(f'{scan_direction}'): value for key, value in variable_dict.items()}

        weights = self.BG(
            band=band,
            samples_dict=samples_dict,
            variable_dict=variable_dict,
            target_dict=target_dict,
            target_grid=target_grid,
        )

        return weights

    def interp_variable_dict(self, **kwargs):


        if self.config.grid_type == 'L1C':
            samples_dict = kwargs['samples_dict']
            variable_dict = kwargs['variable_dict']
            target_dict = None
            target_grid = kwargs['target_grid']
            scan_direction = kwargs['scan_direction']
            band = kwargs['band']

        elif self.config.grid_type == 'L1R':
            samples_dict = kwargs['samples_dict']
            variable_dict = kwargs['variable_dict']
            target_dict = kwargs['target_dict']
            target_grid = kwargs['target_grid']
            scan_direction = kwargs['scan_direction']
            band = kwargs['band']

        if scan_direction is not None:
            variable_dict = {key: value for key, value in variable_dict.items() if key.endswith(scan_direction)}

        weights = self.get_weights(
            band=band,
            samples_dict=samples_dict,
            variable_dict=variable_dict,
            target_dict=target_dict,
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














