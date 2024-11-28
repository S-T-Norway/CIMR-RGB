from numpy import column_stack
from numpy import zeros_like, concatenate, array, exp, sin, meshgrid,nanmean, cos, ravel_multi_index, ogrid, unravel_index, where, nan, take, full, all, sum, zeros, identity, dot, nansum, nan_to_num, sqrt
from tqdm import tqdm
from pyresample import kd_tree, geometry
from scipy.spatial import KDTree

# ---- Testing ----
# import matplotlib
# tkagg = matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.ion()

from .ap_processing  import AntennaPattern, GaussianAntennaPattern, make_integration_grid
from .grid_generator import GridGenerator, GRIDS
from .ids import IDSInterp



class rSIRInterp:
    def __init__(self, config, band):
        self.config = config
        self.band = band

        if self.config.source_antenna_method == 'gaussian':
            self.source_ap = GaussianAntennaPattern(config=self.config,
                                                    antenna_threshold=self.config.source_antenna_threshold)

        else:
            self.source_ap = AntennaPattern(config=self.config,
                                            band=self.band,
                                            antenna_method=self.config.source_antenna_method,
                                            polarisation_method=self.config.polarisation_method,
                                            antenna_threshold=self.config.source_antenna_threshold,
                                            gaussian_params=self.config.source_gaussian_params)

        if self.config.grid_type == 'L1R':

            if self.config.target_antenna_method == 'gaussian':
                self.target_ap = GaussianAntennaPattern(config=self.config,
                                                        antenna_threshold=self.config.target_antenna_threshold)

            else:
                self.target_ap = AntennaPattern(config=self.config,
                                                band=self.config.target_band[0],
                                                antenna_method=self.config.target_antenna_method,
                                                polarisation_method=self.config.polarisation_method,
                                                antenna_threshold=self.config.target_antenna_threshold,
                                                gaussian_params=self.config.target_gaussian_params)

        else:  # L1c
            self.target_ap = GaussianAntennaPattern(config=self.config,
                                                    antenna_threshold=0.001)  # check if this value makes sense for L1c

        self.ids_weights = None

    def get_antenna_patterns(self, band, variable_dict, target_dict, target_lon, target_lat, source_inds, target_inds, target_cell_size):

        pattern_lons = array(variable_dict['longitude'][source_inds])
        pattern_lats = array(variable_dict['latitude'][source_inds])
        pattern_radii = []

        if self.config.source_antenna_method == 'gaussian':
            sigmax = self.config.source_gaussian_params[0]
            sigmay = self.config.source_gaussian_params[1]
            pattern_radii = concatenate((pattern_radii, [self.source_ap.estimate_max_ap_radius(sigmax, sigmay)]))
        else:
            max_radii = [self.source_ap.max_ap_radius[int(nn)] for nn in variable_dict['feed_horn_number'][source_inds]]
            pattern_radii = concatenate((pattern_radii, max_radii))

        pattern_lons = concatenate((pattern_lons, [target_lon]))
        pattern_lats = concatenate((pattern_lats, [target_lat]))

        if self.config.grid_type == 'L1C':
            sigmax = target_cell_size[0]
            sigmay = target_cell_size[1]
            pattern_radii = concatenate((pattern_radii, [self.target_ap.estimate_max_ap_radius(sigmax, sigmay)]))
        elif self.config.target_antenna_method == 'gaussian':
            sigmax = self.config.target_gaussian_params[0]
            sigmay = self.config.target_gaussian_params[1]
            pattern_radii = concatenate((pattern_radii, [self.target_ap.estimate_max_ap_radius(sigmax, sigmay)]))
        else:
            pattern_radii = concatenate((pattern_radii, [self.target_ap.max_ap_radius[int(target_dict['feed_horn_number'][target_inds])]]))


        # Make integration grid
        int_dom_lons, int_dom_lats = make_integration_grid(
            int_projection_definition=self.config.MRF_projection_definition,
            int_grid_definition=self.config.MRF_grid_definition,
            longitude=pattern_lons,
            latitude=pattern_lats,
            ap_radii=pattern_radii
        )

        # Project source patterns to grid
        source_ant_patterns = []
        for sample in source_inds:
            if self.config.source_antenna_method == 'gaussian':
                sample_pattern = self.source_ap.antenna_pattern_to_earth(
                    int_dom_lons=int_dom_lons,
                    int_dom_lats=int_dom_lats,
                    lon_nadir=variable_dict['sub_satellite_lon'][sample],
                    lat_nadir=variable_dict['sub_satellite_lat'][sample],
                    lon_l1b=variable_dict['longitude'][sample],
                    lat_l1b=variable_dict['latitude'][sample],
                    sigmax=self.config.source_gaussian_params[0],
                    sigmay=self.config.source_gaussian_params[1]
                )

            else:
                sample_pattern = self.source_ap.antenna_pattern_to_earth(
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
                    lon_l1b=variable_dict['longitude'][sample],
                    lat_l1b=variable_dict['latitude'][sample]
                )
            sample_pattern /= sum(sample_pattern)
            source_ant_patterns.append(sample_pattern)

        # Get target patterns
        if self.config.grid_type == 'L1C':
            target_ant_pattern = self.target_ap.antenna_pattern_to_earth(
                int_dom_lons=int_dom_lons,
                int_dom_lats=int_dom_lats,
                lon_l1b=target_lon,
                lat_l1b=target_lat,
                sigmax=target_cell_size[0],
                sigmay=target_cell_size[1],
                alpha=0.
            )
        elif self.config.target_antenna_method == 'gaussian':
            target_ant_pattern = self.target_ap.antenna_pattern_to_earth(
                int_dom_lons=int_dom_lons,
                int_dom_lats=int_dom_lats,
                lon_nadir=target_dict['sub_satellite_lon'][target_inds],
                lat_nadir=target_dict['sub_satellite_lat'][target_inds],
                lon_l1b=target_lon,
                lat_l1b=target_lat,
                sigmax=self.config.target_gaussian_params[0],
                sigmay=self.config.target_gaussian_params[1]
            )
        else:
            target_ant_pattern = self.target_ap.antenna_pattern_to_earth(
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

        return source_ant_patterns, target_ant_pattern

    def rsir_algorithm(self, source_ant_patterns, target_ant_pattern, earth_samples, K=5):
        a0 = zeros_like(source_ant_patterns[0]).flatten()
        ant_sum = zeros_like(source_ant_patterns[0]).flatten()
        for count, ant_pattern in enumerate(source_ant_patterns):
            ant_pattern = nan_to_num(ant_pattern, nan=0.0)
            a0 += ant_pattern.flatten() * earth_samples[count]
            ant_sum += ant_pattern.flatten()
        a0 = a0/ant_sum

        for iteration in range(K):
            F = []
            D =[]
            U = []
            for count, ant_pattern in enumerate(source_ant_patterns):
                ant_pattern = nan_to_num(ant_pattern, nan=0.0)
                f_u = ant_pattern.flatten()*a0
                f_l = nansum(ant_pattern)
                f = nansum(f_u)/f_l
                d = sqrt(earth_samples[count]/f)

                if d>=1:
                    u = u = ((1/(2*f))*(1 - (1/d)) + (1/(a0*d)))**-1
                else:
                    u =  ((0.5*f))*(1 - d) + (a0 * d)

                F.append(f)
                D.append(d)
                U.append(u)

            # Update a
            a_update = zeros_like(a0).flatten()

            for sample_count, ant_pattern in enumerate(source_ant_patterns):
                ant_pattern = nan_to_num(ant_pattern, nan=0.0)
                a_update += ant_pattern.flatten() * U[sample_count]
            a_update = a_update.flatten()/ant_sum
            a0 = a_update

        t_out = nansum(target_ant_pattern.flatten() * a_update)

        return t_out

    def rsir(self, band, variable, samples_dict, variable_dict, target_dict, target_grid):

        indexes = samples_dict['indexes']
        fill_value = len(variable_dict[f"longitude"])

        T_out = []
        for target_cell in tqdm(range(indexes.shape[0])):

            # Getting the target lon, lat
            if self.config.grid_type == 'L1C':
                grid_area = GridGenerator(self.config, self.config.projection_definition,
                                          self.config.grid_definition).get_grid_area()

                target_lon, target_lat = (target_grid[0].flatten('C')[samples_dict['grid_1d_index'][target_cell]],
                                                    target_grid[1].flatten('C')[samples_dict['grid_1d_index'][target_cell]])

                cell_area = grid_area.flatten('C')[samples_dict['grid_1d_index'][target_cell]]
                resolution = sqrt(cell_area)
                target_cell_size = [resolution, resolution]

            elif self.config.grid_type == 'L1R':
                target_lon, target_lat = (target_grid[0][samples_dict['grid_1d_index'][target_cell]],
                                          target_grid[1][samples_dict['grid_1d_index'][target_cell]])
                target_cell_size = None

            # Get Antenna Patterns
            samples = indexes[target_cell, :]
            input_samples = samples[samples != fill_value]
            source_ant_patterns, target_ant_pattern = self.get_antenna_patterns(
                band=band,
                variable_dict=variable_dict,
                target_dict=target_dict,
                target_lon=target_lon,
                target_lat=target_lat,
                source_inds=input_samples,
                target_inds=samples_dict['grid_1d_index'][target_cell],
                target_cell_size=target_cell_size
            )

            t_out = self.rsir_algorithm(
                source_ant_patterns=source_ant_patterns,
                target_ant_pattern=target_ant_pattern,
                earth_samples= variable_dict[variable][input_samples],
                K = self.config.rsir_iteration
            )
            T_out.append(t_out)

        return array(T_out)

    def IDS(self, samples_dict, variable):
        # First approximation is to use IDS weights to calculate NEDT
        if self.ids_weights is None:
            distances = samples_dict['distances']
            self.ids_weights = IDSInterp(config=self.config).get_weights(distances=distances)
            weights = self.ids_weights
        else:
            weights = self.ids_weights

        # Get variable data
        indexes_out = samples_dict['indexes']
        max_index = len(variable) - 1
        valid_indices_mask = indexes_out < max_index
        extracted_values = take(variable, indexes_out.clip(0, max_index))
        extracted_values = where(valid_indices_mask, extracted_values, nan)

        # Apply IDS
        output_var = extracted_values * weights
        output_var = nansum(output_var, axis=1) / nansum(weights, axis=1)

        return output_var

    def get_nedt(self, samples_dict, nedt):

        if self.ids_weights is None:
            distances = samples_dict['distances']
            self.ids_weights = IDSInterp(config=self.config).get_weights(distances=distances)
            weights = self.ids_weights
        else:
            weights = self.ids_weights

        # Get NEDT data
        indexes_out = samples_dict['indexes']
        max_index = len(nedt) - 1
        valid_indices_mask = indexes_out < max_index
        extracted_values = take(nedt, indexes_out.clip(0, max_index))
        extracted_values = where(valid_indices_mask, extracted_values, nan)

        weights_sq = weights ** 2
        nedt = nansum(weights_sq * extracted_values, axis=1) / (nansum(weights, axis=1) ** 2)

        return nedt

    def interp_variable_dict(self, **kwargs):

        # Edit this once you know what you need
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

        # Preparing variable_dict
        if scan_direction is not None:
            variable_dict = {key: value for key, value in variable_dict.items() if key.endswith(scan_direction)}

        if scan_direction:
            scan_direction = f"_{scan_direction}"
        else:
            scan_direction = ""

        # Adding a empty attitude variable for SMAP
        if f"attitude{scan_direction}" not in variable_dict:
            variable_dict[f'attitude{scan_direction}'] = full(variable_dict[f"longitude{scan_direction}"].shape,
                                                              None)

        variable_dict={key.removesuffix(f'{scan_direction}'): value for key, value in variable_dict.items()}

        variable_dict_out = {}
        for variable in variable_dict:
            # Check if you want to regrid this variable
            if variable.removesuffix(f"{scan_direction}") not in self.config.variables_to_regrid:
                continue

            # rSIR only works  with Brightness Temperatures, we use IDS for the rest
            if variable.removesuffix(f"{scan_direction}") not in ['bt_h', 'bt_v', 'bt_3', 'bt_4']:
                print(variable)
                if 'nedt' in variable:
                    variable_dict_out[f"{variable}{scan_direction}"] = self.get_nedt(
                        samples_dict=samples_dict,
                        nedt=variable_dict[variable]
                    )
                else:
                    variable_dict_out[f"{variable}{scan_direction}"] = self.IDS(
                        samples_dict=samples_dict,
                        variable=variable_dict[variable]
                    )
            else:
                # Perform rSIR
                variable_dict_out[f"{variable}{scan_direction}"] = self.rsir(
                    band=band,
                    variable = variable,
                    samples_dict=samples_dict,
                    variable_dict=variable_dict,
                    target_dict=target_dict,
                    target_grid=target_grid,
                )

        return variable_dict_out
