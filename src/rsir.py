from numpy.lib.shape_base import column_stack
from requests.packages import target

from ap_processing import AntennaPattern
from  numpy import zeros_like, array, exp, sin, meshgrid,nanmean, cos, ravel_multi_index, ogrid,  unravel_index, where, nan, take, full, all, sum, zeros, identity, dot, nansum, nan_to_num, sqrt
from tqdm import tqdm
from grid_generator import GridGenerator, GRIDS
from pyresample import kd_tree, geometry
from scipy.spatial import KDTree

# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class rSIRInterp:
    def __init__(self, config, band):
        self.config = config
        self.band = band
        self.source_ap = AntennaPattern(config=self.config,
                                                        band=self.band,
                                                        antenna_method=self.config.source_antenna_method,
                                                        polarisation_method=self.config.polarisation_method,
                                                        antenna_threshold=self.config.source_antenna_threshold,
                                                        gaussian_params=self.config.source_gaussian_params)
        if self.config.grid_type == 'L1R':
            self.target_ap = AntennaPattern(config=self.config,
                                              band=self.config.target_band[0],
                                              antenna_method=self.config.target_antenna_method,
                                              polarisation_method=self.config.polarisation_method,
                                              antenna_threshold=self.config.target_antenna_threshold,
                                              gaussian_params=self.config.target_gaussian_params)
        else:
            target_ap = None


    def get_antenna_patterns(self, band, variable_dict, target_dict, target_lon, target_lat, source_inds, target_inds):
        target_ant_pattern = None
        target_idx = None

        # Make integration grid
        int_dom_lons, int_dom_lats = self.source_ap.make_integration_grid(
            longitude=variable_dict['longitude'][source_inds],
            latitude=variable_dict['latitude'][source_inds]
        )

        # Testing
        # plt.figure()
        # plt.scatter(int_dom_lons.flatten(), int_dom_lats.flatten())
        # plt.scatter(variable_dict['longitude'][source_inds], variable_dict['latitude'][source_inds])
        # plt.scatter(target_lon, target_lat)
        # plt.show()
        # plt.savefig('/home/beywood/Desktop/test/integration_grid.png')

        # Project source patterns to grid
        source_ant_patterns = []
        for sample in source_inds:  # CHECK SAMPLE
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
        if self.config.grid_type == 'L1R':

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

        elif self.config.grid_type == 'L1C':
            target_ant_pattern = None
            grid_coords = column_stack((int_dom_lons.flatten(), int_dom_lats.flatten()))
            tree = KDTree(grid_coords)
            _, target_idx = tree.query([target_lon, target_lat])

        return source_ant_patterns, target_ant_pattern, target_idx


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

        return a_update

    def rsir(self, band, variable, samples_dict, variable_dict, target_dict, target_grid):

        indexes = samples_dict['indexes']
        fill_value = len(variable_dict[f"longitude"])
        T_out = []
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
            source_ant_patterns, target_ant_pattern, target_idx = self.get_antenna_patterns(
                band=band,
                variable_dict=variable_dict,
                target_dict=target_dict,
                target_lon=target_lon,
                target_lat=target_lat,
                source_inds=input_samples,
                target_inds=samples_dict['grid_1d_index'][target_cell]
            )

            a = self.rsir_algorithm(
                source_ant_patterns=source_ant_patterns,
                target_ant_pattern=target_ant_pattern,
                earth_samples= variable_dict[variable][input_samples],
                K = self.config.rsir_iteration
            )

            if self.config.grid_type == 'L1R':
                t_out = nansum(target_ant_pattern.flatten() * a)
            else:
                if self.config.MRF_grid_definition == self.config.grid_definition:
                    t_out = a[target_idx]
                else:
                    # We need to perform some sort of binning. Add uncertainty at some point
                    out_res = GRIDS[self.config.grid_definition]['res']
                    mrf_res = GRIDS[self.config.MRF_grid_definition]['res']
                    scale = round(out_res/mrf_res)
                    a_flat = a
                    a = a.reshape(source_ant_patterns[0].shape)
                    target_y, target_x = unravel_index(target_idx, a.shape)
                    if scale % 2 == 0:
                        half_scale = int(scale/2)
                        # Dealing with the cases as the edge of the grid.
                        y_start = max(target_y - half_scale, 0)
                        y_end = min(target_y + half_scale, a.shape[0])
                        x_start = max(target_x - half_scale, 0)
                        x_end = min(target_x + half_scale, a.shape[1])
                        t_bin = a[y_start:y_end, x_start:x_end]
                    else:
                        half_scale = int((scale - 1) / 2)
                        y_start = max(target_y - half_scale, 0)
                        y_end = min(target_y + half_scale + 1, a.shape[0])
                        x_start = max(target_x - half_scale, 0)
                        x_end = min(target_x + half_scale + 1, a.shape[1])
                        t_bin = t_bin = a[y_start:y_end, x_start:x_end]

                t_out = nanmean(t_bin)
            T_out.append(t_out)
        return array(T_out)


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

            # rSIR only works  with Brightness Temperatures
            if variable.removesuffix(f"{scan_direction}") not in ['bt_h', 'bt_v', 'bt_3', 'bt_4']:
                continue

            variable_dict_out[variable] = self.rsir(
                band=band,
                variable = variable,
                samples_dict=samples_dict,
                variable_dict=variable_dict,
                target_dict=target_dict,
                target_grid=target_grid,
            )

        return variable_dict_out