from ap_processing import AntennaPattern
from  numpy import zeros_like, unravel_index, where, nan, take, full, all, sum, zeros, identity, dot, nansum
from tqdm import tqdm
from grid_generator import GridGenerator, GRIDS

# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class rSIRInterp:
    def __init__(self, config):
        self.config = config

    def get_antenna_patterns(self, ap, longitude, latitude, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel,
                             processing_scan_angle, feed_horn_number, attitude, target_lon, target_lat):

        # Make integration grid
        int_dom_lons, int_dom_lats = ap.make_integration_grid(
            longitude=longitude,
            latitude=latitude
        )

        # Project source patterns to grid
        source_ant_patterns = []
        for sample in range(longitude.shape[0]):
            sample_pattern = ap.antenna_pattern_to_earth(
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
                lon_l1b=longitude[sample],
                lat_l1b=latitude[sample]
            )
            sample_pattern /= sum(sample_pattern)
            source_ant_patterns.append(sample_pattern)

        # Project target patterns to grid
        target_ant_pattern = ap.target_gaussian(int_dom_lons,
                                                int_dom_lats,
                                                target_lon,
                                                target_lat)
        target_ant_pattern /= sum(target_ant_pattern)

        # Checking the shift factor, there shouldnt be an array where they are all zero
        for count, pattern in enumerate(source_ant_patterns):
            if all(pattern == 0):
                print(f"Antenna pattern not succesfully projected to grid! - DEBUG!")

        return source_ant_patterns, target_ant_pattern

    def rsir_algorithm(self, source_ant_patterns, target_ant_pattern):
        a0 = zeros_like(target_ant_pattern).flatten()
        ant_sum = zeros_like(target_ant_pattern).flatten()


    def rsir(self,variable, ap, samples_dict, variable_dict, scan_direction):


        indexes = samples_dict['indexes']
        fill_value = len(variable_dict[f"longitude{scan_direction}"])

        for target_cell in tqdm(range(indexes.shape[0])):

            # Getting the target lon, lat
            target_row, target_col = unravel_index(samples_dict['grid_1d_index'][target_cell], (
            GRIDS[self.config.grid_definition]['n_rows'], GRIDS[self.config.grid_definition]['n_cols']))
            target_lon, target_lat = GridGenerator(self.config).rowcol_to_lonlat(target_row, target_col)

            # Get Antenna Patterns
            samples = indexes[target_cell, :]
            input_samples = samples[samples != fill_value]

            source_ant_patterns, target_ant_pattern = self.get_antenna_patterns(
                ap=ap,
                longitude=variable_dict[f'longitude{scan_direction}'][input_samples],
                latitude=variable_dict[f'latitude{scan_direction}'][input_samples] ,
                x_pos=variable_dict[f'x_position{scan_direction}'][input_samples],
                y_pos=variable_dict[f'y_position{scan_direction}'][input_samples],
                z_pos=variable_dict[f'z_position{scan_direction}'][input_samples],
                x_vel=variable_dict[f'x_velocity{scan_direction}'][input_samples],
                y_vel=variable_dict[f'y_velocity{scan_direction}'][input_samples],
                z_vel=variable_dict[f'z_velocity{scan_direction}'][input_samples],
                processing_scan_angle=variable_dict[f'processing_scan_angle{scan_direction}'][input_samples],
                feed_horn_number=variable_dict[f'feed_horn_number{scan_direction}'][input_samples],
                attitude=variable_dict[f'attitude{scan_direction}'][input_samples],
                target_lon=target_lon,
                target_lat=target_lat
            )


            T_out = self.rsir_algorithm(
                source_ant_patterns=source_ant_patterns,
                target_ant_pattern=target_ant_pattern,

            )


    def interp_variable_dict(self, samples_dict, variable_dict, scan_direction=None, band=None):

        if scan_direction:
            scan_direction = f"_{scan_direction}"
        else:
            scan_direction = ""

        ap = AntennaPattern(
            config=self.config,
            band=band
        )

        variable_dict_out = {}
        for variable in variable_dict:
            # Check if you want to regrid this variable
            if variable.removesuffix(f"{scan_direction}") not in self.config.variables_to_regrid:
                continue

            # rSIR only works  with Brightness Temperatures
            if variable.removesuffix(f"{scan_direction}") not in ['bt_h', 'bt_v', 'bt_3', 'bt_4']:
                continue

            variable_dict_out[variable] = self.rsir(
                variable = variable,
                ap = ap,
                samples_dict=samples_dict,
                variable_dict=variable_dict,
                scan_direction=scan_direction
            )