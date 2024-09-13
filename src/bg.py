from ap_processing import AntennaPattern
from  numpy import where, nan, take

# ---- Testing ----
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class BGInterp:
    def __init__(self, config):
        self.config = config



    def get_antenna_patterns(self, ap, longitude, latitude, x_pos, y_pos, z_pos, x_vel, y_vel,
                             z_vel, processing_scan_angle, feed_horn_number, attitude):

        # Make integration grid
        int_dom_lons, int_dom_lats = ap.make_integration_grid(
            longitude=longitude,
            latitude=latitude
        )

        # Project patterns to grid
        ant_patterns = []
        for sample in range(longitude.shape[0]):
            sample_pattern=ap.antenna_pattern_to_earth(
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
                attitude=attitude[sample]
            )
            ant_patterns.append(sample_pattern)

        return ant_patterns

    # Remove variable_dict and add only the variables that we need (x, y, z, nadir etc)
    def get_weights(self, samples_dict, ap, longitude, latitude,
                    x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, processing_scan_angle):
        pass

    def BG(self, variable, samples_dict, ap, longitude,
           latitude, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, processing_scan_angle,
           feed_horn_number, attitude):

        indexes = samples_dict['indexes']
        # Fill value
        fill_value = len(longitude)
        # Get antenna patterns
        # Get weights
        # Apply weights to variable
        for target_cell in range(indexes.shape[0]):
            print(target_cell)
            samples = indexes[target_cell, :]
            filtered_samples = samples[samples != fill_value]
            ant_patterns = self.get_antenna_patterns(
                ap=ap,
                longitude=longitude[filtered_samples],
                latitude=latitude[filtered_samples],
                x_pos=x_pos[filtered_samples],
                y_pos=y_pos[filtered_samples],
                z_pos=z_pos[filtered_samples],
                x_vel=x_vel[filtered_samples],
                y_vel=y_vel[filtered_samples],
                z_vel=z_vel[filtered_samples],
                processing_scan_angle=processing_scan_angle[filtered_samples],
                feed_horn_number=feed_horn_number[filtered_samples],
                attitude=attitude[filtered_samples]
            )

        return None


    def interp_variable_dict(self, samples_dict, variable_dict, scan_direction=None, band=None):

        ap = AntennaPattern(
            config=self.config,
            band=band
        )


        variable_dict_out = {}
        for variable in variable_dict:
            print(variable)
            if scan_direction:
                if scan_direction not in variable:
                    continue
                elif variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                else:
                    # Calculate the weights if not already calculated
                    if not hasattr(self, f"weights_{scan_direction}"):
                        # Only need attitude for CIMR
                        if f"attitude_{scan_direction}" in variable_dict:
                            attitude = variable_dict[f"attitude_{scan_direction}"]
                        else:
                            attitude = None

                        self.BG(
                            variable=variable_dict[variable],
                            samples_dict=samples_dict,
                            ap=ap,
                            longitude=variable_dict[f"longitude_{scan_direction}"],
                            latitude=variable_dict[f"latitude_{scan_direction}"],
                            x_pos=variable_dict[f'x_position_{scan_direction}'],
                            y_pos=variable_dict[f'y_position_{scan_direction}'],
                            z_pos=variable_dict[f'z_position_{scan_direction}'],
                            x_vel=variable_dict[f'x_velocity_{scan_direction}'],
                            y_vel=variable_dict[f'y_velocity_{scan_direction}'],
                            z_vel=variable_dict[f'z_velocity_{scan_direction}'],
                            processing_scan_angle=variable_dict[f'processing_scan_angle_{scan_direction}'],
                            feed_horn_number=variable_dict[f'feed_horn_number_{scan_direction}'],
                            attitude = attitude
                        )
















