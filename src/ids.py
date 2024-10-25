from numpy import take, where, nan, nansum

class IDSInterp:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_weights(distances):
        dist_sq = distances ** 2
        weights = 1 / dist_sq
        return weights

    def IDS(self, samples_dict, variable):
        distances = samples_dict['distances']
        indexes_out = samples_dict['indexes']

        # Get IDS weights (only needs to be calculated once, also fore/aft)
        weights = self.get_weights(distances)

        # Get variable data
        max_index = len(variable) - 1
        valid_indices_mask = indexes_out < max_index
        extracted_values = take(variable, indexes_out.clip(0, max_index))
        extracted_values = where(valid_indices_mask, extracted_values, nan)

        # Apply IDS
        output_temp = extracted_values * weights
        output_temp = nansum(output_temp, axis =1)/ nansum(weights, axis=1)

        return output_temp

    def interp_variable_dict(self, samples_dict, variable_dict, target_grid=None, scan_direction=None, band=None):

        variable_dict_out = {}
        for variable in variable_dict:
            if scan_direction:
                if scan_direction not in variable:
                    continue
                elif variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                else:
                    print(variable)
                    variable_dict_out[variable] = self.IDS(
                        samples_dict=samples_dict,
                        variable=variable_dict[variable]
                    )
            else:
                if variable not in self.config.variables_to_regrid:
                    # Check this
                    continue
                else:
                    variable_dict_out[variable] = self.IDS(
                        samples_dict=samples_dict,
                        variable=variable_dict[variable]
                    )

        return variable_dict_out