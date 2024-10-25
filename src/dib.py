from numpy import take, isfinite, where, nan, nanmean, clip, where


class DIBInterp:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def DIB(samples_dict, variable):
        indexes_out = samples_dict['indexes']

        # Extract values from variable using indexes and masking
        max_index = len(variable)-1
        valid_indices_mask = indexes_out < max_index
        extracted_values = take(variable, clip(indexes_out, 0, max_index))
        extracted_values = where(valid_indices_mask, extracted_values, nan)

        # DIB takes the average of all samples considered
        average_values = nanmean(extracted_values, axis=1)

        return average_values

    def interp_variable_dict(self, samples_dict, variable_dict, target_grid, scan_direction=None, band=None):

        variable_dict_out = {}
        for variable in variable_dict:
            if scan_direction:
                if scan_direction not in variable:
                    continue
                elif variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                else:
                    print(variable)
                    variable_dict_out[variable] = self.DIB(
                        samples_dict=samples_dict,
                        variable=variable_dict[variable]
                    )
            else:
                if variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                else:
                    variable_dict_out[variable] = self.DIB(
                        samples_dict=samples_dict,
                        variable=variable_dict[variable]
                    )
        return variable_dict_out