from numpy import inf, isinf, where, full, nan, take

class NNInterp:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def NN(samples_dict, variable):
        values = take(variable, samples_dict['indexes'])
        return values

    def interp_variable_dict(self, samples_dict, variable_dict, scan_direction=None, band=None):

        variable_dict_out = {}
 
        for variable in variable_dict:

            if scan_direction:
                if scan_direction not in variable:
                    continue
                elif variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                else:
                    print(variable)
                    variable_dict_out[variable] = self.NN(
                        samples_dict=samples_dict,
                        variable=variable_dict[variable]
                    )
            else:
                if variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                else:
                    variable_dict_out[variable] = self.NN(
                        samples_dict=samples_dict,
                        variable=variable_dict[variable]
                    )

        return variable_dict_out

