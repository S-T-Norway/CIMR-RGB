from numpy import inf, isinf, where, full, nan, take

class NNInterp:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger 

    @staticmethod
    def NN(samples_dict, variable):
        print(samples_dict['indexes'].shape)
        if len(samples_dict['indexes'].shape) != 1:
            samples_dict['indexes'] = samples_dict['indexes'][:, 0]
        values = take(variable, samples_dict['indexes'])
        return values


    def interp_variable_dict(self, samples_dict, variable_dict, target_grid, scan_direction=None, band=None, **args):

        variable_dict_out = {}
 
        for variable in variable_dict:

            if scan_direction:

                if scan_direction not in variable:
                    continue

                elif variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                
                else:
                    self.logger.info(variable)

                    variable_dict_out[variable] = self.NN(
                        samples_dict = samples_dict,
                        variable     = variable_dict[variable]
                    )
            else:
                if variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                else:
                    variable_dict_out[variable] = self.NN(
                        samples_dict = samples_dict,
                        variable     = variable_dict[variable]
                    )

        return variable_dict_out

