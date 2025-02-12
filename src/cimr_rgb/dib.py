from numpy import take, isfinite, where, nan, nanmean, clip, where


class DIBInterp:

    def __init__(self, config):
        self.config = config


    @staticmethod
    def DIB(samples_dict, variable):

        """
        Returns the interpolated value on the target points.

        Parameters:
            samples_dict (dictionary of arrays with shape (# target points, self.config.max_neighbours)):
                'distances': distance of a target point to the nearest neighbours
                'indexes': index of the nearest neighbours in the flattened array of source points
                'grid_1d_index': index of the target point in the flattened array of target points
            variable (1D array_like): values of the variable to be regridded on the source points

        Returns:
            array with shape (# target points, 1): values of the regridded variables on the target data points
        """

        indexes_out = samples_dict['indexes']

        # Extract values from variable using indexes and masking
        max_index = len(variable)-1
        valid_indices_mask = indexes_out < max_index
        extracted_values = take(variable, clip(indexes_out, 0, max_index))
        extracted_values = where(valid_indices_mask, extracted_values, nan)

        # DIB takes the average of all samples considered
        average_values = nanmean(extracted_values, axis=1)

        return average_values


    def interp_variable_dict(self, samples_dict, variable_dict, target_grid, scan_direction=None, band=None, **args):

        """
        Returns the interpolated value on the target points, for all variables.

        Parameters:
            samples_dict (dictionary of arrays with shape (# target points, self.config.max_neighbours)):
                'distances': distance of a target point to the nearest neighbours
                'indexes': index of the nearest neighbours in the flattened array of source points
                'grid_1d_index': index of the target point in the flattened array of target points
            variable_dict (dictionary of arrays with shape (# source points, 1)): values of the variable to be regridded
                keys are L1b variables names in the source data (with no suffix if split_fore_aft=True, otherwise with either _fore or _after suffix)
                values are 1d arrays with the values of variables on the source points
            target_grid (None): not used
            scan_direction (None or str): None if split_fore_aft=False, eith 'fore' or 'aft' if split_fore_aft=True
            band (None): not used

        Returns:
            dictionary of arrays with shape (# target points, 1): 
                keys are the names of the regridded variables (with no suffix if split_fore_aft=True, otherwise with either _fore or _after suffix)
                values are 1d arrays with the interpolated values of regridded variables on the target points         
        """

        variable_dict_out = {}
        for variable in variable_dict:
            if scan_direction:
                if scan_direction not in variable:
                    continue
                elif variable.removesuffix(f'_{scan_direction}') not in self.config.variables_to_regrid:
                    continue
                else:
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
