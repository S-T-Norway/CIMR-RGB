import pytest
import numpy as np
import cimr_rgb.nn as nn


variable1 = dict()
samples1  = dict()
exp1      = dict()
variable1['varA'] = np.array([0.1, 0.7, 1. , 0.2, 0.5])
variable1['varB'] = np.array([1.3, 8.4, 9.7, 9.3, 6.1])
variable1['varC'] = np.array([4.9, 0.9, 4.5, 2.4, 3.1])
samples1['indexes']   = np.array([4, 3, 0, 0, 3])
exp1['varA'] = np.array([0.5, 0.2, 0.1, 0.1, 0.2])                                                           


@pytest.mark.parametrize(
    ("samples_dict", "variable_dict", "expected"),
    [
        pytest.param(samples1, variable1, exp1, id='nominal_case')
    ]
)
def test_nn(mocker, samples_dict, variable_dict, expected):

    config = mocker.patch("cimr_rgb.config_file.ConfigFile").return_value
    config.variables_to_regrid = expected.keys()

    interpolator = nn.NNInterp(config)
    variable_dict_out = interpolator.interp_variable_dict(samples_dict, variable_dict, None)

    for key in config.variables_to_regrid:
        assert (expected[key] == variable_dict_out[key]).all(), f"Variable {key}: expected {expected[key]}, got {variable_dict_out[key]}"

