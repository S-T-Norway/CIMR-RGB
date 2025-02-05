import pytest
import numpy as np
import matplotlib.pyplot as plt
import cimr_rgb.ap_processing as ap
import cimr_rgb.grid_generator as gg
import cimr_rgb.config_file as cf
import cimr_rgb.utils as utils


######################################################################
# test: vincenty_sphere_distance
######################################################################

@pytest.mark.parametrize(
    ("lon1, lat1, lon2, lat2, expected"),
    [
        pytest.param(0, 0, 0, 0, 0, id="same_point"),
        pytest.param(0, 0, 0, 1, 111195, id="_1deg_lat_difference"),
        pytest.param(-73.778691, 40.639928, -0.461388, 51.477500, 5539429, id="NYC_to_London"),
        pytest.param(0, 0, 180, 0, np.pi * ((6378137. + 6356752.)/2.), id="antipodal_points")
    ]
)
def test_vincenty_sphere_distance(lon1, lat1, lon2, lat2, expected):
    result = ap.vincenty_sphere_distance(lon1, lat1, lon2, lat2)
    assert np.isclose(result, expected, rtol=1e-3), f"Expected {expected}, got {result}"


######################################################################
# test: make_integration_grid
######################################################################


lon1 = [-0.04638499, -0.01526461, 0.01585578, 0.04697616]
lat1 = [0.03521133, 0.01167435, -0.01186263, -0.03539961]
lon1, lat1 = np.meshgrid(lon1, lat1)

lon2 = [179.79830863, 179.82942901, 179.86054939, 179.89166978, 179.92279016, 179.95391054, 179.98503093, -179.98443975, -179.95331937, -179.92219899, -179.89107861]
lat2 = [0.05874832, 0.03521133, 0.01167435, -0.01186263, -0.03539961, -0.0589366]
lon2, lat2 = np.meshgrid(lon2, lat2)

lon3 = [179.67302854, 179.76638948, 179.85975042, 179.95311136, -179.95331948, -179.85995853]
lat3 = [ 0.24717165, 0.17656039, 0.10594939, 0.03533855, -0.03527224, -0.10588308, -0.17649408, -0.24710533]
lon3, lat3 = np.meshgrid(lon3, lat3)

lon4 = np.array([[6.34019175, 7.30575953, 8.26717334],
                 [6.2344801 , 7.18426739, 8.13010235],
                 [6.1322225 , 7.06672972, 7.99747347]])
lat4 = np.array([[85.25575488, 85.24617289, 85.23524559],
                 [85.1755867 , 85.16616323, 85.15541593],
                 [85.09540007, 85.08612998, 85.0755569 ]])

lon5 = np.array([[-176.9872125 ,  176.9872125  ], 
                 [-176.63353934,  176.63353934],
                 [-176.18592517,  176.18592517],
                 [-175.60129465,  175.60129465],
                 [-174.80557109,  174.80557109],
                 [-173.65980825,  173.65980825],
                 [-171.86989765,  171.86989765],
                 [-168.69006753,  168.69006753],
                 [-161.56505118,  161.56505118],
                 [-135.        ,  135.        ],
                 [ -45.        ,   45.        ]])
lat5 = np.array([[86.9334288 , 86.9334288 ],
                 [87.25535221, 87.25535221],
                 [87.57711983, 87.57711983],
                 [87.89867364, 87.89867364],
                 [88.21991174, 88.21991174],
                 [88.54064028, 88.54064028],
                 [88.86044451, 88.86044451],
                 [89.17826187, 89.17826187],
                 [89.49038255, 89.49038255],
                 [89.7720928 , 89.7720928 ],
                 [89.7720928 , 89.7720928 ]])

#!!!! TEST EMPTY INTEGRATION GRID!! 

@pytest.mark.parametrize(
    ("int_projection_definition, int_grid_definition, longitude, latitude, ap_radii, expected"),
    [
        pytest.param("G", "EASE2_G3km", 0, 0, 3000, (lon1, lat1), id='regular_global_grid'),
        pytest.param("G", "EASE2_G3km", [179.8, 179.8, -179.9, -179.9], [-0.04, 0.05, -0.04, 0.05], 0, (lon2, lat2), id='global_grid_across_idl_r=0'),
        pytest.param("G", "EASE2_G9km", 179.9, 0, 20000, (lon3, lat3), id='global_grid_across_idl_r>0'),
        pytest.param("N", "EASE2_N9km", [7.1, 7.2], [85.1, 85.2], 0, (lon4, lat4), id='regular_polar_grid'),
        pytest.param("N", "EASE2_N36km", [179, 179, -178, 178], [87, 88, 90, 89], 0, (lon5, lat5), id='polar_grid_across_idl'),
    ]
)
def test_make_integration_grid(int_projection_definition, int_grid_definition, longitude, latitude, ap_radii, expected):

    lon, lat = ap.make_integration_grid(int_projection_definition, int_grid_definition, longitude, latitude, ap_radii)
    lonexp, latexp = expected

    assert(lon.shape == lonexp.shape), f"Expected size of longitude grid {lonexp.shape}, got {lon.shape}"
    assert(lat.shape == latexp.shape), f"Expected size of latitude grid {latexp.shape}, got {lat.shape}"

    assert np.isclose(lon, lonexp, rtol=1e-3).all(), f"Longitude grids are different"
    assert np.isclose(lat, latexp, rtol=1e-3).all(), f"Latitude grids are different"


######################################################################
# test: utils.intersection_with_sphere
######################################################################

@pytest.mark.parametrize(
    ("alpha", "R", "H", "expected"),
    [
        pytest.param(0, 1., 1., (0., 1.), id="nadir"),
        pytest.param(30., 3., 1., (np.sqrt(3)-np.sqrt(5)/2., 1+np.sqrt(15)/2.), id='30deg')
    ]
)
def test_intersection_with_sphere(alpha, R, H, expected):
    x, y = utils.intersection_with_sphere(np.deg2rad(alpha), R, H)

    assert np.isclose(x, expected[0], rtol=1e-3)
    assert np.isclose(y, expected[1], rtol=1e-3)


######################################################################
# test: estimate_max_ap_radius
######################################################################

@pytest.mark.parametrize(
     ("tilt_angle", "max_altitude", "max_theta_antenna_patterns", "antenna_method", 
     "antenna_threshold", "gaussian_params", "expected"),
    [
        pytest.param(0, 100000., 0., "gaussian_projected", 0., [0, 0], 0., id='gaussian_zero_radius'),
        pytest.param(0, 100000., 2., "gaussian_projected", 0., [1., 1.], 1.1*3492.1105641045438, id='gaussian_2deg_from_nadir'),
        pytest.param(0, 100000., 100., "instrument", 0., None, 1.1*3492.1105641045438, id='instrument_2deg_from_nadir')
    ]
)
def test_estimate_max_ap_radius(mocker, tilt_angle, max_altitude, max_theta_antenna_patterns, 
                                antenna_method, antenna_threshold, gaussian_params, expected): 

    # the value returned by estimate_max_ap_radius is saved to the max_ap_radius attribute of
    # an antenna_pattern, so we can check directly its value after initialization

    config = mocker.patch("cimr_rgb.config_file.ConfigFile").return_value
    config.antenna_tilt_angle = tilt_angle
    config.input_data_type = "CIMR" #not relevant here, but using CIMR avoids a -pi
    config.antenna_patterns_path = './mock_antenna_patterns'
    config.max_altitude = max_altitude
    config.max_theta_antenna_patterns = max_theta_antenna_patterns

    antenna_pattern = ap.AntennaPattern(config, "L", antenna_method, "scalar", antenna_threshold, gaussian_params)

    assert antenna_pattern.max_ap_radius[0] == expected, f"Expected {expected}, got {antenna_pattern.max_ap_radius[0]}"


######################################################################
# test: antenna_pattern_to_earth
######################################################################

lon1, lat1 = ap.make_integration_grid("G", "EASE2_G3km", 0., 0., 10000.)
pos1 = [6378137.0+500000., 0., 0]
vel1 = [1., 0., 0.]
exp1 = np.zeros_like(lon1)
exp1[4, 3] = 0.25
exp1[4, 4] = 0.25
exp1[5, 3] = 0.25
exp1[5, 4] = 0.25

# choosing the instrument here just determined if the attitude should be computed from the valocity (SMAP) or it's passed to the function (CIMR)

@pytest.mark.parametrize(
    ("instrument", "tilt_angle", "int_dom_lons", "int_dom_lats", "pos", "vel", "antenna_method", 
     "antenna_threshold", "gaussian_params", "processing_scan_angle", 
     "attitude", "lon_l1b", "lat_l1b", "expected"),
    [
        pytest.param("SMAP", 180., lon1, lat1, [6378137.+100000., 0., 0], [0., 1., 0.],
                     "instrument", 0., None, 
                     0., None, None, None, exp1, id='smap_above_(0,0)'),
    ]
)
def test_antenna_pattern_to_earth(mocker, instrument, tilt_angle, int_dom_lons, int_dom_lats, pos, vel, antenna_method, 
     antenna_threshold, gaussian_params, processing_scan_angle, 
     attitude, lon_l1b, lat_l1b, expected):

    config = mocker.patch("cimr_rgb.config_file.ConfigFile").return_value
    config.antenna_tilt_angle = tilt_angle
    config.input_data_type = instrument
    if instrument == "SMAP":
        config.antenna_patterns_path = './mock_antenna_patterns/L/CIMR-PAP-FR-L0-TPv1.0.0.h5'
    else:
        config.antenna_patterns_path = './mock_antenna_patterns'
    config.max_altitude = 100000.
    config.max_theta_antenna_patterns = 10.
    config.scan_angle_feed_offset = {"L": {0 : 0.}}
    if lon_l1b is None or lat_l1b is None:
        config.boresight_shift = False
    else:
        config.boresight_shift = False

    antenna_pattern = ap.AntennaPattern(config, "L", antenna_method, "scalar", antenna_threshold, gaussian_params)

    ginterp = antenna_pattern.antenna_pattern_to_earth(int_dom_lons, int_dom_lats, pos[0], pos[1],
                                 pos[2], vel[0], vel[1], vel[2], processing_scan_angle,
                                 0, attitude=attitude, lon_l1b=lon_l1b, lat_l1b=lat_l1b)

    assert np.isclose(ginterp, expected, rtol=1e-2).all(), "Projected gain is different from expected"