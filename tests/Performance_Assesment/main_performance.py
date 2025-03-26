
import cimr_grasp.grasp_io as grasp_io
import metrics
from performance_assessment_pipeline import TestCase, TestRunner
import numpy as np

repo_root = grasp_io.find_repo_root()

# Base configuration file path
config_path = repo_root.joinpath('tests/Performance_Assesment/performance_assessment_base.xml')

# Test data folder
test_data_folder = repo_root.joinpath('dpr/Test_cards')

# Output results path
output_results_path = repo_root.joinpath('/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/results/rsir_enhanced_max_neighbours_checking_search_radius.csv')

# Antenna patterns path
antenna_patterns_path = repo_root.joinpath('dpr/antenna_patterns')

# define TestRunner object
runner = TestRunner(config_path, antenna_patterns_path, test_data_folder, output_results_path, reset_results_data=False)


input_central_america = repo_root.joinpath(
    "dpr/L1B/CIMR/SCEPS_l1b_sceps_geo_central_america_scene_1_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1.nc")
input_polar_scene = repo_root.joinpath(
    "dpr/L1B/CIMR/SCEPS_l1b_sceps_geo_polar_scene_1_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1.nc")
input_test_scene_1 = repo_root.joinpath(
    "dpr/L1B/CIMR/SCEPS_l1b_devalgo_test_scene_1_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1.nc")
input_test_scene_2 = repo_root.joinpath(
    "dpr/L1B/CIMR/SCEPS_l1b_devalgo_test_scene_2_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1.nc")
input_test_scene_3 = repo_root.joinpath(
    "/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/L1B/CIMR/SCEPS_l1b_devalgo_test_scene_3_unfiltered_asc_tot_minimal_nom_nedt_apc_tot_v2p1-001.nc")
id_central_america = 'SCEPS_central_america'
id_polar_scene = 'SCEPS_polar_scene'
id_test_scene_1 = 'SCEPS_test_scene_1'
id_test_scene_2 = 'SCEPS_test_scene_2'
id_test_scene_3 = 'SCEPS_test_scene_3'
global_grids = ['EASE2_G3km', 'EASE2_G9km', 'EASE2_G36km']
polar_grids = ['EASE2_N3km', 'EASE2_N9km', 'EASE2_N36km']
list_metrics = [metrics.normalised_difference, metrics.mean_absolute_error, metrics.root_mean_square_error,
                metrics.standard_deviation_error, metrics.pointwise_correlation, metrics.valid_pixel_overlap, metrics.mean_absolute_percentage_error]



rsir_enhanced_max_neighbours = [TestCase(algorithm='RSIR',
                      band='L',
                      search_radius=100,
                      max_neighbours=n,
                      MRF_grid_resolution='3km',
                      rsir_iteration=n,
                      ) for n in np.arange(2, 200, 5)]

runner.run_tests(rsir_enhanced_max_neighbours,  list_metrics, id_central_america, input_central_america, ['EASE2_G9km'], reduced_grid_inds=[480, 671, 845, 996])
