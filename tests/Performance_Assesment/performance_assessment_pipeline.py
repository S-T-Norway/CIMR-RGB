from itertools import product
import xml.etree.ElementTree as ET
import pickle
import sys
import os
import psutil
import gc
from numpy import full, nan, isnan, where, float16
import csv
from tqdm import tqdm
# Add the path to your RGB scripts
sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb')
from config_file import ConfigFile
from data_ingestion import DataIngestion
from regridder import ReGridder, GRIDS
import metrics


# import matplotlib
# tkagg = matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.ion()


def get_valid_testing_configurations(search_radius, band, regridding_algorithm, grid_definition):
    # Generate all combinations
    testing_combinations = product(search_radius, band, regridding_algorithm, grid_definition)

    # Filter out combinations we dont want (maybe a smarter way to do this?)
    valid_combos = []
    for SR, B, RA, GD in testing_combinations:
        if B == 'L':
            if SR < 15:
                continue
            if GD == 'EASE2_G3km':
                continue
        elif B == 'C':
            if GD == 'EASE2_G36km':
                continue
            if SR > 30:
                continue
        elif B == 'X':
            if GD == 'EASE2_G36km':
                continue
            if SR > 30:
                continue
        elif B == 'KA':
            if GD != 'EASE2_G3km':
                continue
            if SR > 20:
                continue
        elif B == 'K':
            if GD != 'EASE2_G3km':
                continue
            if SR > 20:
                continue

        valid_combos.append((SR, B, RA, GD))

    # Turning into dict


    return valid_combos

def modify_config_params(config_path, search_radius, band, regridding_algorithm, grid_definition):
    tree = ET.parse(config_path)
    root = tree.getroot()

    xml_params = {
        "ReGridderParams/search_radius": search_radius,
        "InputData/target_band": band,
        "InputData/source_band": band,
        "ReGridderParams/regridding_algorithm": regridding_algorithm,
        "GridParams/grid_definition": grid_definition
    }

    for param_name, new_value in xml_params.items():
        elem = root.find(param_name)
        if elem is not None:
            elem.text = str(new_value)
        else:
            print(f"Couldnt find {param_name} in config")

    # Write the new config
    tree.write(config_path)

def run_RGB(config_path):
    config_object = ConfigFile(config_path)
    data_dict = DataIngestion(config_object).ingest_data()
    data_dict_out = ReGridder(config_object).regrid_data(data_dict)
    return data_dict_out

def open_reference_data(reference_data_id, grid_resolution, test_card_folder, band):
    reference_data_file = (f"{test_card_folder}/{reference_data_id}/"
                           f"EASE_grids/{reference_data_id}_{grid_resolution}.pkl")
    with open(reference_data_file, 'rb') as f:
        truth_data_dict = pickle.load(f)
    return truth_data_dict[f'{band}_BAND']

def RGB_dict_to_ease(data_dict, variables, grid_definition):
    data_dict_ease = {}
    for band in data_dict:
        data_dict_band = {}
        grid_shape = GRIDS[grid_definition]['n_rows'], GRIDS[grid_definition]['n_cols']
        grid_rows, grid_cols = data_dict[band]['cell_row'], data_dict[band]['cell_col']

        for variable in variables:
            variable_out = full(grid_shape, nan)

            for count, sample in enumerate(data_dict[band][variable]):
                row = grid_rows[count]
                col = grid_cols[count]
                variable_out[row, col] = sample
            data_dict_band[variable] = variable_out

        data_dict_ease[band] = data_dict_band
    return data_dict_ease

if __name__ == "__main__":

    # Base configuration file path
    config_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/performance_assessment_base.xml'

    # Test data folder
    test_data_folder = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Test_cards'
    reference_data_id = 'SCEPS_central_america'

    # Output results path
    output_results_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/results'

    # Values to be tested
    search_radius = [5, 10, 15, 20, 30, 40, 50]
    band = ["L", "C", "X", "K", "KA"]
    regridding_algorithm = ["NN", "DIB", "IDS"]
    grid_definition = ['EASE2_G36km', 'EASE2_G9km', 'EASE2_G3km']
    # split_fore_aft = ['True', 'False'] # Todo: Think about how to do this


    # Get a list of different configurations to be tested
    test_configs = get_valid_testing_configurations(
        search_radius=search_radius,
        band=band,
        regridding_algorithm=regridding_algorithm,
        grid_definition=grid_definition
    )

    # I went with the following solution for saving the results. (Open to other ideas).
    # The images are saved in a dictionary, with a certain test_ID, then the test configurations
    # and performance metrics are output to a CSV file with the same test_ID.
    results_images_dict = {}
    with (open(os.path.join(output_results_path, 'results.csv'),mode='w', newline='', encoding='utf-8') as f):
        writer = csv.writer(f)
        writer.writerow(['test_ID', 'search_radius', 'band', 'regridding_algorithm', 'grid_definition', 'ND', 'MAE', 'RMS', 'SDE', 'PC'])

        for test_ID, test_config in tqdm(enumerate(test_configs), total=len(test_configs)):
            search_radius = test_config[0]
            band = test_config[1]
            regridding_algorithm = test_config[2]
            grid_definition = test_config[3]

            print(f"""Running test {test_ID} with config:"
                  band = {band}
                  grid_definition = {grid_definition}
                  search_radius = {search_radius}
                  regridding_algorithm = {regridding_algorithm}""")


            # Edit the base configuration file with the new combinations
            modify_config_params(config_path, search_radius, band, regridding_algorithm, grid_definition)

            # Run the RGB
            rgb_dict = run_RGB(config_path)

            # Convert RGB Data to an EASE grid
            rgb_image = RGB_dict_to_ease(rgb_dict, ['bt_h'], grid_definition)[band]['bt_h']

            # Open the reference data
            ref_image = open_reference_data(reference_data_id, grid_definition, test_data_folder, band)['bt']

            # # Compare images/ Apply performance metrics
            # Todo: We need to apply a metric that analyses how many output cells are actually filled
            # For example, 36km grid with 5km SR will only have a few output.

            ND = metrics.normalised_difference(rgb_image, ref_image)
            MAE = metrics.mean_absolute_error(rgb_image, ref_image)
            RMS = metrics.root_mean_square_error(rgb_image, ref_image)
            SDE = metrics.standard_deviation_error(rgb_image, ref_image)
            PC = metrics.pointwise_correlation(rgb_image, ref_image)

            # Add test configs nad performance metrics to CSV
            writer.writerow([test_ID, search_radius, band, regridding_algorithm, grid_definition, ND, MAE, RMS, SDE, PC])

            # Add reduced images to dictionary
            valid_mask = ~isnan(ref_image) & ~isnan(rgb_image)
            valid_inds = where(valid_mask)
            results_images_dict[test_ID] = rgb_image[min(valid_inds[0]):max(valid_inds[0]), min(valid_inds[1]):max(valid_inds[1])].astype(float16)
            size_gb = sys.getsizeof(results_images_dict) / (1024 ** 3)
            process = psutil.Process(os.getpid())
            mem_gb = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            del rgb_image, ref_image, valid_mask, valid_inds
            gc.collect()


    # Save the image results dict
    with open(os.path.join(output_results_path, 'results_images.pkl'), 'wb') as f:
        pickle.dump(results_images_dict, f)











