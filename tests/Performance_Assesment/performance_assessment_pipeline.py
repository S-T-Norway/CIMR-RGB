from itertools import product
import xml.etree.ElementTree as ET
import sys
import os
import pickle
import psutil
import gc
import time, tracemalloc
from numpy import full, nan, isnan, where, float16
import numpy as np
import csv
from tqdm import tqdm
from cimr_rgb.config_file import ConfigFile
from cimr_rgb.data_ingestion import DataIngestion
from cimr_rgb.regridder import ReGridder, GRIDS
import metrics
import cimr_grasp.grasp_io as grasp_io


# import matplotlib
# tkagg = matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.ion()


def run_RGB(config_path):
    config_object = ConfigFile(config_path)
    data_dict = DataIngestion(config_object).ingest_data()
    data_dict_out = ReGridder(config_object).regrid_data(data_dict)
    return data_dict_out


def open_reference_data(reference_data_id, grid_resolution, test_card_folder, band):
    reference_data_file = (f"{test_card_folder}/{reference_data_id}/"
                           f"{reference_data_id}_{grid_resolution}.npz")
    truth_data_dict = np.load(reference_data_file)
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


class TestCase(object):

    def __init__(self, algorithm, band, **params):

        """
        Initializing a test case, which is a configuration for a specific algorithm

        Parameters:
            algorithm (string defining the regridding algorithm)
            band (string defining the band to regrid)
            params (keyword arguments, they need to be parameters in the RGB config file or "MRF_grid_resolution")

        Returns:
            an instance of the TestCase class
        """

        self.algorithm = algorithm
        self.band = band
        self.params = params
        self.testID = None #the ID will be assigned by the TestRunner
        

    def rewrite_config(self, config_path, antenna_patterns_path, input_data_path, grid, reduced_grid_inds=None):

        tree = ET.parse(config_path)
        root = tree.getroot()

        xml_params = dict()
        xml_params["InputData/path"] = input_data_path
        xml_params["InputData/antenna_patterns_path"] = antenna_patterns_path
        xml_params["ReGridderParams/regridding_algorithm"] = self.algorithm
        xml_params["InputData/target_band"] = self.band
        xml_params["InputData/source_band"] = self.band
        xml_params["OutputData/save_to_disk"] = False
        xml_params["OutputData/output_path"] = '.'
        xml_params["GridParams/grid_definition"] = grid
        xml_params["GridParams/projection_definition"] = grid[grid.rfind('_')+1] #the projection definition is the character after "_"

        if reduced_grid_inds is None:
            xml_params["GridParams/reduced_grid_inds"] = ''
        else:
            xml_params["GridParams/reduced_grid_inds"] = ' '.join(map(str, list(reduced_grid_inds)))

        for param in self.params:
            if param == 'MRF_grid_resolution':        # this is needed since actually only the MRF resolution is a parameter of the algorithm,
                mrf_proj = grid[grid.rfind('_')+1]    # while the MRF projection has to be forced to be the same as the output grid
                mrf_grid = grid                       # (and that's the param needed in the config)
                mrf_grid = grid[:grid.rfind('_')+2] + str(self.params[param])
                xml_params["ReGridderParams/MRF_grid_definition"] = mrf_grid
                xml_params["ReGridderParams/MRF_projection_definition"] = mrf_proj
            else:
                xml_params["ReGridderParams/"+str(param)]= self.params[param]

        for param_name, new_value in xml_params.items():

            elem = root.find(param_name)
            if elem is not None:
                elem.text = str(new_value)
            else:
                raise Exception(f"Couldn't find {param_name} in config")

        tree.write(config_path)


class TestRunner(object):

    base_headers = ['testID', 'scene', 'algorithm', 'band', 'grid', 'regridding_time_s', 'memory_usage_MB']
    test_count = 0

    def __init__(self, config_path, antenna_patterns_path, test_data_folder, output_results_path, reset_results_data=False):

        self.config_path = config_path
        self.antenna_patterns_path = antenna_patterns_path
        self.test_data_folder = test_data_folder
        parent_folder = os.path.dirname(output_results_path)
        filename_no_ext, _ = os.path.splitext(os.path.basename(output_results_path))
        self.output_results_csv = os.path.join(parent_folder, filename_no_ext+'.csv')
        self.output_results_img = os.path.join(parent_folder, filename_no_ext+'.npz')

        os.makedirs(parent_folder, exist_ok=True)

        if reset_results_data or not os.path.isfile(self.output_results_csv):
            if os.path.isfile(self.output_results_img):
                os.remove(self.output_results_img)
            with open(self.output_results_csv, mode='w+', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(TestRunner.base_headers)
            np.savez(self.output_results_img)
        else:
            try:
                with open(self.output_results_csv, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    last_row = None
                    for row in reader:
                        last_row = row
                    if last_row:
                        TestRunner.test_count = int(last_row[0])
                    else:
                        TestRunner.test_count = 0
            except:
                raise EOFError(f"Error in parsing {self.output_results_csv} Either delete it or set reset_results_data=True")

    def run_tests(self, list_tests, list_metrics, list_reference_data_ids, list_input_data_paths, list_grids, reduced_grid_inds=None):

        """
        Function running the RGB for the test cases, computing the metrics that compare with the reference images. 
        Metrics results are added to a csv file, while images are saved in an npz file. 

        Parameters:
            list_tests (list of TestCase instances)
            list_metrics (list of functions, each taking two arrays as input)
            list_reference_data_ids (list of strings, corresponding to the file names of reference data)
            list_input_data_paths (list of strings, corresponding to the paths to the L1B data)
            list_grids (list of strings defining the output grids)
            reduced_grid_inds (list of integers, defining the indexes of a subgrid of the output grid)
        
        Returns:
            nothing
        """

        # convert to list if it's just one instance
        list_tests = np.atleast_1d(list_tests)
        list_metrics = np.atleast_1d(list_metrics)
        list_reference_data_ids = np.atleast_1d(list_reference_data_ids)
        list_input_data_paths = np.atleast_1d(list_input_data_paths)
        list_grids = np.atleast_1d(list_grids)
        if len(list_reference_data_ids) != len(list_input_data_paths):
            raise ValueError(f"The number of reference data IDs and input_data_paths should be the same (now, {len(list_reference_data_ids)} and {len(list_input_data_paths)})")

        if os.path.isfile(self.output_results_img):
            images_archive = np.load(self.output_results_img)
            results_images_dict = dict()
            for key in images_archive.files:
                try:
                    results_images_dict[key] = images_archive[key]
                except Exception as e:
                    print(f"Error loading the image corresponding to test {key}.. skipped")
        else:
            results_images_dict = {}

        list_params = list(map(str, np.unique(np.concatenate([list(test.params.keys()) for test in list_tests]))))
        list_metrics_names = [metric.__name__ for metric in list_metrics]

        with open(self.output_results_csv, 'r') as f:
            headers = next(csv.reader(f)) 

        missing_params  = [header for header in list_params if header not in headers]
        missing_metrics = [header for header in list_metrics_names if header not in headers]
        missing_headers = missing_params + missing_metrics
        current_params_and_metrics = headers[len(TestRunner.base_headers):]

        headers = TestRunner.base_headers + missing_params + current_params_and_metrics + missing_metrics

        if missing_headers:
            rows = []
            with open(self.output_results_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                X = reader.fieldnames  # Extract original headers
                for row in reader:
                    rows.append(row)
            with open(self.output_results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row[key] if key in row else "" for key in headers})       
        
        for reference_data_id,input_data_path in zip(list_reference_data_ids, list_input_data_paths):
            for test in list_tests:
                for grid in list_grids:

                    TestRunner.test_count +=1
                    test.testID = TestRunner.test_count

                    print(f"Running test {self.test_count} on {reference_data_id} with the following parameters:")
                    print(f"algorithm = {test.algorithm}")
                    print(f"band = {test.band}")
                    print(f"grid = {grid}")
                    for param in test.params:
                        print(f"{param} = {test.params[param]}")

                    test.rewrite_config(self.config_path, self.antenna_patterns_path, input_data_path, grid, reduced_grid_inds)

                    t0 = time.time()
                    tracemalloc.start()
                    # Run the RGB
                    rgb_dict = run_RGB(self.config_path)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    t1 = time.time()
                    print(f"Regridding time = {(t1-t0):.1f} s")
                    print(f"Peak memory usage = {(peak/1024**2):.2f} MB")
                    # print("----------------------------------------------------")

                    # Convert RGB Data to an EASE grid
                    rgb_image = RGB_dict_to_ease(rgb_dict, ['bt_h'], grid)[test.band]['bt_h']

                    # Open the reference data
                    ref_image = open_reference_data(reference_data_id, grid, self.test_data_folder, test.band)

                    # Add test configs and performance metrics to CSV
                    row = [test.testID, reference_data_id, test.algorithm, test.band, grid, t1-t0, peak/1024**2]

                    with (open(self.output_results_csv, mode='a', newline='', encoding='utf-8') as f):
                        writer = csv.writer(f)
                        for header in headers[len(TestRunner.base_headers):]:
                            if header in test.params:
                                row.append(test.params[header])
                            elif header in list_metrics_names:
                                imetric = list_metrics_names.index(header)
                                row.append(list_metrics[imetric](rgb_image, ref_image))
                            else:
                                row.append('')
                        writer.writerow(row)

                    # Add reduced images to dictionary
                    valid_mask = ~isnan(ref_image) & ~isnan(rgb_image)
                    valid_inds = where(valid_mask)
                    new_result_image = rgb_image[min(valid_inds[0]):max(valid_inds[0]), min(valid_inds[1]):max(valid_inds[1])].astype(float16)
                    size_gb = sys.getsizeof(results_images_dict) / (1024 ** 3)
                    process = psutil.Process(os.getpid())
                    mem_gb = process.memory_info().rss / (1024 ** 3)  # Convert to GB
                    del rgb_image, ref_image, valid_mask, valid_inds
                    gc.collect()

                    # Save the image results dict
                    results_images = np.load(self.output_results_img)
                    results_images_dict = {key: results_images[key] for key in results_images.files}
                    results_images_dict[str(test.testID)] = new_result_image
                    np.savez_compressed(self.output_results_img, **results_images_dict)

        return
