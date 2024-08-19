from data_ingestion import DataIngestion
import os
from process_antenna_pattern import AntennaPattern
from grid_generator import GridGenerator
import numpy as np
from numpy import nanmin, nanmax, where, nan, arange, meshgrid, array, isnan, unique, column_stack, ravel_multi_index, \
    unravel_index
from scipy.spatial import KDTree
from utils import interleave_bits, deinterleave_bits
import time
from collections import defaultdict
from regridder import ReGridder
import pickle
from process_antenna_pattern import AntennaPattern
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Make a smaller output grid for testing
def reduce_grid(row_start, row_end, col_start, col_end, num_rows, num_cols):
    indices = np.arange(num_rows * num_cols).reshape(num_rows, num_cols)
    sub_grid_indices = indices[row_start:row_end + 1, col_start:col_end + 1]
    flattened_indices = sub_grid_indices.flatten()
    return flattened_indices


class BGInterpolation():
    def __init__(self, config_object, antenna_pattern_object):
        self.config = config_object
        self.antenna_pattern = antenna_pattern_object

    def get_grid(self, data_dict):
        # Get target grid
        source_x, source_y, target_x, target_y = ReGridder(self.config).initiate_grid(
            target_lon=data_dict['lons_target'],
            target_lat=data_dict['lats_target'],
        )
        # remove nans from source points
        non_nan_mask = ~np.isnan(source_x)
        self.config.non_nan_mask = non_nan_mask
        source_x = source_x[non_nan_mask]
        source_y = source_y[non_nan_mask]
        target_x, target_y = meshgrid(target_x, target_y)
        target_grid = np.column_stack((target_x.flatten(), target_y.flatten()))
        source_points = column_stack((source_x, source_y))
        return target_grid, source_points

    @staticmethod
    def wrap_coordinates(points, min_x, max_x):
        wrap_value = max_x - min_x
        wrapped_points = []
        original_indices = []
        for i, point in enumerate(points):
            x, y = point
            wrapped_points.append((x, y))
            original_indices.append(i)
            if x < min_x + wrap_value / 2:
                wrapped_points.append((x + wrap_value, y))
                original_indices.append(i)
            if x > max_x - wrap_value / 2:
                wrapped_points.append((x - wrap_value, y))
                original_indices.append(i)
        return np.array(wrapped_points), np.array(original_indices)

    @staticmethod
    def get_distances(source_points, indices, q_x, q_y, wrap_value):
        dists = []
        for index in indices:
            p_x, p_y = source_points[index]
            # Calculate the shortest distance accounting for wrap-around
            dist_x = min(abs(q_x - p_x), abs(q_x - (p_x + wrap_value)), abs(q_x - (p_x - wrap_value)))
            dist_y = abs(q_y - p_y)
            dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
            dists.append(dist)
        return array(dists)

    @staticmethod
    def filter_samples(samples, distances, max_samples=6):
        # Convert samples to numpy array for easier indexing
        samples = np.array(samples)
        distances = np.array(distances)

        # Create a dictionary to store scan lines and their samples
        scan_lines = defaultdict(list)
        for i, sample in enumerate(samples):
            scan_line = unravel_index(sample, (779, 241))[0]  # Assuming scan line is the first coordinate
            scan_lines[scan_line].append((distances[i], sample))

        # Sort each scan line's samples by distance
        for scan_line in scan_lines:
            scan_lines[scan_line].sort()
        selected_samples = []
        considered_scan_lines = set()
        while len(selected_samples) < max_samples:
            # Find the nearest neighbor
            nearest_sample = None
            nearest_distance = float('inf')
            nearest_scan_line = None
            for scan_line, samples_list in scan_lines.items():
                if scan_line not in considered_scan_lines and samples_list:
                    distance, sample = samples_list[0]
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_sample = sample
                        nearest_scan_line = scan_line
            if nearest_sample is None:
                break
            # Add the nearest sample and up to two additional closest neighbors on the same scan line
            selected_samples.append(nearest_sample)
            considered_scan_lines.add(nearest_scan_line)

            scan_line_samples = scan_lines[nearest_scan_line]
            count = 1
            for distance, sample in scan_line_samples[1:]:
                if count >= 3 or len(selected_samples) >= max_samples:
                    break
                selected_samples.append(sample)
                count += 1
            # Remove considered samples from the scan line
            scan_lines[nearest_scan_line] = scan_lines[nearest_scan_line][count:]
        return selected_samples

    def wrapped_search(self,tree,source_coordinates, query_points, radius, original_indices, min_x, max_x):
        wrap_value = max_x - min_x
        results_dict = {}
        for count, query_point in enumerate(query_points):
            q_x, q_y = query_point

            # Primary search
            results = tree.query_ball_point((q_x, q_y), radius)

            # Handle wrap-around for x-coordinate < min_x and > max_x
            if q_x - radius < min_x:
                results += tree.query_ball_point((q_x + wrap_value, q_y), radius)
            if q_x + radius > max_x:
                results += tree.query_ball_point((q_x - wrap_value, q_y), radius)

            # Get unique original indices
            original_results = list(set(original_indices[results]))
            if len(original_results) == 0:
                continue

            # Obtain distances between samples and target point (also considering toroidal map)
            dists = self.get_distances(source_coordinates, original_results, q_x, q_y,wrap_value)

            # Filter samples to obtain a max of 6 samples (ideally 3 on two revolutions)
            filtered_samples = self.filter_samples(original_results, dists)
            results_dict[count] = list(filtered_samples)

        return results_dict

    def points_selection(self, source_points, target_grid, min_x, max_x, search_radius):
        # The wrapped points function deals with the fact the in the longitude dimension
        # we are dealing with is on a Toroidal map. (i.e continuous)
        wrapped_points, original_indices = self.wrap_coordinates(source_points, min_x, max_x)
        search_tree = KDTree(wrapped_points)
        # Extracts samples to use for BG for each point in the target grid.
        samples_dict = self.wrapped_search(search_tree,source_points, target_grid, search_radius, original_indices, min_x, max_x)
        return samples_dict

    def bg_interpolation(self, target_cell_lon, target_cell_lat, l1b_inds, variable,
                         approx_integrals=False,        #if true, next 2 options are ignored. Not implemented yet. 
                         source_antenna_pattern_approx=None, 
                         target_antenna_pattern_approx='gaussian',
                         parameters=None):
        
        assert source_antenna_pattern_approx in [None, 'gaussian', 'axisymmetric', 'boresight_inferred']
        assert target_antenna_pattern_approx in ['gaussian', 'axisymmetric', 'boresight_inferred']

        AP = self.antenna_pattern

        if approx_integrals:
            #not implemented yet
            assert np.isin(['a, b'], parameters.keys()).all()
        else:
            if source_antenna_pattern_approx == 'gaussian' or target_antenna_pattern_approx == 'gaussian':
                assert np.isin(['sigma_lat', 'sigma_lon',  'rotation'], list(parameters.keys())).all()

        longitudes = [target_cell_lon]
        latitudes  = [target_cell_lat]

        for i in l1b_inds:
            scan_ind = i // 241  # remember the dimensions should come from configuration/self
            earth_sample_ind = i % 241   
            longitudes.append(AP.get_l1b_data('tb_lon', scan_ind, earth_sample_ind))
            latitudes.append(AP.get_l1b_data('tb_lat', scan_ind, earth_sample_ind))

        int_dom_lons, int_dom_lats = AP.make_integration_grid(longitudes, latitudes)

        if approx_integrals:
            pass
            #.....
            #compute the arrays g, v, u here.. using some approximation

        else:

            if target_antenna_pattern_approx == 'gaussian':
                target_ant_pattern = AP.make_gaussian(
                    grid_lons = int_dom_lons, 
                    grid_lats = int_dom_lats, 
                    lon0 = target_cell_lon, 
                    lat0 = target_cell_lat, 
                    slon = parameters['sigma_lon'], 
                    slat = parameters['sigma_lat'],
                    rot = parameters['rotation'])
                
            elif target_antenna_pattern_approx == 'axisymmetric':
                target_ant_pattern = AP.antenna_pattern_to_earth_simplified(
                    scan_ind=scan_ind,
                    earth_sample_ind=earth_sample_ind,
                    int_dom_lons=int_dom_lons,
                    int_dom_lats=int_dom_lats,
                )          
            elif target_antenna_pattern_approx == 'boresight_inferred':
                target_ant_pattern = AP.antenna_pattern_from_boresight(
                    scan_ind=l1b_inds[0]//241,              #the scan_index will determine satellite position and nadir point! I choose the first one.. to be discussed
                    boresight_lon=target_cell_lon,
                    boresight_lat=target_cell_lat,
                    int_dom_lons=int_dom_lons,
                    int_dom_lats=int_dom_lats,
                )

            target_ant_pattern /= np.sum(target_ant_pattern)

            ant_patterns = []
            earth_samples = []

            count = 0

            # Extract patterns for input l1b points, maybe make this loop a function
            for i in l1b_inds:

                scan_ind = i // 241  # remember the dimensions should come from configuration/self
                earth_sample_ind = i % 241

                # Get the antenna pattern
                if source_antenna_pattern_approx == 'gaussian':
                    source_cell_lon, source_cell_lat = AP.boresight_to_earth(scan_ind, earth_sample_ind)
                    ant_pattern = AP.make_gaussian(
                        grid_lons = int_dom_lons, 
                        grid_lats = int_dom_lats, 
                        lon0 = source_cell_lon, 
                        lat0 = source_cell_lat, 
                        slon = parameters['sigma_lon'], 
                        slat = parameters['sigma_lat'],
                        rot = parameters['rotation'])

                elif source_antenna_pattern_approx == 'axisymmetric':
                    ant_pattern = AP.antenna_pattern_to_earth_simplified(
                        scan_ind=scan_ind,
                        earth_sample_ind=earth_sample_ind,
                        int_dom_lons=int_dom_lons,
                        int_dom_lats=int_dom_lats,
                    )
                
                elif source_antenna_pattern_approx == 'boresight_inferred':
                    boresight_lon = AP.get_l1b_data('tb_lon', scan_ind, earth_sample_ind)
                    boresight_lat = AP.get_l1b_data('tb_lat', scan_ind, earth_sample_ind)
                    if boresight_lon < -1000 or boresight_lat < -1000:
                        print('data point with invalid coordinates', boresight_lon, boresight_lat)
                        continue
                    ant_pattern = AP.antenna_pattern_from_boresight(
                        scan_ind=scan_ind,
                        boresight_lon=boresight_lon,
                        boresight_lat=boresight_lat,
                        int_dom_lons=int_dom_lons,
                        int_dom_lats=int_dom_lats,
                    )                    

                else:
                    ant_pattern = AP.antenna_pattern_to_earth(
                        scan_ind=scan_ind,
                        earth_sample_ind=earth_sample_ind,
                        int_dom_lons=int_dom_lons,
                        int_dom_lats=int_dom_lats,
                    )

                ant_pattern /= np.sum(ant_pattern)
                ant_patterns.append(ant_pattern)

                earth_samples.append(AP.get_l1b_data(
                    var=variable,
                    scan_ind=scan_ind,
                    earth_sample_ind=earth_sample_ind
                ))

                count += 1 

            g = np.zeros((count, count))
            v = np.zeros(count)
            u = np.zeros(count)

            for i in range(count):
                u[i] = np.sum(ant_patterns[i])
                v[i] = np.sum(ant_patterns[i] * target_ant_pattern)
                for j in range(count):
                    g[i, j] = np.sum(ant_patterns[i] * ant_patterns[j])
        
        k = 0. #regularization factor
        g = g + k*np.identity(count )
        ginv = np.linalg.inv(g)

        a = ginv @ (v + (1 - u.T @ (ginv @ v)) / (u.T @ (ginv @ u)) * u)
        
        return np.dot(a, earth_samples)


    def regrid_l1c(self, data_dict):
        target_grid, source_points = self.get_grid(data_dict)

        # Make a smaller grid for testing
        reduced_grid_inds = reduce_grid(520, 540, 3290, 3310, 1624, 3856)    #ocean
        #reduced_grid_inds = reduce_grid(1110, 1440, 3080, 3170, 1624, 3856)  #


        # Split samples into Fore and aft
        mask_dict = {'aft': (data_dict['antenna_scan_angle'] >= self.config.aft_angle_min) & (
                data_dict['antenna_scan_angle'] <= self.config.aft_angle_max)}
        mask_dict['fore'] = ~mask_dict['aft']

        # --------------  POINTS SELECTION -----------------#
        # Define parameters for points selection, these should be dealt
        # with in the config file eventually.
        bg_search_radius = 4 # Pixels
        grid_resolution = 9 # km
        search_radius = (bg_search_radius/2 * grid_resolution)*1000 # m
        min_x, max_x = -17367530.44, -17367530.44 + 3856*9008.05 # From grid generator
        # start_time = time.time()
        print('starting point selection')
        # samples_dict = self.points_selection(source_points, target_grid, min_x, max_x, search_radius)
        # with open('samples_dict.pkl', 'wb') as file:
        #     pickle.dump(samples_dict, file)
        # print(f"Samples Selection took {time.time() - start_time} seconds")
        # Open precalculated samples dict
        with open('samples_dict.pkl', 'rb') as f:
            samples_dict = pickle.load(f)
        # --------------  POINTS SELECTION -----------------#


        bg_out = {}
        fore_sample_frequency = {}
        aft_sample_frequency = {}
        samples_processed = 0
        for scan_direction in ['fore', 'aft']:
            bg_out_temp = {}
            # Temporary solution to split fore and aft
            mask_dict = mask_dict[scan_direction]
            for sample in samples_dict:
                if sample not in reduced_grid_inds:
                    continue
                l1b_inds = []
                for l1b_ind in samples_dict[sample]:
                    if mask_dict[l1b_ind] == True:
                        l1b_inds.append(l1b_ind)
                    else:
                        continue
                if len(l1b_inds) == 0:
                    continue
                if scan_direction == 'fore':
                    fore_sample_frequency[sample] = len(l1b_inds)
                elif scan_direction == 'aft':
                    aft_sample_frequency[sample] = len(l1b_inds)
                # Temporary solution to split fore and aft

                target_cell_x, target_cell_y = target_grid[sample]
                target_cell_lon, target_cell_lat = GridGenerator(config).xy_to_lonlat(target_cell_x, target_cell_y)
                # Temporarily ignore high/low lats until antenna grid fixed

                if target_cell_lat >83 or target_cell_lat < -100:
                    continue

                print(f"processing sample = {sample}")
                if np.isnan(target_cell_lon) or np.isnan(target_cell_lat):
                    continue
                print(f"Target Cell: {target_cell_lon, target_cell_lat}")

                variable = 'tb_v'
                t_interp = self.bg_interpolation(target_cell_lon, target_cell_lat, l1b_inds, variable, False, 'boresight_inferred', 'boresight_inferred')
                print(f"BG Interpolated Value = {t_interp}")
                samples_processed+=1
                print(f"samples processed = {samples_processed}")
                bg_out_temp[sample] = t_interp
            bg_out[scan_direction] = bg_out_temp
            break
        return bg_out, target_grid, fore_sample_frequency, aft_sample_frequency









if __name__ == '__main__':
    # Ingest and Extract L1B Data
    ingestion_object = DataIngestion(os.path.join(os.getcwd(), '..', 'config.xml'))
    config = ingestion_object.config
    data_dict = ingestion_object.ingest_data()

    # Backus Gilbert Re-gridding
    start_time = time.time()

    lon, lat = 124.9500000000116, 6.749999992828501

    # indexes = np.array([81038, 81039, 81040, 81041, 81042, 81043, 81044, 81280, 81281,
    #    81282, 81283, 81284, 81285, 81286, 81521, 81522, 81523, 81524,
    #    81525, 81526, 81527, 81763, 81764, 81765, 81766, 81767, 81768,
    #    81769, 89340, 89581, 89582, 89583, 89584, 89585, 89586, 89587,
    #    89588, 89822, 89823, 89824, 89825, 89826, 89827, 89828, 89829,
    #    90063, 90064, 90065, 90066, 90067, 90068, 90069, 90070, 90308,
    #    90309, 90310])
    
    indexes = np.array([81038, 81285, 81523, 81769, 89340, 89581, 89823, 90066, 90308])
    
    # parameters = dict()
    # parameters['sigma_lon'] = 0.5
    # parameters['sigma_lat'] = 0.5
    # parameters['rotation']  = 0.

    ap = AntennaPattern(config)

    start_time = time.time()
    
    # t_interp = BGInterpolation(config, ap).bg_interpolation(lon, lat, indexes, 'tb_v', False, 'boresight_inferred', 'boresight_inferred')
    
    bg_out, target_grid, fore_sample_frequency, aft_sample_frequency  = BGInterpolation(config, ap).regrid_l1c(data_dict)
    with open('bg_out.pkl', 'wb') as file:
        pickle.dump(bg_out, file)
    
    
    print(time.time() - start_time)
    # print(t_interp)