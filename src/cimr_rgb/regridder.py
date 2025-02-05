import logging
import warnings
import functools

from typing import Tuple

import numpy as np
from numpy import (
    meshgrid,
    isinf,
    where,
    full,
    nan,
    nanmin,
    zeros,
    unravel_index,
    all,
    sum,
    ones,
    unique,
    inf,
    ravel_multi_index,
    split,
    stack,
    arange,
)
from pyresample import kd_tree, geometry

# import matplotlib
# tkagg = matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from cimr_rgb.nn import NNInterp
from cimr_rgb.ids import IDSInterp
from cimr_rgb.dib import DIBInterp
from cimr_rgb.bg import BGInterp
from cimr_rgb.rsir import rSIRInterp
from cimr_rgb.grid_generator import GridGenerator, GRIDS
from cimr_rgb.iterative_methods import MIIinterp
from cimr_rgb.utils import great_circle_distance
from cimr_rgb.rgb_logging import RGBLogging

EARTH_RADIUS = 6378000


class ReGridder:
    def __init__(self, config):
        self.config = config
        self.algos = {
            "NN": lambda band=None: NNInterp(self.config),
            "DIB": lambda band=None: DIBInterp(self.config),
            "IDS": lambda band=None: IDSInterp(self.config),
            "BG": lambda band: BGInterp(self.config, band),
            "RSIR": lambda band: rSIRInterp(self.config, band),
            "LW": lambda band: MIIinterp(self.config, band, "LW"),
            "CG": lambda band: MIIinterp(self.config, band, "CG"),
        }

        # If config_object is None, then it won't have logger as attribute
        if config is not None:
            if config.logger is not None:
                self.logger = config.logger
            self.logpar_decorate = config.logpar_decorate
        else:
            # No formatting will be performed
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            self.logpar_decorate = False

    def get_algorithm(self, algorithm_name, band=None):
        algo = self.algos.get(algorithm_name)

        return algo(band)

    # Check if you need the full data dict once the function is complete
    def get_grid(self, data_dict=None):
        # Get target grid

        if self.config.grid_type == "L1C":
            # target_x, target_y = GridGenerator(self.config,
            #     projection_definition=self.config.projection_definition,
            #     grid_definition=self.config.grid_definition).generate_grid_xy(
            #     return_resolution=False
            # )
            tracked_func = RGBLogging.rgb_decorate_and_execute(
                decorate=self.logpar_decorate,
                decorator=RGBLogging.track_perf,
                logger=self.logger,
            )(
                GridGenerator(
                    config_object=self.config,
                    projection_definition=self.config.projection_definition,
                    grid_definition=self.config.grid_definition,
                ).generate_grid_xy
            )
            target_x, target_y = tracked_func(return_resolution=False)

            x_shape = len(target_x)
            y_shape = len(target_y)

            target_x, target_y = meshgrid(target_x, target_y)

            target_x = target_x.flatten()
            target_y = target_y.flatten()

            # target_lon, target_lat = GridGenerator(self.config,
            #                                        projection_definition=self.config.projection_definition,
            #                                        grid_definition=self.config.grid_definition
            #                                        ).xy_to_lonlat(
            #     x=target_x,
            #     y=target_y
            # )

            tracked_func = RGBLogging.rgb_decorate_and_execute(
                decorate=self.logpar_decorate,
                decorator=RGBLogging.track_perf,
                logger=self.logger,
            )(
                GridGenerator(
                    config_object=self.config,
                    projection_definition=self.config.projection_definition,
                    grid_definition=self.config.grid_definition,
                ).xy_to_lonlat
            )
            target_lon, target_lat = tracked_func(x=target_x, y=target_y)

            # Rehshape to original matrix
            target_lon = target_lon.reshape(y_shape, x_shape)
            target_lat = target_lat.reshape(y_shape, x_shape)
            target_grid = [target_lon, target_lat]

        elif self.config.grid_type == "L1R":
            # For the generation of a target grid in L1r, we just needs the lats and lons of
            # the target band.
            target_band = self.config.target_band
            target_lon = data_dict[target_band[0]]["longitude"]
            target_lat = data_dict[target_band[0]]["latitude"]
            target_grid = [target_lon, target_lat]

        return target_grid

    # def get_neighbours(
    #     self, source_lon, source_lat, target_lon, target_lat, search_radius, neighbours
    # ):
    #     source_def = geometry.SwathDefinition(lons=source_lon, lats=source_lat)

    #     if self.config.grid_type == "L1C":
    #         target_def = geometry.GridDefinition(lons=target_lon, lats=target_lat)

    #     elif self.config.grid_type == "L1R":
    #         target_def = geometry.SwathDefinition(lons=target_lon, lats=target_lat)

    #     # warnings.showwarning = RGBLogging.custom_warning_handler(logger = self.logger)

    #     # Capturing Warnings from kde_tree_get_neighbour_info()
    #     # (Use functools.partial to provide `logger` while deferring the other arguments)
    #     warnings.showwarning = functools.partial(
    #         RGBLogging.custom_warning_handler, self.logger
    #     )

    #     valid_input_index, valid_output_index, index_array, distance_array = (
    #         kd_tree.get_neighbour_info(
    #             source_geo_def=source_def,
    #             target_geo_def=target_def,
    #             neighbours=neighbours,
    #             radius_of_influence=search_radius,
    #         )
    #     )

    #     if distance_array.ndim == 1:
    #         inf_mask = ~isinf(distance_array)
    #         reduced_distance = distance_array[inf_mask]
    #         reduced_index = index_array[inf_mask]
    #         # Original indices is the original 1D location of the point in the EASE grid
    #         original_indices = where(inf_mask)[0]

    #     elif distance_array.ndim == 2:
    #         inf_mask = ~all(np.isinf(distance_array), axis=1)
    #         reduced_distance = distance_array[inf_mask]
    #         reduced_index = index_array[inf_mask]
    #         original_indices = where(inf_mask)[0]

    #     # Remember that valid_output_index could be necessary for something in the future
    #     return reduced_distance, reduced_index, original_indices, valid_input_index

    def get_neighbours(
        self,
        source_lon: np.ndarray,
        source_lat: np.ndarray,
        target_lon: np.ndarray,
        target_lat: np.ndarray,
        search_radius: float,
        neighbours: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve neighbor information for given source and target coordinates using a kd-tree search.

        This function creates geometry definitions for the source and target based on the grid type
        specified in the configuration (``self.config.grid_type``). It then calls
        ``kd_tree.get_neighbour_info`` to compute neighbor indices and distances within a given
        search radius. Infinite distances are filtered out before returning the results.

        **Important:**
          When ``self.config.grid_type`` is ``"L1C"``, the ``target_lon`` and ``target_lat``
          arrays **must be provided as 2D arrays** (e.g., shaped as (n_rows, n_cols)) because
          a GridDefinition expects 2D coordinate arrays. For grid type ``"L1R"``, these arrays
          may be provided as 1D arrays.

        All exceptions (ValueErrors and others) are logged via ``self.logger``.

        Parameters
        ----------
        source_lon : np.ndarray
            1D array of longitudes for the source grid.
        source_lat : np.ndarray
            1D array of latitudes for the source grid.
        target_lon : np.ndarray
            Array of longitudes for the target grid. For grid type "L1C", this must be a 2D array.
        target_lat : np.ndarray
            Array of latitudes for the target grid. For grid type "L1C", this must be a 2D array.
        search_radius : float
            The radius of influence for the neighbor search. Must be non-negative.
        neighbours : int
            The number of nearest neighbors to search for. Must be a positive integer.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
              - reduced_distance: Array of distances for neighbors that are within the search radius.
              - reduced_index: Array of neighbor indices corresponding to the distances.
              - original_indices: Array of original indices (from the input array) for entries with finite distances.
              - valid_input_index: Array of valid input indices as returned by ``kd_tree.get_neighbour_info``.

        Raises
        ------
        ValueError
            If any of the following conditions is met:
              - The source coordinate arrays have mismatched lengths.
              - The target coordinate arrays have mismatched lengths.
              - ``search_radius`` is negative.
              - ``neighbours`` is not a positive integer.
              - ``self.config.grid_type`` is not one of ``"L1C"`` or ``"L1R"``.
              - For grid type ``"L1C"``, if ``target_lon`` or ``target_lat`` are not 2D.
              - The distance array returned by ``kd_tree.get_neighbour_info`` has an unsupported
                number of dimensions.
        Exception
            Any unexpected exception is logged and re-raised.

        Examples
        --------
        >>> import numpy as np
        >>> # Assume `instance` is an instance of the class containing get_neighbours,
        >>> # and that instance.config.grid_type is set appropriately.
        >>> source_lon = np.array([0.1, 1.1, 2.1])
        >>> source_lat = np.array([0.0, 0.0, 0.0])
        >>> # For grid type "L1C", target arrays must be 2D.
        >>> target_lon = np.array([[0.0, 1.0, 2.0, 3.0]])
        >>> target_lat = np.array([[0.0, 0.0, 0.0, 0.0]])
        >>> search_radius = 0.2
        >>> neighbours = 1
        >>> red_dist, red_idx, orig_idx, valid_in = instance.get_neighbours(
        ...     source_lon, source_lat, target_lon, target_lat, search_radius, neighbours)
        >>> print(red_dist)  # Array of distances for the valid neighbor matches
        """

        try:
            # Validate coordinate array lengths.
            if len(source_lon) != len(source_lat):
                self.logger.error(
                    f"source_lon and source_lat must have the same length. Got {len(source_lon)} and {len(source_lat)}."
                )
                raise ValueError("source_lon and source_lat must have the same length.")
            if len(target_lon) != len(target_lat):
                self.logger.error(
                    f"target_lon and target_lat must have the same length. Got {len(target_lon)} and {len(target_lat)}."
                )
                raise ValueError("target_lon and target_lat must have the same length.")

            # Validate search_radius and neighbours.
            if search_radius < 0:
                self.logger.error(
                    f"search_radius must be non-negative. Got {search_radius}."
                )
                raise ValueError("search_radius must be non-negative.")
            if not isinstance(neighbours, int) or neighbours <= 0:
                self.logger.error(
                    f"neighbours must be a positive integer. Got {neighbours}."
                )
                raise ValueError("neighbours must be a positive integer.")

            # Create geometry definition for the source.
            source_def = geometry.SwathDefinition(lons=source_lon, lats=source_lat)

            # Create geometry definition for the target based on grid type.
            grid_type = getattr(self.config, "grid_type", None)
            if grid_type == "L1C":
                # Enforce that target coordinates are 2D for a GridDefinition.
                if target_lon.ndim != 2 or target_lat.ndim != 2:
                    self.logger.error(
                        f"For grid type 'L1C', target_lon and target_lat must be 2D arrays. Got shapes {target_lon.shape} and {target_lat.shape}."
                    )
                    raise ValueError(
                        "For grid type 'L1C', target_lon and target_lat must be 2D arrays."
                    )
                target_def = geometry.GridDefinition(lons=target_lon, lats=target_lat)
            elif grid_type == "L1R":
                target_def = geometry.SwathDefinition(lons=target_lon, lats=target_lat)
            else:
                self.logger.error(f"Unsupported grid_type: {grid_type}")
                raise ValueError(f"Unsupported grid_type: {grid_type}")

            # Set up warning capturing using the custom warning handler.
            warnings.showwarning = functools.partial(
                RGBLogging.custom_warning_handler, self.logger
            )

            # Retrieve neighbor information via the kd-tree.
            valid_input_index, valid_output_index, index_array, distance_array = (
                kd_tree.get_neighbour_info(
                    source_geo_def=source_def,
                    target_geo_def=target_def,
                    neighbours=neighbours,
                    radius_of_influence=search_radius,
                )
            )

            # Process the returned distance array.
            if distance_array.ndim == 1:
                inf_mask = ~np.isinf(distance_array)
                reduced_distance = distance_array[inf_mask]
                reduced_index = index_array[inf_mask]
                original_indices = np.where(inf_mask)[0]
            elif distance_array.ndim == 2:
                row_mask = ~np.all(np.isinf(distance_array), axis=1)
                reduced_distance = distance_array[row_mask]
                reduced_index = index_array[row_mask]
                original_indices = np.where(row_mask)[0]
            else:
                self.logger.error(
                    f"Unsupported dimensions for distance_array: expected 1 or 2, got {distance_array.ndim}"
                )
                raise ValueError(
                    f"Unsupported dimensions for distance_array: expected 1 or 2, got {distance_array.ndim}"
                )

            return reduced_distance, reduced_index, original_indices, valid_input_index

        except Exception as e:
            # Log any exception with traceback details.
            self.logger.error(f"Exception in get_neighbours: {e}", exc_info=True)
            raise

    # We need data dict or just pass specific variables?
    def get_l1c_samples(
        self, variable_dict, target_grid
    ):  # also need to change the name of this for l1r
        samples_dict = {}

        search_radius = self.config.search_radius
        neighbours = self.config.max_neighbours

        samples_dict = {}
        if self.config.split_fore_aft:
            for scan_direction in ["fore", "aft"]:
                target_lon = target_grid[0]
                target_lat = target_grid[1]
                source_lon = variable_dict[f"longitude_{scan_direction}"]
                source_lat = variable_dict[f"latitude_{scan_direction}"]

                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(self.get_neighbours)

                reduced_distance, reduced_index, original_indices, valid_input_index = (
                    tracked_func(
                        source_lon=source_lon,
                        source_lat=source_lat,
                        target_lon=target_lon,
                        target_lat=target_lat,
                        search_radius=search_radius,
                        neighbours=neighbours,
                    )
                )

                samples_dict_temp = {}
                samples_dict_temp["distances"] = reduced_distance
                samples_dict_temp["indexes"] = reduced_index
                samples_dict_temp["grid_1d_index"] = original_indices
                samples_dict_temp["valid_input_index"] = valid_input_index
                samples_dict[scan_direction] = samples_dict_temp

        else:
            # Dont split fore/aft
            target_lon = target_grid[0]
            target_lat = target_grid[1]
            source_lon = variable_dict["longitude"]
            source_lat = variable_dict["latitude"]

            tracked_func = RGBLogging.rgb_decorate_and_execute(
                decorate=self.logpar_decorate,
                decorator=RGBLogging.track_perf,
                logger=self.logger,
            )(self.get_neighbours)
            reduced_distance, reduced_index, original_indices, valid_input_index = (
                tracked_func(
                    source_lon=source_lon,
                    source_lat=source_lat,
                    target_lon=target_lon,
                    target_lat=target_lat,
                    search_radius=search_radius,
                    neighbours=neighbours,
                )
            )

            samples_dict = {}
            samples_dict["distances"] = reduced_distance
            samples_dict["indexes"] = reduced_index
            samples_dict["grid_1d_index"] = original_indices
            samples_dict["valid_input_index"] = valid_input_index

        return samples_dict

    def sample_selection(self, variable_dict, target_grid):
        # samples_dict = self.get_l1c_samples(variable_dict, target_grid)
        tracked_func = RGBLogging.rgb_decorate_and_execute(
            decorate=self.logpar_decorate,
            decorator=RGBLogging.track_perf,
            logger=self.logger,
        )(self.get_l1c_samples)
        samples_dict = tracked_func(variable_dict, target_grid)

        for variable in variable_dict:
            if "fore" in variable:
                variable_dict[variable] = variable_dict[variable][
                    samples_dict["fore"]["valid_input_index"]
                ]
            elif "aft" in variable:
                variable_dict[variable] = variable_dict[variable][
                    samples_dict["aft"]["valid_input_index"]
                ]
            else:
                variable_dict[variable] = variable_dict[variable][
                    samples_dict["valid_input_index"]
                ]

        # Remove valid_input_index as we don't use it again, and reduce_grid_inds won't
        # work if its inside (this can be changed if it turns out we need it)
        if self.config.split_fore_aft:
            del samples_dict["fore"]["valid_input_index"]
            del samples_dict["aft"]["valid_input_index"]
        else:
            del samples_dict["valid_input_index"]

        if self.config.reduced_grid_inds:
            if self.config.split_fore_aft:
                # samples_dict['fore'] = self.reduce_grid_inds(samples_dict['fore'])

                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(self.reduce_grid_inds)
                samples_dict["fore"] = tracked_func(samples_dict["fore"])

                # samples_dict['aft'] = self.reduce_grid_inds(samples_dict['aft'])
                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(self.reduce_grid_inds)
                samples_dict["aft"] = tracked_func(samples_dict["aft"])
            else:
                # samples_dict = self.reduce_grid_inds(samples_dict)
                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(self.reduce_grid_inds)
                samples_dict = tracked_func(samples_dict)

        return samples_dict, variable_dict

    def reduce_grid_inds(self, samples_dict):
        if self.config.grid_type == "L1C":
            grid_shape = (
                GRIDS[self.config.grid_definition]["n_rows"],
                GRIDS[self.config.grid_definition]["n_cols"],
            )

        elif self.config.grid_type == "L1R":
            grid_shape = self.config.scan_geometry[self.config.target_band[0]]

        rows_out, cols_out = unravel_index(samples_dict["grid_1d_index"], grid_shape)

        row_min = self.config.reduced_grid_inds[0]
        row_max = self.config.reduced_grid_inds[1]
        col_min = self.config.reduced_grid_inds[2]
        col_max = self.config.reduced_grid_inds[3]

        indices = where(
            (rows_out >= row_min)
            & (rows_out <= row_max)
            & (cols_out >= col_min)
            & (cols_out <= col_max)
        )[0]

        filtered_samples_dict = {
            key: value[indices] for key, value in samples_dict.items()
        }

        return filtered_samples_dict

    def create_output_grid_inds(self, grid_1d_index):
        # Can be unified
        # Also am I making a 3D output variables for number of horns ?
        if self.config.grid_type == "L1C":
            grid_shape = (
                GRIDS[self.config.grid_definition]["n_rows"],
                GRIDS[self.config.grid_definition]["n_cols"],
            )
            # Do we not need scan and earth sample number here? becuase grid_1D index is referenced to the
            # non cleaned up input samples.
            row, col = unravel_index(grid_1d_index, grid_shape)
        elif self.config.grid_type == "L1R":
            # Now we need to build the grid from the original size of the swath
            grid_shape = self.config.num_target_scans, self.config.num_target_samples
            # Im not sure grid_1D index is valid because it is referring to the shape once things have been removed
            # as of yet I dont think anything is being removed, but if I were to for example apply quality control
            # the size of the input array would change.
            row, col = unravel_index(grid_1d_index, grid_shape)

        return row, col

    def sample_selection_brute_force(self, variable_dict):
        samples_dict = {}

        # target_x, target_y = GridGenerator(self.config,
        #                                    projection_definition=self.config.projection_definition,
        #                                    grid_definition=self.config.grid_definition).generate_grid_xy(
        #     return_resolution=False
        # )
        tracked_func = RGBLogging.rgb_decorate_and_execute(
            decorate=self.logpar_decorate,
            decorator=RGBLogging.track_perf,
            logger=self.logger,
        )(
            GridGenerator(
                config_object=self.config,
                projection_definition=self.config.projection_definition,
                grid_definition=self.config.grid_definition,
            ).generate_grid_xy
        )
        target_x, target_y = tracked_func(return_resolution=False)

        # target_lon, target_lat = GridGenerator(self.config,
        #                                       projection_definition=self.config.projection_definition,
        #                                       grid_definition=self.config.grid_definition).xy_to_lonlat(
        #    x=target_x,
        #    y=target_y
        # )

        tracked_func = RGBLogging.rgb_decorate_and_execute(
            decorate=self.logpar_decorate,
            decorator=RGBLogging.track_perf,
            logger=self.logger,
        )(
            GridGenerator(
                config_object=self.config,
                projection_definition=self.config.projection_definition,
                grid_definition=self.config.grid_definition,
            ).xy_to_lonlat
        )
        target_lon, target_lat = tracked_func(x=target_x, y=target_y)

        if self.config.split_fore_aft:
            for scan_direction in ["fore", "aft"]:
                source_lon, source_lat = (
                    variable_dict[f"longitude_{scan_direction}"],
                    variable_dict[f"latitude_{scan_direction}"],
                )

                # source_x, source_y = GridGenerator(self.config,
                #                                    projection_definition=self.config.projection_definition,
                #                                    grid_definition=self.config.grid_definition).lonlat_to_xy(
                #     lon = source_lon,
                #     lat = source_lat
                # )
                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(
                    GridGenerator(
                        config_object=self.config,
                        projection_definition=self.config.projection_definition,
                        grid_definition=self.config.grid_definition,
                    ).lonlat_to_xy
                )
                source_x, source_y = tracked_func(lon=source_lon, lat=source_lat)

                pixel_map = zeros(source_x.shape[0])
                distances = zeros(source_x.shape[0])
                grid_shape = (
                    GRIDS[self.config.grid_definition]["n_rows"],
                    GRIDS[self.config.grid_definition]["n_cols"],
                )

                for count, source_measurement in enumerate(source_x):
                    x_distances = abs(target_x - source_measurement)
                    y_distances = abs(target_y - source_y[count])
                    x_index = where(x_distances == nanmin(x_distances))[0][0]
                    y_index = where(y_distances == nanmin(y_distances))[0][0]
                    pixel_map[count] = ravel_multi_index((y_index, x_index), grid_shape)
                    lon_1, lat_1 = source_lon[count], source_lat[count]
                    lon_2, lat_2 = target_lon[x_index], target_lat[y_index]
                    distances[count] = great_circle_distance(lon_1, lat_1, lon_2, lat_2)

                # Package the pixel dictionary into the same format as pyresample
                samples_dict_temp = {}
                target_grid_1d_len = grid_shape[0] * grid_shape[1]
                fill_value = len(variable_dict[f"longitude_{scan_direction}"])
                unique_pixel, unique_counts = unique(pixel_map, return_counts=True)

                distances_out = full((len(unique_pixel), max(unique_counts)), inf)
                indexes = full(
                    shape=(len(unique_pixel), max(unique_counts)), fill_value=fill_value
                )
                grid_1d_index = full(len(unique_pixel), fill_value)

                for count, pixel in enumerate(unique_pixel):
                    pixel_inds = where(pixel_map == pixel)[0]
                    indexes[count, 0 : len(pixel_inds)] = pixel_inds
                    distances_out[count, 0 : len(pixel_inds)] = distances[pixel_inds]
                    grid_1d_index[count] = int(pixel)

                samples_dict_temp["distances"] = distances_out
                samples_dict_temp["indexes"] = indexes
                samples_dict_temp["grid_1d_index"] = grid_1d_index
                samples_dict[scan_direction] = samples_dict_temp
        else:
            source_lon, source_lat = (
                variable_dict[f"longitude"],
                variable_dict[f"latitude"],
            )
            # source_x, source_y = GridGenerator(self.config,
            #                                    projection_definition=self.config.projection_definition,
            #                                    grid_definition=self.config.grid_definition).lonlat_to_xy(
            #     lon=source_lon,
            #     lat=source_lat
            # )
            tracked_func = RGBLogging.rgb_decorate_and_execute(
                decorate=self.logpar_decorate,
                decorator=RGBLogging.track_perf,
                logger=self.logger,
            )(
                GridGenerator(
                    config_object=self.config,
                    projection_definition=self.config.projection_definition,
                    grid_definition=self.config.grid_definition,
                ).lonlat_to_xy
            )
            source_x, source_y = tracked_func(lon=source_lon, lat=source_lat)

            pixel_map = zeros(source_x.shape[0])
            distances = zeros(source_x.shape[0])
            grid_shape = (
                GRIDS[self.config.grid_definition]["n_rows"],
                GRIDS[self.config.grid_definition]["n_cols"],
            )

            for count, source_measurement in enumerate(source_x):
                x_distances = abs(target_x - source_measurement)
                y_distances = abs(target_y - source_y[count])
                x_index = where(x_distances == nanmin(x_distances))[0][0]
                y_index = where(y_distances == nanmin(y_distances))[0][0]
                pixel_map[count] = ravel_multi_index((y_index, x_index), grid_shape)
                lon_1, lat_1 = source_lon[count], source_lat[count]
                lon_2, lat_2 = target_lon[x_index], target_lat[y_index]
                distances[count] = great_circle_distance(lon_1, lat_1, lon_2, lat_2)

            # Package the pixel dictionary into the same format as pyresample
            samples_dict_temp = {}

            target_grid_1d_len = grid_shape[0] * grid_shape[1]

            fill_value = len(variable_dict[f"longitude"])

            unique_pixel, unique_counts = unique(pixel_map, return_counts=True)

            distances_out = full((len(unique_pixel), max(unique_counts)), inf)
            indexes = full(
                shape=(len(unique_pixel), max(unique_counts)), fill_value=fill_value
            )
            grid_1d_index = full(len(unique_pixel), fill_value)

            for count, pixel in enumerate(unique_pixel):
                pixel_inds = where(pixel_map == pixel)[0]
                indexes[count, 0 : len(pixel_inds)] = pixel_inds
                distances_out[count, 0 : len(pixel_inds)] = distances[pixel_inds]
                grid_1d_index[count] = int(pixel)

            samples_dict_temp["distances"] = distances_out
            samples_dict_temp["indexes"] = indexes
            samples_dict_temp["grid_1d_index"] = grid_1d_index
            samples_dict = samples_dict_temp

        return samples_dict, variable_dict

    def reshape_output_l1r_dict(self, data_dict_out):
        data_dict_reshaped = {}

        for band in data_dict_out:
            variable_dict = data_dict_out[band]
            variable_dict_reshape = {}

            # Get output geomtry
            num_feed_horns = int(self.config.num_horns[self.config.target_band[0]])
            output_scans = int(self.config.scan_geometry[self.config.target_band[0]][0])
            output_samples = int(
                self.config.scan_geometry[self.config.target_band[0]][1]
            )

            # Build the scan geometry
            for variable in variable_dict:
                reshaped_variable = full((output_scans, output_samples), nan)
                if "fore" in variable:
                    scan_direction = "_fore"
                elif "aft" in variable:
                    scan_direction = "_aft"
                else:
                    scan_direction = ""

                if variable in [f"regridding_l1b_orphans{scan_direction}"]:
                    variable_dict_reshape[variable] = variable_dict[variable]
                    continue

                if variable in [
                    f"cell_row{scan_direction}",
                    f"cell_col{scan_direction}",
                ]:
                    # Don't include cell_row or cell_col in output dict for L1R
                    continue

                cell_row = variable_dict[f"cell_row{scan_direction}"]
                cell_col = variable_dict[f"cell_col{scan_direction}"]

                for count, sample in enumerate(variable_dict[variable]):
                    reshaped_variable[int(cell_row[count]), int(cell_col[count])] = (
                        sample
                    )

                # Now the variable is reconstructed, split back into feedhorn geometry
                reshaped_variable = split(reshaped_variable, num_feed_horns, axis=1)
                reshaped_variable = stack(reshaped_variable, axis=2)

                variable_dict_reshape[variable] = reshaped_variable

            data_dict_reshaped[band] = variable_dict_reshape

        return data_dict_reshaped

    def regrid_data(self, data_dict):  # Need to change the name of this.
        # Get target grid
        if self.config.grid_type == "L1C":
            # target_grid = self.get_grid()
            tracked_func = RGBLogging.rgb_decorate_and_execute(
                decorate=self.logpar_decorate,
                decorator=RGBLogging.track_perf,
                logger=self.logger,
            )(self.get_grid)
            target_grid = tracked_func()

            target_dict = None

        elif self.config.grid_type == "L1R":
            # target_grid = self.get_grid(data_dict)
            tracked_func = RGBLogging.rgb_decorate_and_execute(
                decorate=self.logpar_decorate,
                decorator=RGBLogging.track_perf,
                logger=self.logger,
            )(self.get_grid)
            target_grid = tracked_func(data_dict)

            target_dict = data_dict[self.config.target_band[0]]

        data_dict_out = {}

        for band in data_dict:
            if band not in self.config.source_band and self.config.grid_type == "L1R":
                continue

            self.logger.info(f"Regridding band: `{band}`")

            variable_dict = data_dict[band]

            if self.config.search_radius:
                # samples_dict, variable_dict = self.sample_selection(variable_dict, target_grid)
                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(self.sample_selection)
                samples_dict, variable_dict = tracked_func(variable_dict, target_grid)

            else:
                # samples_dict, variable_dict = self.sample_selection_brute_force(variable_dict)
                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(self.sample_selection_brute_force)
                samples_dict, variable_dict = tracked_func(variable_dict)

            if self.config.split_fore_aft:
                variable_dict_out = {}

                for scan_direction in ["fore", "aft"]:
                    self.logger.info(f"Scan direction is: `{scan_direction}`")

                    # Create an argument dictionary for interp_variable_dict, not all the algorithms need
                    # all the arguments.
                    args = {
                        "samples_dict": samples_dict[scan_direction],
                        "variable_dict": variable_dict,
                        "target_dict": target_dict,
                        "target_grid": target_grid,
                        "scan_direction": scan_direction,
                        "band": band,
                    }

                    # Regrid Variables
                    # algorithm = self.get_algorithm(algorithm_name=self.config.regridding_algorithm,
                    #                               band=band)
                    tracked_func = RGBLogging.rgb_decorate_and_execute(
                        decorate=self.logpar_decorate,
                        decorator=RGBLogging.track_perf,
                        logger=self.logger,
                    )(self.get_algorithm)
                    algorithm = tracked_func(
                        algorithm_name=self.config.regridding_algorithm, band=band
                    )

                    # Tracking performance of this method
                    # variable_dict_out[scan_direction] = algorithm.interp_variable_dict(**args)

                    tracked_func = RGBLogging.rgb_decorate_and_execute(
                        decorate=self.logpar_decorate,
                        decorator=RGBLogging.track_perf,
                        logger=self.logger,
                    )(algorithm.interp_variable_dict)

                    variable_dict_out[scan_direction] = tracked_func(**args)

                    # Add cell_row and cell_col indexes
                    # cell_row, cell_col = self.create_output_grid_inds(grid_1d_index=samples_dict[scan_direction]['grid_1d_index'])

                    tracked_func = RGBLogging.rgb_decorate_and_execute(
                        decorate=self.logpar_decorate,
                        decorator=RGBLogging.track_perf,
                        logger=self.logger,
                    )(self.create_output_grid_inds)

                    cell_row, cell_col = tracked_func(
                        grid_1d_index=samples_dict[scan_direction]["grid_1d_index"]
                    )

                    variable_dict_out[scan_direction][f"cell_row_{scan_direction}"] = (
                        cell_row
                    )
                    variable_dict_out[scan_direction][f"cell_col_{scan_direction}"] = (
                        cell_col
                    )

                    # Testing - rebuilding the scan geometry
                    test = 0

                    # Add regridding_n_samples
                    if "regridding_n_samples" in self.config.variables_to_regrid:
                        if samples_dict[scan_direction]["distances"].ndim == 1:
                            variable_dict_out[scan_direction][
                                "regridding_n_samples"
                            ] = ones(
                                len(samples_dict[scan_direction]["distances"]),
                                dtype=int,
                            )

                        else:
                            variable_dict_out[scan_direction][
                                f"regridding_n_samples_{scan_direction}"
                            ] = sum(
                                ~isinf(samples_dict[scan_direction]["distances"]),
                                axis=1,
                            )

                    # Add regridding_l1b_orphans (make own function)
                    if "regridding_l1b_orphans" in self.config.variables_to_regrid:
                        fill_value = len(variable_dict[f"longitude_{scan_direction}"])
                        unique_indexes = unique(
                            samples_dict[scan_direction]["indexes"][
                                samples_dict[scan_direction]["indexes"] != fill_value
                            ]
                        )
                        used_scans = variable_dict[f"scan_number_{scan_direction}"][
                            unique_indexes
                        ].astype(int)
                        used_samples = variable_dict[f"sample_number_{scan_direction}"][
                            unique_indexes
                        ].astype(int)

                        if self.config.input_data_type == "SMAP":
                            # Create output array the same size as the input data
                            l1b_orphans = zeros(self.config.scan_geometry[band])
                            l1b_orphans[used_scans, used_samples] = 1

                        elif self.config.input_data_type == "CIMR":
                            scan_geometry = self.config.scan_geometry[band]

                            num_feed_horns = self.config.num_horns[band]

                            l1b_orphans = zeros((
                                scan_geometry[0],
                                int(scan_geometry[1] / num_feed_horns),
                                num_feed_horns,
                            ))

                            feed_horn_number = variable_dict[
                                f"feed_horn_number_{scan_direction}"
                            ]

                            used_feed_horn_number = feed_horn_number[
                                unique_indexes
                            ].astype(int)

                            for feed_horn in range(num_feed_horns):
                                feed_horn_inds = where(
                                    used_feed_horn_number == feed_horn
                                )[0]
                                feed_horn_scans = used_scans[feed_horn_inds]
                                feed_horn_samples = used_samples[feed_horn_inds]
                                l1b_orphans[
                                    feed_horn_scans, feed_horn_samples, feed_horn
                                ] = 1

                        variable_dict_out[scan_direction][
                            f"regridding_l1b_orphans_{scan_direction}"
                        ] = l1b_orphans

                # Combine fore and aft variables into single dictionary
                combined_dict = {
                    **variable_dict_out["fore"],
                    **variable_dict_out["aft"],
                }
                variable_dict_out = combined_dict
            else:
                # Don't split fore/aft

                # Create an argument dictionary for interp_variable_dict, not all the algorithms need
                # all the arguments.
                args = {
                    "samples_dict": samples_dict,
                    "variable_dict": variable_dict,
                    "target_dict": target_dict,
                    "target_grid": target_grid,
                    "scan_direction": None,
                    "band": band,
                }

                # Regrid Variables
                # algorithm = self.get_algorithm(algorithm_name=self.config.regridding_algorithm,
                #                               band=band)
                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(self.get_algorithm)
                algorithm = tracked_func(
                    algorithm_name=self.config.regridding_algorithm, band=band
                )

                # variable_dict_out = algorithm.interp_variable_dict(**args)
                tracked_func = RGBLogging.rgb_decorate_and_execute(
                    decorate=self.logpar_decorate,
                    decorator=RGBLogging.track_perf,
                    logger=self.logger,
                )(algorithm.interp_variable_dict)

                variable_dict_out = tracked_func(**args)

                # Add cell_row and cell_col indexes
                cell_row, cell_col = self.create_output_grid_inds(
                    samples_dict["grid_1d_index"]
                )
                variable_dict_out["cell_row"] = cell_row
                variable_dict_out["cell_col"] = cell_col
                # if self.config.regridding_algorithm not in ['LW', 'CG']:
                #     tracked_func  = RGBLogging.rgb_decorate_and_execute(
                #         decorate  = self.logpar_decorate,
                #         decorator = RGBLogging.track_perf,
                #         logger    = self.logger
                #         )(self.create_output_grid_inds)
                #     cell_row, cell_col = tracked_func(samples_dict['grid_1d_index'])
                #
                #     variable_dict_out['cell_row'] = cell_row
                #     variable_dict_out['cell_col'] = cell_col
                #
                # else:
                #     cell_row = arange(self.config.reduced_grid_inds[0], self.config.reduced_grid_inds[1])
                #     cell_col = arange(self.config.reduced_grid_inds[2], self.config.reduced_grid_inds[3])
                #
                #     cell_col, cell_row = meshgrid(cell_col, cell_row)
                #     cell_row = cell_row.flatten()
                #     cell_col = cell_col.flatten()
                #     variable_dict_out['cell_row'] = cell_row
                #     variable_dict_out['cell_col'] = cell_col

                # Add regridding_n_samples
                if "regridding_n_samples" in self.config.variables_to_regrid:
                    if samples_dict["distances"].ndim == 1:
                        variable_dict_out["regridding_n_samples"] = ones(
                            len(samples_dict["distances"]), dtype=int
                        )
                    else:
                        variable_dict_out["regridding_n_samples"] = sum(
                            ~isinf(samples_dict["distances"]), axis=1
                        )

                # Add regridding_l1b_orphans
                if "regridding_l1b_orphans" in self.config.variables_to_regrid:
                    fill_value = len(variable_dict["longitude"])
                    unique_indexes = unique(
                        samples_dict["indexes"][samples_dict["indexes"] != fill_value]
                    )
                    used_scans = variable_dict[f"scan_number"][unique_indexes].astype(
                        int
                    )
                    used_samples = variable_dict[f"sample_number"][
                        unique_indexes
                    ].astype(int)

                    if self.config.input_data_type == "SMAP":
                        # Create output array the same size as the input data
                        l1b_orphans = zeros(self.config.scan_geometry[band])
                        l1b_orphans[used_scans, used_samples] = 1

                    elif self.config.input_data_type == "CIMR":
                        scan_geometry = self.config.scan_geometry[band]
                        num_feed_horns = self.config.num_horns[band]

                        l1b_orphans = zeros((
                            scan_geometry[0],
                            int(scan_geometry[1] / num_feed_horns),
                            num_feed_horns,
                        ))

                        feed_horn_number = variable_dict["feed_horn_number"]

                        used_feed_horn_number = feed_horn_number[unique_indexes].astype(
                            int
                        )

                        for feed_horn in range(num_feed_horns):
                            feed_horn_inds = where(used_feed_horn_number == feed_horn)[
                                0
                            ]
                            feed_horn_scans = used_scans[feed_horn_inds]
                            feed_horn_samples = used_samples[feed_horn_inds]
                            l1b_orphans[
                                feed_horn_scans, feed_horn_samples, feed_horn
                            ] = 1

                    variable_dict_out["regridding_l1b_orphans"] = l1b_orphans

            data_dict_out[band] = variable_dict_out

            self.logger.info(f"Finished regridding band: `{band}`")

        # Reshape CIMR data
        if self.config.grid_type == "L1R":
            tracked_func = RGBLogging.rgb_decorate_and_execute(
                decorate=self.logpar_decorate,
                decorator=RGBLogging.track_perf,
                logger=self.logger,
            )(self.reshape_output_l1r_dict)
            data_dict_out = tracked_func(data_dict_out)

        return data_dict_out
