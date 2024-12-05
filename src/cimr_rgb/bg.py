import logging

from numpy import (
    where,
    nan,
    take,
    full,
    all,
    sum,
    zeros,
    identity,
    dot,
    nansum,
    unravel_index,
    rad2deg,
    sqrt,
    newaxis,
    eye,
    einsum,
    concatenate,
    array,
)
from numpy.linalg import inv
from tqdm import tqdm

# ---- Testing ----
# import matplotlib
# tkagg = matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from cimr_rgb.grid_generator import GridGenerator, GRIDS
from cimr_rgb.ap_processing import (
    AntennaPattern,
    GaussianAntennaPattern,
    make_integration_grid,
)


class BGInterp:
    def __init__(self, config, band):
        self.config = config
        self.band = band

        # If config_object is None, then it won't have logger as attribute
        if self.config is not None:
            if self.config.logger is not None:
                self.logger = self.config.logger
            self.logpar_decorate = self.config.logpar_decorate
        else:
            # No formatting will be performed
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())
            self.logpar_decorate = False

        if self.config.source_antenna_method == "gaussian":
            self.source_ap = GaussianAntennaPattern(
                config=self.config,
                antenna_threshold=self.config.source_antenna_threshold,
            )

        else:
            self.source_ap = AntennaPattern(
                config=self.config,
                band=self.band,
                antenna_method=self.config.source_antenna_method,
                polarisation_method=self.config.polarisation_method,
                antenna_threshold=self.config.source_antenna_threshold,
                gaussian_params=self.config.source_gaussian_params,
            )

        if self.config.grid_type == "L1R":
            if self.config.target_antenna_method == "gaussian":
                self.target_ap = GaussianAntennaPattern(
                    config=self.config,
                    antenna_threshold=self.config.target_antenna_threshold,
                )

            else:
                self.target_ap = AntennaPattern(
                    config=self.config,
                    band=self.config.target_band[0],
                    antenna_method=self.config.target_antenna_method,
                    polarisation_method=self.config.polarisation_method,
                    antenna_threshold=self.config.target_antenna_threshold,
                    gaussian_params=self.config.target_gaussian_params,
                )

        else:  # L1c
            self.target_ap = GaussianAntennaPattern(
                config=self.config, antenna_threshold=0.001
            )  # check if this value makes sense for L1c

    def get_antenna_patterns(
        self,
        variable_dict,
        target_dict,
        target_lon,
        target_lat,
        source_inds,
        target_inds,
        target_cell_size,
    ):
        pattern_lons = array(variable_dict["longitude"][source_inds])
        pattern_lats = array(variable_dict["latitude"][source_inds])
        pattern_radii = []

        if self.config.source_antenna_method == "gaussian":
            sigmax = self.config.source_gaussian_params[0]
            sigmay = self.config.source_gaussian_params[1]
            pattern_radii = concatenate((
                pattern_radii,
                [self.source_ap.estimate_max_ap_radius(sigmax, sigmay)],
            ))
        else:
            max_radii = [
                self.source_ap.max_ap_radius[int(nn)]
                for nn in variable_dict["feed_horn_number"][source_inds]
            ]
            pattern_radii = concatenate((pattern_radii, max_radii))

        pattern_lons = concatenate((pattern_lons, [target_lon]))
        pattern_lats = concatenate((pattern_lats, [target_lat]))

        if self.config.grid_type == "L1C":
            sigmax = target_cell_size[0]
            sigmay = target_cell_size[1]
            pattern_radii = concatenate((
                pattern_radii,
                [self.target_ap.estimate_max_ap_radius(sigmax, sigmay)],
            ))
        elif self.config.target_antenna_method == "gaussian":
            sigmax = self.config.target_gaussian_params[0]
            sigmay = self.config.target_gaussian_params[1]
            pattern_radii = concatenate((
                pattern_radii,
                [self.target_ap.estimate_max_ap_radius(sigmax, sigmay)],
            ))
        else:
            pattern_radii = concatenate((
                pattern_radii,
                [
                    self.target_ap.max_ap_radius[
                        int(target_dict["feed_horn_number"][target_inds])
                    ]
                ],
            ))

        int_dom_lons, int_dom_lats = make_integration_grid(
            int_projection_definition=self.config.MRF_projection_definition,
            int_grid_definition=self.config.MRF_grid_definition,
            longitude=pattern_lons,
            latitude=pattern_lats,
            ap_radii=pattern_radii,
        )

        # Project source patterns to grid
        source_ant_patterns = []
        for sample in source_inds:
            if self.config.source_antenna_method == "gaussian":
                sample_pattern = self.source_ap.antenna_pattern_to_earth(
                    int_dom_lons=int_dom_lons,
                    int_dom_lats=int_dom_lats,
                    lon_nadir=variable_dict["sub_satellite_lon"][sample],
                    lat_nadir=variable_dict["sub_satellite_lat"][sample],
                    lon_l1b=variable_dict["longitude"][sample],
                    lat_l1b=variable_dict["latitude"][sample],
                    sigmax=self.config.source_gaussian_params[0],
                    sigmay=self.config.source_gaussian_params[1],
                )
                fraction_above_threshold = 1.0
            else:
                sample_pattern = self.source_ap.antenna_pattern_to_earth(
                    int_dom_lons=int_dom_lons,
                    int_dom_lats=int_dom_lats,
                    x_pos=variable_dict["x_position"][sample],
                    y_pos=variable_dict["y_position"][sample],
                    z_pos=variable_dict["z_position"][sample],
                    x_vel=variable_dict["x_velocity"][sample],
                    y_vel=variable_dict["y_velocity"][sample],
                    z_vel=variable_dict["z_velocity"][sample],
                    processing_scan_angle=variable_dict["processing_scan_angle"][
                        sample
                    ],
                    feed_horn_number=variable_dict["feed_horn_number"][sample],
                    attitude=variable_dict["attitude"][sample],
                    lon_l1b=variable_dict["longitude"][sample],
                    lat_l1b=variable_dict["latitude"][sample],
                )
                fraction_above_threshold = (
                    1.0
                    - self.source_ap.fraction_below_threshold[
                        int(variable_dict["feed_horn_number"][sample])
                    ]
                )
            # sample_pattern /= (sum(sample_pattern)/fraction_above_threshold)
            sample_pattern /= sum(sample_pattern)
            source_ant_patterns.append(sample_pattern)

        # Get target patterns
        if self.config.grid_type == "L1C":
            target_ant_pattern = self.target_ap.antenna_pattern_to_earth(
                int_dom_lons=int_dom_lons,
                int_dom_lats=int_dom_lats,
                lon_l1b=target_lon,
                lat_l1b=target_lat,
                sigmax=target_cell_size[0],
                sigmay=target_cell_size[1],
                alpha=0.0,
            )
            fraction_above_threshold = 1.0
        elif self.config.target_antenna_method == "gaussian":
            target_ant_pattern = self.target_ap.antenna_pattern_to_earth(
                int_dom_lons=int_dom_lons,
                int_dom_lats=int_dom_lats,
                lon_nadir=target_dict["sub_satellite_lon"][target_inds],
                lat_nadir=target_dict["sub_satellite_lat"][target_inds],
                lon_l1b=target_lon,
                lat_l1b=target_lat,
                sigmax=self.config.target_gaussian_params[0],
                sigmay=self.config.target_gaussian_params[1],
            )
            fraction_above_threshold = 1.0
        else:
            target_ant_pattern = self.target_ap.antenna_pattern_to_earth(
                int_dom_lons=int_dom_lons,
                int_dom_lats=int_dom_lats,
                x_pos=target_dict["x_position"][target_inds],
                y_pos=target_dict["y_position"][target_inds],
                z_pos=target_dict["z_position"][target_inds],
                x_vel=target_dict["x_velocity"][target_inds],
                y_vel=target_dict["y_velocity"][target_inds],
                z_vel=target_dict["z_velocity"][target_inds],
                processing_scan_angle=target_dict["processing_scan_angle"][target_inds],
                feed_horn_number=target_dict["feed_horn_number"][target_inds],
                attitude=target_dict["attitude"][target_inds],
                lon_l1b=target_lon,
                lat_l1b=target_lat,
            )
            fraction_above_threshold = (
                1
                - self.target_ap.fraction_below_threshold[
                    int(target_dict["feed_horn_number"][target_inds])
                ]
            )
        # target_ant_pattern /= (sum(target_ant_pattern)/fraction_above_threshold)
        target_ant_pattern /= sum(target_ant_pattern)

        return source_ant_patterns, target_ant_pattern

    def BG(self, samples_dict, variable_dict, target_dict, target_grid):
        indexes = samples_dict["indexes"]
        fill_value = len(variable_dict["longitude"])
        weights = full((indexes.shape[0], indexes.shape[1]), nan)

        for target_cell in tqdm(range(indexes.shape[0])):
            # Getting the target lon, lat
            if self.config.grid_type == "L1C":
                grid_area = GridGenerator(
                    self.config,
                    self.config.projection_definition,
                    self.config.grid_definition,
                ).get_grid_area()

                target_lon, target_lat = (
                    target_grid[0].flatten("C")[
                        samples_dict["grid_1d_index"][target_cell]
                    ],
                    target_grid[1].flatten("C")[
                        samples_dict["grid_1d_index"][target_cell]
                    ],
                )

                cell_area = grid_area.flatten("C")[
                    samples_dict["grid_1d_index"][target_cell]
                ]
                resolution = sqrt(cell_area)
                target_cell_size = [resolution, resolution]

            elif self.config.grid_type == "L1R":
                target_cell_size = None
                target_lon, target_lat = (
                    target_grid[0][samples_dict["grid_1d_index"][target_cell]],
                    target_grid[1][samples_dict["grid_1d_index"][target_cell]],
                )

            # Get Antenna Patterns
            samples = indexes[target_cell, :]
            input_samples = samples[samples != fill_value]
            # print(f"input_samples: {input_samples}")
            source_ant_patterns, target_ant_pattern = self.get_antenna_patterns(
                variable_dict=variable_dict,
                target_dict=target_dict,
                target_lon=target_lon,
                target_lat=target_lat,
                source_inds=input_samples,
                target_inds=samples_dict["grid_1d_index"][target_cell],
                target_cell_size=target_cell_size,
            )

            # BG algorithm
            num_input_samples = len(input_samples)
            g = zeros((num_input_samples, num_input_samples))
            v = zeros(num_input_samples)
            u = zeros(num_input_samples)

            for i in range(num_input_samples):
                u[i] = sum(source_ant_patterns[i])
                v[i] = sum(source_ant_patterns[i] * target_ant_pattern)
                for j in range(num_input_samples):
                    g[i, j] = sum(source_ant_patterns[i] * source_ant_patterns[j])

            # Regularisation Factor
            k = self.config.bg_smoothing
            # Error Covariance Matrix
            E = identity(num_input_samples)

            g = g + k * E
            ginv = inv(g)

            # Weights
            a = ginv @ (v + (1 - u.T @ (ginv @ v)) / (u.T @ (ginv @ u)) * u)
            weights[target_cell, : len(input_samples)] = a

        return weights

    def get_weights(
        self, samples_dict, variable_dict, target_dict, target_grid, scan_direction
    ):
        # Preparing variable_dict
        if scan_direction:
            scan_direction = f"_{scan_direction}"
        else:
            scan_direction = ""

        # Adding a empty attitude variable for SMAP
        if f"attitude{scan_direction}" not in variable_dict:
            variable_dict[f"attitude{scan_direction}"] = full(
                variable_dict[f"longitude{scan_direction}"].shape, None
            )

        variable_dict = {
            key.removesuffix(f"{scan_direction}"): value
            for key, value in variable_dict.items()
        }

        weights = self.BG(
            samples_dict=samples_dict,
            variable_dict=variable_dict,
            target_dict=target_dict,
            target_grid=target_grid,
        )

        return weights

    def get_nedt(self, weights, samples_dict, nedt):
        fill_value = len(nedt)
        indexes = samples_dict["indexes"]
        mask = indexes == fill_value
        vars = full(indexes.shape, nan)
        vars[~mask] = nedt[indexes[~mask]]
        vars_scaled = vars[:, :, newaxis] * eye(weights.shape[1])
        nedt = einsum("ij,ijk,ik->i", weights, vars_scaled, weights)

        return nedt

    def interp_variable_dict(self, **kwargs):
        if self.config.grid_type == "L1C":
            samples_dict = kwargs["samples_dict"]
            variable_dict = kwargs["variable_dict"]
            target_dict = None
            target_grid = kwargs["target_grid"]
            scan_direction = kwargs["scan_direction"]
            band = kwargs["band"]

        elif self.config.grid_type == "L1R":
            samples_dict = kwargs["samples_dict"]
            variable_dict = kwargs["variable_dict"]
            target_dict = kwargs["target_dict"]
            target_grid = kwargs["target_grid"]
            scan_direction = kwargs["scan_direction"]
            band = kwargs["band"]

        if scan_direction is not None:
            variable_dict = {
                key: value
                for key, value in variable_dict.items()
                if key.endswith(scan_direction)
            }

        weights = self.get_weights(
            samples_dict=samples_dict,
            variable_dict=variable_dict,
            target_dict=target_dict,
            target_grid=target_grid,
            scan_direction=scan_direction,
        )

        variable_dict_out = {}

        for variable in variable_dict:
            # Check if you want to regrid this variable
            if (
                variable.removesuffix(f"_{scan_direction}")
                not in self.config.variables_to_regrid
            ):
                continue

            if "nedt" in variable:
                # print(variable)
                self.logger.info(f"`{variable}`")
                variable_dict_out[variable] = self.get_nedt(
                    weights, samples_dict, variable_dict[variable]
                )

            # Apply weights to the variable you want to regrid
            fill_value = len(variable_dict[variable])

            # Combining BG with samples
            indexes = samples_dict["indexes"]
            mask = indexes == fill_value
            vars = full(indexes.shape, nan)
            vars[~mask] = variable_dict[variable][indexes[~mask]]
            vars_out = nansum(weights * vars, axis=1)
            variable_dict_out[variable] = vars_out

        return variable_dict_out
