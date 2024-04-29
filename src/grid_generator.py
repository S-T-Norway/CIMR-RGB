"""
This module contains the GridGenerator class, which is responsible for generating
the grid for the RGB, that is, providing the coordinates for a user selected projection
for which the input data shall be re-gridded to. This includes a swath projection.
The class also provides independent functionality to transfer coordinates between
different coordinate systems.

The DataIngestion class utilizes configuration files to manage the ingestion
process.
"""

import numpy as np
import pyproj

MAP_EQUATORIAL_RADIUS = 6378137.0
E = 0.081819190843  # EASEv2 Map Eccentricity
E2 = E ** 2

GRIDS = {'EASE2_G9km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                        'res': 9008.05, 'n_cols': 3856, 'n_rows': 1624},
         'EASE2_N9km': {'epsg': 6931, 'x_min': -9000000.0, 'y_max': 9000000.0,
                        'res': 9000.0, 'n_cols': 2000, 'n_rows': 2000},
         'EASE2_S9km': {'epsg': 6932, 'x_min': -9000000.0, 'y_max': 9000000.0,
                        'res': 9000.0, 'n_cols': 2000, 'n_rows': 2000},
         'EASE2_G36km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                         'res': 36032.22, 'n_cols': 964, 'n_rows': 406},
         'EASE2_N36km': {'epsg': 6931, 'x_min': -9000000.0, 'y_max': 9000000.0,
                         'res': 36000.0, 'n_cols': 500, 'n_rows': 500},
         'EASE2_S36km': {'epsg': 6932, 'x_min': -9000000.0, 'y_max': 9000000.0,
                         'res': 36000.0, 'n_cols': 500, 'n_rows': 500}
         }

PROJECTIONS = {'G': "+proj=cea +lat_ts=30 +lon_0=0 +lat_0=0 "
                    "+x_0=0 +y_0=0 +datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs",
               'N': "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 "
                    "+datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs",
               'S': "+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 "
                    "+datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs"}


class GridGenerator:
    """
    This class is responsible for generating the grid for the RGB, that is,
    providing the coordinates for a user selected projection for which the
    input data shall be re-gridded to.

    Attributes:
    -----------
    config_object: xml.etree.ElementTree.Element
            Root element of the configuration file

    projection: str
            The projection definition for the grid, which should also be defined in the config file.

    Methods:
    --------
    generate_grid_xy(self, return_resolution=False):
        Generates the grid in x and y coordinates from a given grid definition.
        Grid definitions can be found in the GRIDS dictionary at the start of the module.

    generate_grid_lonlat(self):
        Generates the grid in longitude and latitude coordinates from a given grid definition
        and projection.

    lonlat_to_xy(self, lon, lat):
        Converts longitude and latitude coordinates to x and y coordinates for a given projection.

    xy_to_lonlat(self, x, y):
        Converts x and y coordinates to longitude and latitude coordinates for a given projection.

    xy_to_rowcol(self, x, y):
        Converts x and y coordinates to row and column indices for a given grid definition.

    generate_swath_grid(target_lons, target_lats):
        TBD
    """

    def __init__(self, config_object):
        """
        Initializes the GridGenerator object with the configuration object.

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        """
        self.config = config_object
        if self.config.grid_type == 'L1c':
            self.projection = PROJECTIONS[config_object.projection_definition]

    def generate_grid_xy(self, return_resolution=False):
        """
        Generates the grid in x and y coordinates from a given grid definition.
        Grid definitions can be found in the GRIDS dictionary at the start of the module.

        Parameters
        ----------
        return_resolution: bool
            If True, the function will return the resolution of the grid.

        Returns
        -------
        - If return_resolution is True, returns (xs, ys, res) where:
            xs (numpy.ndarray of float): Array of x-coordinates.
            ys (numpy.ndarray of float): Array of y-coordinates.
            res (float): Grid resolution.
        - If return_resolution is False, returns (xs, ys) where:
            xs (numpy.ndarray of float): Array of x-coordinates.
            ys (numpy.ndarray of float): Array of y-coordinates.
        """
        res = GRIDS[self.config.grid_definition]['res']
        n_cols = GRIDS[self.config.grid_definition]['n_cols']
        n_rows = GRIDS[self.config.grid_definition]['n_rows']
        # EASE Grids are by convention defined from the center of the pixel
        x_min = GRIDS[self.config.grid_definition]['x_min'] + res / 2
        y_max = GRIDS[self.config.grid_definition]['y_max'] - res / 2
        xs = np.zeros((n_cols, 1))

        for i in range(n_cols):
            xs[i] = x_min + (i * res)

        ys = np.zeros((n_rows, 1))
        for i in range(n_rows):
            ys[i] = y_max - (i * res)
        if return_resolution:
            return xs, ys, res

        return xs, ys

    def generate_grid_lonlat(self):
        """
        Generates the grid in longitude and latitude coordinates from a given grid definition
        and projection.

        Returns
        -------
        lons: (numpy.ndarray of float)
            Array of longitudes.
        lats: (numpy.ndarray of float)
            Array of latitudes.
        """
        xs, ys = self.generate_grid_xy(return_resolution=False)
        grid_x, grid_y = np.meshgrid(xs, ys)
        lons, lats = pyproj.Proj(self.projection)(grid_x, grid_y, inverse=True)
        return lons, lats

    def lonlat_to_xy(self, lon, lat):
        """
        Converts longitude and latitude coordinates to x and y coordinates for a given projection.
        Parameters
        ----------
        lon: (float or numpy.ndarray of float)
            Longitude/s in decimal degrees
        lat: (float or numpy.ndarray of float)
            Latitude/s in decimal degrees

        Returns
        -------
        x: (float or numpy.ndarray of float)
            x-coordinate/s
        y: (float or numpy.ndarray of float)
            y-coordinate/s
        """
        epsilon = 1e-6
        params = self.projection.split()
        lon_0 = float(next((param.split('=')[1] for param in params if 'lon_0' in param), None))
        dlon = lon - lon_0
        phi = np.deg2rad(lat)
        lam = np.deg2rad(dlon)
        sin_phi = np.sin(phi)

        q = (1 - E2) * (
            (sin_phi / (1 - E2 * sin_phi ** 2)) -
            (1 / (2 * E)) * np.log((1 - E * sin_phi) / (1 + E * sin_phi))
        )

        # qp=q(phi=90)
        qp = 1 - (
            ((1 - E2) / (2 * E)) * np.log((1 - E) / (1 + E))
        )

        if self.config.projection_definition == 'G':
            lat_ts_value = float(
                next(
                    (param.split('=')[1] for param in params if 'lat_ts' in param),
                    None
                )
            )
            sin_phi_1 = np.sin(np.deg2rad(lat_ts_value))
            cos_phi_1 = np.cos(np.deg2rad(lat_ts_value))
            k0 = cos_phi_1 / np.sqrt(1 - (E2 * sin_phi_1 * sin_phi_1))
            x = MAP_EQUATORIAL_RADIUS * k0 * lam
            y = (MAP_EQUATORIAL_RADIUS * q) / (2 * k0)
            return x, y

        if self.config.projection_definition == 'N':
            pole_diff = abs(qp - q)
            inds = pole_diff >= epsilon
            rho = MAP_EQUATORIAL_RADIUS * np.sqrt(qp - q) * inds

            x = rho * np.sin(lam)
            y = -rho * np.cos(lam)
            return x, y

        if self.config.projection_definition == 'S':
            pole_diff = abs(qp + q)
            inds = pole_diff >= epsilon
            rho = MAP_EQUATORIAL_RADIUS * np.sqrt(qp + q) * inds
            x = rho * np.sin(lam)
            y = rho * np.cos(lam)
            return x, y

        if self.config.projection_definition == 'S':
            pass

    def xy_to_lonlat(self, x, y):
        """
        Converts x and y coordinates to longitude and latitude coordinates for a given projection.
        Parameters
        ----------
        x: (float or numpy.ndarray of float)
            x-coordinate/s
        y: (float or numpy.ndarray of float)
            y-coordinate/s

        Returns
        -------
        lon: (float or numpy.ndarray of float):
            Longitude/s in decimal degrees
        lat: (float or numpy.ndarray of float)
            Latitude/s in decimal degrees
        """
        params = self.projection.split()
        lon_0 = float(next((param.split('=')[1] for param in params if 'lon_0' in param), None))
        E4 = E ** 4
        E6 = E ** 6
        qp = 1 - (((1 - E2) / (2 * E)) * np.log((1 - E) / (1 + E)))
        beta = None
        lam = None

        if self.config.projection_definition == 'G':
            lat_ts_value = float(
                next(
                    (param.split('=')[1] for param in params if 'lat_ts' in param),
                    None
                )
            )
            sin_phi_1 = np.sin(np.deg2rad(lat_ts_value))
            cos_phi_1 = np.cos(np.deg2rad(lat_ts_value))
            k0 = cos_phi_1 / np.sqrt(1 - (E2 * sin_phi_1 * sin_phi_1))
            beta = np.arcsin((2 * y * k0) / (MAP_EQUATORIAL_RADIUS * qp))
            lam = x / (MAP_EQUATORIAL_RADIUS * k0)

        elif self.config.projection_definition == 'N':
            rho = np.sqrt(x ** 2 + y ** 2)
            beta = np.arcsin(1 - (rho ** 2 / (MAP_EQUATORIAL_RADIUS ** 2 * qp)))
            lam = np.arctan2(x, y)

        elif self.config.projection_definition == 'S':
            rho = np.sqrt(x ** 2 + y ** 2)
            beta = -1 * np.arcsin(1 - (rho ** 2 / (MAP_EQUATORIAL_RADIUS ** 2 * qp)))
            lam = np.arctan2(x, y)

        phi = (((beta +
                 ((E2 / 3) + (31 * E4 / 180) + (517 * E6 / 5040)) * np.sin(2 * beta)) +
                ((23 * E4 / 360) + (251E6 / 3780)) * np.sin(4 * beta)) +
               (761E6 / 45360) * np.sin(6 * beta))

        lat = np.rad2deg(phi)
        lon = lon_0 + np.rad2deg(lam)
        return lon, lat

    def xy_to_rowcol(self, x, y):
        """
        Converts x and y coordinates to row and column indices for a given grid definition.

        Parameters
        ----------
        x: (float or numpy.ndarray of float)
            x-coordinate/s
        y: (float or numpy.ndarray of float)
            y-coordinate/s

        Returns
        -------
        row: (float or numpy.ndarray of float)
            Row index/indices

        col: (float or numpy.ndarray of float)
            Column index/indices
        """
        _, _, res = self.generate_grid_xy(return_resolution=True)
        n_cols = GRIDS[self.config.grid_type]['n_cols']
        n_rows = GRIDS[self.config.grid_type]['n_rows']
        r0 = (n_cols - 1) / 2
        s0 = (n_rows - 1) / 2
        col = r0 + (x / res)
        row = s0 - (y / res)

        # Make a check that there are is nothing out of range
        # TBD

        return row, col

    @staticmethod
    def generate_swath_grid(target_lons, target_lats):
        """
        TBD
        """
        pass
