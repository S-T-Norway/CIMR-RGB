"""
This module contains the GridGenerator class, which is responsible for generating
the grid for the RGB, that is, providing the coordinates for a user selected projection
for which the input data shall be re-gridded to. This includes a swath projection.
The class also provides independent functionality to transfer coordinates between
different coordinate systems.

The DataIngestion class utilizes configuration files to manage the ingestion
process.
"""


import os
import pathlib as pb 
import logging 
import importlib.resources as pkg_resources 

import numpy as np
import pyproj

from cimr_rgb.rgb_logging import RGBLogging  


MAP_EQUATORIAL_RADIUS = 6378137.0
E = 0.081819190843  # EASEv2 Map Eccentricity
E2 = E ** 2

GRIDS = {'EASE2_G1km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                        'res': 1000.9, 'n_cols': 34704, 'n_rows': 14616, 'lat_min': -86, 'lat_max': 86},
         'EASE2_G3km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                        'res': 3002.69, 'n_cols': 11568, 'n_rows': 4872, 'lat_min': -86, 'lat_max': 86},
         'EASE2_G9km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                        'res': 9008.05, 'n_cols': 3856, 'n_rows': 1624, 'lat_min': -86, 'lat_max': 86},
         'EASE2_N3km': {'epsg': 6931, 'x_min': -9000000.0, 'y_max': 9000000.0,
                        'res': 3000, 'n_cols': 6000, 'n_rows': 6000, 'lat_min': 0},
         'EASE2_N9km': {'epsg': 6931, 'x_min': -9000000.0, 'y_max': 9000000.0,
                        'res': 9000.0, 'n_cols': 2000, 'n_rows': 2000, 'lat_min': 0},
         'EASE2_S3km': {'epsg': 6932, 'x_min': -9000000.0, 'y_max': 9000000.0,
                        'res': 3000, 'n_cols': 6000, 'n_rows': 6000, 'lat_min': 0},#, 'lat_min': 0},
         'EASE2_S9km': {'epsg': 6932, 'x_min': -9000000.0, 'y_max': 9000000.0,
                        'res': 9000.0, 'n_cols': 2000, 'n_rows': 2000, 'lat_min': 0},
         'EASE2_G36km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83,
                         'res': 36032.22, 'n_cols': 964, 'n_rows': 406, 'lat_min': -86, 'lat_max': 86},
         'EASE2_N36km': {'epsg': 6931, 'x_min': -9000000.0, 'y_max': 9000000.0,
                         'res': 36000.0, 'n_cols': 500, 'n_rows': 500, 'lat_min': 0},
         'EASE2_S36km': {'epsg': 6932, 'x_min': -9000000.0, 'y_max': 9000000.0,
                         'res': 36000.0, 'n_cols': 500, 'n_rows': 500, 'lat_min': 0},
         'STEREO_N6.25km': {'epsg': 3413, 'x_min': -3850000, 'y_max': 5850000,
                            'res': 6250, 'n_cols': 1216, 'n_rows': 1792, 'lat_min': 60},
         'STEREO_N12.5km': {'epsg': 3413, 'x_min': -3850000, 'y_max': 5850000,
                            'res': 12500, 'n_cols': 608, 'n_rows': 896, 'lat_min': 60},
         'STEREO_N25km': {'epsg': 3413, 'x_min': -3850000, 'y_max': 5850000,
                          'res': 25000, 'n_cols': 304, 'n_rows': 448, 'lat_min': 60},
         'STEREO_S6.25km': {'epsg': 3976, 'x_min': -3950000, 'y_max': 4350000,
                            'res': 6250, 'n_cols': 1264, 'n_rows': 1328, 'lat_min': -60},
         'STEREO_S12.5km': {'epsg': 3976, 'x_min': -3950000, 'y_max': 4350000,
                            'res': 12500, 'n_cols': 632, 'n_rows': 664, 'lat_min': -60},
         'STEREO_S25km': {'epsg': 3976, 'x_min': -3950000, 'y_max': 4350000,
                          'res': 25000, 'n_cols': 316, 'n_rows': 332, 'lat_min': -60},
         'MERC_G25km': {'epsg': 3395, 'x_min':-20037508.342789244, 'y_max': 19929239.11337915,
                       'res': 25000, 'n_cols': 1604, 'n_rows': 1595, 'lat_min': -85, 'lat_max': 85},
         'MERC_G12.5km': {'epsg': 3395, 'x_min':-20037508.342789244, 'y_max': 19929239.11337915,
                       'res': 12500, 'n_cols': 3207, 'n_rows':3189 , 'lat_min': -85, 'lat_max': 85},
         'MERC_G6.25km': {'epsg': 3395, 'x_min':-20037508.342789244, 'y_max': 19929239.11337915,
                          'res': 6250, 'n_cols': 6413, 'n_rows': 6378, 'lat_min': -85, 'lat_max': 85},
         }


PROJECTIONS = {
    'G': "+proj=cea +lat_ts=30 +lon_0=0 +lat_0=0 "
         "+x_0=0 +y_0=0 +datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs",
    'N': "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 "
         "+datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs",
    'S': "+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 "
         "+datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs",
    'PS_N': "+proj=stere +lat_0=90 +lon_0=-45 +lat_ts=70 +k=1 +x_0=0 +y_0=0 "
                "+datum=WGS84 +ellps=WGS84 +units=m +no_defs",
    'PS_S': "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
                "+datum=WGS84 +ellps=WGS84 +units=m +no_defs",
    'UPS_N': "+proj=stere +lat_0=90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 "
             "+datum=WGS84 +units=m +no_defs +type=crs",
    'UPS_S': "+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 "
             "+datum=WGS84 +units=m +no_defs +type=crs",
    'MERC_G': "+proj=merc +k=1 +lon_0=0 +x_0=0 +y_0=0 "
              "+datum=WGS84 +units=m +no_defs +type=crs"
}


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

    def __init__(self, config_object, projection_definition, grid_definition):  # , logger = None):
        """
        Initializes the GridGenerator object with the configuration object.

        Parameters
        ----------
        config_object: xml.etree.ElementTree.Element
            Root element of the configuration file
        """

        self.config = config_object

        # If config_object is None, then it won't have logger as attribute 
        if config_object is not None: 
            if config_object.logger is not None: 
                self.logger = config_object.logger 
            self.decorate = config_object.logpar_decorate  
        else:
            # No formatting will be performed 
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler()) 
            self.decorate = False 


        self.projection_definition = projection_definition
        self.grid_definition = grid_definition
        self.projection = PROJECTIONS[self.projection_definition]
        self.resolution = GRIDS[grid_definition]['res']  # m
        self.n_cols = GRIDS[grid_definition]['n_cols']
        self.n_rows = GRIDS[grid_definition]['n_rows']
        self.x_min = GRIDS[grid_definition]['x_min']
        self.x_max = GRIDS[grid_definition]['x_min'] + self.resolution * self.n_cols
        self.y_min = GRIDS[grid_definition]['y_max'] - self.resolution * self.n_rows
        self.y_max = GRIDS[grid_definition]['y_max']
        self.grid_area = None


    def generate_grid_xy_ease2(self, return_resolution: bool = False
                               ) -> tuple[np.ndarray | float, np.ndarray | float] | \
                                    tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
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

        res = GRIDS[self.grid_definition]['res']
        n_cols = GRIDS[self.grid_definition]['n_cols']
        n_rows = GRIDS[self.grid_definition]['n_rows']
        # EASE Grids are by convention defined from the center of the pixel
        x_min = GRIDS[self.grid_definition]['x_min'] + res / 2
        y_max = GRIDS[self.grid_definition]['y_max'] - res / 2
        xs = np.zeros((n_cols, 1))

        for i in range(n_cols):
            xs[i] = x_min + (i * res)

        ys = np.zeros((n_rows, 1))

        for i in range(n_rows):
            ys[i] = y_max - (i * res)

        if return_resolution:
            return xs, ys, res

        return xs, ys


    def generate_grid_xy_stereo(self, return_resolution: bool = False
                               ) -> tuple[np.ndarray | float, np.ndarray | float] | \
                                    tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
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

        res = GRIDS[self.grid_definition]['res']
        n_cols = GRIDS[self.grid_definition]['n_cols']
        n_rows = GRIDS[self.grid_definition]['n_rows']
        # EASE Grids are by convention defined from the center of the pixel
        x_min = GRIDS[self.grid_definition]['x_min'] + res / 2
        y_max = GRIDS[self.grid_definition]['y_max'] - res / 2
        xs = np.zeros((n_cols, 1))

        for i in range(n_cols):
            xs[i] = x_min + (i * res)

        ys = np.zeros((n_rows, 1))

        for i in range(n_rows):
            ys[i] = y_max - (i * res)

        if return_resolution:
            return xs, ys, res

        return xs, ys


    def generate_grid_xy_mercator(self, return_resolution: bool = False
                               ) -> tuple[np.ndarray | float, np.ndarray | float] | \
                                    tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:

        mercator_proj = pyproj.Proj(PROJECTIONS[self.projection_definition])

        # Define bounding box in degrees
        min_lon, max_lon = -180, 180  # Longitude range in degrees
        min_lat, max_lat = GRIDS[self.grid_definition]['lat_min'], GRIDS[self.grid_definition]['lat_max']  # Latitude range in degrees (Mercator excludes poles)

        # Define resolution in meters
        resolution_m = GRIDS[self.grid_definition]['res']  # Set the desired grid resolution in meters (e.g., 10 km)

        # Convert bounding box to Mercator x, y using pyproj
        min_x, min_y = mercator_proj(min_lon, min_lat)
        max_x, max_y = mercator_proj(max_lon, max_lat)

        # Calculate the number of points to ensure equal spacing
        num_x = int((max_x - min_x) / resolution_m) + 1  # Number of points in the x-direction
        num_y = int((max_y - min_y) / resolution_m) + 1  # Number of points in the y-direction

        # Generate x (longitude) and y (latitude) arrays in meters
        x = np.linspace(min_x, max_x, num_x)  # Evenly spaced x-coordinates
        y = np.linspace(max_y, min_y, num_y)  # Evenly spaced y-coordinates, top-to-bottom

        # Create 2D grid
        grid_x, grid_y = np.meshgrid(x, y)

        if return_resolution: 
            return x, y, resolution_m

        return x, y


    def generate_grid_xy(self, return_resolution: bool = False
                               ) -> tuple[np.ndarray | float, np.ndarray | float] | \
                                    tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
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

        if "EASE2" in self.grid_definition:
            #result = self.generate_grid_xy_ease2(return_resolution=return_resolution)
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.generate_grid_xy_ease2)
            result = tracked_func(return_resolution=return_resolution)

        elif "STEREO" in self.grid_definition:
            #result = self.generate_grid_xy_stereo(return_resolution=return_resolution)
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.generate_grid_xy_stereo)
            result = tracked_func(return_resolution=return_resolution)

        elif "MERC" in self.grid_definition:
            return_resolution=False
            #result = self.generate_grid_xy_mercator(return_resolution=return_resolution)
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.generate_grid_xy_mercator)
            result = tracked_func(return_resolution=return_resolution)

        else:
            raise NotImplementedError(f"Grid {self.grid_definition} is not implemented.")


        if return_resolution:
            return result[0], result[1], result[2]

        return result[0], result[1]



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

        #xs, ys = self.generate_grid_xy(return_resolution=False)
        tracked_func  = RGBLogging.rgb_decorate_and_execute(
            decorate  = self.decorate, 
            decorator = RGBLogging.track_perf, 
            logger    = self.logger 
            )(self.generate_grid_xy)
        xs, ys = tracked_func(return_resolution=False)

        grid_x, grid_y = np.meshgrid(xs, ys)

        lons, lats = pyproj.Proj(self.projection)(grid_x, grid_y, inverse=True)

        return lons, lats


    def lonlat_to_xy_laea(self, lon: np.ndarray | float, lat: np.ndarray | float, pole: str = 'N'
                               ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Lambert's Azimuthal Equal Area (North and South)
        """

        # Assigning the sign value (+/-) depending on the pole
        sign = -1 if pole == 'N' else 1 if pole == 'S' \
            else (lambda: (_ for _ in ()).throw(
            ValueError("Invalid value for pole. Must be 'N' or 'S'.")
        ))

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

        qp = 1 - (
                ((1 - E2) / (2 * E)) * np.log((1 - E) / (1 + E))
        )

        pole_diff = abs(qp + sign * q)
        inds = pole_diff >= epsilon
        rho = MAP_EQUATORIAL_RADIUS * np.sqrt(qp + sign * q) * inds

        x = rho * np.sin(lam)
        y = sign * rho * np.cos(lam)

        return x, y


    def lonlat_to_xy_cea(self,
                         lon: np.ndarray | float,
                         lat: np.ndarray | float
                         ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Cylindrical Equal Area (Global)
        """

        # print(f"lonlat_to_xy_cea: lon.shape = {lon.shape}")

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

        # qp = q(phi = 90)
        qp = 1 - (
                ((1 - E2) / (2 * E)) * np.log((1 - E) / (1 + E))
        )

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

        # print(f"lonlat_to_xy_cea: x.shape = {x.shape}")

        return x, y


    def lonlat_to_xy_stereo(self,
                            lon: np.ndarray | float,
                            lat: np.ndarray | float,
                            pole: str = 'N'
                            ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Polar Stereographic projection using pyproj library:
        https://proj.org/en/9.5/operations/projections/stere.html

        Spatial Reference:
        https://spatialreference.org/ref/epsg/3413/
        https://spatialreference.org/ref/epsg/3413/proj4.txt
        """

        # print(f"lonlat_to_xy_stereo: lon.shape = {lon.shape}")

        if pole == 'N':
            projection = PROJECTIONS['PS_N']
        elif pole == 'S':
            projection = PROJECTIONS['PS_S']
        else:
            ValueError("Invalid value for pole. Must be 'N' or 'S'.")

        projection = pyproj.Proj(projection)
        x, y = projection(longitude=lon, latitude=lat)

        return x, y


    def lonlat_to_xy_ups(self,
                         lon: np.ndarray | float,
                         lat: np.ndarray | float,
                         pole: str = 'N'
                         ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Universal Polar Stereographic projection using pyproj library:
        https://proj.org/en/9.5/operations/projections/ups.html

        Spatial Reference:
        - N,E:
            https://spatialreference.org/ref/epsg/32661/
            https://spatialreference.org/ref/epsg/32661/proj4.txt
        - S,E:
            https://spatialreference.org/ref/epsg/32761/
            https://spatialreference.org/ref/epsg/32761/proj4.txt
        """

        if pole == 'N':
            projection = PROJECTIONS['UPS_N']
        elif pole == 'S':
            projection = PROJECTIONS['UPS_S']
        else:
            ValueError("Invalid value for pole. Must be 'N' or 'S'.")

        projection = pyproj.Proj(projection)
        x, y = projection(longitude=lon, latitude=lat)

        return x, y


    def lonlat_to_xy_merc(self,
                          lon: np.ndarray | float,
                          lat: np.ndarray | float
                         ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        (Global) Mercator projection using pyproj library:
        https://proj.org/en/9.5/operations/projections/merc.html

        Spatial Reference:
        https://spatialreference.org/ref/esri/54004/
        https://spatialreference.org/ref/esri/54004/proj4.txt
        """

        projection = PROJECTIONS['MERC_G']
        projection = pyproj.Proj(projection)
        x, y = projection(longitude=lon, latitude=lat)

        return x, y


    def lonlat_to_xy(self,
                     lon: np.ndarray | float,
                     lat: np.ndarray | float
                     ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Generic method. Converts longitude and latitude coordinates to x and y
        coordinates for a given projection.

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

        if self.projection_definition == 'G':

            #x, y = self.lonlat_to_xy_cea(lon=lon, lat=lat)
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.lonlat_to_xy_cea)
            x, y = tracked_func(lon=lon, lat=lat)

        elif self.projection_definition == 'N':

            #x, y = self.lonlat_to_xy_laea(lon=lon, lat=lat, pole='N')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.lonlat_to_xy_laea)
            x, y = tracked_func(lon = lon, lat = lat, pole = 'N')

        elif self.projection_definition == 'S':

            #x, y = self.lonlat_to_xy_laea(lon=lon, lat=lat, pole='S')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.lonlat_to_xy_laea)
            x, y = tracked_func(lon = lon, lat = lat, pole = 'S')

        elif self.projection_definition == 'PS_N':

            #x, y = self.lonlat_to_xy_stereo(lon=lon, lat=lat, pole='N')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.lonlat_to_xy_stereo)
            x, y = tracked_func(lon=lon, lat=lat, pole = 'N')

        elif self.projection_definition == 'PS_S':

            #x, y = self.lonlat_to_xy_stereo(lon=lon, lat=lat, pole='S')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.lonlat_to_xy_stereo)
            x, y = tracked_func(lon=lon, lat=lat, pole = 'S')

        elif self.projection_definition == 'UPS_N':

            #x, y = self.lonlat_to_xy_ups(lon=lon, lat=lat, pole='N')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.lonlat_to_xy_ups)
            x, y = tracked_func(lon=lon, lat=lat, pole = 'N')

        elif self.projection_definition == 'UPS_S':

            #x, y = self.lonlat_to_xy_ups(lon=lon, lat=lat, pole='S')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.lonlat_to_xy_ups)
            x, y = tracked_func(lon=lon, lat=lat, pole = 'S')

        elif self.projection_definition == 'MERC_G':

            #x, y = self.lonlat_to_xy_merc(lon=lon, lat=lat)
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.lonlat_to_xy_merc)
            x, y = tracked_func(lon=lon, lat=lat)

        else:
            raise NotImplementedError(f"Invalid projection code.")

        return x, y


    def xy_to_lonlat_laea(self,
                          x: np.ndarray | float,
                          y: np.ndarray | float,
                          pole: str
                         ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Inverse of Lambert's Azimuthal Equal Area (North and South) projection
        """

        # Assigning the sign value (+/-) depending on the pole
        sign = 1 if pole == 'N' else -1 if pole == 'S' \
            else (lambda: (_ for _ in ()).throw(
            ValueError("Invalid value for pole. Must be 'N' or 'S'.")
        ))

        params = self.projection.split()
        lon_0 = float(next((param.split('=')[1] for param in params if 'lon_0' in param), None))

        E4 = E ** 4
        E6 = E ** 6
        qp = 1 - (((1 - E2) / (2 * E)) * np.log((1 - E) / (1 + E)))
        beta = None
        lam = None

        rho = np.sqrt(x ** 2 + y ** 2)
        beta = sign * np.arcsin(1 - (rho ** 2 / (MAP_EQUATORIAL_RADIUS ** 2 * qp)))
        lam = np.arctan2(x, sign * (-y))

        phi = (beta +
               ((E2 / 3) + (31 * E4 / 180) + (517 * E6 / 5040)) * np.sin(2 * beta) +
               ((23 * E4 / 360) + (251 * E6 / 3780)) * np.sin(4 * beta) +
               (761 * E6 / 45360) * np.sin(6 * beta))

        lat = np.rad2deg(phi)
        lon = lon_0 + np.rad2deg(lam)

        return lon, lat


    def xy_to_lonlat_cea(self,
                         x: np.ndarray | float,
                         y: np.ndarray | float
                         ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Inverse of Cylindrical Equal Area (Global) projection.
        """

        params = self.projection.split()
        lon_0 = float(next((param.split('=')[1] for param in params if 'lon_0' in param), None))

        E4 = E ** 4
        E6 = E ** 6
        qp = 1 - (((1 - E2) / (2 * E)) * np.log((1 - E) / (1 + E)))
        beta = None
        lam = None

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

        phi = (beta +
               ((E2 / 3) + (31 * E4 / 180) + (517 * E6 / 5040)) * np.sin(2 * beta) +
               ((23 * E4 / 360) + (251 * E6 / 3780)) * np.sin(4 * beta) +
               (761 * E6 / 45360) * np.sin(6 * beta))

        lat = np.rad2deg(phi)
        lon = lon_0 + np.rad2deg(lam)

        return lon, lat


    def xy_to_lonlat_stereo(self,
                            x: np.ndarray | float,
                            y: np.ndarray | float,
                            pole: str = 'N'
                            ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Polar Stereographic projection using pyproj library:
        https://proj.org/en/9.5/operations/projections/stere.html

        Spatial Reference:
        https://spatialreference.org/ref/epsg/3413/
        https://spatialreference.org/ref/epsg/3413/proj4.txt
        """

        if pole == 'N':
            projection = PROJECTIONS['PS_N']
        elif pole == 'S':
            projection = PROJECTIONS['PS_S']
        else:
            ValueError("Invalid value for pole. Must be 'N' or 'S'.")

        projection = pyproj.Proj(projection)
        lon, lat = projection(x, y, inverse=True)

        return lon, lat


    def xy_to_lonlat_ups(self,
                         x: np.ndarray | float,
                         y: np.ndarray | float,
                         pole: str = 'N'
                         ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        Universal Polar Stereographic projection using pyproj library:
        https://proj.org/en/9.5/operations/projections/ups.html

        Spatial Reference:
        - N,E:
            https://spatialreference.org/ref/epsg/32661/
            https://spatialreference.org/ref/epsg/32661/proj4.txt
        - S,E:
            https://spatialreference.org/ref/epsg/32761/
            https://spatialreference.org/ref/epsg/32761/proj4.txt
        """

        if pole == 'N':
            projection = PROJECTIONS['UPS_N']
        elif pole == 'S':
            projection = PROJECTIONS['UPS_S']
        else:
            ValueError("Invalid value for pole. Must be 'N' or 'S'.")

        projection = pyproj.Proj(projection)
        lon, lat = projection(x, y, inverse=True)

        return lon, lat


    def xy_to_lonlat_merc(self,
                          x: np.ndarray | float,
                          y: np.ndarray | float
                          ) -> tuple[np.ndarray | float, np.ndarray | float]: 
        """
        (Global) Mercator projection using pyproj library:
        https://proj.org/en/9.5/operations/projections/merc.html

        Spatial Reference:
        https://spatialreference.org/ref/esri/54004/
        https://spatialreference.org/ref/esri/54004/proj4.txt
        """

        projection = PROJECTIONS['MERC_G']
        projection = pyproj.Proj(projection)
        lon, lat = projection(x, y, inverse=True)

        return lon, lat


    def xy_to_lonlat(self,
                     x: np.ndarray | float,
                     y: np.ndarray | float
                     ) -> tuple[np.ndarray | float, np.ndarray | float]: 
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

        if self.projection_definition == 'G':

            #lon, lat = self.xy_to_lonlat_cea(x=x, y=y)
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.xy_to_lonlat_cea)
            lon, lat = tracked_func(x=x, y=y)

        elif self.projection_definition == 'N':

            #lon, lat = self.xy_to_lonlat_laea(x=x, y=y, pole='N')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.xy_to_lonlat_laea)
            lon, lat = tracked_func(x=x, y=y, pole = 'N')

        elif self.projection_definition == 'S':

            #lon, lat = self.xy_to_lonlat_laea(x=x, y=y, pole='S')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.xy_to_lonlat_laea)
            lon, lat = tracked_func(x = x, y = y, pole = 'S')

        elif self.projection_definition == 'PS_N':

            #lon, lat = self.xy_to_lonlat_stereo(x=x, y=y, pole='N')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.xy_to_lonlat_stereo)
            lon, lat = tracked_func(x = x, y = y, pole = 'N')

        elif self.projection_definition == 'PS_S':

            #lon, lat = self.xy_to_lonlat_stereo(x=x, y=y, pole='S')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.xy_to_lonlat_stereo)
            lon, lat = tracked_func(x = x, y = y, pole = 'S')

        elif self.projection_definition == 'UPS_N':

            #lon, lat = self.xy_to_lonlat_ups(x=x, y=y, pole='N')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.xy_to_lonlat_ups)
            lon, lat = tracked_func(x = x, y = y, pole = 'N')

        elif self.projection_definition == 'UPS_S':

            #lon, lat = self.xy_to_lonlat_ups(x=x, y=y, pole='S')
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.xy_to_lonlat_ups)
            lon, lat = tracked_func(x = x, y = y, pole = 'S')

        elif self.projection_definition == 'MERC_G':

            #lon, lat = self.xy_to_lonlat_merc(x=x, y=y)
            tracked_func  = RGBLogging.rgb_decorate_and_execute(
                decorate  = self.decorate, 
                decorator = RGBLogging.track_perf, 
                logger    = self.logger 
                )(self.xy_to_lonlat_merc)
            lon, lat = tracked_func(x = x, y = y)

        else:
            raise NotImplementedError(f"Invalid projection code.")

        return lon, lat

        # def xy_to_lonlat(self, x, y):



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

        n_cols = self.n_cols
        n_rows = self.n_rows
        r0 = (n_cols - 1) / 2
        s0 = (n_rows - 1) / 2
        col = r0 + (x / res)
        row = s0 - (y / res)

        # Make a check that there is nothing out of range
        # TBD

        return row, col


    def rowcol_to_xy(self, row, col):

        _, _, res = self.generate_grid_xy(return_resolution=True)
        
        n_cols = self.n_cols
        n_rows = self.n_rows
        r0 = (n_cols - 1) / 2
        s0 = (n_rows - 1) / 2
        x  = res*(col-r0)
        y  = res*(s0-row)

        return x, y


    def rowcol_to_lonlat(self, row, col):

        _, _, res = self.generate_grid_xy(return_resolution=True)

        n_cols = self.n_cols
        n_rows = self.n_rows
        r0 = (n_cols - 1) / 2
        s0 = (n_rows - 1) / 2
        x  = res*(col-r0)
        y  = res*(s0-row)

        return self.xy_to_lonlat(x, y)



    def get_grid_area(self):

        if self.grid_area is None:

            resolution = self.resolution

            if "EASE2" in self.grid_definition:
                grid_area = resolution**2
                grid_area = np.ones((self.n_rows, self.n_cols)) * grid_area

            elif "STEREO" in self.grid_definition:

                import netCDF4 as nc

                #target_dir = os.path.join(os.path.dirname(os.getcwd()), "dpr/Grids/NSIDC_PS")

                # Access the target directory within the package
                target_dir = pkg_resources.files('cimr_rgb.dpr.Grids.NSIDC_PS')  # Adjust the package path if necessary

                # Iterate through files in the target directory
                for file in target_dir.iterdir():
                    if f"{self.projection_definition}{str(int(resolution/1000))}" in file.name:
                        # Open the NetCDF file
                        with nc.Dataset(file) as dataset:
                            grid_area = np.array(dataset.variables['cell_area'][:])
                        break

                # Open netcdf file
                #for file in os.listdir(target_dir):
                #    if f"{self.projection_definition}{str(int(resolution/1000))}" in file:
                #        # open file with netcdf
                #        dataset = nc.Dataset(os.path.join(target_dir, file))
                #        grid_area = np.array(dataset.variables['cell_area'][:])
                #        dataset.close()
                #        break

            elif "MERC" in self.grid_definition:
                # Todo: calculation of actual mercator grid area
                grid_area = resolution**2
                grid_area = np.ones((self.n_rows, self.n_cols)) * grid_area

            self.grid_area = grid_area

        return self.grid_area
