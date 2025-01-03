import logging
import logging.config as logconfig
import re
import pathlib as pb
import datetime
import itertools as it

import numpy as np
import netCDF4 as nc
import pyproj
from shapely.geometry import Polygon


from cimr_rgb.grid_generator import GRIDS, PROJECTIONS

# TODO: Ad rows and cols variables <= all variables are
# in 1D flattened array and these rows and cols will allow
# the user to build the grid.

# For CIMR:
# rows are # of samples
# cols are
# z dim are the scans

# ncdump does not display chunking information by default, so we adding it here as metadata
CDL = {
    "LOCAL_ATTRIBUTES": {
        "Measurement": {
            # TODO: Change these number to int32 (now it is int64)
            "cell_col": {
                "units": "Grid y-coordinate",
                "long_name": "Grid column index for the chosen output grid",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,2147483647",  # depends on the variable type
                # "_Storage": "",
                # "_ChunkSizes": "",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Grid row index for the chosen output grid. This variable is used to reconstruct the chosen output grid.",
            },
            "cell_row": {
                "units": "Grid x-coordinate",
                "long_name": "Grid row Index for the chosen output grid",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,2147483647",  # depends on the variable type
                # "_Storage": "",
                # "_ChunkSizes": "",
                # "_FillValue": nc.default_fillvals['f8'], # Int
                "comment": "Grid row Index for the chosen output grid. This variable is used to reconstruct the chosen output grid.",
            },
            "bt_h": {
                "units": "K",
                "long_name": "H-polarised TOA Brightness Temperatures",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,2147483647",  # depends on the variable type
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Earth-Gridded TOA h-polarised"
                + " [L|C|X|KU|KA]_BAND_[fore|aft] BTS"
                + " interpolated on a TBD-km grid",
                # L1C: self.config.grid_definition
                # L1R: self.config.target_band[0] <= add this instead of TBD-km
            },
            "bt_v": {
                "units": "K",
                "long_name": "V-polarised TOA Brightness Temperatures",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,2147483647",  # depends on the variable type
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Earth-Gridded TOA v-polarised"
                + " [L|C|X|KU|KA]_BAND_[fore|aft] BTS"
                + " interpolated on a TBD-km grid",
            },
            "bt_3": {
                "units": "K",
                "long_name": "Stokes 3-polarised TOA Brightness Temperatures",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Earth-Gridded TOA [L|C|X|KU|KA]_BAND_[fore|aft] BTS "
                + "interpolated on a TBD-km grid, third stokes parameter "
                + "of the surface polarisation basis",
            },
            "bt_4": {
                "units": "K",
                "long_name": "Stokes 4-polarised TOA Brightness Temperatures",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Earth-Gridded TOA [L|C|X|KU|KA]_BAND_[fore|aft] BTS "
                + "interpolated on a TBD-km grid, fourth stokes parameter "
                + "of the surface polarisation basis",
            },
            "faraday_rot_angle": {
                "units": "deg",
                "long_name": "Interpolated Faraday Rotation Angle of acquisitions",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] faraday "
                + "rotation angle corresponding to the measured BT value. "
                + "The value of the faraday rotation angle will be scaled "
                + "with the interpolation weights of all faraday rotation "
                + "angles Earth samples used in the interpolation of that "
                + "grid cell.",
            },
            "geometric_rot_angle": {
                "units": "deg",
                "long_name": "Interpolated Geometric Rotation angle of acquisitions.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] geometric "
                + "rotation angle corresponding to the measured BT value. "
                + "The value of the geometric rotation angle will be "
                + "scaled with the interpolation weights of all geometric "
                + "rotation angles Earth samples used in the "
                + "interpolation of that grid cell.",
            },
            "nedt_h": {
                "units": "K",
                "long_name": "Radiometric resolution of each measured BT.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Radiometric resolution of each measured BT.",
            },
            "nedt_v": {
                "units": "K",
                "long_name": "Radiometric resolution of each measured BT.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Radiometric resolution of each measured BT.",
            },
            "nedt_3": {
                "units": "K",
                "long_name": "Radiometric resolution of each measured BT.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Radiometric resolution of each measured BT.",
            },
            "nedt_4": {
                "units": "K",
                "long_name": "Radiometric resolution of each measured BT.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Radiometric resolution of each measured BT.",
            },
            "tsu": {
                "units": "K",
                "long_name": "Total standard uncertainty for each measured BT.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Total standard uncertainty for each measured BT.",
            },
            "instrument_status": {
                "units": "N/A",
                "long_name": "Instrument Calibration or Observation mode.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Instrument Calibration or Observation mode, "
                + "for all samples. L1c values will consider the majority "
                + "status values from input L1b samples.",
            },
            "land_sea_content": {
                "units": "N/A",
                "long_name": "Land/Sea content of the measured pixel.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Land/Sea content of the measured pixel, "
                + "200 for full sea content, 0 for full land content.",
            },
            "regridding_n_samples": {
                "units": "N/A",
                "long_name": "Number of earth samples used for interpolation",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Number of L1b [h|v|t3|t4] polarised "
                + "[L|C|X|KU|KA]_BAND_[fore|aft] brightness temperature "
                + "Earth samples used in the [Backus-Gilbert|rSIR|LW] "
                + "remapping interpolation.",
            },
            "regridding_quality_measure": {
                "units": "N/A",
                "long_name": "Algorithm Specific Optimal Value of Regularization Parameter.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "The optimal value of a parameter for the "
                + "[Backus-Gilbert|rSIR|LW]_[fore|aft] that controls the "
                + "trade-off between noise amplification and "
                + "regularisation. For BG it is the optimal value for the "
                + "smoothing parameter, while for [rSIR|LW] it is the "
                + "number of iterations to achieve a chosen level of "
                + "residual error. In case of [NN|IDS|DIB] regularisation "
                + "is not performed and the parameter will take on the "
                + "_FillValue.",
            },
            "regridding_l1b_orphans": {
                "units": "N/A",
                "long_name": "Indication of L1b orphaned Earth samples.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "TBD",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Whether each [L|C|X|KU|KA]_BAND L1b measurement sample was "
                + "unused (1) or used (0) in [Backus-Gilbert|rSIR|LW] regridding "
                + "interpolation of [fore|aft] scan samples. In the fore-scan "
                + "regridding nearly all aft scan samples would be orphan "
                + "(unused), for instance, and vice versa. It would also occur if "
                + "the swath stretches outside the projection window. Orphaned "
                + "samples may also occur if nearest neighbour or linear "
                # + "interpolation (among the TBD methods) is used.",
                + "interpolation is used.",
            },
        },
        "Navigation": {
            "acq_time_utc": {
                "units": "N/A",
                "long_name": "Interpolated UTC Acquisition time of Earth Sample acquisitions.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,N/A",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "UTC acquisition times expressed in seconds "
                + "(seconds since 2000-01-01 00:00:00 UTC). The value of "
                + "time_earth will be scaled with the interpolation "
                + "weights of all time_earth Earth samples used in the "
                + "interpolation of that grid cell. ",
            },
            "azimuth": {
                "units": "deg",
                "long_name": "Interpolated Earth Azimuth angle of the acquisitions.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,359.99",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] "
                + "Earth observation azimuth angles of the acquisitions, "
                + "positive counterclockwise from due east. The value of "
                + "observation azimuth angle will be scaled with the "
                + "interpolation weights of all observation azimuth angle "
                + "Earth samples used in the interpolation of that grid "
                "cell.",
            },
            "latitude": {
                "units": "deg",
                "long_name": "Latitude of the centre of a TBD-km PROJ grid cell.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "-90, 90",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Latitude of the centre of a TBD-km PROJ grid cell.",
                # L1C: self.config.grid_definition
                # L1R: self.config.target_band[0] <= add this instead of TBD-km  Boresight location of the self.config.target_band[0] footprint
            },
            "longitude": {
                "units": "deg",
                "long_name": "Longitude of the centre of a TBD-km PROJ grid cell.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "-180, 179.99",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Longitude of the centre of a TBD-km PROJ grid cell.",
            },
            "oza": {
                "units": "deg",
                "long_name": "Interpolated Observation Zenith Angle of acquisitions.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,359.99",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] "
                + "Earth Observation zenith angles of the acquisitions. "
                + "The value of OZA will be scaled with the interpolation "
                + "weights of all observation OZA Earth samples used in "
                + "the interpolation of that grid cell. The OZA is defined "
                + "as the included angle between the antenna Boresight "
                + "vector and the normal to the Earth's surface.",
            },
            "processing_scan_angle": {
                "units": "deg",
                "long_name": "Interpolated scan angle of acquisitions",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,359.99",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "The processing scan angle of the L1b "
                + "[L|C|X|KU|KA]_BAND_[fore|aft] Earth view samples. The "
                + "value of scan angle will be scaled with the "
                + "interpolation weights of all scan angle Earth samples "
                + "used in the interpolation of that grid cell. "
                + "Measurements from different feed horns are combined. "
                + "The scan angle is defined as the azimuth angle of the "
                + "antenna boresight measured from the ground track "
                + "vector. The scan angle is 90° when the boresight points "
                + "in the same direction as the ground track vector and "
                + "increases clockwise when viewed from above.",
            },
            "solar_azimuth": {
                "units": "deg",
                "long_name": "Interpolated solar azimuth angle of acquisitions",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,359.99",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] "
                + "solar azimuth angle of acquisitions.The value of "
                + "solar_azimuth will be scaled with the interpolation "
                + "weights of all solar_azimuth Earth samples used in the "
                + "interpolation of that grid cell.",
            },
            "solar_zenith": {
                "units": "deg",
                "long_name": "Interpolated solar zenith angle of acquisitions",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,359.99",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] "
                + "solar zenith angle of acquisitions.The value of "
                + "solar_zenith will be scaled with the interpolation "
                + "weights of all solar_zenith Earth samples used in the "
                + "interpolation of that grid cell.",
            },
        },
        "Processing_flags": {
            "processing_flags": {
                "units": "N/A",
                "long_name": "L1c processing performance related information.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,65535",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": "0",#nc.default_fillvals['f8'],
                # "comment": "A TBD-bit binary string of 1’s and 0’s "
                # + "indicating a variety of TBD information related to the "
                "comment": "A binary string of 1’s and 0’s "
                + "providing information related to the "
                + "processing of L1c/r data.",
                # TODO: TBD-bit should either be 8 bit or 16 bits
            }
        },
        "Quality_information": {
            "calibration_flag": {
                "units": "K",
                # "long_name": "A TBD-bit binary string of 1’s and 0’s indicating the quality "
                # + "of the L1b [L|C|X|KU|KA]_BAND Earth samples used to derive the "
                # + "L1c data. A ‘0’ indicates that the L1c samples met a certain "
                # + "quality criterion and a ‘1’ that it did not. Bit position ‘0’ "
                # + "refers to the least significant bit. The calibration flag "
                # + "summarises the calibration quality for each channel and scan. "
                # + "The data quality flag summarises the BT data quality for each "
                # + "channel and scan.",
                "long_name": "A 16-bit binary string of 1’s and 0’s indicating the quality "
                + "of the L1b [L|C|X|KU|KA]_BAND Earth samples used to derive the "
                + "L1c data.",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,65535",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256,256",
                # "_FillValue": "0"
                "comment": "A 16-bit binary string of 1’s and 0’s indicating the quality "
                + "of the L1b [L|C|X|KU|KA]_BAND Earth samples used to derive the "
                + "L1c data. A ‘0’ indicates that the L1c samples met a certain "
                + "quality criterion and a ‘1’ that it did not. Bit position ‘0’ "
                + "refers to the least significant bit. The calibration flag "
                + "summarises the calibration quality for each channel and scan. "
                + "The data quality flag summarises the BT data quality for each "
                + "channel and scan.",
                # TODO: put 16 or remove it.
            },
            "data_quality_flag": {
                "units": "K",
                # "long_name": "A TBD-bit binary string of 1’s and 0’s indicating the quality "
                # + "of the L1b [L|C|X|KU|KA]_BAND Earth samples used to derive the "
                # + "L1c data. A ‘0’ indicates that the L1c samples met a certain "
                # + "quality criterion and a ‘1’ that it did not. Bit position ‘0’ "
                # + "refers to the least significant bit. The calibration flag "
                # + "summarises the calibration quality for each channel and scan. "
                # + "The data quality flag summarises the BT data quality for each "
                # + "channel and scan.",
                "long_name": "",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,65535",
                # "_Storage": "chunked",
                # "_ChunkSizes": "256,256",
                # "_FillValue": "0"
                "comment": "A 16-bit binary string of 1’s and 0’s indicating the quality "
                + "of the L1b [L|C|X|KU|KA]_BAND Earth samples used to derive the "
                + "L1c data. A ‘0’ indicates that the L1c samples met a certain "
                + "quality criterion and a ‘1’ that it did not. Bit position ‘0’ "
                + "refers to the least significant bit. The calibration flag "
                + "summarises the calibration quality for each channel and scan. "
                + "The data quality flag summarises the BT data quality for each "
                + "channel and scan.",
            },
        },
    },
    # "L1C_SPECIFIC_ATTRIBUTES": {
    #     "cell_col": {
    #         "units": "Grid y-coordinate",
    #         "long_name": "Grid column Index for the chosen output grid",
    #         "grid_mapping": "crs",
    #         "coverage_content_type": "Grid",
    #         "valid_range": "TBD",
    #         "_Storage": "chunked",
    #         "_ChunkSizes": "256, 256",
    #         #"_FillValue": nc.default_fillvals['f8'],
    #         "comment": "Grid column Index for the chosen output grid"
    #     },
    #     "cell_row": {
    #         "units": "Grid x-coordinate",
    #         "long_name": "Grid row Index for the chosen output grid",
    #         "grid_mapping": "crs",
    #         "coverage_content_type": "Grid",
    #         "valid_range": "TBD",
    #         "_Storage": "chunked",
    #         "_ChunkSizes": "256, 256",
    #         #"_FillValue": nc.default_fillvals['f8'], # Int
    #         "comment": "Grid row Index for the chosen output grid"
    #     },
    # },
    # "GLOBAL_ATTRIBUTES" : {
    #     "conventions": "CF-1.6",
    #     "id": "TBD",
    #     "naming_authority": "European Space Agency",
    #     "history": "TBD",
    #     "source": "TBD",
    #     "processing_level": "TBD", #f"{self.config.grid_type}",
    #     "comment": "TBD",
    #     "acknowledgement": "TBD",
    #     "license": "None",
    #     "standard_name_vocabulary": "TBD",
    #     "date_created": "TBD", #f"{l1c_utc_time}",
    #     "creator_name": "TBD",
    #     "creator_email": "TBD",
    #     "creator_url": "TBD",
    #     "institution": "European Space Agency",
    #     "project": "CIMR Re-Gridding Toolbox",
    #     "program": "TBD",
    #     "contributor_name": "TBD",
    #     "contributor_role": "TBD",
    #     "publisher_name": "TBD",
    #     "publisher_email": "TBD",
    #     "publisher_url": "TBD",
    #     "geospatial_bounds": "TBD",
    #     "geospatial_bounds_crs": "TBD",
    #     "geospatial_bounds_vertical_crs": "TBD",
    #     "geospatial_lat_min": "TBD",
    #     "geospatial_lat_max": "TBD",
    #     "geospatial_lon_min": "TBD",
    #     "geospatial_lon_max": "TBD",
    #     "time_coverage_start": "TBD",
    #     "time_coverage_end": "TBD",
    #     "time_coverage_duration": "TBD",
    #     "time_coverage_resolution": "TBD",
    #     "geospatial_lat_units": "degrees north",
    #     "geospatial_lat_resolution": "TBD",
    #     "geospatial_lon_units": "degrees north",
    #     "geospatial_lon_resolution": "TBD",
    #     "date_modified": "TBD",
    #     "date_issued": "TBD",
    #     "date_metadata_modified": "TBD",
    #     "product_version": "TBD",
    #     "platform": "TBD", #f"{self.config.input_data_type}",
    #     "instrument": "TBD", #f"{self.config.input_data_type}",
    #     "metadata_link": "TBD",
    #     "keywords": "TBD",
    #     "keywords_vocabulary": "TBD",
    #     "references": "TBD",
    #     "input_level1b_filenames": "TBD", #f"{pb.Path(self.config.input_data_path).resolve().name}", #"TBD",
    #     "level_01_atbd": "TBD",
    #     "mission_requirement_document": "TBD",
    #     "antenna_pattern_files": "TBD",
    #     "antenna_pattern_source": "TBD"
    # }
}


class ProductGenerator:
    def __init__(self, config):
        self.config = config

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

    # TODO: We face the following problem: the global dimensions should
    #       coincide with local ones for the variable names. Therefore,
    #       I need to loop through bands and determine how many dimensions
    #       should be in there, e.g. if we have L and X bands, then we
    #       have n_feeds_L_BAND and n_feeds_X_BANDS only, if there are
    #       more then there will be more. Hence, I will create a separate
    #       method to work with global metadata and a separate on to work
    #       with local ones.

    # Getting netCDF dataset
    def generate_global_metadata(self, data_dict: dict) -> dict:
        """
        geospatial_lat_min and geospatial_lat_max:

            These attributes define the minimum and maximum latitude values (in degrees) that the data covers.
            They indicate the vertical spatial extent of the dataset in terms of geographic coordinates.

        geospatial_lon_min and geospatial_lon_max:

            These attributes define the minimum and maximum longitude values (in degrees) that the data covers.
            They indicate the horizontal spatial extent of the dataset in terms of geographic coordinates.

        geospatial_bounds:

            This attribute specifies the spatial bounds of the dataset as a polygon, typically formatted as POLYGON((lon1 lat1, lon2 lat2, ..., lonN latN, lon1 lat1)).
            It provides the exact spatial geometry for the region covered by the data.

        geospatial_bounds_crs:

            Defines the coordinate reference system (CRS) used for the geospatial_bounds attribute.
            For example, it could be EPSG:4326, which corresponds to the WGS84 geographic coordinate system.

        geospatial_bounds_vertical_crs:

            Specifies the vertical coordinate reference system, if the dataset includes altitude or depth information.
            Common examples include EGM96 (Earth Gravitational Model 1996) or other vertical datums.
        """

        # Convert the datetime object to ISO 8601 format
        # Converting the date defined in config.xml into ISO 8601 formatted date
        custom_date = f"{self.config.timestamp}"

        # Parse the custom formatted date string into a datetime object
        date_obj = datetime.datetime.strptime(custom_date, self.config.timestamp_fmt)
        # We use the current time to set the timestamp
        iso_formatted_date = date_obj.isoformat() + "Z"  # Add 'Z' for UTC timezone

        GLOBAL_ATTRIBUTES = {
            "conventions": "CF-1.6",
            # "id": "TBD",
            "naming_authority": "European Space Agency",
            # "history": "TBD",
            # "source": "TBD",
            "processing_level": f"{self.config.grid_type}",
            "comment": f"Test data set output that represents an example {self.config.grid_type} product for evaluation of {self.config.input_data_type} instrument",
            # "acknowledgement": "TBD",
            "license": "MIT",
            # "standard_name_vocabulary": "TBD",
            "date_created": f"{iso_formatted_date}",  # f"{self.config.timestamp}",
            "creator_name": f"{self.config.creator_name}",  # "Maksym Brilenkov",
            "creator_email": f"{self.config.creator_email}",  # "brilenkov@strcorp.no",
            "creator_url": f"{self.config.creator_url}",  # "https://www.stcorp.no/",
            "creator_institution": f"{self.config.creator_institution}",  # "Science and Technology",
            "project": "CIMR Re-Gridding toolBox (RGB)",
            # "program": "TBD",
            # "contributor_name": "TBD",
            # "contributor_role": "TBD",
            # "publisher_name": "TBD",
            # "publisher_email": "TBD",
            # "publisher_url": "TBD",
            # "time_coverage_start": "TBD",
            # "time_coverage_end": "TBD",
            # "time_coverage_duration": "TBD",
            # "time_coverage_resolution": "TBD",
            "date_modified": f"{iso_formatted_date}",  # f"{self.config.timestamp}",
            "date_issued": f"{iso_formatted_date}",  # f"{self.config.timestamp}",
            "date_metadata_modified": f"{iso_formatted_date}",  # f"{self.config.timestamp}",
            "product_version": f"{self.config.product_version}",
            "platform": f"{self.config.input_data_type}",
            "instrument": f"{self.config.input_data_type}",
            # "metadata_link": "TBD",
            "keywords": "satellites, passive microwave radiometry",
            # "keywords_vocabulary": "TBD",
            # "references": "TBD",
            "input_level1b_filename": f"{pb.Path(self.config.input_data_path).resolve().name}",  # "TBD",
            "level_01_atbd": "Level 0, 1 Algorithms Theoretical Baseline Document Description and Performance analysis, Thales Alenia Space, 20/06/2022",
            "mission_requirement_document": "Copernicus Imaging Microwave Radiometer (CIMR) Mission Requirements Document, version 5, ESA-EOPSM-CIMR-MRD-3236, 11/02/2023",
        }

        # -------------------------------------------------
        # Antenna Patterns's Attributes
        # -------------------------------------------------
        if self.config.regridding_algorithm not in ["IDS", "NN", "DIB"]:
            if self.config.input_data_type == "CIMR":
                beamfiles = self.get_cimr_antenna_patterns_list()
                GLOBAL_ATTRIBUTES["antenna_pattern_files"] = f"{beamfiles}"
                GLOBAL_ATTRIBUTES["antenna_pattern_source"] = (
                    "Antenna patterns were generated using CIMR-GRASP software. These "
                    + "were derived from the original antenna patterns provided by Thales "
                    + "Alenia Space in September 2022, where V-pol and H-pol complex "
                    + "amplitudes were assumed to be identical."
                )
            elif self.config.input_data_type == "AMSR2":
                GLOBAL_ATTRIBUTES["antenna_pattern_files"] = "None"
                GLOBAL_ATTRIBUTES["antenna_pattern_source"] = (
                    "Antenns patterns were generated using a set of internal simulation routines of CIMR RGB."
                )
            elif self.config.input_data_type == "SMAP":
                GLOBAL_ATTRIBUTES["antenna_pattern_files"] = (
                    "RadiometerAntPattern_170830_v011.h5"
                )
                GLOBAL_ATTRIBUTES["antenna_pattern_source"] = (
                    # "Antenns patterns were generated using a set of internal simulation routines of CIMR RGB."
                    "SMAP collaboration."
                )
            else:
                raise ValueError(
                    "Invalid input_data_type was provided. Can be one of the following: [CIMR, AMSR2, SMAP]."
                )
        else:
            GLOBAL_ATTRIBUTES["antenna_pattern_files"] = (
                f"Not used for {self.config.regridding_algorithm} algorithm."
            )
            GLOBAL_ATTRIBUTES["antenna_pattern_source"] = "None"

        # -------------------------------------------------
        # Geospatial Bounds: Getting CRS and other useful variables
        # Example: Compute for a specific grid
        # grid_name = self.config.grid_definition  # "EASE2_G1km"
        # grid = GRIDS[grid_name]
        # print(grid)
        # projection_string = PROJECTIONS[
        #    grid_name.split("_")[0]
        # ]  # Extract appropriate PROJECTION

        # TODO: Uncomment this to have POLYGONs configured
        # geospatial_attrs = self.compute_geospatial_attributes(
        #     longitude=data_dict["L"]["longitude_fore"],
        #     latitude=data_dict["L"]["latitude_fore"],
        #     cell_row=data_dict["L"]["cell_row_fore"],
        #     cell_col=data_dict["L"]["cell_col_fore"],
        #     res=GRIDS[self.config.grid_definition]["res"],
        #     epsg_num=GRIDS[self.config.grid_definition]["epsg"],
        #     # grid, projection_string
        #     # grid=GRIDS[self.config.grid_definition],
        #     # projection_string=PROJECTIONS[self.config.projection_definition],
        # )

        # # Print the derived attributes
        # for key, value in geospatial_attrs.items():
        #     self.logger.debug(f"{key}: {value}")
        # # print(data_dict["L"].keys())
        # # TODO: The min and max values are not calculated correctly and should
        # #       (probably) be hardcoded into GRIDS
        # GLOBAL_ATTRIBUTES["geospatial_bounds"] = geospatial_attrs["geospatial_bounds"]
        # GLOBAL_ATTRIBUTES["geospatial_bounds_crs"] = geospatial_attrs[
        #     "geospatial_bounds_crs"
        # ]
        # # lat_min, lat_max = self.get_minmax_vals(array=data_dict["L"]["latitude_fore"])
        # GLOBAL_ATTRIBUTES["geospatial_lat_min"] = geospatial_attrs[
        #     "geospatial_lat_min"
        # ]  # lat_min
        # GLOBAL_ATTRIBUTES["geospatial_lat_max"] = geospatial_attrs[
        #     "geospatial_lat_max"
        # ]  # lat_max
        # # lon_min, lon_max = self.get_minmax_vals(array=data_dict["L"]["longitude_fore"])
        # GLOBAL_ATTRIBUTES["geospatial_lon_min"] = geospatial_attrs[
        #     "geospatial_lon_min"
        # ]  # lon_min
        # GLOBAL_ATTRIBUTES["geospatial_lon_max"] = geospatial_attrs[
        #     "geospatial_lon_max"
        # ]  # lon_max
        # GLOBAL_ATTRIBUTES["geospatial_lat_units"] = "degrees"
        # GLOBAL_ATTRIBUTES["geospatial_lat_resolution"] = geospatial_attrs[
        #     "geospatial_lat_resolution"
        # ]
        # GLOBAL_ATTRIBUTES["geospatial_lon_units"] = "degrees"
        # GLOBAL_ATTRIBUTES["geospatial_lon_resolution"] = geospatial_attrs[
        #     "geospatial_lon_resolution"
        # ]

        return GLOBAL_ATTRIBUTES

    def get_minmax_vals(self, array):
        min_value = np.nanmin(array)  # Ignores NaN
        max_value = np.nanmax(array)  # Ignores NaN

        return min_value, max_value

    def add_bounding_box(self, input_array, mask_value=np.nan):
        """
        Add a bounding box around a 2D array with the edges set to 1
        and the rest set to a masked value or kept unchanged.

        Args:
            array (numpy.ndarray): Input 2D array.
            mask_value (float): Value to mask the interior of the array.

        Returns:
            numpy.ndarray: Array with the bounding box added.
        """

        # Mask the NaN values
        masked_data = np.ma.masked_where(np.isnan(input_array), input_array)

        # Create an output masked array filled with NaNs (or a masked array)
        output = np.ma.masked_all_like(masked_data)

        # Loop through each row
        for i, row in enumerate(masked_data):
            # Find indices of non-masked elements
            non_masked_indices = np.where(~row.mask)[0]
            if non_masked_indices.size > 0:  # Check if there are non-masked values
                first_idx = non_masked_indices[0]
                last_idx = non_masked_indices[-1]
                # Store the first and last non-masked elements in the output array
                output[i, first_idx] = masked_data[i, first_idx]
                output[i, last_idx] = masked_data[i, last_idx]

        return output  # masked_array #bounding_box #array_with_bbox

    def construct_2D_array(
        self, input_array: np.ndarray, cell_row: np.ndarray, cell_col: np.ndarray
    ) -> np.ndarray:
        """
        Constructs a 2D array by mapping input values to specified grid cells based on row and column indices.

        Parameters
        ----------
        input_array : numpy.ndarray
            The array of input values to be mapped onto the grid.
        cell_row : numpy.ndarray or list of int
            Array or list containing the row indices of the grid cells where the input values will be placed.
        cell_col : numpy.ndarray or list of int
            Array or list containing the column indices of the grid cells where the input values will be placed.

        Returns
        -------
        output_array : numpy.ndarray
            A 2D array with the shape defined by the grid's configuration, where the input values are assigned
            to the specified row and column indices. Unassigned cells will contain NaN.
        """

        grid_shape = (
            GRIDS[self.config.grid_definition]["n_rows"],
            GRIDS[self.config.grid_definition]["n_cols"],
        )
        # create nan array with shape of grid_shape
        output_array = np.full(grid_shape, fill_value=np.nan)

        for i in range(len(cell_row)):
            output_array[int(cell_row[i]), int(cell_col[i])] = input_array[i]

        return output_array

    # This is for L1C only
    def compute_geospatial_attributes(
        self,
        longitude,
        latitude,
        cell_row,
        cell_col,
        res,
        epsg_num,  # grid, projection_string
    ):
        """
        Compute geospatial attributes for a given grid and its projection.

        Args:
            grid (dict): Dictionary containing grid parameters like 'x_min', 'y_max', 'res', 'n_cols', 'n_rows'.
            projection_string (str): PROJ string for the grid's projection.

        Returns:
            dict: Dictionary containing derived geospatial attributes.
        """
        # res, n_cols, n_rows = grid["res"], grid["n_cols"], grid["n_rows"]

        # Retrieving max/min values for lons and lats
        lon_min, lon_max = self.get_minmax_vals(array=longitude)
        lat_min, lat_max = self.get_minmax_vals(array=latitude)

        # Constructing 2D arrays
        lon_grid = self.construct_2D_array(
            input_array=longitude, cell_row=cell_row, cell_col=cell_col
        )
        lat_grid = self.construct_2D_array(
            input_array=latitude, cell_row=cell_row, cell_col=cell_col
        )

        # Retrieving bounding boxes
        lons_masked = self.add_bounding_box(input_array=lon_grid)
        lats_masked = self.add_bounding_box(input_array=lat_grid)

        # Retrieving only non-masked values
        valid_lons = lons_masked.compressed()  # Extract non-masked longitudes
        valid_lats = lats_masked.compressed()  # Extract non-masked latitudes

        # Combine lons and lats into coordinate pairs
        coordinates = list(zip(valid_lons, valid_lats))
        # Ensure the polygon is closed by adding the first point to the end
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        # Create a Shapely Polygon
        polygon = Polygon(coordinates)

        # Print the POLYGON WKT (Well-Known Text) representation
        # print("POLYGON:", polygon)
        print("POLYGON WKT:", polygon.wkt)
        # exit()

        # Define geospatial bounds as a polygon
        geospatial_bounds = polygon.wkt  # f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, {lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"

        # Extract grid properties
        # x_min, y_max = grid["x_min"], grid["y_max"]
        # res, n_cols, n_rows = grid["res"], grid["n_cols"], grid["n_rows"]

        # Calculate bounding box in projected coordinates
        # x_max = x_min + (n_cols * res)
        # y_min = y_max - (n_rows * res)

        # print(f"x_min = {x_min}, x_max = {x_max}")
        # print(f"y_min = {y_min}, y_max = {y_max}")

        # transform to geographic coordinates
        # projection = pyproj.Proj(projection_string)
        # lon_min, lat_max = projection(x_min, y_max, inverse=True)
        # lon_max, lat_min = projection(x_max, y_min, inverse=True)

        # print(f"lat_min = {lat_min}, lat_max = {lat_max}")
        # print(f"lon_min = {lon_min}, lon_max = {lon_max}")

        # lon_min, lat_min = projection(x_min, y_min, inverse=True)
        # lon_max, lat_max = projection(x_max, y_max, inverse=True)

        # print(f"lat_min = {lat_min}, lat_max = {lat_max}")
        # print(f"lon_min = {lon_min}, lon_max = {lon_max}")

        # # Set up the transformer
        # transformer = pyproj.Transformer.from_proj(
        #     proj_from=projection_string, proj_to="epsg:4326", always_xy=True
        # )

        # # Transform grid corners
        # lon_min, lat_min = transformer.transform(x_min, y_min)
        # lon_max, lat_max = transformer.transform(x_max, y_max)

        # print(f"lat_min = {lat_min}, lat_max = {lat_max}")
        # print(f"lon_min = {lon_min}, lon_max = {lon_max}")

        # print(projection_string)

        # exit()

        # Transform a point and its neighbor one resolution unit apart
        # lon1, lat1 = projection(  # transformer.transform(
        #     x_min, y_max
        # )  # Top left corner of the grid/bounding box
        # lon2, lat2 = projection(  # transformer.transform(
        #     x_min + res, y_max - res
        # )

        # Calculate lat/lon resolution
        lat_res = abs(lat_max - lat_min) / res  # abs(lat2 - lat1)
        lon_res = abs(lon_max - lon_min) / res  # abs(lon2 - lon1)

        print(lon_res, lat_res)
        # exit()

        # Return geospatial attributes
        return {
            "geospatial_lat_min": lat_min,
            "geospatial_lat_max": lat_max,
            "geospatial_lon_min": lon_min,
            "geospatial_lon_max": lon_max,
            "geospatial_bounds": geospatial_bounds,
            "geospatial_bounds_crs": f"EPSG:{epsg_num}",  # {grid['epsg']}",  # "EPSG:4326",
            "geospatial_lat_resolution": lat_res,
            "geospatial_lon_resolution": lon_res,
        }

    # TODO: Method to get a list of antenna patterns used in processing
    #       to populate metadata in the nc file
    def get_cimr_antenna_patterns_list(self) -> str:
        """
        Retrieves a list of antenna patterns used in processing and formats them as a single string.

        This method gathers all antenna pattern file names for each target band specified in the
        configuration and combines them into a comma-separated string.

        Returns
        -------
        beamfiles_paths : str
            A comma-separated string of antenna pattern file names from all target bands.

        Notes
        -----
        - The antenna patterns are retrieved from directories specified by the configuration's
          `antenna_patterns_path` for each target band.
        - The file names are extracted from the corresponding subdirectories for each band.
        """

        beamfiles_paths = []

        for band in self.config.target_band:
            ap_path = pb.Path(self.config.antenna_patterns_path).joinpath(band)
            beamfiles_paths.append([pattern.name for pattern in ap_path.glob("*")])

        beamfiles_paths = ", ".join(list(it.chain.from_iterable(beamfiles_paths)))

        return beamfiles_paths

    def create_global_dimensions(
        self, glob_dim_name: str, glob_dim_val: int | None, dataset: nc.Dataset
    ):
        """
        Creates a global dimension in a netCDF dataset if it does not already exist.

        This method checks whether the specified dimension exists in the given netCDF dataset.
        If not, it creates the dimension with the provided name and value.

        Parameters
        ----------
        glob_dim_name : str
            The name of the global dimension to create.
        glob_dim_val : int or None
            The size of the dimension to create. Use `None` for unlimited dimensions.
        dataset : netCDF4.Dataset
            The netCDF dataset in which the dimension will be created.

        Notes
        -----
        - The method logs the creation of a new dimension using the configured logger.
        - If the dimension already exists, no action is performed.
        """

        if glob_dim_name not in dataset.dimensions:
            self.logger.info(
                f"Creating Dimension: {glob_dim_name}, value: {glob_dim_val}"
            )
            dataset.createDimension(glob_dim_name, glob_dim_val)

    def get_global_dimensions(
        self,
        band_name: str,
        var_name: str,
        var_shape: int | tuple,
        dataset: nc.Dataset,
        data_dict: dict,
    ) -> dict:
        """
        Retrieves and constructs a dictionary of global dimensions for a netCDF dataset.

        This method creates a dictionary of global dimensions based on the dataset type
        (L1C or L1R), band name, and variable details. It also populates the dimensions
        using specific logic depending on the input data type (CIMR, SMAP, or AMSR2).

        Parameters
        ----------
        band_name : str
            The name of the band for which dimensions are being defined.
        var_name : str
            The name of the variable for which dimensions are being processed.
        var_shape : tuple of int
            The shape of the variable data to infer dimension sizes.
        dataset : netCDF4.Dataset
            The netCDF dataset in which dimensions will be checked or added.
        data_dict : dict
            A dictionary containing data for different bands to help infer dimensions.

        Returns
        -------
        glob_dims_dict : dict
            A dictionary where keys are dimension names and values are their sizes. If sizes
            are not yet determined, the values will be `None`.

        Notes
        -----
        - Dimensions are created based on the `grid_type` configuration (`L1C` or `L1R`):
          - For `L1C`: Includes dimensions such as `time`, `x`, and `y`.
          - For `L1R`: Focuses on scan-related dimensions like `n_l1b_scans`.
        - Dimension population logic depends on the `input_data_type`:
          - `CIMR`: Uses `populate_global_dimensions_cimr` for CIMR-specific rules.
          - `SMAP`: Uses `populate_global_dimensions_smap` for SMAP-specific rules.
          - `AMSR2`: Uses `populate_global_dimensions_amsr2` for AMSR2-specific rules.
        """

        # Empty dict to populate it with the dimensions
        glob_dims_dict = {}
        for band_name in data_dict.keys():
            glob_dims_dict[f"n_feeds_{band_name}_BAND"] = None
            glob_dims_dict[f"n_samples_{band_name}_BAND"] = None
        glob_dims_dict["n_l1b_scans"] = None

        if self.config.grid_type == "L1C":
            # # Creating Dimentions according to cdl
            glob_dims_dict["time"] = None
            glob_dims_dict["y"] = None
            glob_dims_dict["x"] = None

        # For each dataset (AMSR2, CIMR, SMAP), we will use different logic to
        # populate the dataset with values.
        if self.config.input_data_type == "CIMR":
            glob_dims_dict = self.populate_global_dimensions_cimr(
                glob_dims_dict=glob_dims_dict,
                band_name=band_name,
                var_name=var_name,
                var_shape=var_shape,
            )
        elif self.config.input_data_type == "SMAP":
            glob_dims_dict = self.populate_global_dimensions_smap(
                glob_dims_dict=glob_dims_dict
            )
        elif self.config.input_data_type == "AMSR2":
            glob_dims_dict = self.populate_global_dimensions_amsr2(
                glob_dims_dict=glob_dims_dict
            )

        return glob_dims_dict
        # -----------------------------------

    def populate_global_dimensions_cimr(
        self,
        glob_dims_dict,
        band_name,
        var_name,
        var_shape,
    ):
        """Method to populate global dimensions for CIMR project."""
        if self.config.grid_type == "L1C":
            glob_dims_dict["time"] = 1
            # This variable is special since its shape is of L1R not L1C data
            if "regridding_l1b_orphans" in var_name:
                glob_dims_dict["n_l1b_scans"] = var_shape[0]
                # TODO: There is shape mismatch, which needs to be investigated
                # glob_dims_dict[f"n_samples_{band_name}_BAND"] = var_shape[1]
                # glob_dims_dict[f"n_feeds_{band_name}_BAND"] = var_shape[2]
        elif self.config.grid_type == "L1R":
            glob_dims_dict["n_l1b_scans"] = var_shape[0]

        # print(var_name, var_shape)
        return glob_dims_dict

    # TODO: Populate the ones in case of AMSR2 and SMAP
    def populate_global_dimensions_amsr2(self, glob_dims_dict):
        return glob_dims_dict

    def populate_global_dimensions_smap(self, glob_dims_dict):
        return glob_dims_dict

    def generate_new_product(self, data_dict: dict):
        """Main method of the class. Its purpose is to create the the end data product."""

        # Getting filename (in simplified format)
        outfile = self.get_processor_filename_in_simplified_fmt()
        # Getting global attributes
        GLOBAL_ATTRIBUTES = self.generate_global_metadata(data_dict=data_dict)
        # Open the file to and populate it with data
        with nc.Dataset(outfile, "w", format="NETCDF4") as dataset:
            # Set each global attribute in the netCDF file
            for attr, value in GLOBAL_ATTRIBUTES.items():
                dataset.setncattr(attr, value)

            # Creating nested groups according to cdl
            if self.config.grid_type == "L1C":
                top_group = dataset.createGroup(f"{self.config.projection_definition}")

            elif self.config.grid_type == "L1R":
                # target_band is a list
                top_group = dataset.createGroup(
                    f"{self.config.target_band[0]}_BAND_TARGET"
                )
            # Loop through the parameters defined inside CDL and compare their
            # names to the ones provided inside pickled file. If they coincide
            # we write them into specific group (defined in CDL). In addition,
            # CDL values for CIMR have dimensions (time, x, y) while SMAP has
            # only 1, so we also programmatically figure out the dimensons of
            # the numpy array provided and save the data accordingly.
            #
            # [Note]: We start looping in this way and not from data_dict, because
            # data_dict may contain _fore|_aft suffixes
            for group_field, group_vals in CDL["LOCAL_ATTRIBUTES"].items():
                self.logger.info(f"Creating group: {group_field}")

                group = top_group.createGroup(group_field)

                # Looping through data dictionary and retrieving its variables (per band)
                for band_name, band_var in data_dict.items():
                    # ----------------------
                    # cell_row and cell_col
                    # ----------------------
                    # Removing/skipping cell_row and cell_col if we encounter L1R
                    # (since it is not needed for L1R)
                    if self.config.grid_type == "L1R" and (
                        band_var == "cell_row" or band_var == "cell_col"
                    ):
                        continue
                    # ----------------------
                    # Processing Flags
                    # ----------------------
                    # Processing_flags are defined as a separate field
                    # and not as sub field of specific band
                    if group_field == "Processing_flags":
                        band_group = group
                    else:
                        band_group = group.createGroup(f"{band_name}_BAND")

                    # -----------------------------------
                    # Creating Global Dimensions
                    # -----------------------------------
                    for var_name, var_val in band_var.items():
                        var_shape = var_val.shape
                        glob_dims_dict = self.get_global_dimensions(
                            band_name=band_name,
                            var_name=var_name,
                            var_shape=var_shape,
                            dataset=dataset,
                            data_dict=data_dict,
                        )
                    # Adding newly created dimensions to a dataset
                    for glob_dim_name, glob_dim_val in glob_dims_dict.items():
                        self.create_global_dimensions(
                            glob_dim_name, glob_dim_val, dataset
                        )

                    # -----------------------------------
                    # Creating netCDF Variables
                    # -----------------------------------
                    for var_name, var_val in band_var.items():
                        var_shape = var_val.shape
                        # -----------------------------------
                        # Getting chunk sizes
                        # -----------------------------------
                        # var_dim, chunk_size = self.determine_dimension(
                        #     band_name=band_name, var_shape=var_shape
                        # )

                        # Creating a list of complete variables to regrid based on CDL
                        # and whether user chose to split scans into fore and aft
                        if self.config.split_fore_aft:
                            fore = [key + "_fore" for key in group_vals.keys()]
                            aft = [key + "_aft" for key in group_vals.keys()]
                            regrid_vars = fore + aft
                        else:
                            regrid_vars = [key for key in group_vals.keys()]

                        # Removing the _fore and _aft from the variable name
                        # to get the metadata from CDL (it is almost the same for
                        # both of them anyway). The idea is to compare actual
                        # variable to the variable from the CDL above
                        regrid_var = (
                            var_name.replace("_fore", "")
                            if "_fore" in var_name
                            else var_name.replace("_aft", "")
                            if "_aft" in var_name
                            else var_name
                        )

                        if var_name in regrid_vars:
                            # -----------------------------------
                            # Getting varables' dimensions and chunk sizes
                            # -----------------------------------
                            var_dim, chunk_size = self.determine_dimension(
                                band_name=band_name, var_shape=var_shape
                            )
                            if "regridding_l1b_orphans" in var_name:
                                var_dim = self.determine_dimension_l1r(
                                    band_name, var_shape
                                )
                                chunk_size = (10, 256, 1)

                                # var_type = self.get_netcdf_dtype(var_val.dtype)
                                # var_fill = nc.default_fillvals[var_type]

                                # self.logger.debug(
                                #     f"{group_field}, {band_name}, {var_name}, {var_type}, {var_fill}, {var_dim}"
                                # )

                                # # Determine the appropriate slice based on variable shape
                                # slices = tuple(slice(None) for _ in var_shape)
                                # var_data = band_group.createVariable(
                                #     varname=var_name,
                                #     datatype=var_type,  # "double",
                                #     dimensions=var_dim,  # ('x'),
                                #     fill_value=var_fill,  # group_vals[regrid_var]["_FillValue"]
                                # )
                                # var_data[slices] = var_val

                            # Converting numpy datatype into netCDF one
                            var_type = self.get_netcdf_dtype(var_val.dtype)
                            # Getting the default fill value to mask inappropriate values
                            var_fill = nc.default_fillvals[var_type]

                            self.logger.debug(
                                f"Creating a variable `{var_name}` of type `{var_type}` with the following attributes:"
                            )
                            self.logger.debug(
                                f"Group Field: {group_field}, Band Name: {band_name}, _FillValue: {var_fill}, Dimensions: {var_dim}"
                            )

                            # Determine the appropriate slice based on variable shape
                            slices = tuple(slice(None) for _ in var_shape)

                            # Creating a variable
                            #
                            # var_chunked = ncfile.createVariable(
                            #     varname,          # Variable name as a string
                            #     datatype,         # Data type (e.g., 'f4' for float32, 'i4' for int32)
                            #     dimensions,       # Dimensions as a tuple (e.g., ('x', 'y'))
                            #     chunksizes=None,  # Optional chunk sizes for each dimension (e.g., (100, 100))
                            #     fill_value=None,  # Optional fill value for missing data
                            #     zlib=False,       # Optional compression flag (True to enable compression)
                            #     complevel=4,      # Optional compression level (1-9, higher = more compression)
                            #     contiguous=False, # Optional flag to set storage as contiguous (True) or chunked (False)
                            #     endian='native',  # Optional byte order ('native', 'little', 'big')
                            #     least_significant_digit=None  # Optional precision control for float data
                            # )
                            if chunk_size is None:
                                var_data = band_group.createVariable(
                                    varname=var_name,
                                    datatype=var_type,  # "double",
                                    dimensions=var_dim,  # ('x'),
                                    fill_value=var_fill,  # group_vals[regrid_var]["_FillValue"]
                                )
                            else:
                                var_data = band_group.createVariable(
                                    varname=var_name,
                                    datatype=var_type,
                                    dimensions=var_dim,
                                    fill_value=var_fill,
                                    contiguous=False,
                                    chunksizes=chunk_size,
                                )
                            # Assign values to the variable
                            var_data[slices] = var_val

                            # -----------------------------------
                            # Setting Local Attributes
                            # -----------------------------------
                            # Loop through the dictionary and set attributes for the variable
                            for attr_name, attr_value in group_vals[regrid_var].items():
                                if attr_name == "long_name":
                                    pattern = r"TBD-km"
                                    substitution = f"{self.config.grid_definition}"
                                    attr_value = re.sub(
                                        pattern, substitution, attr_value
                                    )
                                    var_data.setncattr(attr_name, attr_value)
                                elif attr_name == "comment":
                                    pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_\[fore\|aft\]"

                                    if self.config.split_fore_aft:
                                        substitution = (
                                            f"{band_name}_BAND_fore"
                                            if "_fore" in var_name
                                            else f"{band_name}_BAND_aft"
                                        )
                                    else:
                                        substitution = f"{band_name}_BAND"

                                    attr_value = re.sub(
                                        pattern, substitution, attr_value
                                    )

                                    # Checking whther there is any patter left of the following format
                                    pattern = r"\[L\|C\|X\|KU\|KA\]_BAND"
                                    substitution = f"{band_name}_BAND"
                                    attr_value = re.sub(
                                        pattern, substitution, attr_value
                                    )
                                    # Checking whther there is any patter left of the following format
                                    pattern = r"TBD-km"
                                    substitution = f"{self.config.grid_definition}"
                                    attr_value = re.sub(
                                        pattern, substitution, attr_value
                                    )

                                    pattern = r"\[fore\|aft\]"
                                    if self.config.split_fore_aft:
                                        substitution = (
                                            "fore" if "_fore" in var_name else "aft"
                                        )
                                    else:
                                        # Just leave it the way it is
                                        substitution = "[fore|aft]"
                                    attr_value = re.sub(
                                        pattern, substitution, attr_value
                                    )
                                    # Setting comment attribute
                                    var_data.setncattr(attr_name, attr_value)
                                else:  # attr_name != "comment":
                                    # Use setncattr to assign the attribute (from the CDL dict)
                                    var_data.setncattr(attr_name, attr_value)
                                    # Assigning another attribute to a variable
                                    if chunk_size is None:
                                        var_data.setncattr("_Storage", "contiguous")
                                    else:
                                        var_data.setncattr("_Storage", "chunked")
                                        var_data.setncattr(
                                            "_ChunkSizes", f"{chunk_size}"
                                        )

            # exit()

    # TODO: This is an old method and it should be removed
    # def generate_product(self, data_dict: dict):
    #     """
    #     Method to generate resulting product
    #     """

    #     # # --------------------------
    #     # # Working this the filename
    #     # if self.config.grid_definition is not None:
    #     #     grid_res = re.search(
    #     #         r"(\d+(?:\.\d+)?)km", self.config.grid_definition
    #     #     ).group()
    #     # else:
    #     #     grid_res = ""
    #     # # Get the current date and time
    #     # # l1c_utc_time = datetime.datetime.now()

    #     # # Format the date and time as "YYYYMMDDHHMMSS"
    #     # l1c_utc_time = (
    #     #     self.config.timestamp
    #     # )  # self.config.file_time_signature #l1c_utc_time.strftime("%Y%m%d%H%M%S")
    #     # # print(l1c_utc_time)
    #     # # exit()

    #     # if self.config.grid_definition is not None:
    #     #     outfile = f"{self.config.input_data_type}_{self.config.grid_type}_{self.config.regridding_algorithm}_{grid_res}_{l1c_utc_time}.nc"
    #     # else:
    #     #     outfile = f"{self.config.input_data_type}_{self.config.grid_type}_{self.config.regridding_algorithm}_{l1c_utc_time}.nc"

    #     # outfile = pb.Path(f"{self.config.output_path}/{outfile}").resolve()
    #     # # --------------------------

    #     outfile = self.get_processor_filename_in_simplified_fmt()

    #     GLOBAL_ATTRIBUTES = self.generate_global_metadata(data_dict=data_dict)

    #     with nc.Dataset(outfile, "w", format="NETCDF4") as dataset:
    #         # Set each global attribute in the netCDF file
    #         for attr, value in GLOBAL_ATTRIBUTES.items():
    #             dataset.setncattr(attr, value)

    #         # -------------------------------------------------
    #         # Creating Global Dimensions
    #         # The L1R template has the following dims:

    #         # CIMR_E2ESv110_L1B_Product_Format_v0.6.nc

    #         # netcdf CIMR_E2ESv110_L1B_Product_Format_v0.6 {
    #         # dimensions:
    #         # 	n_feeds_L_BAND = 1 ;
    #         # 	n_feeds_C_BAND = 4 ;
    #         # 	n_feeds_X_BAND = 4 ;
    #         # 	n_feeds_KU_BAND = 8 ;
    #         # 	n_feeds_KA_BAND = 8 ;
    #         # 	n_scans = 2 ;
    #         # 	n_samples_L_BAND = 138 ;
    #         # 	n_samples_C_BAND = 549 ;
    #         # 	n_samples_X_BAND = 561 ;
    #         # 	n_samples_KU_BAND = 1538 ;
    #         # 	n_samples_KA_BAND = 2079 ;

    #         # if self.config.grid_type == "L1R":
    #         #     # TODO: this should be in per band calculations
    #         #     # dataset.createDimension('n_feeds_L_BAND', 1) #None)
    #         #     # dataset.createDimension('n_feeds_X_BAND', 4) #None)
    #         #     # dataset.createDimension('n_feeds_C_BAND', 4) #None)
    #         #     # dataset.createDimension('n_feeds_KU_BAND', 8) #None)
    #         #     # dataset.createDimension('n_feeds_KA_BAND', 8) #None)

    #         #     # dataset.createDimension('n_scans', 2) #None)
    #         #     # dataset.createDimension('n_samples_L_BAND', 138) #None)
    #         #     # dataset.createDimension('n_samples_X_BAND', 561) #None)
    #         #     # dataset.createDimension('n_samples_C_BAND', 549) #None)
    #         #     # dataset.createDimension('n_samples_KU_BAND', 1538) #None)
    #         #     # dataset.createDimension('n_samples_KA_BAND', 2079) #None)

    #         #     for band_name in data_dict.keys():
    #         #         if self.config.input_data_type == "CIMR":
    #         #             if band_name == "L":
    #         #                 feed_nums = 1
    #         #         dataset.createDimension(f"n_feeds_{band_name}_BAND", feed_nums)
    #         #         dataset.createDimension(f"n_samples_{band_name}_BAND", None)

    #         #     dataset.createDimension("n_scans", None)

    #         # elif self.config.grid_type == "L1C":
    #         #     # Also, there should be per band calculations for n_feeds
    #         #     # For L1C dimensions
    #         #     # time = 0 // currently 1 <= single integer value
    #         #     # x = {256..16384}
    #         #     # y = {256..16384}
    #         #     # n_l1b_scans = TBD
    #         #     # n_samples = 0
    #         #     # n_feeds_[L_BAND|C_BAND|X_BAND|KA_BAND|KU_BAND] = [1, 4, 4, 8, 8]
    #         #     for band_name in data_dict.keys():
    #         #         dataset.createDimension(f"n_feeds_{band_name}_BAND", None)
    #         #         dataset.createDimension(f"n_samples_{band_name}_BAND", None)

    #         #     dataset.createDimension("n_scans", None)
    #         #     dataset.createDimension("n_l1b_scans", None)

    #         #     # Creating Dimentions according to cdl
    #         #     dataset.createDimension("time", 1)  # None)
    #         #     dataset.createDimension("y", None)
    #         #     dataset.createDimension("x", None)

    #         # print(self.config.target_band)

    #         # Creating nested groups according to cdl
    #         if self.config.grid_type == "L1C":
    #             top_group = dataset.createGroup(f"{self.config.projection_definition}")

    #         elif self.config.grid_type == "L1R":
    #             # target_band is a list
    #             top_group = dataset.createGroup(
    #                 f"{self.config.target_band[0]}_BAND_TARGET"
    #             )

    #         # Loop through the parameters defined inside CDL and compare their
    #         # names to the ones provided inside pickled file. If they coincide
    #         # we write them into specific group (defined in CDL). In addition,
    #         # CDL values for CIMR have dimensions (time, x, y) while SMAP has
    #         # only 1, so we also programmatically figure out the dimensons of
    #         # the numpy array provided and save the data accordingly.
    #         #
    #         # [Note]: We start looping in this way and not from data_dict, because
    #         # data_dict may contain _fore|_aft suffixes
    #         for group_field, group_vals in CDL["LOCAL_ATTRIBUTES"].items():
    #             self.logger.info(
    #                 f"Creating group: {group_field}"
    #             )  # . Group val: {group_vals}")

    #             # group = data_group.createGroup(group_field)
    #             group = top_group.createGroup(group_field)

    #             # Some fields of Quality_information are not sub group of bands
    #             # if group_field == "Quality_information":

    #             # Looping through data dictionary and retrieving its variables (per band)
    #             for band_name, band_var in data_dict.items():
    #                 # Removing/skipping cell_row and cell_col if we encounter L1R
    #                 # (since it is not needed for L1R)
    #                 if self.config.grid_type == "L1R" and (
    #                     band_var == "cell_row" or band_var == "cell_col"
    #                 ):
    #                     continue

    #                 # Processing_flags are defined as a separate field
    #                 # and not as sub field of specific band
    #                 if group_field == "Processing_flags":
    #                     band_group = group
    #                 else:
    #                     band_group = group.createGroup(f"{band_name}_BAND")

    #                 for var_name, var_val in band_var.items():
    #                     var_shape = var_val.shape

    #                     # TODO: Expand the following for the case of AMSR2 and SMAP
    #                     # -----------------------------------
    #                     # Creating Global Dimensions
    #                     # (if they do not yet exist)
    #                     # -----------------------------------
    #                     if self.config.input_data_type == "CIMR":
    #                         if self.config.grid_type == "L1C":
    #                             # Also, there should be per band calculations for n_feeds
    #                             # For L1C dimensions
    #                             # time = 0 // currently 1 <= single integer value
    #                             # x = {256..16384}
    #                             # y = {256..16384}
    #                             # n_l1b_scans = TBD
    #                             # n_samples = 0
    #                             # n_feeds_[L_BAND|C_BAND|X_BAND|KA_BAND|KU_BAND] = [1, 4, 4, 8, 8]

    #                             # Creating Dimensions according to cdl
    #                             glob_dim_name = "time"
    #                             glob_dim_val = 1
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )

    #                             glob_dim_name = "y"
    #                             glob_dim_val = None
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )

    #                             glob_dim_name = "x"
    #                             glob_dim_val = None
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )
    #                             # ('n_l1b_scans', 'n_samples', 'n_feeds')
    #                             if "regridding_l1b_orphans" in var_name:
    #                                 glob_dim_name = "n_l1b_scans"
    #                                 glob_dim_val = var_shape[0]
    #                                 self.create_global_dimensions(
    #                                     glob_dim_name, glob_dim_val, dataset
    #                                 )

    #                                 glob_dim_name = f"n_samples_{band_name}_BAND"
    #                                 glob_dim_val = var_shape[1]
    #                                 self.create_global_dimensions(
    #                                     glob_dim_name, glob_dim_val, dataset
    #                                 )
    #                                 glob_dim_name = f"n_feeds_{band_name}_BAND"
    #                                 glob_dim_val = var_shape[2]
    #                                 self.create_global_dimensions(
    #                                     glob_dim_name, glob_dim_val, dataset
    #                                 )
    #                         elif self.config.grid_type == "L1R":
    #                             # L1R does not contain x, y, time as L1C does
    #                             glob_dim_name = "n_l1b_scans"
    #                             glob_dim_val = var_shape[0]
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )
    #                             # TODO: The feeds are identified not as target band
    #                             # but as a source band and hence we have a problem,
    #                             # when source band is C (4 feeds) and target band
    #                             # is L (1 feed). This should be fixed but after we
    #                             # figure out the source/target band stuff for L1R
    #                             # data. The same goes for `samples`.
    #                             glob_dim_name = f"n_samples_{band_name}_BAND"
    #                             glob_dim_val = None  # var_shape[1]
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )
    #                             glob_dim_name = f"n_feeds_{band_name}_BAND"
    #                             glob_dim_val = None  # var_shape[2]
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )
    #                     else:
    #                         for band_name in data_dict.keys():
    #                             glob_dim_name = f"n_feeds_{band_name}_BAND"
    #                             glob_dim_val = None
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )
    #                             glob_dim_name = f"n_samples_{band_name}_BAND"
    #                             glob_dim_val = None
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )

    #                         glob_dim_name = "n_l1b_scans"
    #                         glob_dim_val = None
    #                         self.create_global_dimensions(
    #                             glob_dim_name, glob_dim_val, dataset
    #                         )

    #                         if self.config.grid_type == "L1C":
    #                             # Also, there should be per band calculations for n_feeds
    #                             # For L1C dimensions
    #                             # time = 0 // currently 1 <= single integer value
    #                             # x = {256..16384}
    #                             # y = {256..16384}
    #                             # n_l1b_scans = TBD
    #                             # n_samples = 0
    #                             # n_feeds_[L_BAND|C_BAND|X_BAND|KA_BAND|KU_BAND] = [1, 4, 4, 8, 8]

    #                             # Creating Dimentions according to cdl
    #                             glob_dim_name = "time"
    #                             glob_dim_val = 1
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )

    #                             glob_dim_name = "y"
    #                             glob_dim_val = None
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )

    #                             glob_dim_name = "x"
    #                             glob_dim_val = None
    #                             self.create_global_dimensions(
    #                                 glob_dim_name, glob_dim_val, dataset
    #                             )
    #                     # -----------------------------------

    #                     # TODO:
    #                     # For L1r we will have 3 dimensions: scan, sample, and feed <= scan is the same for all bands, but sample and feed are not
    #                     # For L1C we will have 3 dimensions: time, y, and x <= the same for all bands
    #                     #
    #                     # So add if else statement
    #                     # if self.config.grid_type == "L1C":
    #                     #     var_dim = self.determine_dimension_l1c(var_shape)

    #                     # elif self.config.grid_type == "L1R":
    #                     #     var_dim = self.determine_dimension_l1r(band_name, var_shape)

    #                     var_dim, chunk_size = self.determine_dimension(
    #                         band_name=band_name, var_shape=var_shape
    #                     )
    #                     print(var_dim, var_shape)

    #                     # -----------------------------------
    #                     # Getting chunk sizes
    #                     # -----------------------------------
    #                     # if self.config.grid_type == "L1C":
    #                     #    if len(var_dim) == 1:
    #                     #        chunk_size = 256
    #                     #    else:
    #                     #        chunk_size = 1

    #                     # Creating a list of complete variables to regrid based on CDL
    #                     # and whether user chose to split scans into fore and aft
    #                     if self.config.split_fore_aft:
    #                         # print("True")
    #                         fore = [key + "_fore" for key in group_vals.keys()]
    #                         aft = [key + "_aft" for key in group_vals.keys()]
    #                         regrid_vars = fore + aft
    #                     else:
    #                         regrid_vars = [key for key in group_vals.keys()]

    #                     # Removing the _fore and _aft from the variable name
    #                     # to get the metadata from CDL (it is almost the same for
    #                     # both of them anyway). The idea is to compare actual
    #                     # variable to the variable from the CDL above
    #                     regrid_var = (
    #                         var_name.replace("_fore", "")
    #                         if "_fore" in var_name
    #                         else var_name.replace("_aft", "")
    #                         if "_aft" in var_name
    #                         else var_name
    #                     )
    #                     # print(var_name, regrid_var)

    #                     if (var_name in regrid_vars) and (
    #                         "regridding_l1b_orphans" not in var_name
    #                     ):
    #                         var_type = self.get_netcdf_dtype(var_val.dtype)
    #                         var_fill = nc.default_fillvals[var_type]

    #                         self.logger.debug(
    #                             f"{group_field}, {band_name}, {var_name}, {var_type}, {var_fill}, {var_dim}"
    #                         )

    #                         # Determine the appropriate slice based on variable shape
    #                         slices = tuple(slice(None) for _ in var_shape)
    #                         # print(slices, var_shape)
    #                         # print(var_shape)
    #                         # print(var_dim)

    #                         # var_chunked = ncfile.createVariable(
    #                         #     varname,          # Variable name as a string
    #                         #     datatype,         # Data type (e.g., 'f4' for float32, 'i4' for int32)
    #                         #     dimensions,       # Dimensions as a tuple (e.g., ('x', 'y'))
    #                         #     chunksizes=None,  # Optional chunk sizes for each dimension (e.g., (100, 100))
    #                         #     fill_value=None,  # Optional fill value for missing data
    #                         #     zlib=False,       # Optional compression flag (True to enable compression)
    #                         #     complevel=4,      # Optional compression level (1-9, higher = more compression)
    #                         #     contiguous=False, # Optional flag to set storage as contiguous (True) or chunked (False)
    #                         #     endian='native',  # Optional byte order ('native', 'little', 'big')
    #                         #     least_significant_digit=None  # Optional precision control for float data
    #                         # )
    #                         var_data = band_group.createVariable(
    #                             varname=var_name,
    #                             datatype=var_type,  # "double",
    #                             dimensions=var_dim,  # ('x'),
    #                             fill_value=var_fill,  # group_vals[regrid_var]["_FillValue"]
    #                             contiguous=False,
    #                             chunksizes=chunk_size,
    #                         )
    #                         # Assign values to the variable
    #                         var_data[slices] = var_val

    #                         # Loop through the dictionary and set attributes for the variable
    #                         for attr_name, attr_value in group_vals[regrid_var].items():
    #                             # TODO: The _FillValue field is kind of obsolete, because we
    #                             # define it above via fill_value parameter (but lets leave it here for now)
    #                             if attr_name != "_FillValue" and attr_name != "comment":
    #                                 # print(attr_name)
    #                                 # Use setncattr to assign the attribute
    #                                 var_data.setncattr(attr_name, attr_value)

    #                             elif attr_name == "comment":
    #                                 pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_\[fore\|aft\]"

    #                                 if self.config.split_fore_aft:
    #                                     substitution = (
    #                                         f"{band_name}_BAND_fore"
    #                                         if "_fore" in var_name
    #                                         else f"{band_name}_BAND_aft"
    #                                     )
    #                                 else:
    #                                     substitution = f"{band_name}_BAND"
    #                                 # print(substitution)

    #                                 # pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_" #\[fore\|aft\]"
    #                                 # substitution = f"{band_name}_BAND_"
    #                                 attr_value = re.sub(
    #                                     pattern, substitution, attr_value
    #                                 )
    #                                 # var_data.setncattr(attr_name, attr_value)

    #                                 # Checking whther there is any patter left of the following format
    #                                 pattern = r"\[L\|C\|X\|KU\|KA\]_BAND"
    #                                 substitution = f"{band_name}_BAND"
    #                                 attr_value = re.sub(
    #                                     pattern, substitution, attr_value
    #                                 )

    #                                 pattern = r"\[fore\|aft\]"
    #                                 if self.config.split_fore_aft:
    #                                     substitution = (
    #                                         "fore" if "_fore" in var_name else "aft"
    #                                     )
    #                                 else:
    #                                     # Just leave it the way it is
    #                                     substitution = "[fore|aft]"
    #                                 attr_value = re.sub(
    #                                     pattern, substitution, attr_value
    #                                 )

    #                                 # Setting comment attribute
    #                                 var_data.setncattr(attr_name, attr_value)

    #                     elif (var_name in regrid_vars) and (
    #                         "regridding_l1b_orphans" in var_name
    #                     ):
    #                         # elif ("regridding_l1b_orphans" in var_name):
    #                         var_dim = self.determine_dimension_l1r(band_name, var_shape)
    #                         var_type = self.get_netcdf_dtype(var_val.dtype)
    #                         var_fill = nc.default_fillvals[var_type]

    #                         self.logger.debug(
    #                             f"{group_field}, {band_name}, {var_name}, {var_type}, {var_fill}, {var_dim}"
    #                         )

    #                         # Determine the appropriate slice based on variable shape
    #                         slices = tuple(slice(None) for _ in var_shape)
    #                         var_data = band_group.createVariable(
    #                             varname=var_name,
    #                             datatype=var_type,  # "double",
    #                             dimensions=var_dim,  # ('x'),
    #                             fill_value=var_fill,  # group_vals[regrid_var]["_FillValue"]
    #                         )
    #                         var_data[slices] = var_val
    #                         # Loop through the dictionary and set attributes for the variable
    #                         for attr_name, attr_value in group_vals[regrid_var].items():
    #                             if attr_name != "_FillValue" and attr_name != "comment":
    #                                 var_data.setncattr(attr_name, attr_value)
    #                             elif attr_name == "comment":
    #                                 pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_\[fore\|aft\]"

    #                                 if self.config.split_fore_aft:
    #                                     substitution = (
    #                                         f"{band_name}_BAND_fore"
    #                                         if "_fore" in var_name
    #                                         else f"{band_name}_BAND_aft"
    #                                     )
    #                                 else:
    #                                     substitution = f"{band_name}_BAND"
    #                                 # print(substitution)

    #                                 # pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_" #\[fore\|aft\]"
    #                                 # substitution = f"{band_name}_BAND_"
    #                                 attr_value = re.sub(
    #                                     pattern, substitution, attr_value
    #                                 )
    #                                 # var_data.setncattr(attr_name, attr_value)

    #                                 # Checking whther there is any patter left of the following format
    #                                 pattern = r"\[L\|C\|X\|KU\|KA\]_BAND"
    #                                 substitution = f"{band_name}_BAND"
    #                                 attr_value = re.sub(
    #                                     pattern, substitution, attr_value
    #                                 )

    #                                 pattern = r"\[fore\|aft\]"
    #                                 if self.config.split_fore_aft:
    #                                     substitution = (
    #                                         "fore" if "_fore" in var_name else "aft"
    #                                     )
    #                                 else:
    #                                     # Just leave it the way it is
    #                                     substitution = "[fore|aft]"
    #                                 attr_value = re.sub(
    #                                     pattern, substitution, attr_value
    #                                 )

    #                                 # Setting comment attribute
    #                                 var_data.setncattr(attr_name, attr_value)

    def get_netcdf_dtype(self, np_dtype: np.dtype) -> str:
        """
        Retrieve the correct netCDF-4 string literal based on numpy data type.

        netCDF4 types are:

        Integer Types:
        'i1'   : byte     - 1-byte signed integer (int8)
        'u1'   : ubyte    - 1-byte unsigned integer (uint8)
        'i2'   : short    - 2-byte signed integer (int16)
        'u2'   : ushort   - 2-byte unsigned integer (uint16)
        'i4'   : int      - 4-byte signed integer (int32)
        'u4'   : uint     - 4-byte unsigned integer (uint32)
        'i8'   : int64    - 8-byte signed integer (int64)
        'u8'   : uint64   - 8-byte unsigned integer (uint64)

        Floating-Point Types:
        'f4'   : float    - 4-byte floating-point (float32)
        'f8'   : double   - 8-byte floating-point (float64)

        Character and String Types:
        'S1'   : char     - 1-byte character data
        str    : string   - Variable-length string data

        Variable-Length Types:
        Depends on element type (e.g., `str` for variable-length strings)

        User-Defined Types (Supported with Definitions):
        enum   : Enumerated type (base integer type dependent)
        opaque : Opaque data type (length defined in bytes)
        compound : Custom compound structure (user-defined sub-elements)

        Parameters:
        ----------
            np_dtype (numpy.dtype): The numpy data type to convert.

        Returns:
        --------
            str: The corresponding netCDF-4 string literal.
        """

        dtype_map = {
            np.dtype("int8"): "i1",  # 1-byte signed integer
            np.dtype("uint8"): "u1",  # 1-byte unsigned integer
            np.dtype("int16"): "i2",  # 2-byte signed integer
            np.dtype("uint16"): "u2",  # 2-byte unsigned integer
            np.dtype("int32"): "i4",  # 4-byte signed integer
            np.dtype("uint32"): "u4",  # 4-byte unsigned integer
            np.dtype("int64"): "i8",  # 8-byte signed integer
            np.dtype("uint64"): "u8",  # 8-byte unsigned integer
            np.dtype("float32"): "f4",  # 4-byte floating-point
            np.dtype("float64"): "f8",  # 8-byte floating-point
            np.dtype("S1"): "S1",  # 1-byte character
            np.dtype("str"): str,  # Variable-length string
        }

        return dtype_map.get(np_dtype, None)

    def determine_dimension(self, var_shape: tuple, band_name: str) -> tuple:
        """
        Generic method to determine the dimension.
        """
        if self.config.grid_type == "L1C":
            var_dim, chunk_size = self.determine_dimension_l1c(var_shape)
            return var_dim, chunk_size

        elif self.config.grid_type == "L1R":
            var_dim = self.determine_dimension_l1r(band_name, var_shape)
            return var_dim, None
        else:
            raise ValueError("Incorrect grid_type supplied.")

    def determine_dimension_l1r(self, band_name: str, var_shape: tuple) -> tuple:
        """
        Determines the dimension names based on the shape of the variable.

        Parameters
        ----------
        var_shape : tuple
            A tuple representing the shape of the variable (e.g., (10000,),
            (10000, 10000), or (1, 111, 111)).

        Returns
        -------
        tuple
            A tuple containing the dimension names:
            - ('n_feeds',) for 1D shapes (e.g., (10000,))
            - ('n_samples', 'n_feeds') for 2D shapes (e.g., (10000, 10000))
            - ('n_l1b_scans', 'n_samples', 'n_feeds') for 3D shapes (e.g., (1, 111, 111))

        Exceptions
        ----------
        ValueError
            Raised if the shape of the variable has more than 3 dimensions
            or an unsupported shape is provided.
        """

        if len(var_shape) == 1:
            # 1D case
            return ("n_feeds_{band_name}_BAND",)
        elif len(var_shape) == 2:
            # 2D case
            return (f"n_samples_{band_name}_BAND", f"n_feeds_{band_name}_BAND")
        elif len(var_shape) == 3:
            # 3D case
            return (
                "n_l1b_scans",
                f"n_samples_{band_name}_BAND",
                f"n_feeds_{band_name}_BAND",
            )
        else:
            # Return a generic message or handle error for unknown shapes
            raise ValueError(
                f"Unsupported shape with {len(var_shape)} dimensions: {var_shape}"
            )

    def determine_dimension_l1c(self, var_shape: tuple) -> tuple:
        """
        Determines the dimension names based on the shape of the variable.

        Parameters
        ----------
        var_shape : tuple
            A tuple representing the shape of the variable (e.g., (10000,),
            (10000, 10000), or (1, 111, 111)).

        Returns
        -------
        tuple
            A tuple containing the dimension names:
            - ('x',) for 1D shapes (e.g., (10000,))
            - ('y', 'x') for 2D shapes (e.g., (10000, 10000))
            - ('time', 'y', 'x') for 3D shapes (e.g., (1, 111, 111))

        Exceptions
        ----------
        ValueError
            Raised if the shape of the variable has more than 3 dimensions
            or an unsupported shape is provided.
        """

        # Returns also chunk size
        if len(var_shape) == 1:
            # 1D case
            return ("x",), (256,)
        elif len(var_shape) == 2:
            # 2D case
            return ("y", "x"), None
        elif len(var_shape) == 3:
            # 3D case
            return ("time", "y", "x"), None
        else:
            # Return a generic message or handle error for unknown shapes
            raise ValueError(
                f"Unsupported shape with {len(var_shape)} dimensions: {var_shape}"
            )

    # TODO: Since it defines the name of operational processor,
    # we are not using this method now + it needs to be finished
    def get_processor_filename(self):
        if self.config.projection_definition == "N":
            proj_str = "N"
        elif self.config.projection_definition == "S":
            proj_str = "S"
        elif self.config.projection_definition == "G":
            proj_str = "G"
        elif self.config.projection_definition == "PS_N":
            proj_str = "P"
        elif self.config.projection_definition == "PS_S":
            proj_str = "Q"

        # Get the current date and time
        l1c_utc_time = datetime.datetime.now()

        # Format the date and time as "YYYYMMDDHHMMSS"
        l1c_utc_time = l1c_utc_time.strftime("%Y%m%d%H%M%S")

        # print(GRIDS.keys())
        # print(self.config.grid_definition)
        grid_res = re.search(
            r"(\d+(?:\.\d+)?)(?=km)", self.config.grid_definition
        ).group()

        grid_type = self.config.grid_type[1:]

        outfile = f"W_NO-ST-OSLOSAT{self.config.input_data_type}-{grid_type}_C_ESA_{l1c_utc_time}_G_D_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS_T_N_{proj_str}_{grid_res}.nc"

        return outfile

    def get_processor_filename_in_simplified_fmt(self):
        """Returns the preprocessor filename in a simplified format:"""
        # --------------------------
        # Working this the filename
        if self.config.grid_definition is not None:
            grid_res = re.search(
                r"(\d+(?:\.\d+)?)km", self.config.grid_definition
            ).group()
        else:
            grid_res = ""
        # Get the current date and time
        # l1c_utc_time = datetime.datetime.now()

        # Format the date and time as "YYYYMMDDHHMMSS"
        l1c_utc_time = (
            self.config.timestamp
        )  # self.config.file_time_signature #l1c_utc_time.strftime("%Y%m%d%H%M%S")
        # print(l1c_utc_time)
        # exit()

        if self.config.grid_definition is not None:
            outfile = f"{self.config.input_data_type}_{self.config.grid_type}_{self.config.regridding_algorithm}_{grid_res}_{l1c_utc_time}.nc"
        else:
            outfile = f"{self.config.input_data_type}_{self.config.grid_type}_{self.config.regridding_algorithm}_{l1c_utc_time}.nc"

        outfile = pb.Path(f"{self.config.output_path}/{outfile}").resolve()
        # --------------------------
        return outfile
