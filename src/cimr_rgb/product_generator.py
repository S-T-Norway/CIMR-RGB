import logging
import logging.config as logconfig
import re
import pathlib as pb
import datetime
import itertools as it

import numpy as np
import netCDF4 as nc
import pyproj


from cimr_rgb.grid_generator import GRIDS, PROJECTIONS

# TODO: Ad rows and cols variables <= all variables are
# in 1D flattened array and these rows and cols will allow
# the user to build the grid.

# For CIMR:
# rows are # of samples
# cols are
# z dim are the scans

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
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
                # "_FillValue": nc.default_fillvals['f8'],
                "comment": "Grid row index for the chosen output grid. This variable is used to reconstruct the chosen output grid.",
            },
            "cell_row": {
                "units": "Grid x-coordinate",
                "long_name": "Grid row Index for the chosen output grid",
                "grid_mapping": "crs",
                "coverage_content_type": "Grid",
                "valid_range": "0,2147483647",  # depends on the variable type
                # "_Storage": "chunked",
                # "_ChunkSizes": "256, 256",
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
                + "interpolation (among the TBD methods) is used.",
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
        # // global_attributes:
        # :conventions = “CF-1.6”;
        # :id = “TBD”;
        # :naming_authority = “European Space Agency”;
        # :history = “TBD”;
        # :source = “TBD”;
        # :processing_level = “L1c”;
        # :comment = “TBD”
        # :acknowledgement = “TBD”;
        # :licence = None
        # :standard_name_vocabulary = “TBD”;
        # :date_created = “TBD”;
        # :creator_name = “TBD”;
        # :creator_email = “TBD”;
        # :creator_url = “TBD”;
        # :institution “European Space Agency”
        # :project = “CIMR Re-Gridding Toolbox”
        # :program = “TBD”;
        # :contributor_name = “TBD”;
        # :contributor_role = “TBD”;
        # :publisher_name = “TBD”;
        # :publisher_email = “TBD”;
        # :publisher_url = “TBD”;
        # :geospatial_bounds = “TBD”;
        # :geospatial_bounds_crs = “TBD”;
        # :geospatial_bounds_vertical_crs = “TBD”;
        # :geospatial_lat_min = “TBD”;
        # :geospatial_lat_max = “TBD”;
        # :geospatial_lon_min = “TBD”;
        # :geospatial_lon_max = “TBD”;
        # :time_coverage_start = “TBD”;
        # :time_coverage_end = “TBD”;
        # :time_coverage_duration = “TBD”;
        # :time_coverage_resolution = “TBD”;
        # :geospatial_lat_units = “degrees north”;
        # :geospatial_lat_resolution = “TBD”;
        # :geospatial_lon_units = “degrees north”
        # :geospatial_lon_resolution = “TBD”;
        # :date_modified = “TBD”;
        # :date_issued = “TBD”;
        # :date_metadata_modified = “TBD”;
        # :product_version = “TBD”;
        # :platform = “CIMR”
        # :instrument = “CIMR”
        # :metadata_link = “TBD”;
        # :keywords = “TBD”;
        # :keywords_vocabulary = “TBD”;
        # :references = “TBD”;
        # :input_level1b_filenames = “TBD”;
        # :level_01_atbd = “TBD”;
        # :mission_requirement_document = “TBD”;
        # :antenna_pattern_file = “TBD”;
        # :antenna_pattern_source = “TBD”;

        # TODO:- Make so the keys with antenna patterns will not be present if
        #      they are not used or the discrption should be changed into like
        #      it was simulated by RGB software
        #
        # Define global attributes
        GLOBAL_ATTRIBUTES = {
            "conventions": "CF-1.6",
            # "id": "TBD",
            "naming_authority": "European Space Agency",
            # "history": "TBD",
            # "source": "TBD",
            "processing_level": f"{self.config.grid_type}",
            "comment": f"Test data set output that represents an example {self.config.grid_type} product for evaluation of {self.config.input_data_type} instrument",
            # "acknowledgement": "TBD",
            "license": "None",
            # "standard_name_vocabulary": "TBD",
            "date_created": f"{self.config.timestamp} CET",
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
            # "date_modified": "TBD",
            # "date_issued": "TBD",
            "date_metadata_modified": f"{self.config.timestamp} CET",
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

        geospatial_attrs = self.compute_geospatial_attributes(
            grid=GRIDS[self.config.grid_definition],
            projection_string=PROJECTIONS[self.config.projection_definition],
        )

        # Print the derived attributes
        for key, value in geospatial_attrs.items():
            self.logger.info(f"{key}: {value}")
        # print(data_dict["L"].keys())
        # TODO: The min and max values are not calculated correctly and should
        #       (probably) be hardcoded into GRIDS
        GLOBAL_ATTRIBUTES["geospatial_bounds"] = geospatial_attrs["geospatial_bounds"]
        GLOBAL_ATTRIBUTES["geospatial_bounds_crs"] = geospatial_attrs[
            "geospatial_bounds_crs"
        ]
        # lat_min, lat_max = self.get_minmax_vals(array=data_dict["L"]["latitude_fore"])
        GLOBAL_ATTRIBUTES["geospatial_lat_min"] = geospatial_attrs[
            "geospatial_lat_min"
        ]  # lat_min
        GLOBAL_ATTRIBUTES["geospatial_lat_max"] = geospatial_attrs[
            "geospatial_lat_max"
        ]  # lat_max
        # lon_min, lon_max = self.get_minmax_vals(array=data_dict["L"]["longitude_fore"])
        GLOBAL_ATTRIBUTES["geospatial_lon_min"] = geospatial_attrs[
            "geospatial_lon_min"
        ]  # lon_min
        GLOBAL_ATTRIBUTES["geospatial_lon_max"] = geospatial_attrs[
            "geospatial_lon_max"
        ]  # lon_max
        GLOBAL_ATTRIBUTES["geospatial_lat_units"] = "degrees"
        GLOBAL_ATTRIBUTES["geospatial_lat_resolution"] = geospatial_attrs[
            "geospatial_lat_resolution"
        ]
        GLOBAL_ATTRIBUTES["geospatial_lon_units"] = "degrees"
        GLOBAL_ATTRIBUTES["geospatial_lon_resolution"] = geospatial_attrs[
            "geospatial_lon_resolution"
        ]
        # exit()

        return GLOBAL_ATTRIBUTES

    def get_minmax_vals(self, array):
        min_value = np.nanmin(array)  # Ignores NaN
        max_value = np.nanmax(array)  # Ignores NaN
        print(min_value)
        print(max_value)

        return min_value, max_value

    # This is for L1C only
    def compute_geospatial_attributes(self, grid, projection_string):
        """
        Compute geospatial attributes for a given grid and its projection.

        Args:
            grid (dict): Dictionary containing grid parameters like 'x_min', 'y_max', 'res', 'n_cols', 'n_rows'.
            projection_string (str): PROJ string for the grid's projection.

        Returns:
            dict: Dictionary containing derived geospatial attributes.
        """
        # Extract grid properties
        x_min, y_max = grid["x_min"], grid["y_max"]
        res, n_cols, n_rows = grid["res"], grid["n_cols"], grid["n_rows"]

        # Calculate bounding box in projected coordinates
        x_max = x_min + (n_cols * res)
        y_min = y_max - (n_rows * res)

        print(f"x_min = {x_min}, x_max = {x_max}")
        print(f"y_min = {y_min}, y_max = {y_max}")

        # transform to geographic coordinates
        projection = pyproj.Proj(projection_string)
        lon_min, lat_max = projection(x_min, y_max, inverse=True)
        lon_max, lat_min = projection(x_max, y_min, inverse=True)

        print(f"lat_min = {lat_min}, lat_max = {lat_max}")
        print(f"lon_min = {lon_min}, lon_max = {lon_max}")

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

        print(projection_string)

        # exit()

        # Transform a point and its neighbor one resolution unit apart
        lon1, lat1 = projection(  # transformer.transform(
            x_min, y_max
        )  # Top left corner of the grid/bounding box
        lon2, lat2 = projection(  # transformer.transform(
            x_min + res, y_max - res
        )

        # Calculate lat/lon resolution
        lat_res = abs(lat_max - lat_min) / res  # abs(lat2 - lat1)
        lon_res = abs(lon_max - lon_min) / res  # abs(lon2 - lon1)

        print(lon_res, lat_res)
        # exit()

        # Define geospatial bounds as a polygon (counter-clockwise order)
        geospatial_bounds = f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, {lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"

        # Return geospatial attributes
        return {
            "geospatial_lat_min": lat_min,
            "geospatial_lat_max": lat_max,
            "geospatial_lon_min": lon_min,
            "geospatial_lon_max": lon_max,
            "geospatial_bounds": geospatial_bounds,
            "geospatial_bounds_crs": f"EPSG:{grid['epsg']}",  # "EPSG:4326",
            "geospatial_lat_resolution": lat_res,
            "geospatial_lon_resolution": lon_res,
        }

    # TODO: Method to get a list of antenna patterns used in processing
    #       to populate metadata inthe nc file
    def get_cimr_antenna_patterns_list(self):
        beamfiles_paths = []

        for band in self.config.target_band:
            ap_path = pb.Path(self.config.antenna_patterns_path).joinpath(band)
            beamfiles_paths.append([pattern.name for pattern in ap_path.glob("*")])

        beamfiles_paths = ", ".join(list(it.chain.from_iterable(beamfiles_paths)))

        return beamfiles_paths

    def generate_product(self, data_dict: dict):
        # (Old dict that contains) Params from CDL. It can be used
        # later still, so left here as a comment
        # params_to_save = {
        #    "Quality_information": {
        #        "navigation_status_flag": {
        #            "units": "N/A",
        #            "long_name": "Quality information flag summarising the " + \
        #                "navigation quality of each scan.",
        #            "grid_mapping": "crs",
        #            "coverage_content_type": "Grid",
        #            "valid_range": "0,65535",
        #            "_Storage": "chunked",
        #            "_ChunkSizes": "256, 256",
        #            "_FillValue": "0",
        #            "comment": "A TBD-bit binary string of 1’s and 0's " +\
        #            "indicating the quality of the L1b acquisition " + \
        #            "conditions. A ‘0’ indicates that the L1c samples met a " + \
        #            "certain quality criterion and a ‘1’ that it did not. " + \
        #            "Bit position ‘0’ refers to the least significant bit. " + \
        #            "navigation_status_flag summarises the navigation " + \
        #            "quality Of each scan."
        #            },
        #        "scan_quality_flag": {
        #            "units": "N/A",
        #            "long_name": "Quality information flag summarising the " + \
        #                "overall scan quality.",
        #            "grid_mapping": "crs",
        #            "coverage_content_type": "Grid",
        #            "valid_range": "0,65535",
        #            "_Storage": "chunked",
        #            "_ChunkSizes": "256, 256",
        #            "_FillValue": "0",#nc.default_fillvals['f8'],
        #            "comment": "A TBD-bit binary string of 1’s and 0’s " + \
        #            "indicating the quality of the L1b acquisition " + \
        #            "conditions. A ‘0’ indicates that the L1c samples met a " + \
        #            "certain quality criterion and a ‘1’ that it did not. " + \
        #            "Bit position ‘0’ refers to the least significant bit. " + \
        #            "scan_quality_flag summarises the scan quality"
        #            },
        #            "temperatures_flag": {
        #                "units": "N/A",
        #                "long_name": "Quality information indicating degraded " + \
        #                    "instrument temperature cases.",
        #                "grid_mapping": "crs",
        #                "coverage_content_type": "Grid",
        #                "valid_range": "0,65535",
        #                "_Storage": "chunked",
        #                "_ChunkSizes": "256, 256",
        #                "_FillValue": "0",#nc.default_fillvals['f8'],
        #                "comment": "A TBD-bit binary string of 1’s and 0’s " + \
        #                "indicating the quality of the L1b acquisition " + \
        #                "conditions. A ‘0’ indicates that the L1c samples met a " + \
        #                "certain quality criterion and a ‘1’ that it did not. " + \
        #                "Bit position ‘0’ refers to the least significant bit. " + \
        #                "temperatures_flag to indicates degraded instrument " + \
        #                "temperature cases."
        #                }
        #            }
        #        }

        # TODO: Remove pickled object and pass in proper dictionary to be saved
        # file_path = pb.Path("dpr/data_dict_out.pkl")
        ## Open the file in read-binary mode and load the object
        # with open(file_path, 'rb') as file:
        #    loaded_object = pickle.load(file)

        # params_to_save = CDL

        # outfile = "test_l1c.nc"
        # outfile = self.get_processor_filename()
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

        # print(f"L band cell_row_aft:  {data_dict['L']['cell_row_aft']}")
        # print(f"X band cell_row_aft:  {data_dict['X']['cell_row_aft']}")
        # exit()
        GLOBAL_ATTRIBUTES = self.generate_global_metadata(data_dict=data_dict)

        with nc.Dataset(outfile, "w", format="NETCDF4") as dataset:
            # Set each global attribute in the netCDF file
            # for attr, value in CDL["GLOBAL_ATTRIBUTES"].items():#GLOBAL_ATTRIBUTES.items():
            for attr, value in GLOBAL_ATTRIBUTES.items():
                dataset.setncattr(attr, value)

            # The L1R template has the following dims:

            # CIMR_E2ESv110_L1B_Product_Format_v0.6.nc

            # netcdf CIMR_E2ESv110_L1B_Product_Format_v0.6 {
            # dimensions:
            # 	n_feeds_L_BAND = 1 ;
            # 	n_feeds_C_BAND = 4 ;
            # 	n_feeds_X_BAND = 4 ;
            # 	n_feeds_KU_BAND = 8 ;
            # 	n_feeds_KA_BAND = 8 ;
            # 	n_scans = 2 ;
            # 	n_samples_L_BAND = 138 ;
            # 	n_samples_C_BAND = 549 ;
            # 	n_samples_X_BAND = 561 ;
            # 	n_samples_KU_BAND = 1538 ;
            # 	n_samples_KA_BAND = 2079 ;
            if self.config.grid_type == "L1R":
                # TODO: this should be in per band calculations
                # dataset.createDimension('n_feeds_L_BAND', 1) #None)
                # dataset.createDimension('n_feeds_X_BAND', 4) #None)
                # dataset.createDimension('n_feeds_C_BAND', 4) #None)
                # dataset.createDimension('n_feeds_KU_BAND', 8) #None)
                # dataset.createDimension('n_feeds_KA_BAND', 8) #None)

                # dataset.createDimension('n_scans', 2) #None)
                # dataset.createDimension('n_samples_L_BAND', 138) #None)
                # dataset.createDimension('n_samples_X_BAND', 561) #None)
                # dataset.createDimension('n_samples_C_BAND', 549) #None)
                # dataset.createDimension('n_samples_KU_BAND', 1538) #None)
                # dataset.createDimension('n_samples_KA_BAND', 2079) #None)

                for band_name in data_dict.keys():
                    dataset.createDimension(f"n_feeds_{band_name}_BAND", None)
                    dataset.createDimension(f"n_samples_{band_name}_BAND", None)

                dataset.createDimension("n_scans", None)

            elif self.config.grid_type == "L1C":
                # Also, there should be per band calculations for n_feeds
                # For L1C dimensions
                # time = 0 // currently 1 <= single integer value
                # x = {256..16384}
                # y = {256..16384}
                # n_l1b_scans = TBD
                # n_samples = 0
                # n_feeds_[L_BAND|C_BAND|X_BAND|KA_BAND|KU_BAND] = [1, 4, 4, 8, 8]
                for band_name in data_dict.keys():
                    dataset.createDimension(f"n_feeds_{band_name}_BAND", None)
                    dataset.createDimension(f"n_samples_{band_name}_BAND", None)

                dataset.createDimension("n_scans", None)

                # Creating Dimentions according to cdl
                dataset.createDimension("time", 1)  # None)
                dataset.createDimension("y", None)
                dataset.createDimension("x", None)

            # print(self.config.target_band)

            # Creating nested groups according to cdl
            if self.config.grid_type == "L1C":
                top_group = dataset.createGroup(f"{self.config.projection_definition}")

            elif self.config.grid_type == "L1R":
                # target_band is a list
                top_group = dataset.createGroup(
                    f"{self.config.target_band[0]}_BAND_TARGET"
                )
            # exit()
            # top_group  = dataset.createGroup(f"{self.config.projection_definition}")
            # Don't need this field as of latest diagram
            # data_group        = projection_group.createGroup("Data")

            # Loop through the parameters defined inside CDL and compare their
            # names to the ones provided inside pickled file. If they coincide
            # we write them into specific group (defined in CDL). In addition,
            # CDL values for CIMR have dimensions (time, x, y) while SMAP has
            # only 1, so we also programmatically figure out the dimensonf of
            # the numpy array provided and save the data accordingly.
            #
            # [Note]: We start looping in this way and not from data_dict, because
            # data_dict may contain _fore|_aft suffixes
            for group_field, group_vals in CDL["LOCAL_ATTRIBUTES"].items():
                # if group_field == "GLOBAL_ATTRIBUTES":
                #    continue
                # else:

                self.logger.info(
                    f"Creating group: {group_field}"
                )  # . Group val: {group_vals}")

                # group = data_group.createGroup(group_field)
                group = top_group.createGroup(group_field)

                # Some fields of Quality_information are not sub group of bands
                # if group_field == "Quality_information":

                # Looping through data dictionary and retrieving its variables (per band)
                for band_name, band_var in data_dict.items():
                    # Removing/skipping cell_row and cell_col if we encounter L1R
                    # (since it is not needed for L1R)
                    if self.config.grid_type == "L1R" and (
                        band_var == "cell_row" or band_var == "cell_col"
                    ):
                        continue

                    # Processing_flags are defined as a separate field
                    # and not as sub field of specific band
                    if group_field == "Processing_flags":
                        band_group = group
                    else:
                        band_group = group.createGroup(f"{band_name}_BAND")

                    for var_name, var_val in band_var.items():
                        var_shape = var_val.shape

                        # TODO:
                        # For L1r we will have 3 dimensions: scan, sample, and feed <= scan is the same for all bands, but sample and feed are not
                        # For L1C we will have 3 dimensions: time, y, and x <= the same for all bands
                        #
                        # So add if else statement
                        if self.config.grid_type == "L1C":
                            var_dim = self.determine_dimension_l1c(var_shape)

                        elif self.config.grid_type == "L1R":
                            var_dim = self.determine_dimension_l1r(band_name, var_shape)
                        # Removing cell_row and cell_col from dictionary
                        # since it is not needed for L1R

                        # TODO: Create a generic method
                        # var_dim  = self.determine_dimension(grid_type, band_name, var_shape)

                        # Creating a list of complete cariables to regrid based on CDL
                        # and whether user chose to split scans into fore and aft
                        if self.config.split_fore_aft:
                            # print("True")
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
                        # print(var_name, regrid_var)

                        if (var_name in regrid_vars) and (
                            "regridding_l1b_orphans" not in var_name
                        ):
                            # var_type = type(var_val[0,0,0])

                            # print(var_name, var_val.dtype, dtype_map.get(var_val.dtype, None), dtype_map.get(var_val.dtype))

                            # print(type(var_val[0]))
                            # print(type(var_val))
                            # print(var_name, var_val.dtype)
                            # exit()
                            var_type = self.get_netcdf_dtype(var_val.dtype)
                            var_fill = nc.default_fillvals[var_type]

                            self.logger.debug(
                                f"{group_field}, {band_name}, {var_name}, {var_type}, {var_fill}, {var_dim}"
                            )

                            # Determine the appropriate slice based on variable shape
                            slices = tuple(slice(None) for _ in var_shape)
                            # print(slices, var_shape)
                            # print(var_shape)
                            # print(var_dim)

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
                            var_data = band_group.createVariable(
                                varname=var_name,
                                datatype=var_type,  # "double",
                                dimensions=var_dim,  # ('x'),
                                fill_value=var_fill,  # group_vals[regrid_var]["_FillValue"]
                            )
                            # print(var_name)
                            # Assign values to the variable
                            var_data[slices] = var_val
                            # print(var_data)
                            # exit()

                            # Loop through the dictionary and set attributes for the variable
                            for attr_name, attr_value in group_vals[regrid_var].items():
                                # TODO: The _FillValue field is kind of obsolete, because we
                                # define it above via fill_value parameter (but lets leave it here for now)
                                if attr_name != "_FillValue" and attr_name != "comment":
                                    # print(attr_name)
                                    # Use setncattr to assign the attribute
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
                                    # print(substitution)

                                    # pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_" #\[fore\|aft\]"
                                    # substitution = f"{band_name}_BAND_"
                                    attr_value = re.sub(
                                        pattern, substitution, attr_value
                                    )
                                    # var_data.setncattr(attr_name, attr_value)

                                    # Checking whther there is any patter left of the following format
                                    pattern = r"\[L\|C\|X\|KU\|KA\]_BAND"
                                    substitution = f"{band_name}_BAND"
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

                        elif (var_name in regrid_vars) and (
                            "regridding_l1b_orphans" in var_name
                        ):
                            # elif ("regridding_l1b_orphans" in var_name):
                            var_dim = self.determine_dimension_l1r(band_name, var_shape)
                            var_type = self.get_netcdf_dtype(var_val.dtype)
                            var_fill = nc.default_fillvals[var_type]

                            self.logger.debug(
                                f"{group_field}, {band_name}, {var_name}, {var_type}, {var_fill}, {var_dim}"
                            )

                            # Determine the appropriate slice based on variable shape
                            slices = tuple(slice(None) for _ in var_shape)
                            var_data = band_group.createVariable(
                                varname=var_name,
                                datatype=var_type,  # "double",
                                dimensions=var_dim,  # ('x'),
                                fill_value=var_fill,  # group_vals[regrid_var]["_FillValue"]
                            )
                            var_data[slices] = var_val
                            # Loop through the dictionary and set attributes for the variable
                            for attr_name, attr_value in group_vals[regrid_var].items():
                                if attr_name != "_FillValue" and attr_name != "comment":
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
                                    # print(substitution)

                                    # pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_" #\[fore\|aft\]"
                                    # substitution = f"{band_name}_BAND_"
                                    attr_value = re.sub(
                                        pattern, substitution, attr_value
                                    )
                                    # var_data.setncattr(attr_name, attr_value)

                                    # Checking whther there is any patter left of the following format
                                    pattern = r"\[L\|C\|X\|KU\|KA\]_BAND"
                                    substitution = f"{band_name}_BAND"
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
                            # print(var_name)
                            # print(var_val.shape)
                            # exit()

    # def create_cdf_var(self, var_shape, var_name, var_val, band_group, group_vals):

    #    if len(var_shape) == 1:
    #        var_data = band_group.createVariable(
    #                var_name,
    #                "double",
    #                ('x'),
    #                fill_value = group_vals[var_name]["_FillValue"]
    #                )
    #        var_data[:] = var_val
    #    elif len(var_shape) == 2:
    #        var_data = band_group.createVariable(
    #                var_name,
    #                "double", ('x', 'y'),
    #                fill_value = group_vals[var_name]["_FillValue"]
    #                )
    #        var_data[:, :] = var_val
    #    elif len(var_shape) == 3:
    #        var_data = band_group.createVariable(
    #                var_name,
    #                "double",
    #                ('time', 'x', 'y'),
    #                fill_value = group_vals[var_name]["_FillValue"]
    #                )
    #        var_data[:, :, :] = var_val
    #    else:
    #        # Return a generic message or handle error for unknown shapes
    #        raise ValueError(f"Unsupported shape with {len(var_shape)} dimensions: {var_shape}")

    #    # TODO: fix the re pattern to also include fore|aft
    #    # when this loop will be appropriate

    #    # Loop through the dictionary and set attributes for the variable
    #    #for attr_name, attr_value in group_vals[var_name].items():
    #    #    if attr_name != "_FillValue" and attr_name != "comment":
    #    #        #print(attr_name)
    #    #        # Use setncattr to assign the attribute
    #    #        var_data.setncattr(attr_name, attr_value)
    #    #    elif attr_name == "comment":
    #    #        pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_" #\[fore\|aft\]"
    #    #        substitution = f"{band_name}_BAND_"
    #    #        attr_value = re.sub(pattern, substitution, attr_value)
    #    #        var_data.setncattr(attr_name, attr_value)

    def get_netcdf_dtype(self, np_dtype: np.dtype):
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
            np_dtype (numpy.dtype): The numpy data type to convert.

        Returns:
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

    def determine_dimension(self, var_shape: tuple) -> tuple: ...

    def determine_dimension_l1r(self, band_name, var_shape: tuple) -> tuple:
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
            - ('n_scans', 'n_sample', 'n_feeds') for 3D shapes (e.g., (1, 111, 111))

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
                "n_scans",
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

        if len(var_shape) == 1:
            # 1D case
            return ("x",)
        elif len(var_shape) == 2:
            # 2D case
            return ("y", "x")
        elif len(var_shape) == 3:
            # 3D case
            return ("time", "y", "x")
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
