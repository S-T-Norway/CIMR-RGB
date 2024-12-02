from logging import config
import re 
#import pickle 
import pathlib as pb 
import datetime 

import numpy  as np 
import netCDF4 as nc 


from cimr_rgb.grid_generator import GRIDS 

# TODO: Ad rows and cols variables <= all variables are 
# in 1D flattened array and these rows and cols will allow 
# the user to build the grid.

# For CIMR:  
# rows are # of samples
# cols are 
# z dim are the scans 

CDL = {
    "Measurement": {
        "bt_h": { 
            "units": "K",
            "long_name": "H-polarised TOA Brightness Temperatures",
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Earth-Gridded TOA h-polarised" + \
             " [L|C|X|KU|KA]_BAND_[fore|aft] BTS" + \
             " interpolated on a TBD-km grid" 
        }, 
        "bt_v": {
            "units": "K",
            "long_name": "V-polarised TOA Brightness Temperatures",
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Earth-Gridded TOA v-polarised" + \
             " [L|C|X|KU|KA]_BAND_[fore|aft] BTS" + \
             " interpolated on a TBD-km grid"
        }, 
        "bt_3": {
            "units": "K",
            "long_name": "Stokes 3-polarised TOA Brightness Temperatures", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Earth-Gridded TOA [L|C|X|KU|KA]_BAND_[fore|aft] BTS " + \
            "interpolated on a TBD-km grid, third stokes parameter " + \
            "of the surface polarisation basis"
        }, 
        "bt_4": {
            "units": "K",
            "long_name": "Stokes 4-polarised TOA Brightness Temperatures", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Earth-Gridded TOA [L|C|X|KU|KA]_BAND_[fore|aft] BTS " + \
            "interpolated on a TBD-km grid, fourth stokes parameter " + \
            "of the surface polarisation basis"
        }, 
        "faraday_rot_angle":   {
            "units": "deg",
            "long_name": "Interpolated Faraday Rotation Angle of acquisitions", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] faraday " + \
            "rotation angle corresponding to the measured BT value. " + \
            "The value of the faraday rotation angle will be scaled " + \
            "with the interpolation weights of all faraday rotation " + \
            "angles Earth samples used in the interpolation of that " + \
            "grid cell."
        }, 
        "geometric_rot_angle": {
            "units": "deg",
            "long_name": "Interpolated Geometric Rotation angle of acquisitions.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] geometric " + \
            "rotation angle corresponding to the measured BT value. " + \
            "The value of the geometric rotation angle will be " + \
            "scaled with the interpolation weights of all geometric " + \
            "rotation angles Earth samples used in the " + \
            "interpolation of that grid cell." 
        }, 
        "nedt_h": {
            "units": "K",
            "long_name": "Radiometric resolution of each measured BT.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Radiometric resolution of each measured BT." 
        }, 
        "nedt_v": {
            "units": "K",
            "long_name": "Radiometric resolution of each measured BT.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Radiometric resolution of each measured BT." 
        }, 
        "nedt_3": {
            "units": "K",
            "long_name": "Radiometric resolution of each measured BT.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Radiometric resolution of each measured BT." 
        }, 
        "nedt_4": {
            "units": "K",
            "long_name": "Radiometric resolution of each measured BT.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Radiometric resolution of each measured BT." 
        }, 
        "tsu":  {
            "units": "K",
            "long_name": "Total standard uncertainty for each measured BT.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Total standard uncertainty for each measured BT." 
        }, 
        "instrument_status": {
            "units": "N/A",
            "long_name": "Instrument Calibration or Observation mode.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Instrument Calibration or Observation mode, " + \
            "for all samples. L1c values will consider the majority " + \
            "status values from input L1b samples."
        }, 
        "land_sea_content":  {
            "units": "N/A",
            "long_name": "Land/Sea content of the measured pixel.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Land/Sea content of the measured pixel, " + \
            "200 for full sea content, 0 for full land content."
        }, 
        "regridding_n_samples": {
            "units": "N/A",
            "long_name": "Number of earth samples used for interpolation", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
             "comment": "Number of L1b [h|v|t3|t4] polarised " + \
             "[L|C|X|KU|KA]_BAND_[fore|aft] brightness temperature " + \
             "Earth samples used in the [Backus-Gilbert|rSIR|LW] " + \
             "remapping interpolation."
            }, 
        "regridding_quality_measure": {
            "units": "N/A",
            "long_name": "Algorithm Specific Optimal Value of Regularization Parameter.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "The optimal value of a parameter for the " + \
             "[Backus-Gilbert|rSIR|LW]_[fore|aft] that controls the " + \
             "trade-off between noise amplification and " + \
             "regularisation. For BG it is the optimal value for the " + \
             "smoothing parameter, while for [rSIR|LW] it is the " + \
             "number of iterations to achieve a chosen level of " + \
             "residual error. In case of [NN|IDS|DIB] regularisation " + \
             "is not performed and the parameter will take on the " + \
             "_FillValue."  
        }, 
        "regridding_l1b_orphans": {
            "units": "N/A",
            "long_name": "Indication of L1b orphaned Earth samples.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "TBD",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Whether each [L|C|X|KU|KA]_BAND L1b measurement sample was " + \
            "unused (1) or used (0) in [Backus-Gilbert|rSIR|LW] regridding " + \
            "interpolation of [fore|aft] scan samples. In the fore-scan " + \
            "regridding nearly all aft scan samples would be orphan " + \
            "(unused), for instance, and vice versa. It would also occur if " + \
            "the swath stretches outside the projection window. Orphaned " + \
            "samples may also occur if nearest neighbour or linear " + \
            "interpolation (among the TBD methods) is used."
        } 
    }, 
    "Navigation": {
        "acq_time_utc": {
            "units": "N/A",
            "long_name": "Interpolated UTC Acquisition time of Earth Sample acquisitions.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,N/A",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "UTC acquisition times expressed in seconds " +\
            "(seconds since 2000-01-01 00:00:00 UTC). The value of " + \
            "time_earth will be scaled with the interpolation " + \
            "weights of all time_earth Earth samples used in the " + \
            "interpolation of that grid cell. "
        }, 
        "azimuth": {
            "units": "deg",
            "long_name": "Interpolated Earth Azimuth angle of the acquisitions.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,359.99",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] " + \
            "Earth observation azimuth angles of the acquisitions, " + \
            "positive counterclockwise from due east. The value of " + \
            "observation azimuth angle will be scaled with the " + \
            "interpolation weights of all observation azimuth angle " + \
            "Earth samples used in the interpolation of that grid " \
            "cell." 
        }, 
        "latitude": {
            "units": "deg",
            "long_name": "Latitude of the centre of a TBD-km PROJ grid cell.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "-90, 90",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256", 
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Latitude of the centre of a TBD-km PROJ grid cell."
        }, 
        "longitude": {
            "units": "deg",
            "long_name": "Longitude of the centre of a TBD-km PROJ grid cell.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "-180, 179.99",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256", 
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Longitude of the centre of a TBD-km PROJ grid cell."
        }, 
        "oza": {
            "units": "deg",
            "long_name": "Interpolated Observation Zenith Angle of acquisitions.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,359.99",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] " + \
            "Earth Observation zenith angles of the acquisitions. " + \
            "The value of OZA will be scaled with the interpolation " + \
            "weights of all observation OZA Earth samples used in " + \
            "the interpolation of that grid cell. The OZA is defined " + \
            "as the included angle between the antenna Boresight " + \
            "vector and the normal to the Earth's surface."
        }, 
        "processing_scan_angle": {
            "units": "deg",
            "long_name": "Interpolated scan angle of acquisitions", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,359.99",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",   
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "The processing scan angle of the L1b " + \
            "[L|C|X|KU|KA]_BAND_[fore|aft] Earth view samples. The " + \
            "value of scan angle will be scaled with the " + \
            "interpolation weights of all scan angle Earth samples " + \
            "used in the interpolation of that grid cell. " + \
            "Measurements from different feed horns are combined. " + \
            "The scan angle is defined as the azimuth angle of the " + \
            "antenna boresight measured from the ground track " + \
            "vector. The scan angle is 90° when the boresight points " + \
            "in the same direction as the ground track vector and " + \
            "increases clockwise when viewed from above." 
        }, 
        "solar_azimuth": {
            "units": "deg",
            "long_name": "Interpolated solar azimuth angle of acquisitions", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,359.99",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",   
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] " + \
            "solar azimuth angle of acquisitions.The value of " + \
            "solar_azimuth will be scaled with the interpolation " + \
            "weights of all solar_azimuth Earth samples used in the " + \
            "interpolation of that grid cell." 
        }, 
        "solar_zenith": {
            "units": "deg",
            "long_name": "Interpolated solar zenith angle of acquisitions", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,359.99",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": nc.default_fillvals['f8'], 
            "comment": "Level 1b [L|C|X|KU|KA]_BAND_[fore|aft] " + \
            "solar zenith angle of acquisitions.The value of " + \
            "solar_zenith will be scaled with the interpolation " + \
            "weights of all solar_zenith Earth samples used in the " + \
            "interpolation of that grid cell."
        }, 

    }, 
    "Processing_flags": {
        "processing_flags": {
            "units": "N/A",
            "long_name": "L1c processing performance related information.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,65535",
            "_Storage": "chunked",
            "_ChunkSizes": "256, 256",  
            #"_FillValue": "0",#nc.default_fillvals['f8'], 
            "comment": "A TBD-bit binary string of 1’s and 0’s " + \
            "indicating a variety of TBD information related to the " + \
            "processing of L1c/r data."
        }
    },  
    "Quality_information": {
        "calibration_flag": {
            "units": "K",
            "long_name": "A TBD-bit binary string of 1’s and 0’s indicating the quality " + \
            "of the L1b [L|C|X|KU|KA]_BAND Earth samples used to derive the " + \
            "L1c data. A ‘0’ indicates that the L1c samples met a certain " + \
            "quality criterion and a ‘1’ that it did not. Bit position ‘0’ " + \
            "refers to the least significant bit. The calibration flag " + \
            "summarises the calibration quality for each channel and scan. " + \
            "The data quality flag summarises the BT data quality for each " + \
            "channel and scan.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,65535",
            "_Storage": "chunked",
            "_ChunkSizes": "256,256",
            #"_FillValue": "0" 
        }, 
        "data_quality_flag": {
            "units": "K",
            "long_name": "A TBD-bit binary string of 1’s and 0’s indicating the quality " + \
            "of the L1b [L|C|X|KU|KA]_BAND Earth samples used to derive the " + \
            "L1c data. A ‘0’ indicates that the L1c samples met a certain " + \
            "quality criterion and a ‘1’ that it did not. Bit position ‘0’ " + \
            "refers to the least significant bit. The calibration flag " + \
            "summarises the calibration quality for each channel and scan. " + \
            "The data quality flag summarises the BT data quality for each " + \
            "channel and scan.", 
            "grid_mapping": "crs",
            "coverage_content_type": "Grid",
            "valid_range": "0,65535",
            "_Storage": "chunked",
            "_ChunkSizes": "256,256",
            #"_FillValue": "0" 
        } 
    } 
} 





class ProductGenerator: 

    def __init__(self, config):
        self.config = config 
        self.logger = config.logger  
        


    def generate_product(self, data_dict: dict): 

        # (Old dict that contains) Params from CDL. It can be used 
        # later still, so left here as a comment  
        #params_to_save = {
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
        #file_path = pb.Path("dpr/data_dict_out.pkl")
        ## Open the file in read-binary mode and load the object
        #with open(file_path, 'rb') as file:
        #    loaded_object = pickle.load(file)

        #params_to_save = CDL 

        #outfile = "test_l1c.nc"
        #outfile = self.get_processor_filename()
        grid_res = re.search(r'(\d+(?:\.\d+)?)km', self.config.grid_definition).group() 
        # Get the current date and time
        l1c_utc_time = datetime.datetime.now()

        # Format the date and time as "YYYYMMDDHHMMSS"
        l1c_utc_time = l1c_utc_time.strftime("%Y%m%d%H%M%S")

        outfile = f"{self.config.input_data_type}_{self.config.grid_type}_{self.config.regridding_algorithm}_{grid_res}_{l1c_utc_time}.nc" 

        outfile = pb.Path(f"{self.config.output_path}/{outfile}").resolve()
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


        # TODO: - Convert this into a method? 
        #       - Add product_version parameter file into config.xml 
        # Define global attributes
        GLOBAL_ATTRIBUTES = {
            "conventions": "CF-1.6",
            "id": "TBD",
            "naming_authority": "European Space Agency",
            "history": "TBD",
            "source": "TBD",
            "processing_level": f"{self.config.grid_type}",
            "comment": "TBD",
            "acknowledgement": "TBD",
            "license": "None",
            "standard_name_vocabulary": "TBD",
            "date_created": f"{l1c_utc_time}",
            "creator_name": "TBD",
            "creator_email": "TBD",
            "creator_url": "TBD",
            "institution": "European Space Agency",
            "project": "CIMR Re-Gridding Toolbox",
            "program": "TBD",
            "contributor_name": "TBD",
            "contributor_role": "TBD",
            "publisher_name": "TBD",
            "publisher_email": "TBD",
            "publisher_url": "TBD",
            "geospatial_bounds": "TBD",
            "geospatial_bounds_crs": "TBD",
            "geospatial_bounds_vertical_crs": "TBD",
            "geospatial_lat_min": "TBD",
            "geospatial_lat_max": "TBD",
            "geospatial_lon_min": "TBD",
            "geospatial_lon_max": "TBD",
            "time_coverage_start": "TBD",
            "time_coverage_end": "TBD",
            "time_coverage_duration": "TBD",
            "time_coverage_resolution": "TBD",
            "geospatial_lat_units": "degrees north",
            "geospatial_lat_resolution": "TBD",
            "geospatial_lon_units": "degrees north",
            "geospatial_lon_resolution": "TBD",
            "date_modified": "TBD",
            "date_issued": "TBD",
            "date_metadata_modified": "TBD",
            "product_version": "TBD",
            "platform": f"{self.config.input_data_type}",
            "instrument": f"{self.config.input_data_type}",
            "metadata_link": "TBD",
            "keywords": "TBD",
            "keywords_vocabulary": "TBD",
            "references": "TBD",
            "input_level1b_filenames": f"{pb.Path(self.config.input_data_path).resolve().name}", #"TBD",
            "level_01_atbd": "TBD",
            "mission_requirement_document": "TBD",
            "antenna_pattern_files": "TBD",
            "antenna_pattern_source": "TBD"
        }

        with nc.Dataset(outfile, "w", format = "NETCDF4") as dataset: 

            # Set each global attribute in the netCDF file
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
                dataset.createDimension('n_feeds_L_BAND', 1) #None)
                dataset.createDimension('n_feeds_X_BAND', 4) #None)
                dataset.createDimension('n_feeds_C_BAND', 4) #None)
                dataset.createDimension('n_feeds_KU_BAND', 8) #None)
                dataset.createDimension('n_feeds_KA_BAND', 8) #None)

                dataset.createDimension('n_scans', 2) #None)
                dataset.createDimension('n_samples_L_BAND', 138) #None)
                dataset.createDimension('n_samples_X_BAND', 561) #None)
                dataset.createDimension('n_samples_C_BAND', 549) #None)
                dataset.createDimension('n_samples_KU_BAND', 1538) #None)
                dataset.createDimension('n_samples_KA_BAND', 2079) #None)


            # For L1C dimensions  
            # time = 0 // currently 1 <= single integer value 
            # x = {256..16384}
            # y = {256..16384}
            # n_l1b_scans = TBD
            # n_samples = 0
            # n_feeds_[L_BAND|C_BAND|X_BAND|KA_BAND|KU_BAND] = [1, 4, 4, 8, 8]

            # Creating Dimentions according to cdl 
            dataset.createDimension('time', 1) #None)
            dataset.createDimension('y', None)
            dataset.createDimension('x', None) 

            print(self.config.target_band)



            # Creating nested groups according to cdl 
            if self.config.grid_type == "L1C": 
                top_group  = dataset.createGroup(f"{self.config.projection_definition}") 
            elif self.config.grid_type == "L1R":
                # target_band is a list 
                top_group  = dataset.createGroup(f"{self.config.target_band[0]}_BAND_TARGET") 
            #exit() 
            #top_group  = dataset.createGroup(f"{self.config.projection_definition}") 
            # Don't need this field as of latest diagram 
            #data_group        = projection_group.createGroup("Data")

            # Loop through the parameters defined inside CDL and compare their
            # names to the ones provided inside pickled file. If they coincide
            # we write them into specific group (defined in CDL). In addition,
            # CDL values for CIMR have dimensions (time, x, y) while SMAP has
            # only 1, so we also programmatically figure out the dimensonf of
            # the numpy array provided and save the data accordingly. 
            for group_field, group_vals in CDL.items(): 

                self.logger.info(f"Creating group: {group_field}")#. Group val: {group_vals}")

                #group = data_group.createGroup(group_field)
                group = top_group.createGroup(group_field)

                # Some fields of Quality_information are not sub group of bands 
                #if group_field == "Quality_information": 

                # Looping through data dictionary and retrieving its variables (per band) 
                for band_name, band_var in data_dict.items(): 

                    # Processing_flags are defined as a separate field 
                    # and not as sub field of specific band
                    if group_field == "Processing_flags": 
                        band_group = group 
                    else: 
                        band_group = group.createGroup(f"{band_name}_BAND")

                    for var_name, var_val in band_var.items(): 

                        var_shape = var_val.shape

                        # Creating a list of complete cariables to regrid based on CDL 
                        # and whether user chose to split scans into fore and aft 
                        if self.config.split_fore_aft: 
                            #print("True")
                            fore = [ key + "_fore" for key in group_vals.keys() ]
                            aft  = [ key + "_aft"  for key in group_vals.keys() ]
                            regrid_vars = fore + aft 
                        else: 
                            regrid_vars = [ key for key in group_vals.keys() ]

                        # Removing the _fore and _aft from the variable name 
                        # to get the metadata from CDL (it is almost the same for 
                        # both of them anyway). The idea is to compare actual 
                        # variable to the variable from the CDL above  
                        regrid_var = var_name.replace("_fore", "") if "_fore" in var_name \
                                else var_name.replace("_aft", "")  if "_aft" in var_name \
                                else var_name
                        #print(var_name, regrid_var)

                        
                        if var_name in regrid_vars: 


                            #var_type = type(var_val[0,0,0])

                            # print(var_name, var_val.dtype, dtype_map.get(var_val.dtype, None), dtype_map.get(var_val.dtype))

                            #print(type(var_val[0]))
                            #print(type(var_val))
                            #print(var_name, var_val.dtype)
                            #exit() 
                            var_type = self.get_netcdf_dtype(var_val.dtype)
                            var_fill = nc.default_fillvals[var_type] 

                            # TODO: 
                            # For L1r we will have 3 dimensions: scan, sample, and feed <= scan is the same for all bands, but sample and feed are not  
                            # For L1C we will have 3 dimensions: time, y, and x <= the same for all bands    
                            # 
                            # So add if else statement 
                            if self.config.grid_type == "L1C": 

                                var_dim  = self.determine_dimension_l1c(var_shape)

                            elif self.config.grid_type == "L1R":

                                var_dim  = self.determine_dimension_l1r(band_name, var_shape)


                            # TODO: Create a generic method  
                            # var_dim  = self.determine_dimension(grid_type, band_name, var_shape)



                            self.logger.info(f"{group_field}, {band_name}, {var_name}, {var_type}, {var_fill}, {var_dim}")

                            # Determine the appropriate slice based on variable shape
                            slices = tuple(slice(None) for _ in var_shape)
                            #print(slices, var_shape)

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
                                    varname    = var_name, 
                                    datatype   = var_type, #"double", 
                                    dimensions = var_dim, #('x'), 
                                    fill_value = var_fill #group_vals[regrid_var]["_FillValue"]
                                    ) 
                            # Assign values to the variable
                            var_data[slices] = var_val
                            #print(var_data)
                            #exit() 

                            # Loop through the dictionary and set attributes for the variable
                            for attr_name, attr_value in group_vals[regrid_var].items():

                                # TODO: The _FillValue field is kind of obsolete, because we 
                                # define it above via fill_value parameter (but lets leave it here for now)
                                if attr_name != "_FillValue" and attr_name != "comment": 

                                    #print(attr_name)
                                    # Use setncattr to assign the attribute
                                    var_data.setncattr(attr_name, attr_value)

                                elif attr_name == "comment": 

                                    pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_\[fore\|aft\]" 

                                    if self.config.split_fore_aft: 
                                        substitution = f"{band_name}_BAND_fore" if "_fore" in var_name \
                                                else f"{band_name}_BAND_aft" 
                                    else: 
                                        substitution = f"{band_name}_BAND" 
                                    #print(substitution) 

                                    #pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_" #\[fore\|aft\]" 
                                    #substitution = f"{band_name}_BAND_" 
                                    attr_value = re.sub(pattern, substitution, attr_value)
                                    #var_data.setncattr(attr_name, attr_value)

                                    # Checking whther there is any patter left of the following format 
                                    pattern = r"\[L\|C\|X\|KU\|KA\]_BAND" 
                                    substitution = f"{band_name}_BAND" 
                                    attr_value = re.sub(pattern, substitution, attr_value)


                                    pattern = r"\[fore\|aft\]" 
                                    if self.config.split_fore_aft: 
                                        substitution = "fore" if "_fore" in var_name \
                                                else "aft" 
                                    else: 
                                        # Just leave it the way it is 
                                        substitution = "[fore|aft]" 
                                    attr_value = re.sub(pattern, substitution, attr_value)

                                    # Setting comment attribute 
                                    var_data.setncattr(attr_name, attr_value)


    #def create_cdf_var(self, var_shape, var_name, var_val, band_group, group_vals):

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
            np.dtype('int8'):    'i1',     # 1-byte signed integer
            np.dtype('uint8'):   'u1',    # 1-byte unsigned integer
            np.dtype('int16'):   'i2',    # 2-byte signed integer
            np.dtype('uint16'):  'u2',   # 2-byte unsigned integer
            np.dtype('int32'):   'i4',    # 4-byte signed integer
            np.dtype('uint32'):  'u4',   # 4-byte unsigned integer
            np.dtype('int64'):   'i8',    # 8-byte signed integer
            np.dtype('uint64'):  'u8',   # 8-byte unsigned integer
            np.dtype('float32'): 'f4',  # 4-byte floating-point
            np.dtype('float64'): 'f8',  # 8-byte floating-point
            np.dtype('S1'):      'S1',       # 1-byte character
            np.dtype('str'):      str,       # Variable-length string
        }


        return dtype_map.get(np_dtype, None) 


    def determine_dimension(self, var_shape: tuple) -> tuple:
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
            return ('x',)
        elif len(var_shape) == 2:
            # 2D case
            return ('y', 'x')
        elif len(var_shape) == 3:
            # 3D case
            return ('time', 'y', 'x')
        else:
            # Return a generic message or handle error for unknown shapes
            raise ValueError(f"Unsupported shape with {len(var_shape)} dimensions: {var_shape}")


    def determine_dimension_l1r(self, band_name, var_shape): 
        ... 

    def determine_dimension_l1c(self, var_shape): 
        ... 



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

        #print(GRIDS.keys())
        #print(self.config.grid_definition)
        grid_res = re.search(r'(\d+(?:\.\d+)?)(?=km)', self.config.grid_definition).group() 

        grid_type = self.config.grid_type[1:]

        outfile = f"W_NO-ST-OSLOSAT{self.config.input_data_type}-{grid_type}_C_ESA_{l1c_utc_time}_G_D_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS_T_N_{proj_str}_{grid_res}.nc"

        return outfile 
