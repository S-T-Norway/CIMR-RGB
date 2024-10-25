import re 
import pickle 
import pathlib as pb 


import numpy   as np 
import netCDF4 as nc 




class ProductGenerator: 

    def __init__(self, config):
        self.config = config 
        self.logger = config.logger  


    def generate_l1c_product(self, data_dict: dict()): 

        # TODO: Change the lists into dictionaries and add metadata (as well as
        # proper dimensions to variables, since right now it is only x, y but
        # can also be n_l1b_scans) 
        # 
        # TODO: Double check all metadata below 
        # 
        # Params from CDL 
        params_to_save = {
                "Measurement": {
                    "bt_h_fore": { 
                        "units": "K",
                        "long_name": "H-polarised TOA Brightness Temperatures",
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": f"Earth-Gridded TOA h-polarised" + \
                         f" [L|C|X|KU|KA]_BAND_fore BTS" + \
                         f" interpolated on a TBD-km grid"
                         }, 
                    "bt_h_aft": { 
                        "units": "K",
                        "long_name": "H-polarised TOA Brightness Temperatures",
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": f"Earth-Gridded TOA h-polarised" + \
                         f" [L|C|X|KU|KA]_BAND_aft BTS" + \
                         f" interpolated on a TBD-km grid"
                         },  
                    "bt_v_fore": {
                        "units": "K",
                        "long_name": "V-polarised TOA Brightness Temperatures",
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": f"Earth-Gridded TOA v-polarised" + \
                         f" [L|C|X|KU|KA]_BAND_fore BTS" + \
                         f" interpolated on a TBD-km grid"
                        }, 
                    "bt_v_aft":  {
                        "units": "K",
                        "long_name": "V-polarised TOA Brightness Temperatures",
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": f"Earth-Gridded TOA v-polarised" + \
                         f" [L|C|X|KU|KA]_BAND_aft BTS" + \
                         f" interpolated on a TBD-km grid"
                        }, 
                    "bt_3_fore": {
                        "units": "K",
                        "long_name": "Stokes 3-polarised TOA Brightness Temperatures", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Earth-Gridded TOA [L|C|X|KU|KA]_BAND_fore BTS " + \
                         "interpolated on a TBD-km grid, third stokes parameter " + \
                         "of the surface polarisation basis"
                        }, 
                    "bt_3_aft":  {
                        "units": "K",
                        "long_name": "H-polarised TOA Brightness Temperatures",
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Earth-Gridded TOA [L|C|X|KU|KA]_BAND_aft BTS " + \
                         "interpolated on a TBD-km grid, third stokes parameter " + \
                         "of the surface polarisation basis"
                        }, 
                    "bt_4_fore": {
                        "units": "K",
                        "long_name": "Stokes 4-polarised TOA Brightness Temperatures", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Earth-Gridded TOA [L|C|X|KU|KA]_BAND_fore BTS " + \
                          "interpolated on a TBD-km grid, fourth stokes parameter " + \
                          "of the surface polarisation basis"
                        }, 
                    "bt_4_aft":  {
                        "units": "K",
                        "long_name": "Stokes 4-polarised TOA Brightness Temperatures", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Earth-Gridded TOA [L|C|X|KU|KA]_BAND_aft BTS " + \
                          "interpolated on a TBD-km grid, fourth stokes parameter " + \
                          "of the surface polarisation basis"
                        }, 
                    "faraday_rot_angle_fore":   {
                        "units": "deg",
                        "long_name": "Interpolated Faraday Rotation Angle of acquisitions", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                         "comment": "Level 1b [L|C|X|KU|KA]_BAND_fore faraday " + \
                           "rotation angle corresponding to the measured BT value. " + \
                           "The value of the faraday rotation angle will be scaled " + \
                           "with the interpolation weights of all faraday rotation " + \
                           "angles Earth samples used in the interpolation of that " + \
                           "grid cell."
                        }, 
                    "faraday_rot_angle_aft":    {
                        "units": "deg",
                        "long_name": "Interpolated Faraday Rotation Angle of acquisitions", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                         "comment": "Level 1b [L|C|X|KU|KA]_BAND_aft faraday " + \
                           "rotation angle corresponding to the measured BT value. " + \
                           "The value of the faraday rotation angle will be scaled " + \
                           "with the interpolation weights of all faraday rotation " + \
                           "angles Earth samples used in the interpolation of that " + \
                           "grid cell."
                        }, 
                    "geometric_rot_angle_fore": {
                        "units": "deg",
                        "long_name": "Interpolated Geometric Rotation angle of acquisitions.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                         "comment": "Level 1b [L|C|X|KU|KA]_BAND_fore geometric " + \
                         "rotation angle corresponding to the measured BT value. " + \
                         "The value of the geometric rotation angle will be " + \
                         "scaled with the interpolation weights of all geometric " + \
                         "rotation angles Earth samples used in the " + \
                         "interpolation of that grid cell." 
                        }, 
                    "geometric_rot_angle_aft":  {
                        "units": "deg",
                        "long_name": "Interpolated Geometric Rotation angle of acquisitions.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_ChunkSizes": [256, 256],  # List instead of a string for chunk sizes
                        "_FillValue": nc.default_fillvals['f8'], 
                         "comment": "Level 1b [L|C|X|KU|KA]_BAND_aft geometric " + \
                         "rotation angle corresponding to the measured BT value. " + \
                         "The value of the geometric rotation angle will be " + \
                         "scaled with the interpolation weights of all geometric " + \
                         "rotation angles Earth samples used in the " + \
                         "interpolation of that grid cell." 
                        }, 
                    "ndet_fore": {
                        "units": "K",
                        "long_name": "Radiometric resolution of each measured BT.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": f"Radiometric resolution of each measured BT." 
                        }, 
                    "ndet_aft":  {
                        "units": "K",
                        "long_name": "Radiometric resolution of each measured BT.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": f"Radiometric resolution of each measured BT." 
                        }, 
                    "tsu_fore":  {
                        "units": "K",
                        "long_name": "Total standard uncertainty for each measured BT.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Total standard uncertainty for each measured BT." 
                        }, 
                    "tsu_aft":   {
                        "units": "K",
                        "long_name": "Total standard uncertainty for each measured BT.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Total standard uncertainty for each measured BT." 
                        }, 
                    "instrument_status_fore": {
                        "units": "N/A",
                        "long_name": "Instrument Calibration or Observation mode.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Instrument Calibration or Observation mode, " + \
                        "for all samples. L1c values will consider the majority " + \
                        "status values from input L1b samples."
                        }, 
                    "instrument_status_aft":  {
                        "units": "N/A",
                        "long_name": "Instrument Calibration or Observation mode.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Instrument Calibration or Observation mode, " + \
                        "for all samples. L1c values will consider the majority " + \
                        "status values from input L1b samples."
                        }, 
                    "land_sea_content_fore":  {
                        "units": "N/A",
                        "long_name": "Land/Sea content of the measured pixel.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Land/Sea content of the measured pixel, " + \
                        "200 for full sea content, 0 for full land content."
                        }, 
                    "land_sea_content_aft":   {
                        "units": "N/A",
                        "long_name": "Land/Sea content of the measured pixel.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Land/Sea content of the measured pixel, " + \
                        "200 for full sea content, 0 for full land content."
                        }, 
                    "regridding_n_samples_fore": {
                        "units": "N/A",
                        "long_name": "Number of earth samples used for interpolation", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                         "comment": "Number of L1b [h|v|t3|t4] polarised " + \
                         "[L|C|X|KU|KA]_BAND [fore|aft] brightness temperature " + \
                         "Earth samples used in the [Backus-Gilbert|rSIR|LW] " + \
                         "remapping interpolation."
                        }, 
                    "regridding_n_samples_aft":  {
                        "units": "N/A",
                        "long_name": "Number of earth samples used for interpolation", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                         "comment": "Number of L1b [h|v|t3|t4] polarised " + \
                         "[L|C|X|KU|KA]_BAND [fore|aft] brightness temperature " + \
                         "Earth samples used in the [Backus-Gilbert|rSIR|LW] " + \
                         "remapping interpolation."
                        }, 
                    "regridding_quality_measure_fore": {
                        "units": "N/A",
                        "long_name": "Algorithm Specific Optimal Value of Regularization Parameter.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "The optimal value of a parameter for the " + \
                         "[Backus-Gilbert|rSIR|LW] [fore|aft] that controls the " + \
                         "trade-off between noise amplification and " + \
                         "regularisation. For BG it is the optimal value for the " + \
                         "smoothing parameter, while for [rSIR|LW] it is the " + \
                         "number of iterations to achieve a chosen level of " + \
                         "residual error. In case of [NN|IDS|DIB] regularisation " + \
                         "is not performed and the parameter will take on the " + \
                         "_FillValue."  
                         }, 
                    "regridding_quality_measure_aft":  {
                        "units": "N/A",
                        "long_name": "Algorithm Specific Optimal Value of Regularization Parameter.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "The optimal value of a parameter for the " + \
                         "[Backus-Gilbert|rSIR|LW] [fore|aft] that controls the " + \
                         "trade-off between noise amplification and " + \
                         "regularisation. For BG it is the optimal value for the " + \
                         "smoothing parameter, while for [rSIR|LW] it is the " + \
                         "number of iterations to achieve a chosen level of " + \
                         "residual error. In case of [NN|IDS|DIB] regularisation " + \
                         "is not performed and the parameter will take on the " + \
                         "_FillValue."  
                        }, 
                    "regridding_l1b_orphans_fore": {
                        "units": "N/A",
                        "long_name": "Indication of L1b orphaned Earth samples.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Whether each [L|C|X|KU|KA]_BAND L1b measurement sample was " + \
                        "unused (1) or used (0) in [Backus-Gilbert|rSIR|LW] regridding " + \
                        "interpolation of [fore|aft] scan samples. In the fore-scan " + \
                        "regridding nearly all aft scan samples would be orphan " + \
                        "(unused), for instance, and vice versa. It would also occur if " + \
                        "the swath stretches outside the projection window. Orphaned " + \
                        "samples may also occur if nearest neighbour or linear " + \
                        "interpolation (among the TBD methods) is used."
                        }, 
                    "regridding_l1b_orphans_aft":  {
                        "units": "N/A",
                        "long_name": "Indication of L1b orphaned Earth samples.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "TBD",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
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
                "Navigation":  {
                    "acq_time_utc_fore": {
                        "units": "N/A",
                        "long_name": "Interpolated UTC Acquisition time of Earth Sample acquisitions.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,N/A",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "UTC acquisition times expressed in seconds " +\
                        "(seconds since 2000-01-01 00:00:00 UTC). The value of " + \
                        "time_earth will be scaled with the interpolation " + \
                        "weights of all time_earth Earth samples used in the " + \
                        "interpolation of that grid cell. "
                        }, 
                    "acq_time_utc_aft": {
                        "units": "N/A",
                        "long_name": "Interpolated UTC Acquisition time of Earth Sample acquisitions.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,N/A",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "UTC acquisition times expressed in seconds " +\
                        "(seconds since 2000-01-01 00:00:00 UTC). The value of " + \
                        "time_earth will be scaled with the interpolation " + \
                        "weights of all time_earth Earth samples used in the " + \
                        "interpolation of that grid cell. "
                        }, 
                    "azimuth_fore": {
                        "units": "deg",
                        "long_name": "Interpolated Earth Azimuth angle of the acquisitions.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                         "comment": "Level 1b [L|C|X|KU|KA]_BAND_fore " + \
                         "Earth observation azimuth angles of the acquisitions, " + \
                         "positive counterclockwise from due east. The value of " + \
                         "observation azimuth angle will be scaled with the " + \
                         "interpolation weights of all observation azimuth angle " + \
                         "Earth samples used in the interpolation of that grid " \
                         "cell." 
                        }, 
                    "azimuth_aft": {
                        "units": "deg",
                        "long_name": "Interpolated Earth Azimuth angle of the acquisitions.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                         "comment": "Level 1b [L|C|X|KU|KA]_BAND_aft " + \
                         "Earth observation azimuth angles of the acquisitions, " + \
                         "positive counterclockwise from due east. The value of " + \
                         "observation azimuth angle will be scaled with the " + \
                         "interpolation weights of all observation azimuth angle " + \
                         "Earth samples used in the interpolation of that grid " \
                         "cell." 
                        }, 
                    "latitude_fore": {
                        "units": "deg",
                        "long_name": "Latitude of the centre of a TBD-km PROJ grid cell.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "-90, 90",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256", 
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Latitude of the centre of a TBD-km PROJ grid cell."
                        }, 
                    "latitude_aft": {
                        "units": "deg",
                        "long_name": "Latitude of the centre of a TBD-km PROJ grid cell.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "-90, 90",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256", 
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Latitude of the centre of a TBD-km PROJ grid cell."
                        }, 
                    "longitude_fore": {
                        "units": "deg",
                        "long_name": "Longitude of the centre of a TBD-km PROJ grid cell.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "-180, 179.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256", 
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Longitude of the centre of a TBD-km PROJ grid cell."
                        }, 
                    "longitude_aft": {
                        "units": "deg",
                        "long_name": "Longitude of the centre of a TBD-km PROJ grid cell.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "-180, 179.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256", 
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Longitude of the centre of a TBD-km PROJ grid cell."
                        }, 
                    "oza_fore": {
                        "units": "deg",
                        "long_name": "Interpolated Observation Zenith Angle of acquisitions.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Level 1b [L|C|X|KU|KA]_BAND_fore " + \
                        "Earth Observation zenith angles of the acquisitions. " + \
                        "The value of OZA will be scaled with the interpolation " + \
                        "weights of all observation OZA Earth samples used in " + \
                        "the interpolation of that grid cell. The OZA is defined " + \
                        "as the included angle between the antenna Boresight " + \
                        "vector and the normal to the Earth's surface."
                        }, 
                    "oza_aft": {
                        "units": "deg",
                        "long_name": "Interpolated Observation Zenith Angle of acquisitions.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Level 1b [L|C|X|KU|KA]_BAND_aft " + \
                        "Earth Observation zenith angles of the acquisitions. " + \
                        "The value of OZA will be scaled with the interpolation " + \
                        "weights of all observation OZA Earth samples used in " + \
                        "the interpolation of that grid cell. The OZA is defined " + \
                        "as the included angle between the antenna Boresight " + \
                        "vector and the normal to the Earth's surface."
                        }, 
                    "processing_scan_angle_fore": {
                        "units": "deg",
                        "long_name": "Interpolated scan angle of acquisitions", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",   
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "The processing scan angle of the L1b " + \
                        "[L|C|X|KU|KA]_BAND_fore Earth view samples. The " + \
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
                    "processing_scan_angle_aft": {
                        "units": "deg",
                        "long_name": "Interpolated scan angle of acquisitions", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",   
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "The processing scan angle of the L1b " + \
                        "[L|C|X|KU|KA]_BAND_aft Earth view samples. The " + \
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
                    "solar_azimuth_fore": {
                        "units": "deg",
                        "long_name": "Interpolated solar azimuth angle of acquisitions", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",   
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Level 1b [L|C|X|KU|KA]_BAND_fore " + \
                        "solar azimuth angle of acquisitions.The value of " + \
                        "solar_azimuth will be scaled with the interpolation " + \
                        "weights of all solar_azimuth Earth samples used in the " + \
                        "interpolation of that grid cell." 
                        }, 
                    "solar_azimuth_aft": {
                        "units": "deg",
                        "long_name": "Interpolated solar azimuth angle of acquisitions", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",   
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Level 1b [L|C|X|KU|KA]_BAND_aft " + \
                        "solar azimuth angle of acquisitions.The value of " + \
                        "solar_azimuth will be scaled with the interpolation " + \
                        "weights of all solar_azimuth Earth samples used in the " + \
                        "interpolation of that grid cell." 
                        }, 
                    "solar_zenith_fore": {
                        "units": "deg",
                        "long_name": "Interpolated solar zenith angle of acquisitions", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Level 1b [L|C|X|KU|KA]_BAND_fore " + \
                        "solar zenith angle of acquisitions.The value of " + \
                        "solar_zenith will be scaled with the interpolation " + \
                        "weights of all solar_zenith Earth samples used in the " + \
                        "interpolation of that grid cell."
                        }, 
                    "solar_zenith_aft": {
                        "units": "deg",
                        "long_name": "Interpolated solar zenith angle of acquisitions", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,359.99",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": nc.default_fillvals['f8'], 
                        "comment": "Level 1b [L|C|X|KU|KA]_BAND_aft " + \
                        "solar zenith angle of acquisitions.The value of " + \
                        "solar_zenith will be scaled with the interpolation " + \
                        "weights of all solar_zenith Earth samples used in the " + \
                        "interpolation of that grid cell."
                        } 
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
                        "_FillValue": "0",#nc.default_fillvals['f8'], 
                        "comment": "A TBD-bit binary string of 1’s and 0’s " + \
                        "indicating a variety of TBD information related to the " + \
                        "processing of L1c/r data."
                        }
                    },  
                "Quality_information": {
                    "navigation_status_flag": {
                        "units": "N/A",
                        "long_name": "Quality information flag summarising the " + \
                            "navigation quality of each scan.",  
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,65535",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": "0", 
                        "comment": "A TBD-bit binary string of 1’s and 0's " +\
                        "indicating the quality of the L1b acquisition " + \
                        "conditions. A ‘0’ indicates that the L1c samples met a " + \
                        "certain quality criterion and a ‘1’ that it did not. " + \
                        "Bit position ‘0’ refers to the least significant bit. " + \
                        "navigation_status_flag summarises the navigation " + \
                        "quality Of each scan."  
                        }, 
                    "scan_quality_flag": {
                        "units": "N/A",
                        "long_name": "Quality information flag summarising the " + \
                            "overall scan quality.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,65535",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": "0",#nc.default_fillvals['f8'], 
                        "comment": "A TBD-bit binary string of 1’s and 0’s " + \
                        "indicating the quality of the L1b acquisition " + \
                        "conditions. A ‘0’ indicates that the L1c samples met a " + \
                        "certain quality criterion and a ‘1’ that it did not. " + \
                        "Bit position ‘0’ refers to the least significant bit. " + \
                        "scan_quality_flag summarises the scan quality" 
                        }, 
                    "temperatures_flag": {
                        "units": "N/A",
                        "long_name": "Quality information indicating degraded " + \
                            "instrument temperature cases.", 
                        "grid_mapping": "crs",
                        "coverage_content_type": "Grid",
                        "valid_range": "0,65535",
                        "_Storage": "chunked",
                        "_ChunkSizes": "256, 256",  
                        "_FillValue": "0",#nc.default_fillvals['f8'], 
                        "comment": "A TBD-bit binary string of 1’s and 0’s " + \
                        "indicating the quality of the L1b acquisition " + \
                        "conditions. A ‘0’ indicates that the L1c samples met a " + \
                        "certain quality criterion and a ‘1’ that it did not. " + \
                        "Bit position ‘0’ refers to the least significant bit. " + \
                        "temperatures_flag to indicates degraded instrument " + \
                        "temperature cases." 
                        }
                    } 
                }

        # TODO: Remove pickled object and pass in proper dictionary to be saved 
        #file_path = pb.Path("dpr/data_dict_out.pkl")
        ## Open the file in read-binary mode and load the object
        #with open(file_path, 'rb') as file:
        #    loaded_object = pickle.load(file)

        # <Projection> -> Data -> Measurement -> <Band>
        outfile = pb.Path(f"{self.config.output_path}/test_l1c.nc").resolve()
        with nc.Dataset(outfile, "w", format = "NETCDF4") as dataset: 

            # Creating Dimentions according to cdl 
            dataset.createDimension('time', None)
            dataset.createDimension('x', None)
            dataset.createDimension('y', None)

            # Creating nested groups according to cdl 
            projection_group  = dataset.createGroup(f"{self.config.projection_definition}") 
            data_group        = projection_group.createGroup("Data")

            # Loop through the parameters defined inside CDL and compare their
            # names to the ones provided inside pickled file. If they coincide
            # we write them into specific group (defined in CDL). In addition,
            # CDL values for CIMR have dimensions (time, x, y) while SMAP has
            # only 1, so we also programmatically figure out the dimensonf of
            # the numpy array provided and save the data accordingly. 
            for group_field, group_vals in params_to_save.items(): 

                group = data_group.createGroup(group_field)

                for band_name, band_var in data_dict.items(): 

                    band_group = group.createGroup(f"{band_name}_BAND")

                    for var_name, var_val in band_var.items(): 

                        var_shape = var_val.shape

                        if var_name in group_vals: 

                            self.logger.info(f"{group_field}, {band_name}, {var_name}")

                            #print(group_vals[var_name]["_FillValue"])

                            if len(var_shape) == 1: 
                                var_data = band_group.createVariable(
                                        var_name, 
                                        "double", 
                                        ('x'), 
                                        fill_value = group_vals[var_name]["_FillValue"]
                                        ) 
                                var_data[:] = var_val 
                            elif len(var_shape) == 2:  
                                var_data = band_group.createVariable(
                                        var_name, 
                                        "double", ('x', 'y'), 
                                        fill_value = group_vals[var_name]["_FillValue"]
                                        ) 
                                var_data[:, :] = var_val 
                            elif len(var_shape) == 3: 
                                var_data = band_group.createVariable(
                                        var_name, 
                                        "double", 
                                        ('time', 'x', 'y'), 
                                        fill_value = group_vals[var_name]["_FillValue"]
                                        ) 
                                var_data[:, :, :] = var_val 
                            else:
                                # Return a generic message or handle error for unknown shapes
                                raise ValueError(f"Unsupported shape with {len(var_shape)} dimensions: {var_shape}")

                            # TODO: fix the re pattern to also include fore|aft
                            # when this loop will be appropriate

                            # Loop through the dictionary and set attributes for the variable
                            for attr_name, attr_value in group_vals[var_name].items():
                                if attr_name != "_FillValue" and attr_name != "comment": 
                                    #print(attr_name)
                                    # Use setncattr to assign the attribute
                                    var_data.setncattr(attr_name, attr_value)
                                elif attr_name == "comment": 
                                    pattern = r"\[L\|C\|X\|KU\|KA\]_BAND_" #\[fore\|aft\]" 
                                    substitution = f"{band_name}_BAND_" 
                                    attr_value = re.sub(pattern, substitution, attr_value)
                                    var_data.setncattr(attr_name, attr_value)



    # TODO: Perhaps this one is obsolete? 
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
            - ('x', 'y') for 2D shapes (e.g., (10000, 10000))
            - ('time', 'x', 'y') for 3D shapes (e.g., (1, 111, 111))
        
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
            return ('x', 'y')
        elif len(var_shape) == 3:
            # 3D case
            return ('time', 'x', 'y')
        else:
            # Return a generic message or handle error for unknown shapes
            raise ValueError(f"Unsupported shape with {len(var_shape)} dimensions: {var_shape}")
