"""

"""
# Python STD imports 
import sys 
import os 
import re 
import pathlib as pb  
import time 
import functools 
import xml.etree.ElementTree as ET 
import argparse 

# 3d-party tools 
import numpy as np 
import scipy as sp 
import h5py 
from   colorama import Fore, Style#, Back   
import tqdm 

# Custom modules 
import cimr_grasp.grasp_io as io 
import cimr_grasp.grasp_utils as utils 

# TODO: This should be modified in the future 
# Add the root of your project to PYTHONPATH
rootpath = pb.Path('.').resolve().parents[0]
syspath = str(rootpath)
sys.path.append(syspath) 
#from cimr_grasp.rgb_logging import RGBLogging 
from cimr_rgb.rgb_logging import RGBLogging 




def get_beamdata(beamfile: pb.Path | str, 
                 half_space: str, 
                 file_version: float, 
                 cimr: dict(), 
                 logger) -> dict():  
    """
    Parses GRASP `grd` file and returns data as dictionary object. 

    Parameters:
    -----------
    beamfile: str or Path
        The path to the beamfile to work at.

    half_space: str 
        The current antenna pattern's half space to work with. 

    cimr: dict 
        Beam object to be returned. 

    file_version: float  
        Version of a current file iteration. 

    Returns:
    --------
    cimr: dict 
        Returns an object that contains parsed beam data (amplitude components
        and the grids those are defined on). 

    Raises:
    -------
    NotImplementedError 
        If IGRID value is different than one. 
    """
    
        
    # This part is to be inline with Joe's data format. But we are using it
    # onlyt for tqdm output in the teminal since BHS is more intuitive than BK 
    if half_space == "FR": 
        bn = "FHS"
    elif half_space == "BK": 
        bn = "BHS"

    # Precompile regex pattern to find numbers in each consecutive line  
    reline_pattern = re.compile(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?') 
    
    with open(beamfile, mode = "r", encoding = "UTF-8") as bfile: 

        header = io.get_header(bfile)

        # Retrieving data after ++++ 
        
        # First 3 lines after ++++  
        info = [line.strip("\n") for i, line in enumerate(bfile)] 
        line_shift = 3

        for i in range(0, line_shift):
            line_numbers = reline_pattern.findall(info[i])

            if i == 0 :
                ktype  = int(line_numbers[0])
            elif i == 1:
                nset   = int(line_numbers[0])
                icomp  = int(line_numbers[1])
                ncomp  = int(line_numbers[2])
                igrid  = int(line_numbers[3])
            elif i == 2:
                ix     = int(line_numbers[0])
                iy     = int(line_numbers[1])

        # Raising an error if igrid is not = 1 (see GRASP TICRA manual for the
        # info abut all other values it can have)
        logger.info(f"KTYPE = {ktype}")
        logger.info(f"NSET  = {nset}, ICOMP = {icomp}, NCOMP = {ncomp}, IGRID = {igrid}") 
        if igrid == 1: 
            logger.info(f"Antenna patterns are provided in the (u,v) coordinates and will be converted into (theta,phi)")
        else: 
            raise NotImplementedError(f"The module functionality is implemented only for IGRID value = 1 since CIMR patterns were provided in this format.")


        # The following lines are repeated NSET of times (as per GRASP manual instructions) 
        for i_set in range(nset):
            
            for k in range(line_shift, line_shift+2):
                
                line_numbers = reline_pattern.findall(info[k])

                if k == 3:
                    xs     = float(line_numbers[0])
                    ys     = float(line_numbers[1])
                    xe     = float(line_numbers[2])
                    ye     = float(line_numbers[3])
                elif k == 4:
                    nx     = int(line_numbers[0])
                    ny     = int(line_numbers[1])
                    klimit = int(line_numbers[2])
                    
            line_shift = line_shift + 2 

            logger.info(f"IX = {ix}, IY = {iy}")
            logger.info(f"XS = {xs}, YS = {ys}, XE = {xe}, YE = {ye}")
            logger.info(f"NX = {nx}, NY = {ny}")
            
            # Grid spacing 
            dx   = (xe - xs) / (nx - 1)
            dy   = (ye - ys) / (ny - 1) 
            xcen = dx * ix 
            ycen = dy * iy 

            logger.info(f"DX   = {dx},   DY = {dy}")
            logger.info(f"XCEN = {xcen}, YCEN = {ycen}")

            
            # The output components 
            """
            From SMAP file, page 6, at the bottom of the page: 
            https://nsidc.org/sites/default/files/smap_anc_l1l3l4.pdf    
            
            G1v: V-pol, Co-pol component, real part
            G2v: V-pol, Co-pol component, imagine part
            G3v: V-pol, Cross-pol component, real part
            G4v: V-pol, Cross-pol component, magine part
            
            G1h: H-pol, Co-pol component, real part
            G2h: H-pol, Co-pol component, imagine part
            G3h: H-pol, Cross-pol component, real part
            G4h: H-pol, Cross-pol component, imagine part 
            """
            # From Numpy docs:
            # https://numpy.org/doc/stable/user/basics.types.html 
            # """
            # Note that, above, we could have used the Python float object as a
            # dtype instead of numpy.float64. NumPy knows that int refers to
            # numpy.int_, bool means numpy.bool, that float is numpy.float64
            # and complex is numpy.complex128. The other data-types do not have
            # Python equivalents. 
            # """ 
            G1h = np.full((ny, nx), 0., dtype=float)
            G2h = np.full((ny, nx), 0., dtype=float)
            G3h = np.full((ny, nx), 0., dtype=float)
            G4h = np.full((ny, nx), 0., dtype=float)

            G1v = np.full((ny, nx), 0., dtype=float)
            G2v = np.full((ny, nx), 0., dtype=float)
            G3v = np.full((ny, nx), 0., dtype=float)
            G4v = np.full((ny, nx), 0., dtype=float)
            
            # j_ is J in GRASP manual 
            # 
            # Here j_ is the row number after (nx, ny, klimit) row in a
            # file. So, to get the current row in a grd file, we need to add
            # j_ and line_shift 
        
            for j_ in tqdm.tqdm(range(0, ny), desc=f"| {bn}: Working on chunks (1 chunk = IS rows in a file)", unit=" chunk"): 
                
                line_numbers = reline_pattern.findall(info[j_ + line_shift])
                
                
                # Get the number of columns to read 
                if klimit == 0: 
                    # It is IS in GRASP manual --- the column number of 1st datapoint  
                    is_ = 1  
                    # It is IN in GRASP manual --- the # of datapoints in row J (j_)
                    in_ = nx   
                elif klimit == 1: 
                    is_ = int(line_numbers[0]) 
                    in_ = int(line_numbers[1]) 

                """
                In GRASP manual it is written that this line should be looped
                as: 
                
                F1(I), F2(I), I = IS, IE 

                where IS is is_ in our case and IE = IS + IN - 1.  

                If we now subtract (IE - IS) to start the array from index 0
                instead of IS, then we get IN - 1. Python subtracts one by
                default, so we loop to in_ which is IN in our case. 
                """

                for ict in range(in_):
                #for ict in tqdm.tqdm(range(nxr), desc="NX", leave=False, unit=" col"):

                    line_numbers = reline_pattern.findall(info[j_ + line_shift + (ict + 1)])

                    # If is_ = 1, then ic and ict are exactly the same it seems 
                    # the matlab version starts with index 1, but python starts with 0 
                    ic = is_ + ict - 1   
                    
                    # We were given only horizontal component files, so we copy
                    # those values into vertical as well 
                    G1h[ic, j_] = float(line_numbers[2])
                    G2h[ic, j_] = float(line_numbers[3]) 
                    G3h[ic, j_] = float(line_numbers[0])
                    G4h[ic, j_] = float(line_numbers[1]) 
            
                    G1v[ic, j_] = float(line_numbers[2])
                    G2v[ic, j_] = float(line_numbers[3]) 
                    G3v[ic, j_] = float(line_numbers[0])
                    G4v[ic, j_] = float(line_numbers[1]) 
                    
                
                # To go to the next block of points within the file we need to
                # increase the line counter 
                line_shift = line_shift + in_ 

    u_grid, v_grid = utils.generate_uv_grid(xcen, ycen, xs, ys, nx, ny, dx, dy)

    
    # Getting Vectors Back 
    # 
    # See the note in the generate grid as to why we do it this way
    u = np.unique(u_grid[0, :]) 
    v = np.unique(v_grid[:, 0])  
    
    # Mandatory Parameters 
    cimr["Grid"]['u']    = u   
    cimr["Grid"]['v']    = v    

    cimr["Gain"]['G1h']  = G1h 
    cimr["Gain"]['G2h']  = G2h 
    cimr["Gain"]['G3h']  = G3h 
    cimr["Gain"]['G4h']  = G4h 

    cimr["Gain"]['G1v']  = G1v 
    cimr["Gain"]['G2v']  = G2v 
    cimr["Gain"]['G3v']  = G3v 
    cimr["Gain"]['G4v']  = G4v 

    cimr['Version']      = file_version 

    # Optional Parameters (mainly to restore the grid if needed) 
    cimr["Grid"]['xcen'] = xcen 
    cimr["Grid"]['ycen'] = ycen 

    cimr["Grid"]['xs']   = xs 
    cimr["Grid"]['ys']   = ys 
    
    cimr["Grid"]['nx']   = nx 
    cimr["Grid"]['ny']   = ny 
    
    cimr["Grid"]['dx']   = dx 
    cimr["Grid"]['dy']   = dy 

    return cimr 
    

def recenter_beamdata(cimr: dict(), logger) -> dict(): 
    """
    Method to recenter original beam.  

    The beam center in the (u,v) grid is dictated by xcen and ycen
    variables calculated above. However, as it turned out, the maximum
    beam value is not located in the center, but instead shifted in
    space. Therefore, we need to find where the maximum value is located
    on the u,v grid and offset u,v values to recenter the beam grid on
    beam's maximum value.  

    Parameters:
    -----------
    cimr: dict 
        Dictionary that contains beam data to be modified and returned. 

    Returns:
    --------
    cimr: dict 
        Dictionary that contains beam data to be modified and returned. 
    """

    Ghh_max_index = utils.get_max_index(cimr["temp"]['Ghh']) 
    Ghv_max_index = utils.get_max_index(cimr["temp"]['Ghv']) 
    Gvv_max_index = utils.get_max_index(cimr["temp"]['Gvv']) 
    Gvh_max_index = utils.get_max_index(cimr["temp"]['Gvh']) 

    logger.info(f"Ghh_max_index = {Ghh_max_index}")  
    logger.info(f"Ghv_max_index = {Ghv_max_index}")  
    logger.info(f"Gvv_max_index = {Gvv_max_index}")  
    logger.info(f"Gvh_max_index = {Gvh_max_index}")  
    
    # Get the maximum value
    Ghh_max_value = cimr["temp"]['Ghh'][Ghh_max_index]
    Ghv_max_value = cimr["temp"]['Ghv'][Ghv_max_index]
    Gvv_max_value = cimr["temp"]['Gvv'][Gvv_max_index]
    Gvh_max_value = cimr["temp"]['Gvh'][Gvh_max_index]

    logger.info(f"Ghh_max_value = {Ghh_max_value}")
    logger.info(f"Ghv_max_value = {Ghv_max_value}")
    logger.info(f"Gvv_max_value = {Gvv_max_value}")
    logger.info(f"Gvh_max_value = {Gvh_max_value}")
    
    # Get the coordinates corresponding to maximum gain inside the mesh grids
    # (u, v). This is our new central value.  
    u_coordinate = cimr['Grid']['u_grid'][Ghh_max_index]
    v_coordinate = cimr['Grid']['v_grid'][Ghh_max_index] 
    logger.info(f"u_coordinate = {u_coordinate}")
    logger.info(f"v_coordinate = {v_coordinate}")
    
    # "Shift" is the distance between two coordinates (the center of the beam
    # and the coordinate that corresponds to its maximum gain value). So we
    # just take an absolute difference  
    # 
    # [Note]: Due to floating point precision, we can get crap after 15th
    # point, so I am cutting it off.  
    u_shift = float(format(np.abs(cimr["Grid"]['xcen'] - u_coordinate), '.15f'))       
    v_shift = float(format(np.abs(cimr["Grid"]['ycen'] - v_coordinate), '.15f'))       
    #v_shift = np.abs(ycen - v_coordinate)      
    logger.info(f"u_shift = {u_shift}")
    logger.info(f"v_shift = {v_shift}")
    
    # If the maximum gain coordinate is negative then we add the shift value
    # (go right to reach zero), else --- we subtract (go left).  
    if u_coordinate < 0: 
        cimr['Grid']['u_grid']  = cimr['Grid']['u_grid'] + u_shift 
    else: 
        cimr['Grid']['u_grid']  = cimr['Grid']['u_grid'] - u_shift 
        
    if v_coordinate < 0: 
        cimr['Grid']['v_grid']  = cimr['Grid']['v_grid'] + v_shift 
    else: 
        cimr['Grid']['v_grid']  = cimr['Grid']['v_grid'] - v_shift 
    
    # Converting (u,v) into (theta,phi)
    # 
    # [Note]: We cannot get unique values for theta and phi (write them down as
    # vectors), because the converted grid is not rectilinear anymore.
    # Therefore, we have to interpolate gain values onto rectilinear grid.
    # Otherwise, scipy functionality will be very slow and limited later on,
    # when we are dealing with conversion from satellite reference fram to
    # Earth reference frame (i.e., projection).    
    #theta_grid, phi_grid = utils.convert_uv_to_tp(u_grid, v_grid) 

    # Updating resulting dictionary  
    # (to use these values later on)
    #cimr["Grid"]['u_grid']     = u_grid 
    #cimr["Grid"]['v_grid']     = v_grid 

    # TODO: Put this statement somewhere else because we do not convert the
    # original grid to theta phi, but convert the new theta, into the u, v to
    # enable proper chunking of data later on. 
    #cimr["Grid"]['theta_grid'] = theta_grid 
    #cimr["Grid"]['phi_grid']   = phi_grid 
    
    #cimr["Gain"]['Ghh']        = Ghh
    #cimr["Gain"]['Ghv']        = Ghv
    #cimr["Gain"]['Gvv']        = Gvv
    #cimr["Gain"]['Gvh']        = Gvh

    #logger.info(cimr["Gain"].keys())
    #logger.info(cimr["Grid"].keys())

    return cimr 


# TODO: This method does not work properly if we have a numpy arrays inside
#       nested dictionaries. BUT, it works for the empty dictionaries  
def is_nested_dict_empty(d):
    if not isinstance(d, dict) or not d:  # If it's not a dictionary or the dictionary is empty
        return not d
    return all(is_nested_dict_empty(v) for v in d.values())


def run_cimr_grasp(datadir:  str | pb.Path, 
         outdir:             str | pb.Path, 
         file_version:       str, 
         beamfiles_paths:    list(), 
         grid_res_phi:       float = 0.1, 
         grid_res_theta:     float = 0.1, 
         chunk_data:         bool  = True, 
         num_chunks:         int   = 4, 
         overlap_margin:     float = 0.1, 
         interp_method:      str   = "linear",  
         use_bhs:            bool  = False, 
         recenter_beam:      bool  = True, 
         use_rgb_logging:    bool  = False, 
         use_rgb_decoration: bool  = False, 
         logger = None 
         ) -> None:     
    """
    Main method (entry point to the program). It performs the following steps: 
    
    - Parsing original .grd file and saves into HDF5   
    - Recentering the beam grid to center on the max gain value 
    - Creating (theta, phi) grid with a given resolution and creating its (x,y) respresentation 
    - Interpolating (in chunks) the original (u,v) grid into coarser (x,y) 
    - Saving the resulting data into HDF5 file 

    [**Note**]: The data format is described in the `CIMR_Antenna_Patterns_Format.ipynb` 
    located inside `notebooks` within the repo.   
    
    Parameters:
    -----------
    datadir: str or Path
        The path to the data directory where all beam files are located. 

    outdir: str or Path  
        The path to the output directory where to store all results of execution. 

    file_version: str    
        Version of the parsed files to be produced. 

    recenter_beam: bool 
        Parameter that defines whether to recenter beam or not 

    use_bhs: bool 
        Parameter that defines whether to parse and preprocess BHS files or not. 
    """

    
    # ========================
    # Parsing Antenna Patterns  
    # ========================
    
    # Reconstructing the file names to operate on necessary parts later on 
    apat_name_info = {}
    
    for beamfile in beamfiles_paths:
        
        tobesplit = str(beamfile.stem) 
        band, horn, freq, pol, half_space = io.parse_file_name(tobesplit)

        if band not in apat_name_info:
            apat_name_info[band] = [freq, pol, {}]
        
        if horn not in apat_name_info[band][2]:
            apat_name_info[band][2][horn] = []
        
        apat_name_info[band][2][horn].append(half_space)

    
    logger.info(f"==============================") 
    logger.info(f"Parsing the Antenna Patterns") 
    logger.info(f"==============================") 
    

    # Creating directory to store parsed files  
    parsed_dir = outdir.joinpath("parsed", f"v{file_version}")
    io.rec_create_dir(parsed_dir, logger = logger)

    preprocessed_dir = outdir.joinpath("preprocessed", f"v{file_version}")
    io.rec_create_dir(preprocessed_dir, logger = logger)

    logger.info(f"{Fore.GREEN}Data Directory:{Fore.RESET}\n| {Fore.BLUE}{datadir}{Style.RESET_ALL}") 
    logger.info(f"{Fore.GREEN}Parsed Directory:{Fore.RESET}\n| {Fore.BLUE}{parsed_dir}{Style.RESET_ALL}") 
    logger.info(f"{Fore.GREEN}Preprocessed Directory:{Fore.RESET}\n| {Fore.BLUE}{preprocessed_dir}{Style.RESET_ALL}") 


    # Main parsing loop 
    for band in apat_name_info.keys(): 
        
        for horn, half_spaces in apat_name_info[band][2].items(): 
            
            freq = apat_name_info[band][0] 
            pol  = apat_name_info[band][1] 
            
            for half_space in half_spaces: 

                # Since we do not require BHS, we skip the relevant part of the
                # loop  
                if not use_bhs and half_space == "BK": 
                    logger.info(f"| use_bhs = {use_bhs}; skipping {half_space}.")
                    continue 

                cimr = {"Gain": {}, "Grid": {}} 
                cimr_is_empty = False  

                # Reconstructing the full path to the file 
                if band == "L": 
                    infile = band + "-" + freq + "-" + pol + "-" + half_space + ".grd" 
                else: 
                    infile = band + str(horn) + "-" + freq + "-" + pol + "-" + half_space + ".grd" 
                
                infile = pb.Path(str(datadir) + "/" + band + "/" + infile)  
                logger.info(f"{Fore.YELLOW}------------------------------{Style.RESET_ALL}") 
                logger.info(f"{Fore.GREEN}Working with Input File: {infile.name}{Style.RESET_ALL}") 

                # Parsing Original GRASP Beam Files 
                # Output filename 
                parsedfile_prefix = f"CIMR-OAP-{half_space}" 
                parsedfile_suffix = "UV"
                outfile_oap = pb.Path(str(parsed_dir) + f"/{parsedfile_prefix}-" + band + horn + f"-{parsedfile_suffix}v{file_version}.h5")   
                
                if io.check_outfile_existance(outfile_oap) == True: 
                     cimr_is_empty = is_nested_dict_empty(cimr) 
                else: 
                    
                    logger.info(f"------------------------------") 
                    logger.info(f"Parsing")
                    logger.info(f"------------------------------") 
                    
                    start_time_pars = time.perf_counter() 

                    cimr = get_beamdata(infile, half_space, file_version, cimr, logger)      
                    end_time_pars = time.perf_counter() - start_time_pars 
                    logger.info(f"Finished Parsing in: {end_time_pars:.2f}s") 
                    
                    logger.info(f"{Fore.BLUE}Saving Output File: {outfile_oap.name}{Style.RESET_ALL}") 
                    
                    with h5py.File(outfile_oap, 'w') as hdf5_file:
                        io.save_dict_to_hdf5(hdf5_file, cimr)


                # Performing Beam Recentering and Interpolation on rectilinear grid  
                preprocfile_prefix = f"CIMR-PAP-{half_space}" 
                preprocfile_suffix = "TP"
                outfile_pap = pb.Path(str(preprocessed_dir) + f"/{preprocfile_prefix}-" + band + horn + f"-{preprocfile_suffix}v{file_version}.h5")   
                
                if io.check_outfile_existance(outfile_pap) == True: 
                    continue 
                else: 
                    if cimr_is_empty: 
                        logger.info("Loading data object...") 
                        start_time_pars = time.time() 
                        
                        with h5py.File(outfile_oap, 'r') as hdf5_file:
                            cimr = io.load_hdf5_to_dict(hdf5_file)
                            
                        end_time_pars = time.time() - start_time_pars 
                        logger.info(f"Finished Loading in: {end_time_pars:.2f}s") 


                    # Creating a set of temporary values to be removed from memory 
                    cimr["temp"] = {} 

                    cimr = utils.construct_complete_gains(cimr)  
                    
                    cimr["Grid"]["u_grid"], cimr["Grid"]["v_grid"] = utils.generate_uv_grid(
                                                              xcen = cimr["Grid"]['xcen'], 
                                                              ycen = cimr["Grid"]['ycen'], 
                                                              xs   = cimr["Grid"]['xs'], 
                                                              ys   = cimr["Grid"]['ys'], 
                                                              nx   = cimr["Grid"]['nx'], 
                                                              ny   = cimr["Grid"]['ny'], 
                                                              dx   = cimr["Grid"]['dx'], 
                                                              dy   = cimr["Grid"]['dy'], 
                                                            )  

                    # If no recentering, then just create a temporary grid to
                    # pass those values further and then delete them once we
                    # are done 
                    if recenter_beam: 
                        logger.info(f"------------------------------") 
                        logger.info(f"ReCentering")
                        logger.info(f"------------------------------") 
                        
                        start_time_recen = time.perf_counter() 
                        
                        cimr = recenter_beamdata(cimr, logger)
                        
                        end_time_recen = time.perf_counter() - start_time_recen
                        logger.info(f"Finished Recentering in: {end_time_recen:.2f}s") 

                    
                    logger.info(f"------------------------------") 
                    logger.info(f"Interpolating")
                    logger.info(f"------------------------------") 
                    
                    start_time_interpn = time.perf_counter()  
                    
                    cimr = utils.interp_beamdata_into_uv(cimr = cimr, 
                                                         logger = logger, 
                                                         grid_res_phi   = grid_res_phi, 
                                                         grid_res_theta = grid_res_theta, 
                                                         chunk_data     = chunk_data, 
                                                         num_chunks     = num_chunks, 
                                                         overlap_margin = overlap_margin, 
                                                         interp_method  = interp_method
                                                         ) 

                    end_time_interpn = time.perf_counter() - start_time_interpn
                    logger.info(f"Finished Interpolation in: {end_time_interpn:.2f}s") 

                    logger.info(f"{Fore.BLUE}Saving Output File: {outfile_pap.name}{Style.RESET_ALL}") 
                    with h5py.File(outfile_pap, 'w') as hdf5_file:
                        io.save_dict_to_hdf5(hdf5_file, cimr)
                    

            logger.info(f"| {Fore.YELLOW}------------------------------{Style.RESET_ALL}") 


# TODO: Put this stuff inside of this method into io? 
def resolve_config_path(path_string: pb.Path | str) -> pb.Path:
    if "~" in str(path_string) and not '$' in str(path_string): 
        # Use expanduser to handle '~' for home directory if present
        expanded_path = os.path.expanduser(path_string)
        return pb.Path(expanded_path).resolve() 
    elif "$" in str(path_string) and not '~' in str(path_string): 
        # Use expandvars to replace environment variables like $HOME
        expanded_path = os.path.expandvars(path_string)
        return pb.Path(expanded_path).resolve() 
    elif '$' in str(path_string) and '~' in str(path_string): 
        error_output = ("`outdir` contains both $ and ~ which " + 
                       "will result in incorrect path resolution. " + 
                       f"Check `outdir` variable in {config_file}") 
        raise ValueError(error_output)
    else: 
        # If we have a relative path, we expand it 
        expanded_path = pb.Path(path_string)
        if not expanded_path.is_absolute(): 
            return expanded_path.resolve() 
        else: 
            return expanded_path 


def get_bool(par_val: str, par_name: str) -> bool: 

    par_val = par_val.lower() 

    if par_val == "true": 
        par_val = True
    elif par_val == "false": 
        par_val = False 
    else: 
        raise ValueError(f"Parameter `{par_name}` can either be True or False.")

    return par_val 


def load_grasp_config(config_file = "grasp_config.xml"):

    # Parse the XML file
    tree = ET.parse(config_file)
    root = tree.getroot()

    config = {}

    # Read the paths
    # getting value of datadir, and if doesn't exist, returning an error. 
    config['datadir']   = resolve_config_path(
            path_string = pb.Path(root.find('paths/datadir').text)
            ) 
    if not config['datadir'].is_dir():
        raise FileNotFoundError(f"The directory '{config['datadir']}' does not exist.")
    if not any(config['datadir'].iterdir()):
        raise FileNotFoundError(f"The directory '{config['datadir']}' is empty.")


    # getting value of outdir, checking for errors and creating directories
    # recursively if they do not exist 
    config['outdir']    = resolve_config_path(
            path_string = root.find('paths/outdir').text
            ) 
    io.rec_create_dir(config['outdir'])   

    # Read the parameters
    parameters = root.find('parameters')


    # Bool params 
    config['use_bhs']        = get_bool(
                   par_name  = 'use_bhs', 
                   par_val   = parameters.find('use_bhs').text
                   ) 
    config['recenter_beam']  = get_bool(
                   par_name  = 'recenter_beam', 
                   par_val   = parameters.find('recenter_beam').text
                   ) 
    config['chunk_data']     = get_bool(
                   par_name  = 'chunk_data', 
                   par_val   = parameters.find('chunk_data').text
                   ) 
    # Float params 
    config['grid_res_phi']   = float(parameters.find('grid_res_phi').text)
    config['grid_res_theta'] = float(parameters.find('grid_res_theta').text)
    config['overlap_margin'] = float(parameters.find('overlap_margin').text)

    # Int params 
    config['num_chunks']     = int(parameters.find('num_chunks').text)

    # Str params 
    config['interp_method']  = parameters.find('interp_method').text
    config['file_version']   = parameters.find('file_version').text


    # Read the logging settings
    logging = root.find('logging')
    config['use_rgb_logging']    = get_bool(
                        par_name = 'use_rgb_logging', 
                        par_val  = logging.find('use_rgb_logging').text
                        )  
    config['use_rgb_decoration'] = get_bool(
                        par_name = 'use_rgb_logging', 
                        par_val  = logging.find('use_rgb_decoration').text
                        )  
    config['logger_config']      = resolve_config_path(
            path_string = root.find('logging/logger_config').text
            ) 

    return config


def main():

    start_time_tot = time.perf_counter() 
    # -----------------------------
    # Default value for config file 
    config_file = pb.Path("configs", "grasp_config.xml").resolve()

    # Getting the value for parameter file from cmd 
    parser = argparse.ArgumentParser(description = "Update XML configuration parameters.")
    # Will use the default value of config_file if none is provided via command line: 
    # https://docs.python.org/3/library/argparse.html#nargs 
    parser.add_argument('config_file', type = str, help = "Path to the XML parameter file.", 
                    nargs = "?", default = config_file)

    args = parser.parse_args() 
    config_file = resolve_config_path(args.config_file) 

    # -----------------------------

    # Params from parameter file 
    config             = load_grasp_config(config_file = config_file)

    outdir             = config["outdir"]  
    datadir            = config["datadir"]  
    use_bhs            = config["use_bhs"]  
    recenter_beam      = config["recenter_beam"] 
    grid_res_phi       = config["grid_res_phi"] 
    grid_res_theta     = config["grid_res_theta"]  
    chunk_data         = config["chunk_data"] 
    num_chunks         = config["num_chunks"]  
    overlap_margin     = config["overlap_margin"] 
    interp_method      = config["interp_method"]   
    file_version       = config["file_version"]  
    # Logging functionality 
    use_rgb_logging    = config["use_rgb_logging"]  
    use_rgb_decoration = config["use_rgb_decoration"] 
    logger_config      = config["logger_config"]

    
    logdir = config['outdir'].joinpath("logs")  
    io.rec_create_dir(logdir)   

    # Creating a logger object based on the user preference 
    if use_rgb_logging and logger_config is not None: 
        rgb_logging    = RGBLogging(logdir = logdir, log_config = logger_config)
        rgb_logger     = rgb_logging.get_logger("rgb") 

    # -----------------------------

    rgb_logger.debug(f"Starting the script using the following libraries:")
    rgb_logger.debug(f"numpy      {np.__version__}" ) 
    rgb_logger.debug(f"scipy      {sp.__version__}" ) 
    rgb_logger.debug(f"h5py       {h5py.__version__}") 
    rgb_logger.debug(f"tqdm       {tqdm.__version__}") 
    #rgb_logger.debug(f"matplotlib {matplotlib.__version__}") 

    # Parameters with which CIMR GRASP is to be run 
    rgb_logger.info("---------")

    rgb_logger.info(f"CIMR GRASP Configuration")

    rgb_logger.info("---------")

    rgb_logger.info(f"Output Directory:        {outdir}") 
    rgb_logger.info(f"Data Directory:          {datadir}")   
    rgb_logger.info(f"Use BHS:                 {use_bhs}") 
    rgb_logger.info(f"Recenter Beam:           {recenter_beam}") 
    rgb_logger.info(f"Grid Resolution (Phi):   {grid_res_phi}") 
    rgb_logger.info(f"Grid Resolution (Theta): {grid_res_theta}")       
    rgb_logger.info(f"Chunk Data:              {chunk_data}") 
    rgb_logger.info(f"Number of Chunks:        {num_chunks}") 
    rgb_logger.info(f"Overlap Margin:          {overlap_margin}") 
    rgb_logger.info(f"Interpolation Method:    {interp_method}")
    rgb_logger.info(f"File Version:            {file_version}")

    rgb_logger.info(f"Use CIMR RGB Logger :    {use_rgb_logging}")
    rgb_logger.info(f"Use CIMR RGB Decoration: {use_rgb_decoration}")
    rgb_logger.info(f"Logger Config:           {logger_config}")

    rgb_logger.info("---------")

    #exit() 

    #if not pb.Path(outdir).exists(): 
    #    rgb_logger.info(f"Creating output directory:\n{outdir}")
    #    pb.Path(outdir).mkdir()

    #if not datadir.is_dir():
    #    raise FileNotFoundError(f"The directory '{datadir}' does not exist.")

    # Getting all beam paths inside dpr/AP 
    beamfiles_paths = datadir.glob("*/*")   
    #for beamfile in beamfiles_paths:
    #    print(beamfile)
    #exit() 


    run_cimr_grasp(datadir  = datadir, 
         outdir             = outdir, 
         file_version       = file_version,
         beamfiles_paths    = beamfiles_paths, 
         use_bhs            = use_bhs, 
         recenter_beam      = recenter_beam, 
         grid_res_phi       = grid_res_phi, 
         grid_res_theta     = grid_res_theta, 
         chunk_data         = chunk_data, 
         num_chunks         = num_chunks, 
         overlap_margin     = overlap_margin, 
         interp_method      = interp_method,   
         use_rgb_logging    = use_rgb_logging, 
         use_rgb_decoration = use_rgb_decoration, 
         logger             = rgb_logger 
         )    
    end_time_tot = time.perf_counter() - start_time_tot
    rgb_logger.info(f"Finished Script in: {end_time_tot:.2f}s") 
    rgb_logger.info(f"------------------------------") 




if __name__ == '__main__': 

    main() 

    
    
    
    
    
