"""

"""
# Python STD imports 
import re 
import pathlib as pb  
#import glob 
#import numbers 
import time 
import functools 
import psutil 

# 3d-party tools 
import numpy as np 
import scipy as sp 
import h5py 
#import xarray as xr 
#import dask 
#import dask.array as da 
#from dask.diagnostics import ProgressBar
#from dask.distributed import Client, LocalCluster 
import matplotlib
#import matplotlib.pyplot as plt 
from   colorama import Fore, Style#, Back   
import tqdm 

# Custom modules 
import grasp_io as io 
import grasp_utils as utils 
from rgb_logging import RGBLogging 


# TODO: Think of different name? 
#def rgb_profiled(time_it = False, profile_it = False, log = False):
#
#    def decorator(func):
#
#        def wrapper(*args, **kwargs):
#
#            message = ""
#
#            if time_it: 
#                print("Starting Timer") 
#                start_time   = time.perf_counter()
#
#            if profile_it: 
#                print("Starting Profiler")
#                start_cpu    = psutil.cpu_percent(interval=None)
#                start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
#
#
#             
#            result = func(*args, **kwargs)
#
#            if time_it: 
#                end_time     = time.perf_counter()
#                time_taken   = end_time - start_time
#
#                message = f"Function '{func.__name__}' executed in {time_taken:.2f}s. "
#
#            if profile_it: 
#                end_cpu = psutil.cpu_percent(interval=None)
#                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
#                cpu_usage = end_cpu - start_cpu
#                memory_usage = end_memory - start_memory
#
#                message = message + "\n" + (
#                           f"CPU usage: {cpu_usage:.2f}%, "
#                           f"Memory usage change: {memory_usage:.2f} MB.")
#            
#            print(message)
#
#            return result
#
#        return wrapper 
#
#    return decorator  
#
#@rgb_profiled(time_it = True, profile_it = True) 
#def add(a = 2, b = 1):
#    return a + b 
#
#add() 
#exit() 


#def conditional_decorator(condition, decorator):
#    """
#    A factory that returns a conditional decorator.
#    If `condition` is True, applies `decorator`. Otherwise, returns the original function.
#    
#    Parameters:
#    - condition (bool): The condition to check.
#    - decorator (function): The decorator to apply if the condition is True.
#    """
#    def wrapper(func):
#        # If the condition is true, apply the decorator
#        if condition:
#            return decorator(func)
#        # Otherwise, return the original function
#        return func
#    return wrapper
#
## Timing decorator
#def timeit(func):
#
#    @functools.wraps(func)
#    def wrapper(*args, **kwargs):
#
#        start_time = time.perf_counter()
#
#        result = func(*args, **kwargs)
#
#        end_time = time.perf_counter()
#        time_taken = end_time - start_time
#
#        print(f"Function '{func.__name__}' executed in {time_taken:.2f} seconds.")
#
#        return result
#
#    return wrapper
#
#def time_and_track(log=True, to_print=True):
#    """
#    Decorator for timing and resource usage (CPU, Memory).
#    """
#    def decorator(func):
#        def wrapper(*args, **kwargs):
#            # Record the start time and resource usage
#            start_time = time.perf_counter()
#            start_cpu = psutil.cpu_percent(interval=None)
#            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
#            
#            # Execute the wrapped function
#            result = func(*args, **kwargs)
#            
#            # Record the end time and resource usage
#            end_time = time.perf_counter()
#            end_cpu = psutil.cpu_percent(interval=None)
#            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
#            
#            # Calculate time taken, CPU, and memory usage
#            time_taken = end_time - start_time
#            cpu_usage = end_cpu - start_cpu
#            memory_usage = end_memory - start_memory
#            
#            message = (f"Function '{func.__name__}' executed in {time_taken:.4f} seconds, "
#                       f"CPU usage: {cpu_usage:.2f}%, "
#                       f"Memory usage change: {memory_usage:.2f} MB.")
#
#            # Log and/or print the message
#            if log:
#                self.logger.info(message)
#            if to_print:
#                print(message)
#
#            return result
#        return wrapper
#    return decorator



#@conditional_decorator(condition=True, decorator=timeit) 
#@timeit
#@rgb_decorator 
def get_beamdata(beamfile: pb.Path, half_space: str, file_version: float, cimr: dict()) -> dict():  
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
        print(f"| KTYPE = {ktype}")
        print(f"| NSET  = {nset}, ICOMP = {icomp}, NCOMP = {ncomp}, IGRID = {igrid}") 
        if igrid == 1: 
            print(f"| Antenna patterns are provided in the (u,v) coordinates and will be converted into (theta,phi)")
        else: 
            raise NotImplementedError(f"| The module functionality is implemented only for IGRID value = 1 since CIMR patterns were provided in this format.")


        # The following lines are repeated NSET of times (as per GRASP manual instructions) 
        for i_set in range(nset):
            
            for k in range(line_shift, line_shift+2):
                
                #line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', info[k])
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

            print(f"| IX = {ix}, IY = {iy}")
            print(f"| XS = {xs}, YS = {ys}, XE = {xe}, YE = {ye}")
            print(f"| NX = {nx}, NY = {ny}")
            
            # Grid spacing 
            dx   = (xe - xs) / (nx - 1)
            dy   = (ye - ys) / (ny - 1) 
            xcen = dx * ix 
            ycen = dy * iy 

            print(f"| DX   = {dx},   DY = {dy}")
            print(f"| XCEN = {xcen}, YCEN = {ycen}")

            
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
            G1h = np.full((ny, nx), np.nan, dtype=float)
            G2h = np.full((ny, nx), np.nan, dtype=float)
            G3h = np.full((ny, nx), np.nan, dtype=float)
            G4h = np.full((ny, nx), np.nan, dtype=float)

            G1v = np.full((ny, nx), np.nan, dtype=float)
            G2v = np.full((ny, nx), np.nan, dtype=float)
            G3v = np.full((ny, nx), np.nan, dtype=float)
            G4v = np.full((ny, nx), np.nan, dtype=float)
            
            # j_ is J in GRASP manual 
            # 
            # Here j_ is the row number after (nx, ny, klimit) row in a
            # file. So, to get the current row in a grd file, we need to add
            # j_ and line_shift 
        
            for j_ in tqdm.tqdm(range(0, ny), desc=f"| {bn}: Working on chunks (1 chunk = IS rows in a file)", unit=" chunk"): 
                
                #line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
                #                          info[j_ + line_shift])
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

                    #line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
                    #                          info[j_ + line_shift + (ict + 1)])

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
                    
                    # Grid points (x,y) run through the values  
                    # 
                    # [Note]: this is how the grid generated accroding to GRASP
                    # manula. But this way results in more NaN grid values than
                    # the way below, so we will be using that instead.  
                    
                    #u0[ic, j_] = xcen + xs + dx * (ic - 1) 
                    #v0[ic, j_] = ycen + ys + dy * (j_ - 1)
                    
                    #u0[ic] = xcen + xs + dx * (ic - 1) 

                    ## Converting the cartesian (u,v) grid into (theta, phi)   
                    #theta[ic, j_], phi[ic, j_] = utils.convert_uv_to_tp(u0[ic, j_], v0[ic, j_])
                
                # To go to the next block of points within the file we need to
                # increase the line counter 
                line_shift = line_shift + in_ 

    # Generating the grid 
    #u0 = xcen + xs
    #u1 = u0 + dx * (nx - 1)
    #v0 = ycen + ys
    #v1 = v0 + dy * (ny - 1)

    #u_grid, v_grid = np.mgrid[u0:(u1 + dx):dx, v0:(v1 + dy):dy]  

    u_grid, v_grid = utils.generate_uv_grid(xcen, ycen, xs, ys, nx, ny, dx, dy)

    
    # Getting Vectors Back 
    # 
    # See the note in the generate grid as to why we do it this way
    #u = np.unique(u_grid[:, 0]) 
    #v = np.unique(v_grid[0, :])  
    u = np.unique(u_grid[0, :]) 
    v = np.unique(v_grid[:, 0])  
    #print(u) 
    #u = u.T 
    #print(v) 
    #exit() 
    
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
    

def recenter_beamdata(cimr: dict()) -> dict(): 
    """
    Method to recenter original beam.  

    The beam center in the (u,v) grid is dictated by xcen and ycen
    variables calculated above. However, as it turned out, the maximum
    beam value is not located in the center, but instead shifted in
    space. Therefore, we need to find where the maximum value is located
    on the u,v grid and offset u,v values to recenter the beam grid on
    beam's maximum value.  
    """

    Ghh_max_index = utils.get_max_index(cimr["temp"]['Ghh']) 
    Ghv_max_index = utils.get_max_index(cimr["temp"]['Ghv']) 
    Gvv_max_index = utils.get_max_index(cimr["temp"]['Gvv']) 
    Gvh_max_index = utils.get_max_index(cimr["temp"]['Gvh']) 

    print(f"| Ghh_max_index = {Ghh_max_index}")  
    print(f"| Ghv_max_index = {Ghv_max_index}")  
    print(f"| Gvv_max_index = {Gvv_max_index}")  
    print(f"| Gvh_max_index = {Gvh_max_index}")  
    
    # Get the maximum value
    Ghh_max_value = cimr["temp"]['Ghh'][Ghh_max_index]
    Ghv_max_value = cimr["temp"]['Ghv'][Ghv_max_index]
    Gvv_max_value = cimr["temp"]['Gvv'][Gvv_max_index]
    Gvh_max_value = cimr["temp"]['Gvh'][Gvh_max_index]

    print(f"| Ghh_max_value = {Ghh_max_value}")
    print(f"| Ghv_max_value = {Ghv_max_value}")
    print(f"| Gvv_max_value = {Gvv_max_value}")
    print(f"| Gvh_max_value = {Gvh_max_value}")
    
    # Get the coordinates corresponding to maximum gain inside the mesh grids
    # (u, v). This is our new central value.  
    u_coordinate = cimr['Grid']['u_grid'][Ghh_max_index]
    v_coordinate = cimr['Grid']['v_grid'][Ghh_max_index] 
    print(f"| u_coordinate = {u_coordinate}")
    print(f"| v_coordinate = {v_coordinate}")
    
    # "Shift" is the distance between two coordinates (the center of the beam
    # and the coordinate that corresponds to its maximum gain value). So we
    # just take an absolute difference  
    # 
    # [Note]: Due to floating point precision, we can get crap after 15th
    # point, so I am cutting it off.  
    u_shift = float(format(np.abs(cimr["Grid"]['xcen'] - u_coordinate), '.15f'))       
    v_shift = float(format(np.abs(cimr["Grid"]['ycen'] - v_coordinate), '.15f'))       
    #v_shift = np.abs(ycen - v_coordinate)      
    print(f"| u_shift = {u_shift}")
    print(f"| v_shift = {v_shift}")
    
    # If the maximum gain coordinate is negative then we add the shift value
    # (go right to reach zero), else --- we subtract (go left).  
    if u_coordinate < 0: 
        cimr['Grid']['u_grid']  = cimr['Grid']['u_grid'] + u_shift 
    else: 
        cimr['Grid']['u_grid']  = cimr['Grid']['u_grid'] - u_shift 
        #u_grid = u_grid - u_shift 
        
    if v_coordinate < 0: 
        cimr['Grid']['v_grid']  = cimr['Grid']['v_grid'] + v_shift 
        #v_grid = v_grid + v_shift 
    else: 
        cimr['Grid']['v_grid']  = cimr['Grid']['v_grid'] - v_shift 
        #v_grid = v_grid - v_shift 
    
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

    #print(cimr["Gain"].keys())
    #print(cimr["Grid"].keys())

    return cimr 


# TODO: This method does not work properly if we have a numpy arrays inside
#       nested dictionaries. BUT, it works for the empty dictionaries  
def is_nested_dict_empty(d):
    if not isinstance(d, dict) or not d:  # If it's not a dictionary or the dictionary is empty
        return not d
    return all(is_nested_dict_empty(v) for v in d.values())


def main(datadir:            str | pb.Path, 
         outdir:             str | pb.Path, 
         file_version:       str, 
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

    
    #print(f"| {Fore.YELLOW}=============================={Style.RESET_ALL}") 
    #print(f"| {Fore.BLUE}Parsing the Antenna Patterns{Style.RESET_ALL}") 
    #print(f"| {Fore.YELLOW}=============================={Style.RESET_ALL}") 
    logger.info(f"==============================") 
    logger.info(f"Parsing the Antenna Patterns") 
    logger.info(f"==============================") 
    

    # Creating directory to store parsed files  
    parsed_dir = outdir.joinpath("parsed", f"v{file_version}")
    io.rec_create_dir(parsed_dir, logger = logger)

    preprocessed_dir = outdir.joinpath("preprocessed", f"v{file_version}")
    io.rec_create_dir(preprocessed_dir, logger = logger)

    #print(f"| {Fore.GREEN}Data Directory:{Fore.RESET}\n| {Fore.BLUE}{datadir}{Style.RESET_ALL}") 
    #print(f"| {Fore.GREEN}Parsed Directory:{Fore.RESET}\n| {Fore.BLUE}{parsed_dir}{Style.RESET_ALL}") 
    #print(f"| {Fore.GREEN}Preprocessed Directory:{Fore.RESET}\n| {Fore.BLUE}{preprocessed_dir}{Style.RESET_ALL}") 
    logger.info(f"Data Directory:\n{datadir}") 
    logger.info(f"Parsed Directory:\n{parsed_dir}") 
    logger.info(f"Preprocessed Directory:\n{preprocessed_dir}") 


    # Main parsing loop 
    for band in apat_name_info.keys(): 
        
        for horn, half_spaces in apat_name_info[band][2].items(): 
            
            freq = apat_name_info[band][0] 
            pol  = apat_name_info[band][1] 
            
            for half_space in half_spaces: 

                # Since we do not require BHS, we skip the relevant part of the
                # loop  
                if not use_bhs and half_space == "BK": 
                    print(f"| use_bhs = {use_bhs}; skipping {half_space}.")
                    continue 

                cimr = {"Gain": {}, "Grid": {}} 
                cimr_is_empty = False  

                # Reconstructing the full path to the file 
                if band == "L": 
                    infile = band + "-" + freq + "-" + pol + "-" + half_space + ".grd" 
                else: 
                    infile = band + str(horn) + "-" + freq + "-" + pol + "-" + half_space + ".grd" 
                
                infile = pb.Path(str(datadir) + "/" + band + "/" + infile)  
                print(f"| {Fore.YELLOW}------------------------------{Style.RESET_ALL}") 
                print(f"| {Fore.GREEN}Working with Input File: {infile.name}{Style.RESET_ALL}") 

                # Parsing Original GRASP Beam Files 
                # Output filename 
                parsedfile_prefix = f"CIMR-OAP-{half_space}" 
                parsedfile_suffix = "UV"
                outfile_oap = pb.Path(str(parsed_dir) + f"/{parsedfile_prefix}-" + band + horn + f"-{parsedfile_suffix}v{file_version}.h5")   
                
                if io.check_outfile_existance(outfile_oap) == True: 
                     cimr_is_empty = is_nested_dict_empty(cimr) 
                else: 
                    
                    print(f"| ------------------------------") 
                    print(f"| Parsing")
                    print(f"| ------------------------------") 
                    
                    start_time_pars = time.time() 

                    cimr = get_beamdata(infile, half_space, file_version, cimr) #_uv, cimr_tp)     
                    exit() 
                    end_time_pars = time.time() - start_time_pars 
                    print(f"| Finished Parsing in: {end_time_pars:.2f}s") 
                    
                    print(f"| {Fore.BLUE}Saving Output File: {outfile_oap.name}{Style.RESET_ALL}") 
                    
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
                        print("| Loading data object...") 
                        start_time_pars = time.time() 
                        
                        with h5py.File(outfile_oap, 'r') as hdf5_file:
                            cimr = io.load_hdf5_to_dict(hdf5_file)
                            
                        end_time_pars = time.time() - start_time_pars 
                        print(f"| Finished Loading in: {end_time_pars:.2f}s") 


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
                        print(f"| ------------------------------") 
                        print(f"| ReCentering")
                        print(f"| ------------------------------") 
                        
                        start_time_recen = time.time() 
                        
                        cimr = recenter_beamdata(cimr)
                        
                        end_time_recen = time.time() - start_time_recen
                        print(f"| Finished Recentering in: {end_time_recen:.2f}s") 

                    
                    print(f"| ------------------------------") 
                    print(f"| Interpolating")
                    print(f"| ------------------------------") 
                    
                    start_time_interpn = time.time()  
                    
                    cimr = utils.interp_beamdata_into_uv(cimr = cimr, 
                                                         grid_res_phi   = grid_res_phi, 
                                                         grid_res_theta = grid_res_theta, 
                                                         chunk_data     = chunk_data, 
                                                         num_chunks     = num_chunks, 
                                                         overlap_margin = overlap_margin, 
                                                         interp_method  = interp_method
                                                         ) 

                    end_time_interpn = time.time() - start_time_interpn
                    print(f"| Finished Interpolation in: {end_time_interpn:.2f}s") 

                    print(f"| {Fore.BLUE}Saving Output File: {outfile_pap.name}{Style.RESET_ALL}") 
                    with h5py.File(outfile_pap, 'w') as hdf5_file:
                        io.save_dict_to_hdf5(hdf5_file, cimr)
                    

            print(f"| {Fore.YELLOW}------------------------------{Style.RESET_ALL}") 

    
if __name__ == '__main__': 
    

    start_time_tot = time.time() 
    print(f"| Starting the script using the following libraries:")
    print(f"| numpy      {np.__version__}" ) 
    print(f"| scipy      {sp.__version__}" ) 
    print(f"| h5py       {h5py.__version__}") 
    print(f"| tqdm       {tqdm.__version__}") 
    print(f"| matplotlib {matplotlib.__version__}") 
    #print(f"| colorama   {colorama.__version__}") 

    # TODO: 
    # - Create a parameter file that will take as input this info. 
    # - Create a logger object and perfomr proper logging of the functionality.   

    # Getting the root of the repo 
    root_dir = io.find_repo_root() 

    # Params to be put inside parameter file  
    outdir             = pb.Path(f"{root_dir}/output").resolve()
    datadir            = pb.Path(f"{root_dir}/dpr/AP").resolve() 
    use_bhs            = False 
    recenter_beam      = True    
    grid_res_phi       = 0.1 
    grid_res_theta     = 0.1 
    chunk_data         = True 
    num_chunks         = 4 
    overlap_margin     = 0.1 
    interp_method      = "linear"  
    file_version       = '0.6.1' 
    # Logging functionality 
    use_rgb_logging    = True  
    use_rgb_decoration = True 


    if not pb.Path(outdir).exists(): 
        print(f"Creating output directory:\n{outdir}")
        pb.Path(outdir).mkdir()

    if not datadir.is_dir():
        raise FileNotFoundError(f"The directory '{datadir}' does not exist.")

    # Getting all beam paths inside dpr/AP 
    beamfiles_paths = datadir.glob("*/*")   

    # Creating a logger object based on the user preference 
    if use_rgb_logging: 
        logger_config = pb.Path(f"{root_dir}/src/cimr_grasp/logger_config.json") 
        rgb_logging   = RGBLogging(log_config = logger_config)
        rgb_logger    = rgb_logging.get_logger("rgb") 


    #main(datadir, outdir, file_version)    
    main(datadir            = datadir, 
         outdir             = outdir, 
         file_version       = file_version,
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
         logger = rgb_logger 
         )    
    
    end_time_tot = time.time() - start_time_tot
    print(f"| Finished Script in: {end_time_tot:.2f}s") 
    print(f"| ------------------------------") 

