import pathlib as pb  
import re 
import glob 
import numbers 
import time 

import numpy as np 
import scipy as sp 
import h5py 
import xarray as xr 
import dask 
import dask.array as da 
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster 
import matplotlib
#import matplotlib.pyplot as plt 
from   colorama import Fore, Back, Style   
import tqdm 

import grasp_io as io 
import grasp_utils as utils 

# TODO: - Use xarrays instead of python dictionaries: https://tutorial.xarray.dev/overview/xarray-in-45-min.html 
#       - Use netCDF instead of HDF5 (?)

# Update by 2024-08-08: We are going to use SMAP as the baseline for the
# standardized format for parsed antenna patterns 

# Update by 2024-04-13: Since arrays are complex neither HDF5 nor NetCDF can
# save them so we are stuck with matlab or native numpy/scipy routines (such s
# npy and npz files)

# TODO: Shouldn't this be an integer value? 
def get_max_index(G) -> (int, int): 
    """
    Returns the index that correponds to the max value of NxN array.   
    
    Parameters:
    -----------
    G: ndarray 
        Gain value to get the maximum value from.  
    Returns:
    --------
     : tuple(int, int) 
        Index value that corresponds to the maximum value of G array.   
    """

    return np.unravel_index(np.nanargmax(np.abs(G)), G.shape) 


def get_beamdata(beamfile, half_space, file_version, cimr): #cimr, apat_hdf5): 
    """
    Opens GRASP `grd` file defined in uv-coordinates (IGRID value is 1) and
    returns electric field values on a (theta, phi) grid. The processing
    includes: 
    
    - Parsing original .grd file  
    - Recentering the beam grid to center on the max gain value 
    - Converting (u,v) into (theta,phi) grids 
    - Interpolating the resulting non-rectilinear grid of (theta,phi) into rectilinear (theta,phi)

    [**Note**]: The data format is described in the `CIMR_Antenna_Patterns_Format.ipynb` 
    located inside `notebooks` within the repo.   

    Parameters:
    -----------
    beamfile: str or Path
        The path to the beamfile to work at.

    half_space: str 
        The current antenna pattern's half space to work with. 

    cimr: dict 
        Beam object to be returned. 

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
            #line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
            #                          info[i])
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

                #exit() 

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
                    etheta[ic, j_], phi[ic, j_] = convert_uv_to_tp(u0[ic, j_], v0[ic, j_])
                
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
    u = np.unique(u_grid[:, 0]) 
    v = np.unique(v_grid[0, :]) 
    
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

    ## Building the complex array and getting the index that corresponds to its
    ## maximum value. 
    #Ghh = cimr["Gain"]['G1h'] + 1j * cimr["Gain"]['G2h']
    #Ghv = cimr["Gain"]['G3h'] + 1j * cimr["Gain"]['G4h']
    #Gvv = cimr["Gain"]['G1v'] + 1j * cimr["Gain"]['G2v']
    #Gvh = cimr["Gain"]['G3v'] + 1j * cimr["Gain"]['G4v']
    
    Ghh_max_index = get_max_index(cimr["temp"]['Ghh']) 
    Ghv_max_index = get_max_index(cimr["temp"]['Ghv']) 
    Gvv_max_index = get_max_index(cimr["temp"]['Gvv']) 
    Gvh_max_index = get_max_index(cimr["temp"]['Gvh']) 

    #print(f"Ghh_max_index is of type {type(Ghh_max_index[0])}") 

    print(f"| Ghh_max_index = {Ghh_max_index}")  
    print(f"| Ghv_max_index = {Ghv_max_index}")  
    print(f"| Gvv_max_index = {Gvv_max_index}")  
    print(f"| Gvh_max_index = {Gvh_max_index}")  

    #exit() 
    
    # Get the maximum value
    Ghh_max_value = cimr["temp"]['Ghh'][Ghh_max_index]
    Ghv_max_value = cimr["temp"]['Ghv'][Ghv_max_index]
    Gvv_max_value = cimr["temp"]['Gvv'][Gvv_max_index]
    Gvh_max_value = cimr["temp"]['Gvh'][Gvh_max_index]

    print(f"| Ghh_max_value = {Ghh_max_value}")
    print(f"| Ghv_max_value = {Ghh_max_value}")
    print(f"| Gvv_max_value = {Ghh_max_value}")
    print(f"| Gvh_max_value = {Ghh_max_value}")
    
    #u_grid, v_grid = np.meshgrid(cimr["Grid"]['u'], cimr["Grid"]['v']) #np.mgrid[u0:(u1 + dx):dx, v0:(v1 + dy):dy]  
    #u_grid, v_grid = u_grid.T, v_grid.T 
    #u_grid, v_grid = utils.generate_uv_grid(xcen = cimr["Grid"]['xcen'], 
    #                                  ycen = cimr["Grid"]['ycen'], 
    #                                  xs   = cimr["Grid"]['xs'], 
    #                                  ys   = cimr["Grid"]['ys'], 
    #                                  nx   = cimr["Grid"]['nx'], 
    #                                  ny   = cimr["Grid"]['ny'], 
    #                                  dx   = cimr["Grid"]['dx'], 
    #                                  dy   = cimr["Grid"]['dy']
    #                                  )  
    #print(u_grid)
    #print() 
    #print(v_grid)
    #exit() 
    
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

def interpolate_beamdata(cimr): 
    """
    Method to interpolate beamdata into rectilinear grid. 
    """
    
    #theta_grid, phi_grid = convert_uv_to_tp(u_grid, v_grid) 
    
    cimr["Grid"]['phi_grid']   = cimr["Grid"]['phi_grid'].flatten()
    cimr["Grid"]['theta_grid'] = cimr["Grid"]['theta_grid'].flatten()
    
    cimr["temp"]['Ghh']        = cimr["temp"]['Ghh'].flatten()  
    cimr["temp"]['Ghv']        = cimr["temp"]['Ghv'].flatten()  
    cimr["temp"]['Gvv']        = cimr["temp"]['Gvv'].flatten() 
    cimr["temp"]['Gvh']        = cimr["temp"]['Gvh'].flatten()  

    # Removing NaN values from the data (it is the same as to do: 
    # arr = arr[~np.isnan(arr)])
    mask_phi   = np.logical_not(np.isnan(cimr["Grid"]['phi_grid'])) 
    mask_theta = np.logical_not(np.isnan(cimr["Grid"]['theta_grid']))
    
    mask_Ghh   = np.logical_not(np.isnan(cimr["Gain"]['Ghh']))
    mask_Ghv   = np.logical_not(np.isnan(cimr["Gain"]['Ghv']))
    mask_Gvv   = np.logical_not(np.isnan(cimr["Gain"]['Gvv']))
    mask_Gvh   = np.logical_not(np.isnan(cimr["Gain"]['Gvh']))

    # Logical AND (intersection of non-NaN values in all arrays) 
    mask       = mask_theta * mask_phi * mask_Ghh * mask_Gvv * mask_Gvh * mask_Ghv 
    
    cimr["Grid"]['phi_grid']   = cimr["Grid"]['phi_grid'][mask]
    cimr["Grid"]['theta_grid'] = cimr["Grid"]['theta_grid'][mask]
    
    cimr["Gain"]['Ghh']        = cimr["Gain"]['Ghh'][mask]  
    cimr["Gain"]['Ghv']        = cimr["Gain"]['Ghv'][mask]  
    cimr["Gain"]['Gvv']        = cimr["Gain"]['Gvv'][mask]  
    cimr["Gain"]['Gvh']        = cimr["Gain"]['Gvh'][mask]  

    # TODO: Add programmatic way to do this. Technically, we can do this by
    # using max and min values of phi and theta grids. 
    
    phi_max    = np.max(cimr["Grid"]['phi_grid'])
    phi_min    = np.min(cimr["Grid"]['phi_grid'])

    buffermask = cimr["Grid"]['phi_grid'] > phi_max * 0.975 #6.2
    cimr["Grid"]['phi_grid']   = np.concatenate((cimr["Grid"]['phi_grid'],   cimr["Grid"]['phi_grid'][buffermask] - 2. * np.pi))
    cimr["Grid"]['theta_grid'] = np.concatenate((cimr["Grid"]['theta_grid'], cimr["Grid"]['theta_grid'][buffermask]))
    
    cimr["Gain"]['Ghh']        = np.concatenate((cimr["Gain"]['Ghh'], cimr["Gain"]['Ghh'][buffermask]))  
    cimr["Gain"]['Ghv']        = np.concatenate((cimr["Gain"]['Ghv'], cimr["Gain"]['Ghv'][buffermask]))   
    cimr["Gain"]['Gvv']        = np.concatenate((cimr["Gain"]['Gvv'], cimr["Gain"]['Gvv'][buffermask]))  
    cimr["Gain"]['Gvh']        = np.concatenate((cimr["Gain"]['Gvh'], cimr["Gain"]['Gvh'][buffermask]))   

    buffermask = cimr["Grid"]['phi_grid'] < 0.1
    cimr["Grid"]['phi_grid']   = np.concatenate((cimr["Grid"]['phi_grid'],   cimr["Grid"]['phi_grid'][buffermask] + 2. * np.pi))
    cimr["Grid"]['theta_grid'] = np.concatenate((cimr["Grid"]['theta_grid'], cimr["Grid"]['theta_grid'][buffermask]))
    
    cimr["Gain"]['Ghh']        = np.concatenate((cimr["Gain"]['Ghh'], cimr["Gain"]['Ghh'][buffermask]))  
    cimr["Gain"]['Ghv']        = np.concatenate((cimr["Gain"]['Ghv'], cimr["Gain"]['Ghv'][buffermask]))   
    cimr["Gain"]['Gvv']        = np.concatenate((cimr["Gain"]['Gvv'], cimr["Gain"]['Gvv'][buffermask]))  
    cimr["Gain"]['Gvh']        = np.concatenate((cimr["Gain"]['Gvh'], cimr["Gain"]['Gvh'][buffermask]))   
    
    # Adding buffer for theta 
    # 
    # [Note]: This is done because we are getting several (exactly 1 in the
    # begginning and 3 in the end) NaN values in the grids after interpolation,
    # which happens due to the fact that interpolator does have enough
    # neighboring points on edges. Therefore, we are adding more points to the
    # left, while the right values are simply put to 0 (see explanation below). 
    theta_max  = np.max(cimr["Grid"]['theta_grid'])
    buffermask = cimr["Grid"]['theta_grid'] < 0.1
    #phi_grid   = np.concatenate((phi_grid, phi_grid[buffermask]))
    #theta_grid = np.concatenate((theta_grid, -theta_grid[buffermask]))
    
    cimr["Grid"]['phi_grid']   = np.concatenate((cimr["Grid"]['phi_grid'], cimr["Grid"]['phi_grid'][buffermask]))
    cimr["Grid"]['theta_grid'] = np.concatenate((cimr["Grid"]['theta_grid'], -cimr["Grid"]['theta_grid'][buffermask]))
    #Ghh        = np.concatenate((Ghh, Ghh[buffermask]))  
    #Ghv        = np.concatenate((Ghv, Ghv[buffermask]))   
    #Gvv        = np.concatenate((Gvv, Gvv[buffermask]))  
    #Gvh        = np.concatenate((Gvh, Gvh[buffermask]))   
    cimr["Gain"]['Ghh']        = np.concatenate((cimr["Gain"]['Ghh'], cimr["Gain"]['Ghh'][buffermask]))  
    cimr["Gain"]['Ghv']        = np.concatenate((cimr["Gain"]['Ghv'], cimr["Gain"]['Ghv'][buffermask]))   
    cimr["Gain"]['Gvv']        = np.concatenate((cimr["Gain"]['Gvv'], cimr["Gain"]['Gvv'][buffermask]))  
    cimr["Gain"]['Gvh']        = np.concatenate((cimr["Gain"]['Gvh'], cimr["Gain"]['Gvh'][buffermask]))   
    
    # Should be smaller than the buffer zone defined above
    res = 0.01 
    
    #def interpolate_gain(theta, phi, gain): 
    #    return sp.interpolate.LinearNDInterpolator(list(zip(phi, theta)), gain)
    
    # TODO: The code below takes up the whole minute, so need to be sped up
    
    # This line is from online tutorial. Just leave it be. 
    #start_time_recen = time.time() 
    
    # Creating rectilinear grid 
    phi        = np.arange(0, 2. * np.pi + res, res)
    theta      = np.arange(0, theta_max, res)
    phi, theta = np.meshgrid(phi, theta)  
    
    # The code below is done in this way because we may run out of memory otherwise 

    # Tried to precompute the triangulation beforehand in the attempt to speed
    # up the code. However, it doesn't seem to work. For instance, the code
    # with Delauney part results in: 
    # 
    # D: 58.31 
    # fhh: 51.37 
    # Ghh: 106.55 
    # fhv: 53.65 
    # Ghv: 128.90
    # fvv: 55.50
    # Gvv: 133.74
    # gvh: 53.80 
    # Gvh: 134.40 
    # 
    # While, without Delaunay we get: 
    # 
    # fhh: 54.11
    # Ghh: 88.94 
    # fhv: 57.13 
    # Ghv: 134.32 
    # fvv: 57.46 
    # Gvv: 117.50 
    # fvh: 57.99 
    # Gvh: 128.28 
    # 
    # Similar things happen for other bands as well. 
    # 

    #import dask as dk  

    #client = Client() 
    #client = Client(threads_per_worker=1, n_workers=2) 
    # Setup Dask cluster with a specified number of threads or processes
    #cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    #client = Client(cluster) 
    #    
    #if False: #True: #cimr['Grid']['nx'] > 1000: 
    #    start_time_inter = time.time() 
    #    points = list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])) 
    #    triangulation = sp.spatial.Delaunay(points)  
    #    end_time_inter = time.time() - start_time_inter 
    #    print(f"| Finished with Delaunay in: {end_time_inter:.2f}s")
    #else: 
    #    triangulation = list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])) 

    #start_time_inter = time.time() 
    #points = list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])) 
    #triangulation = sp.spatial.Delaunay(points)  
    #print(triangulation[1:10]) 

    #triangulation = dk.delayed(sp.spatial.Delaunay)(points)  
    #result = triangulation.compute() 
    #print(result[1:10]) 

    #end_time_inter = time.time() - start_time_inter 

    # Use delayed to parallelize the interpolation process
    #@dask.delayed


    def interpolate_temperature(x, y, z, X, Y, interp_method = "linear"):

        grid_points = np.vstack([X.ravel(), Y.ravel()]).T 

        Z = sp.interpolate.griddata(grid_points, z, (x, y), method=interp_method) 

        #interp = sp.interpolate.LinearNDInterpolator(list(zip(x, y)), z)
        #Z = interp(X, Y)

        return Z

    start_time_inter = time.time() 

    cimr["Gain"]['Ghh'] = da.nan_to_num(interpolate_temperature().T, nan=0.0)  

    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Ghh in: {end_time_inter:.2f}s")


    #phi        = da.arange(0, 2. * np.pi + res, res)
    #theta      = da.arange(0, theta_max, res)
    #phi, theta = da.meshgrid(phi, theta)  

    ## Measure the time of the delayed task creation
    #start_task_creation_time = time.time()

    ## Call the delayed function for interpolation
    #Z_delayed = interpolate_and_meshgrid(cimr['Grid']['phi_grid'],
    #                             cimr['Grid']['theta_grid'],
    #                             cimr['Gain']['Ghh'], phi.compute(),
    #                             theta.compute())

    #task_creation_time = time.time() - start_task_creation_time 

    ## Trigger the computation
    #start_computation_time = time.time()

    #with ProgressBar():
    #    Z = Z_delayed.compute()

    #computation_time = time.time() - start_computation_time 

    #print(f"Time to create delayed task: {task_creation_time:.4f} seconds")
    #print(f"Time to compute task: {computation_time:.4f} seconds") 

    #print(Z.shape)
    #print(Z[0,0])
    #exit() 

    start_time_inter = time.time() 
    cimr["Gain"]['Ghh'] = da.nan_to_num(Z.T, nan=0.0)  
    end_time_inter = time.time() - start_time_inter 
    print(cimr['Gain']['Ghh'])
    print(f"| Finished with Ghh in: {end_time_inter:.2f}s")
    del Z 

    client.shutdown() 

    #start_time_inter = time.time() 
    ##fhh        = sp.interpolate.LinearNDInterpolator(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])), cimr["Gain"]['Ghh']) 
    ##fhh        = sp.interpolate.LinearNDInterpolator(triangulation, cimr["Gain"]['Ghh']) 

    ##end_time_inter = time.time() - start_time_inter 
    ##print(f"| Finished with fhh in: {end_time_inter:.2f}s")

    #start_time_inter = time.time() 
    #triangulation = da.from_array(triangulation, chunks = (1000))  
    #print(triangulation.shape) 
    #print(cimr['Gain']['Ghh'].shape) 
    #cimr['Gain']['Ghh'] = da.from_array(cimr["Gain"]['Ghh'], chunks = (1000))
    #fhh        = dk.delayed(sp.interpolate.LinearNDInterpolator)(triangulation, cimr["Gain"]['Ghh']) 
    #with ProgressBar(): 
    #    fhh.compute() 
    ##fhh        = client.submit(sp.interpolate.LinearNDInterpolator, triangulation, cimr["Gain"]['Ghh']) 

    #end_time_inter = time.time() - start_time_inter 
    #print(f"| Finished with fhh in: {end_time_inter:.2f}s")
    exit() 
    
    start_time_inter = time.time() 
    cimr["Gain"]['Ghh'] = np.nan_to_num(fhh(phi, theta).T, nan=0.0)  
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Ghh in: {end_time_inter:.2f}s")
    del fhh 

    G1h, G2h   = np.real(cimr["Gain"]['Ghh']), np.imag(cimr["Gain"]['Ghh'])  
    del cimr["Gain"]['Ghh'] 
    
    
    start_time_inter = time.time() 
    #fhv        = sp.interpolate.LinearNDInterpolator(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])), cimr["Gain"]['Ghv']) 
    fhv        = sp.interpolate.LinearNDInterpolator(triangulation, cimr["Gain"]['Ghv']) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with fhv in: {end_time_inter:.2f}s")
    
    start_time_inter = time.time() 
    cimr["Gain"]['Ghv'] = np.nan_to_num(fhv(phi, theta).T, nan=0.0)
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Ghv in: {end_time_inter:.2f}s")
    del fhv 
    
    G3h, G4h   = np.real(cimr["Gain"]['Ghv']), np.imag(cimr["Gain"]['Ghv'])  
    del cimr["Gain"]['Ghv'] 

    
    start_time_inter = time.time() 
    #fvv        = sp.interpolate.LinearNDInterpolator(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])), cimr["Gain"]['Gvv']) 
    fvv        = sp.interpolate.LinearNDInterpolator(triangulation, cimr["Gain"]['Gvv']) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with fvv in: {end_time_inter:.2f}s")
    
    start_time_inter = time.time() 
    cimr["Gain"]['Gvv'] = np.nan_to_num(fvv(phi, theta).T, nan=0.0) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Gvv in: {end_time_inter:.2f}s")
    del fvv 
    
    G1v, G2v   = np.real(cimr["Gain"]['Gvv']), np.imag(cimr["Gain"]['Gvv']) 
    del cimr["Gain"]['Gvv'] 

    
    
    start_time_inter = time.time() 
    #fvh        = sp.interpolate.LinearNDInterpolator(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])), cimr["Gain"]['Gvh']) 
    fvh        = sp.interpolate.LinearNDInterpolator(triangulation, cimr["Gain"]['Gvh']) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with fvh in: {end_time_inter:.2f}s")
    
    start_time_inter = time.time() 
    cimr["Gain"]['Gvh'] = np.nan_to_num(fvh(phi, theta).T, nan=0.0) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Gvh in: {end_time_inter:.2f}s")
    del fvh 
    
    G3v, G4v   = np.real(cimr["Gain"]['Gvh']), np.imag(cimr["Gain"]['Gvh'])  
    del cimr["Gain"]['Gvh'] 
    
    exit() 
    #end_time_recen = time.time() - start_time_recen
    #print(f"| Finished Interpolation in: {end_time_recen:.2f}s") 
    

    # SciPy uses Delauney triangulation, which we can precompute and thus speed
    # up the code 
    # Precompute the triangulation
    #start_time_recen = time.time() 
    #
    #triang = sp.spatial.Delaunay(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])))
    #fhh    = sp.interpolate.LinearNDInterpolator(triang, cimr["Gain"]['Ghh']) #, fill_value=0)
    #fhv    = sp.interpolate.LinearNDInterpolator(triang, cimr["Gain"]['Ghv']) #, fill_value=0)
    #fvv    = sp.interpolate.LinearNDInterpolator(triang, cimr["Gain"]['Gvv']) #, fill_value=0)
    #fvh    = sp.interpolate.LinearNDInterpolator(triang, cimr["Gain"]['Gvh']) #, fill_value=0)
    #
    ## Creating rectilinear grid 
    #phi        = np.arange(0, 2. * np.pi + res, res)
    #theta      = np.arange(0, theta_max, res)
    #phi, theta = np.meshgrid(phi, theta)  
    #
    ## Interpolating the function and substituting the last NaN values in the
    ## arrays with zeros, because they are not intersecting the Earth (once you
    ## do the projection) 
    #
    #cimr["Gain"]['Ghh'] = np.nan_to_num(fhh(phi, theta).T, nan=0.0)  
    #cimr["Gain"]['Ghv'] = np.nan_to_num(fhv(phi, theta).T, nan=0.0)
    #cimr["Gain"]['Gvv'] = np.nan_to_num(fvv(phi, theta).T, nan=0.0) 
    #cimr["Gain"]['Gvh'] = np.nan_to_num(fvh(phi, theta).T, nan=0.0) 
    #
    #end_time_recen = time.time() - start_time_recen
    #print(f"| Finished Interpolation in: {end_time_recen:.2f}s") 


    # Creating rectilinear grid 
    #phi        = np.arange(0, 2. * np.pi + res, res)
    #theta      = np.arange(0, theta_max, res)
    #phi, theta = np.meshgrid(phi, theta)  
    
    # Interpolating the function and substituting the last NaN values in the
    # arrays with zeros, because they are not intersecting the Earth (once you
    # do the projection) 
    
    #cimr["Gain"]['Ghh'] = np.nan_to_num(fhh(phi, theta).T, nan=0.0)  
    #print("| Finished iwht Ghh")
    #del fhh 
    #cimr["Gain"]['Ghv'] = np.nan_to_num(fhv(phi, theta).T, nan=0.0)
    #print("| Finished with Ghv")
    #del fhv 
    #cimr["Gain"]['Gvv'] = np.nan_to_num(fvv(phi, theta).T, nan=0.0) 
    #print("| Finished with Gvv")
    #del fvv 
    #cimr["Gain"]['Gvh'] = np.nan_to_num(fvh(phi, theta).T, nan=0.0) 
    #print("| Finished with Gvh")
    #del fvh 

    phi, theta = phi.T, theta.T 

    # Getting back initial vectors and converting them into degrees 
    phi        = np.rad2deg(np.unique(phi[:, 0])) 
    theta      = np.rad2deg(np.unique(theta[0, :])) 
    
    # Splitting the arrays into real and imaginary parts 
    #G1h, G2h   = np.real(cimr["Gain"]['Ghh']), np.imag(cimr["Gain"]['Ghh'])  
    #G3h, G4h   = np.real(cimr["Gain"]['Ghv']), np.imag(cimr["Gain"]['Ghv'])  
    #G1v, G2v   = np.real(cimr["Gain"]['Gvv']), np.imag(cimr["Gain"]['Gvv']) 
    #G3v, G4v   = np.real(cimr["Gain"]['Gvh']), np.imag(cimr["Gain"]['Gvh'])  
    
    # Getting the resulting dictionary  
    # (and removing unnecessary fields)
                    
    cimr["Gain"]['G1h'] = G1h 
    del G1h 
    cimr["Gain"]['G2h'] = G2h 
    del G2h 
    cimr["Gain"]['G3h'] = G3h 
    del G3h 
    cimr["Gain"]['G4h'] = G4h 
    del G4h 

    cimr["Gain"]['G1v'] = G1v 
    del G1v 
    cimr["Gain"]['G2v'] = G2v 
    del G2v 
    cimr["Gain"]['G3v'] = G3v 
    del G3v 
    cimr["Gain"]['G4v'] = G4v 
    del G4v 
    
    #cimr["Grid"]['u']     = u_grid #u_values #u0  
    #cimr["Grid"]['v']     = v_grid #v_values #v0   
    #cimr["Grid"]['u_cen'] = xcen #u_coordinate  
    #cimr["Grid"]['v_cen'] = ycen #v_coordinate   
    cimr["Grid"]['theta'] = theta #_grid  
    cimr["Grid"]['phi']   = phi #_grid   
    
    #print(cimr["Gain"].keys())
    #print(cimr["Grid"].keys())
                    
    return cimr 

# TODO: This method does not work properly if we have a numpy arrays inside
# nested dictionaries 
def is_nested_dict_empty(d):
    if not isinstance(d, dict) or not d:  # If it's not a dictionary or the dictionary is empty
        return not d
    return all(is_nested_dict_empty(v) for v in d.values())


#def main(datadir, outdir, ref_tilt_ang_deg, downscale_factor, vsign, nu, nv, file_version, save_tp=True, save_uv=True):     
def main(datadir, outdir, file_version, recenter_beam = True) -> None:     
    """
    Main method (entry point to the program)
    
    Parameters:
    -----------
    datadir: str or Path
        The path to the data directory where all beam files are located. 

    outdir: str or Path  
        The path to the output directory where to store all results of execution. 

    file_version: float  
        Version of the parsed files to be produced. 
    recenter_beam: bool 
        Parameter that defines whether to recenter beam or not 
    """

    # List of all beam files 
    beamfiles_in  = [] 
    beamfiles_out = [] 
    # List of all band dirs containing respective beam files 
    bands_dirs    = [] 
    # List of the output directories to be created  
    bands_outdirs = [] 
    i = 0 
    j = 0
    
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

    #print(apat_name_info) 

    
    print(f"| {Fore.YELLOW}=============================={Style.RESET_ALL}") 
    print(f"| {Fore.BLUE}Parsing the Antenna Patterns{Style.RESET_ALL}") 
    print(f"| {Fore.YELLOW}=============================={Style.RESET_ALL}") 
    

    # Creating directory to store parsed files  
    parsed_dir = outdir.joinpath("parsed", f"v{file_version}")
    io.rec_create_dir(parsed_dir)

    preprocessed_dir = outdir.joinpath("preprocessed", f"v{file_version}")
    io.rec_create_dir(preprocessed_dir)

    #if not preprocessed_dir.exists(): 
    #    print(f"| Creating 'preprocessed' directory:\n{preprocessed_dir}")
    #    pathlib.Path(preprocessed_dir).mkdir()
    #exit() 


    #parsedfile_prefix = "CIMR-AP-FULL" 
    #parsedfile_suffix = "XE"
    
    print(f"| {Fore.GREEN}Data Directory:{Fore.RESET}\n| {Fore.BLUE}{datadir}{Style.RESET_ALL}") 
    print(f"| {Fore.GREEN}Parsed Directory:{Fore.RESET}\n| {Fore.BLUE}{parsed_dir}{Style.RESET_ALL}") 
    print(f"| {Fore.GREEN}Preprocessed Directory:{Fore.RESET}\n| {Fore.BLUE}{preprocessed_dir}{Style.RESET_ALL}") 

    
    # Main parsing loop 
    for band in apat_name_info.keys(): 
        
        for horn, half_spaces in apat_name_info[band][2].items(): 
            
            freq = apat_name_info[band][0] 
            pol  = apat_name_info[band][1] 
            
            # Object to be saved into file 
            #cimr = {"Gain": {}, "Grid": {}, "Version": file_version} 
            #cimr = {"Gain": {}, "Grid": {}} 
            #cimr_is_empty = False  
            
            #def check_a_n_empty(m):
            #    return isinstance(m, dict) and not m.get('A') and not m.get('N')


            # Example usage
            #nested_dict = {"a": {}, "b": {"c": {}}}
            #print(is_nested_dict_empty(nested_dict))  # Output: True

            #nested_dict = {"a": {}, "b": {"c": {"d": 1}}}
            #print(is_nested_dict_empty(nested_dict))  # Output: False
            #print(is_nested_dict_empty(cimr))  # Output: True
            #exit() 
     

            # Example usage
            #m = {'A': {}, 'N': {}, 'S': 'version'}
            #print(check_a_n_empty(m))  # Output: True

            #m = {'A': {}, 'N': {'some_key': 'some_value'}, 'S': 'version'}
            #print(check_a_n_empty(m))  # Output: False
 
            
            for half_space in half_spaces: 

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
                #    continue 
                #    #print(cimr.keys())
                     cimr_is_empty = is_nested_dict_empty(cimr) 
                #if io.check_outfile_existance(outfile_oap) != True: 
                else: 
                    
                    print(f"| ------------------------------") 
                    print(f"| Parsing")
                    print(f"| ------------------------------") 
                    
                    start_time_pars = time.time() 

                    cimr = get_beamdata(infile, half_space, file_version, cimr) #_uv, cimr_tp)     
                    
                    end_time_pars = time.time() - start_time_pars 
                    print(f"| Finished Parsing in: {end_time_pars:.2f}s") 
                    
                    print(f"| {Fore.BLUE}Saving Output File: {outfile_oap.name}{Style.RESET_ALL}") 
                    
                    with h5py.File(outfile_oap, 'w') as hdf5_file:
                        io.save_dict_to_hdf5(hdf5_file, cimr)


                #print(cimr.keys())
                #exit() 

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

                    print(cimr['Grid'].keys()) 
                    print(cimr['Gain'].keys()) 

                    # Creating a set of temporary values to be removed from memory 
                    cimr["temp"] = {} 
                    cimr = utils.construct_complete_gains(cimr) 
                    print(cimr['Grid'].keys()) 
                    print(cimr['Gain'].keys()) 
                    print(cimr['temp'].keys()) 
                    cimr["Grid"]["u_grid"], cimr["Grid"]["v_grid"] = utils.generate_uv_grid(
                                                              xcen = cimr["Grid"]['xcen'], 
                                                              ycen = cimr["Grid"]['ycen'], 
                                                              xs   = cimr["Grid"]['xs'], 
                                                              ys   = cimr["Grid"]['ys'], 
                                                              nx   = cimr["Grid"]['nx'], 
                                                              ny   = cimr["Grid"]['ny'], 
                                                              dx   = cimr["Grid"]['dx'], 
                                                              dy   = cimr["Grid"]['dy']
                                                            )  
                    print(cimr['temp'].keys()) 
                    #exit() 

                    # If no recentering, then just create a temporary grid to
                    # pass those values further and then delete them once we
                    # are done 
                    #if recenter_beam: 
                    #    print(f"| ------------------------------") 
                    #    print(f"| ReCentering")
                    #    print(f"| ------------------------------") 
                    #    
                    #    start_time_recen = time.time() 
                    #    
                    #    cimr = recenter_beamdata(cimr)
                    #    
                    #    end_time_recen = time.time() - start_time_recen
                    #    print(f"| Finished Recentering in: {end_time_recen:.2f}s") 

                    
                    print(f"| ------------------------------") 
                    print(f"| Interpolating")
                    print(f"| ------------------------------") 
                    
                    start_time_interpn = time.time()  
                    
                    #cimr = interpolate_beamdata(cimr)
                    cimr = utils.interp_beamdata_into_uv(cimr) 
                    #exit() 

                    # Deleting redundant fields  
                    del cimr["Grid"]['dx'] 
                    del cimr["Grid"]['dy'] 
                    del cimr["Grid"]['nx'] 
                    del cimr["Grid"]['ny'] 
                    del cimr["Grid"]['xs'] 
                    del cimr["Grid"]['ys']    
                    #del cimr["Grid"]['u'] 
                    #del cimr["Grid"]['v'] 
                    del cimr["Grid"]['xcen'] 
                    del cimr["Grid"]['ycen'] 
                    #del cimr["Grid"]['u_grid'] 
                    #del cimr["Grid"]['v_grid'] 
                    #del cimr["Grid"]['phi_grid'] 
                    #del cimr["Grid"]['theta_grid'] 
                    
                    #del cimr["Gain"]['Ghh'] 
                    #del cimr["Gain"]['Ghv'] 
                    #del cimr["Gain"]['Gvv'] 
                    #del cimr["Gain"]['Gvh'] 

                    #print(cimr["Gain"].keys())
                    #print(cimr["Grid"].keys())

                    end_time_interpn = time.time() - start_time_interpn
                    print(f"| Finished Interpolation in: {end_time_interpn:.2f}s") 

                    print(f"| {Fore.BLUE}Saving Output File: {outfile_pap.name}{Style.RESET_ALL}") 
                    with h5py.File(outfile_pap, 'w') as hdf5_file:
                        io.save_dict_to_hdf5(hdf5_file, cimr)
                    
                    #exit() 
                

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

    # TODO: Create a parameter file that will take as input this info  

    # Getting the root of the repo 
    root_dir = io.find_repo_root() 

    file_version = 0.5 

    # Params to be put inside parameter file 
    outdir  = pb.Path(f"{root_dir}/output").resolve()
    datadir = pb.Path(f"{root_dir}/dpr/AP").resolve() 

    if not pb.Path(outdir).exists(): 
        print(f"Creating output directory:\n{outdir}")
        pb.Path(outdir).mkdir()

    if not datadir.is_dir():
        raise FileNotFoundError(f"The directory '{datadir}' does not exist.")

    # Getting all beam paths inside dpr/AP 
    beamfiles_paths = datadir.glob("*/*")   


    #main(datadir, outdir, ref_tilt_ang_deg, downscale_factor, vsign, nu, nv, file_version)    
    main(datadir, outdir, file_version)    
    
    end_time_tot = time.time() - start_time_tot
    print(f"| Finished Script in: {end_time_tot:.2f}s") 
    print(f"| ------------------------------") 

