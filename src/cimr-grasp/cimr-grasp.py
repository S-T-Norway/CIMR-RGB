import pathlib 
import re 
import glob 
import numbers 
import time 

import numpy as np 
import scipy as sp 
import h5py 
import xarray as xr 
import matplotlib
import matplotlib.pyplot as plt 
from   colorama import Fore, Back, Style   
import tqdm 

import grasp_io as io 

# TODO: - Use xarrays instead of python dictionaries: https://tutorial.xarray.dev/overview/xarray-in-45-min.html 
#       - Use netCDF instead of HDF5 (?)

# Update by 2024-08-08: We are going to use SMAP as the baseline for the
# standardized format for parsed antenna patterns 

# Update by 2024-04-13: Since arrays are complex neither HDF5 nor NetCDF can
# save them so we are stuck with matlab or native numpy/scipy routines (such s
# npy and npz files)

def get_max_index(G): 
    """
    Returns the index that correponds to the max value of NxN array.   
    
    Parameters:
    -----------
    G: ndarray 
        Gain value to get the maximum value from.  
    Returns:
    --------
     : float 
        Index value that corresponds to the maximum value of G array.   
    """

    return np.unravel_index(np.nanargmax(np.abs(G)), G.shape) 



def uv_to_tp(u,v): 
    """
    Converting (u,v) into (theta,phi) and returning the grid in radians.   
    
    According to the GRASP manual, the relations between (u, v) and (theta,
    phi) coordinates are:  
    
    $$
    u=\sin\theta\cos\phi
    $$
    $$
    v=\sin\theta\sin\phi
    $$ 

    which makes up the unit vector to the field point as  

    $$
    \hat{r} = \left( u, v, \sqrt{1 - u^2 - v^2} \right)
    $$ 

    The reverse relations then are: 
    
    $$
    \theta=\arccos{\sqrt{1-u^2-v^2}}
    $$
    $$
    \phi = \arctan\left(\frac{v}{u}\right)
    $$ 

    where $\phi$ is of [-180, 180] and $\theta$ is [-90, 90] (in degrees).  

    See, e.g. for different conventions: 
    https://en.wikipedia.org/wiki/Spherical_coordinate_system 

    Parameters:
    -----------
    u: float or ndarray 
        U coordinate in director cosine coordinate system 
        
    v: float or ndarray 
        V coordinate in director cosine coordinate system 

    Returns:
    --------
    theta: float or ndarray  
        Theta angle value 
        
    phi  : float or ndarray 
        Phi angle value 
    """

    #theta = np.degrees(np.arccos(np.sqrt(1 - u**2 - v**2))) 
    #phi   = np.degrees(np.arctan2(v, u)) 
    
    theta = np.arccos(np.sqrt(1 - u**2 - v**2)) 
    phi   = np.arctan2(v, u) 

    # Following SMAP convention, we need values for phi to be [0, 360]. 
    # 
    # [Note]: pcolor from matplotlib won't be able to properly output it on the
    # screen after this operation, but the arrays we get are correct
    # nevertheless.  
    #phi[phi < 0] += np.rad2deg(2.0 * np.pi) 
    phi[phi < 0] += 2.0 * np.pi 
      
    return theta, phi 



#def get_beamdata(beamfile, half_space, cimr_uv, cimr_tp): #cimr, apat_hdf5): 
def get_beamdata(beamfile, half_space, cimr): #cimr, apat_hdf5): 
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
    
    with open(beamfile, mode = "r", encoding = "UTF-8") as bfile: 

        header = io.get_header(bfile)

        # Retrieving data after ++++ 
        
        # First 3 lines after ++++  
        info = [line.strip("\n") for i, line in enumerate(bfile)] 
        line_shift = 3
        for i in range(0, line_shift):
            line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
                                      info[i])

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
                
                line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', info[k])

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
            G1h = np.full((ny, nx), np.nan, dtype=float)
            G2h = np.full((ny, nx), np.nan, dtype=float)
            G3h = np.full((ny, nx), np.nan, dtype=float)
            G4h = np.full((ny, nx), np.nan, dtype=float)

            G1v = np.full((ny, nx), np.nan, dtype=float)
            G2v = np.full((ny, nx), np.nan, dtype=float)
            G3v = np.full((ny, nx), np.nan, dtype=float)
            G4v = np.full((ny, nx), np.nan, dtype=float)
            
            #u0 = np.full((ny, nx), np.nan, dtype=float)
            #v0 = np.full((ny, nx), np.nan, dtype=float)
            
            #u0 = np.full(ny, np.nan, dtype=float)
            #v0 = np.full(ny, np.nan, dtype=float)

            #theta = np.full((ny, nx), np.nan, dtype=float)
            #phi   = np.full((ny, nx), np.nan, dtype=float)
            
            
            # i_row is J in GRASP manual 
            # 
            # Here i_row is the row number after (nx, ny, klimit) row in a
            # file. So, to get the current row in a grd file, we need to add
            # i_row and line_shift 
            
            #for i_row in tqdm(range(0, ny), desc=f"Working on row {i_row+1}", unit=i_row):
            #for i_row in tqdm.tqdm(range(0, ny), desc=f"| {bn}: Working on chunks (1 chunk = IS rows in a file)", unit=" chunk"): 
            
            #X = []
            #Y = [] 

            #for j_ in range(0, ny): 
            for j_ in tqdm.tqdm(range(0, ny), desc=f"| {bn}: Working on chunks (1 chunk = IS rows in a file)", unit=" chunk"): 
                
                line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
                                          info[j_ + line_shift])
                
                # Get the number of columns to read 
                if klimit == 0: 
                    # It is IS in GRASP manual --- the column number of 1st datapoint  
                    is_ = 1 #ix0 = 1 
                    # It is IN in GRASP manual --- the # of datapoints in row J (i_row)
                    in_ = nx #nxr = nx  
                elif klimit == 1: 
                    is_ = int(line_numbers[0]) #ix0 = int(line_numbers[0]) 
                    in_ = int(line_numbers[1]) #nxr = int(line_numbers[1])

                """
                In GRASP manual it is written that this line should be looped
                as: 
                
                F1(I), F2(I), I = IS, IE 

                where IS is is_ in our case and IE = IS + IN - 1.  

                If we now subtract (IE - IS) to start the array from index 0
                instead of IS, then we get IN - 1. Python subtracts one by
                default, so we loop to in_ which is IN in our case. 
                """

                #Y.append(ycen + ys + dy * (j_ - 1)) 
                #
                #ie_ = is_ + in_ 
                #print(f"| IS = {is_}, IN = {in_}, IE = {ie_}")

                #x_test = np.full(in_, np.nan, dtype=float)
                #for i_ in range(is_, ie_): 
                #    print(i_)
                #    line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
                #                              info[j_ + line_shift + (i_ - is_ + 1)])
                #    G3h[i_, j_] = float(line_numbers[0])
                #    print(G3h[i_, j_])

                #    x_test[i_ - is_] = xcen + xs + dx * (i_ - 1) 
                #X.append(x_test)
                #
                #print(X)

                #line_shift = line_shift + in_ #nxr 
                #print(line_shift) 
                #if j_ >= 3: 
                #    exit() 

                
                #v0[i_row] = ycen + ys + dy * (i_row - 1)
                
                #print(line_shift) 
                #for ict in range(nxr):
                for ict in range(in_):
                #for ict in tqdm.tqdm(range(nxr), desc="NX", leave=False, unit=" col"):

                    line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
                                              info[j_ + line_shift + (ict + 1)])
                    # If is_ = 1, then ic and ict are exactly the same it seems 
                    # the matlab version starts with index 1, but python starts with 0 
                    ic = is_ + ict - 1 #ix0 + ict - 1  
                    #print(f"is_={is_}: ic={ic}") 
                    
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
                    
                    ## Grid points (x,y) run through the values  
                    #u0[ic, i_row] = xcen + xs + dx * (ic - 1) 
                    #v0[ic, i_row] = ycen + ys + dy * (i_row - 1)
                    
                    #u0[ic] = xcen + xs + dx * (ic - 1) 

                    ## Converting the cartesian (u,v) grid into (theta, phi)   
                    #theta[ic, i_row], phi[ic, i_row] = uv_to_tp(u0[ic, i_row], v0[ic, i_row])
                
                # To go to the next block of points within the file we need to
                # increase the line counter 
                line_shift = line_shift + in_ #nxr 
                #print(line_shift) 
                #if j_ >= 3: 
                #    exit() 



        u0 = xcen + xs
        u1 = u0 + dx * (nx - 1)
        v0 = ycen + ys
        v1 = v0 + dy * (ny - 1)

        # The uv coordinates 
        #u_values = np.arange(u0, u1 + dx, dx)
        #v_values = np.arange(v0, v1 + dy, dy)
        ## Theta.phi coordinates 
        #theta = np.full(ny, np.nan, dtype=float)
        #phi   = np.full(ny, np.nan, dtype=float)
        ##theta, phi = uv_to_tp(u_values, v_values)
        ##theta, phi = [], [] 
        #for i in range(len(u0)): 
            #theta[i], phi[i] = uv_to_tp(u_values[i], v_values[i]) 
        #    theta[i], phi[i] = uv_to_tp(u0[i], v0[i]) 
            
        #theta = np.full((ny, nx), np.nan, dtype=float)
        #phi   = np.full((ny, nx), np.nan, dtype=float)
        
        #u_vec = np.full(nx, np.nan, dtype=float)
        #v_vec = np.full(ny, np.nan, dtype=float)

        #is_ = 1 
        #for ic in range(0, nx): 
        #    #ic = is_ + ict - 1 #ix0 + ict - 1  
        #    u_vec[ic]    = u0 + dx * (ic - 1) 
        #for i_row in range(0, ny): 
        #    v_vec[i_row] = v0 + dy * (i_row - 1)

        #print(u_vec)
        #print(v_vec)
        #exit() 
        
        u_grid, v_grid = np.mgrid[u0:(u1 + dx):dx, v0:(v1 + dy):dy]  
        
    print(f"| ------------------------------") 
    print(f"| ReCentering")
    print(f"| ------------------------------") 
    
    start_time_recen = time.time() 
     
    # The beam center in the (u,v) grid is dictated by xcen and ycen
    # variables calculated above. However, as it runed out, the maximum
    # beam value is not located in the center, but instead shifted in
    # space. Therefore, we need to find where the maximum value is located
    # on the u,v grid and offset u,v values to recenter the beam grid on
    # beam's maximum value.  

    # Building the complex array and getting the index that corresponds to its
    # maximum value. 
    Ghh = G1h + 1j * G2h 
    Ghv = G3h + 1j * G4h 
    Gvv = G1v + 1j * G2v
    Gvh = G3v + 1j * G4v 

    
    #G1h_max_index  = get_max_index(G1h) 
    #G2h_max_index  = get_max_index(G2h) 
    #G3h_max_index  = get_max_index(G3h) 
    #G4h_max_index  = get_max_index(G4h) 
    #
    #print(f"G1h_max_index = {G1h_max_index}")  
    #print(f"G2h_max_index = {G2h_max_index}")  
    #print(f"G3h_max_index = {G3h_max_index}")  
    #print(f"G4h_max_index = {G4h_max_index}")  

    #exit() 
    
    Ghh_max_index = get_max_index(Ghh) 
    Ghv_max_index = get_max_index(Ghv) 
    Gvv_max_index = get_max_index(Gvv) 
    Gvh_max_index = get_max_index(Gvh) 

    print(f"| Ghh_max_index = {Ghh_max_index}")  
    print(f"| Ghv_max_index = {Ghv_max_index}")  
    print(f"| Gvv_max_index = {Gvv_max_index}")  
    print(f"| Gvh_max_index = {Gvh_max_index}")  
    
    # Get the maximum value
    Ghh_max_value = Ghh[Ghh_max_index]
    Ghv_max_value = Ghv[Ghv_max_index]
    Gvv_max_value = Gvv[Gvv_max_index]
    Gvh_max_value = Gvh[Gvh_max_index]

    print(f"| Ghh_max_value = {Ghh_max_value}")
    print(f"| Ghv_max_value = {Ghh_max_value}")
    print(f"| Gvv_max_value = {Ghh_max_value}")
    print(f"| Gvh_max_value = {Ghh_max_value}")
    
    # Get the coordinates corresponding to maximum gain inside the mesh grids
    # (u, v). This is our new central value.  
    u_coordinate = u_grid[Ghh_max_index]
    v_coordinate = v_grid[Ghh_max_index] 
    print(f"| u_coordinate = {u_coordinate}")
    print(f"| v_coordinate = {v_coordinate}")


    # Calculating coordinate shift. The value can be positive or negative.  
    #u_shift = xcen + u_coordinate     
    #v_shift = ycen + v_coordinate     

    # "Shift" is the distance between two coordinates (the center of the beam
    # and the coordinate that corresponds to its maximum gain value). So we
    # just take an absolute difference  
    # 
    # [Note]: Due to floating point precision, we can get crap after 15th
    # point, so I am cutting it off.  
    u_shift = float(format(np.abs(xcen - u_coordinate), '.15f'))       
    v_shift = float(format(np.abs(ycen - v_coordinate), '.15f'))       
    #v_shift = np.abs(ycen - v_coordinate)      
    print(f"| u_shift = {u_shift}")
    print(f"| v_shift = {v_shift}")
    
    # If the maximum gain coordinate is negative then we add the shift value
    # (go right to reach zero), else --- we subtract (go left).  
    if u_coordinate < 0: 
        u_grid = u_grid + u_shift 
    else: 
        u_grid = u_grid - u_shift 
        
    if v_coordinate < 0: 
        v_grid = v_grid + v_shift 
    else: 
        v_grid = v_grid - v_shift 
    
    # Getting unique values which represent coordinates 
    #u_vec = np.unique(u_grid[:, 0]) 
    #v_vec = np.unique(v_grid[0, :]) 

    #print(u_grid[:, :])

    #print() 
    #print(u_vec)


    #print("v")
    #print(v_grid[:, :])
    #print(v_vec)
    #exit() 
    
    #print(u_grid[800:900][:])  
    #print(v_grid[800:900][:])  
    #print() 
    #theta, phi = uv_to_tp(u_grid, v_grid) 
    
    # We cannot get unique values for theta and phi, because the converted
    # grid is not rectilinear anymore   
    theta_grid, phi_grid = uv_to_tp(u_grid, v_grid) 
    
    end_time_recen = time.time() - start_time_recen
    print(f"| Finished Recentering in: {end_time_recen:.2f}s") 

    # TODO: Write it into a separate method 
    # 
    # Interpolating to get the rectilinear grid 
    print(f"| ------------------------------") 
    print(f"| Interpolating")
    print(f"| ------------------------------") 
    
    start_time_interpn = time.time()  
    
    phi_grid   = phi_grid.flatten()
    theta_grid = theta_grid.flatten()
    Ghh        = Ghh.flatten()  
    Ghv        = Ghv.flatten()  
    Gvv        = Gvv.flatten() 
    Gvh        = Gvh.flatten()  

    #print(phi_grid)
    #print(Ghh)
    #
    #print(phi_grid.shape)
    #print(Ghh.shape)
    
    # Removing NaN values from the data (it is the same as to do: 
    # arr = arr[~np.isnan(arr)])
    mask_phi   = np.logical_not(np.isnan(phi_grid))
    mask_theta = np.logical_not(np.isnan(theta_grid))
    mask_Ghh   = np.logical_not(np.isnan(Ghh))
    mask_Ghv   = np.logical_not(np.isnan(Ghv))
    mask_Gvv   = np.logical_not(np.isnan(Gvv))
    mask_Gvh   = np.logical_not(np.isnan(Gvh))

    # Logical AND (intersection of non-NaN values in all arrays) 
    mask       = mask_theta * mask_phi * mask_Ghh * mask_Gvv * mask_Gvh * mask_Ghv 

    phi_grid   = phi_grid[mask]
    theta_grid = theta_grid[mask]
    Ghh        = Ghh[mask]  
    Ghv        = Ghv[mask]  
    Gvv        = Gvv[mask]  
    Gvh        = Gvh[mask]  

    #print(phi_grid)
    #print(Ghh)
    #
    #print(phi_grid.shape)
    #print(Ghh.shape)

    ##print(2 * np.pi)
    #print(np.any(np.isnan(Ghh)))
    #print(np.any(np.isnan(Gvv)))
    #print(np.any(np.isnan(Gvh)))
    #print(np.any(np.isnan(Ghv)))
    
    # TODO: Add programmatic way to do this. Technically, we can do this by
    # using max and min values of phi and theta grids. 
    
    #print(np.min(phi_grid), phi_grid.max()) 
    #print(np.min(theta_grid), theta_grid.max()) 
    phi_max = np.max(phi_grid)
    phi_min = np.min(phi_grid)

    #print(f"phi_max = {phi_max}, phi_min = {phi_min}")
    #print(phi_max * 0.975)
    #print(phi_min * 1.025)
    #exit()

    buffermask = phi_grid > phi_max * 0.975 #6.2
    phi_grid   = np.concatenate((phi_grid, phi_grid[buffermask] - 2. * np.pi))
    theta_grid = np.concatenate((theta_grid, theta_grid[buffermask]))
    #phi   = np.concatenate((phi, phi[buffermask] - 2. * np.pi))
    #theta = np.concatenate((theta, theta[buffermask]))
    Ghh        = np.concatenate((Ghh, Ghh[buffermask]))  
    Ghv        = np.concatenate((Ghv, Ghv[buffermask]))   
    Gvv        = np.concatenate((Gvv, Gvv[buffermask]))  
    Gvh        = np.concatenate((Gvh, Gvh[buffermask]))   

    buffermask = phi_grid < 0.1
    phi_grid   = np.concatenate((phi_grid, phi_grid[buffermask] + 2. * np.pi))
    theta_grid = np.concatenate((theta_grid, theta_grid[buffermask]))
    Ghh        = np.concatenate((Ghh, Ghh[buffermask]))  
    Ghv        = np.concatenate((Ghv, Ghv[buffermask]))   
    Gvv        = np.concatenate((Gvv, Gvv[buffermask]))  
    Gvh        = np.concatenate((Gvh, Gvh[buffermask]))   

    # Adding buffer for theta 
    # 
    # [Note]: This is done because we are getting several (exactly 1 in the
    # begginning and 3 in the end) NaN values in the grids after interpolation,
    # which happens due to the fact that interpolator does have enough
    # neighboring points on edges. Therefore, we are adding more points to the
    # left, while the right values are simply put to 0 (see explanation below). 
    buffermask = theta_grid < 0.1
    phi_grid   = np.concatenate((phi_grid, phi_grid[buffermask]))
    theta_grid = np.concatenate((theta_grid, -theta_grid[buffermask]))
    Ghh        = np.concatenate((Ghh, Ghh[buffermask]))  
    Ghv        = np.concatenate((Ghv, Ghv[buffermask]))   
    Gvv        = np.concatenate((Gvv, Gvv[buffermask]))  
    Gvh        = np.concatenate((Gvh, Gvh[buffermask]))   


    # Should be smaller than the buffer zone defined above
    res = 0.01 
    
    #print(np.min(phi_grid), phi_grid.max()) 
    #print(np.min(theta_grid), theta_grid.max()) 
    #exit() 

    #def interpolate_gain(theta, phi, gain): 
    #    return sp.interpolate.LinearNDInterpolator(list(zip(phi, theta)), gain)
    fhh        = sp.interpolate.LinearNDInterpolator(list(zip(phi_grid, theta_grid)), Ghh) #, fill_value=0)
    fhv        = sp.interpolate.LinearNDInterpolator(list(zip(phi_grid, theta_grid)), Ghv) #, fill_value=0)
    fvv        = sp.interpolate.LinearNDInterpolator(list(zip(phi_grid, theta_grid)), Gvv) #, fill_value=0)
    fvh        = sp.interpolate.LinearNDInterpolator(list(zip(phi_grid, theta_grid)), Gvh) #, fill_value=0)
    # Creating rectilinear grid 
    phi        = np.arange(0, 2. * np.pi + res, res)
    theta      = np.arange(0, np.max(theta_grid), res)
    phi, theta = np.meshgrid(phi, theta)  
    
    #print(np.min(phi), phi.max()) 
    #print(np.min(theta), theta.max()) 
    #exit() 
    #print(phi.shape)
    #print(phi_grid.shape)
    #Ghh = fhh((2.0, 0.2)) #np.meshgrid(phi, theta)).T 
    #Ghh = fhh(X, Y) #np.meshgrid(phi, theta)).T 
    
    #print(Ghh)
   
    #Ghh = fhh(np.meshgrid(phi, theta)).T 
    #Ghv = fhv(np.meshgrid(phi, theta)).T 
    #Gvv = fvv(np.meshgrid(phi, theta)).T 
    #Ghh = fhh(np.meshgrid(phi, theta)).T 
    
    # Interpolating the function and substituting the last NaN values in the
    # arrays with zeros, because they are not intersecting the Earth (once you
    # do the projection) 
    
    Ghh = np.nan_to_num(fhh(phi, theta).T, nan=0.0)  
    Gvv = np.nan_to_num(fvv(phi, theta).T, nan=0.0) 
    Gvh = np.nan_to_num(fvh(phi, theta).T, nan=0.0) 
    Ghv = np.nan_to_num(fhv(phi, theta).T, nan=0.0)

    phi, theta = phi.T, theta.T 

    # Getting back initial vectors and converting them into degrees 
    phi   = np.rad2deg(np.unique(phi[:, 0])) 
    theta = np.rad2deg(np.unique(theta[0, :])) 
    
    # Splitting the arrays into real and imaginary parts 
    G1h, G2h = np.real(Ghh), np.imag(Ghh)  
    G3h, G4h = np.real(Ghv), np.imag(Ghv)  
    G1v, G2v = np.real(Gvv), np.imag(Gvv) 
    G3v, G4v = np.real(Gvh), np.imag(Gvh)  
    

    #Ghh = np.nan_to_num(fhh(0., 0.01) 
    # (630, 158)
    #print(Ghh)
    #print(np.shape(Ghh)) 

    #print(Gvv)
    #print(Gvh)
    #print(Ghv)
    
    end_time_interpn = time.time() - start_time_interpn
    print(f"| Finished Interpolation in: {end_time_interpn:.2f}s") 

    #exit() 
    

    #theta, phi = uv_to_tp(u0, v0) 

    ## I am getting errors associated with interpolation later on, so it
    ## is better to save the grid as a ranged instead of meshgrid.  
    ##apat['u'], apat['v'] = np.meshgrid(u_values, v_values)
    ##apat['u'] = apat['u'].T
    ##apat['v'] = apat['v'].T

    #apat['u'] = u_values 
    #apat['v'] = v_values 

    ## Creating a grid from values 
    #apat['u_grid'], apat['v_grid'] = np.mgrid[u0:(u1 + dx):dx, v0:(v1 + dy):dy]  

    #print(f"{np.shape(apat['u_grid'])}")


    # Start value for the grid (top left corner)  
    #u1 = xcen + xs
    #v1 = ycen + ys

    #print(u1, v1) 

    #theta1, phi1 = uv_to_tp(u1, v1)  
    #print(theta1, phi1) 
    #
    #u2 = u1 + dx * (nx - 1)
    #v2 = v1 + dy * (ny - 1)
    #
    #theta2, phi2 = uv_to_tp(u2, v2)  
    #print(theta2, phi2) 
        

    # Getting the resulting dictionary  
    
    cimr["Gain"]['G1h']   = G1h 
    cimr["Gain"]['G2h']   = G2h 
    cimr["Gain"]['G3h']   = G3h 
    cimr["Gain"]['G4h']   = G4h 

    cimr["Gain"]['G1v']   = G1v 
    cimr["Gain"]['G2v']   = G2v 
    cimr["Gain"]['G3v']   = G3v 
    cimr["Gain"]['G4v']   = G4v 
    
    #cimr["Grid"]['u']     = u_grid #u_values #u0  
    #cimr["Grid"]['v']     = v_grid #v_values #v0   
    #cimr["Grid"]['u_cen'] = xcen #u_coordinate  
    #cimr["Grid"]['v_cen'] = ycen #v_coordinate   
    cimr["Grid"]['theta'] = theta #_grid  
    cimr["Grid"]['phi']   = phi #_grid   

        
    return cimr    


#def main(datadir, outdir, ref_tilt_ang_deg, downscale_factor, vsign, nu, nv, file_version, save_tp=True, save_uv=True):     
def main(datadir, outdir, file_version):     
    """
    Main method (entry point to the program)
    
    Parameters:
    -----------
    datadir: str or Path
        The path to the data directory where all beam files are located. 

    outdir: str or Path  
        The path to the output directory where to store all results of execution. 

    file_version: float  
        Version of the parsed files to be prodiced. 
    """

    # List of all beams files 
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
    parsed_dir = outdir.joinpath("parsed")
    if not parsed_dir.exists(): 
        print(f"| Creating parsed directory:\n{parsed_dir}")
        pathlib.Path(parsed_dir).mkdir()

    parsedfile_prefix = "CIMR-AP-FULL" 
    parsedfile_suffix = "XE"
    
    print(f"| {Fore.GREEN}Data Directory:{Fore.RESET}\n| {Fore.BLUE}{datadir}{Style.RESET_ALL}") 
    print(f"| {Fore.GREEN}Parsed Directory:{Fore.RESET}\n| {Fore.BLUE}{parsed_dir}{Style.RESET_ALL}") 
    # Main parsing loop 
    for band in apat_name_info.keys(): 
        
        for horn, half_spaces in apat_name_info[band][2].items(): 
            
            freq = apat_name_info[band][0] 
            pol  = apat_name_info[band][1] 
            
            # Object to be saved into file 
            cimr = {"Gain": {}, "Grid": {}, "Version": file_version} 
            
            for half_space in half_spaces: 

                # Reconstructing the full path to the file 
                if band == "L": 
                    infile = band + "-" + freq + "-" + pol + "-" + half_space + ".grd" 
                else: 
                    infile = band + str(horn) + "-" + freq + "-" + pol + "-" + half_space + ".grd" 
                
                infile = pathlib.Path(str(datadir) + "/" + band + "/" + infile)  
                print(f"| {Fore.YELLOW}------------------------------{Style.RESET_ALL}") 
                print(f"| {Fore.GREEN}Working with Input File: {infile.name}{Style.RESET_ALL}") 

                # Output filename 
                parsedfile_prefix = f"CIMR-AP-{half_space}" 
                parsedfile_suffix = "TP"
                outfile_tp = pathlib.Path(str(parsed_dir) + f"/{parsedfile_prefix}-" + band + horn + f"-{parsedfile_suffix}v{file_version}.h5")   
                
                if io.check_outfile_existance(outfile_tp) == True: 
                    continue 
                else: 
                    cimr = get_beamdata(infile, half_space, cimr) #_uv, cimr_tp)     
                    
                    print(f"| {Fore.BLUE}Saving Output File: {outfile_tp.name}{Style.RESET_ALL}") 
                    
                    with h5py.File(outfile_tp, 'w') as hdf5_file:
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

    # TODO: Create a parameter file that will take as input this info  

    # Getting the root of the repo 
    root_dir = io.find_repo_root() 

    # Params to be put inside parameter file 
    outdir  = pathlib.Path(f"{root_dir}/output").resolve()
    datadir = pathlib.Path(f"{root_dir}/dpr/AP").resolve() 

    if not pathlib.Path(outdir).exists(): 
        print(f"Creating output directory:\n{outdir}")
        pathlib.Path(outdir).mkdir()

    if not datadir.is_dir():
        raise FileNotFoundError(f"The directory '{datadir}' does not exist.")

    # Getting all beam paths inside dpr/AP 
    beamfiles_paths = datadir.glob("*/*")   

    file_version = 0.3 

    #main(datadir, outdir, ref_tilt_ang_deg, downscale_factor, vsign, nu, nv, file_version)    
    main(datadir, outdir, file_version)    
    
    end_time_tot = time.time() - start_time_tot
    print(f"| Finished Script in: {end_time_tot:.2f}s") 
    print(f"| ------------------------------") 

