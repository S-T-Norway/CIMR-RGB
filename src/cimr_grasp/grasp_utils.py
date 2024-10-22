"""

"""
import time 

import numpy as np 
import scipy as sp




def convert_uv_to_tp(u: float | np.ndarray, 
                     v: float | np.ndarray) -> (float | np.ndarray, float | np.ndarray): 
    """
    Converting (u,v) into (theta,phi) and returning the grid in degrees.   
    
    According to the GRASP manual, the relations between (u, v) and (theta,
    phi) coordinates are:  
    
    $$
    u=\\sin\\theta\\cos\\phi
    $$
    $$
    v=\\sin\\theta\\sin\\phi
    $$ 

    which makes up the unit vector to the field point as  
    $$
    \\hat{r} = \\left( u, v, \\sqrt{1 - u^2 - v^2} \\right)
    $$ 

    The reverse relations then are: 
    
    $$
    \\theta=\\arccos{\\sqrt{1-u^2-v^2}}
    $$
    $$
    \\phi = \\arctan\\left(\\frac{v}{u}\\right)
    $$ 

    where $\\phi$ is of [-180, 180] and $\\theta$ is [-90, 90] (in degrees).  

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
        


def convert_tp_to_uv(theta: float | np.ndarray, 
                     phi: float | np.ndarray) -> (float | np.ndarray, float | np.ndarray): 
    """
    Method converts the cartesian (u,v) coordinates into (theta, phi).  

    Parameters:
    -----------
    theta: float or ndarray  
        Theta angle value 
    phi  : float or ndarray 
        Phi angle value 

    Returns:
    --------
    u: float or ndarray 
        U coordinate in director cosine coordinate system 
    v: float or ndarray 
        V coordinate in director cosine coordinate system 
    """

    u = np.sin(theta) * np.cos(phi) 
    v = np.sin(theta) * np.sin(phi) 

    return u, v 


def get_max_index(G: np.ndarray) -> (int, int): 
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


def generate_uv_grid(xcen: float, 
                     ycen: float, 
                     xs:   float, 
                     ys:   float, 
                     nx:   int, 
                     ny:   int, 
                     dx:   float, 
                     dy:   float) -> (np.ndarray, np.ndarray): 
    """
    Returns ... .   
    
    Parameters:
    -----------
    xcen: float  

    ycen: float  

    xs  : float  

    ys  : float  

    nx  : int  

    ny  : int  

    dx  : float  

    dy  : float  

    Returns:
    --------
    u_grid : np.ndarray 

    v_grid : np.ndarray 

    """ 

    # Generating the grid 
    u0 = xcen + xs
    u1 = u0 + dx * (nx - 1)
    v0 = ycen + ys
    v1 = v0 + dy * (ny - 1)

     
    # [Note]: numpy's mgrid creates a grid which is a transpose of meshgrid.
    # Thus, we need to transpose the resulting grid as well.  
    u_grid, v_grid = np.mgrid[u0:(u1 + dx):dx, v0:(v1 + dy):dy]  
    u_grid, v_grid = u_grid.T, v_grid.T 

    return u_grid, v_grid 


def construct_complete_gains(cimr: dict()) -> dict(): 
    """
    Building the complex array and getting the index that corresponds to its
    maximum value. 
    
    Parameters: 
    -----------
    cimr: dict 
        Dictionary that contains beam data.  

    Returns:
    --------
    cimr: dict 
        Modified data dictionary.  
    
    """ 

    cimr["temp"]['Ghh'] = cimr["Gain"]['G1h'] + 1j * cimr["Gain"]['G2h']
    cimr["temp"]['Ghv'] = cimr["Gain"]['G3h'] + 1j * cimr["Gain"]['G4h']
    cimr["temp"]['Gvv'] = cimr["Gain"]['G1v'] + 1j * cimr["Gain"]['G2v']
    cimr["temp"]['Gvh'] = cimr["Gain"]['G3v'] + 1j * cimr["Gain"]['G4v']

    return cimr 


def interp_gain(X:      np.ndarray, 
                Y:      np.ndarray, 
                gain:   np.ndarray, 
                x_down: np.ndarray, 
                y_down: np.ndarray, 
                interp_method: str = "linear") -> np.ndarray:

    grid_points = np.vstack([X.ravel(), Y.ravel()]).T 

    gain = sp.interpolate.griddata(grid_points, gain, (x_down, y_down), method=interp_method) 

    return gain   


def interp_gain_in_chunks(gain: np.ndarray, 
                          n_rows: int, 
                          n_cols: int, 
                          x_lin: np.ndarray, 
                          y_lin: np.ndarray, 
                          X: np.ndarray, 
                          Y: np.ndarray, 
                          x_down: np.ndarray, 
                          y_down: np.ndarray,  
                          logger, 
                          num_chunks: int = 4, 
                          overlap_margin: float = 0.1, 
                          interp_method: str = "linear") -> np.ndarray: 
    """

    ... 

    To avoid discontinuities we define the overlap margin and chunk our code appropriately 
    the margine is 10% for now.  

    """

    gain_down = np.zeros_like(x_down, dtype=complex)  

    # doing the same for chunks 
    chunk_size_row = n_rows // num_chunks 
    chunk_size_col = n_cols // num_chunks 


    # We shift to left everything after the first chunk to properly stitch them together 
    chunk_shift_row = 0 
    chunk_shift_col = 0

    x_min_list = [None] * (num_chunks + 1)
    x_max_list = [None] * (num_chunks + 1)

    y_min_list = [None] * (num_chunks + 1)
    y_max_list = [None] * (num_chunks + 1) 

    # Creating a loop over all chunks + some leftover values 
    for i in range(0, num_chunks + 1, 1):  

        if i < num_chunks: 
            start_val_row  = i * chunk_size_row - chunk_shift_row
            end_val_row    = (i + 1) * chunk_size_row - chunk_shift_row
        else: 
            start_val_row  = i * chunk_size_row - chunk_shift_row
            end_val_row    = len(x_lin) #- 1
            
        for j in range(0, num_chunks + 1, 1): 

            if j < num_chunks:
            
                # Creating chunks from the original grid X, Y 
                start_val_col  = j * chunk_size_col - chunk_shift_col
                end_val_col    = (j + 1) * chunk_size_col - chunk_shift_col

            else: 
                start_val_col  = j * chunk_size_col - chunk_shift_col
                end_val_col    = len(y_lin) #- 1

            
            x_origin_chunk = X[start_val_row:end_val_row, start_val_col:end_val_col]
            y_origin_chunk = Y[start_val_row:end_val_row, start_val_col:end_val_col]
            gain_chunk     = gain[start_val_row:end_val_row, start_val_col:end_val_col]

            # The grid boundaries should stay the same regaraless of the amount of
            # chunks we choose, because we are downsampling the grid (making it
            # coarser; have less values). Thus, we can define the boundaries for
            # the first chunk in the following manner.

            # The issue here is that the values are not exactly matching, so some
            # small amount of points is lost. So, it makes sense to create a list
            # of values to retrieve later on
            if j == 0:  
                x_min_list[j] = x_origin_chunk.min() 
                abs_length = np.abs(x_lin[-1] - x_lin[0]) 
                x_max_list[j] = x_origin_chunk.max() - overlap_margin * chunk_size_row * abs_length / n_rows

            elif j == num_chunks: 
                x_min_list[j] = x_max_list[j-1] 
                x_max_list[j] = x_origin_chunk.max() 
            else: 
                x_min_list[j] = x_max_list[j-1]
                x_max_list[j] = x_origin_chunk.max() - \
                        overlap_margin * chunk_size_row * np.abs(x_lin[-1] - x_lin[0]) / n_rows 

            if i == 0: 
                y_min_list[i] = y_origin_chunk.min() 
                y_max_list[i] = y_origin_chunk.max() - \
                        overlap_margin * chunk_size_col * np.abs(y_lin[-1] - y_lin[0]) / n_cols 
            elif i == num_chunks: 
                y_min_list[i] = y_max_list[i-1] 
                y_max_list[i] = y_origin_chunk.max() 
            else:
                y_min_list[i] = y_max_list[i-1]
                y_max_list[i] = y_origin_chunk.max() - \
                        overlap_margin * chunk_size_col * np.abs(y_lin[-1] - y_lin[0]) / n_cols 

            # Now, since the resulting grid is non-rectilinear, the interpolated values are scattered 
            # around the downscaled grid. Therefore, we need to define the mask which will capture those
            # temperature values within the chunk of new grid. 
            mask = (x_down >= x_min_list[j]) * (x_down <= x_max_list[j]) * \
                   (y_down >= y_min_list[i]) * (y_down <= y_max_list[i])

            # The grid coordinates from the new/downscaled grid that will correspond to the temperature values 
            x_down_chunk_cropped = x_down[mask]
            y_down_chunk_cropped = y_down[mask]

            # Flatten the arrays to get the coordinates and temperature values as 1D arrays
            #original_grid_points_chunk = np.vstack([x_origin_chunk.ravel(), y_origin_chunk.ravel()]).T
            # Getting the chunked temperature values into appropriate format
            gain_chunk = gain_chunk.flatten()
            # Interpolating given temperature chunk (portion)
            start_time = time.perf_counter() 

            gain_down_chunk = interp_gain(X      = x_origin_chunk, 
                                          Y      = y_origin_chunk, 
                                          gain   = gain_chunk, 
                                          x_down = x_down_chunk_cropped, 
                                          y_down = y_down_chunk_cropped, 
                                          interp_method = interp_method
                                          )

            # Saving the interpolated portion into the main temperature array
            gain_down[mask] = gain_down_chunk
            end_time = time.perf_counter()  - start_time
            logger.info(f"i = {i}, j = {j} took: {end_time:0.2f}s")
        

            # shifting the array to the left 
            chunk_shift_col = int(chunk_shift_col + 2 * overlap_margin * chunk_size_col)

        # Updating the row element and starting anew for the column shift 
        chunk_shift_row = int(chunk_shift_row + 2 * overlap_margin * chunk_size_row)
        chunk_shift_col = 0

    return gain_down 


def interp_beamdata_into_uv(cimr:           dict(), 
                            logger, 
                            grid_res_phi:   float = 0.1, 
                            grid_res_theta: float = 0.1, 
                            chunk_data:     bool  = True, 
                            num_chunks:     int   = 4, 
                            overlap_margin: float = 0.1, 
                            interp_method:  str   = "linear") -> dict(): 
    """
    Method to interpolate the u,v grid into the rectilinear x,y (coarser
    cartesian system) that corresponds to theta, phi. 

    Parameters:
    -----------
    cimr: dict 
        Dictionary that contains beam data to be modified and returned. 

    grid_res_phi: float (default value = 0.1) 
        Resolution for the coarser grid  for phi component (in degrees). 

    grid_res_theta: float (default value = 0.1) 
        Resolution for the coarser grid  for theta component (in degrees). 

    chunk_data: bool (default value = True) 
        Whether to chunk data. 

    num_chunks: int (default value = 4)  
        Number of chunks to split the grid. 

    Returns:
    --------
    cimr: dict 
        Dictionary that contains beam data to be modified and returned. 
    """

    # New grid 
    grid_points_theta = int(90  / grid_res_theta) 
    grid_points_phi   = int(360 / grid_res_phi) 

    theta = np.linspace(0, 90,  grid_points_theta)
    phi   = np.linspace(0, 360, grid_points_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)   

    # Converting these values into the appropriate x, y values (downdraded u,v
    # grid), i.e., an appropriate cartesian representation 
    x_grid, y_grid = convert_tp_to_uv(theta_grid, phi_grid)  
    x, y = np.ravel(x_grid), np.ravel(y_grid)   

    # Interpolating 
    # [Note]: we iterate over the copy of the keys to prevent runtime error: 
    # 
    #     for key in cimr["temp"].keys():
    # RuntimeError: dictionary changed size during iteration 

    for key in list(cimr["temp"].keys()):  
        logger.info(f"Started processing: {key}")
        start_time_inter = time.perf_counter()  

        
        if chunk_data:
            cimr["temp"][key] = interp_gain_in_chunks(gain           = cimr["temp"][key], 
                                                      n_cols         = cimr["Grid"]["nx"], 
                                                      n_rows         = cimr["Grid"]["ny"],  
                                                      x_lin          = cimr["Grid"]["u"],  
                                                      y_lin          = cimr["Grid"]["v"], 
                                                      X              = cimr["Grid"]["u_grid"],  
                                                      Y              = cimr["Grid"]["v_grid"], 
                                                      x_down         = x,  
                                                      y_down         = y, 
                                                      num_chunks     = num_chunks, 
                                                      overlap_margin = overlap_margin, 
                                                      interp_method  = interp_method, 
                                                      logger         = logger
                                                    )
        else: 
            # No chunks 
            cimr["temp"][key] = cimr["temp"][key].ravel()  
            cimr["temp"][key] = interp_gain(X = cimr["Grid"]["u_grid"],  
                                            Y = cimr["Grid"]["v_grid"], 
                                            gain = cimr["temp"][key], 
                                            x_down = x, 
                                            y_down = y 
                                            ).T  

        cimr['temp'][key] = cimr['temp'][key].reshape(3600, 900) 

        if key == 'Ghh': 
            cimr["Gain"]["G1h"], cimr["Gain"]["G2h"] = np.real(cimr["temp"][key]), np.imag(cimr["temp"][key])  
        elif key == 'Ghv': 
            cimr["Gain"]["G3h"], cimr["Gain"]["G4h"] = np.real(cimr["temp"][key]), np.imag(cimr["temp"][key])  
        elif key == 'Gvv': 
            cimr["Gain"]["G1v"], cimr["Gain"]["G2v"] = np.real(cimr["temp"][key]), np.imag(cimr["temp"][key]) 
        elif key == 'Gvh': 
            cimr["Gain"]["G3v"], cimr["Gain"]["G4v"] = np.real(cimr["temp"][key]), np.imag(cimr["temp"][key])  

        del cimr["temp"][key] 

        end_time_inter = time.perf_counter()  - start_time_inter 
        logger.info(f"Finished with {key} in: {end_time_inter:.2f}s")


    # Clean-up 
    del cimr["temp"] 
    del cimr["Grid"]['u_grid'] 
    del cimr["Grid"]['v_grid'] 
    del cimr["Grid"]['dx'] 
    del cimr["Grid"]['dy'] 
    del cimr["Grid"]['nx'] 
    del cimr["Grid"]['ny'] 
    del cimr["Grid"]['xs'] 
    del cimr["Grid"]['ys']    
    del cimr["Grid"]['xcen'] 
    del cimr["Grid"]['ycen'] 

    # Saving resulting grids 
    cimr["Grid"]['x']     = x 
    cimr["Grid"]['y']     = y    
    cimr["Grid"]['theta'] = theta    
    cimr["Grid"]['phi']   = phi    

    return cimr  

