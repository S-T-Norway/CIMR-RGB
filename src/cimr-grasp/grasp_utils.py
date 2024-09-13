import time 

import numpy as np 
import scipy as sp




def convert_uv_to_tp(u,v): 
    """
    Converting (u,v) into (theta,phi) and returning the grid in radians.   
    
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
        


def convert_tp_to_uv(theta, phi) -> (np.ndarray, np.ndarray): 
    """
    Method converts the cartesian u,v coordinates into theta, pho.  

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

    #x_down = np.ravel(np.sin(theta_grid) * np.cos(phi_grid))
    #y_down = np.ravel(np.sin(theta_grid) * np.sin(phi_grid)) 
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


def generate_uv_grid(xcen, ycen, xs, ys, nx, ny, dx, dy) -> (np.ndarray, np.ndarray): 

    # Generating the grid 
    u0 = xcen + xs
    u1 = u0 + dx * (nx - 1)
    v0 = ycen + ys
    v1 = v0 + dy * (ny - 1)

     
    # [Note]: numpy's mgrid creates a grid which is a transpose of meshgrid.
    # Thus, we need to transpose the resulting grid as well.  
    u_grid, v_grid = np.mgrid[u0:(u1 + dx):dx, v0:(v1 + dy):dy]  
    u_grid, v_grid = u_grid.T, v_grid.T 

    #print(u_grid)
    #print(u_grid.T)

    return u_grid, v_grid 


def construct_complete_gains(cimr: dict()) -> dict(): 
    """
    Building the complex array and getting the index that corresponds to its
    maximum value. 
    """ 

    cimr["temp"]['Ghh'] = cimr["Gain"]['G1h'] + 1j * cimr["Gain"]['G2h']
    cimr["temp"]['Ghv'] = cimr["Gain"]['G3h'] + 1j * cimr["Gain"]['G4h']
    cimr["temp"]['Gvv'] = cimr["Gain"]['G1v'] + 1j * cimr["Gain"]['G2v']
    cimr["temp"]['Gvh'] = cimr["Gain"]['G3v'] + 1j * cimr["Gain"]['G4v']

    return cimr 


#def interp_gain(x_down, y_down, gain, X, Y, interp_method = "linear"):
def interp_gain(X, Y, gain, x_down, y_down, interp_method = "linear"):

    grid_points = np.vstack([X.ravel(), Y.ravel()]).T 

    gain = sp.interpolate.griddata(grid_points, gain, (x_down, y_down), method=interp_method) 

    return gain   


def interp_gain_in_chunks(gain: np.ndarray, n_rows: int, n_cols: int, 
                          x_lin: np.ndarray, y_lin: np.ndarray, 
                          X: np.ndarray, Y: np.ndarray, 
                          x_down: np.ndarray, y_down: np.ndarray,  
                          num_chunks: int = 4, overlap_margin: float = 0.1, 
                          interp_method: str = "linear") -> np.ndarray: 

    # Divide the total grid into 4 chunks (25%)
    #num_chunks = 4
    # To avoid discontinuities we define the overlap margin and chunk our code appropriately 
    # the margine is 10% for now 
    #overlap_margin = 0.1 

    #print(gain.shape)
    #print(n_rows) 
    #print(n_cols) 
    #print(X.shape)  

    #gain = np.nan_to_num(gain, nan=0.0) 

    #x_lin = np.unique(X[:, 0]) 
    #y_lin = np.unique(Y[0, :]) 
    #print(x_lin)
    #print(len(y_lin)) 

    #print(x_lin.shape) 
    #print(y_lin.shape) 
    #exit()

    gain_down = np.zeros_like(x_down, dtype=complex)  
    #print(np.shape(gain_down)) 

    # doing the same for chunks 
    chunk_size_row = n_rows // num_chunks 
    chunk_size_col = n_cols // num_chunks 
    #print(chunk_size_row) 

    # To avoid discontinuities we define the overlap margin and chunk our code appropriately 
    # the margine is 10% for now 
    #overlap_margin = 0.1 #0.1 #1 #0.01 

    # We shift to left everything after the first chunk to properly stitch them together 
    chunk_shift_row = 0 #2 * overlap_margin 
    chunk_shift_col = 0

    x_min_list = [None] * (num_chunks + 1)
    x_max_list = [None] * (num_chunks + 1)

    y_min_list = [None] * (num_chunks + 1)
    y_max_list = [None] * (num_chunks + 1) 
    #print(gain.shape) 
    #print(X.shape)
    #print(X.T)
    ##exit() 
    #X, Y = X.T, Y.T 

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
            
                #print(f"chunk_size_row = {chunk_size_row}")
                ##print(f"chunk_percent_to_subtract  = {overlap_margin * chunk_size_row}")
                #print(f"chunk_shift_col = {chunk_shift_col}")
                #print(f"chunk_size_row - chunk_shift_row = {chunk_size_row - chunk_shift_row}")
                
                # Creating chunks from the original grid X, Y 
                start_val_col  = j * chunk_size_col - chunk_shift_col
                end_val_col    = (j + 1) * chunk_size_col - chunk_shift_col

            else: 
                start_val_col  = j * chunk_size_col - chunk_shift_col
                end_val_col    = len(y_lin) #- 1

            #print(f"start_val_col = {start_val_col}, end_val_col = {end_val_col}")
            #print(f"start_val_row = {start_val_row}, end_val_row = {end_val_row}")
            
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
                #x_max_list[j] = x_origin_chunk.max() - overlap_margin * chunk_size_row * np.abs(x_lin[-1] - x_lin[0]) / n_rows
                x_max_list[j] = x_origin_chunk.max() - overlap_margin * chunk_size_row * abs_length / n_rows
                #print(abs_length)

            elif j == num_chunks: 
                x_min_list[j] = x_max_list[j-1] 
                x_max_list[j] = x_origin_chunk.max() 
            else: 
                x_min_list[j] = x_max_list[j-1]
                x_max_list[j] = x_origin_chunk.max() - overlap_margin * chunk_size_row * np.abs(x_lin[-1] - x_lin[0]) / n_rows 

            if i == 0: 
                y_min_list[i] = y_origin_chunk.min() 
                y_max_list[i] = y_origin_chunk.max() - overlap_margin * chunk_size_col * np.abs(y_lin[-1] - y_lin[0]) / n_cols 
            elif i == num_chunks: 
                y_min_list[i] = y_max_list[i-1] 
                y_max_list[i] = y_origin_chunk.max() 
            else:
                y_min_list[i] = y_max_list[i-1]
                y_max_list[i] = y_origin_chunk.max() - overlap_margin * chunk_size_col * np.abs(y_lin[-1] - y_lin[0]) / n_cols 

            #print(f"j = {j}, x_min_list = {x_min_list[j]}") 
            #print(f"j = {j}, x_max_list = {x_max_list[j]}") 
            #print(f"x_origin_chunk_max = {x_origin_chunk.max()}") 
     
            # Now, since the resulting grid is non-rectilinear, the interpolated values are scattered 
            # around the downscaled grid. Therefore, we need to define the mask which will capture those
            # temperature values within the chunk of new grid. 
            mask = (x_down >= x_min_list[j]) * (x_down <= x_max_list[j]) * (y_down >= y_min_list[i]) * (y_down <= y_max_list[i])
            #print(f"mask true len = {len(mask[mask !=False])}") 

            # The grid coordinates from the new/downscaled grid that will correspond to the temperature values 
            x_down_chunk_cropped = x_down[mask]
            y_down_chunk_cropped = y_down[mask]

            # Flatten the arrays to get the coordinates and temperature values as 1D arrays
            #original_grid_points_chunk = np.vstack([x_origin_chunk.ravel(), y_origin_chunk.ravel()]).T
            # Getting the chunked temperature values into appropriate format
            #gain_chunk = gain_chunk.ravel()
            gain_chunk = gain_chunk.flatten()
            # Interpolating given temperature chunk (portion)
            start_time = time.time()
            #gain_down_chunk = sp.interpolate.griddata(original_grid_points_chunk, gain_chunk, (x_down_chunk_cropped, y_down_chunk_cropped), method='linear')

            gain_down_chunk = interp_gain(X = x_origin_chunk, Y = y_origin_chunk, gain = gain_chunk, 
                                          x_down = x_down_chunk_cropped, y_down = y_down_chunk_cropped, 
                                          interp_method = interp_method)

            # Saving the interpolated portion into the main temperature array
            gain_down[mask] = gain_down_chunk
            end_time = time.time() - start_time
            print(f"| i = {i}, j = {j} took: {end_time:0.2f}s")
        

            # shifting the array to the left 
            chunk_shift_col = int(chunk_shift_col + 2 * overlap_margin * chunk_size_col)

        # Updating the row element and starting anew for the column shift 
        chunk_shift_row = int(chunk_shift_row + 2 * overlap_margin * chunk_size_row)
        chunk_shift_col = 0
    #exit() 

    return gain_down 

def interp_beamdata_into_uv(cimr: dict(), grid_res_phi: float = 0.1, 
                            grid_res_theta: float = 0.1, chunk_data: bool = True) -> dict(): 
    """
    Method to interpolate the u,v grid into the rectilinear u,v that corresponds to theta, phi. 
    """

    #print(cimr["Grid"].keys()) 
    #print(cimr["Gain"].keys()) 

    # Old grid 
    #U, V = generate_uv_grid(xcen = cimr["Grid"]['xcen'], 
    #                        ycen = cimr["Grid"]['ycen'], 
    #                        xs   = cimr["Grid"]['xs'], 
    #                        ys   = cimr["Grid"]['ys'], 
    #                        nx   = cimr["Grid"]['nx'], 
    #                        ny   = cimr["Grid"]['ny'], 
    #                        dx   = cimr["Grid"]['dx'], 
    #                        dy   = cimr["Grid"]['dy']
    #                        )  

    # New grid 
    #print(90/ grid_res_theta)
    grid_points_theta = int(90  / grid_res_theta) 
    grid_points_phi   = int(360 / grid_res_phi) 
    theta = np.linspace(0, 90,  grid_points_theta)
    phi   = np.linspace(0, 360, grid_points_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)   
    #theta_grid, phi_grid = theta_grid.T, phi_grid.T 

    # Converting these values into the appropriate x, y values (downdraded u,v
    # grid), i.e., an appropriate cartesian representation 
    x_grid, y_grid = convert_tp_to_uv(theta_grid, phi_grid)  
    x, y = np.ravel(x_grid), np.ravel(y_grid)   
    #print(np.shape(x_grid))
    #print(np.shape(x))
    #print(cimr["Grid"].keys())

    #print() 
    #print(cimr["Grid"]["u_grid"])
    #exit() 

    #chunk_data = False  
    num_chunks = 8 

    # Interpolating 
    #for key in cimr["temp"].keys(): 
    # Iterate over the copy of the keys to prevent runtime error: 
    # 
    #     for key in cimr["temp"].keys():
    # RuntimeError: dictionary changed size during iteration 
    for key in list(cimr["temp"].keys()):  
        print(f"| Started processing: {key}")
        start_time_inter = time.time() 
        
        if chunk_data:
            cimr["temp"][key] = interp_gain_in_chunks(gain = cimr["temp"][key], 
                                                    n_cols = cimr["Grid"]["nx"], 
                                                    n_rows = cimr["Grid"]["ny"],  
                                                    x_lin  = cimr["Grid"]["u"],  
                                                    y_lin  = cimr["Grid"]["v"], 
                                                    X      = cimr["Grid"]["u_grid"],  
                                                    Y      = cimr["Grid"]["v_grid"], 
                                                    x_down = x,  
                                                    y_down = y  
                                                    )
        else: 
            # No chunks 
            #cimr["temp"]['Ghh'] = cimr["temp"]['Ghh'].ravel()  
            #cimr["temp"]['Ghv'] = cimr["temp"]['Ghv'].ravel()  
            #cimr["temp"]['Gvv'] = cimr["temp"]['Gvv'].ravel() 
            #cimr["temp"]['Gvh'] = cimr["temp"]['Gvh'].ravel()  
            cimr["temp"][key] = cimr["temp"][key].ravel()  
            cimr["temp"][key] = interp_gain(X = cimr["Grid"]["u_grid"],  
                                            Y = cimr["Grid"]["v_grid"], 
                                            gain = cimr["temp"][key], 
                                            x_down = x, 
                                            y_down = y 
                                            ).T  


        if key == 'Ghh': 
            cimr["Gain"]["G1h"], cimr["Gain"]["G2h"] = np.real(cimr["temp"][key]), np.imag(cimr["temp"][key])  
        elif key == 'Ghv': 
            cimr["Gain"]["G3h"], cimr["Gain"]["G4h"] = np.real(cimr["temp"][key]), np.imag(cimr["temp"][key])  
        elif key == 'Gvv': 
            cimr["Gain"]["G1v"], cimr["Gain"]["G2v"] = np.real(cimr["temp"][key]), np.imag(cimr["temp"][key]) 
        elif key == 'Gvh': 
            cimr["Gain"]["G3v"], cimr["Gain"]["G4v"] = np.real(cimr["temp"][key]), np.imag(cimr["temp"][key])  

        del cimr["temp"][key] 

        end_time_inter = time.time() - start_time_inter 
        print(f"| Finished with {key} in: {end_time_inter:.2f}s")

    #start_time_inter = time.time() 
    #
    #cimr["temp"]["Ghh"] = interp_gain(x = u, y = v, temp = cimr["temp"]["Ghh"], 
    #                                  X = cimr["Grid"]["u_grid"],  
    #                                  Y = cimr["Grid"]["v_grid"], 
    #                                  ).T  
    #                                  #X = U, 
    #                                  #Y = V).T  
    #key = "Ghh"
    #arr0 = np.nan_to_num(interp_gain_in_chunks(gain = cimr["temp"][key], 
    #                      n_cols = cimr["Grid"]["nx"], 
    #                      n_rows = cimr["Grid"]["ny"],  
    #                      X = cimr["Grid"]["u_grid"],  
    #                      Y = cimr["Grid"]["v_grid"], 
    #                      x_down = u,  
    #                      y_down = v,  
    #                      ), nan=0.0) 
    #cimr["temp"][key] = np.nan_to_num(interp_gain_in_chunks(gain = cimr["temp"][key], 
    #                      n_cols = cimr["Grid"]["nx"], 
    #                      n_rows = cimr["Grid"]["ny"],  
    #                      X = cimr["Grid"]["u_grid"],  
    #                      Y = cimr["Grid"]["v_grid"], 
    #                      x_down = u,  
    #                      y_down = v,  
    #                      ), nan=0.0) 

    #end_time_inter = time.time() - start_time_inter 
    #print(f"| Finished with Ghh in: {end_time_inter:.2f}s")

    #cimr["Gain"]["G1h"], cimr["Gain"]["G2h"] = np.real(cimr["temp"]['Ghh']), np.imag(cimr["temp"]['Ghh'])  
    #arr1, arr2 = np.real(arr0), np.imag(arr0)   
    #print(cimr["Gain"]["G1h"][], cimr["Gain"]["G2h"][0:10]) 
    #print(arr1[arr1 != 0], arr2[arr2 != 0]) 
    #print(arr0[arr0 != 0]) 
    #exit() 
    #del cimr["temp"]['Ghh'] 


    # Clean-up 
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

