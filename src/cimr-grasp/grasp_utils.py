import time 

import numpy as np 
import scipy as sp




def uv_to_tp(u,v): 
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

def generate_uv_grid(xcen, ycen, xs, ys, nx, ny, dx, dy) -> (np.ndarray, np.ndarray): 

    # Generating the grid 
    u0 = xcen + xs
    u1 = u0 + dx * (nx - 1)
    v0 = ycen + ys
    v1 = v0 + dy * (ny - 1)

    u_grid, v_grid = np.mgrid[u0:(u1 + dx):dx, v0:(v1 + dy):dy]  

    return u_grid, v_grid 


def construct_complete_gains(cimr: dict()) -> dict(): 

    cimr["temp"]['Ghh'] = cimr["Gain"]['G1h'] + 1j * cimr["Gain"]['G2h']
    cimr["temp"]['Ghv'] = cimr["Gain"]['G3h'] + 1j * cimr["Gain"]['G4h']
    cimr["temp"]['Gvv'] = cimr["Gain"]['G1v'] + 1j * cimr["Gain"]['G2v']
    cimr["temp"]['Gvh'] = cimr["Gain"]['G3v'] + 1j * cimr["Gain"]['G4v']

    return cimr 


def interp_gain(x, y, temp, X, Y, interp_method = "linear"):

    grid_points = np.vstack([X.ravel(), Y.ravel()]).T 

    temp = sp.interpolate.griddata(grid_points, temp, (x, y), method=interp_method) 

    #interp = sp.interpolate.LinearNDInterpolator(list(zip(x, y)), z)
    #Z = interp(X, Y)

    return temp  

def interp_beamdata_into_uv(cimr: dict()) -> dict(): 
    """
    Method to interpolate the u,v grid into the rectilinear u,v that corresponds to theta, phi. 
    """

    #print(cimr["Grid"].keys()) 
    #print(cimr["Gain"].keys()) 
    cimr["temp"]['Ghh'] = cimr["temp"]['Ghh'].flatten()  
    cimr["temp"]['Ghv'] = cimr["temp"]['Ghv'].flatten()  
    cimr["temp"]['Gvv'] = cimr["temp"]['Gvv'].flatten() 
    cimr["temp"]['Gvh'] = cimr["temp"]['Gvh'].flatten()  

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
    theta = np.linspace(0, 90, 100)
    phi   = np.linspace(0, 360, 100)
    theta_grid, phi_grid = np.meshgrid(theta, phi)   
    theta_grid, phi_grid = theta_grid.T, phi_grid.T 

    # Converting these values into the appropriate x, y values (downdraded u,v
    # grid), i.e., an appropriate cartesian representation 
    u, v = convert_tp_to_uv(theta_grid, phi_grid)  
    u, v = np.ravel(u), np.ravel(v)   


    # Interpolating 
    # No chunks 
    #for key in cimr[].keys(): 
    start_time_inter = time.time() 
    
    cimr["temp"]["Ghh"] = interp_gain(x = u, y = v, temp = cimr["temp"]["Ghh"], 
                                      X = cimr["temp"]["u_grid"],  
                                      Y = cimr["temp"]["v_grid"], 
                                      ).T  
                                      #X = U, 
                                      #Y = V).T  

    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Ghh in: {end_time_inter:.2f}s")

    cimr["Gain"]["G1h"], cimr["Gain"]["G2h"] = np.real(cimr["temp"]['Ghh']), np.imag(cimr["temp"]['Ghh'])  
    del cimr["temp"]['Ghh'] 


    # With chunks 
    # TODO: Put it into its own method 

    del cimr["temp"]['u_grid'] 
    del cimr["temp"]['v_grid'] 

    cimr["Grid"]['u']     = u   
    cimr["Grid"]['v']     = v    
    cimr["Grid"]['theta'] = theta    
    cimr["Grid"]['phi']   = phi    

    return cimr  

