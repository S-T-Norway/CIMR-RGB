import numbers 

import numpy as np 
import scipy as sp 
import h5py 


# TODO: There are problems with interpolation in thsi code since the vlaues are
# completely different from Joe's values. 

# Function to recursively load HDF5 file into a dictionary
def load_hdf5_to_dict(hdf5_group):
    data_dict = {}
    for key, value in hdf5_group.items():
        if isinstance(value, h5py.Group):
            # If the value is a group, recursively call the function for the subgroup
            data_dict[key] = load_hdf5_to_dict(value)
        elif isinstance(value, h5py.Dataset):
            # If the value is a dataset, load the data into a NumPy array
            #data_dict[key] = np.array(value)
            data_dict[key] = value[()] 
        else:
            # If the value is an attribute, load it into the dictionary
            data_dict[key] = hdf5_group.attrs[key]
    return data_dict

def save_dict_to_hdf5(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, create a subgroup
            subgroup = hdf5_group.create_group(key)
            # Recursively call the function for the subgroup
            save_dict_to_hdf5(subgroup, value)
        elif isinstance(value, np.ndarray):
            # If the value is a NumPy array, create a dataset
            hdf5_group.create_dataset(key, data=value)
        #elif isinstance(value, np.isscalar(value)):
        elif isinstance(value, numbers.Number):
            # If the value is a NumPy array, create a dataset
            hdf5_group.create_dataset(key, data=value)
        else:
            # If the value is other type, create an attribute
            #print(key)
            hdf5_group.attrs[key] = value
            #print(hdf5_group.attrs[key])
            #hdf5_group.create_dataset(key, data=value)

#    def save_dict_group(group, data_dict):
#        for key, value in data_dict.items():
#            if isinstance(value, dict):
#                subgroup = group.create_group(key)
#                save_dict_group(subgroup, value)
#            elif isinstance(value, np.ndarray):
#                group.create_dataset(key, data=value)
#            else:
#                group.attrs[key] = value

def flip_apat_y(gb): 
    """
    Flip antenna pattern in Y axis. 

    Parameters: 
    -----------
    gb : dict 
        Antenna pattern to flip. 

    Returns: 
    -------- 
    o : dict 
        Flipped antenna pattern. 
    """

    o = gb.copy()

    o['hpol_copol'] = gb['hpol_copol'][:, ::-1].copy()
    o['hpol_cxpol'] = gb['hpol_cxpol'][:, ::-1].copy()
    o['vpol_copol'] = gb['vpol_copol'][:, ::-1].copy()
    o['vpol_cxpol'] = gb['vpol_cxpol'][:, ::-1].copy()

    return o 

def rot_apat(R, C, phi):
    """
    Applies rotation matrix. 
    """
    
    # converts degrees into radians and then takes cosine and sine 
    ct = np.cos(np.radians(phi))
    st = np.sin(np.radians(phi))

    Rp =  ct * R + st * C
    Cp = -st * R + ct * C

    return Rp, Cp


def rot_uv(b, az0, alt0, az1, alt1):
    """
    Rotate b (director cosine coordinates) from (az0,alt0) to (az1,alt1).
    Output is in director cosine coordinates.

    Steps:
        Rotate to 0 azimuth.
        Rotate about the y-axis to desired altitude.
        Rotate about z-axis from 0 azimuth to the desired azimuth.

    """

    altrot = 0.0
    azrot  = 0.0

    # Rotate about z-axis to 0 azimuth.
    rot0 = np.array([[ np.cos(np.radians(-az0)), -np.sin(np.radians(-az0)), 0 ],
                     [ np.sin(np.radians(-az0)),  np.cos(np.radians(-az0)), 0 ],
                     [ 0, 0, 1 ]])

    # Rotate about the y-axis to the new altitude alt1.
    dalt = alt1 - alt0
    rot1 = np.array([[ np.cos(np.radians(dalt)), 0, -np.sin(np.radians(dalt)) ],
                     [ 0, 1, 0 ],
                     [ np.sin(np.radians(dalt)), 0,  np.cos(np.radians(dalt)) ]])

    # Rotate about the z-axis to the new azimuth az1.
    daz = az1
    rot2 = np.array([[ np.cos(np.radians(daz)), -np.sin(np.radians(daz)), 0 ],
                     [ np.sin(np.radians(daz)),  np.cos(np.radians(daz)), 0 ],
                     [ 0, 0, 1]])

    # Create rotated cartesian vectors.
    br = np.dot(rot2, np.dot(rot1, np.dot(rot0, b)))

    # It should be (1501, 1501), i.e. square matrix 
    print(f"rot_uv: br shape: {np.shape(br)}")
    #print(br)

    return br

def rot_uv2(u, v, z, az0, alt0, az1, alt1):
    """
    Rotate b (director cosine coordinates) from (az0,alt0) to (az1,alt1).
    Output is in director cosine coordinates. 

    Steps:
        Rotate to 0 azimuth.
        Rotate about the y-axis to desired altitude.
        Rotate about z-axis from 0 azimuth to the desired azimuth.

    """

    u = np.array(u)
    v = np.array(v)
    z = np.array(z)

    #dco = np.array([np.array(u).flatten(), np.array(v).flatten(), np.array(z).flatten()])
    dco = np.array([u.flatten(), v.flatten(), z.flatten()])
    #print(np.shape(dco))
    #dco = np.array([u, v, z])

    dcr = rot_uv(dco, az0, alt0, az1, alt1)
    print(f"rot_uv2: dcr shape: {np.shape(dcr)}")
    print("")
    #print(type(dcr))
    #exit()

    #print(np.shape(u))
    #print(u)
    #exit()

    #print(u)
    ur = dcr[0].reshape(u.shape)
    vr = dcr[1].reshape(u.shape)
    zr = dcr[2].reshape(u.shape)
    print(f"rot_uv2: ur shape: {np.shape(ur)}")
    print("")
    # print(np.shape(ur))
    # print(np.shape(vr))
    # print(np.shape(zr))
    # exit()

    return ur, vr, zr

def trans_apat_fhs_bhs_fast(gf, gb, vsign): 
    """
    Transforms the antenna patterns from BHS basis into FHS polarization basis. 

    Parameters:
    -----------
    gf : dict  
        GRASP FHS  
    gb : dict  
        GRASP BHS 
    vsign : int 
        Sign for the vertical pattern (?) 

    Returns: 
    --------
    of : dict 
        Output FHS 
    ob : dict 
        Output BHS 
    """

    # Optionally set the v pol patterns from the h pol patterns.
    # TODO: Figure why? 
    if (abs(vsign) > 0.5): 
        gf['vpol_copol'] = -gf['hpol_copol'] * vsign
        gf['vpol_cxpol'] =  gf['hpol_cxpol'] * vsign

    # For the fhs just adjust signs.
    of = gf.copy()

    # Change signs on patterns to bring them into the (Rx,Cx,Ry,Cy)
    # convention. Note that copol and cxpol have already been swapped from
    # the GRASP convention. 
    # 
    # [**Note**]: This convention was introduced by Joe 
    of['hpol_copol'] =  gf['hpol_copol']
    of['hpol_cxpol'] = -gf['hpol_cxpol']
    of['vpol_copol'] = -gf['vpol_copol']
    of['vpol_cxpol'] = -gf['vpol_cxpol']


    #print(f"gb[1074,16] before flipping: {gb['hpol_cxpol'][1074, 16]}")

    # Flipping the antenna pattern in y axis (?)
    gb = flip_apat_y(gb)
    ob = gb.copy() 
    #print(gb.keys()) 
    #exit()

    #print(f"gb[1074,16] after flipping: {gb['hpol_cxpol'][1074, 16]}")
    # Define the GRASP xyz coordinates. Note that my u is opposite GRASP x.
    gb['x']  = -gb['u']
    gb['y']  =  gb['v']
    gb['z']  =  np.sqrt(1 - gb['x']**2 - gb['y']**2)

    # Polar angles for the xyz system.
    gb['theta'] = np.degrees(np.arccos(gb['z']))
    gb['phi']   = np.degrees(np.arctan2(gb['y'], gb['x']))

    # Compute the fhs angles in the bhs. At each gridpoint on the bhs grid these
    # are the corresponding spherical coordinate angles for the fhs. They can be
    # used to compute the (theta,phi) pol basis vectors wrt the fhs coordinate
    # system.
    gb['thetap']  = 180 - gb['theta']
    gb['phip']    = -gb['phi']

    # pra --- polarization rotation angle 
    gb['pra'] = 2*gb['phi'] + 180

    # Note that the function rot_apat takes as arguments (copol cxpol, polrot)
    # in that order where copol and cxpol are in the GRASP Ludwig-3 convention
    # and polrot is the clockwise rotation of (copol,cxpol).
    #
    # In this convention hpol_copol is GRASP cxpol while hpol_cxpol is GRASP
    # copol.
    #
    # Rotate GRASP bhs Ludwig-3 (co,cross) pol into fhs Ludwig-3 (co,cross).
    #
    # Note that gb.hpol_cxpol = GRASP co-pol and gb.hpol_copol = GRASP
    # cross-pol.
    #
    gb['hpol_cxpolr'], gb['hpol_copolr'] = rot_apat(gb['hpol_cxpol'],
                                                    gb['hpol_copol'],
                                                    gb['pra'])

    # Optionally use rotated GRASP (co,cross) pol to define vpol patterns.
    if (abs(vsign) > 0.5): 
        gb['vpol_copolr'] = -gb['hpol_copolr']*vsign
        gb['vpol_cxpolr'] =  gb['hpol_cxpolr']*vsign

    # Change signs on patterns to bring them into the (Rx, Cx, Ry, Cy) convention.
    ob['hpol_cxpol'] = -gb['hpol_cxpolr']
    ob['hpol_copol'] =  gb['hpol_copolr']
    ob['vpol_cxpol'] = -gb['vpol_cxpolr']
    ob['vpol_copol'] = -gb['vpol_copolr']

    # Store the pol rotation angle.
    ob['pra'] = gb['pra']
    of['pra'] = np.zeros_like(of['vpol_copol'])

    return of, ob 



def xe_to_uv(xi, eta, zeta, tilt_angle_deg):
    """
    Transform (xi,eta) coordinates to (u,v,z) coordinates for a given reflector
    tilt angle (off-nadir angle).
    
        eta  =  u
        xi   = -v
        zeta =  z
    """

    if zeta < -10:
        zeta = -np.sqrt(1 - xi**2 - eta**2)
    # [**Note**]: It should be elif otherwise will get an error: 
    # ValueError: The truth value of an array with more than one element is ambiguoug. 
    elif zeta > 10:
        zeta = np.sqrt(1 - xi**2 - eta**2)

    alt_uv = 90 + tilt_angle_deg
    alt_xe = 90

    u, v, z = rot_uv2(eta, -xi, zeta, 0.0, alt_xe, 0.0, alt_uv)

    u = np.real(u)
    v = np.real(v)
    z = np.real(z)

    return u, v, z

def uv_to_xe(u, v, z, tilt_angle_deg):
    """
    Transform (u,v) coordinates to (xi,eta,zeta) coordinates for a given reflector tilt angle (off-nadir angle).

      eta  =  u
      xi   = -v
      zeta =  z

    """
    print(z)

    if z < -10:
        z = -np.sqrt(1 - u**2 - v**2)
    # In the original code it is just if 
    # [**Note**]: It should be elif otherwise will get an error: 
    # ValueError: The truth value of an array with more than one element is ambiguoug. 
    elif z > 10:
        z = np.sqrt(1 - u**2 - v**2)

    alt_uv = 90 + tilt_angle_deg
    alt_xe = 90

    #print(np.shape(eta))
    #exit()
    
    eta, xi, zeta = rot_uv2(u, v, z, 0.0, alt_uv, 0.0, alt_xe)

    print(f"uv_to_xe: eta shape: {np.shape(eta)}")
    #exit()

    xi   = -np.real(xi)
    eta  =  np.real(eta)
    zeta =  np.real(zeta)

    return xi, eta, zeta

def uv_to_fs(u, v, z):
    """
    Transform (u,v,z) coordinates to full-space (ub,vb).
    """

    print("uv_to_fs")
    ifhs = np.array(np.where(z < 0)).flatten()  
    ibhs = np.array(np.where(z >= 0)).flatten()  
    #print(np.shape(ifhs))
    #print(np.shape(u))
    #print(np.shape(z))
    #print(v[ibhs])
    #print(type(ibhs))
    print(f"uv_to_fs: ibhs is: {ibhs}")
    print(f"uv_to_fs: ifhs is: {ifhs}")
    #exit() 
    #if (ibhs.size != 0): 
    #    print(ibhs)
    #exit() 

    # condition: Fix up z.
    # [**Note**]: ibhs is empty array (for some reason) so need to add this
    #print(ibhs.size)
    #if ibhs.size == 0: 
    #    print("Array is empty")
    if ibhs.size != 0: 
        print("ibhs is not empty, calculating z[ibhs]")
        z[ibhs] =  np.sqrt(1 - u[ibhs]**2 - v[ibhs]**2)

    if ifhs.size != 0: 
        print("ifhs is not empty, calculating z[ifhs]")
        z[ifhs] = -np.sqrt(1 - u[ifhs]**2 - v[ifhs]**2)

    r2 = u**2 + v**2
    r = np.sqrt(r2)
    z2 = 1 - r2
    bind = np.where(r2 > 1 - 1e-9)
    z[bind] = np.nan

    phi = np.arctan2(v, u) * 180 / np.pi
    theta = np.full_like(phi, np.nan)

    #theta[ifhs] = np.arcsind(r[ifhs])
    #theta[ibhs] = 180 - np.arcsind(r[ibhs])
    theta = 180 - np.arccos(z)

    theta[bind] = 90.0
    xb = theta * np.cos(np.radians(phi))
    eb = theta * np.sin(np.radians(phi))
    xs = np.sin(np.radians(theta / 2)) * np.cos(np.radians(phi))
    es = np.sin(np.radians(theta / 2)) * np.sin(np.radians(phi))

    return xb, eb, theta, phi



def apat_normalize(a): 
    """Normalizes the input antenna pattern by removing the common factor."""

    n = a.copy()

    fact = 1 / np.sqrt(4 * np.pi)

    n['hpol_copol'] = a['hpol_copol'] * fact
    n['hpol_cxpol'] = a['hpol_cxpol'] * fact
    n['vpol_copol'] = a['vpol_copol'] * fact
    n['vpol_cxpol'] = a['vpol_cxpol'] * fact

    return n

def apat_interp(u, v, x, ui, vi):

    x_amp2 = np.abs(x) ** 2

    # print("Debug #1")
    # print(np.shape(u))
    # print(np.shape(ui))

    # print(u[0])
    # print(u[1])
    # print(u[5])

    # Getting an error: 
    # ValueError: The points in dimension 1 must be strictly ascending or
    # descending. Therefore, I sort the values first. 
    # TODO: Figure out whether this approach is correct or not. Interpolate the
    # complex function.
    #sort_indices = np.argsort(u)
    #print(np.shape(sort_indices))
    #print(u[1,1], sort_indices[1,1])
    #exit()
    # u_sorted = u[sort_indices]
    # v_sorted = v[sort_indices]
    # x_sorted = x[sort_indices]
    #u_sorted = np.take_along_axis(u, sort_indices, axis=0)
    #v_sorted = np.take_along_axis(v, sort_indices, axis=0)
    #x_sorted = np.take_along_axis(x, sort_indices, axis=0)

    # Apply the same permutation to ui and vi
    #sort_indices_ui = np.argsort(ui)
    #ui_sorted = np.take_along_axis(ui, sort_indices_ui, axis=0)
    #vi_sorted = np.take_along_axis(vi, sort_indices_ui, axis=0)
    #combined_indices = np.argsort(np.argsort(np.column_stack((vi, ui))))
    #ui_sorted = np.take_along_axis(ui, combined_indices, axis=0)
    #vi_sorted = np.take_along_axis(vi, combined_indices, axis=0)

    # Sort the input points based on u and v
    #sort_indices = np.lexsort((v, u))
    #u_sorted = np.take_along_axis(u, sort_indices, axis=0)
    #v_sorted = np.take_along_axis(v, sort_indices, axis=0)
    #x_sorted = np.take_along_axis(x, sort_indices, axis=0)

    ## Combine ui and vi and use the combined values for sorting
    #combined_uv = np.column_stack((vi, ui))
    #combined_indices = np.lexsort((combined_uv[:, 0], combined_uv[:, 1]))

    ## Use the sorted indices to reorder ui and vi
    #ui_sorted = np.take_along_axis(ui, combined_indices, axis=0)
    #vi_sorted = np.take_along_axis(vi, combined_indices, axis=0)

    #print(np.shape(u_sorted))
    #exit()

    #x_amp2 = np.abs(x_sorted) ** 2

    #print(ui[1], ui[2], ui[3])
    #print(vi[1], vi[2], vi[3])

    #
    # [**Note**]: In Joe's code arrays have the following shapes: 
    # u           : (2501, 2501) 
    # ui          : (1513332, 1) or (0, 1) 
    # x_int       : (1513332, 1)
    # n.hpol_copol: (1501, 1501)  
    # 
    # However, the sp.intrepolate.interpn gives an error when u and v are
    # two-dimensional: 
    # 
    # ValueError: The points in dimension 1 must be strictly ascending or
    # descending.  
    # 
    # That is why u and v were kept 1D:  
    # u         : (2501, )
    # ui        : (15133332, )  
    # x_int     : (1513332, )
    # x_amp2_int: (1513332, )
    # 
    # In MATLAB, when using interpn with complex input data, the output is
    # automatically converted to real values by default. This behavior is
    # different from scipy.interpolate.interpn in Python, which can handle
    # complex interpolation directly.
    # 


    #print(f"x[0]: {x[0]}")
    #print(f"x[1]: {x[1]}")
    #print(f"x[2]: {x[2]}")
    #print(f"x[3]: {x[3]}")
    #exit() 

    #x_int = sp.interpolate.interpn((u, v), x, np.column_stack((ui, vi)),
    #                               method='linear', bounds_error=False,
    #                               fill_value=np.nan) 
    #print(x_int)
    #print(f"x_int[0]: {x_int[0]}")
    #print(f"x_int[1]: {x_int[1]}")
    #print(f"x_int[2]: {x_int[2]}")
    #print(f"x_int[3]: {x_int[3]}")


    # Check for NaN values
    #nan_indices = np.isnan(x_int)

    ## Check if there are any NaN values
    #has_nan = np.any(nan_indices)

    ## Print the result
    #if has_nan:
    #    print("Array contains NaN values.")
    #else:
    #    print("Array does not contain NaN values.")
    #exit()

    print(u.shape) 
    print(ui.shape) 
    x_int = np.real(sp.interpolate.interpn((u, v), x, np.column_stack((ui, vi)),
                                   method='linear', bounds_error=False,
                                   fill_value=np.nan))  
    #x_int = sp.interpolate.interpn((u, v), x, np.column_stack((ui, vi)),
    #                               method='linear', bounds_error=False,
    #                               fill_value=np.nan)  
    #print(type(x_int[4]))
    #exit() 
    #print(f"x_int[0]: {x_int[0]}")
    #print(f"x_int[1]: {x_int[1]}")
    #print(f"x_int[2]: {x_int[2]}")
    #print(f"x_int[3]: {x_int[3]}")
    #exit()

    # Interpolate the squared amplitude of the complex function.
    x_amp2_int = sp.interpolate.interpn((u, v), x_amp2, 
                                    np.column_stack((ui, vi)),
                                    method='linear', bounds_error=False,
                                    fill_value=np.nan)
    print(f"AMP {x_amp2_int.shape}") 
    print(f"AMP {x_amp2_int[4]}") 
    print(f"AMP {type(x_amp2_int[4])}") 
    #x_amp2_int = sp.interpolate.interpn((u, v), x_amp2, 
    #                                np.column_stack((ui, vi)),
    #                                method='linear', bounds_error=False,
    #                                fill_value=np.nan) 
    #print(f"x_amp2_int shape is: {np.shape(x_amp2_int)}")
    #exit()
    # Interpolate the complex function.
    #print(ui_sorted[0])
    #print(ui_sorted[1])
    #print(ui_sorted[3])
    #print(ui_sorted[9])

    #print(x_sorted[0])
    #x_int = sp.interpolate.interpn((u_sorted, v_sorted), x_sorted, 
    #                    np.column_stack((ui_sorted, vi_sorted)),
    #                    method='linear', bounds_error=False, fill_value=np.nan)

    ## Interpolate the squared amplitude of the complex function.
    #x_amp2_int = sp.interpolate.interpn((u_sorted, v_sorted), x_amp2,  
    #                     np.column_stack((ui, vi)),
    #                     method='linear', bounds_error=False, fill_value=np.nan)


    #if 0:
    #    # Squared amplitude of the interpolation.
    #    x_int_amp2 = np.abs(x_int) ** 2
    #    out = x_int * np.sqrt(x_amp2_int / x_int_amp2)

    #    # Do not do an amplitude modification where the amplitude of the interpolation is zero.
    #    bind = np.where(x_int_amp2 < 1e-40)
    #    out[bind] = x_int[bind]

    #if 1:
    #    rho = np.sqrt(x_amp2_int)
    #    phi = np.angle(x_int)
    #    out = rho * np.exp(1j * phi)
    
    rho = np.sqrt(x_amp2_int)
    phi = np.angle(x_int)
    
    # 
    # The piece of code in MATLAB is: 
    # out = complex(rho.*cos(phi), rho.*sin(phi))
    # which gives only real part of the complex array, while python analog
    # gives complex, so need to convert it to real. The code below gives the
    # complex data but where 0*j, so conversion to real does nothing wrong. 
    # 
    # [**Note**]: The values of out is vastly different from what Joe gets with
    # his code. 
    # 
    out = rho * np.exp(1j * phi) 
    #print(type(out[4]))
    # 
    # The following two codes are equivalent: 
    # out = np.real(rho * np.exp(1j * phi)) 
    # print(out)
    # print(out[0])
    # print(out[1])
    # print(out[2])
    # out = rho * np.cos(phi) + 1j * rho * np.sin(phi) 
    # print(out)
    # print(out[0])
    # print(out[1])
    # print(out[2])

    #out = rho * np.exp(1j * phi) 

    #print(f"apat_interp: out shape: {np.shape(out)}")
    #print(out)
    #exit()
    #exit()

    #out = x_amp2_int 
    # TODO: By comparing first several values, it turns out that scipy interon
    # gives (almost) the same values. See inside the grasp-rgb-ultimate/processed dir  
    # inside the C Horn 01, that is what I compared these to.  
    x_int = sp.interpolate.interpn((u, v), x, np.column_stack((ui, vi)),
                                   method='linear', bounds_error=False,
                                   fill_value=np.nan)  


    print(f"x_int[0] {x_int[0]}")
    print(x_int[1])
    print(x_int[2])
    print(x_int[3])
    print(x_int[4])
    print(x_int[5])
    print(x_int[6])
    
    #print(out[4])
    #print(x_int[4])
    #print(out[5])
    #print(x_int[5])
    #print(out[40])
    #print(x_int[40])
    out = x_int
    print(out[1])

    return out


def apat_tilt(fhs, bhs, tilt_ang_deg, u0, u1, v0, v1, ihs, dtheta, dphi, nu, nv): 
    """
    Tilts antenna pattern files. 

    Arguments
    ---------


    Returns 
    -------
    """

    n  = {} 
    s  = {}
    ut = {}

    # nu = -1 by inout value (for some reason Joe made it -1)
    # so, in principle, this if/else statement can be ignored 
    #if (nu < 0): 
    #    
    #    # Get the reflector boresight (u,v,z).
    #    
    #    # u0 is the start of the grid in u 
    #    # u1 is the end of the grid in u 
    #    # v0 and v1 are the same  
    #    ut['u0'] = u0;
    #    ut['u1'] = u1;
    #    ut['v0'] = v0;
    #    ut['v1'] = v1;
    #    print(f"apat_tilt: ut.u0: {ut['u0']}")
    #    print(f"apat_tilt: ut.u1: {ut['u1']}")
    #    print(f"apat_tilt: ut.v0: {ut['v0']}")
    #    print(f"apat_tilt: ut.v1: {ut['v1']}")

    #    # nu is the amount of data points for u coordinate 
    #    # nv is the amount of data points for v coordinate 
    #    ut['nu'] = int(np.ceil((ut['u1'] - ut['u0']) / fhs['du'] + 1));
    #    ut['nv'] = int(np.ceil((ut['v1'] - ut['v0']) / fhs['dv'] + 1));
    #    print(f"apat_tilt: ut.nu: {ut['nu']}")
    #    print(f"apat_tilt: ut.nv: {ut['nv']}")


    #    ut['du'] = (ut['u1'] - ut['u0']) / (ut['nu'] - 1);
    #    ut['dv'] = (ut['v1'] - ut['v0']) / (ut['nv'] - 1);


    #    print(f"apat_tilt: ut.du: {ut['du']}")
    #    print(f"apat_tilt: ut.dv: {ut['dv']}")
    #    
    #    #exit()

    #    #print("")

    #    #print(f"apat_tilt: ut.v0: {ut['v0']}")
    #    #print(f"apat_tilt: ut.v1: {ut['v1']}")

    #    #print(f"The shape of ut.du is {np.shape(ut['du'])}")
    #    #print(f"The shape of ut.dv is {np.shape(ut['dv'])}")
    #    #exit()

    #else: 

    # This part correspond to the BHS part in the code. Basically, Joe
    # defines custom grid size: nu, nv = (301, 251) 

    # If (nu,nv) specified adjust the uv range to yield specified grid dimensions.
    uc = 0.5 * (u0 + u1);
    vc = 0.5 * (v0 + v1);

    # Find the center gridpoint in the untilted frame using the location of the maximum power.
    #[cdum mind] = nanmax(abs(fhs.hpol_copol(:)).^2+abs(fhs.hpol_cxpol(:)).^2+abs(fhs.vpol_copol(:)).^2+abs(fhs.vpol_cxpol(:)).^2);
    print(f"apat_tilt: fhs.hpol_cxpol: {np.shape(fhs['hpol_cxpol'])}")
    print(f"apat_tilt: fhs.hpol_cxpol: {np.shape(fhs['hpol_cxpol'].flatten())}")
    #cdum, mind = np.nanmax(np.abs(fhs['hpol_copol'].flatten())**2 +
    #                       np.abs(fhs['hpol_cxpol'].flatten())**2 +
    #                       np.abs(fhs['vpol_copol'].flatten())**2 +
    #                       np.abs(fhs['vpol_cxpol'].flatten())**2) 
    mind = np.nanargmax(np.abs(fhs['hpol_copol'].flatten())**2 +
                        np.abs(fhs['hpol_cxpol'].flatten())**2 +
                        np.abs(fhs['vpol_copol'].flatten())**2 +
                        np.abs(fhs['vpol_cxpol'].flatten())**2) 
    print(mind)

    print(fhs.keys())
    print(np.shape(fhs['u_grid'].flatten()))
    print(np.shape(fhs['u_grid'].flatten()[:mind]))
    print(f"apat_tilt: {fhs['u_grid'].flatten()[mind]}")
    #fhs['u'], fhs['v'] = np.mgrid[fhs['u0']:(fhs['u1'] + fhs['du']):fhs['du'], 
    #                              fhs['v0']:(fhs['v1'] + fhs['dv']):fhs['dv']] 

    uc, vc,  zp = xe_to_uv(-fhs['v_grid'].flatten()[mind],
                            fhs['u_grid'].flatten()[mind], -100, tilt_ang_deg)
    

    up, vp1, zp = xe_to_uv(-fhs['v_grid'].flatten()[mind],
                            fhs['u_grid'].flatten()[mind] + fhs['du'], -100,
                            tilt_ang_deg)
    print(fhs['du'])

    ut['du'] = up - uc
    ut['dv'] = fhs['dv']


    print(f"apat_tilt: up: {up}")
    print(f"apat_tilt: ut.du: {ut['du']}")
        

    ut['u0'] = uc - (nu - 1) / 2 * ut['du']
    ut['u1'] = uc + (nu - 1) / 2 * ut['du']
    ut['v0'] = vc - (nv - 1) / 2 * ut['dv']
    ut['v1'] = vc + (nv - 1) / 2 * ut['dv']

    print("")
    
    print(f"apat_tilt: uc: {uc}")
    print(f"apat_tilt: vc: {vc}")

    print("")

    print(f"apat_tilt: nu: {nu}")
    print(f"apat_tilt: nv: {nv}")

    print("")

    print(f"apat_tilt: ut.u0: {ut['u0']}")
    print(f"apat_tilt: ut.u1: {ut['u1']}")
    print(f"apat_tilt: ut.v0: {ut['v0']}")
    print(f"apat_tilt: ut.v1: {ut['v1']}")

    # nu is the amount of data points for u coordinate 
    # nv is the amount of data points for v coordinate 
    ut['nu'] = int(np.ceil((ut['u1'] - ut['u0']) / ut['du'] + 1))  
    ut['nv'] = int(np.ceil((ut['v1'] - ut['v0']) / ut['dv'] + 1))  
    
    print("")
    
    print(f"apat_tilt: ut.nu: {ut['nu']}")
    print(f"apat_tilt: ut.nv: {ut['nv']}")

    #u_vals = np.arange(ut['u0'], ut['u1'] + ut['du'], ut['du']) 
    #v_vals = np.arange(ut['v0'], ut['v1'] + ut['dv'], ut['dv']) 

    # This seems to be a python/numpy analog to matlab ndgrid 
    # [**Note**]: Since python removes the last element of the array, I add
    # du/dv to account for that, otherwise will get shape (1500, 1500) instead
    # of (1501, 1501)
    #ut['u'], ut['v'] = np.mgrid[ut['u0']:ut['u1']:ut['du'], 
    #                            ut['v0']:ut['v1']:ut['dv']] 
    ut['u'], ut['v'] = np.mgrid[ut['u0']:(ut['u1'] + ut['du']):ut['du'], 
                                ut['v0']:(ut['v1'] + ut['dv']):ut['dv']] 


    # The resulting array is (1501, 1502) while it should be (1501, 1501)
    # so need to remove the last value. But it should be done only for the 
    # nu < 0 
    if nu < 0: 
        ut['u'] = ut['u'][:,:-1]
        ut['v'] = ut['v'][:,:-1]

    #print(ut['v'][1500,1499], ut['v'][1500,1501])

    print(f"apat_tilt: ut.u shape: {np.shape(ut['u'])}")
    print(f"apat_tilt: ut.v shape: {np.shape(ut['v'])}")
    #print(np.shape(u_vals))
    #print(np.shape(v_vals))
    #exit()
    #ut['u'], ut['v'] = np.meshgrid(u_vals, v_vals) 
    #ut['u'], ut['v'] = ut['u'].T, ut['v'].T 

    #ut['u'], ut['v'] = np.meshgrid(np.arange(ut['u0'], ut['u1'] + ut['du'], ut['du']), 
    #                               np.arange(ut['v0'], ut['v1'] + ut['dv'], ut['dv']))

    ut['r2'] = ut['u']**2 + ut['v']**2

    print("")
    print(f"apat_tilt: ut.r2 shape: {np.shape(ut['r2'])}")
    print(f"apat_tilt: ut.r2: {ut['r2'][0,250]}")

    #print(ut['u'])
    ut['xi'], ut['eta'], ut['zeta'] = uv_to_xe(ut['u'], ut['v'], 100 * ihs, tilt_ang_deg)
    #print("Debug infinite")
    print(f"apat_tilt: xi shape: {np.shape(ut['xi'])}")
    print(f"apat_tilt: xi(301,251): {ut['xi'][300,250]}")
    print("")
    print(f"apat_tilt: eta shape: {np.shape(ut['eta'])}")
    print(f"apat_tilt: eta(301,251): {ut['eta'][300,250]}")
    print("")
    print(f"apat_tilt: zeta shape: {np.shape(ut['zeta'])}")
    print(f"apat_tilt: zeta(301,251): {ut['zeta'][300,250]}")
    print(f"")
    

    # Below works for the full beam pattern not BHS 
    # Interpolate from the FHS. 
    # the array in Joe's code is (1513332, 1), and if I do not flatten mine
    # here I get (2, 1513332). Also, need to transpose it  
    gind = np.where((ut['zeta'].flatten() < 0.0) & (ut['r2'].flatten() < 1.0))
    gind = np.array(gind).T.flatten() 

    #print(f"apat_tilt: gind: {gind}")
    print(f"apat_tilt: gind shape: {np.shape(gind)}")
    print(f"apat_tilt: gind(1): {(gind[0])}")
    print(f"apat_tilt: gind(50): {(gind[49])}")

    # n.hpol_copol: (1501, 1501) = 2 253 001
    n['hpol_copol'] = np.full_like(ut['eta'], np.nan, dtype=complex).flatten()  
    n['hpol_cxpol'] = np.full_like(ut['eta'], np.nan, dtype=complex).flatten() 
    n['vpol_copol'] = np.full_like(ut['eta'], np.nan, dtype=complex).flatten() 
    n['vpol_cxpol'] = np.full_like(ut['eta'], np.nan, dtype=complex).flatten() 

    print(f"apat_tilt: n.hpol_copol shape: {np.shape(n['hpol_copol'])}")
    
    #print(type(gind))

    #print(f"apat_tilt: n.hpol_cxpol(gind): {n['hpol_cxpol'][gind]}")

    #print(fhs['u'][0])

    #print(fhs['u'][1])
    #print(fhs['v'][0])
    #print(fhs['v'][1])
    #exit()

    #print(f" The shape of ut[eta] is {np.shape(ut['eta'])}")
    #print(gind)
    #exit()

    #print(ut['eta'][gind])
    #ut['eta'] = ut['eta'].reshape(gind.shape)
    #ut['xi']  = ut['xi'].reshape(gind.shape)
    #print(f"{np.shape(ut['eta'])}")
    # 
    #print("")
    #exit()
    # In Joe's code this funsion takes in ut.eta[gind], which somehow reshapes
    # this array from (1501, 1501) into (15133332, 1). I am doing the same thing
    # with reshaping.  
    # 
    # Basically the operation a(b) indicates indexing or subscripting. The
    # result will be a new array containing the elements of a at the positions
    # specified by the indices in b.
    #
    # Example in MATLAB
    # a = rand(1501, 1501);  % Create a random 1501x1501 array
    # b = randi([1, numel(a)], 15133332, 1);  % Create random indices
    # 
    # result = a(b);  % Indexing operation
    # 
    #  'result' will be a column vector containing the elements of 'a' at
    #  positions specified by 'b'
    # disp(size(result));
    # 
    # Python equivalent code

    #ut['eta'] = ut['eta'].flatten()[gind.flatten()]
    #ut['xi']  = ut['xi'].flatten()[gind.flatten()]
    
    # Flattened versions 
    ut_eta = ut['eta'].flatten()
    ut_xi  = ut['xi'].flatten()
    
    #n['hpol_copol'] = n['hpol_copol'].flatten()[gind.flatten()]
    #n['hpol_cxpol'] = n['hpol_cxpol'].flatten()[gind.flatten()]
    #n['vpol_copol'] = n['vpol_copol'].flatten()[gind.flatten()]
    #n['vpol_cxpol'] = n['vpol_cxpol'].flatten()[gind.flatten()]
    
    #n['hpol_copol'] = n['hpol_copol'].reshape(-1, 1)
    #n['hpol_cxpol'] = n['hpol_cxpol'].reshape(-1, 1) 
    #n['vpol_copol'] = n['vpol_copol'].reshape(-1, 1) 
    #n['vpol_cxpol'] = n['vpol_cxpol'].reshape(-1, 1) 
    #gind = gind.flatten() 

    # Basically the following calls are combining the fhs and bhs parts into
    # one grid via interpolation 

    print("Started Interpolation for FHS")
    #n['hpol_copol'][gind] = apat_interp(fhs['u'], fhs['v'],
    #                                    fhs['hpol_copol'], 
    #                                    ut['eta'][gind], -ut['xi'][gind])

    # The values of this interpolation and the one in matlab code is completely
    # different 
    print(gind.size)


    #x = apat_interp(fhs['u'], fhs['v'], fhs['hpol_copol'], ut_eta[gind], -ut_xi[gind])
    #print(x)
    #x = np.full_like(ut['eta'], np.nan, dtype=complex).flatten() 
    #x[gind] = apat_interp(fhs['u'], fhs['v'], fhs['hpol_copol'], ut_eta[gind], -ut_xi[gind])
    #print(x)
    #print(gind)

    #exit() 
    if gind.size != 0: 
        n['hpol_copol'][gind] = apat_interp(fhs['u'], fhs['v'], fhs['hpol_copol'],
                                            ut_eta[gind], -ut_xi[gind])

        n['hpol_cxpol'][gind] = apat_interp(fhs['u'], fhs['v'], fhs['hpol_cxpol'],
                                            ut_eta[gind], -ut_xi[gind])

        n['vpol_copol'][gind] = apat_interp(fhs['u'], fhs['v'], fhs['vpol_copol'],
                                            ut_eta[gind], -ut_xi[gind])
        n['vpol_cxpol'][gind] = apat_interp(fhs['u'], fhs['v'], fhs['vpol_cxpol'],
                                            ut_eta[gind], -ut_xi[gind])


    print(f"apat_tilt: n.hpol_copol shape: {np.shape(n['hpol_copol'])}")
    print(f"apat_tilt: n.hpol_copol(1) : {n['hpol_copol'][0]}")
    print(f"apat_tilt: n.hpol_copol(50): {n['hpol_copol'][49]}")
    #exit() 
    # ---------------------
    # [Debug]: There is an issue with some arrays being empty, so as a debug
    # checking for it. 
    if n['hpol_copol'].size == 0 : 
        print("n.hpol_copol is empty")
    elif n['hpol_cxpol'].size == 0 : 
        print("n.hpol_cxpol is empty")
    elif n['vpol_copol'].size == 0 : 
        print("n.vpol_copol is empty")
    elif n['vpol_cxpol'].size == 0 : 
        print("n.vpol_cxpol is empty")
    

    print(f"{np.shape(n['vpol_copol'])}")
    print("")
    #print(n['vpol_cxpol'].keys()) 
    print(f"apat_tilt: n.vpol_cxpol[0]: {n['vpol_cxpol'][0]}") 
    print(f"apat_tilt: n.vpol_cxpol[1]: {n['vpol_cxpol'][1]}") 
    print(f"apat_tilt: n.vpol_cxpol[5]: {n['vpol_cxpol'][5]}") 
    print(f"apat_tilt: n.vpol_cxpol[6]: {n['vpol_cxpol'][6]}") 

    print("Finished Interpolation for FHS")
    #exit() 

    print("")

    print("Started Interpolation for BHS")
    # Interpolate from the BHS 
    # 
    # [**Note**]: Joe's code gives gind shape: (0, 1), so need to do the
    # following to get the same. 
    gind = np.where((ut['zeta'].flatten() > 0.0) & (ut['r2'].flatten() < 1.0))
    gind = np.array(gind).T.flatten() 
    print(f"apat_tilt: gind shape: {np.shape(gind)}")
    #print(gind.size)
    #exit()
    # [**Note**]: Joe's code gives (1501, 1501) while mine (2254501, ) = (1501, 1501) 
    # 
    # Also, in Joe's code the array is alo empty but the code works for some
    # reason. However, for me it doesn't (as expected) because gind is empty.  
    if gind.size != 0: 
        n['hpol_copol'][gind] = apat_interp(bhs['u'], bhs['v'], bhs['hpol_copol'],
                                            ut_eta[gind], -ut_xi[gind])
        n['hpol_cxpol'][gind] = apat_interp(bhs['u'], bhs['v'], bhs['hpol_cxpol'],
                                            ut_eta[gind], -ut_xi[gind])
        n['vpol_copol'][gind] = apat_interp(bhs['u'], bhs['v'], bhs['vpol_copol'],
                                            ut_eta[gind], -ut_xi[gind])
        n['vpol_cxpol'][gind] = apat_interp(bhs['u'], bhs['v'], bhs['vpol_cxpol'],
                                            ut_eta[gind], -ut_xi[gind])


    print(f"apat_tilt: bhs.u shape: {bhs['u'].shape}")
    print("")
    print(f"apat_tilt: n.vpol_cxpol[0]: {n['vpol_cxpol'][0]}") 
    print(f"apat_tilt: n.vpol_cxpol[1]: {n['vpol_cxpol'][1]}") 
    print(f"apat_tilt: n.vpol_cxpol[5]: {n['vpol_cxpol'][5]}") 
    print(f"apat_tilt: n.vpol_cxpol[6]: {n['vpol_cxpol'][6]}") 
    print("")
    
    # Another debug statement here 
    if n['hpol_copol'].size == 0 : 
        print("n.hpol_copol is empty")
    elif n['hpol_cxpol'].size == 0 : 
        print("n.hpol_cxpol is empty")
    elif n['vpol_copol'].size == 0 : 
        print("n.vpol_copol is empty")
    elif n['vpol_cxpol'].size == 0 : 
        print("n.vpol_cxpol is empty")


    #print(bhs['hpol_copol'])
    #print(bhs['hpol_cxpol'])
    #print(n['hpol_copol'])
    #print(n['hpol_cxpol'])
    #print(n['vpol_copol'])
    #print(n['vpol_cxpol'])
    #exit()

    #print(f"{np.shape(n['hpol_copol'])}")
    print("Finished Interpolation for BHS")
    print("")
    #exit()
    #
    n['du'] = ut['du']
    n['dv'] = ut['dv']
    n['u']  = ut['u']
    n['v']  = ut['v']
    
    # TODO: Properly implement the griddata part 
    # %%%%%%%%%%
    # In Joe's code the shape of tr2 is (2501, 2501) 
    # while in my code it is (2501, )
    # that is why we need to create a meshgrid from values before we calculate
    # the tr2 
    #print(np.shape(fhs['u_grid'])) 
    #print("Debug")

    #fhs['u'], fhs['v'] = np.mgrid[fhs['u0']:(fhs['u1'] + fhs['du']):fhs['du'], 
    #                              fhs['v0']:(fhs['v1'] + fhs['dv']):fhs['dv']] 

    # # Retrieving predefined ndgrid 
    # fhsu, fhsv = fhs['u_grid'], fhs['v_grid'] 
    # print(f"{fhsu.shape}")
    # print(f"{fhsu[1,1]}")
    # print("")
    # exit()
    # #fhsu, fhsv = np.meshgrid(fhs['u'], fhs['v'])
    #
    # #tr2 = fhs['u']**2 + fhs['v']**2
    # tr2 = fhsu**2 + fhsv**2
    # #print(np.shape(tr2))
    # #exit() 
    # # In Joe's code 
    # # apat_tilt.m: the shape of gindf is (98069, 1)
    # # apat_tilt.m: the shape of fhs.u is (2501, 2501)
    # # 
    # # In my code, however, it is (2501, 2501) for fhs[hpol_copol] 
    # # 
    # # print(np.isnan(fhs['hpol_copol']))
    # # print(~np.isnan(fhs['hpol_copol']))
    # # print(np.shape(np.isnan(fhs['hpol_copol'])))
    # # print(np.shape(~np.isnan(fhs['hpol_copol'])))
    # #exit()
    # gindf = np.where((tr2.flatten() > 0.98) & ~np.isnan(fhs['hpol_copol'].flatten()))
    # gindf = np.array(gindf).T.flatten()  
    # #print(f"Shape of the fhs[hpol_copol] is {np.shape(fhs['hpol_copol'])}")
    # #print(f"Shape of the fhs[u] is {np.shape(fhs['u'])}")
    # print(f"apat_tilt: gindf shape: {np.shape(gindf)}")
    #
    # # Do not delete this code for now 
    # #fu = fhs['u'][gindf]
    # #fv = fhs['v'][gindf]
    # #fz = -np.ones_like(fu)
    # fu =  fhsu.flatten()[gindf]
    # fv =  fhsv.flatten()[gindf]
    # fz = -np.ones_like(fu)
    # print(np.shape(fu))
    # print(np.shape(fz))
    # #exit() 
    # #
    # #bhsu, bhsv = np.meshgrid(bhs['u'], bhs['v'])
    # bhsu, bhsv = bhs['u_grid'], bhs['v_grid'] 
    # #exit() 
    # tr2 = bhsu**2 + bhsv**2
    # gindb = np.where((tr2.flatten() > 0.98) & ~np.isnan(bhs['hpol_copol'].flatten()))
    # gindb = np.array(gindb).T.flatten() 
    # bu = bhsu.flatten()[gindb]
    # bv = bhsv.flatten()[gindb]
    # bz = np.ones_like(bu)
    # #print(bu)
    # #
    # # efu, efv, eftheta, efphi = uv_to_fs(fu, fv, fz)
    # # Transform tilted (xi, eta) to extended dc (xi,eta).
    # # 
    # # [Maksym]: What is this extended director cosine (edc) coordinate system? 
    # # 
    # efu, efv, eftheta, efphi = uv_to_fs(fu, fv, fz);
    # #exit() 
    # ebu, ebv, ebtheta, ebphi = uv_to_fs(bu, bv, bz);
    #
    # # Fill in the gaps near the unit circle in tilted dc coordinates.
    # tr3 = n['u']**2 + n['v']**2;
    # #print(f"n.hpol_copol: {n['hpol_copol']}")
    # #exit() 
    # # [**Note**]: In Joe's code the 
    # bind = np.where((tr3.flatten() < 1) & (np.isnan(n['hpol_copol'].flatten())));
    # bind = np.array(bind).T.flatten() 
    #
    # print(f"apat_tilt: bind: {bind}")
    # print(f"apat_tilt: n.u(bind) shape: {np.shape(n['u'][bind])}")
    # #exit() 
    # xi, eta, zeta = uv_to_xe(n['u'][bind], n['v'][bind],
    #                          ihs*100*np.ones_like(n['v'][bind]), tilt_ang_deg)
    #
    # # Need to flip coordinates here.
    # xb, eb, theta, phi = uv_to_fs(eta, -xi, zeta) 
    #
    # print(f"apat_tilt: xb size: {xb.shape}")
    #

    n['hpol_copol'] = np.reshape(n['hpol_copol'], (nu, nv)) 
    n['vpol_copol'] = np.reshape(n['vpol_copol'], (nu, nv)) 
    n['hpol_cxpol'] = np.reshape(n['hpol_cxpol'], (nu, nv)) 
    n['vpol_cxpol'] = np.reshape(n['vpol_cxpol'], (nu, nv)) 
    #print(f"{n['hpol_copol'].shape}")
    
    

    return n, ut, s 







