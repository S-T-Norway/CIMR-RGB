import pathlib 
import re 
import glob 
import numbers 

import numpy as np 
import scipy as sp 
import h5py 
import xarray as xr 
import matplotlib
import matplotlib.pyplot as plt 
from   colorama import Fore, Back, Style   
import tqdm 

import grasp_lib as lib  
import grasp_io  as io 



# TODO: - Use xarrays instead of python dictionaries: https://tutorial.xarray.dev/overview/xarray-in-45-min.html 
#       - Use netCDF instead of HDF5 (?)
#       - Create a standalone script to process the files based o this one  

# Update by 2024-08-08: We are going to use SMAP as the baseline for the
# standardized format for parsed antenna patterns 

# Update by 2024-04-13: Since arrays are complex neither HDF5 nor NetCDF can
# save them so we are stuck with matlab or native numpy/scipy routines (such s
# npy and npz files)

def uv_to_tp(u,v): 
    """
    Converting (u,v) into (theta,phi) 
    
    According to the GRASP manual, the relations between (u, v) and (theta,
    phi) coordinates are:  
    
    u = sin(theta) * cos(phi) 
    v = sin(theta) * sin(phi) 

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
    
    """

    theta = np.degrees(np.arccos(np.sqrt(1 - u**2 - v**2))) 
    phi   = np.degrees(np.arctan2(v, u)) 
    
    #theta = np.arccos(np.sqrt(1 - u**2 - v**2)) 
    #phi   = np.arctan2(v, u) 
    
    return theta, phi 


def get_beamdata(beamfile, half_space, cimr_uv, cimr_tp): #cimr, apat_hdf5): 
    """
    Opens GRASP `grd` file defined in uv-coordinates (IGRID value is 1) and
    returns electric field values on a (theta, phi) and (u,v) grid. 

    """
        
    # bn --- beam name. It gives:  
    # C_Horn_01_BHS
    # C_Horn_01_FHS
    #bn = pathlib.Path(f"{band_name}_Horn_{horn_number.zfill(2)}_{last_part}")
    
    # This part is to be inline with Joe's data format  
    if half_space == "FR": 
        bn = "FHS"
    elif half_space == "BK": 
        bn = "BHS"
    
    # The analog of struct from Joe's code (?) 
    #apat = {} #dict() 


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

        # Raising an error if igrid is not = 1  
        print(f"| IGRID = {igrid}") 
        if igrid == 1: 
            print(f"Antenna patterns are provided in the (u,v) coordinates and will be converted into (theta,phi)")
        else: 
            raise NotImplementedError(f"The module functionality is implemented only for IGRID value = 1 since CIMR patterns were provided in this format.")


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

            print(f"KTYPE = {ktype}")
            #print(f"NSET  = {nset}, ICOMP = {icomp}, NCOMP = {ncomp}, IGRID = {igrid}") 
            print(f"IX    = {ix}, IY = {iy}")
            print(f"XS = {xs}, YS = {ys}, XE = {xe}, YE = {ye}")
            print(f"NX = {nx}, NY = {ny}")
            
            # Grid spacing 
            dx = (xe - xs) / (nx - 1)
            dy = (ye - ys) / (ny - 1) 
            xcen = dx * ix 
            ycen = dy * iy 

        #    hpol_copol = np.full((ny, nx), np.nan, dtype=complex)
        #    hpol_cxpol = np.full((ny, nx), np.nan, dtype=complex)
        #    vpol_copol = np.full((ny, nx), np.nan, dtype=complex)
        #    vpol_cxpol = np.full((ny, nx), np.nan, dtype=complex)

            
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
            
            u0 = np.full((ny, nx), np.nan, dtype=float)
            v0 = np.full((ny, nx), np.nan, dtype=float)

            theta = np.full((ny, nx), np.nan, dtype=float)
            phi   = np.full((ny, nx), np.nan, dtype=float)
            
            
            # Here i_row is the row number after (nx, ny, klimit) row in a
            # file. So, to get the current row in a grd file, we need to add
            # i_row and line_shift 
            
            #for i_row in tqdm(range(0, ny), desc=f"Working on row {i_row+1}", unit=i_row):
            for i_row in tqdm.tqdm(range(0, ny), desc=f"| {bn}: Working on chunks (1 chunk = IS rows in a file)", unit=" chunk"): 
                
                line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
                                          info[i_row + line_shift])
                
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
                
                #for ict in range(nxr):
                for ict in range(in_):
                #for ict in tqdm.tqdm(range(nxr), desc="NX", leave=False, unit=" col"):

                    line_numbers = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?',
                                              info[i_row + line_shift + (ict + 1)])
                    # If ix0 = 1, then ic and ict are exactly the same it seems 
                    # the matlab version starts with index 1, but python starts with 0 
                    ic = is_ + ict - 1 #ix0 + ict - 1  
                    
        #            hpol_copol[ic, i_row] = complex(float(line_numbers[2]), float(line_numbers[3])) 
        #            hpol_cxpol[ic, i_row] = complex(float(line_numbers[0]), float(line_numbers[1])) 
        #            # Joe just duplicated the same values for v pol 
        #            vpol_copol[ic, i_row] = complex(float(line_numbers[2]), float(line_numbers[3])) 
        #            vpol_cxpol[ic, i_row] = complex(float(line_numbers[0]), float(line_numbers[1])) 
                    
                    # We were given only horizontal component files, so we copy
                    # those values into vertical as well 
                    G1h[ic, i_row] = float(line_numbers[2])
                    G2h[ic, i_row] = float(line_numbers[3]) 
                    G3h[ic, i_row] = float(line_numbers[0])
                    G4h[ic, i_row] = float(line_numbers[1]) 
            
                    G1v[ic, i_row] = float(line_numbers[2])
                    G2v[ic, i_row] = float(line_numbers[3]) 
                    G3v[ic, i_row] = float(line_numbers[0])
                    G4v[ic, i_row] = float(line_numbers[1]) 
                    
                    # grid points (x,y) run through the values  
                    u0[ic, i_row] = xcen + xs + dx * (ic - 1) 
                    v0[ic, i_row] = ycen + ys + dy * (i_row - 1)

                    # Converting the cartesian (u,v) grid into (theta, phi)   
                    theta[ic, i_row], phi[ic, i_row] = uv_to_tp(u0[ic, i_row], v0[ic, i_row])
                
                # To go to the next block of points within the file we need to
                # increase the line counter 
                line_shift = line_shift + in_ #nxr 


        #apat["hpol_copol"] = hpol_copol
        #apat["hpol_cxpol"] = hpol_cxpol
        #apat["vpol_copol"] = vpol_copol
        #apat["vpol_cxpol"] = vpol_cxpol
        #apat["u0"] = u0
        #apat["v0"] = v0
        
        # The values in 300 gives nans both data and the grid in this way 
        #print(apat["v0"][300,300]) 
        #print(apat["hpol_copol"][300,300]) 
        #
        #print(apat["u0"][300,300]) 
        #print(apat["hpol_copol"][300,300]) 
        #print(apat["u0"][500,500]) 
        #print(apat["hpol_copol"][500,500]) 


        #apat_hdf5["Gain"]['G1h']   = G1h 
        #apat_hdf5["Gain"]['G2h']   = G2h 
        #apat_hdf5["Gain"]['G3h']   = G3h 
        #apat_hdf5["Gain"]['G4h']   = G4h 

        #apat_hdf5["Gain"]['G1v']   = G1v 
        #apat_hdf5["Gain"]['G2v']   = G2v 
        #apat_hdf5["Gain"]['G3v']   = G3v 
        #apat_hdf5["Gain"]['G4v']   = G4v 
        #
        #apat_hdf5["Grid"]['theta'] = theta  
        #apat_hdf5["Grid"]['phi']   = phi  

        ##print(phi.shape)

        #nan_indices_theta = np.isnan(theta) 
        ##print(nan_indices_theta.shape)
        #
        #nan_indices_phi = np.isnan(phi) 
        ##print(nan_indices_phi.shape)

        #identical = np.array_equal(nan_indices_theta, nan_indices_phi) 
        #print(identical)
        #
        #apat_hdf5["Grid"]['phi']   = apat_hdf5["Grid"]['phi'][~nan_indices_phi].reshape(-1) 
        #print(apat_hdf5["Grid"]['phi'])   
        #print(apat_hdf5["Grid"]['phi'].shape)   
        #exit() 

        # TODO: In his code, Joe creates the grid the following way, but he
        # also saves the one as defined by GRASP manual (see u0, and v0 above).
        # oth seem to be outputting the same thing, but with a little bit
        # differnt grid values (see ipynb for the plots of parsed antenna
        # patters).  
        
        #apat["du"] = dx 
        #apat["dv"] = dy 

        #u0 = xcen + xs
        #u1 = u0 + dx * (nx - 1)
        #v0 = ycen + ys
        #v1 = v0 + dy * (ny - 1)

        ##apat["u1"] = u1 
        ##apat["v1"] = v1 

        ## The uv coordinates 
        #u_values = np.arange(u0, u1 + dx, dx)
        #v_values = np.arange(v0, v1 + dy, dy)

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
    
    # Theta Phi grid 
    cimr_tp["Gain"]['G1h']   = G1h 
    cimr_tp["Gain"]['G2h']   = G2h 
    cimr_tp["Gain"]['G3h']   = G3h 
    cimr_tp["Gain"]['G4h']   = G4h 

    cimr_tp["Gain"]['G1v']   = G1v 
    cimr_tp["Gain"]['G2v']   = G2v 
    cimr_tp["Gain"]['G3v']   = G3v 
    cimr_tp["Gain"]['G4v']   = G4v 

    cimr_tp["Grid"]['theta'] = theta  
    cimr_tp["Grid"]['phi']   = phi  

    # u,v grid 
    cimr_uv["Gain"]['G1h']   = G1h 
    cimr_uv["Gain"]['G2h']   = G2h 
    cimr_uv["Gain"]['G3h']   = G3h 
    cimr_uv["Gain"]['G4h']   = G4h 

    cimr_uv["Gain"]['G1v']   = G1v 
    cimr_uv["Gain"]['G2v']   = G2v 
    cimr_uv["Gain"]['G3v']   = G3v 
    cimr_uv["Gain"]['G4v']   = G4v 

    cimr_uv["Grid"]['u']     = u0  
    cimr_uv["Grid"]['v']     = v0   
    
    #cimr[f'{bn}'] = apat  #{'horns_1': {bn: apat}}
        
    #return cimr, apat_hdf5  
    return cimr_uv, cimr_tp    


# TODO: Make save_uv and safe_tp variables work, i.e., which files output to disk 
#def main(datadir, outdir, ref_tilt_ang_deg, downscale_factor, vsign, nu, nv, file_version, save_tp=True, save_uv=True):     
def main(datadir, outdir, file_version, save_tp=True, save_uv=True):     
    """
    Main method 
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

    print(apat_name_info) 

    
    print(f"| {Fore.YELLOW}=============================={Style.RESET_ALL}") 
    print(f"| {Fore.BLUE}Parsing the Antenna Patterns{Style.RESET_ALL}") 
    print(f"| {Fore.YELLOW}=============================={Style.RESET_ALL}") 
            
    # Creating directory to store parsed files  
    parsed_dir = outdir.joinpath("parsed")
    if not parsed_dir.exists(): 
        print(f"| Creating parsed directory:\n{parsed_dir}")
        pathlib.Path(parsed_dir).mkdir()

    #parsedfile_prefix = "cimr_apat_full" 
    parsedfile_prefix = "CIMR-AP-FULL" 
    parsedfile_suffix = "XE"
    
    print(f"| {Fore.GREEN}Data Directory:{Fore.RESET}\n| {Fore.BLUE}{datadir}{Style.RESET_ALL}") 
    print(f"| {Fore.GREEN}Parsed Directory:{Fore.RESET}\n| {Fore.BLUE}{parsed_dir}{Style.RESET_ALL}") 
    # Main parsing loop 
    for band in apat_name_info.keys(): 
        
        for horn, half_spaces in apat_name_info[band][2].items(): 
            
            # Output beam file that contains FHS and BHS data 
            #outfile = pathlib.Path(str(parsed_dir) + f"/{parsedfile_prefix}-" + band + horn + f"-{parsedfile_suffix}.mat")   
            #if not pathlib.Path(outfile).exists(): 
            #if not outfile.exists(): 
            #    print(f"| Output File: {Fore.BLUE}{outfile.name} {Fore.RED}{Style.BRIGHT}doesn't exist\n| Starting creation{Style.RESET_ALL}")
            #else: 
            #    print(f"| Output File:\n| {Fore.MAGENTA}{outfile.name} {Fore.CYAN}{Style.BRIGHT}exists.{Style.RESET_ALL}")
            #    continue 
            
            freq = apat_name_info[band][0] 
            pol  = apat_name_info[band][1] 
            
            # Object to be saved into file 
            #cimr = {}
            # This one is for hdf5 standardized format 
            cimr_tp = {"Gain": {}, "Grid": {}, "Version": file_version} 
            cimr_uv = {"Gain": {}, "Grid": {}, "Version": file_version} 
            
            for half_space in half_spaces: 

                # Reconstructing the full path to the file 
                if band == "L": 
                    infile = band + "-" + freq + "-" + pol + "-" + half_space + ".grd" 
                else: 
                    infile = band + str(horn) + "-" + freq + "-" + pol + "-" + half_space + ".grd" 
                
                infile = pathlib.Path(str(datadir) + "/" + band + "/" + infile)  
                print(f"| {Fore.GREEN}Working with Input File: {infile.name}{Style.RESET_ALL}") 

                # Returns cimr object as dictionary with FHS and BHS 
                #cimr, apat_hdf5 = get_beamdata(infile, half_space, cimr, apat_hdf5)     
                cimr_uv, cimr_tp = get_beamdata(infile, half_space, cimr_uv, cimr_tp)     

                #print(apat_hdf5.keys())
                #exit() 

                # Saving parsed data into hdf5 file 
                
                # TODO: These checks DO NOT WORK because you are still calculating things above  
                #  Theta Phi grid 
                parsedfile_prefix = f"CIMR-AP-{half_space}" 
                parsedfile_suffix = "TP"
                outfile_tp = pathlib.Path(str(parsed_dir) + f"/{parsedfile_prefix}-" + band + horn + f"-{parsedfile_suffix}v{file_version}.h5")   
                
                if io.check_outfile_existance(outfile_tp) == True: 
                    continue 
                else: 
                    print(f"| {Fore.BLUE}Saving Output File: {outfile_tp.name}{Style.RESET_ALL}") 
                    with h5py.File(outfile_tp, 'w') as hdf5_file:
                        io.save_dict_to_hdf5(hdf5_file, cimr_tp)

                # u,v grid 
                parsedfile_suffix = "UV"
                outfile_uv = pathlib.Path(str(parsed_dir) + f"/{parsedfile_prefix}-" + band + horn + f"-{parsedfile_suffix}v{file_version}.h5")   
                
                if io.check_outfile_existance(outfile_uv) == True: 
                    continue 
                else: 
                    print(f"| {Fore.BLUE}Saving Output File: {outfile_uv.name}{Style.RESET_ALL}") 
                    with h5py.File(outfile_uv, 'w') as hdf5_file:
                        io.save_dict_to_hdf5(hdf5_file, cimr_uv)


                
                    
                
                
                #if not outfile.exists(): 
                #    print(f"| Output File: {Fore.BLUE}{outfile.name} {Fore.RED}{Style.BRIGHT}doesn't exist\n| Starting creation{Style.RESET_ALL}")
                #else: 
                #    print(f"| Output File:\n| {Fore.MAGENTA}{outfile.name} {Fore.CYAN}{Style.BRIGHT}exists.{Style.RESET_ALL}")
                #    continue 
                #
                #parsedfile_suffix = "TP"
                #outfile_tp = pathlib.Path(str(parsed_dir) + f"/{parsedfile_prefix}-" + band + horn + f"-{parsedfile_suffix}v{file_version}.h5")   
                #

                #
                #if not outfile.exists(): 
                #    print(f"| Output File: {Fore.BLUE}{outfile.name} {Fore.RED}{Style.BRIGHT}doesn't exist\n| Starting creation{Style.RESET_ALL}")
                #else: 
                #    print(f"| Output File:\n| {Fore.MAGENTA}{outfile.name} {Fore.CYAN}{Style.BRIGHT}exists.{Style.RESET_ALL}")
                #    continue 
                #
                ## T = theta, P = phi  
                #print(parsedfile_prefix) 
                #outfile_hdf = pathlib.Path(str(parsed_dir) + f"/{parsedfile_prefix}-" + band + horn + f"-{parsedfile_suffix}v{file_version}.h5")   
                #print(outfile_hdf)
                #
                #print(f"| {Fore.BLUE}Saving Output File: {outfile.name}{Style.RESET_ALL}") 
                #with h5py.File(outfile_hdf, 'w') as hdf5_file:
                #    io.save_dict_to_hdf5(hdf5_file, apat_hdf5)
                ##exit() 
            
            # Saving it to matlab file 
            #print(f"| {Fore.BLUE}Saving Output File: {outfile.name}{Style.RESET_ALL}") 
            ##sp.io.savemat(outfile, cimr)
            
            print(f"| {Fore.YELLOW}------------------------------{Style.RESET_ALL}") 

    
    # TODO: This part can be removed (later), because we won't be using Joe's
    # approach for processing the antenna patterns  
    
    ## ==============================
    ## Preprocessing Antenna Patterns  
    ## ==============================
    ## [**Note**]: In Joe's code, v and u are on the 2D grid, while here I am
    ## writing it as 1D grid. This stems from the fact that the scipy interpn
    ## function does not accept these values as 2D, only as 1D. Therefore, every
    ## time there is an operation on these values the mshgrid should be established.  
    ## Otherwise, the original grid should be kept, which is what I am doing.  
    #print(f"| {Fore.YELLOW}=============================={Style.RESET_ALL}") 
    #print(f"| {Fore.BLUE}Untilting the Antenna Patterns{Style.RESET_ALL}") 
    #print(f"| {Fore.YELLOW}=============================={Style.RESET_ALL}") 
    #
    #apat = {'BHS': {}, 'FHS': {}} 
    #
    #processedfile_prefix = "CIMR-AP-HBS"
    #processedfile_sufix  = "UV"
    #
    #processed_dir = f"{outdir}/processed"
    #if not pathlib.Path(processed_dir).exists():
    #    pathlib.Path(processed_dir).mkdir()
    #    print(processed_dir)
    #
    #for band in apat_name_info.keys(): 
    #    
    #    for horn in apat_name_info[band][2].keys(): 
    #        
    #        #beamfile_parsed = f"{parsed_dir}/{parsedfile_prefix}_{band}_horn_{horn.zfill(2)}_xe.mat"
    #        beamfile_parsed = f"{parsed_dir}/{parsedfile_prefix}-{band}{horn}-{parsedfile_suffix}.mat"
    #        cimr = sp.io.loadmat(beamfile_parsed)  
    #        
    #        # For some reason it saved the data not as dict of dict of numpy
    #        # arrays but as dict of numpy array of numpy arrays, i.e.,
    #        # cimr['BHS'] is considered by python as numpy array and not a
    #        # dictionary, so need to reconstruct the original dictionary again 
    #        for half_space in ['BHS', 'FHS']: 
    #            apat[f'{half_space}']["hpol_copol"] = cimr[f'{half_space}']['hpol_copol'][0][0]  
    #            apat[f'{half_space}']["hpol_cxpol"] = cimr[f'{half_space}']['hpol_cxpol'][0][0] 
    #            apat[f'{half_space}']["vpol_copol"] = cimr[f'{half_space}']['vpol_copol'][0][0] 
    #            apat[f'{half_space}']["vpol_cxpol"] = cimr[f'{half_space}']['vpol_cxpol'][0][0] 
    #            apat[f'{half_space}']["u0"]         = cimr[f'{half_space}']['u0'][0][0] 
    #            apat[f'{half_space}']["v0"]         = cimr[f'{half_space}']['v0'][0][0] 
    #            apat[f'{half_space}']["du"]         = cimr[f'{half_space}']['du'][0][0]  
    #            apat[f'{half_space}']["dv"]         = cimr[f'{half_space}']['dv'][0][0]  
    #            apat[f'{half_space}']['u']          = cimr[f'{half_space}']['u'][0][0].flatten()   
    #            apat[f'{half_space}']['v']          = cimr[f'{half_space}']['v'][0][0].flatten() 
    #            apat[f'{half_space}']['u_grid']     = cimr[f'{half_space}']['u_grid'][0][0]  
    #            apat[f'{half_space}']['v_grid']     = cimr[f'{half_space}']['v_grid'][0][0]  
    #        
    #        # Feeing up memory (removing data duplicates)  
    #        del cimr 
    #        
    #        apat['FHS'], apat['BHS'] = lib.trans_apat_fhs_bhs_fast(apat['FHS'], apat['BHS'], vsign) 
    #        
    #        # Normalize the patterns by removing the 4*pi factor right out of the gate.
    #        apat['FHS'] = lib.apat_normalize(apat['FHS'])
    #        apat['BHS'] = lib.apat_normalize(apat['BHS'])
    #
    #        # ----------------------------------------------
    #        # TILT HORN BORESIGHT (HBS; MAIN BEAM) PATTERNS.
    #        # ----------------------------------------------
    #        
    #        # We need to downsample the patterns because they are very big in size. Also,
    #        # what we need to do is to work with only Main Beam instead of the Full Beam
    #        # and that is because the assumption is that the sidelobes should be removed. 
    #        
    #        # Get the reflector boresight (u,v,z). 
    #        ut = {} 
    #        
    #        # bu, bv, bz stand for boresight u, boresight v, boresight z. 
    #        ut['bu'], ut['bv'], ut['bz'] = lib.xe_to_uv(0, 0, -1, ref_tilt_ang_deg)
    #        print(f"The value of ut.bu is {ut['bu']}")
    #        print(f"The value of ut.bv is {ut['bv']}")
    #        print(f"The value of ut.bz is {ut['bz']}")
    #        
    #        # Where does these values come from? 
    #        ut['uhw'] = 0.6
    #        ut['vhw'] = 0.6
    #        ut['u0']  = ut['bu'] - ut['uhw']
    #        ut['u1']  = ut['bu'] + ut['uhw']
    #        ut['v0']  = ut['bv'] - ut['vhw']
    #        ut['v1']  = ut['bv'] + ut['vhw']
    #        
    #        sp_dtheta = 0.1
    #        sp_dphi   = 0.1
    #        
    #        # Calculate the reflector patterns
    #        nur = nu * downscale_factor
    #        nvr = nv * downscale_factor
    #        
    #        # ------
    #        # Analog to the Horn Boresight (Main Beam) struct  
    #        hbs, utf, sph = lib.apat_tilt(
    #                apat['FHS'],
    #                apat['BHS'],
    #                #cimr['horns']['C_Horn_01_FHS']['apat'], 
    #                #cimr['horns']['C_Horn_01_BHS']['apat'],
    #                ref_tilt_ang_deg,
    #                ut['u0'], ut['u1'], ut['v0'], ut['v1'], 
    #                -1, sp_dtheta, sp_dphi, nur, nvr)
    #        
    #        print(hbs.keys()) 
    #        
    #        print(hbs['hpol_copol'][0][0])
    #        
    #        beamfile_processed = f"{processedfile_prefix}-{band}{horn}-{processedfile_sufix}.mat"
    #        beamfile_processed = processed_dir + "/" + beamfile_processed   
    #        
    #        sp.io.savemat(beamfile_processed, hbs) 
     
    #return 0 



if __name__ == '__main__': 
    
    # TODO: Create a parameter file that will take as input this info  

    root_dir = io.find_repo_root() 
    print(root_dir)

    # Params to be put inside parameter file 
    outdir  = pathlib.Path(f"{root_dir}/output").resolve()
    #datadir = pathlib.Path("../../../../data/CIMR/beams/ours/").resolve() 
    datadir = pathlib.Path(f"{root_dir}/dpr/AP").resolve() 

    if not pathlib.Path(outdir).exists(): 
        print(f"Creating output directory:\n{outdir}")
        pathlib.Path(outdir).mkdir()

    if not datadir.is_dir():
        raise FileNotFoundError(f"The directory '{datadir}' does not exist.")

    beamfiles_paths = datadir.glob("*/*")   

    # Beam Info 
    #downscale_factor = 1 #3
    #ref_tilt_ang_deg = 46.886  
    #vsign = 1 
    ## Why nu = -1? <= that is because he distinguishes the grid size/type with it  
    ## Grid limits for the antenna pattern 
    #nu     = 301 #-1 
    #nv     = 251 #-1 
    ##do_hbs =  0 
    ##do_sph =  1 

    file_version = 0.1 

    #main(datadir, outdir, ref_tilt_ang_deg, downscale_factor, vsign, nu, nv, file_version)    
    main(datadir, outdir, file_version)    
    

