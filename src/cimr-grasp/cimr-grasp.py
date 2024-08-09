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

import grasp_io  as io 

# TODO: - Use xarrays instead of python dictionaries: https://tutorial.xarray.dev/overview/xarray-in-45-min.html 
#       - Use netCDF instead of HDF5 (?)
#       - Create a standalone script to process the files based on this one  

# Update by 2024-08-08: We are going to use SMAP as the baseline for the
# standardized format for parsed antenna patterns 

# Update by 2024-04-13: Since arrays are complex neither HDF5 nor NetCDF can
# save them so we are stuck with matlab or native numpy/scipy routines (such s
# npy and npz files)

def uv_to_tp(u,v): 
    """
    Converting (u,v) into (theta,phi) and returning the grid in degrees.   
    
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

    theta = np.degrees(np.arccos(np.sqrt(1 - u**2 - v**2))) 
    phi   = np.degrees(np.arctan2(v, u)) 
    
    #theta = np.arccos(np.sqrt(1 - u**2 - v**2)) 
    #phi   = np.arctan2(v, u) 
    
    return theta, phi 


#def get_beamdata(beamfile, half_space, cimr_uv, cimr_tp): #cimr, apat_hdf5): 
def get_beamdata(beamfile, half_space, cimr): #cimr, apat_hdf5): 
    """
    Opens GRASP `grd` file defined in uv-coordinates (IGRID value is 1) and
    returns electric field values on a (theta, phi) and (u,v) grid. 

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

            print(f"| KTYPE = {ktype}")
            #print(f"NSET  = {nset}, ICOMP = {icomp}, NCOMP = {ncomp}, IGRID = {igrid}") 
            print(f"| IX    = {ix}, IY = {iy}")
            print(f"| XS = {xs}, YS = {ys}, XE = {xe}, YE = {ye}")
            print(f"| NX = {nx}, NY = {ny}")
            
            # Grid spacing 
            dx = (xe - xs) / (nx - 1)
            dy = (ye - ys) / (ny - 1) 
            xcen = dx * ix 
            ycen = dy * iy 

            
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
                    
                    # Grid points (x,y) run through the values  
                    u0[ic, i_row] = xcen + xs + dx * (ic - 1) 
                    v0[ic, i_row] = ycen + ys + dy * (i_row - 1)

                    # Converting the cartesian (u,v) grid into (theta, phi)   
                    theta[ic, i_row], phi[ic, i_row] = uv_to_tp(u0[ic, i_row], v0[ic, i_row])
                
                # To go to the next block of points within the file we need to
                # increase the line counter 
                line_shift = line_shift + in_ #nxr 


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
    
    cimr["Gain"]['G1h']   = G1h 
    cimr["Gain"]['G2h']   = G2h 
    cimr["Gain"]['G3h']   = G3h 
    cimr["Gain"]['G4h']   = G4h 

    cimr["Gain"]['G1v']   = G1v 
    cimr["Gain"]['G2v']   = G2v 
    cimr["Gain"]['G3v']   = G3v 
    cimr["Gain"]['G4v']   = G4v 
    
    cimr["Grid"]['u']     = u0  
    cimr["Grid"]['v']     = v0   
    cimr["Grid"]['theta'] = theta  
    cimr["Grid"]['phi']   = phi  
        
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

    print(apat_name_info) 

    
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

    file_version = 0.1 

    #main(datadir, outdir, ref_tilt_ang_deg, downscale_factor, vsign, nu, nv, file_version)    
    main(datadir, outdir, file_version)    
    

