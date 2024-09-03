import pathlib as pb  
import time 


import numpy as np 
import xarray as xr 
import h5py 
import h5netcdf 

import grasp_io as io 



# Load the dataset from the HDF5 file
#ds_loaded = xr.open_dataset('output.h5', engine='h5netcdf')

#print(ds_loaded)
if __name__ == '__main__': 

    start_time_tot = time.time() 
    
    # Getting the root of the repo 
    root_dir = io.find_repo_root() 

    # Params to be put inside parameter file 
    outdir  = pb.Path(f"{root_dir}/output").resolve()
    datadir = pb.Path(f"{root_dir}/dpr/AP").resolve() 


    # Loading the data and converting into xarrays 
    outfile_oap = "/run/media/eva-v3/seagate_small/Maks/CurrProjects/SaT/CIMR-RGB/code/cimr-91/output/parsed/v0.4/CIMR-OAP-FR-C1-UVv0.4.h5"

    with h5py.File(outfile_oap, 'r') as hdf5_file:
        cimr = io.load_hdf5_to_dict(hdf5_file)

    print(cimr) 

    # Load the dataset from the HDF5 file
    ds_loaded = xr.open_dataset(f'{outfile_oap}', engine='h5netcdf')

    print(ds_loaded)
 




    end_time_tot = time.time() - start_time_tot
    print(f"| Finished Script in: {end_time_tot:.2f}s") 
    print(f"| ------------------------------") 
 
