import pathlib 
import re 

import numpy as np 
import h5py 
from   colorama import Fore, Back, Style   


def get_header(beamfile):
    """
    Method to get some useful information from the header file. 
    """
    
    header = [] 
    
    while True: 
        line = beamfile.readline()
        if line.strip() == "++++":
            break
        header.append(line)

    header = [s.strip('\n') for s in header]

    return header 

#def get_horn_number(string):
#    """
#    Method to get a horn number from the file name  
#    """
#    
#    # Split the string by the first hyphen
#    parts = string.split('-', 1)
#    
#    # Extract the first part
#    first_part = parts[0]
#    
#    # Use regular expression to separate number and letters
#    match = re.match(r'(\D+)(\d*)', first_part)
#    
#    # If number is missing, default to 1
#    if match.group(2) == '':
#        number = '1'
#    else:
#        number = match.group(2)
#    
#    letter = match.group(1)
#    
#    return number, letter


def check_outfile_existance(outfile): 

    if not outfile.exists(): 
        print(f"| Output File: {Fore.BLUE}{outfile.name} {Fore.RED}{Style.BRIGHT}doesn't exist\n| Starting creation{Style.RESET_ALL}")
        return False 
    else: 
        print(f"| Output File:\n| {Fore.MAGENTA}{outfile.name} {Fore.CYAN}{Style.BRIGHT}exists.{Style.RESET_ALL}")
        return True  




def parse_file_name(file_name):
    """
    Parses the beamfile name to get the band, horn and half-space identifier. 
    The file name should be of format: 

    S-12345-E-FR", F1-12345-E-BK",
    """
    
    parts = file_name.split('-')
    band = parts[0]
    freq = parts[1]
    pol  = parts[2]
    
    if len(band) == 1: 
        horn = "1" 
    elif len(band) == 2: 
        horn = band[1] 
    else: 
        raise ValueError("Invalid filename format: " + file_name) 
        
    band = band[0]

    half_space = parts[-1].split('.')[0]
        
    return band, horn, freq, pol, half_space 


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
        #else:
        #    # If the value is a scalar or other type, create an attribute
        #    hdf5_group.attrs[key] = value
        elif np.isscalar(value): 
            # If the value is a scalar, also create a dataset
            hdf5_group.create_dataset(key, data=value) #np.array(value))
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}") 

            
# Function to recursively load HDF5 file into a dictionary
def load_hdf5_to_dict(hdf5_group):
    data_dict = {}
    for key, value in hdf5_group.items():
        if isinstance(value, h5py.Group):
            # If the value is a group, recursively call the function for the subgroup
            data_dict[key] = load_hdf5_to_dict(value)
        elif isinstance(value, h5py.Dataset):
            # If the value is a dataset, load the data into a NumPy array
            data_dict[key] = np.array(value)
        else:
            # If the value is an attribute, load it into the dictionary
            data_dict[key] = hdf5_group.attrs[key]
            
    return data_dict


def find_repo_root(start_path: pathlib.Path = None) -> pathlib.Path:
    # If no start path is provided, use the current working directory
    if start_path is None:
        start_path = pathlib.Path.cwd()
    
    # Convert to absolute path
    start_path = start_path.resolve()
    
    # Check each directory from the start path upwards to the root
    for parent in [start_path] + list(start_path.parents):
        if (parent / '.git').is_dir():
            return parent
    
    raise FileNotFoundError("No .git directory found in any parent directories") 
