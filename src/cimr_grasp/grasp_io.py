import os
import pathlib as pb
import re
import logging

import numpy as np
import h5py
from colorama import Fore, Back, Style


def get_header(beamfile: str) -> list:
    """
    Method to get some useful information from the header file.

    Parameters:
    -----------
    beamfile: str
        The (text; .grd) file to parse the header information from.

    Returns:
    --------
    header: list
        List containing header information from the given text file.
    """

    header = []

    while True:
        line = beamfile.readline()
        if line.strip() == "++++":
            break
        header.append(line)

    header = [s.strip("\n") for s in header]

    return header


def check_outfile_existance(outfile: pb.Path) -> bool:
    """
    Checks for the existence of a given (e.g., antenna pattern) file and
    returns boolean value.

    Parameters:
    -----------
    outfile: str or Path
        The file name to check for.

    Returns:
    --------
    bool
        Whether given file exists or not.
    """

    if not outfile.exists():
        print(
            f"| Output File: {Fore.BLUE}{outfile.name} {Fore.RED}{Style.BRIGHT}doesn't exist\n| Starting creation{Style.RESET_ALL}"
        )
        return False
    else:
        print(
            f"| Output File:\n| {Fore.MAGENTA}{outfile.name} {Fore.CYAN}{Style.BRIGHT}exists.{Style.RESET_ALL}"
        )
        return True


def parse_file_name(file_name: pb.Path) -> (str, str, str, str, str):
    """
    Parses the beamfile name to get the band, horn and half-space identifier.
    The file name should be of format:

    S-12345-E-FR" or F1-12345-E-BK"

    Parameters:
    -----------
    file_name: str or Path
        The beam file name to get information from.

    Returns:
    --------
    band: str
        Band number.
    horn: str
        Horn number.
    freq: str
        Frequency value.
    pol:  str
        Polarisation value.
    half_space: str
        The half-space of the antenna pattern sphere.

    Raises:
    -------
    ValueError
        If the aformentioned file format was not uphold.
    """

    # Regex to match the expected patterns of the filename
    pattern = r"([A-Za-z]+\d*)-(\d+)-([A-Z]+)-([A-Z]+)"
    match = re.match(pattern, file_name)

    if not match:
        raise ValueError("Invalid filename format: " + file_name)

    # Extract matched groups
    band = match.group(1)
    freq = match.group(2)
    pol = match.group(3)
    half_space = match.group(4)
    # Check for horn based on whether there's a number in the band name
    horn = (
        "".join(filter(str.isdigit, band))
        if any(char.isdigit() for char in band)
        else "1"
    )
    band = "".join(filter(str.isalpha, band))

    return band, horn, freq, pol, half_space


def save_dict_to_hdf5(hdf5_group, data_dict: dict) -> None:
    """
    Recursively saves (e.g, parsed beam) dictionary data into HDF5.

    Parameters:
    -----------
    hdf_group:
        The HDF5 group to work with.
    """

    for key, value in data_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, create a subgroup
            subgroup = hdf5_group.create_group(key)
            # Recursively call the function for the subgroup
            save_dict_to_hdf5(subgroup, value)
        elif isinstance(value, np.ndarray):
            # If the value is a NumPy array, create a dataset
            hdf5_group.create_dataset(key, data=value)
        elif np.isscalar(value):
            # If the value is a scalar, also create a dataset
            hdf5_group.create_dataset(key, data=value)  # np.array(value))
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")


def load_hdf5_to_dict(hdf5_group) -> dict:
    """
    Loads (e.g, parsed beam) information from the HDF5 file in a form of a
    dictionary.

    Parameters:
    -----------
    hdf_group:
        The HDF5 group to work with.

    Returns:
    --------
    data_dict: dict
        Data to be returned in a python dictionary format.
    """

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


def find_repo_root(start_path: pb.Path = None) -> pb.Path:
    """
    Finds the root path of the repo based off the location of the first `.git`
    directory encountered.

    Parameters:
    -----------
    start_path: Path
        The start path to start the search from/at.

    Returns:
    --------
    parent: str or Path
        Root path to the directory.

    Raises:
    -------
    FileNotFoundError
        No `.git` was found in any of the parent directories.
    """

    # If no start path is provided, use the current working directory
    if start_path is None:
        start_path = pb.Path.cwd()

    # Convert to absolute path
    start_path = start_path.resolve()

    # Check each directory from the start path upwards to the root
    for parent in [start_path] + list(start_path.parents):
        if (parent / ".git").is_dir():
            return parent

    raise FileNotFoundError("No .git directory found in any parent directories")


def rec_create_dir(path: pb.Path | str, logger=None) -> None:
    """
    Recursively create directories.

    Parameters:
    -----------
    path: Path or str
        The path to the directory in question.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    # Convert the path to a Path object
    path = pb.Path(path)

    # Create the directories if they do not exist
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directories created: {path}")
    else:
        logger.debug(f"Path already exists: {path}")


def resolve_config_path(path_string: pb.Path | str) -> pb.Path:
    """
    Resolves a given path string to an absolute `Path` object, handling
    home directory symbols (`~`) and environment variables (`$VAR`) if present.

    This method checks for special symbols in the input path to ensure it points
    to a valid location. It expands `~` to the home directory, replaces environment
    variables (e.g., `$HOME`) if specified, and resolves any relative paths to absolute
    ones. If both `~` and `$` are in the path, a `ValueError` is raised, as this
    could lead to unexpected path behavior.

    Parameters:
    -----------
    path_string : pb.Path | str
        A path string or `Path` object that may contain `~` for the home directory
        or `$VAR` for environment variables.

    Returns:
    --------
    pb.Path
        The fully resolved, absolute path as a `Path` object.

    Exceptions:
    -----------
    ValueError
        Raised if the path contains both `~` and `$`, which could lead to incorrect
        path resolution due to conflicting expansions.
    """

    path_string = str(path_string).strip()

    if "~" in str(path_string) and not "$" in str(path_string):
        # Use expanduser to handle '~' for home directory if present
        expanded_path = os.path.expanduser(path_string)
        return pb.Path(expanded_path).resolve()
    elif "$" in str(path_string) and not "~" in str(path_string):
        # Use expandvars to replace environment variables like $HOME
        expanded_path = os.path.expandvars(path_string)
        return pb.Path(expanded_path).resolve()
    elif "$" in str(path_string) and "~" in str(path_string):
        error_output = (
            "`outdir` contains both $ and ~ which "
            + "will result in incorrect path resolution. "
            +
            # f"Check `outdir` variable in {config_file}")
            f"Check `outdir` variable in configuration file."
        )
        raise ValueError(error_output)
    else:
        # If we have a relative path, we expand it
        expanded_path = pb.Path(path_string)
        if not expanded_path.is_absolute():
            return expanded_path.resolve()
        else:
            return expanded_path


# TODO: This method does not work properly if we have a numpy arrays inside
#       nested dictionaries. BUT, it works for the empty dictionaries
def is_nested_dict_empty(d: dict) -> bool:
    """
    Checks if a nested dictionary is empty, including handling dictionaries
    that may contain `numpy` arrays, nested dictionaries, or other data types.

    This method recursively examines the contents of a nested dictionary and
    returns `True` if all entries are empty or contain only empty dictionaries
    or empty `numpy` arrays. It returns `False` if any entry is non-empty, such
    as containing values within a `numpy` array, a non-empty dictionary, or any
    other data type with content.

    Parameters:
    -----------
    d : dict
        A dictionary, potentially nested, to check for emptiness.

    Returns:
    --------
    bool
        `True` if the dictionary is empty, including any nested dictionaries
        and `numpy` arrays that may be empty. `False` if any entry is non-empty.
    """

    if (
        not isinstance(d, dict) or not d
    ):  # If it's not a dictionary or the dictionary is empty
        return not d

    return all(is_nested_dict_empty(v) for v in d.values())


def get_bool_from_string(par_val: str, par_name: str) -> bool:
    """
    Converts a string representation of a boolean to an actual boolean value.

    This method accepts a string input, which should be either "true" or "false",
    and returns the corresponding boolean value. The input is case-insensitive.
    If the input does not match "true" or "false", a `ValueError` is raised with
    a message indicating that only these values are allowed.

    Parameters:
    ------------------
    par_val : str
        The parameter value as a string, expected to be either "true" or "false".
    par_name : str
        The name of the parameter, used in error messaging to identify the specific
        parameter that caused the exception.

    Returns:
    ------------
    bool
        The converted boolean value (`True` for "true" and `False` for "false").

    Exceptions:
    -----------------
    ValueError
        Raised if `par_val` is not "true" or "false", indicating an invalid input.
    """

    par_val = par_val.lower()

    if par_val == "true":
        par_val_out = True
    elif par_val == "false":
        par_val_out = False
    else:
        raise ValueError(f"Parameter `{par_name}` can either be True or False.")

    return par_val_out
