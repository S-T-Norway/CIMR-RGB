import sys
import os
import re
import pathlib as pb
import time
import functools
import xml.etree.ElementTree as ET
import argparse
import logging

# 3d-party tools
import numpy as np
import scipy as sp
import h5py
from colorama import Fore, Style
import tqdm

# Custom modules
import cimr_grasp.grasp_io as io
import cimr_grasp.grasp_utils as utils
from cimr_rgb.rgb_logging import RGBLogging


def get_beamdata(
    beamfile: pb.Path | str,
    pol: str,
    half_space: str,
    file_version: float,
    cimr: dict,
    logger: logging.Logger,
) -> dict:
    """
    Parses one GRASP `grd` file and adds its polarization component
    to the CIMR dictionary.

    Parameters
    ----------
    beamfile : str or Path
        Path to the GRASP beam file.

    pol : str
        Polarization of the current file. Expected values: "H" or "V".

    half_space : str
        Current antenna pattern half-space, e.g. "FR" or "BK".

    file_version : float or str
        Version of the current file iteration.

    cimr : dict
        CIMR dictionary to be updated.

    logger : logging.Logger
        Logger object.

    Returns
    -------
    cimr : dict
        Updated CIMR dictionary containing the parsed polarization component.
    """

    pol = pol.upper()

    if pol not in {"H", "V"}:
        raise ValueError(
            f"Unsupported polarization '{pol}' for file {beamfile}. "
            "Expected 'H' or 'V'."
        )

    # This part is inline with Joe's data format.
    # We use FHS/BHS only for tqdm/log output.
    if half_space == "FR":
        bn = "FHS"
    elif half_space == "BK":
        bn = "BHS"
    else:
        bn = half_space

    reline_pattern = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")

    with open(beamfile, mode="r", encoding="UTF-8") as bfile:
        header = io.get_header(bfile)

        # Retrieving data after ++++
        info = [line.strip("\n") for i, line in enumerate(bfile)]

    # First 3 lines after ++++
    line_shift = 3

    for i in range(0, line_shift):
        line_numbers = reline_pattern.findall(info[i])

        if i == 0:
            ktype = int(line_numbers[0])

        elif i == 1:
            nset = int(line_numbers[0])
            icomp = int(line_numbers[1])
            ncomp = int(line_numbers[2])
            igrid = int(line_numbers[3])

        elif i == 2:
            ix = int(line_numbers[0])
            iy = int(line_numbers[1])

    logger.info(f"KTYPE = {ktype}")
    logger.info(
        f"NSET = {nset}, ICOMP = {icomp}, NCOMP = {ncomp}, IGRID = {igrid}"
    )

    if igrid == 1:
        logger.info(
            "Antenna patterns are provided in the (u,v) coordinates "
            "and will be converted into (theta,phi)"
        )
    else:
        raise NotImplementedError(
            "The module functionality is implemented only for IGRID value = 1 "
            "since CIMR patterns were provided in this format."
        )

    # The following lines are repeated NSET times.
    # In the current CIMR files we expect NSET = 1.
    if nset != 1:
        raise NotImplementedError(
            f"Only NSET=1 is currently supported. Found NSET={nset} in {beamfile}."
        )

    for i_set in range(nset):
        for k in range(line_shift, line_shift + 2):
            line_numbers = reline_pattern.findall(info[k])

            if k == 3:
                xs = float(line_numbers[0])
                ys = float(line_numbers[1])
                xe = float(line_numbers[2])
                ye = float(line_numbers[3])

            elif k == 4:
                nx = int(line_numbers[0])
                ny = int(line_numbers[1])
                klimit = int(line_numbers[2])

        line_shift = line_shift + 2

    logger.info(f"IX = {ix}, IY = {iy}")
    logger.info(f"XS = {xs}, YS = {ys}, XE = {xe}, YE = {ye}")
    logger.info(f"NX = {nx}, NY = {ny}")

    # Grid spacing
    dx = (xe - xs) / (nx - 1)
    dy = (ye - ys) / (ny - 1)

    xcen = dx * ix
    ycen = dy * iy

    logger.info(f"DX = {dx}, DY = {dy}")
    logger.info(f"XCEN = {xcen}, YCEN = {ycen}")

    # Temporary arrays for the current polarization only.
    #
    # GRASP file convention used by the existing code:
    #   line_numbers[2], line_numbers[3] -> co-pol real/imag
    #   line_numbers[0], line_numbers[1] -> cross-pol real/imag
    Gco_real = np.full((ny, nx), 0.0, dtype=float)
    Gco_imag = np.full((ny, nx), 0.0, dtype=float)
    Gcx_real = np.full((ny, nx), 0.0, dtype=float)
    Gcx_imag = np.full((ny, nx), 0.0, dtype=float)

    for j_ in tqdm.tqdm(
        range(0, ny),
        desc=f"| {bn}-{pol}: Working on chunks (1 chunk = IS rows in a file)",
        unit=" chunk",
    ):
        line_numbers = reline_pattern.findall(info[j_ + line_shift])

        if klimit == 0:
            # Full row. No row limit given.
            is_ = 1
            in_ = nx

        elif klimit == 1:
            # IS and IN are provided at the start of each row.
            is_ = int(line_numbers[0])
            in_ = int(line_numbers[1])

        else:
            raise NotImplementedError(
                f"Unsupported KLIMIT={klimit} in file {beamfile}."
            )

        for ict in range(in_):
            line_numbers = reline_pattern.findall(
                info[j_ + line_shift + (ict + 1)]
            )

            # MATLAB/GRASP starts at 1, Python starts at 0.
            ic = is_ + ict - 1

            Gco_real[ic, j_] = float(line_numbers[2])
            Gco_imag[ic, j_] = float(line_numbers[3])
            Gcx_real[ic, j_] = float(line_numbers[0])
            Gcx_imag[ic, j_] = float(line_numbers[1])

        # Move to the next block of points.
        line_shift = line_shift + in_

    u_grid, v_grid = utils.generate_uv_grid(
        xcen,
        ycen,
        xs,
        ys,
        nx,
        ny,
        dx,
        dy,
    )

    # Getting vectors back.
    u = np.unique(u_grid[0, :])
    v = np.unique(v_grid[:, 0])

    # Store/check grid.
    # The first polarization creates the grid.
    # The second polarization must be on the same grid.
    if "u" in cimr["Grid"]:
        if not np.allclose(cimr["Grid"]["u"], u):
            raise ValueError(
                f"u-grid mismatch while reading {beamfile}. "
                "H and V files must be defined on the same grid."
            )

        if not np.allclose(cimr["Grid"]["v"], v):
            raise ValueError(
                f"v-grid mismatch while reading {beamfile}. "
                "H and V files must be defined on the same grid."
            )

        grid_checks = {
            "xcen": xcen,
            "ycen": ycen,
            "xs": xs,
            "ys": ys,
            "nx": nx,
            "ny": ny,
            "dx": dx,
            "dy": dy,
        }

        for key, value in grid_checks.items():
            old_value = cimr["Grid"][key]

            if isinstance(value, float):
                if not np.isclose(old_value, value):
                    raise ValueError(
                        f"Grid mismatch for '{key}' while reading {beamfile}: "
                        f"{old_value} != {value}"
                    )
            else:
                if old_value != value:
                    raise ValueError(
                        f"Grid mismatch for '{key}' while reading {beamfile}: "
                        f"{old_value} != {value}"
                    )

    else:
        cimr["Grid"]["u"] = u
        cimr["Grid"]["v"] = v

        # Optional parameters, mainly to restore the grid if needed.
        cimr["Grid"]["xcen"] = xcen
        cimr["Grid"]["ycen"] = ycen
        cimr["Grid"]["xs"] = xs
        cimr["Grid"]["ys"] = ys
        cimr["Grid"]["nx"] = nx
        cimr["Grid"]["ny"] = ny
        cimr["Grid"]["dx"] = dx
        cimr["Grid"]["dy"] = dy

    # Add only the current polarization component.
    if pol == "H":
        cimr["Gain"]["G1h"] = Gco_real
        cimr["Gain"]["G2h"] = Gco_imag
        cimr["Gain"]["G3h"] = Gcx_real
        cimr["Gain"]["G4h"] = Gcx_imag

    elif pol == "V":
        cimr["Gain"]["G1v"] = Gco_real
        cimr["Gain"]["G2v"] = Gco_imag
        cimr["Gain"]["G3v"] = Gcx_real
        cimr["Gain"]["G4v"] = Gcx_imag

    cimr["Version"] = file_version

    return cimr

def recenter_beamdata(cimr: dict, logger: logging.Logger) -> dict:
    """
    Method to recenter original beam.

    The beam center in the (u,v) grid is dictated by xcen and ycen
    variables calculated above. However, as it turned out, the maximum
    beam value is not located in the center, but instead shifted in
    space. Therefore, we need to find where the maximum value is located
    on the u,v grid and offset u,v values to recenter the beam grid on
    beam's maximum value.

    Parameters:
    -----------
    cimr: dict
        Dictionary that contains beam data to be modified and returned.

    logger: logging.Logger
        Logger object to properly parse information.

    Returns:
    --------
    cimr: dict
        Dictionary that contains beam data to be modified and returned.
    """

    Ghh_max_index = utils.get_max_index(cimr["temp"]["Ghh"])
    Ghv_max_index = utils.get_max_index(cimr["temp"]["Ghv"])
    Gvv_max_index = utils.get_max_index(cimr["temp"]["Gvv"])
    Gvh_max_index = utils.get_max_index(cimr["temp"]["Gvh"])

    logger.info(f"Ghh_max_index = {Ghh_max_index}")
    logger.info(f"Ghv_max_index = {Ghv_max_index}")
    logger.info(f"Gvv_max_index = {Gvv_max_index}")
    logger.info(f"Gvh_max_index = {Gvh_max_index}")

    # Get the maximum value
    Ghh_max_value = cimr["temp"]["Ghh"][Ghh_max_index]
    Ghv_max_value = cimr["temp"]["Ghv"][Ghv_max_index]
    Gvv_max_value = cimr["temp"]["Gvv"][Gvv_max_index]
    Gvh_max_value = cimr["temp"]["Gvh"][Gvh_max_index]

    logger.info(f"Ghh_max_value = {Ghh_max_value}")
    logger.info(f"Ghv_max_value = {Ghv_max_value}")
    logger.info(f"Gvv_max_value = {Gvv_max_value}")
    logger.info(f"Gvh_max_value = {Gvh_max_value}")

    # Get the coordinates corresponding to maximum gain inside the mesh grids
    # (u, v). This is our new central value.
    u_coordinate = cimr["Grid"]["u_grid"][Ghh_max_index]
    v_coordinate = cimr["Grid"]["v_grid"][Ghh_max_index]
    logger.info(f"u_coordinate = {u_coordinate}")
    logger.info(f"v_coordinate = {v_coordinate}")

    # "Shift" is the distance between two coordinates (the center of the beam
    # and the coordinate that corresponds to its maximum gain value). So we
    # just take an absolute difference
    #
    # [Note]: Due to floating point precision, we can get crap after 15th
    # point, so I am cutting it off.
    u_shift = float(format(np.abs(cimr["Grid"]["xcen"] - u_coordinate), ".15f"))
    v_shift = float(format(np.abs(cimr["Grid"]["ycen"] - v_coordinate), ".15f"))
    logger.info(f"u_shift = {u_shift}")
    logger.info(f"v_shift = {v_shift}")

    # If the maximum gain coordinate is negative then we add the shift value
    # (go right to reach zero), else --- we subtract (go left).
    if u_coordinate < 0:
        cimr["Grid"]["u_grid"] = cimr["Grid"]["u_grid"] + u_shift
    else:
        cimr["Grid"]["u_grid"] = cimr["Grid"]["u_grid"] - u_shift

    if v_coordinate < 0:
        cimr["Grid"]["v_grid"] = cimr["Grid"]["v_grid"] + v_shift
    else:
        cimr["Grid"]["v_grid"] = cimr["Grid"]["v_grid"] - v_shift

    return cimr

def build_apat_name_info(
    beamfiles_paths: list[pb.Path],
    logger: logging.Logger = None,
) -> dict:
    """
    Deconstruct GRASP beam filenames and build a lookup dictionary that keeps
    the H and V polarization files separate.

    Output structure
    ----------------
    apat_name_info[band][horn][half_space][pol] = {
        "path": beamfile,
        "freq": freq,
    }

    Example
    -------
    apat_name_info["C"]["1"]["FR"]["H"]["path"] -> C1-6810-H-FR.grd
    apat_name_info["C"]["1"]["FR"]["V"]["path"] -> C1-6810-V-FR.grd
    """

    apat_name_info = {}

    for beamfile in beamfiles_paths:
        beamfile = pb.Path(beamfile)

        band, horn, freq, pol, half_space = io.parse_file_name(str(beamfile.stem))

        pol = pol.upper()
        half_space = half_space.upper()

        if pol not in {"H", "V"}:
            raise ValueError(
                f"Unsupported polarization '{pol}' in file {beamfile.name}. "
                "Expected 'H' or 'V'."
            )

        if band not in apat_name_info:
            apat_name_info[band] = {}

        if horn not in apat_name_info[band]:
            apat_name_info[band][horn] = {}

        if half_space not in apat_name_info[band][horn]:
            apat_name_info[band][horn][half_space] = {}

        if pol in apat_name_info[band][horn][half_space]:
            existing_file = apat_name_info[band][horn][half_space][pol]["path"]

            raise ValueError(
                f"Duplicate {pol}-polarization file found for "
                f"band={band}, horn={horn}, half_space={half_space}.\n"
                f"Existing file: {existing_file}\n"
                f"New file: {beamfile}"
            )

        apat_name_info[band][horn][half_space][pol] = {
            "path": beamfile,
            "freq": freq,
        }

        if logger is not None:
            logger.debug(
                f"Registered beam file: band={band}, horn={horn}, "
                f"freq={freq}, pol={pol}, half_space={half_space}, "
                f"path={beamfile}"
            )

    return apat_name_info


def run_cimr_grasp(
    datadir: str | pb.Path,
    outdir: str | pb.Path,
    file_version: str,
    beamfiles_paths: list,
    grid_max_theta: float = 90.0,
    grid_res_phi: float = 0.1,
    grid_res_theta: float = 0.1,
    chunk_data: bool = True,
    num_chunks: int = 4,
    overlap_margin: float = 0.1,
    interp_method: str = "linear",
    use_bhs: bool = False,
    recenter_beam: bool = True,
    use_rgb_logging: bool = False,
    use_rgb_decoration: bool = False,
    logger: logging.Logger = None,
) -> None:
    """
    Method performs the following steps:

    - Parsing original .grd file and saves into HDF5
    - Recentering the beam grid to center on the max gain value
    - Creating (theta, phi) grid with a given resolution and creating its (x,y) respresentation
    - Interpolating (in chunks or in full) the original (u,v) grid into coarser (x,y)
    - Saving the resulting data into HDF5 file

    [**Note**]: The data format is described in the `CIMR_Antenna_Patterns_Format.ipynb`
    located inside `notebooks` within the repo.

    Parameters:
    -----------
    datadir: str or Path
        The path to the data directory where all beam files are located.

    outdir: str or Path
        The path to the output directory where to store all results of execution.

    file_version: str
        Version of the parsed files to be produced.

    recenter_beam: bool
        Parameter that defines whether to recenter beam or not

    use_bhs: bool
        Parameter that defines whether to parse and preprocess BHS files or not.

    beamfiles_paths: list
        The list of full paths to all beam files for processing.

    grid_max_theta: float
        Maximum theta in the output grid (useful to save memory for small antenna patterns)

    grid_res_phi: float
        The grid resolution for phi angle.

    grid_res_theta: float
        The grid resolution for theta angle.

    chunk_data: bool
        Whether to perform chunking of the data for interpolation.

    num_chunks: int
        Number of chunks to split data into.

    overlap_margin: float
        The percentage overlap between neighboring chunks.

    interp_method: str
        Interpolation method to use. Possible values are: linear, nearest, and
        cubic. See scipy docs for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    use_rgb_logging: bool
        Whether to use RGB Logging configuration.

    use_rgb_decoration: bool
        Whether to use RGB Logging decorator.

    logger: logging.Logger
        Logger object to properly parse information.
    """

    # ========================
    # Parsing Antenna Patterns
    # ========================

    # Reconstructing the file names to operate on necessary parts later on
    apat_name_info = {}

    # for beamfile in beamfiles_paths:
    #     tobesplit = str(beamfile.stem)
    #     band, horn, freq, pol, half_space = io.parse_file_name(tobesplit)
    #
    #     if band not in apat_name_info:
    #         apat_name_info[band] = [freq, pol, {}]
    #
    #     if horn not in apat_name_info[band][2]:
    #         apat_name_info[band][2][horn] = []
    #
    #     apat_name_info[band][2][horn].append(half_space)

    apat_name_info = build_apat_name_info(
        beamfiles_paths=beamfiles_paths,
        logger=logger,
    )

    logger.info("==============================")
    logger.info("Parsing the Antenna Patterns")
    logger.info("==============================")

    # Creating directory to store parsed files
    parsed_dir = outdir.joinpath("parsed", f"v{file_version}")
    io.rec_create_dir(parsed_dir, logger=logger)

    preprocessed_dir = outdir.joinpath("preprocessed", f"v{file_version}")
    io.rec_create_dir(preprocessed_dir, logger=logger)

    logger.info(
        f"{Fore.GREEN}Data Directory:{Fore.RESET}\n| {Fore.BLUE}{datadir}{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.GREEN}Parsed Directory:{Fore.RESET}\n| {Fore.BLUE}{parsed_dir}{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.GREEN}Preprocessed Directory:{Fore.RESET}\n| {Fore.BLUE}{preprocessed_dir}{Style.RESET_ALL}"
    )

    # Main parsing + preprocessing loop
    required_pols = {"H", "V"}

    required_gain_keys = {
        "G1h", "G2h", "G3h", "G4h",
        "G1v", "G2v", "G3v", "G4v",
    }

    for band in apat_name_info.keys():
        for horn, half_spaces_dict in apat_name_info[band].items():
            for half_space, pol_files in half_spaces_dict.items():

                # Skip BHS/BK unless explicitly requested.
                if not use_bhs and half_space == "BK":
                    logger.info(f"| use_bhs = {use_bhs}; skipping {half_space}.")
                    continue

                missing_pols = required_pols - set(pol_files.keys())

                if missing_pols:
                    raise FileNotFoundError(
                        f"Missing polarization files for "
                        f"band={band}, horn={horn}, half_space={half_space}. "
                        f"Missing: {sorted(missing_pols)}. "
                        f"Available: {sorted(pol_files.keys())}"
                    )

                infile_h = pol_files["H"]["path"]
                infile_v = pol_files["V"]["path"]

                logger.info(
                    f"{Fore.YELLOW}------------------------------{Style.RESET_ALL}"
                )
                logger.info(
                    f"{Fore.GREEN}Working with H Input File: "
                    f"{infile_h.name}{Style.RESET_ALL}"
                )
                logger.info(
                    f"{Fore.GREEN}Working with V Input File: "
                    f"{infile_v.name}{Style.RESET_ALL}"
                )

                # ------------------------------------------------------------
                # Output filenames
                # ------------------------------------------------------------

                horn_output = str(int(horn) - 1)

                parsedfile_prefix = f"CIMR-OAP-{half_space}"
                parsedfile_suffix = "UV"

                outfile_oap = pb.Path(
                    str(parsed_dir)
                    + f"/{parsedfile_prefix}-"
                    + band
                    + horn_output
                    + f"-{parsedfile_suffix}v{file_version}.h5"
                )

                preprocfile_prefix = f"CIMR-PAP-{half_space}"
                preprocfile_suffix = "TP"

                outfile_pap = pb.Path(
                    str(preprocessed_dir)
                    + f"/{preprocfile_prefix}-"
                    + band
                    + horn_output
                    + f"-{preprocfile_suffix}v{file_version}.h5"
                )

                # ------------------------------------------------------------
                # Step 1: parse original GRASP files into OAP/UV HDF5
                # ------------------------------------------------------------

                if io.check_outfile_existance(outfile_oap):
                    logger.info(
                        f"{Fore.BLUE}Parsed file already exists: "
                        f"{outfile_oap.name}{Style.RESET_ALL}"
                    )

                    logger.info("Loading existing parsed object...")

                    start_time_load = time.perf_counter()

                    with h5py.File(outfile_oap, "r") as hdf5_file:
                        cimr = io.load_hdf5_to_dict(hdf5_file)

                    end_time_load = time.perf_counter() - start_time_load
                    logger.info(f"Finished loading parsed object in: {end_time_load:.2f}s")

                else:
                    cimr = {"Gain": {}, "Grid": {}}

                    logger.info("------------------------------")
                    logger.info("Parsing H and V polarizations")
                    logger.info("------------------------------")

                    start_time_pars = time.perf_counter()

                    cimr = get_beamdata(
                        infile_h,
                        "H",
                        half_space,
                        file_version,
                        cimr,
                        logger,
                    )

                    cimr = get_beamdata(
                        infile_v,
                        "V",
                        half_space,
                        file_version,
                        cimr,
                        logger,
                    )

                    missing_gain_keys = required_gain_keys - set(cimr["Gain"].keys())

                    if missing_gain_keys:
                        raise ValueError(
                            f"Incomplete cimr['Gain'] for "
                            f"band={band}, horn={horn}, half_space={half_space}. "
                            f"Missing keys: {sorted(missing_gain_keys)}"
                        )

                    end_time_pars = time.perf_counter() - start_time_pars
                    logger.info(f"Finished Parsing in: {end_time_pars:.2f}s")

                    logger.info(
                        f"{Fore.BLUE}Saving Output File: "
                        f"{outfile_oap.name}{Style.RESET_ALL}"
                    )

                    with h5py.File(outfile_oap, "w") as hdf5_file:
                        io.save_dict_to_hdf5(hdf5_file, cimr)

                # ------------------------------------------------------------
                # Step 2: preprocess OAP/UV into PAP/TP
                # ------------------------------------------------------------

                if io.check_outfile_existance(outfile_pap):
                    logger.info(
                        f"{Fore.BLUE}Preprocessed file already exists: "
                        f"{outfile_pap.name}{Style.RESET_ALL}"
                    )
                    continue

                logger.info("------------------------------")
                logger.info("Preparing complex gains")
                logger.info("------------------------------")

                # Validate again before interpolation.
                missing_gain_keys = required_gain_keys - set(cimr["Gain"].keys())

                if missing_gain_keys:
                    raise ValueError(
                        f"Cannot preprocess incomplete cimr['Gain'] for "
                        f"band={band}, horn={horn}, half_space={half_space}. "
                        f"Missing keys: {sorted(missing_gain_keys)}"
                    )

                # Creating temporary complex gains:
                #   Ghh, Ghv, Gvv, Gvh
                cimr["temp"] = {}
                cimr = utils.construct_complete_gains(cimr)

                # Reconstruct full u/v grids from parsed metadata.
                cimr["Grid"]["u_grid"], cimr["Grid"]["v_grid"] = (
                    utils.generate_uv_grid(
                        xcen=cimr["Grid"]["xcen"],
                        ycen=cimr["Grid"]["ycen"],
                        xs=cimr["Grid"]["xs"],
                        ys=cimr["Grid"]["ys"],
                        nx=cimr["Grid"]["nx"],
                        ny=cimr["Grid"]["ny"],
                        dx=cimr["Grid"]["dx"],
                        dy=cimr["Grid"]["dy"],
                    )
                )

                # Optional recentering.
                if recenter_beam:
                    logger.info("------------------------------")
                    logger.info("ReCentering")
                    logger.info("------------------------------")

                    start_time_recen = time.perf_counter()

                    cimr = recenter_beamdata(cimr, logger)

                    end_time_recen = time.perf_counter() - start_time_recen
                    logger.info(f"Finished Recentering in: {end_time_recen:.2f}s")

                logger.info("------------------------------")
                logger.info("Interpolating")
                logger.info("------------------------------")

                start_time_interpn = time.perf_counter()

                cimr = utils.interp_beamdata_into_uv(
                    cimr=cimr,
                    logger=logger,
                    grid_max_theta=grid_max_theta,
                    grid_res_phi=grid_res_phi,
                    grid_res_theta=grid_res_theta,
                    chunk_data=chunk_data,
                    num_chunks=num_chunks,
                    overlap_margin=overlap_margin,
                    interp_method=interp_method,
                )

                end_time_interpn = time.perf_counter() - start_time_interpn
                logger.info(f"Finished Interpolation in: {end_time_interpn:.2f}s")

                logger.info(
                    f"{Fore.BLUE}Saving Output File: "
                    f"{outfile_pap.name}{Style.RESET_ALL}"
                )

                with h5py.File(outfile_pap, "w") as hdf5_file:
                    io.save_dict_to_hdf5(hdf5_file, cimr)

                logger.info(
                    f"| {Fore.YELLOW}------------------------------{Style.RESET_ALL}"
                )

def load_grasp_config(config_file: str = "grasp_config.xml") -> dict:
    """
    Loads configuration data from an XML file and parses it into a dictionary.

    This function reads an XML file specified by `config_file`, parses its content,
    and returns a dictionary containing the configuration values.

    Parameters:
    -----------
    config_file : str
        The path to the XML configuration file. Defaults to "grasp_config.xml".

    Returns:
    --------
    dict
        A dictionary containing configuration data extracted from the XML file.

    Exceptions:
    -----------
    FileNotFoundError
        Raised if the specified XML file cannot be found.
    ET.ParseError
        Raised if there is an error parsing the XML file.
    FileNotFoundError
        Raised if the data directory does not exist.
    FileNotFoundError
        Raised if the data directory is empty.
    """

    try:
        # Parse the XML file
        tree = ET.parse(config_file)
        root = tree.getroot()

    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")
    except ET.ParseError:
        raise ET.ParseError(
            f"There was an error parsing the configuration file {config_file}."
        )

    config = {}

    # Read the paths
    # getting value of datadir, and if doesn't exist, returning an error.
    config["datadir"] = io.resolve_config_path(
        path_string=pb.Path(root.find("paths/datadir").text)
    )
    if not config["datadir"].is_dir():
        raise FileNotFoundError(f"The directory '{config['datadir']}' does not exist.")
    if not any(config["datadir"].iterdir()):
        raise FileNotFoundError(f"The directory '{config['datadir']}' is empty.")

    # getting value of outdir, checking for errors and creating directories
    # recursively if they do not exist
    config["outdir"] = io.resolve_config_path(
        path_string=root.find("paths/outdir").text
    )
    io.rec_create_dir(config["outdir"])

    # Read the parameters
    parameters = root.find("parameters")

    # Bool params
    config["use_bhs"] = io.get_bool_from_string(
        par_name="use_bhs", par_val=parameters.find("use_bhs").text
    )
    config["recenter_beam"] = io.get_bool_from_string(
        par_name="recenter_beam", par_val=parameters.find("recenter_beam").text
    )
    config["chunk_data"] = io.get_bool_from_string(
        par_name="chunk_data", par_val=parameters.find("chunk_data").text
    )
    # Float params
    config["grid_max_theta"] = float(parameters.find("grid_max_theta").text)
    config["grid_res_phi"] = float(parameters.find("grid_res_phi").text)
    config["grid_res_theta"] = float(parameters.find("grid_res_theta").text)
    config["overlap_margin"] = float(parameters.find("overlap_margin").text)

    # Int params
    config["num_chunks"] = int(parameters.find("num_chunks").text)

    # Str params
    config["interp_method"] = parameters.find("interp_method").text
    config["file_version"] = parameters.find("file_version").text

    # Read the logging settings
    logging = root.find("logging")
    config["use_rgb_logging"] = io.get_bool_from_string(
        par_name="use_rgb_logging", par_val=logging.find("use_rgb_logging").text
    )
    config["use_rgb_decoration"] = io.get_bool_from_string(
        par_name="use_rgb_logging", par_val=logging.find("use_rgb_decoration").text
    )
    config["logger_config"] = io.resolve_config_path(
        path_string=root.find("logging/logger_config").text
    )

    return config


def main():
    """
    Entry point of the program.
    """

    start_time_tot = time.perf_counter()
    # -----------------------------
    # Default value for config file
    config_file = pb.Path("configs", "grasp_config.xml").resolve()

    # Getting the value for parameter file from cmd
    parser = argparse.ArgumentParser(description="Update XML configuration parameters.")
    # Will use the default value of config_file if none is provided via command line:
    # https://docs.python.org/3/library/argparse.html#nargs
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the XML parameter file.",
        nargs="?",
        default=config_file,
    )

    args = parser.parse_args()
    config_file = io.resolve_config_path(args.config_file)

    # -----------------------------

    # Params from parameter file
    config = load_grasp_config(config_file=config_file)

    outdir = config["outdir"]
    datadir = config["datadir"]
    use_bhs = config["use_bhs"]
    recenter_beam = config["recenter_beam"]
    grid_max_theta = config["grid_max_theta"]
    grid_res_phi = config["grid_res_phi"]
    grid_res_theta = config["grid_res_theta"]
    chunk_data = config["chunk_data"]
    num_chunks = config["num_chunks"]
    overlap_margin = config["overlap_margin"]
    interp_method = config["interp_method"]
    file_version = config["file_version"]
    # Logging functionality
    use_rgb_logging = config["use_rgb_logging"]
    use_rgb_decoration = config["use_rgb_decoration"]
    logger_config = config["logger_config"]

    logdir = config["outdir"].joinpath("logs")
    io.rec_create_dir(logdir)

    # Creating a logger object based on the user preference
    if use_rgb_logging and logger_config is not None:
        rgb_logging = RGBLogging(logdir=logdir, log_config=logger_config)
        rgb_logger = rgb_logging.get_logger("rgb-logger")
    else:
        rgb_logger = logging.getLogger(__name__)
        rgb_logger.addHandler(logging.NullHandler())

    # -----------------------------

    rgb_logger.debug("Starting the script using the following libraries:")
    rgb_logger.debug(f"numpy      {np.__version__}")
    rgb_logger.debug(f"scipy      {sp.__version__}")
    rgb_logger.debug(f"h5py       {h5py.__version__}")
    rgb_logger.debug(f"tqdm       {tqdm.__version__}")
    # rgb_logger.debug(f"matplotlib {matplotlib.__version__}")

    # Parameters with which CIMR GRASP is to be run
    rgb_logger.info("---------")

    rgb_logger.info("CIMR GRASP Configuration")

    rgb_logger.info("---------")

    rgb_logger.info(f"Output Directory:        {outdir}")
    rgb_logger.info(f"Data Directory:          {datadir}")
    rgb_logger.info(f"Use BHS:                 {use_bhs}")
    rgb_logger.info(f"Recenter Beam:           {recenter_beam}")
    rgb_logger.info(f"Grid max Theta:          {grid_max_theta}")
    rgb_logger.info(f"Grid Resolution (Phi):   {grid_res_phi}")
    rgb_logger.info(f"Grid Resolution (Theta): {grid_res_theta}")
    rgb_logger.info(f"Chunk Data:              {chunk_data}")
    rgb_logger.info(f"Number of Chunks:        {num_chunks}")
    rgb_logger.info(f"Overlap Margin:          {overlap_margin}")
    rgb_logger.info(f"Interpolation Method:    {interp_method}")
    rgb_logger.info(f"File Version:            {file_version}")

    rgb_logger.info(f"Use CIMR RGB Logger :    {use_rgb_logging}")
    rgb_logger.info(f"Use CIMR RGB Decoration: {use_rgb_decoration}")
    rgb_logger.info(f"Logger Config:           {logger_config}")

    rgb_logger.info("---------")

    # Getting all beam paths inside dpr/AP
    beamfiles_paths = datadir.glob("*/*")

    run_cimr_grasp(
        datadir=datadir,
        outdir=outdir,
        file_version=file_version,
        beamfiles_paths=beamfiles_paths,
        use_bhs=use_bhs,
        recenter_beam=recenter_beam,
        grid_max_theta=grid_max_theta,
        grid_res_phi=grid_res_phi,
        grid_res_theta=grid_res_theta,
        chunk_data=chunk_data,
        num_chunks=num_chunks,
        overlap_margin=overlap_margin,
        interp_method=interp_method,
        use_rgb_logging=use_rgb_logging,
        use_rgb_decoration=use_rgb_decoration,
        logger=rgb_logger,
    )
    end_time_tot = time.perf_counter() - start_time_tot
    rgb_logger.info(f"Finished Script in: {end_time_tot:.2f}s")
    rgb_logger.info(f"------------------------------")


if __name__ == "__main__":
    main()

