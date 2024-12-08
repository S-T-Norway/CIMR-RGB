# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0rc1] - 2024-12-08

### Added

- **Remapping and Regridding Algorithms**:
  - Implemented 6 different algorithms:
    - Backus Gilbert (BG) inversion
    - radiometric Scaterometer Image Reconstruction (RSIR)
    - Landweber (LW)s. [**Note**]: Landweber algorithm is implemented in iterative_methods, but not tested or working on the full pipeline.
    - Conjugate Gradients (CG) methods
    - Inverse Distance Squared (IDS)
    - Nearest Neighbour (NN)
    - Drop-in-the-Bucket (DIB)
- **Map Projections**:
  - Added support for multiple map projections, including:
    - EASE2 grid:
      - Lambert's Azimuthal Equal Area (North and South)
      - Cylindrical Equal Area (Global)
    - Mercator (Global)
    - NSIDC grids: Polar Stereographic (North and South)
- **Radiometer Support**:
  - Integrated support for the following radiometers:
    - CIMR
    - AMSR2
    - SMAP
- **Antenna Patterns**:
  - Ability to simulate Gaussian antenna patterns.
  - Support for working with existing patterns (e.g., CIMR and SMAP).
- **Preprocessing Module**:
  - Added standalone preprocessing module for converting CIMR GRASP files into standardized HDF5 patterns.
  - This module ingests its own configuration file (found in `configs``).
- **Logging**:
  - Added configurable logging support for better debugging and tracing.
- **Testing**:
  - Implemented testing functionality using the Pytest library.
- **Documentation**:
  - Added a Jupyter notebook with (partial) theoretical explanations examples.
- **Executable**:
  - Developed a standalone executable:
    - Accepts configuration XML files as mandatory input.
    - Supports command-line arguments for additional flexibility.

### Changed

- Initial implementation of the library and executable.

### Removed

- N/A for this release.

---
