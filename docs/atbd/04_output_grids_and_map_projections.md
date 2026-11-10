(sec-grids)=
# Output Grids and Map Projections
The RGB offers a variety of output grids and map projections for L1C interpolations. 
An overview of grids and projections available to the user are provided in the CIMR RGB Software
Installation and User manual {doc}`[RD-3]<03_references>`, Tables 2 and 3 respectively. This Section describes the
implementation of these grids within the processing pipeline. 

The [`grid_generator`](https://github.com/S-T-Norway/CIMR-RGB/blob/main/src/cimr_rgb/grid_generator.py)
module contains the definition of available output grids and projections as well as functionality to 
create grids and convert coordinates between different projections. 

(sec-grid-generation)=
## Generating grids

An example of a grid definition string is provided below for an EASE2 Global 9km grid (`EASE2_G9km`):

```python
'EASE2_G9km': {
    'epsg': 6933,
    'x_min': -17367530.44,
    'y_max': 7314540.83,
    'res': 9008.05,
    'n_cols': 3856,
    'n_rows': 1624,
    'lat_min': -86,
    'lat_max': 86}
```
Upon initiation of the RGB, the user selected grid is queried and samples from the L1b data are removed
if they fall out of the boundaries of the grid. 

It is necessary for a grid to be generated in order to interpolate L1B BTs onto. The grids are generated in a Cartesian coordinate system 
defined by the input projection (`epsg`), where the upper left corner is equal to the grid definition string's (`x_min`,`y_max`), the coordinates
of the centre of each grid cell, which are the target locations of the interpolation, are defined as:

$$
xs[i_{col}] = x_{min} + res/2 + (i_{col}*res)
$$ (eq-grid-x)

$$
ys[i_{row}] = y_{max} - res/2 - (i_{row}*res)
$$ (eq-grid-y)

for

$$
i_{col} = 0,..., n_{cols} - 1, i_{row} = 0,...,n_{rows} - 1
$$ (eq-grid-res)

The procedure outlined above is used for all EASE2 and stereographic grids, for which commonly used
grids have already been defined for this purpose. For the Mercator grids, in the absence of a standard
grid used by the community, it was necessary to create a custom definition. This was achieved by selecting the
longitude and latitude bounds for which the grid is required, converting to the Mercator projection and creating 
the correct number of rows and columns that fit the desired resolution.

Once the grid coordinates have been created, these are considered the target locations and are passed 
through the pipeline. Upon completion of the interpolation, a `cell_row` and `cell_col` variable is provided
with the output data (indexing every BT to a grid cell), allowing the user to easily reconstruct the data
onto the chosen grid.

(sec-coord-conversion)=
## Converting coordinates between projections

The [`grid_generator`](https://github.com/S-T-Norway/CIMR-RGB/blob/main/src/cimr_rgb/grid_generator.py) module 
provides functionality to convert between (`lon`, `lat`), (`x`,`y`) and (`row`,`col`) for all provided projections. 
The module is frequently called throughout the RGB processing chain to convert between these coordinate systems when 
required or more convenient for a certain numerical operation. 

Each of the six projections offered comes with their own set of conversion formulae. For the projections used in the
EASE2 grids (Cylindrical Equal Area (CEA) and Lambert's Azimuthal Equal Area (LAEA)), the conversion formulae
were implemented directly according to the EASE2 definition {cite:p}`brodzik/etal:2012,nsidc:easeconv`. For the remainder of the projections (Stereographic and Mercator), the 
[`Proj`](https://github.com/OSGeo/PROJ) library is used. 

For a comprehensive overview of projections and their associated transformations, {cite:t}`snyder:1987:map_projections` serves as an invaluable resource. 

