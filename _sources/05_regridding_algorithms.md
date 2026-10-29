# 4. Regridding Algorithms
The RGB provides functionality for a variety of regridding/ interpolation methods, whose theoretical basis are described in this Section. 

All algorithms, except for NN, rely on multiple input samples to perform interpolation. The selection of these input samples
is determined by two configuration parameters: `search_radius` and `max_neighbours`. 

- `search_radius` specifies the radius from the target location within which the algorithm searches for neighboring samples.
- `max_neighbours` defines the maximum number of neighbours that shall be used for the interpolation. 

If the number of neighbours found within a specified `search_radius` exceeds the value of `max_neighbors`, the algorithm will select
only the closest `N = max_neighbors` samples for interpolation. 


## 4.1. Nearest Neighbour


### 4.1.1. Statement of Function

The Nearest Neighbour (NN) regridding algorithm assigns each point in the target grid the value of the closest point in 
the source grid. This method is especially useful when precise, discrete values need to be preserved, as it does not involve
interpolation or averaging. 

### 4.1.2. Heritage

NN doesn't traditionally have a heritage in formal satellite products due to it's simplicity and non-smoothing characteristics. 
It is, however, the fastest algorithm and is useful for sanity checks in the development of more complext algorithms, or when 
using new data. 

### 4.1.3. Limitations and Assumptions

This approach is fast and simple but will likely introduce abrupt changes and artefacts in the output data due to the lack 
of smoothing. Moreover, there will be no reduction in noise by averaging. It also throws away the vast majority of available input
data, leading to potentially inaccurate output results. However, the effective antenna pattern is only broadened to the size of 
the output grid cell, as opposed to other methods whose effective antenna pattern is largely widened by considering points
far away from the target location.

### 4.1.4. Approximations

No approximations are made for this algorithm.

### 4.1.5. Algorithm Definition
The NN algorithm assigns each point in the target grid the value of the closest point in the source grid. If there are $N$ source samples $\{T_{B1}, \ldots, T_{BN}\}$ that fall within a search radius of a
given cell, the corresponding gridded $T_B$ is expressed as

$$
T_{Bg} = T_{Bj}
$$ (eq-nn)

where $T_{Bj}$ represents the single sample that is radially closest to the grid cell centre.


### 4.1.6. References

This Section is intentionally left blank. 

## 4.2. Drop-in-the-Bucket

### 4.2.1. Statement of Function

The Drop-in-the-Bucket (DIB) algorithm aggregates data from the source grid into the cells of the target grid, effectively
"dropping" data points into the nearest target cell without weighting (i.e all have equal weights).

### 4.2.2. Heritage

The algorithm doesn't have significant heritage in comparable products but is frequently used in literature to compare more complext algorithms. 


### 4.2.3. Limitations and Assumptions

While the effective noise fluctuations in the source measurements are reduced in the averaging
process, the effective antenna pattern can be significantly broadened due to equal weights given to
source samples that may lie near the cell boundaries, resulting in the gridded result containing
radiance from outside the target cell/ search area.

### 4.2.4. Approximations

No approximations were made.

### 4.2.5. Algorithm Definition

If there are $N$ source samples $\{T_{B1}, \ldots, T_{BN}\}$ that fall within a target cell/search radius, the
corresponding gridded $T_B$ is expressed as

$$
T_{Bg} = \frac{1}{N}\sum_{i=1}^{N} \alpha_i T_{Bi}
$$ (eq-average)

where all $\alpha_i$ are equal to 1.

The RGB offers the DIB algorithm to be applied within a chosen search radius or directly within the cells (squared) themselves. 
Suppose there are more neighbours in the search radius than the specified $N$ = `max_neighbours`.
Then, the algorithm will consider the closest $N$ neighbours.  

### 4.2.6. References

This Section is intentionally left blank.

## 4.3. Inverse Distance Squared

### 4.3.1. Statement of Function

The Inverse Distance Squared (IDS) regridding algorithm interpolates values from the source grid to
the target grid by weighting each source point's influence on the target cell inversely to the square of
the distance. Closer points have a much stronger influence than those far away, resulting in smooth
gradients across the grid.

### 4.3.2. Heritage

This algorithm is often used in microwave radiometry applications, most notably in the SMAP L1C
Radiometer Half-Orbit 36 km EASE-Grid Brightness Temperatures (BTs) {doc}`[RD-4]<./03_references>`, {doc}`[RD-5]<./03_references>` product.

### 4.3.3. Limitations and Assumptions

IDS can introduce smoothing that can mask finer details in the data, especially in the case where
source data is unevenly distributed, as is of course the case for conically scanning radiometers. This
algorithm is limited in cases where finer resolution detail is required and/or when an appropriate
number of neighbouring samples are not within an appropriate proximity to the target cell. This is
due to the fact that the sample temperatures are treated as discrete temperatures at that location,
as opposed to methods utilising antenna patterns, which allow the distribution of temperatures over
larger regions.

### 4.3.4. Approximations

No approximations were made. 

### 4.3.5. Algorithm Definition

The Inverse Distance Squared method assigns weights to each measurement identified in the points
selection depending on its vicinity to the centre of the target output cell.

$$
T_{B_{target}} = \frac{1}{A}\sum_{i=1}^{N} \alpha_i T_{B_{i_{source}}}
$$ (eq-idw-a)

$$
A = \sum_{i=1}^{N} \alpha_i
$$ (eq-idw-b)

$$
\alpha_i = \frac{1}{d_i^2}
$$ (eq-idw-c)

where $d$ is the orthodromic distance based on the Earth's sphere.

### 4.3.6. References

The reader is referred to {doc}`[RD-4]<./03_references>` and {doc}`[RD-5]<./03_references>` for more information. 

## 4.4. Backus Gilbert

### 4.4.1. Statement of Function

The Backus-Gilbert (BG) regridding algorithm interpolates values from the source grid to the target
grid using a weighted average of the source points. The weights are chosen to minimise the fit
error, accounting for the projection of the source and target antenna gain patterns onto the Earth's
surface. The goal is to reproduce, as closely as possible, what the radiometer would have measured
had it been pointed directly at the interpolation point. The algorithm takes advantage of the
non-uniform spatial sampling characteristic of conical microwave radiometer measurements, which
oversample in certain regions.

### 4.4.2. Heritage

The Backus-Gilbert interpolation algorithm was first developed in the 1960's in order to better
understand the internal structure of the Earth. Since its first development, the BG method has been
widely used in a variety of scientific fields, mostly in geophysics and, more recently, in the field of
spatial data interpolation and resolution enhancement. The algorithm was selected for
implementation in the RGB as a result of a literature review of state-of-the-art algorithms, detailed in
the Comparative Performance Assessment Plan {doc}`[AD-2]<./03_references>`.

In the last 15 years or so, BG has been extensively researched in the application of microwave
radiometry and applied as the chosen interpolation method for a variety of satellite data products.
Most notably, the BG algorithm was used in the creation of the “GCOM-W1 AMSR2 Level 1R
Product” {doc}`[RD-6]<./03_references>` and the “SMAP Enhanced L1C Radiometer Half-Orbit 9 km EASE-Grid Brightness
Temperatures” {doc}`[RD-7]<./03_references>` and has also been applied to the Microwave Imager (MWI) and Ice Cloud Imager (ICI)
L2 products {doc}`[RD-8]<./03_references>`.

### 4.4.3. Limitations and Assumptions

Despite its powerful capabilities, the Backus-Gilbert method has faced challenges in widespread
adoption, primarily due to its computational cost and complexity in implementation, especially for
large-scale problems.

Implementations of the Backus-Gilbert method in satellite data processing have had to address
issues such as artefact production due to poor condition numbers in matrix inversions. This is
especially true when the number of neighbours taken into account for the interpolation is large,
making the matrix inversion more computationally intensive and affected by uncertainties.

As multiple input source samples are used for the interpolation, the temperatures are averaged
temporally over these source samples. It is, therefore, assumed that the surface characteristics
remain constant over this period. This assumption is, of course, more applicable in the case where
adjacent samples from the same scan are considered. However, there could be significantly more
time in-between two samples in which one was taken from a forward-looking part of one scan and
the backward-looking part of another. The user should consider this in their choice of configuration
parameters.

### 4.4.4. Approximations

Several approximations are required in order to implement the BG algorithm for the interpolation of
radiometric data, in particular when there is a large number of high-resolution measurements,
entailing a higher computational load.

Antenna patterns for CIMR are provided with high resolution (< 0.01 deg for K and Ka bands) through
the full forward hemisphere. Projecting all the gain values on Earth's surface for each source pattern
is not feasible, and the following strategies are adopted to decrease the computational load:

- Antenna patterns are sub-sampled to lower resolution
- Gain values smaller than a threshold values are set to zero
- Antenna patterns are projected to an integration grid on the Earth's surface (with resolution being a user-defined parameter), which could downgrade further the antenna pattern
resolution

Antenna temperature measurements, described by 4 Stokes parameters, are, in principle, obtained
via a radiative transfer calculation, which can be described with a Mueller Matrix. The full Mueller
matrix for the SMAP and CIMR instrument is provided and can be included in the BG algorithm when
interpolating the 4 Stokes parameters. However, this significantly increases the amount of operations
in the projection step since all 16 values of the Mueller matrix need to be projected to Earth and
recomputed in a Ludwig-3 polarisation basis relative to the Earth's surface. Therefore, an
approximation is performed that derives a scalar value from the Mueller matrix to characterise the
antenna gain, as described in Section XXX. This considerably speeds up the calculation, with the
drawback that all 4 Stokes parameters will be interpolated with the same scalar value of the gain,
neglecting the different response of the feedhorn to different polarisation components.

### 4.4.5. Algorithm Definition

The $T_B$ measured by a receiving antenna is expressed as

$$
T_B = \iint G(x,y) T_B(x,y)\, dx\, dy = \int G(\rho) T_B(\rho)\, ds
$$ (eq-bg-1)

where $G(x,y) = G(\rho)$ is the projected antenna pattern on Earth, i.e., the gain of the antenna at
position $(x,y) = \rho$ on the Earth's Surface, and $T_B(x,y) = T_B(\rho)$ is the microwave energy emitted
from the point $\rho$. $G(\rho)$ is the normalised gain value of the antenna, where $\int G(\rho)\, dS = 1$.

For the interpolation of an output grid cell, multiple source samples and their respective antenna
patterns are considered, each with varying positions and rotations on the 2D surface. The set of
source patterns considered is expressed as:

$$
T_{B_{source,i}} = \int G_{source,i}(\rho) T_{B_{source,i}}(\rho)\, dS
$$ (eq-bg-2)

where $G_i$ $(i = 1, \ldots, N)$.

The target measurement, $T_{B_{target}}$, is measured by another receiving antenna pattern. Depending on
the desired output grid type, this could be the real or simulated antenna pattern of the target band
in the case of an L1R remap or a gaussian fit to the size of the output cell in the case of an L1C
remap. The target measurement is expressed as

$$
T_{B_{target}} = \int F_{target}(\rho) T_{B_{target}}(\rho)\, dS
$$ (eq-bg-3)

where $F(\rho)$ is the target antenna pattern and $\int F(\rho)\, dS = 1$.

$T_{B_{target}}$ is a weighted combination of the source samples expressed as

$$
T_{B_{target}} = \sum_{i=1}^{N} a_i T_{B_{source,i}} = \sum_{i=1}^{N} a_i \left( \int_S G_i(\rho) T_{B_{source,i}}(\rho)\, dS \right)
$$ (eq-bg-4)

where $a_i$ are weighting factors of the source samples, that can be computed to minimise the fit error.

The ideal $a_i$ should satisfy a fit error of zero,

$$
F(\rho) - \sum_{i=1}^{N} a_i G_i(\rho) = 0
$$ (eq-bg-5)

The resulting equation for the weighting factors (written in matrix form) is

$$
\mathbf{a} = \mathbf{V}^{-1} \left[ \mathbf{v} + \left( \frac{1 - \mathbf{u}^T \mathbf{V}^{-1} \mathbf{v}}{\mathbf{u}^T \mathbf{V}^{-1} \mathbf{u}} \right) \mathbf{u} \right]
$$ (eq-bg-6)

where

$$
\mathbf{V} = \mathbf{G} + k\mathbf{I}
$$ (eq-bg-7)

$$
u_i = \int G_i(\rho)\, dS
$$ (eq-bg-8)

$$
v_i = \int G_i(\rho) F(\rho)\, dS
$$ (eq-bg-9)

$$
G_{ij} = \int G_i(\rho) G_j(\rho)\, dS
$$ (eq-bg-10)

$k$ is a factor multiplied by the identity matrix $I$, which varies from 0 to infinity and smooths the
variation of the coefficient $a_i$.

### 4.4.6. References

For further reading on the BG algorithm, we refer the reader to {doc}`[RD-9]<./03_references>`, {doc}`[RD-10]<./03_references>` , {doc}`[RD-8]<./03_references>`,
{doc}`[RD-11]<./03_references>`, {doc}`[RD-12]<./03_references>`, {doc}`[RD-6]<./03_references>`, {doc}`[RD-12]<./03_references>`, {doc}`[RD-7]<./03_references>` and  {doc}`[RD-13]<./03_references>`.

## 4.5. Radiometer Scatterometer Image Reconstruction

### 4.5.1. Statement of Function

The Radiometer Scatterometer Image Reconstruction (rSIR) method is an iterative image reconstruction technique. 
rSIR makes use of a surface temperature distribution defined by the instruments antenna patterns. The algorithm uses 
regularisation in order to trade-off between noise amplification and spatial resolution. It achieves this by selecting
an appropriate number of iterations, which consequently produces only a partial reconstruction of the image. 

The aim is to estimate the surface brightness temperature based on the noise source measurements. The algorithm makes
use of the instruments (or simulated) antenna patterns, which translate to the Measurement Response Function (MRF) on the Earth’s surface and is therefore highly sensitive to
processing decisions pertaining to this. The MRF is calculated using the satellite position, antenna
scan angle, and the antenna gain pattern. The processing of antenna patterns and their associated
MRF’s is discussed in Section XXX. 

### 4.5.2. Heritage

rSIR was selected as an outcome of a literature study on state-of-the-art regridding algorithms
outlined in the Comparative Performance Assessment Plan {doc}`[AD-02]<./03_references>`. The studies by Long et al. 
{doc}`[RD-14]<./03_references>`, {doc}`[RD-15]<./03_references>` applied to SMAP data demonstrate that in comparison 
with other image production methods, rSIR produces images with finer resolution and lower Root Mean Square Errors 
(RMSEs). rSIR, in particular, exhibits slightly lower noise in simulations and finer resolution by certain metrics. The
studies conclude that due to its lower computational requirements and superior performance, rSIR is
preferred for generating enhanced-resolution images for Calibrated Passive Microwave Daily
Equal-Area Scalable Earth Grid 2.0 (EASE2) Brightness Temperature (CETB) products, specifically for
SMAP applications.
rSIR has been applied to SMAP data in a stand-alone NSIDC product, “SMAP Radiometer Twice-Daily
rSIR-Enhanced EASE-Grid 2.0 Brightness Temperatures” {doc}`[RD-16]<./03_references>`. This product (and its associated
documentation) was one of the main sources of reference in the implementation of this algorithm.

### 4.5.3. Limitations and Assumptions

As the rSIR algorithm is an iterative update of temperature distribution over a larger area
surrounding the target location, it is only appropriate to apply it to the brightness temperatures
(`bt_h`, `bt_v`, `bt_3`, `bt_4`). Therefore, all other variables apart from these use IDS for their regrid
when this algorithm is selected.
As multiple input source samples are used to reconstruct the image, the reconstructed image is
averaged temporally over these source samples. It is, therefore, assumed that the surface
characteristics remain constant over this period. This assumption is, of course, more applicable in the
case where adjacent samples from the same scan are considered. However, there could be
significantly more time in-between two samples in which one was taken from a forward-looking part
of one scan and the backward-looking part of another. The user should consider this in their choice
of configuration parameters.

### 4.5.4. Approximations

If a full reconstruction were employed, noise in the source samples could be excessively amplified.
The algorithm is therefore regularised through the iteration number, which effectively imposes a
smoothing constraint on the image.

The antenna pattern processing and, consequently, the MRF, contains a degree of approximation
regardless of the method used to generate it (users can employ instrument antenna patterns or
simulate a Gaussian; see the Software Installation and User Manual {doc}`[AD-01]<./03_references>` for more details). In all
cases, users must choose a gain threshold, below which the antenna pattern is ignored and not
considered part of the "footprint." This threshold is generally taken in the field to be 3 dB (half
power). Moreover, when "instrument" antenna patterns are used, some interpolation is applied to
the patterns before ingestion into the RGB, when "gaussian" or "gaussian_projected" patterns are
selected, the antenna pattern itself is naturally approximated by a Gaussian function (see Sections
XXX to XXX for details on this function). The MRF is further approximated in its superposition onto
the image domain/integration grid, the resolution of which is user-configurable.

### 4.5.5. Algorithm Definition

The target surface brightness temperature distribution $T_{B,target}(x, y)$ is treated as a discrete signal,
sampled at the chosen grid pixel spacing, and is estimated from $N$ source measurements $T_{B,source}$.
$T_{B,target}(x, y)$ is considered over an image domain, $N_x \times N_y$, large enough to contain all considered
samples and associated antenna patterns for the interpolation of a target output cell. $T_{B,target}(x, y)$
is vectorised over this domain, $a_j = T_{B,target}(x_l, y_k)$, where $j = l + N_x k$ and $l, k$ correspond to
the row and column number of the pixel in the image domain. The image domain is analogous to
what is otherwise referred to in this report as the 'integration grid'. The creation of this is described
in Section XXXX.

A particular measurement, $z_i$, can be expressed as

$$
z_i = \sum_{j \in image} h_{ij} a_j
$$ (eq-rsir-1)

where $h_{ij} = MRF(x_l, y_k)$, or the normalised antenna pattern projected to Earth evaluated at the $j^{th}$
pixel centre. The MRF is negligible at a certain distance from the pixel centre and is cut to a
user-selected value, and the sum is therefore only computed over an area local to the measurement
position. The $h_{ij}$ are normalised such that $\sum_j h_{ij} = 1$.

For all measurements considered for a target output cell,

$$
\vec{T} = \mathbf{H}\vec{a}
$$ (eq-rsir-2)

where $H$ contains the samples MRF for each measurement, and $T$ and $a$ are vectors composed of the
measurements $z_i$, and the samples surface brightness temperatures $a_j$, respectively.

The rSIR algorithm solves this equation iteratively. It approximates a maximum-entropy solution to
an underdetermined system and a least-squares solution to an overdetermined system.

The first iteration is a simple estimate of $T_{B,target}$ with the $j^{th}$ pixel, expressed as:

$$
a_j^0 = \frac{\sum_i h_{ij} z_i}{\sum_i h_{ij}}
$$ (eq-rsir-3)

At the $k$th $(k > 0)$ iteration, the forward projection and scale factor are defined as:

$$
f_i^k = \frac{\sum_n h_{in} a_n^k}{\sum_n h_{in}}
$$ (eq-fk)

$$
d_i^k = \sqrt{\frac{z_i}{f_i^k}}
$$ (eq-dk)

respectively, where $n$ is the index to each pixel in the image. The non-linear update term is expressed
as:

$$
u_{ij}^k =
\begin{cases}
\left[\dfrac{1}{2} f_i^k \left(1 - \dfrac{1}{d_i^k}\right) + \dfrac{1}{a_j^k d_i^k}\right]^{-1}, & \text{if } d_i^k \geq 1 \\[8pt]
\left[\dfrac{1}{2} f_i^k \left(1 - d_i^k\right) + a_j^k d_i^k\right], & \text{if } d_i^k < 1
\end{cases}
$$ (eq-rsir-4)

and the $j$th image pixel is computed as:

$$
a_j^{k+1} = \frac{\sum_i h_{ij} u_{ij}^k}{\sum_i h_{ij}}
$$ (eq-rsir-5)

This set of equations is iterated over $k$ for the user-selected $N$ iterations. The set of equations is
minimised by cutting the MRF to within a certain user-selected value of the peak response and set to zero elsewhere.
The set of input samples and associated MRFs input to the algorithm are calculated
based on the satellite position and scan angle and will have different characteristics (ellipticity,
rotation, footprint size) depending on satellite position, the band (antenna pattern) in question, and
gain threshold considered.

The final output temperature is calculated by combining the target MRF and iterated temperature
distribution:

$$
T_{B_{target}} = \sum_{j \in image} h_{j_{target}} a_j^{k+1}
$$ (eq-rsir-6)



### 4.5.6. References

For further reading on the rSIR algorithm, the reader is referred to {doc}`[RD-11]<./03_references>`, 
{doc}`[RD-17]<./03_references>`, and {doc}`[RD-18]<./03_references>`.


## 4.7. NEDT and Uncertainty Propagation
The determination of the uncertainties output as a result of the regridding algorithms is provided in this Section. The
uncertainty calculations are limited to the uncertainties added as a result of the regridding algorithm only. An
additional source of uncertainty in the regridded L1r/c measurement results from the uncertainty of
the antenna pattern gains, the interpolation of the antenna patterns to a coarser resolution, and the
projection of the antenna pattern to Earth.

As the antenna patterns utilised come without documentation/ description of uncertainty, and the
L1b data for all instruments is provided without uncertainty quantifications for the position, velocity,
or attitude of the satellite, it has been decided not to include these considerations in the
approximation.

A covariance matrix is required for uncertainty propagation for some of the regridding algorithms.
The L1b NEDT and the uncertainty correlation of neighbouring arc samples are unavailable at the time of writing. 
In their absence per band NEDT cited in the {doc}`[AD-03]<./03_references>` (Table MRD-2) is applied along the diagonals of the covariance matrix.
Moreover, the configuration parameter `<antenna_pattern_uncertainty>`, which defines a single value corresponding to the 
total uncertainty of the resulting antenna pattern and associated processing. When a value is used for this configurtion 
parameter, its value is added to the NEDTs described in this Section according to the following equation: 

$$
NEDT_{out} = \sqrt(NEDT_{BT}^{2} + NEDT_{AP}^{2})
$$ (eq-un-1)

where $$NEDT_{AP}$$ represents the uncertainty added as a result of the configuration parameter `<antenna_pattern_uncertainty>`.

### 4.7.1. Nearest Neighbour

As only a single sample is used in the grid, the resulting value retains the same noise as the input sample, expressed as: 

$$
\sigma_{eff}^{2} = \sigma^{2}
$$ (eq-un-2)

### 4.7.2. Drop-in-the-Bucket

The uncertainty of the gridded $$T_{B}$$ is expressed as:

$$
\sigma^2_{T_{Bg}} = \frac{1}{N^2} \sum_{i=1}^{N} \sigma^2_{T_{Bi}}
$$ (eq-un-3)

### 4.7.3. Inverse Distance Squared

The uncertainty of the output measurement is expressed as: 

$$
\sigma_{T_{Bgtarget}} =
\frac{1}{A}
\sqrt{
\sum_{i=1}^{N}
\alpha_i^2 \sigma^2_{T_{Bisource}}
}
$$ (eq-un-4)

### 4.7.4. Backus Gilbert

The resulting uncertainty of the output measurement is:

$$
\sigma_T^2 = \mathbf{a}^T \mathbf{E} \mathbf{a}
$$ (eq-un-5)

where $$\mathbf{E}$$ is the measurement error covariance matrix, and $$\mathbf{a}$$ is the vector of weights defined in 
Section XXX. In the absence of uncertainty correlations between neighbouring arc samples, the per band NEDTs stated in 
{doc}`[AD-03]<./03_references>` are used along the diagonals of the of $$\mathbf{E}$$.


### 4.7.5. Radiometer Scatterometer Image Reconstruction

Propagating the uncertainty through this algorithm is not trivial. This is due to the non-linear update
term, $u_{ij}^k$. Mathematical formulations for the propagation were considered; however, this poses
challenges in uncertainty as non-linear functions can distort input distributions, have variable
sensitivity across their domain, and introduce complex interactions between variables. Moreover, a
method could not be found in the literature considered for this report. In addition, there is no
mention of uncertainty in the rSIR SMAP product {doc}`[RD-16]<./03_references>`, {doc}`[RD-17]<./03_references>`, and uncertainty variables are
not provided in its output.

It was decided at this stage to propagate uncertainty for rSIR following the same propagation
formulas as IDS (see Section XXX). The proportion of the antenna pattern considered for the output
temperature is dependent on its proximity to the output location and it is therefore considered
reasonable to propagate the uncertainty in this way.


