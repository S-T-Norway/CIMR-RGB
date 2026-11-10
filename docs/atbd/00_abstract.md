# Abstract

This document is the Algorithm Theoretical Baseline Document (ATBD) of the CIMR Re-Gridding Toolbox (RGB), an open-source Python toolbox
for regridding the brightness temperatures (TBs) of conically scanning microwave radiometers. The RGB was developed for ESA to support the
Copernicus Imaging Microwave Radiometer (CIMR) mission, and takes CIMR, AMSR2 and SMAP L1B data as input. It produces two types of output: L1C, with
the TBs regridded onto Earth-fixed grids (such as EASE2) and map projections, and L1R, with the TBs of one band resampled onto the sample
locations of another band.

The toolbox includes two families of regridding algorithms. Nearest Neighbour, Drop-in-the-Bucket and Inverse Distance Squared combine the
surrounding samples based on their positions only. Backus-Gilbert (BG) and radiometer Scatterometer Image Reconstruction (rSIR) use the antenna patterns
of the samples, projected onto the Earth's surface, to combine them so that their combined response matches a prescribed target pattern. For
each algorithm, this document describes its theoretical basis, assumptions and limitations, and how the NEDT and uncertainties of the input samples
are propagated through the regridding. It also describes the output grids and map projections, and the implementation of the processing chain:
the selection of the input samples, the initialisation and projection of the antenna patterns, the integration grid and the target patterns.

This is version 2 of the ATBD, converted to Jupyter Book format from version 1.2, the version delivered with the initial release of the
toolbox. The RGB now provides the regridding of the CIMR L1C prototype processor developed within the CIMR L2 Product Algorithm
Development (L2PAD) project, and the choices made for the L1C product are described in the CIMR L1C ATBD. The further development of the
toolbox is planned in a Contract Change Notice (CCN) for continued development.
