(sec-intro)=
# Introduction

## Context

The CIMR Re-Gridding Toolbox (RGB) is an open-source Python toolbox for regridding conically scanned microwave radiometer data, developed for ESA
in the EXPRO project CIMR RGB Re-Gridding Tool Box {doc}`[AD-1]<03_references>`. The project concluded in April 2025 with the initial release
of the toolbox (v1.0.0, 15/04/2025) and version 1.2 of this ATBD.

Since then, the RGB has been used within the CIMR Level-2 product Algorithm Development (L2PAD) project:

- the RGB algorithms were reviewed against the needs of the CIMR Level-2 ground prototype processors, and the resulting gap analysis and
  implementation strategies are described in the ATBD for RGB evolution {doc}`[RD-6]<03_references>`;
- the RGB provides the regridding algorithm of the CIMR L1C prototype processor, with the Backus-Gilbert algorithm as baseline. How the RGB is
  applied to produce the L1C is described in the {doc}`CIMR L1C ATBD <l1c:intro>` {doc}`[RD-5]<03_references>`.

The further development of the toolbox will be carried out in a Contract Change Notice (CCN) for continued development, expected to start in
Q4 2026 ({numref}`Chapter %s <sec-roadmap>`).

(sec-purpose-scope)=
## Purpose and Scope

This document contains the Algorithm Theoretical Baseline Document (ATBD) for the CIMR RGB. It serves as a reference for the development,
implementation and theoretical background of the algorithms used for regridding data with the RGB. The primary purpose is to provide a
detailed description of the theoretical basis and implementation strategies for each algorithm included in the toolbox, as well as their
limitations and assumptions. The included algorithms can be broadly divided into two categories:

1. those that do not require knowledge of the antenna patterns: Nearest Neighbour, Drop-in-the-Bucket and Inverse Distance Squared;
2. those that rely on knowledge of both the source and the target antenna patterns: Backus-Gilbert and radiometer Scatterometer Image Reconstruction.

This is version 2 of the ATBD. It converts version 1.2 to Jupyter Book format and describes the toolbox as it is currently implemented.
The changes with respect to version 1.2 are listed in {ref}`content:changelog`.

This document describes the algorithms of the RGB in general. Some choices and parameters for the CIMR L1C product are given in the
CIMR L1C ATBD {doc}`[RD-5]<03_references>`, which is referred to where relevant. The installation and use of the toolbox, and its input and
output data, are described in the Software User Manual {doc}`[RD-3]<03_references>` and the Input/Output Data Definition {doc}`[RD-4]<03_references>`.

(sec-doc-structure)=
## Document Structure

- {doc}`03_references` lists the applicable and reference documents.
- {doc}`04_output_grids_and_map_projections` describes the map projections and grids used in the RGB.
- {doc}`05_regridding_algorithms` describes the regridding algorithms, including their theoretical basis, limitations, assumptions and
  implementation strategies, and concludes with the propagation of the NEDT and uncertainties through the regridding.
- {doc}`06_implementation` discusses the implementation details related to the regridding algorithms, including the antenna patterns and the
  target patterns.
- {doc}`07_roadmap` gives the planned further development of the toolbox.
- {ref}`content:changelog` gives the change log of the document.
