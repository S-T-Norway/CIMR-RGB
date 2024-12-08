# CIMR-RGB

<div align="justify">

The **Remapping/re-Gridding Toolbox (RGB)** is a software suite designed to
perform remapping of CIMR Level-1b (L1b) products, enabling the generation of
inputs suitable for Level-2 (L2) processors. Additionally, RGB supports the
creation of Level-1c (L1c) products, which involve re-gridding L1b sample data
onto an Earth-projected fixed grid. This re-gridding ensures compatibility and
facilitates the comparison of CIMR data with data from other sources.

</div>

> This repo consists of two projects: CIMR GRASP (to parse and preprocess CIMR
> beam files produced via GRASP TICRA software into suitable format) and CIMR
> RGB (to perform analysis).

---

## Table of Contents

| [Installation](#Installation) | [Usage](#Usage) | [Testing](#Testing) | [Tutorials](#Tutorials) |

---

## Installation

The package is installable using `pip`. After configuring python virtual environment, do the following:

```
$ python -m pip install .
```

Or, to install package in editable mode, do:

```
$ python -m pip install -e .
```

<div align="justify">

[**Note**]: The nix files present in the directory **will not** install the
software, but will only configure python virtual environemnt for development.
To install the software on NixOS, insterested user can use `conda` via
`conda-shell` (see e.g. [NixOS 24.05 packages](https://search.nixos.org/packages?channel=24.05&show=conda&from=0&size=50&sort=relevance&type=packages&query=conda-shell+)).

</div>

<div align="justify">

To create python dev environment with nix package manager (assuming `flakes` are enabled), run:

</div>

```
$ nix develop .
```

If you are using `direnv`, just do:

```
$ direnv allow .
```

## Usage

> After installation to packages will be available to run: `cimr-rgb` and
> `cimr-grasp`. To run the project for remapping (or the preprocessing of
> antenna patterns), the config file is required. The template parameter files
> are located inside `configs` directory.

To simply run the software, do:

```
$ cimr-rgb path/to/config.xml
```

For CIMR RGB, the command line options are also available. To check the full list do:

```
$ cimr-rgb --help
```

## Testing

The test suite is available as optional dependency and can be run using `pytest` framework. User can install it by running:

```
$ python -m pip install -e .[tests]
```

To run the tests, simply do:

```
$ pytest -v
```

from within the root of the repo.

## Tutorials

Python notebooks with explanations are available inside the `notebooks` directory in the root of the repo.
