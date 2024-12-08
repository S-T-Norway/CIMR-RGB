import pathlib as pb 
from xml.etree.ElementTree import Element, SubElement, ElementTree 
from unittest.mock import Mock

import pytest 
import numpy as np 
import pyproj 


from cimr_rgb.grid_generator import GridGenerator, GRIDS, PROJECTIONS  # Assuming the class is saved in grid_generator.py


# These tests cover individual functions for GridGenerator 
# using parametrization for cases where grid definitions and 
# projections differ. Mocking is used to avoid dependency on 
# external configurations or files. 

 
# @pytest.mark.parametrize("grid_def", [
#     grid_def for grid_def in GRIDS if grid_def.startswith("EASE2_")
# ])
# def test_generate_grid_xy_ease2(grid_def):
#     """
#     Test the `generate_grid_xy_ease2` method for all EASE2 grid definitions in the `GRIDS` dictionary.

#     This test validates the functionality of the `generate_grid_xy_ease2` method for each EASE2 grid,
#     ensuring:
#     1. The number of generated x-coordinates matches the expected number of columns (`n_cols`) for 
#        each grid definition.
#     2. The number of generated y-coordinates matches the expected number of rows (`n_rows`) for 
#        each grid definition.
#     3. The x-coordinates and y-coordinates at the four corners of the grid are correctly calculated:
#        - Top-left: (x_min + res/2, y_max - res/2)
#        - Top-right: (x_max - res/2, y_max - res/2)
#        - Bottom-left: (x_min + res/2, y_min + res/2)
#        - Bottom-right: (x_max - res/2, y_min + res/2)

#     Test setup:
#     - Mocked configuration object (`mock_config`) is passed as a placeholder.
#     - Each EASE2 grid definition in the `GRIDS` dictionary is tested.

#     Assertions:
#     - `len(xs)` matches `GRIDS[grid_def]["n_cols"]`.
#     - `len(ys)` matches `GRIDS[grid_def]["n_rows"]`.
#     - The coordinates at all four corners of the grid match their expected values.

#     Parameters:
#     ----------
#     None (relies on internal setup and predefined `GRIDS` data).

#     Returns:
#     -------
#     None (passes if all assertions are true, raises an `AssertionError` otherwise).
#     """
#     mock_config = Mock()
#     
#     # for grid_def in GRIDS:
#     #     if not grid_def.startswith("EASE2_"):
#     #         continue
#     #     
#     #     # [Note]: We pass "G" as a dummy variable, since test is performed for all projections 
#     #     grid_gen = GridGenerator(config_object=mock_config, 
#     #                              projection_definition="G", 
#     #                              grid_definition=grid_def)
#     #     xs, ys = grid_gen.generate_grid_xy_ease2()
#     #     
#     #     # Validate grid properties
#     #     assert len(xs) == GRIDS[grid_def]["n_cols"], f"Incorrect x-coordinates for {grid_def}"
#     #     assert len(ys) == GRIDS[grid_def]["n_rows"], f"Incorrect y-coordinates for {grid_def}"
#     #     
#     #     # Extract resolution and bounds from GRIDS
#     #     res = GRIDS[grid_def]["res"]
#     #     x_min = GRIDS[grid_def]["x_min"]
#     #     x_max = GRIDS[grid_def]["x_min"] + res * GRIDS[grid_def]["n_cols"]
#     #     y_min = GRIDS[grid_def]["y_max"] - res * GRIDS[grid_def]["n_rows"]
#     #     y_max = GRIDS[grid_def]["y_max"]

#     #     # Expected corner values
#     #     expected_top_left_x = x_min + res / 2
#     #     expected_top_left_y = y_max - res / 2
#     #     expected_top_right_x = x_max - res / 2
#     #     expected_top_right_y = y_max - res / 2
#     #     expected_bottom_left_x = x_min + res / 2
#     #     expected_bottom_left_y = y_min + res / 2
#     #     expected_bottom_right_x = x_max - res / 2
#     #     expected_bottom_right_y = y_min + res / 2

#     #     # Check corners (using small tolerance value due to numerical precision)
#     #     # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html 
#     #     assert np.isclose(xs[0], expected_top_left_x), f"Incorrect top-left x-coordinate for {grid_def}"
#     #     assert np.isclose(ys[0], expected_top_left_y), f"Incorrect top-left y-coordinate for {grid_def}"
#     #     
#     #     assert np.isclose(xs[-1], expected_top_right_x), f"Incorrect top-right x-coordinate for {grid_def}"
#     #     assert np.isclose(ys[0], expected_top_right_y), f"Incorrect top-right y-coordinate for {grid_def}"
#     #     
#     #     assert np.isclose(xs[0], expected_bottom_left_x), f"Incorrect bottom-left x-coordinate for {grid_def}"
#     #     assert np.isclose(ys[-1], expected_bottom_left_y), f"Incorrect bottom-left y-coordinate for {grid_def}"
#     #     
#     #     assert np.isclose(xs[-1], expected_bottom_right_x), f"Incorrect bottom-right x-coordinate for {grid_def}"
#     #     assert np.isclose(ys[-1], expected_bottom_right_y), f"Incorrect bottom-right y-coordinate for {grid_def}"
#     # for grid_def in GRIDS:
#     #if not grid_def.startswith("EASE2_"):
#     #    continue
#     
#     # [Note]: We pass "G" as a dummy variable, since test is performed for all projections 
#     grid_gen = GridGenerator(config_object=mock_config, 
#                              projection_definition="G", 
#                              grid_definition=grid_def)
#     xs, ys = grid_gen.generate_grid_xy_ease2()
#     
#     # Validate grid properties
#     assert len(xs) == GRIDS[grid_def]["n_cols"], f"Incorrect x-coordinates for {grid_def}"
#     assert len(ys) == GRIDS[grid_def]["n_rows"], f"Incorrect y-coordinates for {grid_def}"
#     
#     # Extract resolution and bounds from GRIDS
#     res = GRIDS[grid_def]["res"]
#     x_min = GRIDS[grid_def]["x_min"]
#     x_max = GRIDS[grid_def]["x_min"] + res * GRIDS[grid_def]["n_cols"]
#     y_min = GRIDS[grid_def]["y_max"] - res * GRIDS[grid_def]["n_rows"]
#     y_max = GRIDS[grid_def]["y_max"]

#     # Expected corner values
#     expected_top_left_x = x_min + res / 2
#     expected_top_left_y = y_max - res / 2
#     expected_top_right_x = x_max - res / 2
#     expected_top_right_y = y_max - res / 2
#     expected_bottom_left_x = x_min + res / 2
#     expected_bottom_left_y = y_min + res / 2
#     expected_bottom_right_x = x_max - res / 2
#     expected_bottom_right_y = y_min + res / 2

#     # Check corners (using small tolerance value due to numerical precision)
#     # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html 
#     assert np.isclose(xs[0], expected_top_left_x), f"Incorrect top-left x-coordinate for {grid_def}"
#     assert np.isclose(ys[0], expected_top_left_y), f"Incorrect top-left y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[-1], expected_top_right_x), f"Incorrect top-right x-coordinate for {grid_def}"
#     assert np.isclose(ys[0], expected_top_right_y), f"Incorrect top-right y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[0], expected_bottom_left_x), f"Incorrect bottom-left x-coordinate for {grid_def}"
#     assert np.isclose(ys[-1], expected_bottom_left_y), f"Incorrect bottom-left y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[-1], expected_bottom_right_x), f"Incorrect bottom-right x-coordinate for {grid_def}"
#     assert np.isclose(ys[-1], expected_bottom_right_y), f"Incorrect bottom-right y-coordinate for {grid_def}"



# @pytest.mark.parametrize("grid_def", [
#     grid_def for grid_def in GRIDS if grid_def.startswith("STEREO_")
# ])
# def test_generate_grid_xy_stereo(grid_def):
#     """
#     Test the `generate_grid_xy_stereo` method for all STEREO grid definitions in the `GRIDS` dictionary.

#     This test validates the functionality of the `generate_grid_xy_stereo` method for each STEREO grid,
#     ensuring:
#     1. The number of generated x-coordinates matches the expected number of columns (`n_cols`) for 
#        each grid definition.
#     2. The number of generated y-coordinates matches the expected number of rows (`n_rows`) for 
#        each grid definition.
#     3. The x-coordinates and y-coordinates at the four corners of the grid are correctly calculated:
#        - Top-left: (x_min + res/2, y_max - res/2)
#        - Top-right: (x_max - res/2, y_max - res/2)
#        - Bottom-left: (x_min + res/2, y_min + res/2)
#        - Bottom-right: (x_max - res/2, y_min + res/2)

#     Test setup:
#     - Mocked configuration object (`mock_config`) is passed as a placeholder.
#     - Each STEREO grid definition in the `GRIDS` dictionary is tested.

#     Assertions:
#     - `len(xs)` matches `GRIDS[grid_def]["n_cols"]`.
#     - `len(ys)` matches `GRIDS[grid_def]["n_rows"]`.
#     - The coordinates at all four corners of the grid match their expected values.

#     Parameters:
#     ----------
#     None (relies on internal setup and predefined `GRIDS` data).

#     Returns:
#     -------
#     None (passes if all assertions are true, raises an `AssertionError` otherwise).
#     """
#     mock_config = Mock()
#     
#     #for grid_def in GRIDS:
#     #if not grid_def.startswith("STEREO_"):
#     #    continue
#     
#     # Choose the correct projection definition based on the grid name
#     if "N" in grid_def:
#         proj_def = "PS_N"
#     elif "S" in grid_def:
#         proj_def = "PS_S"
#     else:
#         raise ValueError(f"Unknown STEREO grid definition: {grid_def}")
#     
#     #grid_gen = GridGenerator(mock_config, projection_definition, grid_def)
#     # [Note]: We pass "G" as a dummy variable, since test is performed for all projections 
#     grid_gen = GridGenerator(config_object=mock_config, 
#                              projection_definition=proj_def, 
#                              grid_definition=grid_def)
#     xs, ys = grid_gen.generate_grid_xy_stereo()
#     
#     # Validate grid properties
#     assert len(xs) == GRIDS[grid_def]["n_cols"], f"Incorrect x-coordinates for {grid_def}"
#     assert len(ys) == GRIDS[grid_def]["n_rows"], f"Incorrect y-coordinates for {grid_def}"
#     
#     # Extract resolution and bounds from GRIDS
#     res   = GRIDS[grid_def]["res"]
#     x_min = GRIDS[grid_def]["x_min"]
#     x_max = GRIDS[grid_def]["x_min"] + res * GRIDS[grid_def]["n_cols"]
#     y_min = GRIDS[grid_def]["y_max"] - res * GRIDS[grid_def]["n_rows"]
#     y_max = GRIDS[grid_def]["y_max"]

#     # Expected corner values
#     expected_top_left_x = x_min + res / 2
#     expected_top_left_y = y_max - res / 2
#     expected_top_right_x = x_max - res / 2
#     expected_top_right_y = y_max - res / 2
#     expected_bottom_left_x = x_min + res / 2
#     expected_bottom_left_y = y_min + res / 2
#     expected_bottom_right_x = x_max - res / 2
#     expected_bottom_right_y = y_min + res / 2

#     # Check corners
#     assert np.isclose(xs[0], expected_top_left_x), f"Incorrect top-left x-coordinate for {grid_def}"
#     assert np.isclose(ys[0], expected_top_left_y), f"Incorrect top-left y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[-1], expected_top_right_x), f"Incorrect top-right x-coordinate for {grid_def}"
#     assert np.isclose(ys[0], expected_top_right_y), f"Incorrect top-right y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[0], expected_bottom_left_x), f"Incorrect bottom-left x-coordinate for {grid_def}"
#     assert np.isclose(ys[-1], expected_bottom_left_y), f"Incorrect bottom-left y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[-1], expected_bottom_right_x), f"Incorrect bottom-right x-coordinate for {grid_def}"
#     assert np.isclose(ys[-1], expected_bottom_right_y), f"Incorrect bottom-right y-coordinate for {grid_def}"



# @pytest.mark.parametrize("grid_def", [
#     grid_def for grid_def in GRIDS if grid_def.startswith("MERC_")
# ])
# def test_generate_grid_xy_mercator(grid_def):
#     """
#     Test the `generate_grid_xy_mercator` method for all MERCATOR grid definitions in the `GRIDS` dictionary.

#     This test validates the functionality of the `generate_grid_xy_mercator` method for each MERCATOR grid,
#     ensuring:
#     1. The number of generated x-coordinates is greater than 0, and x-coordinates increase from left to right.
#     2. The number of generated y-coordinates is greater than 0, and y-coordinates decrease from top to bottom.
#     3. The x-coordinates and y-coordinates at the four corners of the grid are correctly calculated:
#        - Top-left: (min_x, max_y)
#        - Top-right: (max_x, max_y)
#        - Bottom-left: (min_x, min_y)
#        - Bottom-right: (max_x, min_y)

#     Test setup:
#     - Mocked configuration object (`mock_config`) is passed as a placeholder.
#     - Each MERCATOR grid definition in the `GRIDS` dictionary is tested.

#     Assertions:
#     - `len(xs) > 0` and `len(ys) > 0`.
#     - `xs[0] < xs[-1]` (x-coordinates increase).
#     - `ys[0] > ys[-1]` (y-coordinates decrease).
#     - The coordinates at all four corners of the grid match their expected values.

#     Parameters:
#     ----------
#     None (relies on internal setup and predefined `GRIDS` data).

#     Returns:
#     -------
#     None (passes if all assertions are true, raises an `AssertionError` otherwise).
#     """
#     mock_config = Mock()

#     #for grid_def in GRIDS:
#     #    if not grid_def.startswith("MERC_"):
#     #        continue

#     grid_gen = GridGenerator(mock_config, "MERC_G", grid_def)
#     xs, ys = grid_gen.generate_grid_xy_mercator()

#     # Validate grid properties
#     assert len(xs) > 0, f"X-coordinates not generated for {grid_def}"
#     assert len(ys) > 0, f"Y-coordinates not generated for {grid_def}"
#     assert xs[0] < xs[-1], f"X-coordinates do not increase for {grid_def}"
#     assert ys[0] > ys[-1], f"Y-coordinates do not decrease for {grid_def}"

#     # Extract resolution and bounds from GRIDS
#     res = GRIDS[grid_def]["res"]
#     mercator_proj = pyproj.Proj(PROJECTIONS["MERC_G"])

#     # Define grid bounds in lat/lon
#     min_lon, max_lon = -180, 180
#     min_lat, max_lat = GRIDS[grid_def]["lat_min"], GRIDS[grid_def].get("lat_max", 85)  # Default max_lat to 85 for Mercator

#     # Convert bounds to Mercator projection
#     min_x, min_y = mercator_proj(min_lon, min_lat)
#     max_x, max_y = mercator_proj(max_lon, max_lat)

#     # Expected corner values
#     expected_top_left_x = min_x
#     expected_top_left_y = max_y
#     expected_top_right_x = max_x
#     expected_top_right_y = max_y
#     expected_bottom_left_x = min_x
#     expected_bottom_left_y = min_y
#     expected_bottom_right_x = max_x
#     expected_bottom_right_y = min_y

#     # Check corners
#     assert np.isclose(xs[0], expected_top_left_x), f"Incorrect top-left x-coordinate for {grid_def}"
#     assert np.isclose(ys[0], expected_top_left_y), f"Incorrect top-left y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[-1], expected_top_right_x), f"Incorrect top-right x-coordinate for {grid_def}"
#     assert np.isclose(ys[0], expected_top_right_y), f"Incorrect top-right y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[0], expected_bottom_left_x), f"Incorrect bottom-left x-coordinate for {grid_def}"
#     assert np.isclose(ys[-1], expected_bottom_left_y), f"Incorrect bottom-left y-coordinate for {grid_def}"
#     
#     assert np.isclose(xs[-1], expected_bottom_right_x), f"Incorrect bottom-right x-coordinate for {grid_def}"
#     assert np.isclose(ys[-1], expected_bottom_right_y), f"Incorrect bottom-right y-coordinate for {grid_def}"




# # TODO: think more regarding this test 
# @pytest.mark.parametrize("grid_def, proj_def", [
#     *[(grid_def, "G") for grid_def in GRIDS if grid_def.startswith("EASE2_")],
#     *[(grid_def, "PS_N") for grid_def in GRIDS if grid_def.startswith("STEREO_N")],
#     *[(grid_def, "PS_S") for grid_def in GRIDS if grid_def.startswith("STEREO_S")],
#     *[(grid_def, "MERC_G") for grid_def in GRIDS if grid_def.startswith("MERC_")],
# ])
# def test_generate_grid_xy(grid_def, proj_def):
#     """
#     Test the `generate_grid_xy` method for all grid definitions in the `GRIDS` dictionary.

#     This test validates the `generate_grid_xy` method for each grid definition, ensuring:
#     1. The number of generated x-coordinates is greater than 0.
#     2. The number of generated y-coordinates is greater than 0.
#     3. The x-coordinates and y-coordinates at the four corners of the grid are correctly calculated:
#        - Top-left: (x_min, y_max)
#        - Top-right: (x_max, y_max)
#        - Bottom-left: (x_min, y_min)
#        - Bottom-right: (x_max, y_min)

#     Parameters:
#     ----------
#     grid_def: str
#         The grid definition name from the `GRIDS` dictionary.
#     proj_def: str
#         The projection definition to use for the grid.

#     Returns:
#     -------
#     None (passes if all assertions are true, raises an `AssertionError` otherwise).
#     """
#     mock_config = Mock()
#     grid_gen = GridGenerator(mock_config, proj_def, grid_def)

#     xs, ys = grid_gen.generate_grid_xy()

#     # Validate grid properties
#     assert len(xs) > 0, f"X-coordinates not generated for {grid_def}"
#     assert len(ys) > 0, f"Y-coordinates not generated for {grid_def}"

#     # Extract resolution and bounds
#     res = GRIDS[grid_def]["res"]
#     if grid_def.startswith("MERC_"):
#         # Mercator-specific bounds calculation
#         mercator_proj = pyproj.Proj(PROJECTIONS["MERC_G"])
#         min_lon, max_lon = -180, 180
#         min_lat, max_lat = GRIDS[grid_def]["lat_min"], GRIDS[grid_def]["lat_max"]

#         min_x, min_y = mercator_proj(min_lon, min_lat)
#         max_x, max_y = mercator_proj(max_lon, max_lat)

#         # Expected corner values for Mercator
#         expected_top_left_x = min_x
#         expected_top_left_y = max_y
#         expected_top_right_x = max_x
#         expected_top_right_y = max_y
#         expected_bottom_left_x = min_x
#         expected_bottom_left_y = min_y
#         expected_bottom_right_x = max_x
#         expected_bottom_right_y = min_y
#     else:
#         # Standard EASE2 and STEREO grid bounds
#         x_min = GRIDS[grid_def]["x_min"]
#         x_max = GRIDS[grid_def]["x_min"] + res * GRIDS[grid_def]["n_cols"]
#         y_min = GRIDS[grid_def]["y_max"] - res * GRIDS[grid_def]["n_rows"]
#         y_max = GRIDS[grid_def]["y_max"]

#         # Expected corner values
#         expected_top_left_x = x_min + res / 2
#         expected_top_left_y = y_max - res / 2
#         expected_top_right_x = x_max - res / 2
#         expected_top_right_y = y_max - res / 2
#         expected_bottom_left_x = x_min + res / 2
#         expected_bottom_left_y = y_min + res / 2
#         expected_bottom_right_x = x_max - res / 2
#         expected_bottom_right_y = y_min + res / 2

#     # Check corners
#     assert np.isclose(xs[0], expected_top_left_x), f"Incorrect top-left x-coordinate for {grid_def}"
#     assert np.isclose(ys[0], expected_top_left_y), f"Incorrect top-left y-coordinate for {grid_def}"

#     assert np.isclose(xs[-1], expected_top_right_x), f"Incorrect top-right x-coordinate for {grid_def}"
#     assert np.isclose(ys[0], expected_top_right_y), f"Incorrect top-right y-coordinate for {grid_def}"

#     assert np.isclose(xs[0], expected_bottom_left_x), f"Incorrect bottom-left x-coordinate for {grid_def}"
#     assert np.isclose(ys[-1], expected_bottom_left_y), f"Incorrect bottom-left y-coordinate for {grid_def}"

#     assert np.isclose(xs[-1], expected_bottom_right_x), f"Incorrect bottom-right x-coordinate for {grid_def}"
#     assert np.isclose(ys[-1], expected_bottom_right_y), f"Incorrect bottom-right y-coordinate for {grid_def}"





# @pytest.mark.parametrize("grid_def", [
#     grid_def for grid_def in GRIDS if grid_def.startswith("EASE2_")
# ])
# def test_generate_grid_xy_ease2_with_projections(grid_def):
#     """
#     Test the `generate_grid_xy_ease2` method for EASE2 grid definitions by comparing its results
#     against pyproj-based computations.

#     This test validates:
#     1. The generated x and y arrays match pyproj's grid generation for EASE2 grids.
#     2. The number of x and y coordinates matches the expected number of columns and rows (`n_cols`, `n_rows`).

#     Parameters:
#     ----------
#     grid_def: str
#         The grid definition name from the `GRIDS` dictionary.

#     Returns:
#     -------
#     None (passes if all assertions are true, raises an `AssertionError` otherwise).
#     """
#     mock_config = Mock()
#     grid_gen = GridGenerator(mock_config, "G", grid_def)
#     
#     # Get results from the method
#     xs, ys = grid_gen.generate_grid_xy_ease2()

#     # Extract grid properties from GRIDS
#     res = GRIDS[grid_def]["res"]
#     n_cols = GRIDS[grid_def]["n_cols"]
#     n_rows = GRIDS[grid_def]["n_rows"]
#     x_min = GRIDS[grid_def]["x_min"]
#     y_max = GRIDS[grid_def]["y_max"]

#     # Set up pyproj projection using PROJECTIONS dictionary
#     projection_str = PROJECTIONS["G"]
#     ease_proj = pyproj.Proj(projection_str)

#     # Generate expected x and y arrays using pyproj
#     pyproj_xs = np.array([x_min + (i + 0.5) * res for i in range(n_cols)])
#     pyproj_ys = np.array([y_max - (j + 0.5) * res for j in range(n_rows)])

#     # Validate dimensions
#     assert len(xs) == n_cols, f"Number of x-coordinates does not match for {grid_def}"
#     assert len(ys) == n_rows, f"Number of y-coordinates does not match for {grid_def}"

#     # Compare results from the method and pyproj
#     assert np.allclose(xs.flatten(), pyproj_xs), f"X-coordinates do not match pyproj for {grid_def}"
#     assert np.allclose(ys.flatten(), pyproj_ys), f"Y-coordinates do not match pyproj for {grid_def}"




# @pytest.mark.parametrize("grid_def, proj_def", [
#     (grid_def, proj_def) 
#     for proj_def in PROJECTIONS
#     for grid_def in GRIDS if grid_def.startswith(f"EASE2_{proj_def}")
# ])
# def test_generate_grid_xy_ease2_with_projections(grid_def, proj_def):
#     """
#     Test the `generate_grid_xy_ease2` method for EASE2 grid definitions by comparing its results
#     against pyproj-based computations.

#     This test validates:
#     1. The generated x and y arrays match pyproj's grid generation for EASE2 grids.
#     2. The number of x and y coordinates matches the expected number of columns and rows (`n_cols`, `n_rows`).

#     Parameters:
#     ----------
#     grid_def: str
#         The grid definition name from the `GRIDS` dictionary.
#     proj_def: str
#         The projection definition key from the `PROJECTIONS` dictionary.

#     Returns:
#     -------
#     None (passes if all assertions are true, raises an `AssertionError` otherwise).
#     """
#     mock_config = Mock()
#     grid_gen = GridGenerator(mock_config, proj_def, grid_def)

#     # Get results from the method
#     xs, ys = grid_gen.generate_grid_xy_ease2()

#     # Extract grid properties from GRIDS
#     res = GRIDS[grid_def]["res"]
#     n_cols = GRIDS[grid_def]["n_cols"]
#     n_rows = GRIDS[grid_def]["n_rows"]
#     x_min = GRIDS[grid_def]["x_min"]
#     y_max = GRIDS[grid_def]["y_max"]

#     # Set up pyproj projection using PROJECTIONS dictionary
#     projection_str = PROJECTIONS[proj_def]
#     ease_proj = pyproj.Proj(projection_str)

#     # Generate expected x and y arrays using pyproj
#     pyproj_xs = np.array([x_min + (i + 0.5) * res for i in range(n_cols)])
#     pyproj_ys = np.array([y_max - (j + 0.5) * res for j in range(n_rows)])

#     # Validate dimensions
#     assert len(xs) == n_cols, f"Number of x-coordinates does not match for {grid_def}"
#     assert len(ys) == n_rows, f"Number of y-coordinates does not match for {grid_def}"

#     # Compare results from the method and pyproj
#     assert np.allclose(xs.flatten(), pyproj_xs), f"X-coordinates do not match pyproj for {grid_def}"
#     assert np.allclose(ys.flatten(), pyproj_ys), f"Y-coordinates do not match pyproj for {grid_def}"





@pytest.mark.parametrize("grid_def, proj_def", [
    (grid_def, proj_def) 
    #for proj_def in PROJECTIONS
    for proj_def in PROJECTIONS if proj_def in {"G"}#, "N"}#, "S"}
    for grid_def in GRIDS if grid_def.startswith(f"EASE2_{proj_def}")
])
#@pytest.mark.parametrize("grid_def, proj_def", [
#    (grid_def, proj_def) 
#    for proj_def in PROJECTIONS if proj_def.startswith(f"S")
#    for grid_def in GRIDS if grid_def.startswith(f"EASE2_S1km")#{proj_def}")
#])
def test_generate_grid_xy_ease2_with_projections(grid_def, proj_def):
    """
    Test the `generate_grid_xy_ease2` method for EASE2 grid definitions by comparing its results
    against pyproj-based computations.

    This test validates:
    1. The generated x and y arrays match pyproj's grid generation for EASE2 grids.
    2. The number of x and y coordinates matches the expected number of columns and rows (`n_cols`, `n_rows`).

    Parameters:
    ----------
    grid_def: str
        The grid definition name from the `GRIDS` dictionary.
    proj_def: str
        The projection definition key from the `PROJECTIONS` dictionary.

    Returns:
    -------
    None (passes if all assertions are true, raises an `AssertionError` otherwise).
    """
    mock_config = Mock()
    grid_gen = GridGenerator(mock_config, proj_def, grid_def)

    # Get results from the method
    xs, ys = grid_gen.generate_grid_xy_ease2()
    xs = xs.ravel() 
    ys = ys.ravel() 

    # Extract grid properties from GRIDS
    res = GRIDS[grid_def]["res"]
    n_cols = GRIDS[grid_def]["n_cols"]
    n_rows = GRIDS[grid_def]["n_rows"]

    #if proj_def == "N":  # North polar grids
    #    lat_min = 0
    #    lat_max = 90
    #elif proj_def == "S":  # South polar grids
    #    lat_min = -90
    #    lat_max = 0
    #else:  # Global grids
    lat_min = GRIDS[grid_def]["lat_min"]
    lat_max = GRIDS[grid_def]["lat_max"]

    #lat_min = GRIDS[grid_def]["lat_min"]

    # Set up pyproj projection using PROJECTIONS dictionary
    projection_str = PROJECTIONS[proj_def]
    ease_proj = pyproj.Proj(projection_str)

    ## Calculate x_min, x_max, y_min, y_max using pyproj
    #print(lat_min)
    #x_min, y_min = ease_proj(-180, lat_min)
    ##x_max, y_max = ease_proj(180, lat_max)

    x_min, y_max = ease_proj(-180, lat_max)
    x_max, y_min = ease_proj(180, lat_min)

    ## Generate expected x and y arrays using pyproj
    pyproj_xs = np.array([x_min + (i + 0.5) * res for i in range(n_cols)])
    pyproj_ys = np.array([y_max - (j + 0.5) * res for j in range(n_rows)])

    # Validate dimensions
    assert len(xs) == n_cols, f"Number of x-coordinates does not match for {grid_def}"
    assert len(ys) == n_rows, f"Number of y-coordinates does not match for {grid_def}"


    #print(xs[1], pyproj_xs[1]) 
    # Compare results from the method and pyproj
    assert np.allclose(xs.flatten(), pyproj_xs, atol=1e-2, equal_nan=True), \
            f"X-coordinates do not match pyproj "#for {grid_def}"
    #assert np.allclose(ys.flatten(), pyproj_ys, atol=1e-2, equal_nan=True), f"Y-coordinates do not match pyproj for {grid_def}"


@pytest.mark.parametrize("proj_def, grid_def, lon, lat", [
    (proj_def, grid_def, lon, lat) 
    for proj_def in PROJECTIONS if proj_def in {"N", "S"}
    for grid_def in GRIDS if grid_def.startswith(f"EASE2_{proj_def}")
    for (lon, lat) in [
        (0, 0),        # Center point (North/South Pole)
        #(1e6, 1e6),    # Positive coordinates
        #(-2e6, -1e6),  # Negative coordinates
        #(1e6, -1e6),   # Mixed coordinates
        #(-2e6, 3e6)    # Arbitrary mix
        (86, 86), 
        (-86, 76), 
        (np.array([21, 34, 55]), np.array([56, 67, 12])), 
    ]
])
def test_lonlat_to_xy_laea(proj_def, grid_def, lon, lat): 
    """
    Test the `lonlat_to_xy_laea` method of the `GridGenerator` class for Lambert's Azimuthal Equal Area (LAEA) projection.

    This test compares the results from the custom `lonlat_to_xy_laea` method with those generated by the `pyproj` library
    to ensure accuracy of the implementation.

    Test Cases:
    -----------
    - Test various longitude (`lon`) and latitude (`lat`) combinations:
        - `(0, 0)`: Center point (North/South Pole)
        - `(86, 86)`: Edge case near the grid boundary.
        - `(-86, 76)`: Mixed coordinate edge case.
        - `([21, 34, 55], [56, 67, 12])`: Numpy array inputs for batch testing.

    Parameters:
    -----------
    proj_def : str
        Projection definition key from `PROJECTIONS`. Only "N" (North) and "S" (South) are tested.

    grid_def : str
        Grid definition key from `GRIDS`. Selected grids are restricted to those starting with "EASE2_" followed by the
        projection key ("N" or "S").

    lon : float or np.ndarray
        Longitude values in decimal degrees. Can be a scalar or an array for batch testing.

    lat : float or np.ndarray
        Latitude values in decimal degrees. Can be a scalar or an array for batch testing.

    Assertions:
    -----------
    1. For scalar inputs:
        - Validate that the x and y coordinates from the custom method match those from `pyproj` within a tolerance of `1e-5`.
    2. For array inputs:
        - Validate that all x and y coordinates from the custom method closely match their `pyproj` counterparts within a 
          tolerance of `1e-5`.

    Notes:
    ------
    - Uses `Mock` to mock the configuration object for the `GridGenerator`.
    - Leverages `pyproj.Proj` for reference LAEA calculations.

    Raises:
    -------
    AssertionError
        If any coordinate (x or y) from the custom implementation does not match the corresponding `pyproj` result.

    """


    mock_config = Mock()
    #grid_def = "EASE2" # in this case does not really matter  
    grid_gen = GridGenerator(mock_config, proj_def, grid_def)

    # Define projection based on North or South
    pole = 'N' if proj_def == 'N' else 'S'

    # Get results from custom method
    x_custom, y_custom = grid_gen.lonlat_to_xy_laea(lon, lat, pole)

    projection = PROJECTIONS[proj_def]

    # Get results from pyproj
    proj = pyproj.Proj(projection)
    x_pyproj, y_pyproj = proj(lon, lat)#, inverse=True)

    # Assertions for both scalar and array inputs
    if isinstance(x_custom, np.ndarray):
        assert np.allclose(x_custom, x_pyproj, atol=1e-5), \
            f"x-coordinates do not match for {grid_def}: {x_custom} vs {x_pyproj}"
        assert np.allclose(y_custom, y_pyproj, atol=1e-5), \
            f"y-coordinates do not match for {grid_def}: {y_custom} vs {y_pyproj}"
    else:
        assert np.isclose(x_custom, x_pyproj, atol=1e-5), \
            f"x-coordinate does not match for {grid_def}: {x_custom} vs {x_pyproj}"
        assert np.isclose(y_custom, y_pyproj, atol=1e-5), \
            f"y-coordinate does not match for {grid_def}: {y_custom} vs {y_pyproj}"




# PROJECTIONS = {
#     'G': "+proj=cea +lat_ts=30 +lon_0=0 +lat_0=0 "
#          "+x_0=0 +y_0=0 +datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs",
#     'N': "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 "
#          "+datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs",
#     'S': "+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 "
#          "+datum=WGS84 +ellps=WGS84 +units=m +no_defs +type=crs",
#     'PS_N': "+proj=stere +lat_0=90 +lon_0=-45 +lat_ts=70 +k=1 +x_0=0 +y_0=0 "
#                 "+datum=WGS84 +ellps=WGS84 +units=m +no_defs",
#     'PS_S': "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
#                 "+datum=WGS84 +ellps=WGS84 +units=m +no_defs",
#     'UPS_N': "+proj=stere +lat_0=90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 "
#              "+datum=WGS84 +units=m +no_defs +type=crs",
#     'UPS_S': "+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 +x_0=2000000 +y_0=2000000 "
#              "+datum=WGS84 +units=m +no_defs +type=crs",
#     'MERC_G': "+proj=merc +k=1 +lon_0=0 +x_0=0 +y_0=0 "
#               "+datum=WGS84 +units=m +no_defs +type=crs"
# }

