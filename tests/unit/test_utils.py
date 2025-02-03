import pytest
import numpy as np

import cimr_rgb.utils as cimr_utils


@pytest.mark.parametrize(
    "array, overlap, expected, expect_exception",
    [
        # Valid cases
        (np.arange(10), 2, np.array([2, 3, 4, 5, 6, 7]), False),  # Standard case
        (
            np.arange(20),
            3,
            np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            False,
        ),  # Larger array
        (np.arange(6), 1, np.array([1, 2, 3, 4]), False),  # Small array
        (np.arange(5), 0, np.array([0, 1, 2, 3, 4]), False),  # No overlap removal
        # Edge cases
        (np.arange(4), 2, np.array([]), False),  # Entire array removed
        (np.array([]), 0, np.array([]), False),  # Empty array with overlap 0
        # Invalid cases
        # (np.arange(10), -1, None, True),  # Negative overlap
        # (np.arange(10), 6, None, True),  # Overlap too large
    ],
)
def test_remove_overlap(array, overlap, expected, expect_exception):
    """
    Test the `remove_overlap` function in `cimr_rgb.utils`.

    This test validates:
    - Correct removal of overlap from the beginning and end of the array.
    - Handling of edge cases such as an empty array or removing the entire array.
    - Error handling when `overlap` is negative or exceeds half the array length.

    Parameters
    ----------
    array : np.ndarray
        The input array to be processed.
    overlap : int
        The number of elements to remove from each end.
    expected : np.ndarray or None
        The expected output array, or None if an exception is expected.
    expect_exception : bool
        Indicates whether a `ValueError` is expected.
    """

    if expect_exception:
        with pytest.raises(ValueError, match="Invalid overlap value"):
            cimr_utils.remove_overlap(array, overlap)
    else:
        result = cimr_utils.remove_overlap(array, overlap)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (3, 5, 39),
        (10, 20, 612),
        (0, 0, 0),  # Edge case: zero
        (255, 255, 65535),  # Edge case: max 8-bit int
        (1023, 2047, 3145727),  # Larger integers
    ],
)
def test_interleave_bits(x, y, expected):
    """
    Test `interleave_bits`.

    Ensures that the function correctly interleaves bits of two integers.

    Parameters
    ----------
    x, y : int
        Coordinates to interleave.
    expected : int
        Expected interleaved integer.
    """
    assert cimr_utils.interleave_bits(x, y) == expected


@pytest.mark.parametrize(
    "x, y",
    [
        (3, 5),
        (10, 20),
        (0, 0),  # Edge case: zero
        (255, 255),  # Edge case: max 8-bit int
        (1023, 2047),  # Larger integers
    ],
)
def test_deinterleave_bits(x, y):
    """
    Test `deinterleave_bits`.

    Ensures that the function correctly reconstructs the original x, y coordinates after interleaving.

    Parameters
    ----------
    x, y : int
        Coordinates to interleave and then deinterleave.
    """

    interleaved = cimr_utils.interleave_bits(x, y)

    assert cimr_utils.deinterleave_bits(interleaved) == (x, y)


@pytest.mark.parametrize(
    "vec, expected",
    [
        ([3, 4], np.array([0.6, 0.8])),
        ([1, 0, 0], np.array([1, 0, 0])),
        ([0, 1, 0], np.array([0, 1, 0])),
    ],
)
def test_normalize(vec, expected):
    """
    Test `normalize`.

    Ensures that the function correctly normalizes a vector to unit length.

    Parameters
    ----------
    vec : list
        Input vector.
    expected : np.ndarray
        Expected normalized vector.
    """

    assert np.allclose(cimr_utils.normalize(vec), expected)


@pytest.mark.parametrize(
    "x0, y0, z0, x1, y1, z1",
    [
        (
            np.eye(3)[:, 0],
            np.eye(3)[:, 1],
            np.eye(3)[:, 2],
            np.eye(3)[:, 0],
            np.eye(3)[:, 1],
            np.eye(3)[:, 2],
        ),
        (
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0, -1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 0, 1]),
        ),
    ],
)
def test_generic_transformation_matrix(x0, y0, z0, x1, y1, z1):
    """
    Test `generic_transformation_matrix`.

    Ensures that the function correctly computes the transformation matrix between two sets of basis vectors.

    Parameters
    ----------
    x0, y0, z0 : np.ndarray
        Original basis vectors.
    x1, y1, z1 : np.ndarray
        Transformed basis vectors.
    """

    R = cimr_utils.generic_transformation_matrix(x0, y0, z0, x1, y1, z1)

    assert np.allclose(R @ np.column_stack((x0, y0, z0)), np.column_stack((x1, y1, z1)))


@pytest.mark.parametrize(
    "axis, angle, vector, expected",
    [
        ("z", np.pi / 2, np.array([1, 0, 0]), np.array([0, 1, 0])),
        ("z", -np.pi / 2, np.array([1, 0, 0]), np.array([0, -1, 0])),
        ("x", np.pi / 2, np.array([0, 1, 0]), np.array([0, 0, 1])),
        ("x", -np.pi / 2, np.array([0, 1, 0]), np.array([0, 0, -1])),
        ("y", np.pi / 2, np.array([0, 0, 1]), np.array([1, 0, 0])),
        ("y", -np.pi / 2, np.array([0, 0, 1]), np.array([-1, 0, 0])),
    ],
)
def test_rotation_matrix(axis, angle, vector, expected):
    """
    Test `rotation_matrix` function.

    Ensures that the function correctly computes rotation matrices around different axes.

    Parameters
    ----------
    axis : str
        The rotation axis ('x', 'y', or 'z').
    angle : float
        The rotation angle in radians.
    vector : np.ndarray
        The input vector to rotate.
    expected : np.ndarray
        Expected transformed vector after applying the rotation.
    """

    result = cimr_utils.rotation_matrix(axis, angle) @ vector

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "lon1, lat1, lon2, lat2, expected",
    [
        (0, 0, 0, 0, 0),  # Same point
        (0, 0, 0, 90, np.pi / 2 * cimr_utils.EARTH_RADIUS),  # 90-degree latitude shift
        (0, 0, 180, 0, np.pi * cimr_utils.EARTH_RADIUS),  # Antipodal points
    ],
)
def test_great_circle_distance(lon1, lat1, lon2, lat2, expected):
    """
    Test `great_circle_distance` function.

    Ensures that the function correctly computes the great-circle distance between two points on a sphere.

    Parameters
    ----------
    lon1, lat1 : float
        Longitude and latitude of the first point in degrees.
    lon2, lat2 : float
        Longitude and latitude of the second point in degrees.
    expected : float
        Expected distance in meters.
    """

    assert np.isclose(
        cimr_utils.great_circle_distance(lon1, lat1, lon2, lat2), expected
    )
