"""
Utility functions for the RGB.
"""

import numpy as np

EARTH_RADIUS = 6378000


# def remove_overlap(array, overlap):
#     """
#     Removes overlap scans from the start and end of the array.

#     Parameters
#     ----------
#     array: np.ndarray
#         Array to remove overlap from
#     overlap: int
#         Number of scans to remove from the start and end of the array

#     Returns
#     -------
#     np.ndarray
#         Array with overlap scans removed
#     """
#     return delete(
#         array, r_[0:overlap, array.shape[0] - overlap : array.shape[0]], axis=0
#     )


def remove_overlap(array: np.ndarray, overlap: int) -> np.ndarray:
    """
    Removes overlap scans from the start and end of the array.

    Parameters
    ----------
    array : np.ndarray
        Array from which overlap scans will be removed.
    overlap : int
        Number of scans to remove from the start and end of the array.

    Returns
    -------
    np.ndarray
        Array with overlap scans removed.
    """

    if overlap < 0 or 2 * overlap > array.shape[0]:
        raise ValueError(
            "Invalid overlap value. It must be non-negative and <= half of array length."
        )
    return np.delete(
        array, np.r_[0:overlap, array.shape[0] - overlap : array.shape[0]], axis=0
    )


# def interleave_bits(x, y):
#     """
#     This function creates a unique integer from two coordinates by interleaving their bits.

#     Parameters
#     ----------
#     x: int
#         First coordinate
#     y: int
#         Second coordinate

#     Returns
#     -------
#     z: int
#         Unique integer created from the two coordinates
#     """
#     z = 0
#     for i in range(max(x.bit_length(), y.bit_length())):
#         z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1)
#     return int(z)


def interleave_bits(x: int, y: int) -> int:
    """
    Creates a unique integer by interleaving the bits of two integers.

    Parameters
    ----------
    x : int
        First coordinate.
    y : int
        Second coordinate.

    Returns
    -------
    int
        Unique integer generated from interleaving bits of x and y.
    """

    assert isinstance(x, int) and x >= 0, "x must be a non-negative integer."
    assert isinstance(y, int) and y >= 0, "y must be a non-negative integer."

    z = 0
    for i in range(max(x.bit_length(), y.bit_length())):
        z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1)
    return z


# def deinterleave_bits(z):
#     """
#     This function extracts the two coordinates from a unique integer created by interleave_bits.

#     Parameters
#     ----------
#     z: int
#         Unique integer created from two coordinates

#     Returns
#     -------
#     x: int
#         First coordinate
#     y: int
#         Second coordinate
#     """
#     x = y = 0
#     for i in range(0, z.bit_length()):
#         # Extract the bit at position i and shift it to the right position
#         # For x, we take every second bit starting from the LSB (least significant bit)
#         # For y, it's every second bit as well, but starting from the next bit over from x's start
#         x |= (z & (1 << (2 * i))) >> i
#         y |= (z & (1 << (2 * i + 1))) >> (i + 1)
#     return x, y


def deinterleave_bits(z: int) -> tuple[int, int]:
    """
    Extracts the original coordinates from a unique integer.

    Parameters
    ----------
    z : int
        Unique integer created using interleave_bits.

    Returns
    -------
    tuple of (int, int)
        Original coordinates (x, y).
    """

    assert isinstance(z, int) and z >= 0, "z must be a non-negative integer."

    x = y = 0
    for i in range(0, z.bit_length()):
        # Extract the bit at position i and shift it to the right position
        # For x, we take every second bit starting from the LSB (least significant bit)
        # For y, it's every second bit as well, but starting from the next bit over from x's start
        x |= (z & (1 << (2 * i))) >> i
        y |= (z & (1 << (2 * i + 1))) >> (i + 1)
    return x, y


# function to normalize a vector (or a list of vectors) to 1
# def normalize(vec):
#     vec = array(vec)
#     if len(vec.shape) > 1:
#         vec_norm = array(vec) / linalg.norm(vec, axis=1)[:, None]
#     else:
#         vec_norm = array(vec) / linalg.norm(vec)
#     return vec_norm


# TODO: This needs to be checked further
def normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector (or array of vectors) to unit length.

    Parameters
    ----------
    vec : np.ndarray or list
        Vector(s) to normalize.

    Returns
    -------
    np.ndarray
        Normalized vector(s).
    """

    vec = np.array(vec)

    norm = np.linalg.norm(vec, axis=-1, keepdims=True)

    norm[norm == 0] = 1  # Avoid division by zero

    return vec / norm


# def generic_transformation_matrix(x0, y0, z0, x1, y1, z1):
#     A = column_stack((x0, y0, z0))
#     B = column_stack((x1, y1, z1))
#     R = B @ linalg.inv(A)
#     return R


def generic_transformation_matrix(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    z1: np.ndarray,
) -> np.ndarray:
    """
    Computes a generic transformation matrix.

    Parameters
    ----------
    x0, y0, z0 : np.ndarray
        Initial coordinates.
    x1, y1, z1 : np.ndarray
        Transformed coordinates.

    Returns
    -------
    np.ndarray
        Transformation matrix.
    """

    A = np.column_stack((x0, y0, z0))
    B = np.column_stack((x1, y1, z1))

    try:
        return B @ np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Transformation matrix is singular and cannot be inverted.")


# returns the rotation matrix around a given axis and a given angle
# def rotation_matrix(axis, angle):
#     assert axis in ["x", "y", "z"]

#     if axis == "x":
#         matrix = array([
#             [1, 0, 0],
#             [0, cos(angle), -sin(angle)],
#             [0, sin(angle), cos(angle)],
#         ])

#     if axis == "y":
#         matrix = array([
#             [cos(angle), 0, sin(angle)],
#             [0, 1, 0],
#             [-sin(angle), 0, cos(angle)],
#         ])

#     if axis == "z":
#         matrix = array([
#             [cos(angle), -sin(angle), 0],
#             [sin(angle), cos(angle), 0],
#             [0, 0, 1],
#         ])

#     return matrix


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """
    Returns the rotation matrix around a given axis.

    Parameters
    ----------
    axis : str
        Rotation axis ('x', 'y', or 'z').
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """

    assert axis in ["x", "y", "z"], "Axis must be 'x', 'y', or 'z'."

    c, s = np.cos(angle), np.sin(angle)

    matrices = {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }

    return matrices[axis]


# def great_circle_distance(lon_1, lat_1, lon_2, lat_2):
#     phi1, phi2 = radians(lat_1), radians(lat_2)
#     lambda1, lambda2 = radians(lon_1), radians(lon_2)
#     # Spherical Law of Cosines formula
#     delta_sigma = arccos(
#         sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(lambda2 - lambda1)
#     )
#     distance = EARTH_RADIUS * delta_sigma
#     return distance


def great_circle_distance(
    lon_1: float, lat_1: float, lon_2: float, lat_2: float
) -> float:
    """
    Computes the great-circle distance between two latitude-longitude points.

    Parameters
    ----------
    lon_1, lat_1 : float
        Longitude and latitude of the first point in degrees.
    lon_2, lat_2 : float
        Longitude and latitude of the second point in degrees.

    Returns
    -------
    float
        Distance in meters.
    """
    if (lon_1, lat_1) == (lon_2, lat_2):
        return 0.0

    phi1, phi2 = np.radians(lat_1), np.radians(lat_2)
    lambda1, lambda2 = np.radians(lon_1), np.radians(lon_2)

    delta_sigma = np.arccos(
        np.sin(phi1) * np.sin(phi2)
        + np.cos(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)
    )
    distance = EARTH_RADIUS * delta_sigma
    return distance


def intersection_with_sphere(alpha, R, H):
    """
    Finds the closest intersection of a line passing through P(0,H) with angle alpha
    with a circle of radius R centered at (0,0). The solution with positive x is returned.
    
    Parameters:
    R     : float  - Radius of the circle
    H     : float  - Height of point P
    alpha : float  - Angle with nadir in radians
    
    Returns:
    (x, y) : tuple - Coordinates of the closest intersection point
    """
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    
    discriminant = (R+H)**2 * np.cos(alpha)**2 - ((R+H)**2 - R**2)
    if discriminant < 0:
        raise ValueError("No real intersection found (line misses the circle).")
    
    lambda_closest = (R+H) * cos_alpha - np.sqrt(discriminant)

    x = lambda_closest * sin_alpha
    y = R+H - lambda_closest * cos_alpha

    return x, y
