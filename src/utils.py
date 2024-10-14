"""
Utility functions for the RGB.
"""

from numpy import delete, r_, array, linalg, column_stack, cos, sin




def remove_overlap(array, overlap):
    """
    Removes overlap scans from the start and end of the array.

    Parameters
    ----------
    array: np.ndarray
        Array to remove overlap from
    overlap: int
        Number of scans to remove from the start and end of the array

    Returns
    -------
    np.ndarray
        Array with overlap scans removed
    """
    return delete(array, r_[0:overlap, array.shape[0] - overlap:array.shape[0]], axis=0)



def interleave_bits(x, y):
    """
    This function creates a unique integer from two coordinates by interleaving their bits.

    Parameters
    ----------
    x: int
        First coordinate
    y: int
        Second coordinate

    Returns
    -------
    z: int
        Unique integer created from the two coordinates
    """
    z = 0
    for i in range(max(x.bit_length(), y.bit_length())):
        z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1)
    return int(z)



def deinterleave_bits(z):
    """
    This function extracts the two coordinates from a unique integer created by interleave_bits.

    Parameters
    ----------
    z: int
        Unique integer created from two coordinates

    Returns
    -------
    x: int
        First coordinate
    y: int
        Second coordinate
    """
    x = y = 0
    for i in range(0, z.bit_length()):
        # Extract the bit at position i and shift it to the right position
        # For x, we take every second bit starting from the LSB (least significant bit)
        # For y, it's every second bit as well, but starting from the next bit over from x's start
        x |= (z & (1 << (2*i))) >> i
        y |= (z & (1 << (2*i + 1))) >> (i + 1)
    return x, y


# function to normalize a vector (or a list of vectors) to 1
def normalize(vec):
    vec = array(vec)
    if len(vec.shape) > 1:
        vec_norm = array(vec) / linalg.norm(vec, axis=1)[:, None]
    else:
        vec_norm = array(vec) / linalg.norm(vec)
    return vec_norm



def generic_transformation_matrix(x0, y0, z0, x1, y1, z1):
    A = column_stack((x0, y0, z0))
    B = column_stack((x1, y1, z1))
    R = B @ linalg.inv(A)
    return R


# returns the rotation matrix around a given axis and a given angle
def rotation_matrix(axis, angle):
    assert axis in ['x', 'y', 'z']

    if axis == 'x':
        matrix = array([[1, 0, 0],
                           [0, cos(angle), -sin(angle)],
                           [0, sin(angle), cos(angle)]])

    if axis == 'y':
        matrix = array([[cos(angle), 0, sin(angle)],
                           [0, 1, 0],
                           [-sin(angle), 0, cos(angle)]])

    if axis == 'z':
        matrix = array([[cos(angle), -sin(angle), 0],
                           [sin(angle), cos(angle), 0],
                           [0, 0, 1]])

    return matrix
