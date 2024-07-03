"""
Utility functions for the RGB.
"""

from numpy import delete, r_
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
