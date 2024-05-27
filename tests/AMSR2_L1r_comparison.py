import sys
import os
import pickle
import h5py as h5
from numpy import delete, r_
# add path for scripts in another folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# Debugging
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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



# Open RGB remap pickle dict
with open('/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/L1rProducts/AMSR2/RGB/GW1AM2_202308301033_051D_L1SGRTBR_2220220_36_10.pkl', 'rb') as f:
    amsr2_data = pickle.load(f)

# Load JAXA data
with h5.File('/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/L1rProducts/AMSR2/JAXA/GW1AM2_202308301033_051D_L1SGRTBR_2220220.h5', 'r') as data:
    overlap = int(data.attrs['OverlapScans'][0])
    bt_v_10_36 = data['Brightness Temperature (res10,36.5GHz,V)']
    bt_v_10_36_scale = bt_v_10_36.attrs['SCALE FACTOR']
    bt_v_10_36 = remove_overlap(bt_v_10_36[:], overlap)*bt_v_10_36_scale




diff = abs(amsr2_data['bt_v_source'] - bt_v_10_36)
print(f"Max difference: {diff.max()}")
print(f"Min difference: {diff.min()}")
print(f"Mean difference: {diff.mean()}")


fig, axs = plt.subplots(1, 3)
axs[0].imshow(amsr2_data['bt_v_source'])
axs[1].imshow(bt_v_10_36)
axs[2].imshow(diff)



