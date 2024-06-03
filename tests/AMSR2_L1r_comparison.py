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

key_mappings_target = {
    '7': '7.3',
    '10': '10.7',
    '18': '18.7',
    '23': '23.8',
    '36': '36.5',
    '89': '89.0'
}

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

root_path = os.path.join(os.getcwd(), '..')
class AMSR2Comparison:
    def __init__(self, root_path):
        self.root_path = root_path
        self.RGB_dir = os.path.join(root_path, 'dpr/L1R/AMSR2/RGB')
        self.JAXA_dir = os.path.join(root_path, 'dpr/L1R/AMSR2/JAXA')

    def get_RGB_data(self, rgb_granule_name):
        rgb_data = os.path.join(self.RGB_dir,rgb_granule_name + '.pkl')
        with open(rgb_data, 'rb') as f:
            rgb_data = pickle.load(f)
        return rgb_data

    def get_JAXA_data(self, rgb_granule_name):
        test = 0
        jaxa_data = os.path.join(self.JAXA_dir, rgb_granule_name[0:41] + '.h5')

        bands = rgb_granule_name[42:].split('_')
        target_band = bands[1]
        source_band = bands[0]
        source_band = key_mappings_target[source_band]
        if len(target_band) == 1:
            target_band = '0' + target_band

        with h5.File(os.path.join(self.JAXA_dir,jaxa_data), 'r') as data:
            overlap = int(data.attrs['OverlapScans'][0])
            bt_v = data[f'Brightness Temperature (res{target_band},{source_band}GHz,V)']
            bt_v_scale = bt_v.attrs['SCALE FACTOR']
            bt_v = remove_overlap(bt_v[:], overlap)*bt_v_scale
        return bt_v

    def compare_single_granule(self, rgb_granule_name):
        rgb_data = self.get_RGB_data(rgb_granule_name)
        jaxa_data = self.get_JAXA_data(rgb_granule_name)

        # Currently for 'bt_v' only
        diff = abs(rgb_data['bt_v_source'] - jaxa_data)
        print(f"Max difference: {diff.max()}")
        print(f"Min difference: {diff.min()}")
        print(f"Mean difference: {diff.mean()}")

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(rgb_data['bt_v_source'], cmap = 'magma')
        axs[0].set_title('RGB (IDS)', fontsize = 14)
        axs[1].imshow(jaxa_data, cmap = 'magma')
        axs[1].set_title('JAXA (BG)', fontsize = 14)
        im = axs[2].imshow(diff, cmap = 'magma')
        axs[2].set_title('Difference', fontsize = 14)

        fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')



bt_v = AMSR2Comparison(root_path).compare_single_granule('GW1AM2_202308301033_051D_L1SGRTBR_2220220_23_6_9')




