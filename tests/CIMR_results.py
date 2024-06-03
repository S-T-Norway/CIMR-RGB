import os
import pickle

import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

root_path = os.path.join(os.getcwd(), '..')

class CIMResults:
    def __init__(self, root_path):
        self.root_path = root_path
        self.CIMR_dir = os.path.join(root_path, 'dpr/L1C/CIMR/RGB')

    def get_CIMR_data(self, rgb_granule_name):
        cimr_data = os.path.join(self.CIMR_dir, rgb_granule_name + '.pkl')
        with open(cimr_data, 'rb') as f:
            cimr_data = pickle.load(f)
        return cimr_data

    def plot_granule(self, rgb_granule_name):
        cimr_data = self.get_CIMR_data(rgb_granule_name)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im= ax.imshow(cimr_data['C']['bt_h_fore'], cmap='magma')

        ax.set_title('RGB - Simulated CIMR', fontsize=14)
        ax.set_ylabel('EASE2 Grid Rows [-]', fontsize=14)
        ax.set_xlabel('EASE2 Grid Columns [-]', fontsize=14)
        fig.colorbar(im,  orientation='vertical')

        plt.tight_layout()



rgb_granule_name= 'SCEPS_L1C_sceps_geo_polar_scene_1_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1_EASE2_N9km'
cimr_data = CIMResults(root_path).plot_granule(rgb_granule_name)
