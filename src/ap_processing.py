from numpy import deg2rad, logical_and
import h5py
import os


class AntennaPattern:
    def __init__(self, config, band):
        self.config = config
        self.band=band

        if self.config.antenna_method == 'real':
            self.ap_dict = self.load_antenna_patterns()

            if self.config.polarisation_method == 'scalar':
                self.scalar_gain = self.get_scalar_pattern()
            elif self.config.polarisation_method == 'mueller':
                self.mueller_matrix = self.get_mueller_matrix()

        elif self.config == 'gaussian':
            self.ap_dict = self.generate_antenna_patterns()

    @staticmethod
    def extract_gain_dict(file_path,phi_range=None, theta_range=None):
        ap_dict = {}
        with h5py.File(file_path, 'r') as f:
            gains = f['Gain']
            phi = f['Grid']['phi'][:]
            theta = f['Grid']['theta'][:]

            if phi_range is not None:
                mask_phi = logical_and(phi > phi_range[0], phi < phi_range[1])
                phi = phi[mask_phi]

            if theta_range is not None:
                mask_theta = logical_and(theta > theta_range[0], theta < theta_range[1])
                theta = theta[mask_theta]

            gain_dict = {}
            for gain in gains:
                g = gains[gain][:]
                if phi_range is not None:
                    g = g[mask_phi]
                if theta_range is not None:
                    g = g[:, mask_theta]
                gain_dict[gain] = g

            theta = deg2rad(theta)
            phi = deg2rad(phi)

        ap_dict['theta'] = theta
        ap_dict['phi'] = phi
        ap_dict['gain'] = gain_dict

        return ap_dict

    def load_antenna_patterns(self):
        if self.config.input_data_type == "SMAP":
            ap_dict = self.extract_gain_dict(
                file_path = self.config.antenna_pattern_path
            )

        elif self.config.input_data_type == "CIMR":
            test =0
            num_horns = self.config.num_horns[self.band]
            horn_dict = {}
            ap_dict = {}
            for feedhorn in range(num_horns):
                path = os.path.join(
                    self.config.antenna_pattern_path, self.band)
                horn = self.band + str(feedhorn)

                for file in os.listdir(path):
                    if horn in file:
                        ap_dict[horn] = self.extract_gain_dict(
                            file_path=os.path.join(path, file)
                        )

        return ap_dict

    def generate_antenna_patterns(self):
        pass

    def get_scalar_pattern(self):
        pass

    def get_mueller_matrix(self):
        pass