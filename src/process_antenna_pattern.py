import h5py
import numpy as np
from grid_generator import GridGenerator
from pyproj import CRS, Transformer
from utils import normalize, generic_transformation_matrix, rotation_matrix
from scipy.interpolate import RegularGridInterpolator
# Terms to add to config file:
# antenna_pattern_path
#

class AntennaPattern:
    def __init__(self, config_object):
        self.config = config_object
        self.antenna_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Antenna_patterns/SMAP/RadiometerAntPattern_170830_v011.h5' # Should be in config file and provided in repository
        # self.theta, self.phi, self.gain_dict = self.get_ant_pattern(self.antenna_file, gain_label=None, phi_range=None, theta_range=[0,10], threshold=None)

    def get_antenna_patterns(self, antenna_file, normalize=True):
        # It is not smart to keep re-opening the file every time
        # you want to access the antenna patterns. We need to implement
        # the dictionary as in the function below at some point.

        with h5py.File(self.antenna_file, 'r') as f:
            G1h = f['Gain']['G1h'][:]
            G1v = f['Gain']['G1v'][:]
            G2h = f['Gain']['G2h'][:]
            G2v = f['Gain']['G2v'][:]
            phi = np.deg2rad(f['Grid']['phi'][:])
            theta = np.deg2rad(f['Grid']['theta'][:])

        if (normalize):
            dtheta = np.deg2rad(0.1)
            dphi = dtheta
            power1Htot = np.sum(10 ** (G1h / 10.) * np.sin(theta) * dtheta * dphi) / 4. / np.pi
            power1Vtot = np.sum(10 ** (G1v / 10.) * np.sin(theta) * dtheta * dphi) / 4. / np.pi
            power2Htot = np.sum(10 ** (G2h / 10.) * np.sin(theta) * dtheta * dphi) / 4. / np.pi
            power2Vtot = np.sum(10 ** (G2v / 10.) * np.sin(theta) * dtheta * dphi) / 4. / np.pi
            G1h = 10 ** (G1h / 10.) / (power1Htot)
            G1v = 10 ** (G1v / 10.) / (power1Vtot)
            G2h = 10 ** (G2h / 10.) / (power2Htot)
            G2v = 10 ** (G2v / 10.) / (power2Vtot)

        return theta, phi, (G1h, G1v, G2h, G2v)

    @staticmethod
    def get_full_patterns_in_dict(antenna_file, gain_label=None, phi_range=None, theta_range=None, threshold=None):

        with h5py.File(antenna_file, 'r') as f:
            gains = f['Gain']
            phi = f['Grid']['phi'][:]
            theta = f['Grid']['theta'][:]

            theta, phi = np.meshgrid(theta, phi)
            theta = theta.flatten()
            phi = phi.flatten()

            if phi_range is not None:
                mask = np.logical_and(phi > phi_range[0], phi < phi_range[1])
                phi = phi[mask]
                theta = theta[mask]
                # gains = gains[mask]

            if theta_range is not None:
                mask = np.logical_and(theta > theta_range[0], theta < theta_range[1])
                phi = phi[mask]
                theta = theta[mask]
                # gains = gains[mask]

            gain_dict = {}
            for gain in gains:
                if 'mask' in locals():
                    gain_dict[gain] = gains[gain][:].flatten()[mask]
                else:
                    gain_dict[gain] = gains[gain][:].flatten()

            # if threshold is not None:
            #     mask = gains > threshold
            #     phi = phi[mask]
            #     theta = theta[mask]
            #     gains = gains[mask]

            theta = np.deg2rad(theta)
            phi = np.deg2rad(phi)

            return theta, phi, gain_dict

    def make_integration_grid(self, lon_target, lat_target):

        dx = 1000000.  # 2*dx is the size of the interpolation grid

        if np.abs(lat_target) < 83.: # 75
            # Check that this doesnt permanently change the config object in other places in the file.
            # Temporary fix to not permenently change the config object.
            original_grid_definition = self.config.grid_definition
            self.config.grid_definition = 'EASE2_G3km'
            self.config.projection_definition = 'G'
            x0, y0 = GridGenerator(self.config).lonlat_to_xy(lon_target, lat_target)
            xmin = x0 - dx
            xmax = x0 + dx
            ymin = y0 - dx
            ymax = y0 + dx
            xs, ys = GridGenerator(self.config).generate_grid_xy()
            xs = xs[np.logical_and(xs > xmin, xs < xmax)]
            ys = ys[np.logical_and(ys > ymin, ys < ymax)]
            lons, lats = GridGenerator(self.config).xy_to_lonlat(xs, ys)
            lons, lats = np.meshgrid(lons, lats)
            self.config.grid_definition = original_grid_definition
        else:
            original_grid_definition = self.config.grid_definition
            self.config.grid_definition = 'EASE2_N3km'
            self.config.projection_definition = 'N'
            x0, y0 = GridGenerator(self.config).lonlat_to_xy(lon_target, lat_target)
            xmin = x0 - dx
            xmax = x0 + dx
            ymin = y0 - dx
            ymax = y0 + dx
            xs, ys = GridGenerator(self.config).generate_grid_xy()
            xs = xs[np.logical_and(xs > xmin, xs < xmax)]
            ys = ys[np.logical_and(ys > ymin, ys < ymax)]
            xs, ys = np.meshgrid(xs, ys)
            lons, lats = GridGenerator(self.config).xy_to_lonlat(xs, ys)
            self.config.grid_definition = original_grid_definition

        return lons, lats

    @staticmethod
    def make_gaussian(grid_lon, grid_lat, lon0, lat0, sigma):
        R = 0.5 * (6378.1370 + 6356.7523)  # km
        gcd = R * np.arccos(np.sin(np.deg2rad(grid_lat)) * np.sin(np.deg2rad(lat0))
                            + np.cos(np.deg2rad(grid_lat)) * np.cos(np.deg2rad(lat0)) * np.cos(
            np.deg2rad(np.abs(grid_lon - lon0))))  # km
        Z = np.exp(-gcd ** 2 / sigma ** 2)
        return Z

    def get_target_pattern(self, grid_lons, grid_lats, lon, lat):
        sigma = 18.  # km (I choose sigma=2*cell, we can think later about the best width of the gaussian)
        F = self.make_gaussian(grid_lons, grid_lats, lon, lat, sigma)
        F /= np.sum(F)
        return F

    def get_l1b_data(self, var, scan_ind, earth_sample_ind):
        # This is just a placeholder function for now
        # We should be able to get the data directly from the file
        # and this should be passed from the DataIngestion
        # and should be in some sort of data_dict.
        with h5py.File(self.config.input_data_path, 'r') as data:
            spacecraft = data['Spacecraft_Data']
            measurement = data['Brightness_Temperature']
            try:
                var_data = spacecraft[var][:]
            except:
                try:
                    var_data = measurement[var][:].flatten()
                    # This is a temporary solution as we remove the Nans from the data
                    # in order to be able to use the search tree.
                    # There might be problems here if things arent working.

                    var_data = var_data[self.config.non_nan_mask]


                except:
                    print(f"{var} not found in L1b file")
                    return


            if scan_ind is None and earth_sample_ind is None:
                return var_data
            elif earth_sample_ind is None:
                return var_data[scan_ind]
            else:
                flattened_ind = np.ravel_multi_index((scan_ind, earth_sample_ind), (779, 241))
                return var_data[flattened_ind]

    def source_ant_pattern_to_earth(self, scan_ind, earth_sample_ind, int_dom_lons, int_dom_lats):

        # For now we will grab the data directly from the file
        # However, this could/should be passed from the DataIngestion
        # and should be in some sort of data_dict.
        scan_angle = self.get_l1b_data('antenna_scan_angle', scan_ind, earth_sample_ind)
        x_pos = self.get_l1b_data('x_pos', scan_ind, None)
        y_pos = self.get_l1b_data('y_pos', scan_ind, None)
        z_pos = self.get_l1b_data('z_pos', scan_ind, None)
        x_vel = self.get_l1b_data('x_vel', scan_ind, None)
        y_vel = self.get_l1b_data('y_vel', scan_ind, None)
        z_vel = self.get_l1b_data('z_vel', scan_ind, None)
        pitch = self.get_l1b_data('pitch', scan_ind, None)
        roll = self.get_l1b_data('roll', scan_ind, None)
        yaw = self.get_l1b_data('yaw', scan_ind, None)
        lon = self.get_l1b_data('tb_lon', scan_ind, earth_sample_ind)
        lat = self.get_l1b_data('tb_lat', scan_ind, earth_sample_ind)

        crs_ecef = CRS.from_epsg(4978)
        crs_wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_wgs84, crs_ecef, always_xy=True)

        int_dom_alts = np.zeros_like(int_dom_lons)
        X, Y, Z = transformer.transform(int_dom_lons, int_dom_lats, int_dom_alts)
        X = X - x_pos
        Y = Y - y_pos
        Z = Z - z_pos

        xax = normalize([x_vel, y_vel, z_vel])
        zax = normalize([x_pos, y_pos, z_pos])
        yax = np.cross(zax, xax)

        tilt_angle = np.deg2rad(144.54)
        rot_angle = np.deg2rad(scan_angle)

        R = generic_transformation_matrix(xax, yax, zax, [1, 0, 0], [0, 1, 0], [0, 0, 1])
        R = rotation_matrix('y', pitch) @ R
        R = rotation_matrix('x', roll) @ R
        R = rotation_matrix('z', yaw) @ R
        R = rotation_matrix('z', -rot_angle) @ R
        R = rotation_matrix('y', -tilt_angle) @ R

        # hack to obtain an array of (row) vectors, and then rotate to get the resulting (row) vectors
        rot_vec = np.stack((X, Y, Z), axis=-1)
        shape = rot_vec.shape
        rot_vec = rot_vec.reshape(-1, 3) @ R.T
        rot_vec = rot_vec.reshape(shape)

        X = rot_vec[..., 0]
        Y = rot_vec[..., 1]
        Z = rot_vec[..., 2]

        norms = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        norms = np.where(norms == 0, 1, norms)

        X /= norms
        Y /= norms
        Z /= norms

        del (rot_vec)

        # to spherical coordinates
        theta = np.arccos(Z)
        phi = np.sign(Y) * np.arccos(X / np.sqrt(X ** 2 + Y ** 2))

        phi[phi < 0] += 2. * np.pi # Why 2?

        thetaAP, phiAP, gains = self.get_antenna_patterns(self.antenna_file, 'all')
        G11, G12, G21, G22 = gains

        Gtot = G11 + G12 + G21 + G22
        # Interpolation shoul "conserve power" according to Joey Tennerelli
        # To be discussed
        Ginterp = RegularGridInterpolator((phiAP, thetaAP), Gtot)((phi, theta))
        return Ginterp








