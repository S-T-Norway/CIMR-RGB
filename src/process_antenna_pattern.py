import h5py
import numpy as np
from grid_generator import GridGenerator
from pyproj import CRS, Transformer
from utils import normalize, generic_transformation_matrix, rotation_matrix
from scipy.interpolate import RegularGridInterpolator
import time
# Terms to add to config file:
# antenna_pattern_path
#

class AntennaPattern:

    def __init__(self, config_object):
        self.config = config_object
        self.antenna_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Antenna_patterns/SMAP/RadiometerAntPattern_170830_v011.h5' # Should be in config file and provided in repository
        self.theta, self.phi, self.gain_dict = self.get_full_patterns_in_dict(self.antenna_file, phi_range=None, theta_range=None)
        self.scalar_antenna_pattern = self.get_scalar_antenna_pattern()

    def get_scalar_antenna_pattern(self):

        Ghco = self.gain_dict['G1h'] + 1j * self.gain_dict['G2h']
        Ghcx = self.gain_dict['G3h'] + 1j * self.gain_dict['G4h']
        Gvco = self.gain_dict['G1v'] + 1j * self.gain_dict['G2v']
        Gvcx = self.gain_dict['G3v'] + 1j * self.gain_dict['G4v']

        Ghco_norm = np.abs(Ghco)
        Ghcx_norm = np.abs(Ghcx)
        Gvco_norm = np.abs(Gvco)
        Gvcx_norm = np.abs(Gvcx)

        Gv = np.sqrt(Gvco_norm**2 + Gvcx_norm**2)
        Gh = np.sqrt(Ghco_norm**2 + Ghcx_norm**2)

        Gscalar = 0.5*(Gv + Gh)
        mask = Gscalar < 0.01
        Gscalar[mask] = 0.

        return Gscalar


    def get_mueller_matrix(self):

        Ghh = self.gain_dict('G1h') + 1j * self.gain_dict('G2h')
        Ghv = self.gain_dict('G3h') + 1j * self.gain_dict('G4h')
        Gvv = self.gain_dict('G1v') + 1j * self.gain_dict('G2v')
        Gvh = self.gain_dict('G3v') + 1j * self.gain_dict('G4v')           

        n1, n2 = Ghh.shape
        G = np.zeros((4, 4, n1, n2))

        G[0, 0] = np.abs(Gvv)**2
        G[0, 1] = np.abs(Gvh)**2
        G[0, 2] = np.real(Gvh*np.conj(Gvv))
        G[0, 3] = np.imag(Gvh*np.conj(Gvv))
        G[1, 0] = np.abs(Ghv)**2
        G[1, 1] = np.abs(Ghh)**2
        G[1, 2] = np.real(Ghh*np.conj(Ghv))
        G[1, 3] = np.imag(Ghh*np.conj(Ghv))
        G[2, 0] = 2. * np.real(Gvv*np.conj(Ghv))
        G[2, 1] = 2. * np.real(Gvh*np.conj(Ghh))
        G[2, 2] = np.real(Gvv*np.conj(Ghh)+Gvh*np.conj(Ghv))
        G[2, 3] = -np.imag(Gvv*np.conj(Ghh)-Gvh*np.conj(Ghv))
        G[3, 0] = 2. * np.imag(Gvv*np.conj(Ghv))
        G[3, 1] = 2. * np.imag(Gvh*np.conj(Ghh))
        G[3, 2] = np.imag(Gvv*np.conj(Ghh)+Gvh*np.conj(Ghv))
        G[3, 3] = np.real(Gvv*np.conj(Ghh)-Gvh*np.conj(Ghv))

        return G
        
    @staticmethod
    def get_full_patterns_in_dict(antenna_file, phi_range=None, theta_range=None):

        with h5py.File(antenna_file, 'r') as f:

            gains = f['Gain']
            phi = f['Grid']['phi'][:]
            theta = f['Grid']['theta'][:]

            if phi_range is not None:
                mask_phi   = np.logical_and(phi > phi_range[0], phi < phi_range[1])
                phi = phi[mask_phi]

            if phi_range is not None:
                mask_theta = np.logical_and(theta > theta_range[0], theta < theta_range[1])
                theta = theta[mask_theta]

            gain_dict = {}
            for gain in gains:
                g = gains[gain][:]
                if phi_range is not None:
                    g = g[mask_phi]
                if theta_range is not None:
                    g = g[:, mask_theta]
                gain_dict[gain] = g

            theta = np.deg2rad(theta)
            phi   = np.deg2rad(phi)

        return theta, phi, gain_dict
        


    def make_integration_grid(self, lon_target, lat_target):

        #size of the interpolation grid, we could choose it based on the position of the antenna patterns
        dx = 500000.

        if np.abs(lat_target) < 83.: # 75
            # Check that this doesnt permanently change the config object in other places in the file.
            # Temporary fix to not permenently change the config object.
            original_grid_definition = self.config.grid_definition
            self.config.grid_definition = 'EASE2_G3km'
            self.config.projection_definition = 'G'
            x0, y0 = GridGenerator(self.config).lonlat_to_xy(lon_target, lat_target)
            xmin = x0 - dx/2
            xmax = x0 + dx/2
            ymin = y0 - dx/2
            ymax = y0 + dx/2
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
    def make_gaussian(grid_lons, grid_lats, lon0, lat0, slon, slat, rot=0.):

        phi1 = np.deg2rad(grid_lats)
        phi2 = np.deg2rad(lat0)
        delta_lam = np.deg2rad(grid_lons - lon0)
        
        x = grid_lons - lon0
        y = grid_lats - lat0
        
        Z = np.exp(-0.5 * (( ( x*np.cos(rot)+y*np.sin(rot) )**2 )/slon**2  + ( ( -x*np.sin(rot)+y*np.cos(rot)  )**2 )/slat**2))
        
        return Z

    def get_l1b_data(self, var, scan_ind, earth_sample_ind=None):
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
                    exit()
                    return

            if scan_ind is None and earth_sample_ind is None:
                return var_data
            elif earth_sample_ind is None:
                return var_data[scan_ind]
            else:
                flattened_ind = np.ravel_multi_index((scan_ind, earth_sample_ind), (779, 241))
                return var_data[flattened_ind]

    def antenna_pattern_to_earth_simplified(self, scan_ind, earth_sample_ind, int_dom_lons, int_dom_lats, use_full_mueller_matrix=False):

        x_pos = self.get_l1b_data('x_pos', scan_ind, None)
        y_pos = self.get_l1b_data('y_pos', scan_ind, None)
        z_pos = self.get_l1b_data('z_pos', scan_ind, None)    

        lonb, latb = self.boresight_to_earth(scan_ind, earth_sample_ind)      
    
        crs_ecef = CRS.from_epsg(4978)
        crs_wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_wgs84, crs_ecef, always_xy=True)

        int_dom_alts = np.zeros_like(int_dom_lons)
        X, Y, Z = transformer.transform(int_dom_lons, int_dom_lats, int_dom_alts) #m
        xb, yb, zb = transformer.transform(lonb, latb, 0.)

        vector_boresight = np.array([xb - x_pos, yb - y_pos, zb - z_pos]) 
        vector_boresight /= np.linalg.norm(vector_boresight)

        vector_grid_points_x = X - x_pos
        vector_grid_points_y = Y - y_pos
        vector_grid_points_z = Z - z_pos
        vector_grid_points_norm = np.sqrt(vector_grid_points_x**2 + vector_grid_points_y**2 + vector_grid_points_z**2)
        vector_grid_points_x /= vector_grid_points_norm
        vector_grid_points_y /= vector_grid_points_norm
        vector_grid_points_z /= vector_grid_points_norm

        semiaxis_a = 6378137. #m
        semiaxis_b = 6356752. #m
        norm_vec_x = -X/semiaxis_a**2
        norm_vec_y = -Y/semiaxis_a**2
        norm_vec_z = -Z/semiaxis_b**2
        norm_norm_vec = np.sqrt(norm_vec_x**2 + norm_vec_y**2 + norm_vec_z**2)
        norm_vec_x /= norm_norm_vec
        norm_vec_y /= norm_norm_vec
        norm_vec_z /= norm_norm_vec
        
        cos_angle_proj = norm_vec_x*vector_grid_points_x + norm_vec_y*vector_grid_points_y + norm_vec_z*vector_grid_points_z

        theta = np.arccos(vector_grid_points_x * vector_boresight[0] +
                          vector_grid_points_y * vector_boresight[1] + 
                          vector_grid_points_z * vector_boresight[2] )
        
        phi = 0. #this is the big simplification! I assume azimuthal simmetry, so I don't need to care about the scan angle

        if not use_full_mueller_matrix:
            gain = self.scalar_antenna_pattern
            Ginterp  = RegularGridInterpolator((self.phi, self.theta), gain)((phi, theta))
            Ginterp *= cos_angle_proj

        else:
            gain = self.get_mueller_matrix()
            _, _, n1, n2 = gain.shape
            Ginterp = np.zeros((4, 4, n1, n2))
            for i in range(4):
                for j in range(4):
                     Ginterp[i, j]  = RegularGridInterpolator((self.phi, self.theta), gain[i, j])((phi, theta))
                     Ginterp[i, j] *= cos_angle_proj

        return Ginterp


    def antenna_pattern_from_boresight(self, scan_ind,
                                       boresight_lon, boresight_lat, 
                                       int_dom_lons, int_dom_lats, 
                                       use_full_mueller_matrix=False):
                        
        x_pos = self.get_l1b_data('x_pos', scan_ind, None)
        y_pos = self.get_l1b_data('y_pos', scan_ind, None)
        z_pos = self.get_l1b_data('z_pos', scan_ind, None)
        nadir_lon = self.get_l1b_data('sc_nadir_lon', scan_ind, None)
        nadir_lat = self.get_l1b_data('sc_nadir_lat', scan_ind, None)
        
        crs_ecef = CRS.from_epsg(4978)
        crs_wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_wgs84, crs_ecef, always_xy=True)

        int_dom_alts = np.zeros_like(int_dom_lons)
        X, Y, Z = transformer.transform(int_dom_lons, int_dom_lats, int_dom_alts) #m       
        xb, yb, zb = transformer.transform(boresight_lon, boresight_lat, 0.)
        xn, yn, zn = transformer.transform(nadir_lon, nadir_lat, 0.)

        vector_boresight = np.array([xb - x_pos, yb - y_pos, zb - z_pos])
        vector_boresight /= np.linalg.norm(vector_boresight)

        vector_nadir = np.array([xn - x_pos, yn - y_pos, zn - z_pos]) 
        vector_nadir /= np.linalg.norm(vector_nadir)

        zaxis = vector_boresight
        yaxis = np.cross(vector_nadir, vector_boresight)
        xaxis = np.cross(yaxis, zaxis)

        vector_grid_points_x = X - x_pos
        vector_grid_points_y = Y - y_pos
        vector_grid_points_z = Z - z_pos
        vector_grid_points_norm = np.sqrt(vector_grid_points_x**2 + vector_grid_points_y**2 + vector_grid_points_z**2)
        vector_grid_points_x /= vector_grid_points_norm
        vector_grid_points_y /= vector_grid_points_norm
        vector_grid_points_z /= vector_grid_points_norm

        semiaxis_a = 6378137. #m
        semiaxis_b = 6356752. #m
        norm_vec_x = -X/semiaxis_a**2
        norm_vec_y = -Y/semiaxis_a**2
        norm_vec_z = -Z/semiaxis_b**2
        norm_norm_vec = np.sqrt(norm_vec_x**2 + norm_vec_y**2 + norm_vec_z**2)
        norm_vec_x /= norm_norm_vec
        norm_vec_y /= norm_norm_vec
        norm_vec_z /= norm_norm_vec
        
        cos_angle_proj = norm_vec_x*vector_grid_points_x + norm_vec_y*vector_grid_points_y + norm_vec_z*vector_grid_points_z

        cos_theta = vector_grid_points_x * vector_boresight[0] + vector_grid_points_y * vector_boresight[1] + vector_grid_points_z * vector_boresight[2] 

        cos_theta[cos_theta >= 1.] = 1. #sometimes there are values slightly larger than one, likely a numerical issue

        theta = np.arccos(cos_theta)
        
        xphi = vector_grid_points_x * xaxis[0] +  vector_grid_points_y * xaxis[1] +  vector_grid_points_z * xaxis[2]
        yphi = vector_grid_points_x * yaxis[0] +  vector_grid_points_y * yaxis[1] +  vector_grid_points_z * yaxis[2]

        phi = np.arctan2(yphi, xphi)

        phi[phi<0] += 2.*np.pi

        if not use_full_mueller_matrix:
            gain = self.scalar_antenna_pattern
            Ginterp  = RegularGridInterpolator((self.phi, self.theta), gain)((phi, theta))
            Ginterp *= cos_angle_proj

        else:
            gain = self.get_mueller_matrix()
            _, _, n1, n2 = gain.shape
            Ginterp = np.zeros((4, 4, n1, n2))
            for i in range(4):
                for j in range(4):
                     Ginterp[i, j]  = RegularGridInterpolator((self.phi, self.theta), gain[i, j])((phi, theta))
                     Ginterp[i, j] *= cos_angle_proj

        return Ginterp      


    def antenna_pattern_to_earth(self, scan_ind, earth_sample_ind, int_dom_lons, int_dom_lats, use_full_mueller_matrix=False):

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

        crs_ecef = CRS.from_epsg(4978)
        crs_wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_wgs84, crs_ecef, always_xy=True)

        int_dom_alts = np.zeros_like(int_dom_lons)
        X, Y, Z = transformer.transform(int_dom_lons, int_dom_lats, int_dom_alts) #m

        semiaxis_a = 6378137. #m
        semiaxis_b = 6356752. #m
        norm_vec_x = -X/semiaxis_a**2
        norm_vec_y = -Y/semiaxis_a**2
        norm_vec_z = -Z/semiaxis_b**2
        norm_norm_vec = np.sqrt(norm_vec_x**2 + norm_vec_y**2 + norm_vec_z**2)
        norm_vec_x /= norm_norm_vec
        norm_vec_y /= norm_norm_vec
        norm_vec_z /= norm_norm_vec

        X = X - x_pos
        Y = Y - y_pos
        Z = Z - z_pos

        norm_dir_vec = np.sqrt(X**2 + Y**2 + Z**2)
        dir_vec_x = X / norm_dir_vec
        dir_vec_y = Y / norm_dir_vec
        dir_vec_z = Z / norm_dir_vec
        
        cos_angle_proj = norm_vec_x*dir_vec_x + norm_vec_y*dir_vec_y + norm_vec_z*dir_vec_z

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

        phi[phi < 0] += 2. * np.pi

        # Interpolation should "conserve power" according to Joey Tennerelli
        # To be discussed

        if not use_full_mueller_matrix:
            gain = self.scalar_antenna_pattern
            Ginterp  = RegularGridInterpolator((self.phi, self.theta), gain)((phi, theta))
            Ginterp *= cos_angle_proj

        else:
            gain = self.get_mueller_matrix()
            _, _, n1, n2 = gain.shape
            Ginterp = np.zeros((4, 4, n1, n2))
            for i in range(4):
                for j in range(4):
                     Ginterp[i, j]  = RegularGridInterpolator((self.phi, self.theta), gain[i, j])((phi, theta))
                     Ginterp[i, j] *= cos_angle_proj

        return Ginterp
    

    def boresight_to_earth(self, scan_ind, earth_sample_ind):

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
    
        P0         = np.array([x_pos, y_pos, z_pos])
        tilt_angle = np.deg2rad(144.54)
        rot_angle  = np.deg2rad(scan_angle)

        R = rotation_matrix('y', tilt_angle)
        R = rotation_matrix('z', rot_angle)  @ R
        R = rotation_matrix('z', -yaw)       @ R
        R = rotation_matrix('x', -roll)      @ R
        R = rotation_matrix('y', -pitch)     @ R
    
        xax = normalize([x_vel, y_vel, z_vel])
        zax = normalize([x_pos, y_pos, z_pos])
        yax = np.cross(zax, xax)  

        R = generic_transformation_matrix([1,0,0], [0,1,0], [0,0,1], xax, yax, zax) @ R

        boresight = R @ np.array([0., 0., 1.]) + np.array([x_pos, y_pos, z_pos])

        m = boresight - P0
        q = boresight

        a=6378137.
        b=6356752.314245

        A = m[0]**2/a**2 + m[1]**2/a**2 + m[2]**2/b**2
        B = 2*(q[0]*m[0]/a**2 + q[1]*m[1]/a**2 + q[2]*m[2]/b**2)
        C = q[0]**2/a**2 + q[1]**2/a**2 + q[2]**2/b**2 - 1

        t1 = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
        t2 = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
    
        P1 = q + m * t1
        P2 = q + m * t2

        distance1 = np.linalg.norm(P1-P0)
        distance2 = np.linalg.norm(P2-P0)

        crs_ecef = CRS.from_epsg(4978)
        crs_wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_ecef, crs_wgs84, always_xy=True)

        if distance1 < distance2:
            lon, lat, _ = transformer.transform(P1[0], P1[1], P1[2])
        else:
            lon, lat, _ = transformer.transform(P2[0], P2[1], P2[2])

        return lon, lat



############################ test code 

# scan_ind = 81038 // 241
# earth_sample_ind = 81038 % 241

# lon, lat = 124.9500000000116, 6.749999992828501

# from data_ingestion import DataIngestion
# import os

# ingestion_object = DataIngestion(os.path.join(os.getcwd(), '..', 'config.xml'))
# config = ingestion_object.config
# data_dict = ingestion_object.ingest_data()

# AP = AntennaPattern(config)

# int_dom_lons, int_dom_lats = AP.make_integration_grid(lon_target=lon, lat_target=lat)
# pattern = AP.antenna_pattern_from_boresight(scan_ind, lon, lat, int_dom_lons, int_dom_lats)

# import matplotlib.pyplot as plt
# plt.imshow(pattern)
# plt.show()