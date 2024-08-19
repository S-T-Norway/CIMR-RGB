import h5py
import numpy as np
from grid_generator import GridGenerator
from pyproj import CRS, Transformer
from utils import normalize, generic_transformation_matrix, rotation_matrix
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
import time
# Terms to add to config file:
# antenna_pattern_path
#

class AntennaPattern:

    def __init__(self, config_object):
        threshold_dB = -9. #this could maybe be a parameter
        self.config = config_object
        self.antenna_file = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/Antenna_patterns/SMAP/RadiometerAntPattern_170830_v011.h5' # Should be in config file and provided in repository
        # self.antenna_file = '/home/davide/Downloads/CIMR-PAP-FR-L1-TPv0.3.h5'
        theta, phi, gain_dict = self.get_full_patterns_in_dict(self.antenna_file, phi_range=None, theta_range=None)

        self.average_radius = 200000 #this number depends on the instrument. 
        #it can be computed, but not trivial (create radial bins, find the radius where the gain goes below threshold)

        Ghco = gain_dict['G1h'] + 1j * gain_dict['G2h']
        Ghcx = gain_dict['G3h'] + 1j * gain_dict['G4h']
        Gvco = gain_dict['G1v'] + 1j * gain_dict['G2v']
        Gvcx = gain_dict['G3v'] + 1j * gain_dict['G4v']

        Gnorm = 0.5* (np.sqrt(np.abs(Ghco)**2+np.abs(Ghcx)**2) + np.sqrt(np.abs(Gvco)**2+np.abs(Gvcx)**2))

        mask = Gnorm < 10**(threshold_dB/10.)
        Ghco[mask] = 0.
        Ghcx[mask] = 0.
        Gvco[mask] = 0.
        Gvcx[mask] = 0.

        f_gain_hco = RegularGridInterpolator((phi, theta), Ghco)
        f_gain_hcx = RegularGridInterpolator((phi, theta), Ghcx)
        f_gain_vco = RegularGridInterpolator((phi, theta), Gvco)
        f_gain_vcx = RegularGridInterpolator((phi, theta), Gvcx)
            
        self.scalar_gain_function = self.get_scalar_antenna_pattern(f_gain_hco, f_gain_hcx, f_gain_vco, f_gain_vcx)
        self.mueller_matrix_function = self.get_mueller_matrix(f_gain_hco, f_gain_hcx, f_gain_vco, f_gain_vcx)

        return
    

    def get_scalar_antenna_pattern(self, f_gain_hco, f_gain_hcx, f_gain_vco, f_gain_vcx):

        def f_scalar_gain(phi, theta):
            Ghco_norm = np.abs(f_gain_hco((phi, theta)))
            Ghcx_norm = np.abs(f_gain_hcx((phi, theta)))
            Gvco_norm = np.abs(f_gain_vco((phi, theta)))
            Gvcx_norm = np.abs(f_gain_vcx((phi, theta)))
            Gv = np.sqrt(Gvco_norm**2 + Gvcx_norm**2)
            Gh = np.sqrt(Ghco_norm**2 + Ghcx_norm**2)
            Gscalar = 0.5*(Gv + Gh)
            return Gscalar
        
        return f_scalar_gain


    def get_mueller_matrix(self, f_gain_hco, f_gain_hcx, f_gain_vco, f_gain_vcx, threshold_dB=1e40):

        def f_mueller_matrix(phi, theta):

            Ghh = f_gain_hco((phi, theta))    #check the hh means hcross, etc.. 
            Ghv = f_gain_hcx((phi, theta))
            Gvv = f_gain_vco((phi, theta))
            Gvh = f_gain_vcx((phi, theta))

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
        
        return f_mueller_matrix

        
    @staticmethod
    def get_full_patterns_in_dict(antenna_file, phi_range=None, theta_range=None):

        with h5py.File(antenna_file, 'r') as f:

            gains = f['Gain']
            phi = f['Grid']['phi'][:]
            theta = f['Grid']['theta'][:]

            if phi_range is not None:
                mask_phi   = np.logical_and(phi > phi_range[0], phi < phi_range[1])
                phi = phi[mask_phi]

            if theta_range is not None:
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
        

    def make_integration_grid(self, longitudes, latitudes, margin=None):

        longitudes = np.array(longitudes)
        latitudes  = np.array(latitudes)

        gdef = self.config.grid_definition
        pdef = self.config.projection_definition

        if np.all(np.abs(latitudes) <= 75.): # 83.
            self.config.grid_definition = 'EASE2_G3km'
            self.config.projection_definition = 'G'

        elif np.any(latitudes > 75.):
            self.config.grid_definition = 'EASE2_N3km'
            self.config.projection_definition = 'N'

        elif np.any(latitudes < -75.):
            self.config.grid_definition = 'EASE2_S3km'
            self.config.projection_definition = 'S'

        xpoints = []
        ypoints = []
        for lon, lat in zip(longitudes, latitudes):
            x0, y0 = GridGenerator(self.config).lonlat_to_xy(lon, lat)
            xpoints.append(x0)
            ypoints.append(y0)
        if margin is None:
            margin = self.average_radius
        xmin = np.min(xpoints) - margin
        xmax = np.max(xpoints) + margin
        ymin = np.min(ypoints) - margin
        ymax = np.max(ypoints) + margin

        xs, ys = GridGenerator(self.config).generate_grid_xy()
        xeasemin = xs.min() #should be the cell edge, not the cell center
        xeasemax = xs.max() #should be the cell edge, not the cell center

        if self.config.projection_definition == 'G':
            xmin1 = xeasemax
            xmax1 = xeasemin
            if xmin < xeasemin:
                xmin1 = xeasemax - (xeasemin - xmin)
            if xmax > xeasemax:
                xmax1 = xeasemin + (xmax - xeasemax)

            xs = np.concatenate((xs[xs > xmin1], xs[np.logical_and(xs > xmin, xs < xmax)], xs[xs < xmax1]))
            ys = ys[np.logical_and(ys > ymin, ys < ymax)]

        else:
            xs = xs[np.logical_and(xs > xmin, xs < xmax)]
            ys = ys[np.logical_and(ys > ymin, ys < ymax)]

        Xs, Ys = np.meshgrid(xs, ys)        
        lons, lats = GridGenerator(self.config).xy_to_lonlat(Xs, Ys)

        self.config.grid_definition = gdef
        self.config.projection_definition = pdef

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

    def get_l1b_data(self, data_dict, var, scan_ind, earth_sample_ind):
        if var in self.config.variables_1d:
            return data_dict[var][scan_ind]
        else:
            return data_dict[var][earth_sample_ind]



    #this function can be removed
    def antenna_pattern_to_earth_simplified(self, data_dict, scan_ind, earth_sample_ind, int_dom_lons, int_dom_lats, use_full_mueller_matrix=False):

        x_pos = self.get_l1b_data(data_dict,'x_pos', scan_ind, None)
        y_pos = self.get_l1b_data(data_dict,'y_pos', scan_ind, None)
        z_pos = self.get_l1b_data(data_dict,'z_pos', scan_ind, None)

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
            Ginterp  = self.scalar_gain_function(phi, theta)
            Ginterp *= cos_angle_proj

        else:
            Ginterp = self.mueller_matrix_function(phi, theta)
            Ginterp *= cos_angle_proj 

        return Ginterp


    def antenna_pattern_from_boresight(self,data_dict, scan_ind,
                                       boresight_lon, boresight_lat, 
                                       int_dom_lons, int_dom_lats, 
                                       use_full_mueller_matrix=False):
        
        """
        It uses the boresight location on Earth and satellite position from L1b data to project an 
        antenna pattern on Earth. Currently it doens't include roll, pitch and yaw.
        """
                        
        x_pos = self.get_l1b_data(data_dict,'x_pos', scan_ind, None)
        y_pos = self.get_l1b_data(data_dict,'y_pos', scan_ind, None)
        z_pos = self.get_l1b_data(data_dict,'z_pos', scan_ind, None)
        nadir_lon = self.get_l1b_data(data_dict,'sc_nadir_lon', scan_ind, None)
        nadir_lat = self.get_l1b_data(data_dict,'sc_nadir_lat', scan_ind, None)
        
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
            Ginterp  = self.scalar_gain_function(phi, theta)
            Ginterp *= cos_angle_proj

        else:
            Ginterp = self.mueller_matrix_function(phi, theta)
            ## here some code changing the components of the mueller matrix because of the change in polarization basis
            Ginterp *= cos_angle_proj

        return Ginterp      


    def antenna_pattern_to_earth(self, data_dict, scan_ind, earth_sample_ind, int_dom_lons, int_dom_lats, use_full_mueller_matrix=False):

        scan_angle = self.get_l1b_data(data_dict,'antenna_scan_angle', scan_ind, earth_sample_ind)
        x_pos = self.get_l1b_data(data_dict,'x_pos', scan_ind, None)
        y_pos = self.get_l1b_data(data_dict,'y_pos', scan_ind, None)
        z_pos = self.get_l1b_data(data_dict,'z_pos', scan_ind, None)
        x_vel = self.get_l1b_data(data_dict,'x_vel', scan_ind, None)
        y_vel = self.get_l1b_data(data_dict,'y_vel', scan_ind, None)
        z_vel = self.get_l1b_data(data_dict,'z_vel', scan_ind, None)

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

        if not use_full_mueller_matrix:
            Ginterp  = self.scalar_gain_function(phi, theta)
            Ginterp *= cos_angle_proj

        else:
            Ginterp = self.mueller_matrix_function(phi, theta)
            Ginterp *= cos_angle_proj

        return Ginterp    


    def boresight_to_earth(self,data_dict, scan_ind, earth_sample_ind):

        scan_angle = self.get_l1b_data(data_dict,'antenna_scan_angle', scan_ind, earth_sample_ind)
        x_pos = self.get_l1b_data(data_dict,'x_pos', scan_ind, None)
        y_pos = self.get_l1b_data(data_dict,'y_pos', scan_ind, None)
        z_pos = self.get_l1b_data(data_dict,'z_pos', scan_ind, None)
        x_vel = self.get_l1b_data(data_dict,'x_vel', scan_ind, None)
        y_vel = self.get_l1b_data(data_dict,'y_vel', scan_ind, None)
        z_vel = self.get_l1b_data(data_dict,'z_vel', scan_ind, None)
        pitch = self.get_l1b_data(data_dict,'pitch', scan_ind, None)
        roll = self.get_l1b_data(data_dict,'roll', scan_ind, None)
        yaw = self.get_l1b_data(data_dict,'yaw', scan_ind, None)
    
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
# #scan_ind = 27
# #earth_sample_ind = 27

# lon, lat = 124.9500000000116, 6.749999992828501
# #lon, lat = -179.91505, 84.00648

# from data_ingestion import DataIngestion
# import os

# ingestion_object = DataIngestion(os.path.join(os.getcwd(), '..', 'config.xml'))
# config = ingestion_object.config
# data_dict = ingestion_object.ingest_data()

# AP = AntennaPattern(config)

# int_dom_lons, int_dom_lats = AP.make_integration_grid([lon], [lat])

# t = time.time()
# pattern = AP.antenna_pattern_from_boresight(scan_ind, lon, lat, int_dom_lons, int_dom_lats)
# print('time for projection: ', time.time()-t)

# import matplotlib.pyplot as plt
# plt.imshow(pattern); plt.colorbar()
# plt.show()