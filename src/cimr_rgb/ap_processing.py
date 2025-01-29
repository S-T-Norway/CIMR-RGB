import os
from sys import excepthook

import numpy as np
from numpy import (abs, deg2rad, full, sqrt, logical_or, logical_and, conj, real, imag,
                   zeros, array, argmax, cross, average, tan, all, any, min, max,
                   concatenate, meshgrid, sin , cos, sign, arccos, zeros_like, stack,
                   where, pi, flip)
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from pyproj import CRS, Transformer

from cimr_rgb.grid_generator import GridGenerator, GRIDS
from cimr_rgb.utils          import normalize, generic_transformation_matrix


class AntennaPattern:

    def __init__(self, config, band, antenna_method, polarisation_method, antenna_threshold, gaussian_params):
        self.config = config
        self.band=band
        self.antenna_method = antenna_method
        self.polarisation_method = polarisation_method
        self.antenna_threshold = antenna_threshold
        self.gaussian_params = gaussian_params

        if self.antenna_method == 'instrument':

            ap_dict, self.fraction_below_threshold = self.load_antenna_patterns()

            if self.polarisation_method == 'scalar':
                self.scalar_gain = self.get_scalar_pattern(ap_dict)
            elif self.polarisation_method == 'mueller':
                self.mueller_matrix = self.get_mueller_matrix(ap_dict)

            self.max_ap_radius = self.estimate_max_ap_radius(ap_dict)

        elif self.antenna_method == 'gaussian_projected':  #in this case I guess only scalar gain available

            #for consistency, create a number of horn corresponding to the band, all with same gaussian pattern
            ap_dict = dict()
            for i in range(self.config.num_horns[self.band]):
                ap_dict[i] = None
            self.scalar_gain = self.gaussian_antenna_patterns(ap_dict)
            self.max_ap_radius = self.estimate_max_ap_radius(ap_dict)

        return

    @staticmethod
    def extract_gain_dict(file_path, antenna_threshold, theta_max=40):

        import h5py

        theta_max = deg2rad(theta_max)
        ap_dict = {}

        with h5py.File(file_path, 'r') as f:
            gains = f['Gain']
            phi = f['Grid']['phi'][:]
            theta = f['Grid']['theta'][:]

            gain_dict = {}
            for gain in gains:
                gain_dict[gain] = gains[gain][:]

        ap_dict['theta'] = deg2rad(theta)
        ap_dict['phi'] = deg2rad(phi)

        ap_dict['Ghco'] = gain_dict['G1h'] + 1j * gain_dict['G2h']
        ap_dict['Ghcx'] = gain_dict['G3h'] + 1j * gain_dict['G4h']
        ap_dict['Gvco'] = gain_dict['G1v'] + 1j * gain_dict['G2v']
        ap_dict['Gvcx'] = gain_dict['G3v'] + 1j * gain_dict['G4v']

        ap_dict['Gnorm'] = 0.5* (sqrt(abs(ap_dict['Ghco'])**2+abs(ap_dict['Ghcx'])**2)
                                 + sqrt(abs(ap_dict['Gvco'])**2+abs(ap_dict['Gvcx'])**2))

        ap_dict['Gnorm'] /= np.sum( ap_dict['Gnorm'])

        mask = full(ap_dict['Gnorm'].shape, False)

        fraction_below_threshold = 1.

        if antenna_threshold is not None:
            threshold_power = antenna_threshold * np.max(ap_dict['Gnorm'])
            mask = np.logical_or(mask, ap_dict['Gnorm'] < threshold_power)
            fraction_below_threshold = np.sum(ap_dict['Gnorm'][mask]) / np.sum(ap_dict['Gnorm'])

        if theta_max is not None:
            mask = logical_or(mask, ap_dict['theta'] > theta_max)

        ap_dict['Ghco'][mask] = 0.
        ap_dict['Ghcx'][mask] = 0.
        ap_dict['Gvco'][mask] = 0.
        ap_dict['Gvcx'][mask] = 0.
        ap_dict['Gnorm'][mask] = 0.
    
        return ap_dict, fraction_below_threshold

    def load_antenna_patterns(self):

        # JOSEPH: We might also totally remove this IF-ELSE, if the user 
        # is forced to specify the targetBand config parameter also for SMAP
        if self.config.input_data_type == "SMAP":
            ap_dict = {}
            fraction_below_threshold = {}
            ap_dict[0], fraction_below_threshold[0] = self.extract_gain_dict(
                file_path = self.config.antenna_patterns_path,
                antenna_threshold=self.antenna_threshold
            )

        elif self.config.input_data_type == "CIMR":
            num_horns = self.config.num_horns[self.band]
            horn_dict = {}
            ap_dict = {}
            fraction_below_threshold = {}
            for feedhorn in range(num_horns):
                path = os.path.join(
                    self.config.antenna_patterns_path, self.band)
                horn = self.band + str(feedhorn)

                horn_files = [ff for ff in os.listdir(path) if horn in ff]

                assert(len(set(horn_files))==1), "There are zero or more than one antenna pattern files for feedhorn " + horn

                ap_dict[int(feedhorn)], fraction_below_threshold[int(feedhorn)] = self.extract_gain_dict(
                            file_path=os.path.join(path, horn_files[0]),
                            antenna_threshold=self.antenna_threshold
                        )

        return ap_dict, fraction_below_threshold

    def gaussian_antenna_patterns(self, ap_dict):

        if self.antenna_threshold is None:
            # Specify this as default in config and/or docs
            ant_th = 0.001
        else:
            ant_th = self.antenna_threshold

        sigma_u = self.gaussian_params[0]
        sigma_v = self.gaussian_params[1]
        rot     = np.deg2rad(self.gaussian_params[2])

        def f_scalar_gain(phi, theta):
            phi   = np.atleast_1d(phi)
            theta = np.atleast_1d(theta)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            Z = np.exp(-0.5 * (( ( x*np.cos(rot)+y*np.sin(rot) )**2 )/sigma_u**2  + ( ( -x*np.sin(rot)+y*np.cos(rot)  )**2 )/sigma_v**2))
            Z /= Z.sum()
            Z[Z < ant_th * Z.max()] = 0.
            return Z

        scalar_pattern = dict()
        for horn in ap_dict:
            scalar_pattern[horn] = f_scalar_gain

        return scalar_pattern

    def get_scalar_pattern(self, ap_dict):

        scalar_pattern = {}

        for horn in (ap_dict):

            phi = ap_dict[horn]['phi']
            theta = ap_dict[horn]['theta']

            f_gain_hco = RegularGridInterpolator((phi, theta), ap_dict[horn]['Ghco'], bounds_error=False, fill_value=0.)
            f_gain_hcx = RegularGridInterpolator((phi, theta), ap_dict[horn]['Ghcx'], bounds_error=False, fill_value=0.)
            f_gain_vco = RegularGridInterpolator((phi, theta), ap_dict[horn]['Gvco'], bounds_error=False, fill_value=0.)
            f_gain_vcx = RegularGridInterpolator((phi, theta), ap_dict[horn]['Gvcx'], bounds_error=False, fill_value=0.)
            
            def f_scalar_gain(phi, theta):
                Ghco_norm = abs(f_gain_hco((phi, theta)))
                Ghcx_norm = abs(f_gain_hcx((phi, theta)))
                Gvco_norm = abs(f_gain_vco((phi, theta)))
                Gvcx_norm = abs(f_gain_vcx((phi, theta)))
                Gv = sqrt(Gvco_norm**2 + Gvcx_norm**2)
                Gh = sqrt(Ghco_norm**2 + Ghcx_norm**2)
                Gscalar = 0.5*(Gv + Gh)
                return Gscalar

            scalar_pattern[horn] = f_scalar_gain

        return scalar_pattern

    def get_mueller_matrix(self, ap_dict):

        mueller_matrix = dict()

        for horn in (ap_dict):

            phi = ap_dict[horn]['phi']
            theta = ap_dict[horn]['theta']

            f_gain_hco = RegularGridInterpolator((phi, theta), ap_dict[horn]['Ghco'], bounds_error=False, fill_value=0.)
            f_gain_hcx = RegularGridInterpolator((phi, theta), ap_dict[horn]['Ghcx'], bounds_error=False, fill_value=0.)
            f_gain_vco = RegularGridInterpolator((phi, theta), ap_dict[horn]['Gvco'], bounds_error=False, fill_value=0.)
            f_gain_vcx = RegularGridInterpolator((phi, theta), ap_dict[horn]['Gvcx'], bounds_error=False, fill_value=0.)

            def f_mueller_matrix(phi, theta):

                Ghh = f_gain_hco((phi, theta))    #check that hh means hcross, etc.. 
                Ghv = f_gain_hcx((phi, theta))
                Gvv = f_gain_vco((phi, theta))
                Gvh = f_gain_vcx((phi, theta))

                n1, n2 = Ghh.shape
                G = zeros((4, 4, n1, n2))

                G[0, 0] = abs(Gvv)**2
                G[0, 1] = abs(Gvh)**2
                G[0, 2] = real(Gvh*conj(Gvv))
                G[0, 3] = imag(Gvh*conj(Gvv))
                G[1, 0] = abs(Ghv)**2
                G[1, 1] = abs(Ghh)**2
                G[1, 2] = real(Ghh*conj(Ghv))
                G[1, 3] = imag(Ghh*conj(Ghv))
                G[2, 0] = 2. * real(Gvv*conj(Ghv))
                G[2, 1] = 2. * real(Gvh*conj(Ghh))
                G[2, 2] = real(Gvv*conj(Ghh)+Gvh*conj(Ghv))
                G[2, 3] = -imag(Gvv*conj(Ghh)-Gvh*conj(Ghv))
                G[3, 0] = 2. * imag(Gvv*conj(Ghv))
                G[3, 1] = 2. * imag(Gvh*conj(Ghh))
                G[3, 2] = imag(Gvv*conj(Ghh)+Gvh*conj(Ghv))
                G[3, 3] = real(Gvv*conj(Ghh)-Gvh*conj(Ghv))

                return G

            mueller_matrix[horn] = f_mueller_matrix

        return mueller_matrix

    def estimate_max_ap_radius(self, ap_dict):

        tilt_angle = np.deg2rad(self.config.antenna_tilt_angle)

        if self.config.input_data_type == "SMAP":
            tilt_angle = pi - tilt_angle

        satellite_altitude = self.config.max_altitude

        def interesection_with_sphere(alpha, R, H):
            dx = np.sin(alpha)
            dz = -np.cos(alpha)
            a = dx**2 + dz**2
            b = 2 * dz * (R+H)
            c = (R+H)**2 - R**2
            discriminant = b**2 - 4*a*c
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)
            t = t1 if t1 >= 0 else t2
            x = t * dx
            y = 0
            z = R + H + t * dz
            return (x,y,z)

        max_radius = {}

        for horn in ap_dict:
            
            if self.antenna_method == 'instrument':
                Gnorm = ap_dict[horn]['Gnorm']
                ind_max_theta_non_zero = argmax(average(Gnorm, axis=0) <= 0.)
                if ind_max_theta_non_zero==0:
                    ind_max_theta_non_zero = -1
                max_theta_non_zero = ap_dict[horn]['theta'][ind_max_theta_non_zero]

            elif self.antenna_method == 'gaussian_projected':
                if self.antenna_threshold is None:
                    max_theta_non_zero = np.deg2rad(self.config.max_theta_antenna_patterns)
                else:      
                    sigma_u = self.gaussian_params[0]
                    sigma_v = self.gaussian_params[1]
                    sigma_max = np.maximum(sigma_u, sigma_v)
                    r_max = np.sqrt(-2. * sigma_max**2 * np.log(self.antenna_threshold))
                    max_theta_non_zero = np.minimum(np.deg2rad(self.config.max_theta_antenna_patterns), np.arcsin(r_max))

            R = (6378137. + 6356752.)/2. #m
            angle_tangent = np.arcsin(R / (R + satellite_altitude))
            angle_max = np.minimum(tilt_angle + max_theta_non_zero, angle_tangent)
            x1, y1, z1 = interesection_with_sphere(angle_max, R, satellite_altitude)
            x2, y2, z2 = interesection_with_sphere(tilt_angle, R, satellite_altitude)
            angle_center_1 = np.arctan(x1/z1)
            angle_center_2 = np.arctan(x2/z2)
            arch1 = R * angle_center_1
            arch2 = R * angle_center_2
            max_radius[horn] = arch1 - arch2
            max_radius[horn] *= 1.1 #add 10% margin

        return max_radius

    def antenna_pattern_to_earth(self, int_dom_lons, int_dom_lats, x_pos, y_pos,
                                 z_pos, x_vel, y_vel, z_vel, processing_scan_angle,
                                 feed_horn_number, attitude=None,
                                 lon_l1b=None, lat_l1b=None):
        
        tilt_angle = np.deg2rad(self.config.antenna_tilt_angle)
        processing_scan_angle = np.deg2rad(processing_scan_angle)

        if self.config.boresight_shift:
            lonb, latb = self.boresight_to_earth(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel,
                                           processing_scan_angle, self.band, feed_horn_number, attitude)

        if self.config.input_data_type == "SMAP":
            tilt_angle = pi - tilt_angle # maybe we should change this already in the data ingestion!
            processing_scan_angle = pi/2. - processing_scan_angle
            yax = normalize([x_vel, y_vel, z_vel])
            zax = normalize([x_pos, y_pos, z_pos])
            xax = cross(yax, zax)
            attitude = generic_transformation_matrix(xax, yax, zax, [1,0,0], [0,1,0], [0,0,1])
            feed_offset_phi =   0.
            feed_offset_theta = 0. 

        elif self.config.input_data_type == "CIMR":
            attitude = attitude.reshape(3,3)
            feed_offset_phi = deg2rad(self.config.scan_angle_feed_offsets[self.band][int(feed_horn_number)])
            if feed_offset_phi != 0.:  
                feed_offset_theta = np.arcsin(self.config.v0[self.band][int(feed_horn_number)] / (-np.sin(feed_offset_phi))) - tilt_angle
            else:
                feed_offset_theta = np.arcsin(self.config.u0[self.band][int(feed_horn_number)]) - tilt_angle

        adjust_lon = 0
        adjust_lat = 0

        if self.config.boresight_shift:
            adjust_lon = lonb - lon_l1b
        if self.config.boresight_shift:
            adjust_lat = latb - lat_l1b

        # zaxis = feedhorn boresight, and xaxis perpendicolar to it and with upvector pointing in the same direction
        antenna_zaxis = array([-sin(tilt_angle+feed_offset_theta)*cos(processing_scan_angle+feed_offset_phi),
                                   sin(tilt_angle+feed_offset_theta)*sin(processing_scan_angle+feed_offset_phi),
                                  -cos(tilt_angle+feed_offset_theta)])
        antenna_yaxis = array([-cos(tilt_angle+feed_offset_theta)*cos(processing_scan_angle+feed_offset_phi),
                                   cos(tilt_angle+feed_offset_theta)*sin(processing_scan_angle+feed_offset_phi),
                                   sin(tilt_angle+feed_offset_theta)])
        antenna_xaxis = cross(antenna_yaxis, antenna_zaxis)

        crs_ecef = CRS.from_epsg(4978)
        crs_wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_wgs84, crs_ecef, always_xy=True)

        int_dom_alts = zeros_like(int_dom_lons)
        X, Y, Z = transformer.transform(int_dom_lons+adjust_lon, int_dom_lats+adjust_lat, int_dom_alts) #m

        semiaxis_a = 6378137. #m
        semiaxis_b = 6356752. #m
        norm_vec_x = -X/semiaxis_a**2
        norm_vec_y = -Y/semiaxis_a**2
        norm_vec_z = -Z/semiaxis_b**2
        norm_norm_vec = sqrt(norm_vec_x**2 + norm_vec_y**2 + norm_vec_z**2)
        norm_vec_x /= norm_norm_vec
        norm_vec_y /= norm_norm_vec
        norm_vec_z /= norm_norm_vec

        X = X - x_pos
        Y = Y - y_pos
        Z = Z - z_pos

        norm_dir_vec = sqrt(X**2 + Y**2 + Z**2)
        dir_vec_x = X / norm_dir_vec
        dir_vec_y = Y / norm_dir_vec
        dir_vec_z = Z / norm_dir_vec
        
        cos_angle_proj = norm_vec_x*dir_vec_x + norm_vec_y*dir_vec_y + norm_vec_z*dir_vec_z

        rot_vec = stack((X, Y, Z), axis=-1)
        shape = rot_vec.shape

        rot_vec = np.einsum('ij,jk', rot_vec.reshape(-1, 3), attitude.T) #equivalent to @, which is occasionally crashing for no reason (numpy bug?)
        rot_vec = rot_vec.reshape(shape)
        X = rot_vec[..., 0]
        Y = rot_vec[..., 1]
        Z = rot_vec[..., 2]

        norms = sqrt(X ** 2 + Y ** 2 + Z ** 2)
        norms = where(norms == 0, 1, norms)

        X /= norms
        Y /= norms
        Z /= norms

        del (rot_vec)

        # to spherical coordinates
        Xp = X*antenna_xaxis[0] + Y*antenna_xaxis[1] + Z*antenna_xaxis[2]
        Yp = X*antenna_yaxis[0] + Y*antenna_yaxis[1] + Z*antenna_yaxis[2]
        Zp = X*antenna_zaxis[0] + Y*antenna_zaxis[1] + Z*antenna_zaxis[2]

        Xp[Xp>1.] = 1.
        Yp[Yp>1.] = 1.
        Zp[Zp>1.] = 1.

        theta = arccos(Zp)
        phi = sign(Yp) * arccos(Xp / sqrt(Xp** 2 + Yp**2))
        
        phi[phi < 0] += 2. * pi

        Ginterp=self.scalar_gain[int(feed_horn_number)](phi, theta)
        Ginterp *= cos_angle_proj                                       #this messes up the normalization, is it a problem?

        return Ginterp 


    def boresight_to_earth(self, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, processing_scan_angle, band,
                                 feed_horn_number, attitude=None):

        tilt_angle = np.deg2rad(self.config.antenna_tilt_angle)

        if self.config.input_data_type == "SMAP":
            tilt_angle = pi - tilt_angle                          # should we change this already in the data ingestion?
            processing_scan_angle = pi/2. - processing_scan_angle # should we change this already in the data ingestion?   
            yax = normalize([x_vel, y_vel, z_vel])
            zax = normalize([x_pos, y_pos, z_pos])
            xax = cross(yax, zax)
            attitude = generic_transformation_matrix(xax, yax, zax, [1,0,0], [0,1,0], [0,0,1]) #maybe better way to compute it, accounting for roll, pitch and yaw
            feed_offset_phi   =  0
            feed_offset_theta = 0

        elif self.config.input_data_type == "CIMR":
            attitude = attitude.reshape(3,3)
            feed_offset_phi = deg2rad(self.config.scan_angle_feed_offsets[band][feed_horn_number])
            feed_offset_theta = np.arcsin(self.config.v0[band][feed_horn_number] / (-np.sin(feed_offset_phi))) - tilt_angle

        P0 = np.array([x_pos, y_pos, z_pos])

        #boresight in the body frame
        feedhorn_boresight = np.array([-np.sin(tilt_angle+feed_offset_theta)*np.cos(processing_scan_angle+feed_offset_phi), 
                                        np.sin(tilt_angle+feed_offset_theta)*np.sin(processing_scan_angle+feed_offset_phi), 
                                       -np.cos(tilt_angle+feed_offset_theta)])

        feedhorn_boresight = feedhorn_boresight @ attitude + np.array([x_pos, y_pos, z_pos])

        m = feedhorn_boresight - P0
        q = feedhorn_boresight

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


class GaussianAntennaPattern:

    def __init__(self, config, antenna_threshold):

        self.config = config
        self.antenna_threshold = antenna_threshold #dB
        self.fraction_below_threshold = 1.

        return

    def antenna_pattern_to_earth(self, int_dom_lons, int_dom_lats, lon_l1b, lat_l1b, sigmax, sigmay, alpha=None, lon_nadir=None, lat_nadir=None):

        if alpha is None:

            delta_lon = np.deg2rad(lon_l1b - lon_nadir)

            alpha = np.arctan2(np.cos(np.deg2rad(lat_l1b)) * np.sin(delta_lon),
                             np.cos(np.deg2rad(lat_nadir)) * np.sin(np.deg2rad(lat_l1b)) - np.sin(np.deg2rad(lat_nadir)) * np.cos(np.deg2rad(lat_l1b)) * np.cos(delta_lon)
                    )
            
            alpha = np.pi/2. - alpha

        x = haversine_distance(int_dom_lons, lat_l1b, lon_l1b, lat_l1b)
        y = haversine_distance(lon_l1b, int_dom_lats, lon_l1b, lat_l1b)

        x[int_dom_lons < lon_l1b] *= -1
        y[int_dom_lats < lat_l1b] *= -1

        x_rot = x * np.cos(alpha) + y * np.sin(alpha)  #m
        y_rot = -x * np.sin(alpha) + y * np.cos(alpha) #m

        G = np.exp(-((x_rot ** 2) / (2 * sigmax ** 2) + (y_rot ** 2) / (2 * sigmay ** 2)))

        G /= G.sum()

        if self.antenna_threshold is not None:
            mask_zero = (G < self.antenna_threshold * G.max())
            G[mask_zero] = 0.

        return G

    def estimate_max_ap_radius(self, sigmax, sigmay):  
        sigma_max = np.maximum(sigmax, sigmay)
        return np.sqrt(-2. * sigma_max**2 * np.log(self.antenna_threshold))

@staticmethod
def haversine_distance(lon1, lat1, lon2, lat2):

    Rearth  = (6378137. + 6356752.)/2. #m
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = Rearth * c

    return distance #m

@staticmethod
def make_integration_grid(int_projection_definition, int_grid_definition, longitude, latitude, ap_radii):

    Rearth  = (6378137. + 6356752.)/2. #m
    Rcircle = Rearth * np.abs(np.cos(np.deg2rad(latitude)))
    Rcircle = np.max(Rcircle)
    Rpattern = max(ap_radii)
    latdn = np.array(latitude)  - np.rad2deg(Rpattern/Rearth)
    latup = np.array(latitude)  + np.rad2deg(Rpattern/Rearth)
    lonsx = np.array(longitude) - np.rad2deg(Rpattern/Rcircle)
    londx = np.array(longitude) + np.rad2deg(Rpattern/Rcircle)

    integration_grid = GridGenerator(
        config_object = None,
        projection_definition=int_projection_definition,
        grid_definition=int_grid_definition
    )

    mask = latup>90
    latup[mask] = 180. - latup[mask]

    mask = latdn<-90
    latdn[mask] = -180. - latdn[mask]

    mask = londx>180.
    londx[mask] = londx[mask] - 360.

    mask = lonsx<-180.
    lonsx[mask] = lonsx[mask] + 360.

    lons = np.concatenate((lonsx, londx))
    lats = np.concatenate((latdn, latup))

    if int_projection_definition == 'G':

        max1 = np.max(180 - lons[lons > 0]) if np.any(lons > 0) else 0
        max2 = np.max(lons[lons < 0] + 180) if np.any(lons < 0) else 0
        size_wrapped = max1 + max2    
        size_non_wrapped = lons.max() - lons.min()

        if size_non_wrapped <= size_wrapped: #not wrapping across the IDL:
            lonmin = np.min(lons)
            lonmax = np.max(lons)
            latmin = np.min(lats)
            latmax = np.max(lats)
            xmin, ymin = integration_grid.lonlat_to_xy(lonmin, latmin)
            xmax, ymax = integration_grid.lonlat_to_xy(lonmax, latmax)
            xs, ys = integration_grid.generate_grid_xy()
            xs = xs[logical_and(xs > xmin, xs < xmax)]
            ys = ys[logical_and(ys > ymin, ys < ymax)]

        else: #wrapping across the IDL
            lonmax = np.max(lons[lons<0]) #further point from IDL with lon < 0
            lonmin = np.min(lons[lons>0]) #further point from IDL with lon > 0
            latmin = np.min(lats)
            latmax = np.max(lats)
            xmin, ymin = integration_grid.lonlat_to_xy(lonmin, latmin)
            xmax, ymax = integration_grid.lonlat_to_xy(lonmax, latmax)
            xs, ys = integration_grid.generate_grid_xy()
            xs = concatenate((xs[xs > xmin], xs[xs < xmax]))
            ys = ys[logical_and(ys > ymin, ys < ymax)]

        Xs, Ys = meshgrid(xs, ys)

    elif int_projection_definition in ['N', 'S']:

        x, y = integration_grid.lonlat_to_xy(lons, lats)
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        xs, ys = integration_grid.generate_grid_xy()
        xs = xs[logical_and(xs > xmin, xs < xmax)]
        ys = ys[logical_and(ys > ymin, ys < ymax)]
        Xs, Ys = meshgrid(xs, ys)
   
    lons, lats = integration_grid.xy_to_lonlat(Xs, Ys)

    return lons, lats
