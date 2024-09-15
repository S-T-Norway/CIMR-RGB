from numpy import (abs, deg2rad, full, sqrt, logical_and, conj, real, imag,
                   zeros, array, argmax, cross, average, tan, all, any, min, max,
                   concatenate, meshgrid, sin , cos, sign, arccos, zeros_like, stack,
                   where, pi)
from scipy.interpolate import RegularGridInterpolator
from pyproj import CRS, Transformer
import h5py

import os

from grid_generator import GridGenerator
from utils import normalize, generic_transformation_matrix


class AntennaPattern:

    def __init__(self, config, band):
        self.config = config
        self.band=band

        if self.config.antenna_method == 'real':

            ap_dict = self.load_antenna_patterns()

            if self.config.polarisation_method == 'scalar':
                self.scalar_gain = self.get_scalar_pattern(ap_dict)
            elif self.config.polarisation_method == 'mueller':
                self.mueller_matrix = self.get_mueller_matrix(ap_dict)

        elif self.config == 'gaussian':  #in this case I guess only scalar gain available

            self.scalar_gain = self.gaussian_antenna_patterns()

        self.max_ap_radius = self.estimate_max_ap_radius(ap_dict)

        return


    @staticmethod
    def extract_gain_dict(file_path, threshold_dB, theta_max=30):

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

        mask = full(ap_dict['Gnorm'].shape, True)
        if threshold_dB is not None:
            mask = logical_and(mask, ap_dict['Gnorm'] < 10**(threshold_dB/10.))
        if theta_max is not None:
            mask = logical_and(mask, ap_dict['theta'] <= theta_max)

        ap_dict['Ghco'][mask] = 0.
        ap_dict['Ghcx'][mask] = 0.
        ap_dict['Gvco'][mask] = 0.
        ap_dict['Gvcx'][mask] = 0.
        ap_dict['Gnorm'][mask] = 0.

        return ap_dict

    def load_antenna_patterns(self):

        # We might also totally remove this IF-ELSE, if the user 
        # is forced to specify the targetBand config parameter also for SMAP
        if self.config.input_data_type == "SMAP":
            ap_dict = {}
            ap_dict[0] = self.extract_gain_dict(
                threshold_dB=self.config.antenna_threshold,
                file_path = self.config.antenna_pattern_path
            )

        elif self.config.input_data_type == "CIMR":
            num_horns = self.config.num_horns[self.band]
            horn_dict = {}
            ap_dict = {}
            for feedhorn in range(num_horns):
                path = os.path.join(
                    self.config.antenna_pattern_path, self.band)
                horn = self.band + str(feedhorn)

                for file in os.listdir(path):
                    if horn in file:
                        ap_dict[int(feedhorn)] = self.extract_gain_dict(
                            file_path=os.path.join(path, file),
                            threshold_dB=self.config.antenna_threshold
                        )
        return ap_dict

    def gaussian_antenna_patterns(self):

        # JOSEPH: List of parameters needed. For now I just hard-code them below-
        # - resolution in the phi and theta dimensions
        # - sigma_u, sigma_v (aka x, y)
        # - rotation 
        # - max_theta (default pi/2)
        # - threshold_dB (default -100)

        resolution_phi   = 100
        resolution_theta = 100 
        theta_max = np.deg2rad(30.)
        sigma_u = 0.2
        sigma_v = 0.2
        rot = 0.

        def f_scalar_gain(phi, theta):
            phi   = np.atleast_1d(phi)
            theta = np.atleast_1d(theta)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            Z = np.exp(-0.5 * (( ( x*np.cos(rot)+y*np.sin(rot) )**2 )/sigma_u**2  + ( ( -x*np.sin(rot)+y*np.cos(rot)  )**2 )/sigma_v**2))
            Z[theta < theta_max] = 0.
            Z[Z < 10**(threshold_dB/10.)] = 0.
        return Z

        scalar_pattern = {}
        scalar_pattern['L0'] = f_scalar_gain

        return scalar_pattern

    def get_scalar_pattern(self, ap_dict):

        scalar_pattern = {}

        for horn in (ap_dict):

            phi = ap_dict[horn]['phi']
            theta = ap_dict[horn]['theta']

            f_gain_hco = RegularGridInterpolator((phi, theta), ap_dict[horn]['Ghco'])
            f_gain_hcx = RegularGridInterpolator((phi, theta), ap_dict[horn]['Ghcx'])
            f_gain_vco = RegularGridInterpolator((phi, theta), ap_dict[horn]['Gvco'])
            f_gain_vcx = RegularGridInterpolator((phi, theta), ap_dict[horn]['Gvcx'])
            
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

            f_gain_hco = RegularGridInterpolator((phi, theta), ap_dict[horn]['Ghco'])
            f_gain_hcx = RegularGridInterpolator((phi, theta), ap_dict[horn]['Ghcx'])
            f_gain_vco = RegularGridInterpolator((phi, theta), ap_dict[horn]['Gvco'])
            f_gain_vcx = RegularGridInterpolator((phi, theta), ap_dict[horn]['Gvcx'])

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

        # To do:
        # - average satellite flight altitude (this should be updated to be calculated from
        # the specific grid point). Joseph needs to add this variable to data_ingestion
        # - Once this is done, davide/joseph can update the functionality here to reflect the change.

        tilt_angle = self.config.antenna_tilt_angle
        satellite_altitude = self.config.max_altitude


        max_radius = {}

        for horn in ap_dict:
            
            Gnorm = ap_dict[horn]['Gnorm']
            ind_max_theta_non_zero = argmax(average(Gnorm, axis=0))
            max_theta_non_zero = ap_dict[horn]['theta'][ind_max_theta_non_zero]

            # brutal approximation, it's not difficult to make it more accurate

            max_radius[horn] = 1.2 * satellite_altitude * (tan(deg2rad(tilt_angle) + max_theta_non_zero)
                                                            - tan(deg2rad(tilt_angle)))

        return max_radius

    def make_integration_grid(self, longitude, latitude):
        # To do:
        # Davide - remove functionality to change between grids as it is not relevant anymore.

        gdef = self.config.grid_definition
        pdef = self.config.projection_definition

        if all(abs(latitude) <= 75.): # 83.
            self.config.grid_definition = 'EASE2_G3km'
            self.config.projection_definition = 'G'

        elif any(latitude > 75.):
            self.config.grid_definition = 'EASE2_N3km'
            self.config.projection_definition = 'N'

        elif any(latitude < -75.):
            self.config.grid_definition = 'EASE2_S3km'
            self.config.projection_definition = 'S'

        xpoints = []
        ypoints = []
        for lon, lat in zip(longitude, latitude):
            x0, y0 = GridGenerator(self.config).lonlat_to_xy(lon, lat)
            xpoints.append(x0)
            ypoints.append(y0)

        margin = max(list(self.max_ap_radius.values()))
        xmin = min(xpoints) - margin
        xmax = max(xpoints) + margin
        ymin = min(ypoints) - margin
        ymax = max(ypoints) + margin

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

            xs = concatenate((xs[xs > xmin1], xs[logical_and(xs > xmin, xs < xmax)], xs[xs < xmax1]))
            ys = ys[logical_and(ys > ymin, ys < ymax)]

        else:
            xs = xs[logical_and(xs > xmin, xs < xmax)]
            ys = ys[logical_and(ys > ymin, ys < ymax)]

        Xs, Ys = meshgrid(xs, ys)
        lons, lats = GridGenerator(self.config).xy_to_lonlat(Xs, Ys)

        self.config.grid_definition = gdef
        self.config.projection_definition = pdef

        return lons, lats

    def antenna_pattern_to_earth(self, int_dom_lons, int_dom_lats, x_pos, y_pos,
                                 z_pos, x_vel, y_vel, z_vel, processing_scan_angle,
                                 feed_horn_number, attitude):
        
        # JOSEPH: add how to retreive the variables
        # DAVIDE: get the offset angles for each bands .. for now I hardcoded band L just to see if everything works

        tilt_angle = self.config.antenna_tilt_angle

        if self.config.input_data_type == "SMAP":
            yax = normalize([x_vel, y_vel, z_vel])
            zax = normalize([x_pos, y_pos, z_pos])
            xax = cross(yax, zax)
            attitude = generic_transformation_matrix(xax, yax, zax, [1,0,0], [0,1,0], [0,0,1]) #maybe better way to compute it, accounting for roll, pitch and yaw
            feed_offset_phi =   0.  # we can try to infer this by trial and error!
            feed_offset_theta = 0. 

        elif self.config.input_data_type == "CIMR":
            # Davide is attitude in the right shape for you here?
            attitude = attitude.reshape(3,3)
            feed_offset_phi = deg2rad(-1.31647126)   #hardcoded for band L
            feed_offset_theta = deg2rad(-2.7457478)  #hardcoded for band L

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
        X, Y, Z = transformer.transform(int_dom_lons, int_dom_lats, int_dom_alts) #m

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
        rot_vec = rot_vec.reshape(-1, 3) @ attitude.T
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

        theta = arccos(Zp)
        phi = sign(Yp) * arccos(Xp / sqrt(Xp** 2 + Yp**2))

        phi[phi < 0] += 2. * pi

        Ginterp=self.scalar_gain[int(feed_horn_number)](phi[0], theta[0])
        Ginterp *= cos_angle_proj

        return Ginterp    
