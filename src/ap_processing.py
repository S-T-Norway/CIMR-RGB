import numpy as np
import os


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
    def extract_gain_dict(file_path, theta_max=np.pi/2., threshold_dB=-100):

        # JOSEPH: how are the parameters theta_max and threshold_dB passed? for now I hardcode them here

        theta_max = np.deg2rad(30.)
        threshold_dB = 1.

        ap_dict = {}

        with h5py.File(file_path, 'r') as f:
            gains = f['Gain']
            phi = f['Grid']['phi'][:]
            theta = f['Grid']['theta'][:]

            gain_dict = {}
            for gain in gains:
                gain_dict[gain] = gains[gain][:]

        ap_dict['theta'] = np.deg2rad(theta)
        ap_dict['phi'] = np.deg2rad(phi)

        ap_dict['Ghco'] = gain_dict['G1h'] + 1j * gain_dict['G2h']
        ap_dict['Ghcx'] = gain_dict['G3h'] + 1j * gain_dict['G4h']
        ap_dict['Gvco'] = gain_dict['G1v'] + 1j * gain_dict['G2v']
        ap_dict['Gvcx'] = gain_dict['G3v'] + 1j * gain_dict['G4v']

        ap_dict['Gnorm'] = 0.5* (np.sqrt(np.abs(Ghco)**2+np.abs(Ghcx)**2) + np.sqrt(np.abs(Gvco)**2+np.abs(Gvcx)**2))

        mask = np.full(Gnorm.shape, True)
        if threshold_dB is not None:
            mask = np.logical_and(mask, Gnorm < 10**(threshold_dB/10.))
        if theta_max is not None:
            mask = np.logical_and(mask, ap_dict['theta'] <= theta_max)
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
            ap_dict['L0'] = self.extract_gain_dict(                
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
                        ap_dict[horn] = self.extract_gain_dict(
                            file_path=os.path.join(path, file)
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
                Ghco_norm = np.abs(f_gain_hco((phi, theta)))
                Ghcx_norm = np.abs(f_gain_hcx((phi, theta)))
                Gvco_norm = np.abs(f_gain_vco((phi, theta)))
                Gvcx_norm = np.abs(f_gain_vcx((phi, theta)))
                Gv = np.sqrt(Gvco_norm**2 + Gvcx_norm**2)
                Gh = np.sqrt(Ghco_norm**2 + Ghcx_norm**2)
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

            mueller_matrix[horn] = f_mueller_matrix

        return mueller_matrix


    def estimate_max_ap_radius(self, ap_dict):

        # JOSEPH: List of parameters neede (for now just hardcoded):
        # - tilt angle
        # - average satellite flight altitude

        tilt_angle = 45.
        satellite_altitude = 700000. #m

        max_radius = {}

        for horn in ap_dict:
            
            Gnorm = ap_dict[horn]['Gnorm']
            ind_max_theta_non_zero = np.argmax(np.average(Gnorm, axis=0))
            max_theta_non_zero = ap_dict[horn]['theta'][ind_max_theta_non_zero]

            # brutal approximation, it's not difficult to make it more accurate

            max_radius[horn] = 1.2 * satellite_altitude * (np.tan(np.deg2rad(tilt_angle) + max_theta_non_zero) 
                                                            - np.tan(np.deg2rad(tilt_angle)))

        return max_radius



    def make_integration_grid(self, longitudes, latitudes, horn):

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

        margin = self.max_ap_radius[horn]
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



    def antenna_pattern_to_earth(self, sample_dict, variable_dict, int_dom_lons, int_dom_lats):
        
        # JOSEPH: add how to retreive the variables
        # DAVIDE: get the offset angles for each bands .. for now I hardcoded band L just to see if everything works 

        tilt_angle =  # ...............
        scan_angle =  # ...............
        x_pos = # ...............
        y_pos = # ...............
        z_pos = # ...............

        if self.config.input_data_type == "SMAP":
            x_vel = # ...............
            y_vel = # ...............
            z_vel = # ...............
            yax = normalize([x_vel, y_vel, z_vel])
            zax = normalize([x_pos, y_pos, z_pos])
            xax = np.cross(yax, zax)  
            attitude = generic_transformation_matrix(xax, yax, zax, [1,0,0], [0,1,0], [0,0,1]) #maybe better way to compute it, accounting for roll, pitch and yaw
            feed_offset_phi =   0.  # we can try to infer this by trial and error!
            feed_offset_theta = 0. 

        elif self.config.input_data_type == "CIMR":
            attitude = self.get_l1b_data(data_dict, 'attitude', scan_ind, earth_sample_ind).reshape(3,3)
            feed_offset_phi = np.deg2rad(-1.31647126)   #hardcoded for band L
            feed_offset_theta = np.deg2rad(-2.7457478)  #hardcoded for band L

        # zaxis = feedhorn boresight, and xaxis perpendicolar to it and with upvector pointing in the same direction
        antenna_zaxis = np.array([-np.sin(tilt_angle+feed_offset_theta)*np.cos(scan_angle+feed_offset_phi), 
                                   np.sin(tilt_angle+feed_offset_theta)*np.sin(scan_angle+feed_offset_phi), 
                                  -np.cos(tilt_angle+feed_offset_theta)])
        antenna_yaxis = np.array([-np.cos(tilt_angle+feed_offset_theta)*np.cos(scan_angle+feed_offset_phi), 
                                   np.cos(tilt_angle+feed_offset_theta)*np.sin(scan_angle+feed_offset_phi), 
                                   np.sin(tilt_angle+feed_offset_theta)])
        antenna_xaxis = np.cross(antenna_yaxis, antenna_zaxis)

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

        rot_vec = np.stack((X, Y, Z), axis=-1)
        shape = rot_vec.shape
        rot_vec = rot_vec.reshape(-1, 3) @ attitude.T
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
        Xp = X*antenna_xaxis[0] + Y*antenna_xaxis[1] + Z*antenna_xaxis[2]
        Yp = X*antenna_yaxis[0] + Y*antenna_yaxis[1] + Z*antenna_yaxis[2]
        Zp = X*antenna_zaxis[0] + Y*antenna_zaxis[1] + Z*antenna_zaxis[2]

        theta = np.arccos(Zp)
        phi = np.sign(Yp) * np.arccos(Xp / np.sqrt(Xp** 2 + Yp**2))

        phi[phi < 0] += 2. * np.pi

        Ginterp  = self.scalar_gain_function(phi, theta)
        Ginterp *= cos_angle_proj

        return Ginterp    
