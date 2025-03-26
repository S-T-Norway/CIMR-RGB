from numpy import concatenate,array,where, logical_and, zeros, zeros_like, linalg, meshgrid, atleast_1d, minimum, maximum, dot, arange, sum, full
import numpy as np
import matplotlib.pyplot as plt
from .grid_generator import GridGenerator
from .ap_processing import AntennaPattern, GaussianAntennaPattern, make_integration_grid
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata


def landweber(A, Y, lambda_param=1e-4, alpha=None, n_iter=1000, rtol=1e-5):

    """
    Perform Landweber iteration with Tikhonov regularisation to solve AX = Y.
    
    Parameters:
        A (2D numpy array): matrix A in the equation AX=Y
        Y (1D or 2D numpy array): the observed data in the equation AX=Y
        lambda_param (float) regularisation parameter for Tikhonov regularisation.
        alpha (float): step size
        n_iter (int): maximum number of iterations.
        rtol (float): tolerance for the relative error stopping criterion.
    
    Returns:
        numpy array: the estimated solution
        int: number of iterations
    """

    X = np.ones(A.shape[1])*1e-20
    At = A.T

    if alpha is None:
        alpha = 1./np.linalg.norm(np.dot(A.T,A))

    count = 0
    for i in range(n_iter):
        residual = Y - A @ X
        regularisation_term = lambda_param * X
        rel_error = np.linalg.norm(residual) / np.linalg.norm(Y)
        if rel_error < rtol:
            # print(f"Converged in {i+1} iterations with residual {np.linalg.norm(residual)} and relative error {rel_error}")
            return X, count
        else:
            X = X + alpha * (At @ residual - regularisation_term)
            count += 1

    # print(f"Reached maximum iterations without full convergence: residual {np.linalg.norm(residual)} and relative error {rel_error}")
    return X, count


def landweber_nedt(A, varY, lambda_param, alpha, n_iter):

    if alpha is None:
        alpha = 1./np.linalg.norm(np.dot(A.T,A))

    At  = A.T
    AtA = At @ A 
    xlen = A.shape[1]
    B = np.identity(xlen) - alpha*(AtA + lambda_param*np.identity(xlen))

    if len(varY.shape)==1:
        varX = np.zeros(xlen)
        for i in range(n_iter):
            varX = np.sum(A * (varY[:, None] * A), axis=0) + np.sum(B * (varX[:, None] * B), axis=0)

    else:
        varX = np.zeros((xlen,xlen))
        for i in range(n_iter):
            varX = alpha**2 * At @ (varY @ A) + B @ (varX @ B.T)
        varX = np.diagonal(varX)
    
    return varX


def conjugate_gradient_ne(A, Y, lambda_param=1e-4, n_iter=1000, rtol=1e-5):

    """
    Perform Conjugate Gradient iteration with Tikhonov regularisation to solve AX = Y.
    
    Parameters:
        A (2D numpy array): matrix A in the equation AX=Y
        Y (1D pr 2D numpy array): the observed data in the equation AX=Y
        lambda_param (float) regularisation parameter for Tikhonov regularisation.
        n_iter (int): maximum number of iterations.
        rtol (float): tolerance for the relative error stopping criterion.
    
    Returns:
        numpy array: the estimated solution.
        int: number of iterations
    """

    AtA  = A.T @ A
    AtAl = AtA + lambda_param * np.eye(A.shape[1])
    AtY  = A.T @ Y

    X = np.ones(AtA.shape[1])*1e-20  # Initial guess
    r = AtY - AtAl @ X
    p = np.copy(r)
    rnorm_old = np.dot(r, r)

    count = 0
    for i in range(n_iter):
        rel_error = np.sqrt(rnorm_old) / np.linalg.norm(AtY)
        if rel_error < rtol:
            # print(f"Converged in {i+1} iterations with residual {np.sqrt(rnorm_old)} and relative error {rel_error}")
            return X, count
        else:
            AtAlp = AtAl @ p
            a   = rnorm_old / np.dot(p, AtAlp)
            X  += a * p
            r  -= a * AtAlp
            rnorm_new = np.dot(r, r)
            p   = r + rnorm_new / rnorm_old * p
            rnorm_old = rnorm_new
            count +=1

    # print(f"Reached maximum iterations without full convergence: residual {np.sqrt(rnorm_old)} and relative error {rel_error}")
    return X, count


def conjugate_gradient_ne_nedt(A, varY, n_iter):

    #to be implemented

    return None


class MIIinterp:

    def __init__(self, config, band, inversion_method):

        # Add this to config file instead
        assert config.grid_type == 'L1C', "Matrix inversion interpolation methods are L1c, please change grid_type to L1c"

        self.config = config
        self.band   = band
        self.inversion_method = inversion_method
        
        if self.config.source_antenna_method == 'gaussian':
            self.source_ap = GaussianAntennaPattern(config=self.config, 
                                                    antenna_threshold=self.config.source_antenna_threshold)

        else:
            self.source_ap = AntennaPattern(config=self.config,
                                            band=self.band,
                                            antenna_method=self.config.source_antenna_method,
                                            polarisation_method=self.config.polarisation_method,
                                            antenna_threshold=self.config.source_antenna_threshold,
                                            gaussian_params=self.config.source_gaussian_params)


    def apply_inversion(self, variable_dict, samples_dict, target_grid):

        """
        Returns the interpolated value on the target points, for all variables.

        Parameters:
            samples_dict (dictionary of arrays with shape (# target points, self.config.max_neighbours)):
                'distances': distance of a target point to the nearest neighbours
                'indexes': index of the nearest neighbours in the flattened array of source points
                'grid_1d_index': index of the target point in the flattened array of target points
            variable_dict (dictionary of arrays with shape (# source points, 1)): values of the variable to be regridded
                keys are L1b variables names in the source data (with no suffix if split_fore_aft=True, otherwise with either _fore or _after suffix)
                values are 1d arrays with the values of variables on the source points
            target_grid (list of size 2): [array of longitude of target points, array of latitudes of taget points]

        Returns:
            dictionary of arrays with shape (# target points, 1): 
                keys are the names of the regridded variables (with no suffix if split_fore_aft=True, otherwise with either _fore or _after suffix)
                values are 1d arrays with the interpolated values of regridded variables on the target points    
        """

        output_grid = GridGenerator(self.config, self.config.projection_definition, self.config.grid_definition)

        if self.config.source_antenna_method == 'gaussian':
            Rpattern = self.source_ap.estimate_max_ap_radius(self.config.source_gaussian_params[0], self.config.source_gaussian_params[1])
        else:
            Rpattern = max(self.source_ap.max_ap_radius.values())

        #sample coordinates in the integration grid projection (useful later for masking coordinates in a chunk)
        integration_grid =  GridGenerator(self.config, self.config.MRF_projection_definition, self.config.MRF_grid_definition)
        xsample, ysample = integration_grid.lonlat_to_xy(variable_dict['longitude'], variable_dict['latitude'])

        resolution_ratio = output_grid.resolution / integration_grid.resolution

        #corner indexes in the output grid
        if self.config.reduced_grid_inds:
            j0min, j0max, i0min, i0max = self.config.reduced_grid_inds
        else:
            i0min = 0
            j0min = 0
            i0max = output_grid.n_cols
            j0max = output_grid.n_rows

        #corner indexes in the integration grid 
        imin = int(np.floor(i0min*resolution_ratio))
        jmin = int(np.floor(j0min*resolution_ratio))
        imax = int(np.ceil(i0max*resolution_ratio))
        jmax = int(np.ceil(j0max*resolution_ratio))

        # Temporary solution to not regrid variables you dont want to regrid
        variable_dict_out = {}
        variable_dict_int = {}
        for variable in variable_dict:
            if variable not in self.config.variables_to_regrid:
                continue
            variable_dict_out[variable] = np.zeros((output_grid.n_rows, output_grid.n_cols))
            variable_dict_int[variable] = np.zeros((integration_grid.n_rows, integration_grid.n_cols))

        max_chunk_size = self.config.max_chunk_size
        nchunkx = (imax - imin) // max_chunk_size #number of "full" chunks (so it can be zero)
        nchunky = (jmax - jmin) // max_chunk_size

        #generating coordinates on the whole grid, to pass to make_integration_grid
        xs, ys = integration_grid.generate_grid_xy()
        xs = xs[:, 0]
        ys = ys[::-1, 0]

        #loop over chunks
        for i in range(nchunkx+1): #i = column index

            i1 = max_chunk_size * i
            if i < nchunkx:
                i2 = max_chunk_size * (i+1)
            else:
                i2 = imax-imin

            for j in range(nchunky+1): #j row index
                
                j1 = max_chunk_size * j
                if j < nchunky:
                    j2 = max_chunk_size * (j+1)
                else:
                    j2 = jmax-jmin

                #lon and lat of chunk corners
                # CHECKING
                # Firstly, check the input x and ys are correct.
                # Secondly, don't  we already have this information in the target grid?

                chunklon1, chunklat1 = integration_grid.rowcol_to_lonlat(jmin + j1, imin + i1)     # 1 --- 2
                chunklon2, chunklat2 = integration_grid.rowcol_to_lonlat(jmin + j1, imin + i2-1)   # |     |
                chunklon3, chunklat3 = integration_grid.rowcol_to_lonlat(jmin + j2-1, imin + i1)   # |     |
                chunklon4, chunklat4 = integration_grid.rowcol_to_lonlat(jmin + j2-1, imin + i2-1) # 3 --- 4

                #create an integration grid (slightly larger than the chunks)
                int_dom_lons, int_dom_lats = make_integration_grid(
                    integration_grid, xs, ys,
                    int_projection_definition=self.config.MRF_projection_definition,
                    int_grid_definition=self.config.MRF_grid_definition,
                    longitude=[chunklon1, chunklon2, chunklon3, chunklon4],
                    latitude =[chunklat1, chunklat2, chunklat3, chunklat4],
                    ap_radii = self.config.chunk_buffer*Rpattern
                )

                #find integration grid corners
                intx1, inty1 = integration_grid.lonlat_to_xy(int_dom_lons[0,   0], int_dom_lats[0,   0])   # 1 --- 2
                intx2, inty2 = integration_grid.lonlat_to_xy(int_dom_lons[0,  -1], int_dom_lats[0,  -1])   # |     |
                intx3, inty3 = integration_grid.lonlat_to_xy(int_dom_lons[-1,  0], int_dom_lats[-1,  0])   # |     |
                intx4, inty4 = integration_grid.lonlat_to_xy(int_dom_lons[-1, -1], int_dom_lats[-1, -1])   # 3 --- 4

                intx1 -= integration_grid.resolution/2.
                intx2 += integration_grid.resolution/2.
                intx3 -= integration_grid.resolution/2.
                intx4 += integration_grid.resolution/2.
                inty1 += integration_grid.resolution/2.
                inty2 += integration_grid.resolution/2.
                inty3 -= integration_grid.resolution/2.
                inty4 -= integration_grid.resolution/2.

                n_cell_sx = np.sum(int_dom_lons[0]<chunklon1)   #hack: np.sum counts the number of values in a boolean array
                n_cell_dx = np.sum(int_dom_lons[0]>chunklon2)
                n_cell_dn = np.sum(int_dom_lats[:, 0]<chunklat3)
                n_cell_up = np.sum(int_dom_lats[:, 0]>chunklat1)

                if (intx1 < intx2): #global grid with no data across international date line, north and south polar grids
                    mask = ((xsample >= intx1) * (xsample <= intx2) * 
                            (ysample >= inty3) * (ysample <= inty1) )
                else: #global grid with data across international date line 
                    xeasemin = integration_grid.x_min
                    xeasemax = integration_grid.x_max
                    mask = np.logical_or((xsample >= intx1) * (xsample <= xeasemax),
                                         (xsample <= intx2) * (xsample >= xeasemin) )
                    mask = mask * (ysample >= inty3) * (ysample <= inty1)

                if mask.max():
                    print(f"Working on chunk {i*(nchunky+1)+j+1}/{(nchunkx+1)*(nchunky+1)}")
                else:
                    print(f"Working on chunk {i*(nchunky+1)+j+1}/{(nchunkx+1)*(nchunky+1)}, no samples here")
                    continue

                # AX = Y
                Nsamples = len(xsample[mask])
                A = zeros((Nsamples, int_dom_lons.shape[0]*int_dom_lons.shape[1]))
                irow = 0

                for isample in tqdm(range(Nsamples)):

                    if self.config.source_antenna_method == 'gaussian':
                        projected_pattern = self.source_ap.antenna_pattern_to_earth(
                            int_dom_lons=int_dom_lons,
                            int_dom_lats=int_dom_lats,  
                            lon_nadir=variable_dict['sub_satellite_lon'][mask][isample],
                            lat_nadir=variable_dict['sub_satellite_lat'][mask][isample],
                            lon_l1b=variable_dict['longitude'][mask][isample],
                            lat_l1b=variable_dict['latitude'][mask][isample],
                            sigmax=self.config.source_gaussian_params[0],
                            sigmay=self.config.source_gaussian_params[1]
                        )        
                        fraction_above_threshold = 1. 
                    else:        
                        projected_pattern=self.source_ap.antenna_pattern_to_earth(
                            int_dom_lons=int_dom_lons,
                            int_dom_lats=int_dom_lats,
                            x_pos=variable_dict['x_position'][mask][isample], 
                            y_pos=variable_dict['y_position'][mask][isample],
                            z_pos=variable_dict['z_position'][mask][isample],
                            x_vel=variable_dict['x_velocity'][mask][isample],
                            y_vel=variable_dict['y_velocity'][mask][isample],
                            z_vel=variable_dict['z_velocity'][mask][isample],
                            processing_scan_angle=variable_dict['processing_scan_angle'][mask][isample],
                            feed_horn_number=variable_dict['feed_horn_number'][mask][isample],
                            attitude=variable_dict['attitude'][mask][isample],
                            lon_l1b = variable_dict['longitude'][mask][isample],
                            lat_l1b = variable_dict['latitude'][mask][isample]
                            )

                        fraction_above_threshold = 1.- self.source_ap.fraction_below_threshold[int(variable_dict['feed_horn_number'][mask][isample])]

                        if projected_pattern.any():
                            # projected_pattern /= (fraction_above_threshold*sum(projected_pattern))
                            projected_pattern /= sum(projected_pattern)

                    A[irow] = projected_pattern.flatten()

                    irow += 1

                # var_to_regrid = []
                # for var in self.config.variables_to_regrid:
                #     if 'nedt' in var and 'bt'+var[-2:] not in self.config.variables_to_regrid:
                #         var_to_regrid.append('bt'+var[-2:])
                #     if 'nedt' not in var:
                #         var_to_regrid.append(var)

                for variable in variable_dict:
                    if variable not in self.config.variables_to_regrid:
                        continue

                    Y = variable_dict[variable][mask]

                    if self.inversion_method == 'CG':
                        X1, n_iter = conjugate_gradient_ne(A, Y, 
                                                   lambda_param = self.config.regularisation_parameter,
                                                   n_iter       = self.config.max_iterations,
                                                   rtol         = self.config.relative_tolerance
                        )

                    elif self.inversion_method == 'LW':
                        X1, n_iter = landweber(A, Y,
                                       lambda_param = self.config.regularisation_parameter,
                                       n_iter       = self.config.max_iterations,
                                       rtol         = self.config.relative_tolerance
                        )

                    if variable in self.config.variables_to_regrid:
                        isx = n_cell_sx
                        idx = int_dom_lons.shape[1] - n_cell_dx 
                        jup = n_cell_up
                        jdn = int_dom_lons.shape[0] - n_cell_dn
                        variable_dict_int[variable][jmin+j1:jmin+j2, imin+i1:imin+i2] = X1.reshape(int_dom_lons.shape)[jup:jdn, isx:idx]
                        
                    # nedt_var = 'nedt'+variable[-2:]
                    # if 'bt' in variable and nedt_var in self.config.variables_to_regrid:
                    #     if self.inversion_method == 'CG':
                    #         Xnedt = conjugate_gradient_ne_nedt(A, variable_dict[nedt_var][mask], n_iter)
                    #     elif self.inversion_method == 'LW':
                    #         Xnedt = landweber_nedt(A, variable_dict[nedt_var][mask],
                    #                                                alpha = None,
                    #                                                lambda_param = self.config.regularisation_parameter,
                    #                                                n_iter       = n_iter
                    #         )
                    #     Finterp = RegularGridInterpolator((int_dom_lats[:, 0], int_dom_lons[0]), Xnedt.reshape(int_dom_lons.shape), bounds_error=False, fill_value=0.)
                    #     Rows, Cols = np.meshgrid(np.arange(jmin+j1, jmin+j2), np.arange(imin+i1, imin+i2))
                    #     chunklons, chunklats = output_grid.rowcol_to_lonlat(Rows, Cols)
                    #     Xinterp = Finterp((chunklats, chunklons)).T
                    #     variable_dict_out[variable][jmin+j1:jmin+j2, imin+i1:imin+i2] = Xinterp

        for variable in variable_dict_out:

            if resolution_ratio == 1.:
                variable_dict_out[variable] = variable_dict_int[variable]
            else:
                xint, yint = integration_grid.generate_grid_xy()
                xout, yout = output_grid.generate_grid_xy()
                xout, yout = np.meshgrid(xout.flatten(), yout.flatten())
                Finterp = RegularGridInterpolator((yint.flatten(), xint.flatten()), variable_dict_int[variable], bounds_error=False, fill_value=0.)    
                variable_dict_out[variable] = Finterp((yout, xout))

            variable_dict_out[variable] = variable_dict_out[variable].flatten('C')[samples_dict['grid_1d_index']]

        return variable_dict_out

    def apply_inversion_attempt_2(self, variable_dict, samples_dict, target_grid):
        indexes = samples_dict['indexes']
        fill_value = len(variable_dict[f"longitude"])

        T_out = []
        for target_cell in tqdm(range(indexes.shape[0])):
            target_lon, target_lat = (target_grid[0].flatten('C')[samples_dict['grid_1d_index'][target_cell]],
                                                    target_grid[1].flatten('C')[samples_dict['grid_1d_index'][target_cell]])


            # Get antenna patterns
            samples = indexes[target_cell, :]
            input_samples = samples[samples != fill_value]
            if len(input_samples)<1:
                continue

            pattern_lons = array(variable_dict['longitude'][input_samples])
            pattern_lats = array(variable_dict['latitude'][input_samples])
            pattern_lons = concatenate((pattern_lons, [target_lon]))
            pattern_lats = concatenate((pattern_lats, [target_lat]))
            Rpattern = max(self.source_ap.max_ap_radius.values())

            # Create integration grid
            int_dom_lons, int_dom_lats = make_integration_grid(
                int_projection_definition=self.config.MRF_projection_definition,
                int_grid_definition=self.config.MRF_grid_definition,
                longitude=pattern_lons,
                latitude =pattern_lats,
                ap_radii = Rpattern
            )
            # make source antenna patterns shape input_samples x int_dom_lons.flatten()
            A = zeros((len(input_samples), len(int_dom_lons.flatten())))
            irow=0
            for sample in input_samples:
                projected_pattern = self.source_ap.antenna_pattern_to_earth(
                    int_dom_lons=int_dom_lons,
                    int_dom_lats=int_dom_lats,
                    x_pos=variable_dict['x_position'][sample],
                    y_pos=variable_dict['y_position'][sample],
                    z_pos=variable_dict['z_position'][sample],
                    x_vel=variable_dict['x_velocity'][sample],
                    y_vel=variable_dict['y_velocity'][sample],
                    z_vel=variable_dict['z_velocity'][sample],
                    processing_scan_angle=variable_dict['processing_scan_angle'][sample],
                    feed_horn_number=variable_dict['feed_horn_number'][sample],
                    attitude=variable_dict['attitude'][sample],
                    lon_l1b=variable_dict['longitude'][sample],
                    lat_l1b=variable_dict['latitude'][sample]
                )
                projected_pattern /= sum(projected_pattern)
                A[irow] = projected_pattern.flatten()

            Y = variable_dict['bt_h'][input_samples]

            if self.inversion_method == 'CG':
                X1, n_iter = conjugate_gradient_ne(A, Y,
                                                   lambda_param=self.config.regularisation_parameter,
                                                   n_iter=self.config.max_iterations,
                                                   rtol=self.config.relative_tolerance
                                                   )

            elif self.inversion_method == 'LW':
                X1, n_iter = landweber(A, Y,
                                       lambda_param=self.config.regularisation_parameter,
                                       n_iter=self.config.max_iterations,
                                       rtol=self.config.relative_tolerance
                                       )

            t_out = X1.flatten()[where(( int_dom_lons.flatten() == target_lon) & (int_dom_lats.flatten() == target_lat))[0]]
            # print(t_out)
            T_out.append(t_out)

        return array(T_out)


    def interp_variable_dict(self, **kwargs):

        samples_dict   = kwargs['samples_dict']
        variable_dict  = kwargs['variable_dict']
        target_grid    = kwargs['target_grid']
        scan_direction = kwargs['scan_direction']

        ### keep only variable for the scan direction
        if scan_direction is not None:
            variable_dict = {key: value for key, value in variable_dict.items() if key.endswith(scan_direction)}
        if scan_direction:
            scan_direction = f"_{scan_direction}"
        else:
            scan_direction = ""
        if f"attitude{scan_direction}" not in variable_dict:
            variable_dict[f'attitude{scan_direction}'] = full(variable_dict[f"longitude{scan_direction}"].shape, None)

        variable_dict={key.removesuffix(f'{scan_direction}'): value for key, value in variable_dict.items()}

        variable_dict_out = self.apply_inversion(variable_dict, samples_dict, target_grid)

        # Temporary solution to add the fore/aft prefix
        variable_dict_out = {f"{key}{scan_direction}": value for key, value in variable_dict_out.items()}

        return variable_dict_out