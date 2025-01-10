from numpy import concatenate, logical_and, zeros, zeros_like, linalg, meshgrid, atleast_1d, minimum, maximum, dot, arange, sum, full
import numpy as np
import matplotlib.pyplot as plt
from .grid_generator import GridGenerator
from .ap_processing import AntennaPattern, GaussianAntennaPattern, make_integration_grid
from tqdm import tqdm


def landweber(A, Y, lambda_param=1e-4, alpha=None, n_iter=1000, rtol=1e-5):

    """
    Perform Landweber iteration with Tikhonov regularization to solve AX = Y.
    
    Parameters:
    - A: 2D numpy array, the system matrix.
    - Y: 1D or 2D numpy array, the observed data.
    - lambda_param: float, regularization parameter for Tikhonov regularization.
    - alpha: float, step size.
    - n_iter: int, maximum number of iterations.
    - rtol: float, tolerance for the relative error stopping criterion.
    
    Returns:
    - X: numpy array, the estimated solution.
    """

    X = np.ones(A.shape[1])*1e-20
    At = A.T

    if alpha is None:
        alpha = 1./np.linalg.norm(np.dot(A.T,A))

    count = 0
    for i in tqdm(range(n_iter)):
        residual = Y - A @ X
        regularization_term = lambda_param * X
        rel_error = np.linalg.norm(residual) / np.linalg.norm(Y)
        if rel_error < rtol:
            print(f"Converged in {i+1} iterations with residual {np.linalg.norm(residual)} and relative error {rel_error}")
            return X, count
        else:
            X = X + alpha * (At @ residual - regularization_term)
            count += 1

    print(f"Reached maximum iterations without full convergence: residual {np.linalg.norm(residual)} and relative error {rel_error}")
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
        for i in tqdm(range(n_iter)):
            varX = np.sum(A * (varY[:, None] * A), axis=0) + np.sum(B * (varX[:, None] * B), axis=0)

    else:
        varX = np.zeros((xlen,xlen))
        for i in tqdm(range(n_iter)):
            varX = alpha**2 * At @ (varY @ A) + B @ (varX @ B.T)
        varX = np.diagonal(varX)
    
    return varX


def conjugate_gradient_ne(A, Y, lambda_param=1e-4, n_iter=1000, rtol=1e-5):

    """
    Perform Conjugate Gradient iteration with Tikhonov regularization to solve AX = Y.
    
    Parameters:
    - A: 2D numpy array, the system matrix.
    - Y: 1D or 2D numpy array, the observed data.
    - lambda_param: float, regularization parameter for Tikhonov regularization.
    - n_iter: int, maximum number of iterations.
    - rtol: float, tolerance for the relative error stopping criterion.
    
    Returns:
    - X: numpy array, the estimated solution.
    """

    AtA  = A.T @ A
    AtAl = AtA + lambda_param * np.eye(A.shape[1])
    AtY  = A.T @ Y

    X = np.ones(AtA.shape[1])*1e-20  # Initial guess
    r = AtY - AtAl @ X
    p = np.copy(r)
    rnorm_old = np.dot(r, r)

    count = 0
    for i in tqdm(range(n_iter)):
        rel_error = np.sqrt(rnorm_old) / np.linalg.norm(AtY)
        if rel_error < rtol:
            print(f"Converged in {i+1} iterations with residual {np.sqrt(rnorm_old)} and relative error {rel_error}")
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

    print(f"Reached maximum iterations without full convergence: residual {np.sqrt(rnorm_old)} and relative error {rel_error}")
    return X, count


def conjugate_gradient_ne_nedt(A, varY, n_iter):

    #to be implemented

    return None


class MIIinterp:

    def __init__(self, config, band, inversion_method):

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

        output_grid = GridGenerator(self.config, self.config.projection_definition, self.config.grid_definition)

        #define output variable dictionary
        variable_dict_out = {}
        for variable in variable_dict:
            variable_dict_out[variable] = np.zeros(target_grid[0].shape)

        if self.config.source_antenna_method == 'gaussian':
            Rpattern = self.source_ap.estimate_max_ap_radius(self.config.source_gaussian_params[0], self.config.source_gaussian_params[1])
        else:
            Rpattern = max(self.source_ap.max_ap_radius.values())

        #longitude and latitude of reduced grid
        target_lon = target_grid[0].flatten('C')[samples_dict['grid_1d_index']]
        target_lat = target_grid[1].flatten('C')[samples_dict['grid_1d_index']]    

        #sample coordinates in the integration grid (useful later for masking coordinates in a chunk)
        integration_grid =  GridGenerator(self.config, self.config.MRF_projection_definition, self.config.MRF_grid_definition)
        xsample, ysample = integration_grid.lonlat_to_xy(variable_dict['longitude'], variable_dict['latitude'])

        #rows and columns of reduced grid into the total output grid
        rows_output, cols_output = np.unravel_index(samples_dict['grid_1d_index'], (output_grid.n_rows, output_grid.n_cols))
        imin, imax = cols_output.min(), cols_output.max()
        jmin, jmax = rows_output.min(), rows_output.max()

        #TODO: choose optimal max_chunk_size depending on available memory
        max_chunk_size = 100
        nchunkx = (imax - imin) // max_chunk_size #number of "full" chunks (so it can be zero)
        nchunky = (jmax - jmin) // max_chunk_size

        #loop over chunks
        for i in range(nchunkx+1):

            i1 = max_chunk_size * i
            if i < nchunkx:
                i2 = max_chunk_size * (i+1)
            else:
                i2 = imax-imin

            for j in range(nchunky+1):

                print(f"Working on chunk {2*i+j+1}/{(nchunkx+1)*(nchunkx+1)}")
                
                j1 = max_chunk_size * j
                if j < nchunky:
                    j2 = max_chunk_size * (j+1)
                else:
                    j2 = jmax-jmin

                #lon and lat of chunk corners
                chunkx1, chunky1 = output_grid.rowcol_to_xy(jmin + j1, imin + i1)   # 1 --- 2
                chunkx2, chunky2 = output_grid.rowcol_to_xy(jmin + j1, imin + i2)   # |     |
                chunkx3, chunky3 = output_grid.rowcol_to_xy(jmin + j2, imin + i1)   # |     |
                chunkx4, chunky4 = output_grid.rowcol_to_xy(jmin + j2, imin + i2)   # 3 --- 4

                chunklon1, chunklat1 = output_grid.xy_to_lonlat(chunkx1, chunky1)
                chunklon2, chunklat2 = output_grid.xy_to_lonlat(chunkx2, chunky2)
                chunklon3, chunklat3 = output_grid.xy_to_lonlat(chunkx3, chunky3)
                chunklon4, chunklat4 = output_grid.xy_to_lonlat(chunkx4, chunky4)
                
                #create an integration grid (slightly larger than the chunks)
                int_dom_lons, int_dom_lats = make_integration_grid(
                    int_projection_definition=self.config.MRF_projection_definition,
                    int_grid_definition=self.config.MRF_grid_definition,
                    longitude=[chunklon1, chunklon2, chunklon3, chunklon4],
                    latitude =[chunklat1, chunklat2, chunklat3, chunklat4],
                    ap_radii = Rpattern
                )

                #mask samples within the integration grid
                intx1, inty1 = integration_grid.lonlat_to_xy(int_dom_lons[0,   0], int_dom_lats[0,   0])   # 1 --- 2
                intx2, inty2 = integration_grid.lonlat_to_xy(int_dom_lons[0,  -1], int_dom_lats[0,  -1])   # |     |
                intx3, inty3 = integration_grid.lonlat_to_xy(int_dom_lons[-1,  0], int_dom_lats[-1,  0])   # |     |
                intx4, inty4 = integration_grid.lonlat_to_xy(int_dom_lons[-1, -1], int_dom_lats[-1, -1])   # 3 --- 4

                #count number of cells in the integration grid but outside the chunk
                n_cell_sx = np.sum(int_dom_lons[0]<chunklon1)   #hack: np.sum counts the number of values in a boolean array
                n_cell_dx = np.sum(int_dom_lons[0]>chunklon2)
                n_cell_dn = np.sum(int_dom_lats[:, 0]<chunklat3)
                n_cell_up = np.sum(int_dom_lats[:, 0]>chunklat1)                  

                if (intx1 < intx2): #global grid with no data across internatioal date line, north and south polar grids
                    mask = ((xsample >= intx1) * (xsample <= intx2) * 
                            (ysample >= inty3) * (ysample <= inty1) )
                else: #global grid with data across international date line 
                    xeasemin = integration_grid.x_min
                    xeasemax = integration_grid.x_max
                    mask = np.logical_or((xsample >= intx1) * (xsample <= xeasemax),
                                         (xsample <= intx2) * (xsample >= xeasemin) )
                    mask = mask * (ysample >= inty3) * (ysample <= inty1)

                if not mask.max():
                    print('loop to next chunk')
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
                        projected_pattern /= (fraction_above_threshold*sum(projected_pattern))

                    A[irow] = projected_pattern.flatten()

                    irow += 1

                var_to_regrid = []
                for var in self.config.variables_to_regrid:
                    if 'nedt' in var and 'bt'+var[-2:] not in self.config.variables_to_regrid:
                        var_to_regrid.append('bt'+var[-2:])
                    if 'nedt' not in var:
                        var_to_regrid.append(var)

                for variable in var_to_regrid:

                    Y = variable_dict[variable][mask]

                    if self.inversion_method == 'CG':
                        X1, n_iter = conjugate_gradient_ne(A, Y, 
                                                   lambda_param = self.config.regularization_parameter, 
                                                   n_iter       = self.config.max_number_iteration,
                                                   rtol         = self.config.relative_tolerance
                        )

                    elif self.inversion_method == 'LW':
                        X1, n_iter = landweber(A, Y,
                                       lambda_param = self.config.regularization_parameter, 
                                       n_iter       = self.config.max_number_iteration,
                                       rtol         = self.config.relative_tolerance
                        )

                    isx = n_cell_sx
                    idx = int_dom_lons.shape[1] - n_cell_dx 
                    jup = n_cell_up
                    jdn = int_dom_lons.shape[0] - n_cell_dn

                    if variable in self.config.variables_to_regrid:
                        variable_dict_out[variable][jmin+j1:jmin+j2, imin+i1:imin+i2] = X1.reshape(int_dom_lons.shape)[jup:jdn, isx:idx]

                    nedt_var = 'nedt'+variable[-2:]
                    if 'bt' in variable and nedt_var in self.config.variables_to_regrid:
                        if self.inversion_method == 'CG':
                            Xnedt = conjugate_gradient_ne_nedt(A, variable_dict[nedt_var][mask], n_iter)
                        elif self.inversion_method == 'LW':
                            Xnedt = landweber_nedt(A, variable_dict[nedt_var][mask],
                                                                   alpha = None,
                                                                   lambda_param = self.config.regularization_parameter, 
                                                                   n_iter       = n_iter
                            )
                        variable_dict_out[nedt_var][jmin+j1:jmin+j2, imin+i1:imin+i2] = Xnedt.reshape(int_dom_lons.shape)[jup:jdn, isx:idx]

        # JOSEPH: check that the variables are reshaped correctly. right now  variable_dict_out[var] is a 2D array with the shape of the output grid
        for variable in variable_dict_out:
            variable_dict_out[variable] = variable_dict_out[variable][jmin:jmax, imin:imax].flatten()

        return variable_dict_out


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

        return variable_dict_out