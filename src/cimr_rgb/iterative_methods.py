from numpy import concatenate, logical_and, zeros, zeros_like, linalg, meshgrid, atleast_1d, minimum, maximum, dot, arange
import numpy as np
import matplotlib.pyplot as plt
from .grid_generator import GridGenerator
from .ap_processing import AntennaPattern, make_integration_grid
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

    for i in range(n_iter):
        residual = Y - A @ X
        regularization_term = lambda_param * X
        X_new = X + alpha * (At @ residual - regularization_term)
        rel_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)
        if rel_error < rtol:
            print(f"Converged in {i+1} iterations with relative error: {rel_error}")
            return X_new
        X = X_new

    print("Reached maximum iterations without full convergence.")
    return X



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

    AtA = A.T @ A + lambda_param * np.eye(A.shape[1])
    AtY = A.T @ Y

    X = np.ones(AtA.shape[1])*1e-20  # Initial guess
    r = AtY - AtA @ X                # Residual
    p = r.copy()                     # Initial search direction
    rs_old = np.dot(r, r)

    for i in range(n_iter):
        Ap = AtA @ p
        alpha = rs_old / np.dot(p, Ap)
        X_new = X + alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)

        rel_error = np.linalg.norm(X_new - X) / np.linalg.norm(X)
        if rel_error < rtol:
            print(f"Converged in {i+1} iterations with relative error: {rel_error}")
            return X_new

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        X = X_new

    print("Reached maximum iterations without full convergence.")
    return X


class MIIinterp:

    #this can be called ONLY for L1c

    def __init__(self, config, band, inversion_method):
        self.config = config
        self.band   = band
        self.inversion_method = inversion_method

    def interp_variable_dict(self, **kwargs):

        samples_dict   = kwargs['samples_dict']
        variable_dict  = kwargs['variable_dict']
        target_dict    = None
        target_grid    = kwargs['target_grid']
        scan_direction = kwargs['scan_direction']
        band           = kwargs['band']


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
        ###

        output_grid = GridGenerator(self.config, self.config.projection_definition, self.config.grid_definition)

        #define output variable dictionary
        variable_dict_out = {}
        for variable in self.config.variables_to_regrid:
            variable_dict_out[variable] = np.zeros((len(target_grid[0]), len(target_grid[1])))

        #define sample pattern
        source_ap = AntennaPattern(config=self.config,
                                   band=band,
                                   antenna_method = self.config.source_antenna_method,
                                   polarisation_method = self.config.polarisation_method,
                                   antenna_threshold=self.config.source_antenna_threshold,
                                   gaussian_params=self.config.source_gaussian_params)

        #get max radius of patterns among the feedhorns
        Rpattern = max(source_ap.max_ap_radius.values())

        #longitude and latitude of reduced grid
        target_lon = target_grid[0].flatten('C')[samples_dict['grid_1d_index']]
        target_lat = target_grid[1].flatten('C')[samples_dict['grid_1d_index']]    

        #sample coordinates in the integration grid (useful later for masking coordinates in a chunk)
        integration_grid =  GridGenerator(self.config, self.config.MRF_projection_definition, self.config.MRF_grid_definition)
        xsample, ysample = integration_grid.lonlat_to_xy(variable_dict['longitude'], variable_dict['latitude'])

        #rows and columns of reduced grid into the total output grid
        rows_output, cols_output = np.unravel_index(samples_dict['grid_1d_index'], (output_grid.n_rows, output_grid.n_cols))
        imin, imax = rows_output.min(), rows_output.max()
        jmin, jmax = cols_output.min(), cols_output.max()

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
                chunkx1, chunky1 = output_grid.rowcol_to_xy(imin + i1, jmin + j1)   # 1 --- 2
                chunkx2, chunky2 = output_grid.rowcol_to_xy(imin + i1, jmin + j2)   # |     |
                chunkx3, chunky3 = output_grid.rowcol_to_xy(imin + i2, jmin + j1)   # |     |
                chunkx4, chunky4 = output_grid.rowcol_to_xy(imin + i2, jmin + j2)   # 3 --- 4

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

                    projected_pattern=source_ap.antenna_pattern_to_earth(
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

                    projected_pattern /= np.sum(projected_pattern)   #implement ap threshold here
                    A[irow] = projected_pattern.flatten()

                    irow += 1

                for variable in self.config.variables_to_regrid:

                    Y = variable_dict[variable][mask]

                    if self.inversion_method == 'CG':
                        X1 = conjugate_gradient_ne(A, Y, 
                                                   lambda_param = self.config.regularization_parameter, 
                                                   n_iter       = self.config.max_number_iteration,
                                                   rtol         = self.config.relative_tolerance
                        )

                    elif self.inversion_method == 'LW':
                        X1 = landweber(A, Y,
                                       lambda_param = self.config.regularization_parameter, 
                                       n_iter       = self.config.max_number_iteration,
                                       rtol         = self.config.relative_tolerance
                        )

                    isx = n_cell_sx
                    idx = int_dom_lons.shape[1] - n_cell_dx 
                    jup = n_cell_up
                    jdn = int_dom_lons.shape[0] - n_cell_dn

                    variable_dict_out[variable][imin+i1:imin+i2, jmin+j1:jmin+j2] = X1.reshape(int_dom_lons.shape)[jup:jdn, isx:idx]

        plt.figure()
        plt.imshow(variable_dict_out[variable][imin:imax, jmin:jmax])
        plt.show()

        return variable_dict_out