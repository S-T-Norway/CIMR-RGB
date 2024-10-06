
def interpolate_beamdata(cimr): 
    """
    Method to interpolate beamdata into rectilinear grid. 
    """
    
    #theta_grid, phi_grid = convert_uv_to_tp(u_grid, v_grid) 
    
    cimr["Grid"]['phi_grid']   = cimr["Grid"]['phi_grid'].flatten()
    cimr["Grid"]['theta_grid'] = cimr["Grid"]['theta_grid'].flatten()
    
    cimr["temp"]['Ghh']        = cimr["temp"]['Ghh'].flatten()  
    cimr["temp"]['Ghv']        = cimr["temp"]['Ghv'].flatten()  
    cimr["temp"]['Gvv']        = cimr["temp"]['Gvv'].flatten() 
    cimr["temp"]['Gvh']        = cimr["temp"]['Gvh'].flatten()  

    # Removing NaN values from the data (it is the same as to do: 
    # arr = arr[~np.isnan(arr)])
    mask_phi   = np.logical_not(np.isnan(cimr["Grid"]['phi_grid'])) 
    mask_theta = np.logical_not(np.isnan(cimr["Grid"]['theta_grid']))
    
    mask_Ghh   = np.logical_not(np.isnan(cimr["Gain"]['Ghh']))
    mask_Ghv   = np.logical_not(np.isnan(cimr["Gain"]['Ghv']))
    mask_Gvv   = np.logical_not(np.isnan(cimr["Gain"]['Gvv']))
    mask_Gvh   = np.logical_not(np.isnan(cimr["Gain"]['Gvh']))

    # Logical AND (intersection of non-NaN values in all arrays) 
    mask       = mask_theta * mask_phi * mask_Ghh * mask_Gvv * mask_Gvh * mask_Ghv 
    
    cimr["Grid"]['phi_grid']   = cimr["Grid"]['phi_grid'][mask]
    cimr["Grid"]['theta_grid'] = cimr["Grid"]['theta_grid'][mask]
    
    cimr["Gain"]['Ghh']        = cimr["Gain"]['Ghh'][mask]  
    cimr["Gain"]['Ghv']        = cimr["Gain"]['Ghv'][mask]  
    cimr["Gain"]['Gvv']        = cimr["Gain"]['Gvv'][mask]  
    cimr["Gain"]['Gvh']        = cimr["Gain"]['Gvh'][mask]  

    # TODO: Add programmatic way to do this. Technically, we can do this by
    # using max and min values of phi and theta grids. 
    
    phi_max    = np.max(cimr["Grid"]['phi_grid'])
    phi_min    = np.min(cimr["Grid"]['phi_grid'])

    buffermask = cimr["Grid"]['phi_grid'] > phi_max * 0.975 #6.2
    cimr["Grid"]['phi_grid']   = np.concatenate((cimr["Grid"]['phi_grid'],   cimr["Grid"]['phi_grid'][buffermask] - 2. * np.pi))
    cimr["Grid"]['theta_grid'] = np.concatenate((cimr["Grid"]['theta_grid'], cimr["Grid"]['theta_grid'][buffermask]))
    
    cimr["Gain"]['Ghh']        = np.concatenate((cimr["Gain"]['Ghh'], cimr["Gain"]['Ghh'][buffermask]))  
    cimr["Gain"]['Ghv']        = np.concatenate((cimr["Gain"]['Ghv'], cimr["Gain"]['Ghv'][buffermask]))   
    cimr["Gain"]['Gvv']        = np.concatenate((cimr["Gain"]['Gvv'], cimr["Gain"]['Gvv'][buffermask]))  
    cimr["Gain"]['Gvh']        = np.concatenate((cimr["Gain"]['Gvh'], cimr["Gain"]['Gvh'][buffermask]))   

    buffermask = cimr["Grid"]['phi_grid'] < 0.1
    cimr["Grid"]['phi_grid']   = np.concatenate((cimr["Grid"]['phi_grid'],   cimr["Grid"]['phi_grid'][buffermask] + 2. * np.pi))
    cimr["Grid"]['theta_grid'] = np.concatenate((cimr["Grid"]['theta_grid'], cimr["Grid"]['theta_grid'][buffermask]))
    
    cimr["Gain"]['Ghh']        = np.concatenate((cimr["Gain"]['Ghh'], cimr["Gain"]['Ghh'][buffermask]))  
    cimr["Gain"]['Ghv']        = np.concatenate((cimr["Gain"]['Ghv'], cimr["Gain"]['Ghv'][buffermask]))   
    cimr["Gain"]['Gvv']        = np.concatenate((cimr["Gain"]['Gvv'], cimr["Gain"]['Gvv'][buffermask]))  
    cimr["Gain"]['Gvh']        = np.concatenate((cimr["Gain"]['Gvh'], cimr["Gain"]['Gvh'][buffermask]))   
    
    # Adding buffer for theta 
    # 
    # [Note]: This is done because we are getting several (exactly 1 in the
    # begginning and 3 in the end) NaN values in the grids after interpolation,
    # which happens due to the fact that interpolator does have enough
    # neighboring points on edges. Therefore, we are adding more points to the
    # left, while the right values are simply put to 0 (see explanation below). 
    theta_max  = np.max(cimr["Grid"]['theta_grid'])
    buffermask = cimr["Grid"]['theta_grid'] < 0.1
    #phi_grid   = np.concatenate((phi_grid, phi_grid[buffermask]))
    #theta_grid = np.concatenate((theta_grid, -theta_grid[buffermask]))
    
    cimr["Grid"]['phi_grid']   = np.concatenate((cimr["Grid"]['phi_grid'], cimr["Grid"]['phi_grid'][buffermask]))
    cimr["Grid"]['theta_grid'] = np.concatenate((cimr["Grid"]['theta_grid'], -cimr["Grid"]['theta_grid'][buffermask]))
    #Ghh        = np.concatenate((Ghh, Ghh[buffermask]))  
    #Ghv        = np.concatenate((Ghv, Ghv[buffermask]))   
    #Gvv        = np.concatenate((Gvv, Gvv[buffermask]))  
    #Gvh        = np.concatenate((Gvh, Gvh[buffermask]))   
    cimr["Gain"]['Ghh']        = np.concatenate((cimr["Gain"]['Ghh'], cimr["Gain"]['Ghh'][buffermask]))  
    cimr["Gain"]['Ghv']        = np.concatenate((cimr["Gain"]['Ghv'], cimr["Gain"]['Ghv'][buffermask]))   
    cimr["Gain"]['Gvv']        = np.concatenate((cimr["Gain"]['Gvv'], cimr["Gain"]['Gvv'][buffermask]))  
    cimr["Gain"]['Gvh']        = np.concatenate((cimr["Gain"]['Gvh'], cimr["Gain"]['Gvh'][buffermask]))   
    
    # Should be smaller than the buffer zone defined above
    res = 0.01 
    
    #def interpolate_gain(theta, phi, gain): 
    #    return sp.interpolate.LinearNDInterpolator(list(zip(phi, theta)), gain)
    
    # TODO: The code below takes up the whole minute, so need to be sped up

    # This line is from online tutorial. Just leave it be. 
    #start_time_recen = time.time() 
    
    # Creating rectilinear grid 
    phi        = np.arange(0, 2. * np.pi + res, res)
    theta      = np.arange(0, theta_max, res)
    phi, theta = np.meshgrid(phi, theta)  
    
    # The code below is done in this way because we may run out of memory otherwise 

    # Tried to precompute the triangulation beforehand in the attempt to speed
    # up the code. However, it doesn't seem to work. For instance, the code
    # with Delauney part results in: 
    # 
    # D: 58.31 
    # fhh: 51.37 
    # Ghh: 106.55 
    # fhv: 53.65 
    # Ghv: 128.90
    # fvv: 55.50
    # Gvv: 133.74
    # gvh: 53.80 
    # Gvh: 134.40 
    # 
    # While, without Delaunay we get: 
    # 
    # fhh: 54.11
    # Ghh: 88.94 
    # fhv: 57.13 
    # Ghv: 134.32 
    # fvv: 57.46 
    # Gvv: 117.50 
    # fvh: 57.99 
    # Gvh: 128.28 
    # 
    # Similar things happen for other bands as well. 
    # 

    #import dask as dk  

    #client = Client() 
    #client = Client(threads_per_worker=1, n_workers=2) 
    # Setup Dask cluster with a specified number of threads or processes
    #cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    #client = Client(cluster) 
    #    
    #if False: #True: #cimr['Grid']['nx'] > 1000: 
    #    start_time_inter = time.time() 
    #    points = list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])) 
    #    triangulation = sp.spatial.Delaunay(points)  
    #    end_time_inter = time.time() - start_time_inter 
    #    print(f"| Finished with Delaunay in: {end_time_inter:.2f}s")
    #else: 
    #    triangulation = list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])) 

    #start_time_inter = time.time() 
    #points = list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])) 
    #triangulation = sp.spatial.Delaunay(points)  
    #print(triangulation[1:10]) 

    #triangulation = dk.delayed(sp.spatial.Delaunay)(points)  
    #result = triangulation.compute() 
    #print(result[1:10]) 

    #end_time_inter = time.time() - start_time_inter 

    # Use delayed to parallelize the interpolation process
    #@dask.delayed


    def interpolate_temperature(x, y, z, X, Y, interp_method = "linear"):

        grid_points = np.vstack([X.ravel(), Y.ravel()]).T 

        Z = sp.interpolate.griddata(grid_points, z, (x, y), method=interp_method) 

        #interp = sp.interpolate.LinearNDInterpolator(list(zip(x, y)), z)
        #Z = interp(X, Y)

        return Z

    start_time_inter = time.time() 

    cimr["Gain"]['Ghh'] = da.nan_to_num(interpolate_temperature().T, nan=0.0)  

    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Ghh in: {end_time_inter:.2f}s")


    #phi        = da.arange(0, 2. * np.pi + res, res)
    #theta      = da.arange(0, theta_max, res)
    #phi, theta = da.meshgrid(phi, theta)  

    ## Measure the time of the delayed task creation
    #start_task_creation_time = time.time()

    ## Call the delayed function for interpolation
    #Z_delayed = interpolate_and_meshgrid(cimr['Grid']['phi_grid'],
    #                             cimr['Grid']['theta_grid'],
    #                             cimr['Gain']['Ghh'], phi.compute(),
    #                             theta.compute())

    #task_creation_time = time.time() - start_task_creation_time 

    ## Trigger the computation
    #start_computation_time = time.time()

    #with ProgressBar():
    #    Z = Z_delayed.compute()

    #computation_time = time.time() - start_computation_time 

    #print(f"Time to create delayed task: {task_creation_time:.4f} seconds")
    #print(f"Time to compute task: {computation_time:.4f} seconds") 

    #print(Z.shape)
    #print(Z[0,0])
    #exit() 

    start_time_inter = time.time() 
    cimr["Gain"]['Ghh'] = da.nan_to_num(Z.T, nan=0.0)  
    end_time_inter = time.time() - start_time_inter 
    print(cimr['Gain']['Ghh'])
    print(f"| Finished with Ghh in: {end_time_inter:.2f}s")
    del Z 

    client.shutdown() 

    #start_time_inter = time.time() 
    ##fhh        = sp.interpolate.LinearNDInterpolator(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])), cimr["Gain"]['Ghh']) 
    ##fhh        = sp.interpolate.LinearNDInterpolator(triangulation, cimr["Gain"]['Ghh']) 

    ##end_time_inter = time.time() - start_time_inter 
    ##print(f"| Finished with fhh in: {end_time_inter:.2f}s")

    #start_time_inter = time.time() 
    #triangulation = da.from_array(triangulation, chunks = (1000))  
    #print(triangulation.shape) 
    #print(cimr['Gain']['Ghh'].shape) 
    #cimr['Gain']['Ghh'] = da.from_array(cimr["Gain"]['Ghh'], chunks = (1000))
    #fhh        = dk.delayed(sp.interpolate.LinearNDInterpolator)(triangulation, cimr["Gain"]['Ghh']) 
    #with ProgressBar(): 
    #    fhh.compute() 
    ##fhh        = client.submit(sp.interpolate.LinearNDInterpolator, triangulation, cimr["Gain"]['Ghh']) 

    #end_time_inter = time.time() - start_time_inter 
    #print(f"| Finished with fhh in: {end_time_inter:.2f}s")
    exit() 
    
    start_time_inter = time.time() 
    cimr["Gain"]['Ghh'] = np.nan_to_num(fhh(phi, theta).T, nan=0.0)  
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Ghh in: {end_time_inter:.2f}s")
    del fhh 

    G1h, G2h   = np.real(cimr["Gain"]['Ghh']), np.imag(cimr["Gain"]['Ghh'])  
    del cimr["Gain"]['Ghh'] 
    
    
    start_time_inter = time.time() 
    #fhv        = sp.interpolate.LinearNDInterpolator(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])), cimr["Gain"]['Ghv']) 
    fhv        = sp.interpolate.LinearNDInterpolator(triangulation, cimr["Gain"]['Ghv']) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with fhv in: {end_time_inter:.2f}s")
    
    start_time_inter = time.time() 
    cimr["Gain"]['Ghv'] = np.nan_to_num(fhv(phi, theta).T, nan=0.0)
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Ghv in: {end_time_inter:.2f}s")
    del fhv 
    
    G3h, G4h   = np.real(cimr["Gain"]['Ghv']), np.imag(cimr["Gain"]['Ghv'])  
    del cimr["Gain"]['Ghv'] 

    
    start_time_inter = time.time() 
    #fvv        = sp.interpolate.LinearNDInterpolator(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])), cimr["Gain"]['Gvv']) 
    fvv        = sp.interpolate.LinearNDInterpolator(triangulation, cimr["Gain"]['Gvv']) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with fvv in: {end_time_inter:.2f}s")
    
    start_time_inter = time.time() 
    cimr["Gain"]['Gvv'] = np.nan_to_num(fvv(phi, theta).T, nan=0.0) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Gvv in: {end_time_inter:.2f}s")
    del fvv 

    G1v, G2v   = np.real(cimr["Gain"]['Gvv']), np.imag(cimr["Gain"]['Gvv']) 
    del cimr["Gain"]['Gvv'] 

    
    
    start_time_inter = time.time() 
    #fvh        = sp.interpolate.LinearNDInterpolator(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])), cimr["Gain"]['Gvh']) 
    fvh        = sp.interpolate.LinearNDInterpolator(triangulation, cimr["Gain"]['Gvh']) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with fvh in: {end_time_inter:.2f}s")
    
    start_time_inter = time.time() 
    cimr["Gain"]['Gvh'] = np.nan_to_num(fvh(phi, theta).T, nan=0.0) 
    end_time_inter = time.time() - start_time_inter 
    print(f"| Finished with Gvh in: {end_time_inter:.2f}s")
    del fvh 
    
    G3v, G4v   = np.real(cimr["Gain"]['Gvh']), np.imag(cimr["Gain"]['Gvh'])  
    del cimr["Gain"]['Gvh'] 
    
    exit() 
    #end_time_recen = time.time() - start_time_recen
    #print(f"| Finished Interpolation in: {end_time_recen:.2f}s") 
    

    # SciPy uses Delauney triangulation, which we can precompute and thus speed
    # up the code 
    # Precompute the triangulation
    #start_time_recen = time.time() 
    #
    #triang = sp.spatial.Delaunay(list(zip(cimr["Grid"]['phi_grid'], cimr["Grid"]['theta_grid'])))
    #fhh    = sp.interpolate.LinearNDInterpolator(triang, cimr["Gain"]['Ghh']) #, fill_value=0)
    #fhv    = sp.interpolate.LinearNDInterpolator(triang, cimr["Gain"]['Ghv']) #, fill_value=0)
    #fvv    = sp.interpolate.LinearNDInterpolator(triang, cimr["Gain"]['Gvv']) #, fill_value=0)
    #fvh    = sp.interpolate.LinearNDInterpolator(triang, cimr["Gain"]['Gvh']) #, fill_value=0)
    #
    ## Creating rectilinear grid 
    #phi        = np.arange(0, 2. * np.pi + res, res)
    #theta      = np.arange(0, theta_max, res)
    #phi, theta = np.meshgrid(phi, theta)  
    #
    ## Interpolating the function and substituting the last NaN values in the
    ## arrays with zeros, because they are not intersecting the Earth (once you
    ## do the projection) 
    #
    #cimr["Gain"]['Ghh'] = np.nan_to_num(fhh(phi, theta).T, nan=0.0)  
    #cimr["Gain"]['Ghv'] = np.nan_to_num(fhv(phi, theta).T, nan=0.0)
    #cimr["Gain"]['Gvv'] = np.nan_to_num(fvv(phi, theta).T, nan=0.0) 
    #cimr["Gain"]['Gvh'] = np.nan_to_num(fvh(phi, theta).T, nan=0.0) 
    #
    #end_time_recen = time.time() - start_time_recen
    #print(f"| Finished Interpolation in: {end_time_recen:.2f}s") 


    # Creating rectilinear grid 
    #phi        = np.arange(0, 2. * np.pi + res, res)
    #theta      = np.arange(0, theta_max, res)
    #phi, theta = np.meshgrid(phi, theta)  
    
    # Interpolating the function and substituting the last NaN values in the
    # arrays with zeros, because they are not intersecting the Earth (once you
    # do the projection) 
    
    #cimr["Gain"]['Ghh'] = np.nan_to_num(fhh(phi, theta).T, nan=0.0)  
    #print("| Finished iwht Ghh")
    #del fhh 
    #cimr["Gain"]['Ghv'] = np.nan_to_num(fhv(phi, theta).T, nan=0.0)
    #print("| Finished with Ghv")
    #del fhv 
    #cimr["Gain"]['Gvv'] = np.nan_to_num(fvv(phi, theta).T, nan=0.0) 
    #print("| Finished with Gvv")
    #del fvv 
    #cimr["Gain"]['Gvh'] = np.nan_to_num(fvh(phi, theta).T, nan=0.0) 
    #print("| Finished with Gvh")
    #del fvh 

    phi, theta = phi.T, theta.T 

    # Getting back initial vectors and converting them into degrees 
    phi        = np.rad2deg(np.unique(phi[:, 0])) 
    theta      = np.rad2deg(np.unique(theta[0, :])) 
    
    # Splitting the arrays into real and imaginary parts 
    #G1h, G2h   = np.real(cimr["Gain"]['Ghh']), np.imag(cimr["Gain"]['Ghh'])  
    #G3h, G4h   = np.real(cimr["Gain"]['Ghv']), np.imag(cimr["Gain"]['Ghv'])  
    #G1v, G2v   = np.real(cimr["Gain"]['Gvv']), np.imag(cimr["Gain"]['Gvv']) 
    #G3v, G4v   = np.real(cimr["Gain"]['Gvh']), np.imag(cimr["Gain"]['Gvh'])  
    
    # Getting the resulting dictionary  
    # (and removing unnecessary fields)
                    
    cimr["Gain"]['G1h'] = G1h 
    del G1h 
    cimr["Gain"]['G2h'] = G2h 
    del G2h 
    cimr["Gain"]['G3h'] = G3h 
    del G3h 
    cimr["Gain"]['G4h'] = G4h 
    del G4h 

    cimr["Gain"]['G1v'] = G1v 
    del G1v 
    cimr["Gain"]['G2v'] = G2v 
    del G2v 
    cimr["Gain"]['G3v'] = G3v 
    del G3v 
    cimr["Gain"]['G4v'] = G4v 
    del G4v 
    
    #cimr["Grid"]['u']     = u_grid #u_values #u0  
    #cimr["Grid"]['v']     = v_grid #v_values #v0   
    #cimr["Grid"]['u_cen'] = xcen #u_coordinate  
    #cimr["Grid"]['v_cen'] = ycen #v_coordinate   
    cimr["Grid"]['theta'] = theta #_grid  
    cimr["Grid"]['phi']   = phi #_grid   
    
    #print(cimr["Gain"].keys())
    #print(cimr["Grid"].keys())
                    
    return cimr 
