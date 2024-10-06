{
  description = "Flake to reproduce CIMR RGB development environment"; 

  inputs = {
    # Defining which version of nixpkgs to follow 
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05"; 
    #nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; 
    # Set of utils allowing to target different systems 
    flake-utils.url = "github:numtide/flake-utils"; 
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let 
        pkgs = nixpkgs.legacyPackages.${system}; 
        # Python version and associated packages 
        pyver = pkgs.python3; 

        # --------------------------------------------------------------------------------
	# PyResample 
        # --------------------------------------------------------------------------------
	# [Note]: PyResample relies on several different packages which are not
	# shipped inside nixpkgs, so we need to install them first and only
	# then compile the module.  
	versioneer_v028 = with pkgs.python3Packages; buildPythonPackage rec {
          pname     = "versioneer";
          version   = "0.28";
          pyproject = true; 
        
          src = fetchPypi {
            inherit pname version;
            sha256 = "7175ca8e7bb4dd0e3c9779dd2745e5b4a6036304af3f5e50bd896f10196586d6"; 
          };

          build-system = [
            setuptools 
            wheel 
          ];

          doCheck = true; 
        
          meta = with lib; {
            description = "Versioneer - Python utility for managing version strings";
            homepage = "https://github.com/python-versioneer/python-versioneer";
            license = licenses.bsd3;
          };
        };

        # [Note]: donfig 0.8.1 relies only on (==) 0.28 versioneer (not 0.29 or other)
        donfig_v081 = with pkgs.python3Packages; buildPythonPackage rec {
          pname   = "donfig";
          version = "0.8.1";
          pyproject = true; 
        
          src = fetchPypi {
            inherit pname version;
            sha256 = "d1773b550e5f1e0930dee03565ce610d1c53a9f9af95fcc46f587cc70e9d39ff";
          };
        
          build-system = [
            setuptools 
            wheel 
          ]; 

          # Not sure above these lines because it works either way :/ 
          # Propagating the inputs, so they will be available to pyresample 
          propagatedBuildInputs = [ 
            pyyaml 
            versioneer_v028 
          ]; 

          doCheck = true; 
          
          meta = with lib; {
            description = "Donfig - Configuration management for Python";
            homepage = "https://github.com/donfig/donfig";
            license = licenses.bsd3;
          };
        };

        # Pyresample is not available in nixpkgs, so it is required to be compiled
        # from source. It is based on pyproject.toml and relies on several
        # different libraries such numpy, xarray, dask etc., whcih also need to be
        # compiled with the same version of numpy. Therefore, we need to compile
        # numpy first.  
        pyresample_v1282 = with pkgs.python3Packages; buildPythonPackage rec {
          pname = "pyresample"; 
          version = "1.28.2";
          pyproject = true; 

          src = fetchPypi {
            inherit pname version; 
            sha256 = "3f48da510148d9c649dd689ff43ea4a57eb3eae90428b737312fc0502beb3532"; 
          }; 

          build-system = [
            pip 
            setuptools 
            wheel 
          ];

          dependencies = [
            numpy 
            pykdtree 
            xarray  
            cython 
            platformdirs 
            donfig_v081 
            #shapely 
            configobj 
            pyproj 
            # These are to make test work 
            dask 
            matplotlib 
            pytest 
            #pytest-lazy-fixture 
            cartopy 
          ]; 

          doCheck = true; 

          meta = with pkgs.lib; {
            description = "Pyresample - Geospatial resampling of earth data with pyproj and numpy";
            homepage = "https://github.com/pytroll/pyresample";
            license = licenses.bsd3;
          };
        };

        pypkgs = with pkgs; [
          (pyver.withPackages (ppkgs: with ppkgs; [
            ipython 
            jupyter 
	    python-magic 
	    # customly compiled 
	    pyresample_v1282 
	    # pyresample already contains numpy, dask, xarray etc., but maybe
	    # it is a good idea to still write them here 
            numpy 
            scipy 
            xarray 
            pyproj 
            cython 
	    # parsing/io
            h5py 
	    h5netcdf
            netcdf4 
            colorama 
            tqdm 
	    # visualization 
	    cartopy 
	    matplotlib 
	    basemap 
	    basemap-data 
	    # parallelization 
	    joblib 
	    dask 
	    distributed 
	    # other 
	    flake8 

          #  # Building cartopy from source 
          #  #(python3Packages.buildPythonPackage rec {
          #  #  pname = "cartopy";
          #  #  version = "0.23.0"; # You can specify the desired version here

          #  #  src = fetchPypi {
          #  #    inherit pname version;
          #  #    # Gives an error if the link is provided excplicitly 
          #  #    #url = "https://files.pythonhosted.org/packages/a5/00/fed048eeb80129908f8bd5f5762e230864abc8a8fbbc493740bafcb696bd/Cartopy-0.23.0.tar.gz" ;#"file://${toString cartopyArchive}"; 
          #  #    sha256 = "231f37b35701f2ba31d94959cca75e6da04c2eea3a7f14ce1c75ee3b0eae7676"; # Fetch the correct hash from PyPI
          #  #  };

          #  #  propagatedBuildInputs = with python3Packages; [
          #  #    numpy
          #  #    shapely
          #  #    proj
          #  #    pyshp
          #  #    six
          #  #    setuptools_scm
          #  #  ];

          #  #  # Disable tests since not all of them pass and the build may be broken  
          #  #  doCheck = false; 

          #  #  meta = with pkgs.lib; {
          #  #    description = "Cartopy - a cartographic python library with matplotlib support";
          #  #    homepage = "https://scitools.org.uk/cartopy/docs/latest/";
          #  #    license = licenses.mpl20;
          #  #    maintainers = with maintainers; [ ];
          #  #  };
          #  #}) 

          ])) 
        ]; 
        # Other packages (mostly to read hdf5 files etc. from the command-line) 
        otherpkgs = with pkgs; [
          hdf5 
          netcdf 
        ]; 
      in 
      {
        # This approach just does not work, gives an error: file `shell.nix` not found 
        #devShells.default = import ./shell.nix { inherit pkgs; }; 

        # Creating a shell with cartopy compiled from source (the package is broken)
        devShells.default = with pkgs; mkShell {
          buildInputs = pypkgs ++ otherpkgs; 
        }; 
      }
    ); 
}
