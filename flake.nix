{
  description = "Flake to reproduce CIMR RGB development environment"; 

  inputs = {
    # Defining which version of nixpkgs to follow 
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05"; 
    # Set of utils allowing to target different systems 
    flake-utils.url = "github:numtide/flake-utils"; 
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let 
        pkgs = nixpkgs.legacyPackages.${system}; 
        # Python version and associated packages 
        pyver = pkgs.python310; 
        pypkgs = with pkgs; [
          (pyver.withPackages (ppkgs: with ppkgs; [
            jupyter 
            numpy 
            scipy 
            matplotlib 
            #requests 
            pyproj 
            h5py 
            netcdf4 
            colorama 
            xarray 
            tqdm 
            
            # Building cartopy from source 
            (python3Packages.buildPythonPackage rec {
              pname = "cartopy";
              version = "0.23.0"; # You can specify the desired version here

              src = fetchPypi {
                inherit pname version;
                # Gives an error if the link is provided excplicitly 
                #url = "https://files.pythonhosted.org/packages/a5/00/fed048eeb80129908f8bd5f5762e230864abc8a8fbbc493740bafcb696bd/Cartopy-0.23.0.tar.gz" ;#"file://${toString cartopyArchive}"; 
                sha256 = "231f37b35701f2ba31d94959cca75e6da04c2eea3a7f14ce1c75ee3b0eae7676"; # Fetch the correct hash from PyPI
              };

              propagatedBuildInputs = with python3Packages; [
                numpy
                shapely
                proj
                pyshp
                six
                setuptools_scm
              ];

              # Disable tests since not all of them pass and the build may be broken  
              doCheck = false; 

              meta = with pkgs.lib; {
                description = "Cartopy - a cartographic python library with matplotlib support";
                homepage = "https://scitools.org.uk/cartopy/docs/latest/";
                license = licenses.mpl20;
                maintainers = with maintainers; [ ];
              };
            }) 
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
