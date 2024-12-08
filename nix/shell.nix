{ pkgs ? import <nixpkgs> }:
let 
  # Python version and associated packages 
  pyver = pkgs.python3; 
  # [Note]: Do not forget to add the file into git since nix can only
  # track files available to git:
  # git add pyresample.nix 
  pyresampleDeps = import ./pyresample.nix { inherit pkgs; }; 
  # Manually defining packages 
  pypkgs = with pkgs; [
    (pyver.withPackages (ppkgs: with ppkgs; [
      ipython 
      jupyter 
      python-magic 
      # customly compiled 
      #pyresample_v1282 
      pyresampleDeps.pyresample 
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
  otherpkgs = with pkgs; [
    hdf5 
    netcdf 
  ]; 

in 
# see: 
# https://github.com/NixOS/nixpkgs/blob/754290904d5e0ff59e94478febf94ed1ab3cac82/doc/languages-frameworks/python.section.md#develop-local-package-develop-local-package 
with pkgs; mkShell {

  buildInputs = pypkgs ++ otherpkgs; 

}
# TODO: This doesn't since it does not install our package, but it should 
#with pkgs; python3Packages.buildPythonPackage rec {
#  name = "cimr-rgb";
#  src = ./.; #./path/to/package/source;
#  propagatedBuildInputs = pypkgs; #[ pytest numpy pkgs.libsndfile ];
#}
