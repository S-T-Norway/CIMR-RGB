# pyresample.nix

{ pkgs }:

let
  # Define packages required for pyresample
  versioneer_v028 = with pkgs.python3Packages; buildPythonPackage rec {
    pname     = "versioneer";
    version   = "0.28";
    pyproject = true;

    src = fetchPypi {
      inherit pname version;
      sha256 = "7175ca8e7bb4dd0e3c9779dd2745e5b4a6036304af3f5e50bd896f10196586d6";
    };

    build-system = [ setuptools wheel ];
    doCheck = true;

    meta = with lib; {
      description = "Versioneer - Python utility for managing version strings";
      homepage = "https://github.com/python-versioneer/python-versioneer";
      license = licenses.bsd3;
    };
  };

  donfig_v081 = with pkgs.python3Packages; buildPythonPackage rec {
    pname   = "donfig";
    version = "0.8.1";
    pyproject = true;

    src = fetchPypi {
      inherit pname version;
      sha256 = "d1773b550e5f1e0930dee03565ce610d1c53a9f9af95fcc46f587cc70e9d39ff";
    };

    build-system = [ setuptools wheel ];
    propagatedBuildInputs = [ pyyaml versioneer_v028 ];
    doCheck = true;

    meta = with lib; {
      description = "Donfig - Configuration management for Python";
      homepage = "https://github.com/donfig/donfig";
      license = licenses.bsd3;
    };
  };

  pyresample_v1282 = with pkgs.python3Packages; buildPythonPackage rec {
    pname = "pyresample";
    version = "1.28.2";
    pyproject = true;

    src = fetchPypi {
      inherit pname version;
      sha256 = "3f48da510148d9c649dd689ff43ea4a57eb3eae90428b737312fc0502beb3532";
    };

    build-system = [ pip setuptools wheel ];
    dependencies = [
      numpy pykdtree xarray cython platformdirs donfig_v081 configobj
      pyproj dask matplotlib pytest cartopy
    ];
    doCheck = true;

    meta = with pkgs.lib; {
      description = "Pyresample - Geospatial resampling of earth data with pyproj and numpy";
      homepage = "https://github.com/pytroll/pyresample";
      license = licenses.bsd3;
    };
  };

in
{
  # Export pyresample and other dependencies
  pyresample = pyresample_v1282;
  versioneer = versioneer_v028;
  donfig = donfig_v081;
}

