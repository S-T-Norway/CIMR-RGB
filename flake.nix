{
  description = "CIMR RGB Python package and development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;

        # Pin Python explicitly so uv.lock and Nix agree more reliably.
        python = pkgs.python312;

        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        projectName = pyproject.project.name;

        # Load pyproject.toml + uv.lock.
        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };

        # Prefer wheels first. If a wheel is unavailable or incompatible,
        # uv2nix may fall back to building from source depending on the lock
        # data and package.
        uvLockedOverlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        pyprojectOverrides =
          final: prev:
          {
            # Package-specific fixes go here. To be kept empty unless there is
            # a build error. 
          };

        # Base Python package set from pyproject.nix, extended with:
        # 1. common Python build backends,
        # 2. packages resolved from uv.lock,
        # 3. the local fixups.
        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                uvLockedOverlay
                pyprojectOverrides
              ]
            );

        # The actual Nix package for the Python project.
        package = pythonSet.${projectName};

        # A normal, non-editable virtual environment containing all packages
        # and their runtime dependencies.
        runtimeEnv = pythonSet.mkVirtualEnv "${projectName}-env" workspace.deps.default;

        editableOverlay = workspace.mkEditablePyprojectOverlay {
          root = "$REPO_ROOT";
          members = [ projectName ];
        };

        editablePythonSet =
          pythonSet.overrideScope
            (
              lib.composeManyExtensions [
                editableOverlay

                # Needed for editable installs of local workspace packages.
                (
                  final: prev:
                  {
                    ${projectName} = prev.${projectName}.overrideAttrs (old: {
                      nativeBuildInputs =
                        (old.nativeBuildInputs or [ ])
                        ++ final.resolveBuildSystem {
                          editables = [ ];
                        };
                    });
                  }
                )
              ]
            );

        devEnv =
          editablePythonSet.mkVirtualEnv "${projectName}-dev-env" workspace.deps.all;

        inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;

        app =
          mkApplication {
            venv = runtimeEnv;
            package = pythonSet.${projectName};
          };
      in
      {
        packages = {
          default = app;
          ${projectName} = app;
          python-env = runtimeEnv;
        };

        apps = {
          default = {
            type = "app";
            program = "${app}/bin/cimr-rgb";
          };

          cimr-rgb = {
            type = "app";
            program = "${app}/bin/cimr-rgb";
          };

          cimr-grasp = {
            type = "app";
            program = "${app}/bin/cimr-grasp";
          };
        };

        checks = {
          default = app;
        };

        devShells = {
          default = with pkgs; mkShell {
            packages = [
              devEnv
              uv
              hdf5
              netcdf
              proj
              geos
            ];

            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = "${devEnv}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH

              export REPO_ROOT="$(
                git rev-parse --show-toplevel 2>/dev/null || pwd
              )"

              echo "CIMR RGB uv2nix dev shell"
              echo "Python: $(${devEnv}/bin/python --version)"
              echo "Repo root: $REPO_ROOT"
            '';
          };

          impure = pkgs.mkShell {
            packages = [
              python
              pkgs.uv

              pkgs.hdf5
              pkgs.netcdf
              pkgs.proj
              pkgs.geos
            ];

            env = {
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH
              echo "Impure uv shell. Use: uv sync"
            '';
          };
        };
      }
    );
}
