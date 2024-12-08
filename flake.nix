{
  description = "Flake to reproduce CIMR RGB development environment"; 

  inputs = {
    # Defining which version of nixpkgs to follow 
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05"; 
    #nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; 
    # Set of utils allowing to target different systems 
    flake-utils.url = "github:numtide/flake-utils"; 

    # To use pyproject.nix (we are not using as of now) 
    pyproject-nix.url = "github:nix-community/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, pyproject-nix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let 
      
        # inherit (nixpkgs) lib; 

        # # Pyproject.nix organisation 
        # project = pyproject-nix.lib.project.loadPyprojectDynamic {

        #   inherit pyver; 

        #   projectRoot = ./.; 

        # };

        project = pyproject-nix.lib.project.loadPyproject {
          # Read & unmarshal pyproject.toml relative to this project root.
          # projectRoot is also used to set `src` for renderers such as buildPythonPackage.
          projectRoot = ./.;
        };

        pkgs = nixpkgs.legacyPackages.${system}; 
        python = pkgs.python3; 
      in 
      {
        # Creates dev env, but not installs package 
        # [Note]: Add file to git for this approach to work as expected 
        devShells.default = import ./nix/shell.nix { inherit pkgs; }; 

        # Creat:ng a shell with cartopy compiled from source (the package is broken)
        # devShells.default = with pkgs; mkShell {
        #   buildInputs = pypkgs ++ otherpkgs; 
        # }; 

        # devShells.default = with pkgs; 
        #   let
        #     # Returns a function that can be passed to `python.withPackages`
        #     arg = project.renderers.withPackages { inherit pyver; };

        #     # Returns a wrapped environment (virtualenv like) with all our packages
        #     pythonEnv = pyver.withPackages arg;
        #   in
        #   # Create a devShell like normal.
        #   pkgs.mkShell { packages = [ pythonEnv ]; };

        # TODO; This doesn' work because of Pyresample (not in nixpkgs) and
        # also does not provide any executable whatsoever 
        # 
        # Create a development shell containing dependencies from `pyproject.toml`
        #devShells.default =
        #  let
        #    # Returns a function that can be passed to `python.withPackages`
        #    arg = project.renderers.withPackages { inherit python; };

        #    # Returns a wrapped environment (virtualenv like) with all our packages
        #    pythonEnv = python.withPackages arg;
        #    # Adding custom pyresample 
        #    pyresampleDeps = import ./nix/pyresample.nix { inherit pkgs; }; 
        #  in
        #  # Create a devShell like normal.
        #  pkgs.mkShell { packages = [ pythonEnv ]; }; 

        # defaultPackage = import ./default.nix; 

      }
    ); 
}
