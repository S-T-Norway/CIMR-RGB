# CIMR-RGB

> This project consists of two projects: CIMR GRASP (to parse CIMR in suitable format) and CIMR RGB (to perform analysis). 



## Installation 

To install the package do: 
```
$ python -m pip install . 
```
or in editable mode: 
```
$ python -m pip install -e . 
```

## Run 

To run the project do: 
```
$ python cimr_rgb ./config.xml
```
or
```
$ cimr-rgb ./config.xml
```



#### Dev Environment with Nix 

To create python dev environment with nix package manager run: 
```
$ nix develop .  
```
If you are using `direnv`, just do: 
```
$ direnv allow . 
```
[__Note__]: It will take some time during the first run to build the environemnt, since nix builds python from source.  

