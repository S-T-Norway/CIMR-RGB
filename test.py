from importlib.resources import files

# Access the directory
target_dir = files('cimr_rgb.dpr.Grids.NSIDC_PS')

# Iterate through files
for file in target_dir.iterdir():
    print(f"Found: {file.name}")


