from pyproj import CRS, Transformer


def compute_laea_longitudes(proj_string, x_min, x_max, y_min, y_max):
    #transformer = Transformer.from_proj(proj_string, "epsg:4326", always_xy=True)

    ## Transform corners to geographic coordinates
    #lon_min, _ = transformer.transform(x_min, y_min)
    #lon_max, _ = transformer.transform(x_max, y_min)

    # Create a transformer to convert LAEA to geographic coordinates
    transformer = Transformer.from_proj(proj_string, "epsg:4326", always_xy=True)

    # Compute corners
    lon_min, lat_max = transformer.transform(x_min, y_max)  # Top-left corner
    print(f"lon_min = {lon_min}, lat_max = {lat_max}")
    lon_max, lat_max = transformer.transform(x_max, y_max)  # Top-right corner
    print(f"lon_max = {lon_max}, lat_max = {lat_max}")
    lon_min, lat_min = transformer.transform(x_min, y_min)  # Bottom-left corner
    print(f"lon_min = {lon_min}, lat_min = {lat_min}")
    lon_max, lat_min = transformer.transform(x_max, y_min)  # Bottom-right corner
    print(f"lon_max = {lon_max}, lat_min = {lat_min}")

    return lon_min, lon_max


# Projection string for EASE2 North
ease2_n_proj = "+proj=laea +lat_0=90 +lon_0=0 +datum=WGS84 +units=m +no_defs"

# EASE2 North grid properties
x_min, x_max = -9000000, 9000000  # in meters
y_min, y_max = -9000000, 9000000

lon_min, lon_max = compute_laea_longitudes(ease2_n_proj, x_min, x_max, y_min, y_max)
print(f"Longitude range: {lon_min}° to {lon_max}°")
