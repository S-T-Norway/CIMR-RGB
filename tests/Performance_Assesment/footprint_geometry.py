import sys
sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb')
from grid_generator import GridGenerator, GRIDS
from config_file import ConfigFile
from netCDF4 import Dataset
from numpy import array
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.patches import Ellipse



# Open netcdf file
file_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/dpr/L1B/CIMR/SCEPS_l1b_sceps_geo_central_america_scene_1_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1.nc'
band = 'L_BAND'

with Dataset(file_path, "r") as data:
    band_data = data[band]
    longitude = array(band_data['lon'][:,:,0])
    latitude = array(band_data['lat'][:,:,0])
    sub_satellite_lat = array(band_data['sub_satellite_lat'][:])
    sub_satellite_lon = array(band_data['sub_satellite_lon'][:])
    scan_angle = array(band_data['scan_angle'][:])
    scan_geometry = longitude.shape
    bt_h = array(band_data['brightness_temperature_h'][:,:,0])

# Covert cordinates to x, y in EASE
config_path = '/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/Performance_Assesment/quick_config.xml'
config_object = ConfigFile(config_path)
grid_object = GridGenerator(config_object, 'G', 'EASE2_G9km')
x_lon, y_lon = grid_object.lonlat_to_xy(longitude.flatten('C'), latitude.flatten('C'))
x_sub, y_sub = grid_object.lonlat_to_xy(sub_satellite_lon, sub_satellite_lat)


scan_angle = scan_angle.flatten('C')
aft_max = 360
aft_min = 180
scan_angle = scan_angle.flatten('C')
scan_aft = scan_angle[(scan_angle < aft_max)&(scan_angle > aft_min)]
scan_fore = scan_angle[(scan_angle <= aft_min)&(scan_angle >= 0)]

x_lon_fore, y_lon_fore = x_lon[(scan_angle <= aft_min)&(scan_angle >= 0)], y_lon[(scan_angle <= aft_min)&(scan_angle >= 0)]
x_lon_aft, y_lon_aft = x_lon[(scan_angle < aft_max)&(scan_angle > aft_min)], y_lon[(scan_angle < aft_max)&(scan_angle > aft_min)]


# Put the data back into scan geometry
# x_lon_scan = x_lon.reshape(scan_geometry)
# y_lon_scan = y_lon.reshape(scan_geometry)
# x_sub_scan = x_sub.reshape(scan_geometry)
# y_sub_scan = y_sub.reshape(scan_geometry)

half_square = 100000
origin_x, origin_y = x_sub[37,173], y_sub[37,173]
x_min, x_max = origin_x - half_square, origin_x + half_square
y_min, y_max = origin_y - half_square, origin_y + half_square

# Extract all x_lon, x_lat points that are within the bounds
x_lon_focus = x_lon[(x_lon >= x_min) & (x_lon <= x_max) & (y_lon >= y_min) & (y_lon <= y_max)]
y_lon_focus = y_lon[(x_lon >= x_min) & (x_lon <= x_max) & (y_lon >= y_min) & (y_lon <= y_max)]
x_lon_focus_fore = x_lon_fore[(x_lon_fore >= x_min) & (x_lon_fore <= x_max) & (y_lon_fore >= y_min) & (y_lon_fore <= y_max)]
y_lon_focus_fore = y_lon_fore[(x_lon_fore >= x_min) & (x_lon_fore <= x_max) & (y_lon_fore >= y_min) & (y_lon_fore <= y_max)]
x_lon_focus_aft = x_lon_aft[(x_lon_aft >= x_min) & (x_lon_aft <= x_max) & (y_lon_aft >= y_min) & (y_lon_aft <= y_max)]
y_lon_focus_aft = y_lon_aft[(x_lon_aft >= x_min) & (x_lon_aft <= x_max) & (y_lon_aft >= y_min) & (y_lon_aft <= y_max)]

scan_angle_focus = scan_angle[(x_lon >= x_min) & (x_lon <= x_max) & (y_lon >= y_min) & (y_lon <= y_max)]
scan_angle_focus_fore = scan_fore[(x_lon_fore >= x_min) & (x_lon_fore <= x_max) & (y_lon_fore >= y_min) & (y_lon_fore <= y_max)]
scan_angle_focus_aft = scan_aft[(x_lon_aft >= x_min) & (x_lon_aft <= x_max) & (y_lon_aft >= y_min) & (y_lon_aft <= y_max)]

fig, axs = plt.subplots()
relative_points_x = x_lon_focus_fore - origin_x
relative_points_y = y_lon_focus_fore - origin_y
axs.scatter(relative_points_x, relative_points_y)
ellipse_width = 43000
ellipse_height = 73000
for xi, yi, angle in zip(relative_points_x, relative_points_y, scan_angle_focus_fore):
    ellipse = Ellipse(
        xy=(xi, yi),
        width = ellipse_width,
        height = ellipse_height,
        angle = 90-angle,
        edgecolor='r',
        facecolor='none',
        alpha = 0.3
    )
    axs.add_patch(ellipse)
# axs.scatter(0, 0, c='r')
axs.set_xlim(0 - half_square, 0+ half_square)
axs.set_ylim(0-half_square, 0+half_square)
axs.grid(True)
plt.show()






