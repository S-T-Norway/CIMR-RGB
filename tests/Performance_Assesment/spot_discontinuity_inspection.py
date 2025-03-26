import sys
sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb')
from data_ingestion import DataIngestion
from config_file import ConfigFile
from numpy import full, nan, unravel_index

# Testing
import matplotlib
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

config_file = '/dpr/Test_cards/SCEPS_1/devalgo_1_bg.xml'
config_object = ConfigFile(config_file)
data_dict = DataIngestion(config_object).ingest_data()

def plot_neighbourhood_of_discontinuity(data_dict, num_samples, feed_horn, band):
    test_out = full((74,  num_samples), nan)
    band = band
    feed_horn = feed_horn
    fontsize = 16
    for i in range(len(data_dict[band]['feed_horn_number_fore'])):
        if data_dict[band]['feed_horn_number_fore'][i] == feed_horn:
            scan_number = data_dict[band]['scan_number_fore'][i]
            sample_number = data_dict[band]['sample_number_fore'][i]
            test_out[int(scan_number), int(sample_number)] = data_dict[band]['bt_h_fore'][i]
    fig, axs = plt.subplots()
    im1 = axs.imshow(test_out, cmap='viridis')
    fig.colorbar(im1, label = 'BT [K]')
    plt.xlabel('Sample Number [-]', fontsize = fontsize)
    plt.ylabel('Scan Number [-]', fontsize = fontsize)
    plt.suptitle(f'{band} Band central_america_scene', fontsize = fontsize)
    plt.tight_layout()

def plot_feed_horns(data_dict, scan_line, feed_horns, sample_start, sample_end, band, num_samples):
    from numpy import full
    scan_line = scan_line
    feed_horns = feed_horns
    sample_start = sample_start
    sample_end = sample_end
    band = band
    fontsize = 16
    test_out = full((feed_horns, num_samples), nan)

    for count, scan in enumerate(data_dict[band]['scan_number_fore']):
        if scan == scan_line:
            if sample_start <= int(data_dict[band]['sample_number_fore'][count]) <= sample_end:
                feed_horn_number = int(data_dict[band]['feed_horn_number_fore'][count])
                test_out[feed_horn_number, int(data_dict[band]['sample_number_fore'][count])] = \
                data_dict[band]['bt_h_fore'][count]

    fig, axs = plt.subplots()
    im1 = axs.imshow(test_out[:, sample_start:sample_end], extent=[sample_start, sample_end, 0, feed_horns],
                     cmap='viridis')
    fig.colorbar(im1, label='BT [K]')
    plt.xlabel('Sample Number [-]', fontsize=fontsize)
    plt.ylabel('Feed Horn Number [-]', fontsize=fontsize)
    plt.suptitle(f'{band} Band central_america_scene - Scan Line = 31', fontsize=fontsize)

def plot_multiple_feed_horns(data_dict, feed_horns, num_samples, band, extent):
    fig, axs = plt.subplots(1, 4)
    fontsize = 12

    for feed_horn in range(feed_horns):
        test_out = full((74, num_samples), nan)

        for i in range(len(data_dict[band]['feed_horn_number_fore'])):
            if data_dict[band]['feed_horn_number_fore'][i] == feed_horn:
                scan_number = data_dict[band]['scan_number_fore'][i]
                sample_number = data_dict[band]['sample_number_fore'][i]
                test_out[int(scan_number), int(sample_number)] = data_dict[band]['bt_h_fore'][i]

        # For KA band
        if band == 'KA':
            row, col = unravel_index(feed_horn, (2, 4))
            im1 = axs[row, col].imshow(test_out[extent[0]:extent[1], extent[2]:extent[3]],
                                        extent = [extent[2], extent[3], extent[0], extent[1]],
                                        cmap='viridis')
            # axs subtitle

            axs[row, col].set_title(f'Horn {feed_horn}')
            cbar = fig.colorbar(im1, ax=axs[row, col], shrink=0.8)
            axs[row, col].set_xlabel(f'Sample Number [-]', fontsize=fontsize)
            axs[row, col].set_ylabel(f'Scan Number [-]', fontsize=fontsize)
        elif band == 'C':
            # For C band

            im1 = axs[feed_horn].imshow(test_out[extent[0]:extent[1], extent[2]:extent[3]],
                                       extent=[extent[2], extent[3], extent[0], extent[1]],
                                       cmap='viridis')
            # axs subtitle

            axs[feed_horn].set_title(f'Horn {feed_horn}')
            cbar = fig.colorbar(im1, ax=axs[feed_horn], shrink=0.8)
            axs[feed_horn].set_xlabel(f'Sample Number [-]', fontsize=fontsize)
            axs[feed_horn].set_ylabel(f'Scan Number [-]', fontsize=fontsize)

        # Set colour bar for each axs
        # axs[feed_horn].colorbar(im1, label='BT [K]')
        # axs[feed_horn].colorbar(im1, label='BT [K]')
        # plt.xlabel('Sample Number [-]', fontsize=fontsize)
        # plt.ylabel('Scan Number [-]', fontsize=fontsize)
        # plt.suptitle(f'{band} Band central_america_scene', fontsize=fontsize)
        # plt.tight_layout()










# Samples KA = 10395, C = 2747
# plot_neighbourhood_of_discontinuity(data_dict, 10395, 0, 'KA')
# plot_feed_horns(data_dict, 31, 8, 4055, 4085, 'KA', 10395)
# plot_multiple_feed_horns(data_dict, 8, 10395, 'KA', [23, 40, 4055, 4085])

# Samples KA = 10395, C = 2747
plot_neighbourhood_of_discontinuity(data_dict, 2747, 0, 'C')
plot_feed_horns(data_dict, 31, 4, 1070, 1110, 'C', 2747)
plot_multiple_feed_horns(data_dict, 4, 2747, 'C', [23, 40, 1070, 1110])