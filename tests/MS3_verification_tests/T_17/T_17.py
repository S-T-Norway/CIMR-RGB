# Testing script for T.17
# Remapping of L1b CIMR C-band data with LW (Landweber) algorithm on a Mercator global grid
# The remapped data are compatible with L1c data obtained by regridding with IDS algorithm on the same output grid (average relative difference of the brightness temperature < 1%)
import sys
import os

import matplotlib
import pathlib as pb
import subprocess as sbps
tkagg = matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb')
from grid_generator import GRIDS
from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import cimr_grasp.grasp_io as grasp_io


repo_root = grasp_io.find_repo_root()
# Add RGB remapped IDS netCDF here
ids_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_17/CIMR_L1C_IDS_9km_test.nc"
)  # ""
# Add the RGB remapped BG netCDF here
lw_data_path = pb.Path(repo_root).joinpath(
    "output/MS3_verification_tests/T_17/CIMR_L1C_LW_9km_test.nc"
)
GRID = 'EASE2_G9km'
PROJECTION = 'G'
BAND = 'L_BAND'

class CIMR_comparison:

    def __init__(self, ids_data_path, lw_data_path):
        self.ids_data = self.get_netcdf_data(ids_data_path)
        self.lw_data = self.get_netcdf_data(lw_data_path)

    @staticmethod
    def get_netcdf_data(path):
        import netCDF4 as nc
        gridded_vars = {}
        with nc.Dataset(path, 'r') as f:
            data = f[f"{PROJECTION}"]
            measurement = data['Measurement']
            band = measurement[BAND]
            var = array(band['bt_h'][:])
            row = array(band['cell_row'][:])
            col = array(band['cell_col'][:])

            # print(f"cell row  {row.min(), row.max()}")
            # print(f"cell col {col.min(), col.max()}")


            grid = full((GRIDS[GRID]['n_rows'], GRIDS[GRID]['n_cols']), nan)

            for count, sample in enumerate(var):
                if sample == 9.969209968386869e36:
                    continue
                if sample == 0.:
                    continue
                if row[count] == -9223372036854775806:
                    continue
                if col[count] == -9223372036854775806:
                    continue
                grid[int(row[count]), int(col[count])] = sample
            gridded_vars['bt_h'] = grid
            return gridded_vars

    def map_compare(self):

        cmap = 'viridis'
        # bt_h plt
        fig, axs = plt.subplots(1, 3, constrained_layout=True)
        im00 = axs[0].imshow(self.lw_data['bt_h'], cmap=cmap)
        axs[0].set_title('LW Remap (bt_h)')
        im01 = axs[1].imshow(self.ids_data['bt_h'], cmap=cmap)
        axs[1].set_title('IDS Remap (bt_h)')
        bt_h_fore_diff = abs(self.lw_data['bt_h'] - self.ids_data['bt_h'])
        im02 = axs[2].imshow(bt_h_fore_diff, cmap=cmap)
        axs[2].set_title('Difference (bt_h)')
        # aft

        fig.colorbar(im02, ax=axs[2])


        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_h_fore_diff)

        print(f"Average relative difference for bt_h: {fore_mean_diff}")


        # Calculate percentage Differences
        fore_percent_diff = (fore_mean_diff / nanmean(self.ids_data['bt_h'])) * 100

        print(f"Average percentage difference for bt_h: {fore_percent_diff}")


        # Add statistics to the plot
        # axs[0,2].text(100,300, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
        # axs[0,2].text(100, 300, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
        axs[2].text(100, 420,
                fr"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{IDS}}_i - \mathrm{{LW}}_i |$",
                fontsize=14, color="black")

        axs[2].text(100, 470,
                fr"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
                fontsize=14, color="black")


        plt.show()

    @staticmethod
    def scatter_stats(x, y):
        mask = ~isnan(x) & ~isnan(y) & ~isinf(x) & ~isinf(y)
        x = x[mask]
        y = y[mask]
        m, b = polyfit(x, y, 1)
        y_fit = m * x + b

        # Calculate R^2
        ss_res = sum((y - y_fit) ** 2)
        ss_tot = sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return x, y, m, b, y_fit, r_squared


    def scatter_compare(self):
        x = self.lw_data['bt_h'].flatten()
        y = self.ids_data['bt_h'].flatten()
        x_h_fore, y_h_fore, m_h_fore, b_h_fore, y_fit_h_fore, r_squared = self.scatter_stats(x, y)

        # x = self.lw_data['bt_h_aft'].flatten()
        # y = self.ids_data['bt_h_aft'].flatten()
        # x_h_aft, y_h_aft, m_h_aft, b_h_aft, y_fit_h_aft, r_squared = self.scatter_stats(x, y)


        fig, axs = plt.subplots()
        axs.scatter(x_h_fore, y_h_fore)
        axs.plot(x_h_fore, y_fit_h_fore, color='red')
        axs.legend(title=f"$R^2 = {r_squared:.3f}$")
        axs.set_title('bt_h_fore')
        axs.set_xlabel('LW BT [K]')
        axs.set_ylabel('IDS BT [K]')

        # axs[1].scatter(x_h_aft, y_h_aft)
        # axs[1].plot(x_h_aft, y_fit_h_aft, color='red')
        # axs[1].legend(title=f"$R^2 = {r_squared:.3f}$")
        # axs[1].set_title('bt_h_aft')
        # axs[1].set_xlabel('LW BT [K]')
        # axs[1].set_ylabel('IDS BT [K]')

        plt.show()

def run_python_subprocess(config_path):
    """
    Runs a Python script as a subprocess, displaying output in real-time.

    Parameters:
    ----------
    script_path : str
        Path to the Python script to be executed.
    *args : str
        Additional arguments to pass to the script.

    Returns:
    ----------
    int
        Exit code of the subprocess.
    """
    try:
        # Construct the command
        # command = [sys.executable, script_path] + list(args)
        command = [
            "python",
            "-m",
            "cimr_rgb",
            str(config_path),
        ]  # Adjust command if needed

        # Run the subprocess and allow real-time output
        result = sbps.run(
            command,
            stdout=sys.stdout,  # Redirect stdout to the parent process's stdout
            stderr=sys.stderr,  # Redirect stderr to the parent process's stderr
        )

        # Return the exit code
        return result.returncode
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        return -1

if __name__ == "__main__":
    # IDS
    config_path = grasp_io.find_repo_root().joinpath(
        "/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/MS3_verification_tests/T_17/T_17_IDS.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    # # LW
    config_path = grasp_io.find_repo_root().joinpath(
        "/home/beywood/ST/CIMR_RGB/CIMR-RGB/tests/MS3_verification_tests/T_17/T_17_LW.xml"
    )
    exit_code = run_python_subprocess(config_path=config_path)
    print(f"Subprocess exited with code: {exit_code}")

    CIMR_comparison(ids_data_path, lw_data_path).map_compare()
    CIMR_comparison(ids_data_path, lw_data_path).scatter_compare()

# data = CIMR_comparison(ids_data_path, lw_data_path)
# CIMR_comparison(ids_data_path, lw_data_path).map_compare()
# CIMR_comparison(ids_data_path, lw_data_path).scatter_compare()