# Testing script for T.12
# Remapping of L1b SMAP data with IDS (inverse distance squared) algorithm on an EASE2 global grid
# The remmapped data are compatible with SMAP L1c data obtained by NASA, with an average relative difference of brightness temperature < 0.25%

import sys
import os
import pathlib as pb
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


from numpy import array, full, nan, nanmean, polyfit, isnan, isinf
import matplotlib

tkagg = matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.ion()

# sys.path.append('/home/beywood/ST/CIMR_RGB/CIMR-RGB/src/cimr_rgb')
from cimr_rgb.grid_generator import GRIDS
import cimr_grasp.grasp_io as grasp_io

repo_root = grasp_io.find_repo_root()
# Add RGB remapped netCDF here
rgb_data = grasp_io.find_repo_root().joinpath(
    "MS3_verification_tests/T_12/SMAP_L1C_IDS_36km_test.nc"
)  # ""
print(rgb_data)
# Add the NASA version you are comparing to here
nasa_data = ""

GRID = "EASE2_G36km"
PROJECTION = "G"


class SMAP_comparison:
    def __init__(self, rgb_data_path, nasa_data_path):
        self.rgb_data = self.get_netcdf_data(rgb_data_path)
        self.nasa_data = self.get_hdf5_data(nasa_data_path)

    @staticmethod
    def get_netcdf_data(path):
        import netCDF4 as nc

        gridded_vars = {}
        with nc.Dataset(path, "r") as f:
            data = f[f"{PROJECTION}"]
            measurement = data["Measurement"]
            l_band = measurement["L_BAND"]

            for bt in ["bt_h_fore", "bt_h_aft", "bt_v_fore", "bt_v_aft"]:
                if "fore" in bt:
                    row = array(l_band["cell_row_fore"][:])
                    col = array(l_band["cell_col_fore"][:])
                elif "aft" in bt:
                    row = array(l_band["cell_row_aft"][:])
                    col = array(l_band["cell_col_aft"][:])
                else:
                    row = ""
                    col = ""
                var = array(l_band[bt][:])
                grid = full((GRIDS[GRID]["n_rows"], GRIDS[GRID]["n_cols"]), nan)

                for count, sample in enumerate(var):
                    if sample == 9.969209968386869e36:
                        continue
                    grid[row[count], col[count]] = sample
                gridded_vars[bt] = grid
            return gridded_vars

    @staticmethod
    def get_hdf5_data(path):
        import h5py

        gridded_vars = {}
        with h5py.File(path, "r") as f:
            data = f["Global_Projection"]
            row = data["cell_row"][:]
            col = data["cell_column"][:]

            bts = {
                "bt_h_fore": data["cell_tb_h_fore"][:],
                "bt_h_aft": data["cell_tb_h_aft"][:],
                "bt_v_fore": data["cell_tb_v_fore"][:],
                "bt_v_aft": data["cell_tb_v_aft"][:],
            }

            for bt in bts:
                var = array(bts[bt])
                grid = full((GRIDS[GRID]["n_rows"], GRIDS[GRID]["n_cols"]), nan)

                for count, sample in enumerate(var):
                    grid[row[count], col[count]] = sample
                gridded_vars[bt] = grid
            return gridded_vars

    def map_compare(self):
        cmap = "viridis"
        # bt_h plt
        fig, axs = plt.subplots(2, 3, constrained_layout=True)
        im00 = axs[0, 0].imshow(self.rgb_data["bt_h_fore"][:, 550:], cmap=cmap)
        axs[0, 0].set_title("RGB Remap (bt_h_fore)")
        om01 = axs[0, 1].imshow(self.nasa_data["bt_h_fore"][:, 550:], cmap=cmap)
        axs[0, 1].set_title("NASA Remap (bt_h_fore)")
        bt_h_fore_diff = abs(self.rgb_data["bt_h_fore"] - self.nasa_data["bt_h_fore"])
        im02 = axs[0, 2].imshow(bt_h_fore_diff[:, 550:], cmap=cmap)
        axs[0, 2].set_title("Difference (bt_h_fore)")
        # aft
        im10 = axs[1, 0].imshow(self.rgb_data["bt_h_aft"][:, 550:], cmap=cmap)
        axs[1, 0].set_title("RGB Remap (bt_h_aft)")
        im11 = axs[1, 1].imshow(self.nasa_data["bt_h_aft"][:, 550:], cmap=cmap)
        axs[1, 1].set_title("NASA Remap (bt_h_aft)")
        bt_h_aft_diff = abs(self.rgb_data["bt_h_aft"] - self.nasa_data["bt_h_aft"])
        im12 = axs[1, 2].imshow(bt_h_aft_diff[:, 550:], cmap=cmap)
        axs[1, 2].set_title("Difference (bt_h_aft)")
        fig.colorbar(im02, ax=axs[0, 2])
        fig.colorbar(im12, ax=axs[1, 2])

        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_h_fore_diff)
        aft_mean_diff = nanmean(bt_h_aft_diff)
        print(f"Average relative difference for bt_h_fore: {fore_mean_diff}")
        print(f"Average relative difference for bt_h_aft: {aft_mean_diff}")

        # Calculate percentage Differences
        fore_percent_diff = (
            fore_mean_diff / nanmean(self.nasa_data["bt_h_fore"])
        ) * 100
        aft_percent_diff = (aft_mean_diff / nanmean(self.nasa_data["bt_h_aft"])) * 100
        print(f"Average percentage difference for bt_h_fore: {fore_percent_diff}")
        print(f"Average percentage difference for bt_h_aft: {aft_percent_diff}")

        # Add statistics to the plot
        # axs[0,2].text(50,50, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
        # axs[0,2].text(50, 50, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
        axs[0, 2].text(
            50,
            50,
            rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{RGB}}_i - \mathrm{{NASA}}_i |$",
            fontsize=14,
            color="black",
        )

        axs[0, 2].text(
            50,
            100,
            rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
            fontsize=14,
            color="black",
        )

        axs[1, 2].text(
            50,
            50,
            rf"$\mu_{{aft}} =  {aft_mean_diff:.2f} K, \ \text{{or}} \ {aft_percent_diff:.2f}\%$",
            fontsize=14,
            color="black",
        )
        plt.show()

        # bt_v plt
        fig, axs = plt.subplots(2, 3, constrained_layout=True)
        im00 = axs[0, 0].imshow(self.rgb_data["bt_v_fore"][:, 550:], cmap=cmap)
        axs[0, 0].set_title("RGB Remap (bt_v_fore)")
        om01 = axs[0, 1].imshow(self.nasa_data["bt_v_fore"][:, 550:], cmap=cmap)
        axs[0, 1].set_title("NASA Remap (bt_v_fore)")
        bt_v_fore_diff = abs(self.rgb_data["bt_v_fore"] - self.nasa_data["bt_v_fore"])
        im02 = axs[0, 2].imshow(bt_v_fore_diff[:, 550:], cmap=cmap)
        axs[0, 2].set_title("Difference (bt_v_fore)")
        # aft
        im10 = axs[1, 0].imshow(self.rgb_data["bt_v_aft"][:, 550:], cmap=cmap)
        axs[1, 0].set_title("RGB Remap (bt_v_aft)")
        im11 = axs[1, 1].imshow(self.nasa_data["bt_v_aft"][:, 550:], cmap=cmap)
        axs[1, 1].set_title("NASA Remap (bt_v_aft)")
        bt_v_aft_diff = abs(self.rgb_data["bt_v_aft"] - self.nasa_data["bt_v_aft"])
        im12 = axs[1, 2].imshow(bt_v_aft_diff[:, 550:], cmap=cmap)
        axs[1, 2].set_title("Difference (bt_v_aft)")
        fig.colorbar(im02, ax=axs[0, 2])
        fig.colorbar(im12, ax=axs[1, 2])

        # Add Statistics
        # Calculate the average relative difference
        fore_mean_diff = nanmean(bt_v_fore_diff)
        aft_mean_diff = nanmean(bt_h_aft_diff)
        print(f"Average relative difference for bt_v_fore: {fore_mean_diff}")
        print(f"Average relative difference for bt_v_aft: {aft_mean_diff}")

        # Calculate percentage Differences
        fore_percent_diff = (
            fore_mean_diff / nanmean(self.nasa_data["bt_v_fore"])
        ) * 100
        aft_percent_diff = (aft_mean_diff / nanmean(self.nasa_data["bt_v_aft"])) * 100
        print(f"Average percentage difference for bt_v_fore: {fore_percent_diff}")
        print(f"Average percentage difference for bt_v_aft: {aft_percent_diff}")

        # Add statistics to the plot
        # axs[0,2].text(50,50, f"mean(abs(bt_diff)) = {fore_mean_diff}K")
        # axs[0,2].text(50, 50, f"mean(abs(bt_diff)) = {aft_mean_diff}K")
        axs[0, 2].text(
            50,
            50,
            rf"$\mu = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} | \mathrm{{RGB}}_i - \mathrm{{NASA}}_i |$",
            fontsize=14,
            color="black",
        )

        axs[0, 2].text(
            50,
            100,
            rf"$\mu_{{fore}} =  {fore_mean_diff:.2f} K, \ \text{{or}} \ {fore_percent_diff:.2f}\%$",
            fontsize=14,
            color="black",
        )

        axs[1, 2].text(
            50,
            50,
            rf"$\mu_{{aft}} =  {aft_mean_diff:.2f} K, \ \text{{or}} \ {aft_percent_diff:.2f}\%$",
            fontsize=14,
            color="black",
        )
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
        x = self.rgb_data["bt_h_fore"].flatten()
        y = self.nasa_data["bt_h_fore"].flatten()
        x_h_fore, y_h_fore, m_h_fore, b_h_fore, y_fit_h_fore, r_squared = (
            self.scatter_stats(x, y)
        )

        x = self.rgb_data["bt_h_aft"].flatten()
        y = self.nasa_data["bt_h_aft"].flatten()
        x_h_aft, y_h_aft, m_h_aft, b_h_aft, y_fit_h_aft, r_squared = self.scatter_stats(
            x, y
        )

        x = self.rgb_data["bt_v_fore"].flatten()
        y = self.nasa_data["bt_v_fore"].flatten()
        x_v_fore, y_v_fore, m_v_fore, b_v_fore, y_fit_v_fore, r_squared = (
            self.scatter_stats(x, y)
        )

        x = self.rgb_data["bt_v_aft"].flatten()
        y = self.nasa_data["bt_v_aft"].flatten()

        x_v_aft, y_v_aft, m_v_aft, b_v_aft, y_fit_v_aft, r_squared = self.scatter_stats(
            x, y
        )

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].scatter(x_h_fore, y_h_fore)
        axs[0, 0].plot(x_h_fore, y_fit_h_fore, color="red")
        axs[0, 0].legend(title=f"$R^2 = {r_squared:.3f}$")
        axs[0, 0].set_title("bt_h_fore")
        axs[0, 0].set_xlabel("RGB BT [K]")
        axs[0, 0].set_ylabel("NASA BT [K]")

        axs[0, 1].scatter(x_h_aft, y_h_aft)
        axs[0, 1].plot(x_h_aft, y_fit_h_aft, color="red")
        axs[0, 1].legend(title=f"$R^2 = {r_squared:.3f}$")
        axs[0, 1].set_title("bt_h_aft")
        axs[0, 1].set_xlabel("RGB BT [K]")
        axs[0, 1].set_ylabel("NASA BT [K]")

        axs[1, 0].scatter(x_v_fore, y_v_fore)
        axs[1, 0].plot(x_v_fore, y_fit_v_fore, color="red")
        axs[1, 0].legend(title=f"$R^2 = {r_squared:.3f}$")
        axs[1, 0].set_title("bt_v_fore")
        axs[1, 0].set_xlabel("RGB BT [K]")
        axs[1, 0].set_ylabel("NASA BT [K]")

        axs[1, 1].scatter(x_v_aft, y_v_aft)
        axs[1, 1].plot(x_v_aft, y_fit_v_aft, color="red")
        axs[1, 1].legend(title=f"$R^2 = {r_squared:.3f}$")
        axs[1, 1].set_title("bt_v_aft")
        axs[1, 1].set_xlabel("RGB BT [K]")
        axs[1, 1].set_ylabel("NASA BT [K]")

        plt.show()


if __name__ == "__main__":
    SMAP_comparison(rgb_data, nasa_data).map_compare()
    SMAP_comparison(rgb_data, nasa_data).scatter_compare()
