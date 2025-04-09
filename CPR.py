# %%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# %%
def read_cpr(full_path):
    # Open the HDF5 file and read a specific group into xarray
    data_group = xr.open_dataset(full_path, group="ScienceData/Data", chunks="auto")
    data_group = data_group.rename({"phony_dim_10": "nray", "phony_dim_11": "nbin"})

    geo_group = xr.open_dataset(full_path, group="ScienceData/Geo", chunks="auto")
    geo_group = geo_group.rename({"phony_dim_14": "nray", "phony_dim_15": "nbin"})

    xds = xr.merge([data_group, geo_group])
    xds = xds.set_coords(["latitude", "longitude", "profileTime", "binHeight"])
    xds = xds.set_xindex(["profileTime"])
    return xds


# %%
if __name__ == "__main__":

    file_name = "ECA_JXCA_CPR_NOM_1B_20240811T150145Z_20250203T110502Z_01162A"
    file_path = "/data/s6/L1/EarthCare/L1/CPR_NOM_1B/2024/08/11"

    # file_name = "ECA_JXCA_CPR_NOM_1B_20240803T044914Z_20250127T095411Z_01031A"
    # file_path = "/data/s6/L1/EarthCare/L1/CPR_NOM_1B/2024/08/03"

    full_path = os.path.join(
        file_path,
        file_name,
        file_name + ".h5",
    )
    # %%
    xds = read_cpr(full_path)

    # Select a subset of the data (optional, for visualization clarity)
    xds["dbZ"] = xds["radarReflectivityFactor"].pipe(np.log10) * 10  # Convert to dBZ
    subset = xds.isel(nray=slice(400, 600))

    # Create a scatter plot with pixel_values as the hue
    plt.figure(figsize=(10, 6))
    subset.plot.scatter(
        x="latitude",
        y="binHeight",
        hue="dbZ",
        linewidths=0,
        s=5,
        cmap="viridis",  # Choose a colormap
        vmin=-35,
        vmax=25,
    )
    # Add a secondary x-axis for longitude
    ax = plt.gca()  # Get the current axis

    # Map latitude to longitude for the secondary x-axis
    latitude = subset["latitude"].values
    longitude = subset["longitude"].values

    # Create a secondary x-axis at the bottom
    secax = ax.secondary_xaxis("bottom")
    secax.set_xlabel("longitude [{}]".format(subset["longitude"].attrs["units"]))  # Label for the secondary x-axis

    # Interpolate longitude values for the secondary x-axis ticks
    secax.set_xticks(ax.get_xticks())  # Match the primary x-axis ticks
    secax.set_xticklabels(np.interp(ax.get_xticks(), latitude, longitude).round(2))  # Interpolate longitude values for the ticks

    # Set the primary x-axis label
    # ax.set_xlabel("Latitude")

    # Adjust the position of the secondary x-axis to avoid overlap
    secax.spines["bottom"].set_position(("outward", 40))  # Move the secondary x-axis outward

    # Add a title
    plt.title(
        """
    Scatter Plot of CPR Signal (dBZ) Along Track and Bin Height
    {} to {}
    """.format(
            subset["profileTime"][0].values,
            subset["profileTime"][-1].values,
        )
    )
    plt.show()

# %%
