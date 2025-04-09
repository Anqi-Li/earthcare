# %%
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%


def read_msi(full_path, band=5):

    # MSI L1 product definition document
    # Table 4-1: Spectral bands of MSI
    # Band # Band Name # Spectral range (µm)
    # 1 VIS 0.660-0.680
    # 2 VNIR 0.855-0.875
    # 3 SWIR1 1.625-1.675
    # 4 SWIR2 2.160-2.260
    # 5 TIR1 8.35-9.25
    # 6 TIR2 10.35-11.25
    # 7 TIR3 11.55-12.45

    # Open the HDF5 file and read a specific group into xarray
    xds = xr.open_dataset(full_path, group="ScienceData", chunks="auto")
    xds = xds.set_coords(["latitude", "longitude", "time"])
    xds = xds.isel(
        band=band,  # Select a specific band
        across_track=slice(12, -15),  # Remove the edges of the image
    )
    xds = xds.set_xindex(["time"])

    if band in [4, 5, 6, [4,5,6], [4,5], [5,6], [4,6]]:
        # TIR bands' unit are in Kelvin (for easy visualization)
        xds["pixel_values"].attrs = {
            "long_name": "Brightness Temperature",
            "units": "K",
        }
    elif band in [1, 2, 3]:
        xds["pixel_values"].attrs = {
            "long_name": "Radiance",
            "units": "Wm-2sr-1um-1",
        }
    else:
        print("Selected band is not available. (Valid value 0-6)")

    return xds


# %%
if __name__ == "__main__":

    file_name = "ECA_EXAF_MSI_NOM_1B_20240811T154801Z_20250218T170133Z_01162E"
    file_path = "/data/s6/L1/EarthCare/L1/MSI_NOM_1B/2024/08/11"

    full_path = os.path.join(
        file_path,
        file_name,
        file_name + ".h5",
    )
    # %%
    xds = read_msi(full_path, band=5)
    # Select a subset of the data (optional, for visualization clarity)
    subset = xds.isel(along_track=slice(None, 200))

    # Create a scatter plot with pixel_values as the hue
    plt.figure(figsize=(10, 6))
    scatter = subset.plot.scatter(
        x="longitude",
        y="latitude",
        hue="pixel_values",
        s=3,  # Adjust point size
        # alpha=0.1,  # Adjust transparency
        linewidths=0,  # Remove outlines around the scatter points
        cmap="viridis",  # Choose a colormap
    )
    plt.title(
        """
        Scatter Plot of Pixel Values in MSI Band 5 (TIR2 10.35-11.25 µm)
        {} to {}
        """.format(
            subset["time"][0].values,
            subset["time"][-1].values,
        )
    )

    plt.grid(True)
    plt.show()

    # %%
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Select a subset of the data (optional, for visualization clarity)
    subset = xds.isel(along_track=slice(200, None))

    # Get the latitude and longitude range from the subset
    min_lon, max_lon = subset["longitude"].min().item(), subset["longitude"].max().item()
    min_lat, max_lat = subset["latitude"].min().item(), subset["latitude"].max().item()

    # Create a figure with a Cartopy map projection
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())  # Use PlateCarree for latitude/longitude data

    # Set the extent of the map to the latitude and longitude range
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Add coastlines and other geographic features
    ax.coastlines(resolution="110m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    # Add gridlines
    gridlines = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle="--", color="gray")
    gridlines.top_labels = False  # Disable top labels
    gridlines.right_labels = False  # Disable right labels
    gridlines.xlabel_style = {"size": 10, "color": "black"}
    gridlines.ylabel_style = {"size": 10, "color": "black"}

    # Add labels and title
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Coastlines with Gridlines and Axes Labels")

    # Show the plot
    plt.show()

    # %%
