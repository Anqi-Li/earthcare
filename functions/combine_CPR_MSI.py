# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from functions.search_orbit_files import (
    get_orbit_files,
    get_all_orbit_numbers_per_instrument,
)


def read_xmet(
    orbit_number: str,
    base_path: str = "/data/s6/L1/EarthCare/Meteo_Supporting_Files/AUX_MET_1D/",
    aligned: bool = False,
    set_coords: bool = True,
    set_xindex: bool = True,
) -> xr.Dataset:
    """
    Read the XMET file corresponding to the given orbit number.
    """
    group = "ScienceData"
    if aligned:
        base_path = base_path.replace("AUX_MET_1D", "AUX_MET_1D_aligned_CPR")
        group = None

    file_paths = get_orbit_files(orbit_numbers=orbit_number, base_path=base_path)
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No XMET file found for orbit number {orbit_number}")

    elif len(file_paths) > 1:
        raise FileExistsError(
            f"Multiple XMET files found for orbit number {orbit_number}"
        )

    elif len(file_paths) == 1:
        ds = xr.open_dataset(
            file_paths[0],
            group=group,
            chunks="auto",
        )

        if set_coords:
            ds = ds.set_coords(["latitude", "longitude", "geometrical_height"])
        if set_xindex:
            ds = ds.set_xindex(["latitude", "longitude"])

        return ds


# %%
def read_cpr(orbit_number: str) -> xr.Dataset:
    """
    Read the CPR file corresponding to the given orbit number.
    """
    # Define the file path
    full_paths = get_orbit_files(orbit_numbers=orbit_number, inst="CPR")
    if len(full_paths) == 0:
        raise FileNotFoundError(f"No CPR file found for orbit number {orbit_number}")
    elif len(full_paths) > 1:
        raise FileExistsError(
            f"Multiple CPR files found for orbit number {orbit_number}"
        )
    elif len(full_paths) == 1:
        full_path = full_paths[0]
        # Open the HDF5 file and read a specific group into xarray
        data_group = xr.open_dataset(full_path, group="ScienceData/Data", chunks="auto")
        data_group = data_group.rename({"phony_dim_10": "nray", "phony_dim_11": "nbin"})

        geo_group = xr.open_dataset(full_path, group="ScienceData/Geo", chunks="auto")
        geo_group = geo_group.rename({"phony_dim_14": "nray", "phony_dim_15": "nbin"})

        xds = xr.merge([data_group, geo_group])
        xds = xds.set_coords(["latitude", "longitude", "profileTime", "binHeight"])
        xds = xds.set_xindex(["profileTime"])
        return xds


def read_msi(orbit_number, band=[4, 5, 6]):
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
    full_paths = get_orbit_files(orbit_numbers=orbit_number, inst="MSI")
    if len(full_paths) == 0:
        raise FileNotFoundError(f"No MSI file found for orbit number {orbit_number}")
    elif len(full_paths) > 1:
        raise FileExistsError(
            f"Multiple MSI files found for orbit number {orbit_number}"
        )
    elif len(full_paths) == 1:
        full_path = full_paths[0]
        # Open the HDF5 file and read a specific group into xarray
        xds = xr.open_dataset(full_path, group="ScienceData", chunks="auto")
        xds = xds.set_coords(["latitude", "longitude", "time"])
        xds = xds.isel(
            band=band,  # Select a specific band
            across_track=slice(12, -15),  # Remove the edges of the image
        )
        xds = xds.set_xindex(["time"])

        if band in [4, 5, 6, [4, 5, 6], [4, 5], [5, 6], [4, 6]]:
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
def merge_colocated(xds_cpr, xds_msi):
    """
    Merge the CPR and MSI datasets based on their geolocation.
    """
    # Find the closest MSI pixel (across_track) to match CPR profile
    pixel_number = 266
    xds_msi_selected = xds_msi.isel(across_track=pixel_number).sel(
        time=xds_cpr.profileTime
        + np.timedelta64(
            640, "ms"
        ),  # shift the lat/lon match in samppling time manually
        method="nearest",
    )
    # Merge the two datasets
    xds = xr.merge(
        [xds_cpr, xds_msi_selected],
        compat="override",  # pick the cpr values when there is a conflict variable(lat/lon)
    )
    return xds.drop_indexes("profileTime")


def get_cpr_msi_from_orbits(
    orbit_numbers: list | str,
    msi_band: list[int] | int = [4, 5, 6],
    get_xmet: bool = False,
) -> xr.Dataset:
    """
    Combine CPR and MSI data from the given orbit number list.
    """
    if isinstance(orbit_numbers, str):
        orbit_numbers = [orbit_numbers]

    # Initialize empty lists to store the datasets
    xds_list = []
    ds_xmet_list = []
    # Iterate through the orbit numbers
    for orbit_number in orbit_numbers:
        xds_cpr = read_cpr(orbit_number=orbit_number)
        xds_msi = read_msi(orbit_number=orbit_number, band=msi_band)
        xds = merge_colocated(xds_cpr, xds_msi).set_xindex(["latitude", "longitude"])
        # Append the datasets to the lists
        xds_list.append(xds)

        if get_xmet:
            # Get the XMET dataset for the current orbit number
            ds_xmet = read_xmet(orbit_number, aligned=True)
            # ds_xmet = align_xmet_horizontal_grid(ds_xmet, xds)
            ds_xmet_list.append(ds_xmet)

    xds_combined = xr.concat(xds_list, dim="nray")
    xds_combined["dBZ"] = xds_combined["radarReflectivityFactor"].pipe(
        lambda x: 10 * np.log10(x)
    )  # Convert to dBZ
    xds_combined["dBZ"].attrs = {"long_name": "dBZ", "units": "dBZ"}

    if get_xmet:
        ds_xmet_combined = xr.concat(ds_xmet_list, dim="horizontal_grid")
        return xds_combined, ds_xmet_combined
    else:
        return xds_combined


# %%
def interpolate1d_height_grid(
    height_sample: np.ndarray,
    variable_sample: np.ndarray,
    height_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate the variable_sample to the height_grid using linear interpolation.
    """
    # Remove NaN values from the sample data
    mask = ~np.isnan(variable_sample)
    variable_sample = variable_sample[mask]
    height_sample = height_sample[mask]

    # if sample data is empty, return NaN
    if len(variable_sample) == 0:
        return np.full_like(height_grid, np.nan)

    else:
        # Perform linear interpolation
        interpolated_data = griddata(
            height_sample,
            variable_sample,
            height_grid,
            method="linear",
            fill_value=np.nan,
        )
        return interpolated_data


def xr_vectorized_height_interpolation(
    ds: xr.Dataset,
    height_name: str,
    variable_name: str,
    height_grid: np.ndarray,
    height_dim: str,
    new_height_dim: str = "height_grid",
) -> xr.Dataset:
    """
    Interpolate the data in the dataset ds to the height_grid using linear interpolation.
    """
    da_interpolated = xr.apply_ufunc(
        interpolate1d_height_grid,
        ds[height_name],
        ds[variable_name],
        height_grid,
        exclude_dims=set((height_dim,)),
        input_core_dims=[[height_dim], [height_dim], [new_height_dim]],
        output_core_dims=[[new_height_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds[variable_name].dtype],
    )
    da_interpolated[new_height_dim] = height_grid
    da_interpolated.attrs = ds[variable_name].attrs
    return da_interpolated


# %% align geolocation of AUX_MET_L1D dataset with CPR profiles
def align_xmet_horizontal_grid(ds_xmet, xds):
    """
    find the closest horizontal grid point to the xds.nray (latitude, longitude)'
    """
    dist = cdist(
        np.array([*ds_xmet.horizontal_grid.values]),
        np.array([*xds.nray.values]),
        "chebyshev",
    )
    jminflat = dist.argmin(axis=0)
    return ds_xmet.isel(horizontal_grid=jminflat)


# %%
def package_ml_xy(
    xds: xr.Dataset,
    ds_xmet: xr.Dataset,
    height_grid: np.ndarray = np.arange(1e3, 15e3, 100),
    low_dBZ_replacement: float = -50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare the input data for machine learning.
    """
    # Interpolate the data to a uniform height grid
    da_dBZ_height = xr_vectorized_height_interpolation(
        ds=xds,
        height_name="binHeight",
        variable_name="dBZ",
        height_grid=height_grid,
        height_dim="nbin",
    )

    da_T_height = xr_vectorized_height_interpolation(
        ds=ds_xmet,
        height_name="geometrical_height",
        variable_name="temperature",
        height_grid=height_grid,
        height_dim="height",
    )

    # Define the features and target variable
    mask_clearsky = (da_dBZ_height < -25).all(dim="height_grid").compute()
    da_dBZ_height = da_dBZ_height.where(
        np.logical_and(~da_dBZ_height.pipe(np.isinf), da_dBZ_height > -25),
        low_dBZ_replacement,
    )
    X = np.stack([da_dBZ_height, da_T_height], axis=2)
    X = X[~mask_clearsky]  # remove profiles that are clearsky

    y = xds["pixel_values"].T
    y = y[~mask_clearsky]

    nsamples, nx, ny = X.shape
    X_2d = X.reshape((nsamples, nx * ny))

    return X_2d, y


# %%
if __name__ == "__main__":
    # Get all matched orbit numbers
    # Define the base directory and orbit number
    base_path = "/data/s6/L1/EarthCare/L1/"
    orbit_numbers = get_all_orbit_numbers_per_instrument("CPR")
    matched_orbits = []
    matched_file_pairs = []
    for orbit_number in orbit_numbers:
        orbit_files = get_orbit_files(orbit_number, base_path)
        if len(orbit_files) >= 2:
            print(orbit_number)

            matched_orbits.append(orbit_number)
            matched_file_pairs.append(orbit_files)

    # %%
    # orbit_number = matched_orbits[0]
    # orbit_files = matched_file_pairs[0]
    orbit_number = "03613C"  # "01723E"  # "03613C"
    xds_cpr = read_cpr(orbit_number=orbit_number)
    xds_msi = read_msi(orbit_number=orbit_number, band=[4, 5, 6])
    # %%
    xds = merge_colocated(xds_cpr, xds_msi)
    xds["dbZ"] = xds["radarReflectivityFactor"].pipe(np.log10) * 10  # Convert to dBZ
    # %%
    variables = ["dbZ", "pixel_values"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    scatter = xds[variables].plot.scatter(
        ax=axes[0],
        x="nray",
        y="binHeight",
        hue="dbZ",
        cmap="viridis",
        vmin=-35,
        vmax=25,
        linewidths=0,
        s=0.2,
        add_colorbar=False,  # Disable the default colorbar
    )
    xds[variables].pixel_values.plot(ax=axes[1])

    # Add a colorbar at the top
    cbar = fig.colorbar(scatter, ax=axes[0], orientation="horizontal", pad=0.2)
    cbar.set_label("dBZ")  # Add a label to the colorbar

    # Plot the second variable
    xds[variables].pixel_values.plot(ax=axes[1])
    xds_msi.pixel_values.sel(time=xds.profileTime, method="nearest").plot(
        ax=axes[2],
        x="nray",
        y="across_track",
        cmap="viridis",
        add_colorbar=False,
    )
    plt.hlines(266, 0, 9000, color="red", linestyles="--")
    # Show the plot
    plt.show()

    # %% plot sampling location of CPR and MSI (horizontal)
    plt.plot(
        xds_msi.latitude,
        xds_msi.longitude,
        ls="",
        marker=".",
        markersize=0.1,
        color="C1",
    )
    plt.plot(
        xds_cpr.latitude,
        xds_cpr.longitude,
        ls="",
        marker=".",
        markersize=0.1,
        color="C0",
    )
    plt.show()

    # %%
    slice_time = slice(None, None, 200)
    xds_msi_selected = xds_msi.isel(across_track=slice(266, 268)).sel(
        time=xds_cpr.isel(nray=slice_time).profileTime
        + np.timedelta64(
            640, "ms"
        ),  # shift the lat/lon match in samppling time manually
        method="nearest",
    )
    plt.plot(
        xds_msi_selected.latitude,
        xds_msi_selected.longitude,
        ls="-",
        marker="o",
        color="C1",
    )
    plt.plot(
        xds_cpr.isel(nray=slice_time).latitude,
        xds_cpr.isel(nray=slice_time).longitude,
        ls="-",
        marker=".",
        color="C0",
    )
    plt.show()

    # %%
    from functions.combine_CPR_MSI import get_cpr_msi_from_orbits

    orbit_number = "03842C"
    xds, ds_xmet = get_cpr_msi_from_orbits(orbit_number, get_xmet=True)
# %%
