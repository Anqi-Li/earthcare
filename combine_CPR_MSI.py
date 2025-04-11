# %%
import os
import xarray as xr
from MSI import read_msi
from CPR import read_cpr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from datetime import date, datetime, timedelta


# %%
def get_xmet_ds(
    orbit_number: str,
    base_path: str = "/data/s6/L1/EarthCare/Meteo_Supporting_Files/AUX_MET_1D/",
    set_coords: bool = True,
    set_xindex: bool = True,
) -> xr.Dataset:
    """
    Get the XMET file corresponding to the given orbit number.
    """
    for root, _, files in os.walk(base_path):
        for file in files:
            if ".h5" in file and orbit_number in file:
                ds = xr.open_dataset(
                    os.path.join(root, file),
                    group="ScienceData",
                    chunks="auto",
                )
                if set_coords:
                    ds = ds.set_coords(["latitude", "longitude", "geometrical_height"])
                if set_xindex:
                    ds = ds.set_xindex(["latitude", "longitude"])

                return ds

    # If no matching file is found, raise an error
    raise FileNotFoundError(f"No XMET file found for orbit number {orbit_number}")


# %%
def get_aligned_xmet(
    orbit_numbers: list[str],
    xds: xr.Dataset,
) -> xr.Dataset:
    """
    Get the XMET dataset for a list of orbit numbers and align it with the CPR dataset.
    Parameters:
        orbit_numbers: list of str
            List of orbit numbers.
        xds: xarray.Dataset
            The CPR dataset.
    """
    ds_list = []
    for orbit_number in orbit_numbers:
        ds = get_xmet_ds(orbit_number)
        ds_list.append(ds)
    ds_combined = xr.concat(ds_list, dim="horizontal_grid")
    ds_combined_aligned = align_xmet_horizontal_grid(ds_combined, xds)
    return ds_combined_aligned


# %%
def get_all_orbit_numbers_per_instrument(
    inst: str,
    date: str = "",  # format: "YYYY/MM/DD"
    base_path: str = "/data/s6/L1/EarthCare/",
    get_full_path: bool = False,
) -> list:
    """
    Get all orbit numbers for the specified instrument (CPR, MSI, or XMET).
    """
    # Define the base directory depending on the instrument
    if inst == "CPR":
        base_path = os.path.join(base_path, "L1/CPR_NOM_1B", date)
    elif inst == "MSI":
        base_path = os.path.join(base_path, "L1/MSI_NOM_1B", date)
    elif inst == "XMET":
        base_path = os.path.join(base_path, "Meteo_Supporting_Files/AUX_MET_1D", date)

    # Walk through the directory and collect orbit numbers
    orbit_numbers = []
    paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if ".h5" in file:
                orbit_numbers.append(file[-9:-3])
                paths.append(os.path.join(root, file))

    if get_full_path:
        return paths
    else:
        return orbit_numbers


# %% search common orbit numbers
def get_common_orbits(
    instruments: list[str],
    date_list: list[str] = "",  # format: ["YYYY/MM/DD"]
) -> list:
    """
    Get the common orbit numbers for the given instruments and date (format: "YYYY/MM/DD").
    If the date is not given, return all available common orbit numbers.
    """
    common_orbits = []
    for date in date_list:

        # Get all orbit numbers for each instrument
        orbit_numbers = [
            get_all_orbit_numbers_per_instrument(inst, date=date)
            for inst in instruments
        ]

        # Find elements that exist in all lists
        common_orbits_per_date = set(orbit_numbers[0]).intersection(*orbit_numbers[1:])

        common_orbits.extend(common_orbits_per_date)
    return list(common_orbits)


# %% search common orbit numbers given a date range
def get_date_list_from_range(
    date_range: list[str],  # format: ["YYYY/MM/DD", "YYYY/MM/DD"]
) -> list:
    """
    Get the common orbit numbers for the given instruments and date range.
    """
    # Convert date strings to datetime objects
    start_date = datetime.strptime(date_range[0], "%Y/%m/%d")
    end_date = datetime.strptime(date_range[1], "%Y/%m/%d")

    # Generate a list of dates within the range
    date_list = [
        (start_date + timedelta(days=i)).strftime("%Y/%m/%d")
        for i in range((end_date - start_date).days + 1)
    ]

    # Get common orbits for each date in the range
    return date_list


# %%
def get_orbit_files(
    orbit_numbers: str | list[str],
    inst: str = None,
    date: str = "",  # format: "YYYY/MM/DD"
    base_path: str = "/data/s6/L1/EarthCare/",
) -> list:
    """
    Get the matching files for the given orbit number.

    Parameters:
    base_path (str): The directory to search for files.
    orbit_numbers (list or string): A list of orbit numbers to match.

    Returns:
    list: A list of file paths that match the orbit numbers.

    """
    if inst is not None:
        if inst == "CPR":
            base_path = os.path.join(base_path, "L1/CPR_NOM_1B", date)
        elif inst == "MSI":
            base_path = os.path.join(base_path, "L1/MSI_NOM_1B", date)
        elif inst == "XMET":
            base_path = os.path.join(
                base_path, "Meteo_Supporting_Files/AUX_MET_1D", date
            )

    orbit_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if isinstance(orbit_numbers, list):
                if any(orbit_number + ".h5" in file for orbit_number in orbit_numbers):
                    orbit_files.append(os.path.join(root, file))
            elif isinstance(orbit_numbers, str):
                if orbit_numbers + ".h5" in file:
                    orbit_files.append(os.path.join(root, file))
    return orbit_files


# %%
def read_cpr_msi(
    orbit_files: list[str],
    msi_band: int | list[int] = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Reads CPR and MSI data from matching files.

    Parameters:
        orbit_files: list of str
            List of file paths for matching CPR and MSI orbit files.
        msi_band: optional
            Band information to pass to the read_msi function. Default is None.

    Returns:
        xds_cpr: xarray.Dataset
            The CPR dataset.
        xds_msi: xarray.Dataset
            The MSI dataset.
    """
    if len(orbit_files) == 2:
        if "MSI" in orbit_files[0] and "CPR" in orbit_files[1]:
            xds_msi = (
                read_msi(orbit_files[0], msi_band)
                if msi_band is not None
                else read_msi(orbit_files[0])
            )
            xds_cpr = read_cpr(orbit_files[1])
        elif "MSI" in orbit_files[1] and "CPR" in orbit_files[0]:
            xds_msi = (
                read_msi(orbit_files[1], msi_band)
                if msi_band is not None
                else read_msi(orbit_files[0])
            )
            xds_cpr = read_cpr(orbit_files[0])
        else:
            raise ValueError(
                "The file pairs do not match the expected format (one CPR one MSI)."
            )
    else:
        raise ValueError("Please inspect the orbit files")
    return xds_cpr, xds_msi


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


def combine_cpr_msi_from_orbits(
    orbit_numbers: list,
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
        # Search for matching file pairs
        matching_file_pairs = get_orbit_files(orbit_number)
        if len(matching_file_pairs) == 3:
            matching_file_pairs = [
                file for file in matching_file_pairs if "AUX_MET_1D" not in file
            ]

        if len(matching_file_pairs) != 2:
            raise ValueError(
                f"Expected 2 files for orbit number {orbit_number}, but found {len(matching_file_pairs)}."
            )
        xds_cpr, xds_msi = read_cpr_msi(matching_file_pairs, msi_band=[4, 5, 6])
        xds = merge_colocated(xds_cpr, xds_msi).set_xindex(["latitude", "longitude"])
        # Append the datasets to the lists
        xds_list.append(xds)

        if get_xmet:
            # Get the XMET dataset for the current orbit number
            ds_xmet = get_xmet_ds(orbit_number)
            ds_xmet = align_xmet_horizontal_grid(ds_xmet, xds)
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
    orbit_files = get_orbit_files(orbit_number)
    xds_cpr, xds_msi = read_cpr_msi(orbit_files)

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
