# %%
from combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    package_ml_xy,
)
from search_orbit_files import (
    get_common_orbits,
)
import numpy as np
import os
import datetime

# %% select orbit numbers
common_orbits_all = get_common_orbits(
    ["CPR", "MSI", "XMET_aligned"],
    date_range=["2025/01/01", "2025/05/01"],
)


# %%
def save_training_dataset(orbit_number):
    """Save training dataset for a given orbit number.
    Parameters:
    orbit_number (str): The orbit number for which to save the training dataset.
    """

    filename = "./data/training_data/training_data_{}.nc".format(orbit_number)
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File {orbit_number} already exists. Skipping...")
        return

    start = datetime.datetime.now()
    # Load the data from the specified orbit number
    # This function retrieves the data for the specified orbit number,
    # including the MSI bands and XMET data, and filters out ground refelctivity.

    xds, ds_xmet = get_cpr_msi_from_orbits(
        orbit_numbers=orbit_number,
        msi_band=[4, 5, 6],
        get_xmet=True,
        filter_ground=True,
        add_dBZ=True,
    )

    height_grid = np.arange(1e3, 15e3, 100)

    # Package the data into training features (X) and labels (y)
    # It applies a lowest dBZ threshold to filter out weak reflectivity values
    # and interpolate dBz and temperature in an uniformed height grid.

    X_train, y_train = package_ml_xy(
        xds=xds,
        ds_xmet=ds_xmet,
        lowest_dBZ_threshold=-25,
        height_grid=height_grid,
    )

    # Check if the training data is empty
    if X_train.size == 0 or y_train.size == 0:
        print(f"No training data available for orbit {orbit_number}. Skipping...")
        return

    # Package the data into an xarray dataset and preserve the relevant coordinates.
    # This is necessary to ensure that the data can be reshaped correctly later
    ds = (
        (
            y_train.to_dataset()
            .assign(
                x=(("nray", "height_grid", "param"), X_train.reshape(-1, len(height_grid), 2)),
            )
            .assign_coords(
                param=["dBZ", "T"],
                height_grid=height_grid,
            )
        )
        .stack(features=("height_grid", "param"))
        .reset_index(("nray", "features"))
    )
    # Save the dataset to a NetCDF file
    ds.to_netcdf(filename)
    print("time spent:", datetime.datetime.now() - start)


# %%
for i in range(len(common_orbits_all)):
    # Iterate through each orbit number in the list and save the training dataset
    # for that orbit number.
    orbit_number = common_orbits_all[i]
    print(f"{i}/{len(common_orbits_all)}", orbit_number)
    save_training_dataset(orbit_number)

print("All training datasets saved successfully.")
