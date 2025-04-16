# %%
from list_groups_h5 import list_groups
import zipfile
import os
import xarray as xr
import numpy as np
from functions.combine_CPR_MSI import (
    combine_cpr_msi_from_orbits,
    xr_vectorized_height_interpolation,
    get_xmet_ds,
)
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from datetime import datetime
import joblib


# %%
orbit_number = "03890F"

# %% select the closest horizontal grid point to the cpr & msi data
xds = combine_cpr_msi_from_orbits(orbit_numbers=[orbit_number], get_xmet=False)
ds = get_xmet_ds(orbit_number=orbit_number)

xds = xds.reset_index(['latitude', 'longitude']).reset_coords(['latitude', 'longitude'])
ds = ds.reset_index(['latitude', 'longitude']).reset_coords(['latitude', 'longitude'])

# %%
# find the closest horizontal grid point to the xds.nray (latitude, longitude)
dist = cdist(
    np.array([*ds.horizontal_grid.values]),
    np.array([*xds.nray.values]),
    "chebyshev",
)
jminflat = dist.argmin(axis=0)
ds_xmet = ds.isel(horizontal_grid=jminflat)


# %%
# function to find the shortest distance to a hosrizontal grid index of an array from a list of coordinates
def find_closest_horizontal_grid_index(grid_lat_lon, query_lat_lon):
    """
    Find the closest horizontal grid index to a given latitude and longitude.
    Parameters
    ----------
    grid_lat_lon : array-like
        The latitude and longitude of the horizontal grid points.
    query_lat_lon : array-like
        The latitude and longitude of the query points.
    Returns
    -------
    array-like
        The indices of the closest horizontal grid points.
    """
    dist = cdist(
        grid_lat_lon,
        query_lat_lon,
        "chebyshev",
    )
    jminflat = dist.argmin(axis=0)
    return jminflat

start = datetime.now()
xr.apply_ufunc(
    find_closest_horizontal_grid_index,
    # np.array([*ds["horizontal_grid"].values]),
    # np.array([*xds["nray"].values]),
    ds[['latitude', 'longitude']].to_array().values.T,
    xds[['latitude', 'longitude']].to_array().values.T,
    exclude_dims=set(('horizontal_grid',)),
    input_core_dims=[["horizontal_grid"], ["nray"]],
    output_core_dims=[["nray"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[np.int64],
)
print(datetime.now() - start)

#%%
start = datetime.now()
find_closest_horizontal_grid_index(
    # np.array([*ds["horizontal_grid"].values]),
    # np.array([*xds["nray"].values]),
    ds[['latitude', 'longitude']].to_array().values.T,
    xds[['latitude', 'longitude']].to_array().values.T,
)
print(datetime.now() - start)

#%%
start = datetime.now()
ds_T_unstack = ds.temperature.unstack(dim="horizontal_grid")
print(datetime.now() - start)
# %%
