#%%
from list_groups_h5 import list_groups
import zipfile
import os
import xarray as xr
import numpy as np
from combine_CPR_MSI import combine_cpr_msi_from_orbits, xr_vectorized_height_interpolation
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

#%%
filename = "ECA_EXAA_AUX_MET_1D_20250202T234643Z_20250203T002104Z_03890F"
orbit_number = filename[-6:]

#%%
path_to_zip_file = f'./data/{filename}.ZIP'
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall("./data")

# %%
list_groups(f"./data/{filename}.h5")
# %%
ds = xr.open_dataset(f"./data/{filename}.h5", group="ScienceData",)
ds = ds.set_coords(["latitude", "longitude", "geometrical_height"])
ds = ds.set_xindex(["latitude", "longitude"])

# %% select the closest horizontal grid point to the cpr & msi data
xds = combine_cpr_msi_from_orbits(orbit_numbers=[orbit_number])

# find the closest horizontal grid point to the xds.nray (latitude, longitude)
dist = cdist(np.array([*ds.horizontal_grid.values]), np.array([*xds.nray.values]), "chebyshev",)
jminflat = dist.argmin(axis=0)
ds_xmet = ds.isel(horizontal_grid=jminflat)

# %%
height_grid = np.arange(1e3,15e3, 100)
# xr_vectorized_height_interpolation(ds_xmet, "geometrical_height", "temperature", height_grid, "height", "height_grid",)
xr_vectorized_height_interpolation(ds=xds, height_name='binHeight', variable_name='dBZ', height_grid=height_grid, height_dim='nbin', new_height_dim='height_grid',)

