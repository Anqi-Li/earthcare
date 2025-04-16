# %%
from combine_CPR_MSI import (
    get_aligned_xmet,
    get_common_orbits,
    get_orbit_files,
    get_all_orbit_numbers_per_instrument,
)
from CPR import read_cpr
import os
import xarray as xr

# %%
common_orbits = get_common_orbits(["CPR", "MSI", "XMET"])
common_orbit_files = get_orbit_files(common_orbits, "XMET")
base_dir = "/data/s6/L1/EarthCare/Meteo_Supporting_Files/AUX_MET_1D_aligned_CPR"
variables_to_save = ["temperature"]

# %%
for i in range(len(common_orbit_files)):
    _, _, _, _, _, _, _, y, m, d, filename, _ = common_orbit_files[i].split("/")
    save_dir = os.path.join(base_dir, y, m, d, filename)
    save_file_path = os.path.join(save_dir, f"{filename}.nc")
    orbit_number = filename[-6:]
    print(orbit_number)

    # Check if the file already exists
    if not os.path.exists(save_file_path):
        # Get the aligned XMET data
        xmet = get_aligned_xmet(
            orbit_number,
            read_cpr(get_orbit_files(orbit_number, "CPR")[0]).set_xindex(
                ["latitude", "longitude"]
            ),
        )
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the aligned XMET data to a NetCDF file
        xmet[variables_to_save].reset_index("horizontal_grid").to_netcdf(
            path=save_file_path,
            mode="w",
        )
        print(f"File saved to {save_file_path}")
    else:
        print("File already exists, skip.")
# %%
if __name__ == "__main__":

    #%%
    base_dir = "/data/s6/L1/EarthCare/Meteo_Supporting_Files/AUX_MET_1D_aligned_CPR"
    orbits = get_all_orbit_numbers_per_instrument(base_path=base_dir)
    len(orbits)
# %%
