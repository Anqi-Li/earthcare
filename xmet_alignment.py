# %%
from functions.search_orbit_files import (
    get_common_orbits,
    get_orbit_files,
    get_all_orbit_numbers_per_instrument,
)
from functions.combine_CPR_MSI import read_xmet, align_xmet_horizontal_grid, read_cpr
import os
import xarray as xr

REWRITE = False  # Set to True to overwrite existing files
# %%
common_orbits = get_common_orbits(["CPR", "MSI", "XMET"])
common_orbit_files = get_orbit_files(common_orbits, "XMET")
base_dir = "/data/s6/L1/EarthCare/Meteo_Supporting_Files/AUX_MET_1D_aligned_CPR"
variables_to_save = ["temperature"]

# %%
num_of_files_existed = 0
num_of_files_created = 0

for i in range(len(common_orbit_files)):
    _, _, _, _, _, _, _, y, m, d, filename, _ = common_orbit_files[i].split("/")
    save_dir = os.path.join(base_dir, y, m, d, filename)
    save_file_path = os.path.join(save_dir, f"{filename}.nc")
    orbit_number = filename[-6:]
    print(orbit_number)

    # Check if the file already exists
    if not os.path.exists(save_file_path):

        xmet = align_xmet_horizontal_grid(
            read_xmet(orbit_number=orbit_number),
            read_cpr(orbit_number=orbit_number).set_xindex(["latitude", "longitude"]),
        )
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the aligned XMET data to a NetCDF file
        xmet[variables_to_save].reset_index("horizontal_grid").to_netcdf(
            path=save_file_path,
            mode="w",
        )
        print(f"File saved to {save_file_path}")
        num_of_files_created += 1
    else:
        if REWRITE:
            NotImplementedError(
                f"File already exists: {save_file_path}, but REWRITE is set to True."
            )
        print("File already exists, skip.")
        num_of_files_existed += 1


print(f"Number of files created: {num_of_files_created}")
print(f"Number of files existed: {num_of_files_existed}")

# %%
if __name__ == "__main__":

    # %%
    base_dir = "/data/s6/L1/EarthCare/Meteo_Supporting_Files/AUX_MET_1D_aligned_CPR"
    orbits = get_all_orbit_numbers_per_instrument(base_path=base_dir)
    len(orbits)
    # %%
    xr.open_dataset(get_orbit_files(orbits[0], base_path=base_dir)[0])
# %%
