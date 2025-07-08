# %%
from search_orbit_files import (
    get_common_orbits,
    get_orbit_files,
)
from combine_CPR_MSI import read_xmet, align_xmet_horizontal_grid, read_cpr
import os
from tqdm import tqdm

REWRITE = False  # Set to True to overwrite existing files


def create_xmet_aligned_files(
    variables_to_save: list[str],
    dir_save: str,
):
    # %% search for common orbits and only convert those XMET files
    common_orbits = get_common_orbits(["CPR", "MSI", "XMET"])
    common_orbit_files = get_orbit_files(common_orbits, "XMET")

    # %%
    num_of_files_existed = 0
    num_of_files_created = 0

    for i in tqdm(range(len(common_orbit_files))):
        # print(f"Processing file {i + 1}/{len(common_orbit_files)}")

        # Get the file path and orbit number
        xmet_file_path = common_orbit_files[i]
        _, _, _, _, _, _, _, y, m, d, filename, _ = xmet_file_path.split("/")
        save_dir = os.path.join(dir_save, y, m, d, filename)
        save_file_path = os.path.join(save_dir, f"{filename}.nc")
        orbit_number = filename[-6:]

        # Check if the file already exists
        if not os.path.exists(save_file_path):

            xmet = align_xmet_horizontal_grid(
                read_xmet(file_path=xmet_file_path),
                read_cpr(orbit_number=orbit_number).set_xindex(
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
            # print(f"File saved to {save_file_path}")
            num_of_files_created += 1
        else:
            if REWRITE:
                NotImplementedError(
                    f"File already exists: {save_file_path}, but REWRITE is set to True."
                )
            # print("File already exists, skip.")
            num_of_files_existed += 1

    return num_of_files_created, num_of_files_existed


# %%
if __name__ == "__main__":
    dir_save = "/scratch/li/earthcare/AUX_MET_1D_horizontally_aligned"

    variables_to_save = [
        "temperature",
        "pressure",
        "specific_humidity",
        "ozone_mass_mixing_ratio",
        "specific_cloud_liquid_water_content",
        "specific_rain_water_content",
    ]
    num_of_files_created, num_of_files_existed = create_xmet_aligned_files(
        variables_to_save, dir_save
    )
    print(f"Number of files created: {num_of_files_created}")
    print(f"Number of files existed: {num_of_files_existed}")
