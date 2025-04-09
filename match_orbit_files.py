# %%
from combine_CPR_MSI import get_all_orbit_numbers_per_instrument, get_orbit_files

# %% search common orbit numbers
# Get all orbit numbers for each instrument
orbit_numbers_cpr = get_all_orbit_numbers_per_instrument("CPR")
orbit_numbers_msi = get_all_orbit_numbers_per_instrument("MSI")
orbit_numbers_xmet = get_all_orbit_numbers_per_instrument("XMET")

# Find elements that exist in all three lists
common_orbits = (
    set(orbit_numbers_cpr) & set(orbit_numbers_msi) & set(orbit_numbers_xmet)
)
print("Total num of common orbits:", len(common_orbits))

# %% write the matching orbit numbers to a file
with open("./data/test.txt", "w") as f:
    for line in common_orbits:
        f.write(f"{line}\n")

# %% read the matching orbit numbers from the file
with open("./data/matched_orbits.txt", "r") as file:
    # Read all lines into a list
    matching_orbits = [line.strip() for line in file]
    matching_orbits.sort()

# %%
get_orbit_files(list(common_orbits)[100], base_path="/data/s6/L1/EarthCare/")
