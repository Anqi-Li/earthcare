# %%
# This script counts the number of files downloaded per day for each instrument
# and plots the results.
# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from functions.search_orbit_files import (
    get_common_orbits,
    get_orbit_files,
    get_all_orbit_numbers_per_instrument,
)

from datetime import datetime
from collections import Counter


# %%
def count_num_files_per_day(orbit_numbers, inst):
    """
    Count the number of files per day for a given instrument and list of orbit numbers.
    """
    if inst == "Common":
        inst = "CPR"

    # Get file paths for the specified instrument
    file_paths = get_orbit_files(orbit_numbers=orbit_numbers, inst=inst)
    file_paths_unique_orbit = list({f[-9:-3]: f for f in file_paths}.values())

    # Extract dates from file paths (assuming the date is part of the directory structure)
    file_dates = ["".join(path.split("/")[7:10]) for path in file_paths_unique_orbit]

    # Count occurrences of each date
    file_count_per_day = Counter(file_dates)

    # save the result into arrays
    days = []
    counts = []
    for day, count in file_count_per_day.items():
        days.append(day)
        counts.append(count)

    # Convert string dates to datetime objects for better plotting
    days = [datetime.strptime(day, "%Y%m%d") for day in days]

    # Sort the dates and counts
    sorted_indices = np.argsort(days)
    days = np.array(days)[sorted_indices]
    counts = np.array(counts)[sorted_indices]

    return days, counts


# %% Plotting
orbit_dict = {
    "Common": get_common_orbits(instruments=["CPR", "MSI", "XMET"]),
    "CPR": get_all_orbit_numbers_per_instrument(inst="CPR"),
    "MSI": get_all_orbit_numbers_per_instrument(inst="MSI"),
    "XMET": list(set(get_all_orbit_numbers_per_instrument(inst="XMET"))),
    # "XMET_aligned": get_all_orbit_numbers_per_instrument(inst="XMET_aligned"),
}
# %%
count_dict = {}
for inst, orbit_numbers in orbit_dict.items():
    # Get the counts for each instrument
    counts = count_num_files_per_day(orbit_numbers=orbit_numbers, inst=inst)
    count_dict[inst] = counts

# %%
plt.figure(figsize=(10, 5))
for inst in orbit_dict.keys():
    counts = count_dict[inst]
    # Plot the data
    plt.plot(*counts, label=inst, marker="o", linestyle="-", alpha=0.5)
    # plt.step(*counts, label=inst, where='mid', marker="o", linestyle="-")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gcf().autofmt_xdate()  # Rotate date labels
plt.legend()
# plt.xlim(datetime(2025, 1, 1), None)
plt.xlabel("Date")
plt.ylabel("Number of files")
plt.title("Number of downloaded EarthCARE files per day")
plt.grid()
plt.tight_layout()
# plt.show()

# %% save the figure
plt.savefig(
    "/home/anqil/earthcare/figures/number_of_files_per_day.png",
    dpi=300,
    bbox_inches="tight",
)
