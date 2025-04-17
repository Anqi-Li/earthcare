import os
from datetime import datetime, timedelta


# %%
def get_all_orbit_numbers_per_instrument(
    inst: str = None,
    date: str = "",  # format: "YYYY/MM/DD"
    base_path: str = "/data/s6/L1/EarthCare/",
    get_full_path: bool = False,
) -> list:
    """
    Get all orbit numbers for the specified instrument (CPR, MSI, or XMET).
    """

    if inst is not None:
        # Define the base directory depending on the instrument
        if inst == "CPR":
            base_path = os.path.join(base_path, "L1/CPR_NOM_1B", date)
        elif inst == "MSI":
            base_path = os.path.join(base_path, "L1/MSI_NOM_1B", date)
        elif inst == "XMET":
            base_path = os.path.join(
                base_path, "Meteo_Supporting_Files/AUX_MET_1D", date
            )
        elif inst == "XMET_aligned":
            base_path = os.path.join(
                base_path, "Meteo_Supporting_Files/AUX_MET_1D_aligned_CPR", date
            )
    else:
        base_path = os.path.join(base_path, date)

    # Walk through the directory and collect orbit numbers
    orbit_numbers = []
    paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if ".h5" in file or ".nc" in file:
                orbit_numbers.append(file[-9:-3])
                paths.append(os.path.join(root, file))

    if get_full_path:
        return paths
    else:
        return orbit_numbers


# %% search common orbit numbers
def get_common_orbits(
    instruments: list[str],
    date_list: list[str] = [""],  # format: ["YYYY/MM/DD"]
    date_range: list[str] = None,  # format: ["YYYY/MM/DD", "YYYY/MM/DD"]
) -> list:
    """
    Get the common orbit numbers for the given instruments and date (format: "YYYY/MM/DD").
    If the date is not given, return all available common orbit numbers.
    """
    if date_range is not None:
        date_list = get_date_list_from_range(date_range=date_range)

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
    else:
        base_path = os.path.join(base_path, date)

    if isinstance(orbit_numbers, str):
        orbit_numbers = [orbit_numbers]

    orbit_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(
                orbit_number + ".h5" in file or orbit_number + ".nc" in file
                for orbit_number in orbit_numbers
            ):
                orbit_files.append(os.path.join(root, file))

    return orbit_files
