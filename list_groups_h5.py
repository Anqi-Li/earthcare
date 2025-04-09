# %%
import h5py
import os


def list_groups(full_path):
    # Open the HDF5 file and list all groups with h5py
    with h5py.File(full_path, "r") as hdf:
        print("HDF5 File Structure:")

        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")

        hdf.visititems(print_structure)


if __name__ == "__main__":
    # Define the path to the HDF5 file
    file_name = "ECA_JXCA_CPR_NOM_1B_20240803T044914Z_20250127T095411Z_01031A"
    file_path = "/data/s6/L1/EarthCare/L1/CPR_NOM_1B/2024/08/03"

    full_path = os.path.join(
        file_path,
        file_name,
        file_name + ".h5",
    )
    list_groups(file_path)
