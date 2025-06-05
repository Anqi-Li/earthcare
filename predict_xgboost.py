# %%
from functions.combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    package_ml_xy,
)

import xgboost as xgb
import warnings
import os

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)

model_tag = "all_orbits_20250101_20250501_seed42"

# %% load the fitted model
model = xgb.Booster()
model.load_model(f"/home/anqil/earthcare/data/xgb_regressor_{model_tag}.json")
print("Model loaded")
print("Model tag:", model_tag)

# %% read the orbit numbers from the file
with open(f"./data/orbit_numbers_{model_tag}.txt", "r") as f:
    lines = f.readlines()
    common_orbits_train = []
    common_orbits_test = []
    reading_train = True
    for line in lines:
        if "train orbits:" in line:
            continue
        elif "test orbits:" in line:
            reading_train = False
            continue
        if reading_train:
            common_orbits_train.append(line.strip())
        else:
            common_orbits_test.append(line.strip())

# %%
filepath = f"./data/eval_results_{model_tag}"
if not os.path.exists(filepath):
    os.makedirs(filepath)

print("Starting prediction on every test orbit...")
print(f"Number of test orbits: {len(common_orbits_test)}")
for i, orbit_number in enumerate(common_orbits_test):
    print(f"{i}/{len(common_orbits_test)-1}: {orbit_number}")

    # check if the result file exists
    filename = filepath + f"/{model_tag}_{orbit_number}.nc"
    if os.path.exists(filename):
        print(f"File {orbit_number} exists, skipping...")
        continue

    # % load test data
    xds, ds_xmet = get_cpr_msi_from_orbits(
        orbit_numbers=orbit_number,
        get_xmet=True,
        msi_band=[4, 5, 6],
        filter_ground=True,
        add_dBZ=True,
    )
    # %
    X_test, y_test = package_ml_xy(
        xds=xds,
        ds_xmet=ds_xmet,
        lowest_dBZ_threshold=-25,
    )
    if y_test.nray.size == 0:
        print(f"y_test is empty for {orbit_number}, skipping...")
        continue

    # % Predict
    y_pred = model.predict(xgb.DMatrix(X_test))
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f"Root Mean Squared Error: {rmse}")

    y_test.to_dataset(name="y_true").assign(y_pred=(("nray", "band"), y_pred)).reset_index("nray").to_netcdf(
        filename,
    )

print("Evaluation completed.")
