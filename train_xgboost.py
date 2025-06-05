# %%
import xgboost as xgb
from functions.combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    package_ml_xy,
)
from functions.search_orbit_files import (
    get_common_orbits,
)
import random
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

model_tag = "all_orbits_20250101_20250501_seed42"

# %% select orbit numbers
common_orbits_all = get_common_orbits(
    ["CPR", "MSI", "XMET_aligned"],
    date_range=["2025/01/01", "2025/05/01"],
)

# %%
random.seed(42)  # Set a seed for reproducibility
random.shuffle(common_orbits_all)
split_index = int(len(common_orbits_all) * 0.8)
common_orbits_train = common_orbits_all[:split_index]
common_orbits_test = common_orbits_all[split_index:]

# save the orbit numbers to a file
with open("./data/orbit_numbers_{}.txt".format(model_tag), "w") as f:
    f.write("train orbits:\n")
    for orbit in common_orbits_train:
        f.write(orbit + "\n")
    f.write("test orbits:\n")
    for orbit in common_orbits_test:
        f.write(orbit + "\n")

print(model_tag)
print("total number of orbits for training: ", len(common_orbits_train))

# %% load the data to xarray datasets
print("Loading training data...")
start = datetime.now()
orbit_numbers = common_orbits_train
xds, ds_xmet = get_cpr_msi_from_orbits(
    orbit_numbers=orbit_numbers,
    msi_band=[4, 5, 6],
    get_xmet=True,
    filter_ground=True,
    add_dBZ=True,
)

X_train, y_train = package_ml_xy(xds=xds, ds_xmet=ds_xmet, lowest_dBZ_threshold=-25)
dtrain = xgb.DMatrix(X_train, label=y_train)
print("Load training data", datetime.now() - start)

# %% load the data to xarray datasets
print("Loading test data...")
start = datetime.now()
orbit_numbers = common_orbits_test
xds, ds_xmet = get_cpr_msi_from_orbits(
    orbit_numbers=orbit_numbers,
    msi_band=[4, 5, 6],
    get_xmet=True,
    filter_ground=True,
    add_dBZ=True,
)

X_test, y_test = package_ml_xy(xds=xds, ds_xmet=ds_xmet, lowest_dBZ_threshold=-25)
dtest = xgb.DMatrix(X_test, label=y_test)
print("Load test data", datetime.now() - start)

# %% load previoiusly saved model
# Uncomment the following lines if you want to load a previously saved model
model_previous = xgb.Booster()
model_previous.load_model("./data/xgb_regressor_{}.json".format("all_orbits_20250101_20250501"))

# %% Fit model
print("Training XGBoost regression model...")
start = datetime.now()

# Define parameters for the XGBoost model
params = {
    "objective": "reg:squarederror",  # Regression objective
    "eval_metric": "rmse",  # Root Mean Square Error as evaluation metric
    "eta": 0.1,  # Learning rate
    "max_depth": 6,  # Maximum depth of a tree
    "subsample": 0.8,  # Subsample ratio of the training instances
    "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
    "seed": 42,  # Random seed for reproducibility
}

# Train the model
evals_result = {}  # Dictionary to store evaluation results
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,  # Number of boosting rounds
    evals=[(dtest, "validation")],  # Evaluation set
    early_stopping_rounds=10,
    verbose_eval=10,
    evals_result=evals_result,
    xgb_model=model_previous,
)

print("Model training completed in", datetime.now() - start)

# %% save evals_result to a file
with open("./data/xgb_evals_result_{}.json".format(model_tag), "w") as f:
    json.dump(evals_result, f)
print("Evaluation results saved")

# %% Save the model to a file
model.save_model("./data/xgb_regressor_{}.json".format(model_tag))
print("Model saved")
