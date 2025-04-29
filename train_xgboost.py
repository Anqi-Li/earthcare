# %%
import xgboost as xgb
import numpy as np
from functions.combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    package_ml_xy,
)
from functions.search_orbit_files import (
    get_common_orbits,
)
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# %% select orbit numbers
model_tag = "ten_days_three_channels"
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET_aligned"],
    date_range=["2025/02/01", "2025/02/10"],
    # date_list=["2025/02/01"],
)
print(model_tag)
print("total number of orbits: ", len(common_orbits))

# %% load the data to xarray datasets
print("Loading data...")
start = datetime.now()
orbit_numbers = common_orbits
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
num_boost_round = 100  # Number of boosting rounds
model = xgb.train(params, dtrain, num_boost_round)

print("Model training completed in", datetime.now() - start)

# %% Save the model to a file
model.save_model("./data/xgb_regressor_{}.json".format(model_tag))
print("Model saved")

