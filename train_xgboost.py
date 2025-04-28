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
model_tag = "five_day_orbits"
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET_aligned"],
    date_range=["2025/02/01", "2025/02/05"],
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
    msi_band=6,
    get_xmet=True,
    filter_ground=True,
    add_dBZ=True,
)
print("Load xarray data", datetime.now() - start)

# %%
X_train, y_train = package_ml_xy(xds=xds, ds_xmet=ds_xmet, lowest_dBZ_threshold=-25)
dtrain = xgb.DMatrix(X_train, label=y_train)

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

# %% Load the model from a file
# model = xgb.Booster()
# model.load_model("/home/anqil/earthcare/data/xgboost_regressor_one_day_orbit,.json")
# print("Model loaded")

# # %% predict
# print("Predicting...")
# start = datetime.now()
# dtest = xgb.DMatrix(X_train)
# y_pred = model.predict(dtest)
# print("Prediction completed in", datetime.now() - start)

# %%
# from sklearn.metrics import mean_squared_error, r2_score

# y_test = y_train  # Assuming y_test is the same as y_train for this example
# # Calculate evaluation metrics
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# r2 = r2_score(y_test, y_pred)
# print(f"R-squared: {r2}")

# # %% plotting
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.title("XGBoost Regression Predictions vs True Values")
# plt.grid()
# plt.show()
# %%
