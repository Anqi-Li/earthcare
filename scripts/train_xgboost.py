# %%
import glob
from os import path
import xarray as xr
import xgboost as xgb
import random
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

model_tag = "all_orbits_20250101_20250501_seed42"

# %% list all nc files in the directory
path_to_data = "./data/training_data/"
nc_files = glob.glob(path.join(path_to_data, "*.nc"))
nc_files.sort()
random.seed(42)  # Set a seed for reproducibility
random.shuffle(nc_files)
split_index = int(len(nc_files) * 0.8)
nc_files_train = nc_files[:split_index]
nc_files_test = nc_files[split_index:]

# save the orbit numbers to a file
with open("./data/orbit_numbers_{}.txt".format(model_tag), "w") as f:
    f.write("train orbits:\n")
    for file in nc_files_train:
        orbit = file[-9:-3]  # Extract orbit number from filename
        f.write(orbit + "\n")
    f.write("test orbits:\n")
    for file in nc_files_test:
        orbit = file[-9:-3]  # Extract orbit number from filename
        f.write(orbit + "\n")


# %% Load the data from the netCDF files and convert to DMatrix
def get_Dmatrix(nc_files):
    ds = xr.open_mfdataset(
        nc_files,
        combine="nested",
        concat_dim="nray",
        parallel=True,
    )
    dtrain = xgb.DMatrix(ds.x, label=ds.pixel_values)
    return dtrain


print("Loading training data from netCDF files...")
start = datetime.now()
dtrain = get_Dmatrix(nc_files_train)
dtest = get_Dmatrix(nc_files_test)
print("Load training data", datetime.now() - start)

# %% load previoiusly saved model
model_previous = xgb.Booster()
model_previous.load_model("./data/xgb_regressor_{}.json".format(model_tag))

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
    num_boost_round=100,  # Number of boosting rounds
    evals=[(dtrain, "train"), (dtest, "validation")],  # Evaluation set
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

# %%
