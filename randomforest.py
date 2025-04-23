# %% import libraries
from sklearn.ensemble import RandomForestRegressor
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
model_tag = "five_days_orbits,-25dBZ,remove_y_extremes"
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET_aligned"],
    date_range=["2025/02/01", "2025/02/05"],
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

# %% Fit model
regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=0,
    criterion="squared_error",
)

print("fitting random forest regessor")
start = datetime.now()
regressor.fit(X_train, y_train)
print("Model fitting", datetime.now() - start)

# %% save model to file
now = datetime.now().strftime("%Y%m%d%H%M%S")
joblib.dump(regressor, f"./data/rf_regressor_{model_tag}_{now}.joblib")
