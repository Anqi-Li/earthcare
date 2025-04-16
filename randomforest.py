# %% import libraries
from sklearn.ensemble import RandomForestRegressor
from functions.combine_CPR_MSI import (
    combine_cpr_msi_from_orbits,
    package_ml_xy,
)
from functions.search_orbit_files import (
    get_common_orbits,
    get_date_list_from_range,
)
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# %% select orbit numbers
model_tag = ""
date_list = get_date_list_from_range(date_range=["2025/01/30", "2025/02/10"])
# date_list = ["2025/01/30"]
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET"],
    date_list=date_list,
)
print("total number of orbits: ", len(common_orbits))

# %% load the data to xarray datasets
print("Loading data...")
start = datetime.now()
orbit_numbers = common_orbits
xds, ds_xmet = combine_cpr_msi_from_orbits(orbit_numbers=orbit_numbers, get_xmet=True)
# xds = combine_cpr_msi_from_orbits(orbit_numbers=orbit_numbers, get_xmet=False)
print("Load xarray data", datetime.now() - start)

# %%
X_train, y_train = package_ml_xy(xds=xds, ds_xmet=ds_xmet)

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
joblib.dump(regressor, f"./data/my_rf_regressor_{now}{model_tag}.joblib")
