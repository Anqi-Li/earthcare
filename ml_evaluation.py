# %%
from functions.combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    package_ml_xy,
)
from functions.search_orbit_files import get_common_orbits
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xarray as xr
import joblib
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)

# %% load the fitted model
regressor = joblib.load(
    "/home/anqil/earthcare/data/rf_regressor_one_day_orbits,-25dBZ,remove_03868H_20250422235221.joblib"
)

# %%
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET"],
    date_list=["2025/02/01"],
)
len(common_orbits)
# %% load test data
orbit_numbers = common_orbits[:3]
xds, ds_xmet = get_cpr_msi_from_orbits(
    orbit_numbers=orbit_numbers,
    get_xmet=True,
    msi_band=6,
    filter_ground=True,
    add_dBZ=True,
)
# %
X_test, y_test = package_ml_xy(
    xds=xds,
    ds_xmet=ds_xmet,
    lowest_dBZ_threshold=-25,
    height_grid=np.arange(1e3, 15e3, 100)
)

# %% Predict
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# %% scatter plot of the predictions vs true values
plt.figure(figsize=(6, 5))
plt.plot(
    y_test,
    y_pred,
    ".",
    alpha=0.5,
    markersize=3,
    # label=["TIR1 (8.35-9.25)", "TIR2 (10.35-11.25)", "TIR3 (11.55-12.45)"],
)
plt.xlabel("True [K]")
plt.ylabel("Predicted [K]")
plt.title(
    """
Random Forest Regression
Predicting TIR brightness temperature from dBZ
Orbit number: {}
""".format(
        orbit_numbers
    )
)

plt.text(0.1, 0.9, f"MSE: {mse:.2f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f"R2: {r2:.2f}", transform=plt.gca().transAxes)

# Add diagonal line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="True = Predicted")

plt.legend()
plt.show()

# %% Histogram of the predictions vs true values
if len(y_test.shape) == 1:
    y_test = y_test.expand_dims(dim="band").T
    y_pred = y_pred.reshape(-1, 1)
fig, axes = plt.subplots(
    1,
    len(y_test.band),
    figsize=(15, 6),
    sharey=True,
    sharex=True,
)
if len(y_test.band) == 1:
    axes = [axes]
for band in range(len(y_test.band)):
    # hist = axes[band].hist2d(
    #     y_test[:, band],
    #     y_pred[:, band],
    #     bins=(50, 50),
    #     # vmax=0.01,
    #     cmap="Blues",
    #     density=True,
    # )
    h = np.histogram2d(y_test[:, band], y_pred[:, band], bins=(50, 50))
    axes[band].contour(h[1][:-1], h[2][:-1], h[0])

    # Add diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[band].plot(
        [min_val, max_val], [min_val, max_val], "r--", label="True = Predicted"
    )

    # Add a colorbar for the current subplot
    # cbar = fig.colorbar(hist[3], ax=axes[band], orientation="horizontal")
    # axes[band].set_title(["TIR1 (8.35-9.25)", "TIR2 (10.35-11.25)", "TIR3 (11.55-12.45)"][band])
    axes[band].set_xlabel("True [K]")
axes[0].set_ylabel("Predicted [K]")
plt.show()

# %%
