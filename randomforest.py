# %% import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from combine_CPR_MSI import (
    combine_cpr_msi_from_orbits,
    xr_vectorized_height_interpolation,
    get_common_orbits,
    get_common_orbits_date_range,
)
import matplotlib.pyplot as plt
import xarray as xr
import joblib
from datetime import datetime

# %% select orbit numbers
# common_orbits = get_common_orbits_date_range(
#     ["CPR", "MSI", "XMET"],
#     date_range=["2025/01/30", "2025/02/02"],
# )
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET"],
    date_list=["2025/01/30"],
)
print("total number of orbits: ", len(common_orbits))

# %% load the data to xarray datasets
start = datetime.now()
# common_orbits = ["03890F"]
xds, ds_xmet = combine_cpr_msi_from_orbits(orbit_numbers=common_orbits, get_xmet=True)
print(datetime.now() - start)

# %% interpolation to a uniform height grid
height_grid = np.arange(1e3, 15e3, 100)
da_dBZ_height = xr_vectorized_height_interpolation(
    ds=xds,
    height_name="binHeight",
    variable_name="dBZ",
    height_grid=height_grid,
    height_dim="nbin",
)
da_T_height = xr_vectorized_height_interpolation(
    ds=ds_xmet,
    height_name="geometrical_height",
    variable_name="temperature",
    height_grid=height_grid,
    height_dim="height",
)

# %% Define the features and target variable
low_dBZ_replacement = -50
mask_clearsky = (da_dBZ_height < -25).all(dim="height_grid").compute()
da_dBZ_height = da_dBZ_height.where(
    np.logical_and(~da_dBZ_height.pipe(np.isinf), da_dBZ_height > -25),
    low_dBZ_replacement,
)
X = np.stack(
    [da_dBZ_height, da_T_height],
    axis=2,
)
X = X[~mask_clearsky]  # remove profiles that are clearsky

y = xds["pixel_values"].T
y = y[~mask_clearsky]

nsamples, nx, ny = X.shape
X_2d = X.reshape((nsamples, nx * ny))

# %%
# X_train, X_test, y_train, y_test = train_test_split(
#     X_2d, y, test_size=0.2, random_state=0
# )
X_train = X_2d[: int(0.8 * len(X_2d))]
X_test = X_2d[int(0.8 * len(X_2d)) :]
y_train = y[: int(0.8 * len(X_2d))]
y_test = y[int(0.8 * len(X_2d)) :]

# %% Fit model
regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=0,
    criterion="squared_error",
)
regressor.fit(X_train, y_train)
now = datetime.now().strftime("%Y%m%d%H%M%S")
joblib.dump(regressor, f"./data/my_rf_regressor{now}.joblib")

# %% Predict
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# %% plot the predictions
plt.figure(figsize=(6, 5))
plt.plot(
    y_test,
    y_pred,
    ".",
    alpha=0.5,
    markersize=3,
    label=["TIR1 (8.35-9.25)", "TIR2 (10.35-11.25)", "TIR3 (11.55-12.45)"],
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

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, sharex=True)
for band in range(3):
    hist = axes[band].hist2d(
        y_test[:, band],
        y_pred[:, band],
        bins=(50, 50),
        vmax=0.012,
        cmap="Blues",
        density=True,
    )
    # Add a colorbar for the current subplot
    cbar = fig.colorbar(hist[3], ax=axes[band], orientation="horizontal")
    # cbar.set_label("Counts")  # Label for the colorbar    axes[band].ylabel("Predicted [K]")
    axes[band].set_title(["TIR1 (8.35-9.25)", "TIR2 (10.35-11.25)", "TIR3 (11.55-12.45)"][band])
    axes[band].set_xlabel("True [K]")
axes[0].set_ylabel("Predicted [K]")
plt.show()

# %% Plot da_dBZ_height and da_T_height to show the input variables
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Plot da_dBZ_height
im1 = axes[0].imshow(
    da_dBZ_height.T,
    aspect="auto",
    origin="lower",
    extent=[0, da_dBZ_height.shape[0], height_grid.min(), height_grid.max()],
    cmap="viridis",
)
axes[0].set_title("Reflectivity (dBZ)")
axes[0].set_xlabel("Sample Index")
axes[0].set_ylabel("Height [m]")
fig.colorbar(im1, ax=axes[0], label="dBZ")

# Plot da_T_height
im2 = axes[1].imshow(
    da_T_height.T,
    aspect="auto",
    origin="lower",
    extent=[0, da_T_height.shape[0], height_grid.min(), height_grid.max()],
    cmap="plasma",
)
axes[1].set_title("Temperature (K)")
axes[1].set_xlabel("Sample Index")
fig.colorbar(im2, ax=axes[1], label="Temperature [K]")

plt.suptitle(f"Input Variables: Reflectivity and Temperature (Orbit: {orbit_numbers})")
plt.tight_layout()
plt.show()
# %% Plot the orbit trajectory (latitude, longitude)
plt.figure(figsize=(10, 6))
plt.plot(
    xds["longitude"],
    xds["latitude"],
    marker=".",
    label="Orbit Trajectory",
    color="blue",
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Orbit Trajectory (Orbit: {orbit_numbers})")
plt.grid(True)
plt.legend()
plt.show()
