# %%
from functions.combine_CPR_MSI import (
    combine_cpr_msi_from_orbits,
    package_ml_xy,
)
from functions.search_orbit_files import get_common_orbits
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xarray as xr
import joblib
from datetime import datetime
import numpy as np

# %%
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET"],
    date_list=["2025/03/01"],
)
len(common_orbits)
# %%
orbit_numbers = common_orbits[2:3]
xds, ds_xmet = combine_cpr_msi_from_orbits(
    orbit_numbers=orbit_numbers,
    get_xmet=True,
)
# %%
X_test, y_test = package_ml_xy(
    xds=xds,
    ds_xmet=ds_xmet,
    height_grid=np.arange(1e3, 15e3, 100),
)
# %% load the fitted model
regressor = joblib.load("./data/my_rf_regressor_20250410215809.joblib")

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

# # %%
# fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, sharex=True)
# for band in range(3):
#     hist = axes[band].hist2d(
#         y_test[:, band],
#         y_pred[:, band],
#         bins=(50, 50),
#         vmax=0.012,
#         cmap="Blues",
#         density=True,
#     )
#     # Add a colorbar for the current subplot
#     cbar = fig.colorbar(hist[3], ax=axes[band], orientation="horizontal")
#     # cbar.set_label("Counts")  # Label for the colorbar    axes[band].ylabel("Predicted [K]")
#     axes[band].set_title(["TIR1 (8.35-9.25)", "TIR2 (10.35-11.25)", "TIR3 (11.55-12.45)"][band])
#     axes[band].set_xlabel("True [K]")
# axes[0].set_ylabel("Predicted [K]")
# plt.show()

# # %% Plot da_dBZ_height and da_T_height to show the input variables
# fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# # Plot da_dBZ_height
# im1 = axes[0].imshow(
#     da_dBZ_height.T,
#     aspect="auto",
#     origin="lower",
#     extent=[0, da_dBZ_height.shape[0], height_grid.min(), height_grid.max()],
#     cmap="viridis",
# )
# axes[0].set_title("Reflectivity (dBZ)")
# axes[0].set_xlabel("Sample Index")
# axes[0].set_ylabel("Height [m]")
# fig.colorbar(im1, ax=axes[0], label="dBZ")

# # Plot da_T_height
# im2 = axes[1].imshow(
#     da_T_height.T,
#     aspect="auto",
#     origin="lower",
#     extent=[0, da_T_height.shape[0], height_grid.min(), height_grid.max()],
#     cmap="plasma",
# )
# axes[1].set_title("Temperature (K)")
# axes[1].set_xlabel("Sample Index")
# fig.colorbar(im2, ax=axes[1], label="Temperature [K]")

# plt.suptitle(f"Input Variables: Reflectivity and Temperature (Orbit: {orbit_numbers})")
# plt.tight_layout()
# plt.show()
# # %% Plot the orbit trajectory (latitude, longitude)
# plt.figure(figsize=(10, 6))
# plt.plot(
#     xds["longitude"],
#     xds["latitude"],
#     marker=".",
#     label="Orbit Trajectory",
#     color="blue",
# )
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title(f"Orbit Trajectory (Orbit: {orbit_numbers})")
# plt.grid(True)
# plt.legend()
# plt.show()

# %%
