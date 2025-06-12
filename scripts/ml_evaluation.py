# %%
from combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    package_ml_xy,
)
from search_orbit_files import get_common_orbits
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import joblib
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)

# %% load the fitted model
regressor_rf = joblib.load(
    "/home/anqil/earthcare/data/rf_regressor_ten_days_three_channels_20250429224026.joblib"
)

regressor_xgb = xgb.Booster()
regressor_xgb.load_model(
    "/home/anqil/earthcare/data/xgb_regressor_ten_days_three_channels.json"
)

regressor_nn = joblib.load(
    "/home/anqil/earthcare/data/nn_model_ten_days_three_channels.pkl"
)
print("Model loaded")


# %%
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET_aligned"],
    # date_list=["2025/02/10"],
    date_range=["2025/01/01", "2025/01/31"],
)
len(common_orbits)
# %% load test data
orbit_numbers = common_orbits  # "03821H"  #   # "03821C" #"03852B"
xds, ds_xmet = get_cpr_msi_from_orbits(
    orbit_numbers=orbit_numbers,
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

# %% Predict
y_pred_dict = {
    "y_pred_xgb": regressor_xgb.predict(xgb.DMatrix(X_test)),
    "y_pred_rf": regressor_rf.predict(X_test),
    "y_pred_nn": regressor_nn.predict(X_test),
}
mse = [mean_squared_error(y_test, y_pred) for y_pred in y_pred_dict.values()]
print(f"Mean Squared Error: {mse}")

r2 = [r2_score(y_test, y_pred) for y_pred in y_pred_dict.values()]
print(f"R-squared: {r2}")


# %% repackage the test data into xarray for plotting
def repackage_xarray(
    X_test, y_test, y_pred_dict, xds, height_grid=np.arange(1e3, 15e3, 100)
):
    """
    Repackage the test data into xarray for plotting
    """
    if "band" in y_test.dims:
        X_copy_y = y_test.isel(band=0).expand_dims(
            dim=dict(height=len(height_grid), vars=2)
        )
    else:
        X_copy_y = y_test.expand_dims(dim=dict(height=len(height_grid), vars=2))

    X = (
        xr.DataArray(
            X_test.reshape(len(X_test), 140, 2).transpose(1, 2, 0),
            dims=X_copy_y.dims,
            coords=X_copy_y.coords,
        )
        .assign_coords(height=height_grid, vars=["dBZ", "T"])
        .to_dataset(dim="vars")
    )

    y = y_test.to_dataset(name="y_true").assign(
        {
            k: xr.DataArray(
                v,
                dims=y_test.dims,
                coords=y_test.coords,
            )
            for k, v in y_pred_dict.items()
        }
    )

    ds = xr.merge([X, y])
    ds = ds.reindex_like(xds, copy=False).assign_coords(
        time=xds.time, profileTime=xds.profileTime
    )

    return ds


# %%
ds_org = repackage_xarray(X_test, y_test, y_pred_dict, xds).sortby("time").copy()
ds = ds_org.isel(band=0)

# %% plot histograme of the predictions vs true values for each band
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
for band in ds_org.band:
    h = np.histogram2d(
        ds_org.dropna(dim="nray").y_true.sel(band=band),
        ds_org.dropna(dim="nray")["y_pred_xgb"].sel(band=band),
        bins=(30, 30),
        density=True,
    )
    axes[band].contour(h[1][:-1], h[2][:-1], h[0].T, levels=np.linspace(0, 0.003, 8))
    axes[band].set_title(f"Band {band.item()}")
    axes[band].set_xlabel("True [K]")
    # Add diagonal line
    min_val = min(
        ds_org.dropna(dim="nray").y_true.sel(band=band).min(),
        ds_org.dropna(dim="nray")["y_pred_xgb"].sel(band=band).min(),
    )
    max_val = max(
        ds_org.dropna(dim="nray").y_true.sel(band=band).max(),
        ds_org.dropna(dim="nray")["y_pred_xgb"].sel(band=band).max(),
    )
    axes[band].plot(
        [min_val, max_val], [min_val, max_val], "r--", label="True = Predicted"
    )
    # Add text with MSE and R2
    axes[band].text(
        0.1,
        0.9,
        f"RMSE: {np.sqrt(mean_squared_error(ds_org.dropna(dim='nray').y_true.sel(band=band), ds_org.dropna(dim='nray')['y_pred_xgb'].sel(band=band))):.2f}",
        transform=axes[band].transAxes,
    )
    axes[band].text(
        0.1,
        0.85,
        f"R2: {r2_score(ds_org.dropna(dim='nray').y_true.sel(band=band), ds_org.dropna(dim='nray')['y_pred_xgb'].sel(band=band)):.2f}",
        transform=axes[band].transAxes,
    )
    # Add colorbar
    # cbar = plt.colorbar(
    #     axes[band].collections[0],
    #     ax=axes[band],
    #     orientation="vertical",
    #     pad=0.01,
    #     aspect=10,
    # )
    # Misc
axes[0].set_ylabel("Predicted [K]")
plt.suptitle("Orbit number: {}".format(orbit_numbers))
plt.show()

# %% plot along track series
max_dbz = 20
min_dbz = -25
cond_dbz = ds.dBZ > min_dbz

fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex="col", sharey="row")
ds.dBZ.where(cond_dbz).plot.contourf(
    ax=axes[0],
    x="time",
    y="height",
    vmin=min_dbz,
    vmax=max_dbz,
    cbar_kwargs=dict(orientation="horizontal", aspect=30),
)

for k, v in {
    "y_true": "True MSI",
    "y_pred_rf": "Pred RF",
    "y_pred_xgb": "Pred XGB",
    "y_pred_nn": "Pred NN",
}.items():
    ds[k].plot(
        label=v,
        x="time",
        ls="-",
        marker=".",
        alpha=0.3,
        markersize=1,
        ax=axes[1],
    )

axes[1].set_ylabel("Temperature [K]")
axes[1].set_xlabel("Latitude [degrees_north]")
axes[1].set_xticklabels(
    np.interp(axes[1].get_xticks(), mdates.date2num(ds["time"]), ds.latitude).round(2)
)

secax = axes[1].secondary_xaxis("bottom")
secax.set_xlabel("Longitude [degrees_east]")
secax.set_xticks(axes[1].get_xticks())
secax.set_xticklabels(
    np.interp(axes[1].get_xticks(), mdates.date2num(ds["time"]), ds.longitude).round(2)
)
secax.spines["bottom"].set_position(("outward", 40))

axes[0].set_ylim(0, 20e3)
axes[0].set_xlabel("")
axes[0].set_title("CPR dBZ profiles")
axes[1].legend(loc="lower right")
fig.suptitle("Orbit number: {}".format(orbit_numbers))

plt.show()


# %% Histogram of the predictions vs true values for each model
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
label = ["XGBoost", "Random Forest", "Neural Network"]
for i, y_pred in enumerate(ds[y_pred_dict.keys()].dropna(dim="nray").values()):

    # Calculate the histogram
    h = np.histogram2d(
        ds.dropna(dim="nray").y_true, y_pred, bins=(30, 30), density=True
    )
    axes[i].contour(h[1][:-1], h[2][:-1], h[0].T)

    # Add text with MSE and R2
    axes[i].text(
        0.1,
        0.9,
        f"RMSE: {np.sqrt(mean_squared_error(ds.dropna(dim='nray').y_true, y_pred)):.2f}",
        transform=axes[i].transAxes,
    )
    axes[i].text(
        0.1,
        0.85,
        f"R2: {r2_score(ds.dropna(dim='nray').y_true, y_pred):.2f}",
        transform=axes[i].transAxes,
    )

    # Add diagonal line
    min_val = min(ds.dropna(dim="nray").y_true.min(), y_pred.min())
    max_val = max(ds.dropna(dim="nray").y_true.max(), y_pred.max())
    axes[i].plot(
        [min_val, max_val], [min_val, max_val], "r--", label="True = Predicted"
    )

    # Add colorbar
    cbar = plt.colorbar(
        axes[i].collections[0],
        ax=axes[i],
        orientation="vertical",
        pad=0.01,
        aspect=10,
    )
    # Misc
    axes[i].set_title(label[i])
    axes[i].set_xlabel("True [K]")
axes[0].set_ylabel("Predicted [K]")
# plt.suptitle("Orbit number: {}".format(orbit_numbers))
plt.suptitle("number of orbits: {}".format(len(orbit_numbers)))
plt.show()


# %%
