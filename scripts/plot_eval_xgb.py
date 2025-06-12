# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.stats import binned_statistic
import json
import pandas as pd
import xgboost as xgb

# %%
model_tag = "all_orbits_20250101_20250501_seed42"
path_to_data = "../data/training_data/"
model = xgb.Booster()
model.load_model(f"../data/xgb_regressor_{model_tag}.json")
print("Model loaded from file")

# %% Read the orbit numbers from the saved file
with open(f"../data/orbit_numbers_{model_tag}.txt", "r") as f:
    lines = f.readlines()
    orbits_train = []
    orbits_test = []
    reading_train = True
    for line in lines:
        if "train orbits:" in line:
            continue
        elif "test orbits:" in line:
            reading_train = False
            continue
        if reading_train:
            orbits_train.append(line.strip())
        else:
            orbits_test.append(line.strip())


# %%
def get_predict(model, ds):
    dtest = xgb.DMatrix(ds.x)
    y_pred = model.predict(dtest)

    ds = ds.rename({"pixel_values": "y_true"})
    ds["y_pred"] = (ds.y_true.dims, y_pred)
    return ds


# %% find the netCDF files for the training and test orbits
# nc_files_train = [f"{path_to_data}training_data_{orbit}.nc" for orbit in orbits_train]
nc_files_test = [f"{path_to_data}training_data_{orbit}.nc" for orbit in orbits_test]

ds = xr.open_mfdataset(
    nc_files_test,
    combine="nested",
    concat_dim="nray",
    parallel=True,
)

ds = get_predict(model, ds)


# %% Calculate the conditional distribution
def get_cond_distribution(bin_edges, y_true, y_pred):
    """
    Calculate the conditional distribution of y_pred given y_true.
    """
    h_joint, _, _ = np.histogram2d(y_true, y_pred, bins=bin_edges, density=True)
    h_true, _ = np.histogram(y_true, bins=bin_edges, density=True)
    h_cond = h_joint / h_true[:, None]
    return h_cond


def get_bin_statistics(bin_edges, y_true, y_pred):
    """
    Calculate the mean and percentiles of the conditional distribution.
    """
    bin_means, _, _ = binned_statistic(
        y_true,
        y_pred,
        statistic="mean",
        bins=bin_edges,
    )
    bin_per90, _, _ = binned_statistic(
        y_true,
        y_pred,
        statistic=lambda x: np.percentile(x, 90),
        bins=bin_edges,
    )
    bin_per10, _, _ = binned_statistic(
        y_true,
        y_pred,
        statistic=lambda x: np.percentile(x, 10),
        bins=bin_edges,
    )
    return bin_means, bin_per90, bin_per10


# Define the bin edges and midpoints
bin_edges = np.arange(180, 330, 5)
bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2

# %% Calculate the conditional distribution and statistics for each band
h_cond_pred_true_nan_ = []
bin_means_ = []
bin_per90_ = []
bin_per10_ = []
for i in range(3):
    h_cond_pred_true = get_cond_distribution(
        bin_edges,
        ds.isel(band=i)["y_true"],
        ds.isel(band=i)["y_pred"],
    )
    h_cond_pred_true_nan = np.where(h_cond_pred_true > 0, h_cond_pred_true, np.nan)

    bin_means, bin_per90, bin_per10 = get_bin_statistics(
        bin_edges,
        ds.isel(band=i)["y_true"],
        ds.isel(band=i)["y_pred"],
    )

    h_cond_pred_true_nan_.append(h_cond_pred_true_nan)
    bin_means_.append(bin_means)
    bin_per90_.append(bin_per90)
    bin_per10_.append(bin_per10)

# %% Plot the conditional distribution for all bands
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)
for i in range(3):
    ax = axes[i]
    ax.set_title(f"Band {i}")
    ax.plot(bin_mid, bin_means_[i], label="Mean", c="k")
    ax.plot(bin_mid, bin_per90_[i], label="90th percentile")
    ax.plot(bin_mid, bin_per10_[i], label="10th percentile")
    ax.pcolormesh(bin_edges, bin_edges, h_cond_pred_true_nan_[i].T, cmap="Blues", vmin=0, vmax=0.2)
    # ax.contourf(bin_mid, bin_mid, h_cond_pred_true_nan_[i].T, cmap="Blues", levels=np.linspace(0, 0.2, 10))
    ax.plot(bin_mid, bin_mid, color="red", linestyle="--", label="y_true = y_pred")
    ax.set_xlabel("True [K]")

axes[0].set_ylabel("Pred [K]")
axes[0].legend()

# Add a colorbar below all subplots
cbar = plt.colorbar(axes[0].collections[0], ax=axes, orientation="horizontal", label="Density", aspect=40)
# Adjust layout to prevent overlap
plt.subplots_adjust(bottom=0.35)
plt.suptitle(
    f"""
EarthCare CPR -> TIR P(pred | true)
for Model "{model_tag}"
""",
    fontsize=16,
    y=1.1,
)
plt.show()

# %% Plot prediction vs true with training data for one orbit frame
orbit_frame = orbits_test[0]  # Change this to the desired orbit number
ds_one_orbit = xr.open_dataset(f"../data/training_data/training_data_{orbit_frame}.nc")
ds_one_orbit = get_predict(model, ds_one_orbit)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ds_train_dbz = ds_one_orbit.set_xindex(("param", "height_grid")).unstack("features").sel(param="dBZ").x
ds_train_dbz.where(ds_train_dbz != -50).plot(
    ax=axes[0],
    x="nray",
    y="height_grid",
    cmap="viridis",
    vmin=-30,
    vmax=30,
    add_colorbar=True,
    cbar_kwargs={"label": "dBZ", "orientation": "horizontal", "aspect": 40},
)
ds_one_orbit[["y_true", "y_pred"]].isel(band=0).to_array().plot(
    ax=axes[1],
    x="nray",
    hue="variable",
)
ds_one_orbit[["y_true", "y_pred"]].isel(band=0).to_array().diff("variable").plot(
    ax=axes[2],
    x="nray",
)
axes[0].set_ylabel("Height [m]")
axes[0].set_xlabel("")
axes[1].set_ylabel("Brightness Temperature [K]")
axes[1].set_xlabel("")
axes[2].set_ylabel("Difference [K]")
axes[2].set_title("")
axes[-1].set_xlabel("Ray number")
fig.suptitle(
    f"Orbit frame: {orbit_frame}\n" f"Model: {model_tag}",
    fontsize=16,
)
plt.tight_layout()

# %% Feature importance plot
feature_importance = model.get_score(importance_type="gain")
df_importance = pd.DataFrame.from_dict(feature_importance, orient="index", columns=["importance"])
ds_importance = xr.Dataset.from_dataframe(df_importance).assign_coords(index=ds_one_orbit.features).swap_dims(index="features")
ds_importance = ds_importance.set_index(features=["param", "height_grid"]).unstack("features")
ds_importance.importance.plot(
    x="height_grid",
    hue="param",
)
