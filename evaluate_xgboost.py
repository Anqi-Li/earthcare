# %%
from functions.combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    package_ml_xy,
)
from functions.search_orbit_files import get_common_orbits
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import os

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)

model_tag = "all_orbits_20250101_20250501_seed42"

# %% load the fitted model
model = xgb.Booster()
model.load_model(f"/home/anqil/earthcare/data/xgb_regressor_{model_tag}.json")
print("Model loaded")
print("Model tag:", model_tag)

# %% read the orbit numbers from the file
with open(f"./data/orbit_numbers_{model_tag}.txt", "r") as f:
    lines = f.readlines()
    common_orbits_train = []
    common_orbits_test = []
    reading_train = True
    for line in lines:
        if "train orbits:" in line:
            continue
        elif "test orbits:" in line:
            reading_train = False
            continue
        if reading_train:
            common_orbits_train.append(line.strip())
        else:
            common_orbits_test.append(line.strip())

# %%
print("Starting evaluation on test orbits...")
print(f"Number of test orbits: {len(common_orbits_test)}")
for i, orbit_number in enumerate(common_orbits_test):
    print(f"{i}/{len(common_orbits_test)-1}: {orbit_number}")

    # check if the result file exists
    if os.path.exists(f"./data/eval_results_{model_tag}/{orbit_number}.nc"):
        print(f"File {orbit_number} exists, skipping...")
        continue

    # % load test data
    xds, ds_xmet = get_cpr_msi_from_orbits(
        orbit_numbers=orbit_number,
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
    if y_test.nray.size == 0:
        print(f"y_test is empty for {orbit_number}, skipping...")
        continue

    # % Predict
    y_pred = model.predict(xgb.DMatrix(X_test))
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f"Root Mean Squared Error: {rmse}")

    y_test.to_dataset(name="y_true").assign(y_pred=(("nray", "band"), y_pred)).reset_index("nray").to_netcdf(
        f"./data/eval_results_{model_tag}/{orbit_number}.nc",
    )

# %% compile the results (one orbit) to an xarray dataset
# def get_ds_subset(df_subset):
#     ds_subset = xr.merge(df_subset["y_test"].values).rename(pixel_values="y_test")
#     ds_subset = ds_subset.assign({"y_pred": (("nray", "band"), np.concatenate(df_subset["y_pred"].values))})
#     # ds_subset = ds_subset.assign(
#     #     {
#     #         "X_test": (
#     #             ("nray", "z", "var"),
#     #             np.concatenate(df_subset["X_test"].values).reshape(-1, 140, 2),
#     #         )
#     #     }
#     # )
#     return ds_subset


# # %% plot series of X_test, y_test and y_pred
# for i in range(len(df)):
#     df_subset = df.iloc[i : i + 1]
#     ds_subset = get_ds_subset(df_subset)

#     # % plot series of X_test, y_test and y_pred
#     fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
#     axes.set_title("X_test")
#     ds_subset["X_test"].where(ds_subset["X_test"] > -25).isel(var=0).plot.contourf(
#         ax=axes,
#         x="time",
#         y="z",
#         cmap="viridis",
#         levels=np.linspace(-25, 25, 10),
#         add_colorbar=False,
#         # cbar_kwargs=dict(orientation="horizontal", aspect=30),
#     )
#     axes.set_title(df_subset.index.item())

#     axes_twinx = axes.twinx()
#     ds_subset["y_test"].isel(band=0).plot(
#         ax=axes_twinx,
#         x="time",
#         label="y_true",
#         ls="-",
#         marker=".",
#         markersize=2,
#         alpha=0.5,
#     )
#     ds_subset["y_pred"].isel(band=0).plot(
#         ax=axes_twinx,
#         x="time",
#         label="y_pred",
#         ls="-",
#         marker=".",
#         markersize=2,
#         alpha=0.5,
#     )
#     axes_twinx.invert_yaxis()
#     axes_twinx.set_ylabel("Temperature [K]")
#     axes_twinx.legend()

# %% plot 2d histogram of y_test and y_pred in three bands
# ds_subset = get_ds_subset(df.iloc[5:])
# fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
# for band in range(3):
#     ax = axes[band]
#     h = np.histogram2d(
#         ds_subset["y_test"].isel(band=band).values.flatten(),
#         ds_subset["y_pred"].isel(band=band).values.flatten(),
#         bins=(30, 30),
#         density=True,
#     )
#     ax.contour(
#         h[1][:-1],
#         h[2][:-1],
#         h[0].T,
#         # levels=np.logspace(-4, -2, 8),
#     )
#     min_val = min(
#         ds_subset["y_test"].isel(band=band).values.min(),
#         ds_subset["y_pred"].isel(band=band).values.min(),
#     )
#     max_val = max(
#         ds_subset["y_test"].isel(band=band).values.max(),
#         ds_subset["y_pred"].isel(band=band).values.max(),
#     )
#     ax.plot([min_val, max_val], [min_val, max_val], "r--", label="True = Predicted")

#     ax.set_xlabel("y_test")
#     ax.set_ylabel("y_pred")
#     ax.set_title(f"Band {band+1}")

# # %%
# ds_subset = get_ds_subset(df.iloc[:]).load()
# cond = np.logical_and(
#     (ds_subset.y_test - ds_subset.y_pred) > 10, ds_subset.y_test < 240
# )
# ds_subset_investigate = ds_subset.where(cond.all(dim="band"), drop=True)

# %%
# plt.figure()
# df["RMSE"].plot(marker=".")
# plt.title(
#     "RMSE for each orbit frame (202501), mean = {:.2f} [K]".format(df["RMSE"].mean())
# )
# plt.axhline(
#     df["RMSE"].mean(),
#     color="red",
#     linestyle="--",
#     label="Mean RMSE",
# )
# plt.legend()
# plt.xlabel("Orbit Number")
# plt.ylabel("RMSE [K]")
# plt.xticks(rotation=45)
# plt.grid()
# plt.tight_layout()

# plt.savefig(
#     "/home/anqil/earthcare/figures/ML_evaluation_xgb_rmse_orbit_202501.png",
#     dpi=300,
#     bbox_inches="tight",
# )


# # %% randomly select n orbits
# eval_dict = {}
# n_test = 50
# n_orbits = 3
# for i in range(n_test):
#     print(f"{i}/{n_test}")

#     chosen_orbits = np.random.choice(common_orbits_test, n_orbits, replace=False)

#     # % load test data
#     xds, ds_xmet = get_cpr_msi_from_orbits(
#         orbit_numbers=chosen_orbits,
#         get_xmet=True,
#         msi_band=[4, 5, 6],
#         filter_ground=True,
#         add_dBZ=True,
#     )
#     # %
#     X_test, y_test = package_ml_xy(
#         xds=xds,
#         ds_xmet=ds_xmet,
#         lowest_dBZ_threshold=-25,
#     )

#     # % Predict
#     y_pred = model.predict(xgb.DMatrix(X_test))
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Mean Squared Error: {mse}")
#     # r2 = r2_score(y_test, y_pred)
#     # print(f"R-squared: {r2}")

#     eval_dict[i] = np.sqrt(mse)

# # %%
# df_random = pd.DataFrame.from_dict(
#     eval_dict,
#     orient="index",
#     columns=["RMSE"],
# )
# df_random.plot()

# %%
