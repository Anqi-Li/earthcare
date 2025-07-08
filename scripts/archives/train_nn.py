# %% import libraries
from sklearn.neural_network import MLPRegressor
from combine_CPR_MSI import (
    get_cpr_msi_from_orbits,
    package_ml_xy,
)
from search_orbit_files import (
    get_common_orbits,
)
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# %% select orbit numbers
model_tag = "ten_days_channel6_one_neuron"
common_orbits = get_common_orbits(
    ["CPR", "MSI", "XMET_aligned"],
    date_range=["2025/02/01", "2025/02/10"],
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
    remove_underground=True,
    add_dBZ=True,
)

X_train, y_train = package_ml_xy(xds=xds, ds_xmet=ds_xmet, lowest_dBZ_threshold=-25)
print("Load training data", datetime.now() - start)

# %%
# Initialize the neural network model
nn_model = MLPRegressor(
    hidden_layer_sizes=(1,),  # One hidden layer with 10 neurons
    activation="relu",  # Activation function
    solver="adam",  # Optimizer
    max_iter=500,  # Maximum number of iterations
    random_state=42,
)

# Train the model
print("Training the neural network...")
start = datetime.now()

nn_model.fit(X_train, y_train)

print("Model training completed in", datetime.now() - start)

# %% Save the trained model
joblib.dump(nn_model, f"./data/nn_model_{model_tag}.pkl")
print(f"Model saved")
