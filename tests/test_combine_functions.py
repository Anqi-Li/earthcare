"""
Tests for specific functions in combine_CPR_MSI.py
Tests: merge_colocated, xr_vectorized_height_interpolation, align_xmet_horizontal_grid

This test file imports the actual functions from combine_CPR_MSI.py (requires Python 3.10+)
"""

import sys
import os
import xarray as xr
import numpy as np
import pytest

# Add the functions directory to the path so we can import the actual functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "functions"))

try:
    # Import the actual functions from combine_CPR_MSI.py
    from combine_CPR_MSI import merge_colocated, xr_vectorized_height_interpolation, align_xmet_horizontal_grid, interpolate1d_height_grid

    FUNCTIONS_AVAILABLE = True
    print("✓ Successfully imported functions from combine_CPR_MSI.py")
except ImportError as e:
    print(f"✗ Failed to import functions from combine_CPR_MSI.py: {e}")
    FUNCTIONS_AVAILABLE = False


def create_mock_cpr_dataset(n_rays=100, n_bins=50):
    """Create a mock CPR dataset for testing that matches the actual structure"""
    # Create latitude/longitude arrays
    lat_vals = np.linspace(-60, 60, n_rays).astype(np.float64)
    lon_vals = np.linspace(-180, 180, n_rays).astype(np.float64)

    # Create proper datetime64 timestamps for profileTime
    base_time = np.datetime64("2025-01-01T00:00:00")
    time_offsets = np.arange(n_rays) * np.timedelta64(1, "s")  # 1 second increments
    profile_times = base_time + time_offsets

    data_vars = {
        "radarReflectivityFactor": (["nray", "nbin"], np.random.normal(-10, 20, (n_rays, n_bins)).astype(np.float32)),
        "dopplerVelocity": (["nray", "nbin"], np.random.normal(0, 5, (n_rays, n_bins)).astype(np.float32)),
        "binHeight": (["nray", "nbin"], np.tile(np.linspace(20000, 0, n_bins), (n_rays, 1)).astype(np.float32)),
        "latitude": (["nray"], lat_vals),
        "longitude": (["nray"], lon_vals),
        "profileTime": (["nray"], profile_times),
        "surfaceBinNumber": (["nray"], np.random.randint(40, 48, n_rays).astype(np.int16)),
        # Add dBZ for testing
        "dBZ": (["nray", "nbin"], np.random.normal(-20, 15, (n_rays, n_bins)).astype(np.float32)),
    }

    coords = {"nray": np.arange(n_rays), "nbin": np.arange(n_bins)}
    ds = xr.Dataset(data_vars, coords=coords)
    ds = ds.set_coords(["latitude", "longitude", "profileTime", "binHeight"])

    # Set the profileTime index as in the actual function
    ds = ds.set_xindex(["profileTime"])

    return ds


def create_mock_msi_dataset(n_pixels=100, n_channels=7, n_across_track=533):
    """Create a mock MSI dataset for testing with proper across_track size"""
    # Create time series that overlaps with CPR profileTime (using datetime64)
    base_time = np.datetime64("2025-01-01T00:00:00")
    time_offsets = np.arange(n_pixels) * np.timedelta64(1, "s")  # 1 second increments
    time_values = base_time + time_offsets

    data_vars = {
        "pixel_values": (
            ["along_track", "across_track", "band"],
            np.random.normal(280, 20, (n_pixels, n_across_track, n_channels)).astype(np.float32),
        ),
        "latitude": (["along_track", "across_track"], np.random.uniform(-60, 60, (n_pixels, n_across_track)).astype(np.float64)),
        "longitude": (["along_track", "across_track"], np.random.uniform(-180, 180, (n_pixels, n_across_track)).astype(np.float64)),
        "time": (["along_track"], time_values),
    }

    coords = {"along_track": np.arange(n_pixels), "across_track": np.arange(n_across_track), "band": np.arange(n_channels)}

    ds = xr.Dataset(data_vars, coords=coords)
    ds = ds.set_coords(["latitude", "longitude", "time"])
    # Set the time index that merge_colocated expects
    ds = ds.set_xindex(["time"])

    return ds


def create_mock_xmet_dataset(n_horizontal=200, n_height=50):
    """Create a mock XMET dataset for testing align_xmet_horizontal_grid"""
    # Create coordinate grids
    lat_grid = np.linspace(-60, 60, int(np.sqrt(n_horizontal)))
    lon_grid = np.linspace(-180, 180, int(np.sqrt(n_horizontal)))
    lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid)

    # Flatten for horizontal_grid dimension
    lat_flat = lat_2d.flatten()[:n_horizontal]
    lon_flat = lon_2d.flatten()[:n_horizontal]

    # Pad if needed
    if len(lat_flat) < n_horizontal:
        lat_flat = np.pad(lat_flat, (0, n_horizontal - len(lat_flat)), mode="edge")
        lon_flat = np.pad(lon_flat, (0, n_horizontal - len(lon_flat)), mode="edge")

    data_vars = {
        "temperature": (["horizontal_grid", "height"], np.random.normal(250, 30, (n_horizontal, n_height)).astype(np.float32)),
        "humidity": (["horizontal_grid", "height"], np.random.uniform(0, 100, (n_horizontal, n_height)).astype(np.float32)),
        "geometrical_height": (["height"], np.linspace(0, 20000, n_height).astype(np.float32)),
        "latitude": (["horizontal_grid"], lat_flat.astype(np.float64)),
        "longitude": (["horizontal_grid"], lon_flat.astype(np.float64)),
    }

    coords = {"horizontal_grid": np.arange(n_horizontal), "height": np.arange(n_height)}

    ds = xr.Dataset(data_vars, coords=coords)
    ds = ds.set_coords(["latitude", "longitude", "geometrical_height"])

    return ds


# Pytest fixtures
@pytest.fixture
def cpr_dataset():
    """Fixture for CPR dataset"""
    return create_mock_cpr_dataset(n_rays=20, n_bins=30)


@pytest.fixture
def msi_dataset():
    """Fixture for MSI dataset"""
    return create_mock_msi_dataset(n_pixels=20, n_channels=5, n_across_track=533)


@pytest.fixture
def xmet_dataset():
    """Fixture for XMET dataset"""
    return create_mock_xmet_dataset(n_horizontal=100, n_height=25)


# Test functions
@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
def test_merge_colocated_function(cpr_dataset, msi_dataset):
    """Test the merge_colocated function from combine_CPR_MSI.py"""
    # Test the merge function with real imported function
    merged = merge_colocated(cpr_dataset, msi_dataset)

    # Verify the merged dataset contains both CPR and MSI data
    assert "radarReflectivityFactor" in merged.data_vars, "CPR data missing from merged dataset"
    assert "pixel_values" in merged.data_vars, "MSI data missing from merged dataset"

    # Check that coordinates are present
    assert "latitude" in merged.coords or "latitude" in merged.data_vars, "Latitude missing"
    assert "longitude" in merged.coords or "longitude" in merged.data_vars, "Longitude missing"


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
def test_xr_vectorized_height_interpolation_function(cpr_dataset):
    """Test the xr_vectorized_height_interpolation function"""
    # Create a height grid for interpolation
    height_grid = np.linspace(1000, 15000, 20).astype(np.float32)

    # Test interpolating dBZ to height grid
    result = xr_vectorized_height_interpolation(
        ds=cpr_dataset,
        height_name="binHeight",
        variable_name="dBZ",
        height_grid=height_grid,
        height_dim="nbin",
        new_height_dim="height_grid",
    )

    # Verify results
    assert "height_grid" in result.dims, "New height dimension not created"
    assert len(result.height_grid) == len(height_grid), "Height grid size mismatch"
    assert result.sizes["nray"] == cpr_dataset.sizes["nray"], "Ray dimension changed unexpectedly"

    # Check that height_grid coordinate is set correctly
    np.testing.assert_array_equal(result.height_grid.values, height_grid)


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
def test_interpolate1d_height_grid_function():
    """Test the interpolate1d_height_grid function directly"""
    # Test data
    height_sample = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.float32)
    variable_sample = np.array([10, 8, 6, 4, 2], dtype=np.float32)
    height_grid = np.array([1500, 2500, 3500, 4500], dtype=np.float32)

    # Test interpolation
    result = interpolate1d_height_grid(height_sample, variable_sample, height_grid)

    # Verify results
    assert len(result) == len(height_grid), "Output length doesn't match height_grid"
    assert not np.isnan(result).all(), "All results are NaN"

    # Test with expected interpolated values (approximately)
    expected_1500 = 9.0  # Between 10 and 8
    assert abs(result[0] - expected_1500) < 1.0, "Interpolation result unexpected"


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
def test_align_xmet_horizontal_grid_function(cpr_dataset, xmet_dataset):
    """Test the align_xmet_horizontal_grid function"""
    # Create a smaller dataset for testing to avoid dimension mismatch
    cpr_small = cpr_dataset.isel(nray=slice(0, 5))

    # Create the proper index structure that align_xmet_horizontal_grid expects
    cpr_small = cpr_small.set_xindex(["latitude", "longitude"])
    xmet_dataset = xmet_dataset.set_xindex(["latitude", "longitude"])

    # Test the alignment function
    aligned = align_xmet_horizontal_grid(xmet_dataset, cpr_small)

    # Verify the result
    assert isinstance(aligned, xr.Dataset), "Result should be xarray Dataset"
    assert "horizontal_grid" in aligned.dims, "horizontal_grid dimension should exist"

    # Check that required variables are still present
    assert "temperature" in aligned.data_vars, "Temperature variable missing after alignment"
    assert "latitude" in aligned.coords or "latitude" in aligned.data_vars, "Latitude missing after alignment"


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
def test_height_interpolation_with_mock_data(cpr_dataset):
    """Test height interpolation with realistic mock CPR data"""
    # Create a realistic height grid (1km to 15km, 100m resolution)
    height_grid = np.arange(1000, 15000, 100).astype(np.float32)

    # Test interpolating radar reflectivity factor
    result = xr_vectorized_height_interpolation(
        ds=cpr_dataset,
        height_name="binHeight",
        variable_name="radarReflectivityFactor",
        height_grid=height_grid,
        height_dim="nbin",
        new_height_dim="height_interpolated",
    )

    # Verify the interpolation worked
    assert "height_interpolated" in result.dims
    assert len(result.height_interpolated) == len(height_grid)

    # Check that we have reasonable data (not all NaN)
    non_nan_fraction = (~np.isnan(result.values)).sum() / result.size
    assert non_nan_fraction > 0.1, "Too much data lost in interpolation"


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
def test_merge_colocated_with_time_matching(cpr_dataset, msi_dataset):
    """Test merge_colocated with proper time coordinate handling"""
    # The MSI dataset already has a time index, so no need to set it again

    # Test the merge function
    merged = merge_colocated(cpr_dataset, msi_dataset)

    # Basic checks
    assert isinstance(merged, xr.Dataset), "Result should be xarray Dataset"

    # Check that both datasets' variables are present
    cpr_vars = set(cpr_dataset.data_vars.keys())
    msi_vars = set(msi_dataset.data_vars.keys())
    merged_vars = set(merged.data_vars.keys())

    # At least some variables from each dataset should be present
    cpr_present = len(cpr_vars.intersection(merged_vars)) > 0
    msi_present = len(msi_vars.intersection(merged_vars)) > 0

    assert cpr_present, "No CPR variables found in merged dataset"
    assert msi_present, "No MSI variables found in merged dataset"


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason="Functions not available")
def test_function_integration():
    """Test integration of multiple functions together"""
    # Create datasets
    cpr = create_mock_cpr_dataset(n_rays=10, n_bins=20)
    msi = create_mock_msi_dataset(n_pixels=10, n_channels=3, n_across_track=533)
    xmet = create_mock_xmet_dataset(n_horizontal=50, n_height=15)

    # Test 1: Merge CPR and MSI
    merged = merge_colocated(cpr, msi)
    assert isinstance(merged, xr.Dataset), "Merge failed"

    # Test 2: Interpolate height data
    height_grid = np.linspace(2000, 18000, 10).astype(np.float32)
    interpolated = xr_vectorized_height_interpolation(
        ds=cpr, height_name="binHeight", variable_name="dBZ", height_grid=height_grid, height_dim="nbin"
    )
    assert "height_grid" in interpolated.dims, "Height interpolation failed"

    # Test 3: Align XMET data
    aligned_xmet = align_xmet_horizontal_grid(xmet.set_xindex(["latitude", "longitude"]), cpr.set_xindex(["latitude", "longitude"]))
    assert isinstance(aligned_xmet, xr.Dataset), "XMET alignment failed"


if __name__ == "__main__":
    # Run tests manually if not using pytest
    print("Running manual tests for combine_CPR_MSI functions...")

    print(f"Functions available: {FUNCTIONS_AVAILABLE}")

    # Create test datasets
    cpr = create_mock_cpr_dataset(n_rays=5, n_bins=10)
    msi = create_mock_msi_dataset(n_pixels=5, n_channels=3, n_across_track=533)
    xmet = create_mock_xmet_dataset(n_horizontal=20, n_height=8)

    print(f"CPR dataset: {dict(cpr.dims)}")
    print(f"MSI dataset: {dict(msi.dims)}")
    print(f"XMET dataset: {dict(xmet.dims)}")

    # Test basic functionality
    try:
        merged = merge_colocated(cpr, msi)
        print("✓ merge_colocated test passed")
    except Exception as e:
        print(f"✗ merge_colocated test failed: {e}")

    try:
        height_grid = np.linspace(5000, 15000, 5).astype(np.float32)
        interpolated = xr_vectorized_height_interpolation(cpr, "binHeight", "dBZ", height_grid, "nbin")
        print("✓ xr_vectorized_height_interpolation test passed")
    except Exception as e:
        print(f"✗ xr_vectorized_height_interpolation test failed: {e}")

    try:
        aligned = align_xmet_horizontal_grid(xmet.set_xindex(["latitude", "longitude"]), cpr.set_xindex(["latitude", "longitude"]))
        print("✓ align_xmet_horizontal_grid test passed")
    except Exception as e:
        print(f"✗ align_xmet_horizontal_grid test failed: {e}")

    print("Manual tests completed")
