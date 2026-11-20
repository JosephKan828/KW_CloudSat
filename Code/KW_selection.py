"""Select Kelvin-wave events from OLR using netCDF4, NumPy, and SciPy.

Edit the settings below, then run:
    python Code/KW_selection_netcdf4.py

The input must contain daily OLR(time, lat, lon), paired latitudes across
the equator, and an ascending, evenly spaced global longitude grid.
Event CSVs contain indices into the selected dates and input longitudes.
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np
from netCDF4 import Dataset, date2num, num2date
from scipy.ndimage import minimum_filter

# Save plots without opening a window.
matplotlib.use("Agg")
from matplotlib import pyplot as plt


# ====================================================
# 1. Settings: edit these values for your analysis
# ====================================================

project_dir = Path(__file__).resolve().parents[1]
olr_data_path = Path("/work/DATA/Satellite/OLR/olr_anomaly.nc")
events_dir = project_dir / "Files" / "KW_events_netcdf4"
figures_dir = project_dir / "Figure" / "KW_olr_netcdf4"

start_date = datetime(2006, 1, 1)
end_date = datetime(2017, 12, 31)  # Include this entire day.
latitude_bound = 5.0            # Select 5S to 5N.

k_ranges = [(1, 13), (1, 3), (3, 5), (5, 7), (7, 9), (9, 11), (11, 13)]
period_min = 2.5                # days
period_max = 30.0               # days
depth_min = 8.0                 # equivalent depth, m
depth_max = 90.0                # equivalent depth, m

sigma_multiplier = 2.77         # Event threshold = -2.77 * filtered OLR std.
time_window = 7                 # Search +/-3 days for a local minimum.
longitude_half_width = 15.0     # degrees; use 9.375 for the old 31-point window.
lag_days = 10                   # Composite from -10 to +10 days.
make_plots = True

earth_radius = 6.37122e6        # m
gravity = 9.80665               # m/s^2
seconds_per_day = 86400.0


# ====================================================
# 2. Load OLR data
# ====================================================

if start_date > end_date or not 0 < latitude_bound <= 90:
    raise ValueError("Check the date range and latitude bound.")
if not (2 < period_min <= period_max and 0 < depth_min <= depth_max):
    raise ValueError("Periods must exceed 2 days; depth bounds must be positive.")
if sigma_multiplier <= 0 or time_window < 1 or time_window % 2 == 0:
    raise ValueError("Use a positive sigma multiplier and a positive odd time window.")
if not 0 <= longitude_half_width < 180 or lag_days < 0:
    raise ValueError("Check the longitude search radius and composite lag range.")

with Dataset(olr_data_path) as dataset:
    time_var = dataset.variables["time"]
    time_all = np.ma.asarray(time_var[:], dtype=float).filled(np.nan)
    lat_all = np.ma.asarray(dataset.variables["lat"][:], dtype=float).filled(np.nan)
    lon = np.ma.asarray(dataset.variables["lon"][:], dtype=float).filled(np.nan)
    calendar = getattr(time_var, "calendar", "standard")

    if not (np.isfinite(time_all).all() and np.all(np.diff(time_all) > 0)):
        raise ValueError("Time coordinates must be finite and strictly increasing.")
    if not (np.isfinite(lat_all).all() and
            (np.all(np.diff(lat_all) > 0) or np.all(np.diff(lat_all) < 0))):
        raise ValueError("Latitude coordinates must be finite and monotonic.")

    # Convert the requested dates to the file's time units.
    time_start = date2num(start_date, time_var.units, calendar=calendar)
    time_stop = date2num(end_date + timedelta(days=1), time_var.units, calendar=calendar)
    selected_time = np.where((time_all >= time_start) & (time_all < time_stop))[0]
    selected_lat = np.where(np.abs(lat_all) <= latitude_bound)[0]

    if selected_time.size < 2 or selected_lat.size == 0:
        raise ValueError("The selection needs at least two dates and one latitude.")

    # Both selections are contiguous, so read only this slab of the file.
    time_slice = slice(selected_time[0], selected_time[-1] + 1)
    lat_slice = slice(selected_lat[0], selected_lat[-1] + 1)
    time = np.asarray(num2date(time_all[time_slice], time_var.units, calendar=calendar))
    lat = lat_all[lat_slice]

    olr_var = dataset.variables["olr"]
    if olr_var.dimensions != ("time", "lat", "lon"):
        raise ValueError("Expected OLR dimensions in the order (time, lat, lon).")
    olr = olr_var[time_slice, lat_slice, :]
    olr = np.ma.asarray(olr, dtype=np.float64).filled(np.nan)
    olr_units = getattr(olr_var, "units", "W/m^2")

# FFT filtering requires finite data and a regular time-longitude grid.
if not np.isfinite(olr).all():
    raise ValueError("OLR contains missing/non-finite values; fill gaps before filtering.")
time_steps = np.array([(right - left).total_seconds() / seconds_per_day
                       for left, right in zip(time[:-1], time[1:])])
if not np.allclose(time_steps, 1.0, atol=1e-8, rtol=0):
    raise ValueError("OLR must have daily sampling without missing dates.")
if not np.allclose(lat, -lat[::-1], atol=1e-6, rtol=0):
    raise ValueError("Latitudes must form matching pairs across the equator.")
if lon.size < 4 or not np.isfinite(lon).all():
    raise ValueError("The longitude grid needs at least four finite points.")
longitude_spacing = lon[1] - lon[0]
if (longitude_spacing <= 0
        or not np.allclose(np.diff(lon), longitude_spacing, atol=1e-6, rtol=0)
        or not np.isclose(lon.size * longitude_spacing, 360, atol=1e-5, rtol=0)):
    raise ValueError("Longitude must be evenly spaced, ascending, and cover 360 degrees.")

n_time = time.size
n_lon = lon.size
for k_min, k_max in k_ranges:
    if not 1 <= k_min <= k_max < n_lon / 2:
        raise ValueError("Each band must satisfy 1 <= k_min <= k_max < n_lon/2.")

# Convert a search radius in degrees to an odd number of grid points.
half_width_points = int(np.floor(longitude_half_width / longitude_spacing + 0.5))
longitude_window = 2 * half_width_points + 1
if longitude_window > n_lon:
    raise ValueError("The longitude search window is larger than the grid.")
print(f"Loaded {n_time} days, {lat.size} latitudes, {n_lon} longitudes.", flush=True)
print(f"Minimum search: {time_window} days x {longitude_window} longitude points.", flush=True)


# ====================================================
# 3. Remove the background and symmetrize across the equator
# ====================================================

# Subtract one time-longitude mean at each latitude.
olr -= olr.mean(axis=(0, 2), keepdims=True)

# Reversing the latitude axis pairs each latitude with its opposite latitude.
olr_symm = (olr + olr[:, ::-1, :]) / 2
olr_raw = olr_symm.mean(axis=1)  # Shape: (time, longitude).
del olr, olr_symm              # Only the latitude mean is needed below.


# ====================================================
# 4. Transform OLR into frequency-wavenumber space
# ====================================================

# Use FFT over time and IFFT over longitude, as in the original script.
# With this convention, eastward waves have frequency and wavenumber
# of the same sign. The FFT assumes the time record is periodic.
spectrum = np.fft.fft(olr_raw, axis=0)
spectrum = np.fft.ifft(spectrum, axis=1) * n_lon

frequency = np.fft.fftfreq(n_time, d=1.0)              # cycles/day
wavenumber = np.rint(np.fft.fftfreq(n_lon, d=1 / n_lon))
wavenumber_grid, frequency_grid = np.meshgrid(wavenumber, frequency)
abs_frequency = np.abs(frequency_grid)
abs_wavenumber = np.abs(wavenumber_grid)

# Kelvin dispersion relation: omega = c*k, where c = sqrt(g*h).
# Convert planetary wavenumber to frequency in cycles/day.
speed_min = np.sqrt(gravity * depth_min)
speed_max = np.sqrt(gravity * depth_max)
earth_circumference = 2 * np.pi * earth_radius
dispersion_min = speed_min * abs_wavenumber * seconds_per_day / earth_circumference
dispersion_max = speed_max * abs_wavenumber * seconds_per_day / earth_circumference

events_dir.mkdir(parents=True, exist_ok=True)


# ====================================================
# 5. Filter, select events, and plot each wavenumber band
# ====================================================

for k_min, k_max in k_ranges:
    band_name = f"k={k_min}~{k_max}"

    # Keep eastward waves within the wavenumber, period, and depth bounds.
    # Absolute values apply identical bounds to conjugate spectral pairs.
    kelvin_mask = (
        (frequency_grid * wavenumber_grid > 0)
        & (abs_wavenumber >= k_min) & (abs_wavenumber <= k_max)
        & (abs_frequency >= 1 / period_max) & (abs_frequency <= 1 / period_min)
        & (abs_frequency >= dispersion_min) & (abs_frequency <= dispersion_max)
    )
    filtered_spectrum = spectrum * kelvin_mask

    # Reverse the transforms to recover filtered OLR(time, longitude).
    kw_recon = np.fft.ifft(filtered_spectrum, axis=0)
    kw_recon = np.fft.fft(kw_recon, axis=1) / n_lon
    kw_recon = kw_recon.real

    # An event must be a local minimum AND fall below the amplitude threshold.
    olr_std = float(kw_recon.std())
    threshold = -sigma_multiplier * olr_std
    local_minimum = minimum_filter(
        kw_recon,
        size=(time_window, longitude_window),
        mode=("constant", "wrap"),  #type: ignore # Stop at time boundaries; wrap longitude.
        cval=np.inf,
    )
    is_event = (kw_recon == local_minimum) & (kw_recon <= threshold) & (olr_std > 0)
    time_idx, lon_idx = np.where(is_event)
    # Equal minima are retained. No wave tracking or extra clustering is applied.

    np.savetxt(
        events_dir / f"{band_name}.csv",
        np.column_stack((time_idx, lon_idx)),
        fmt="%d", delimiter=",", header="time_idx,lon_idx", comments="",
    )
    print(f"{band_name}: std={olr_std:.4f}, threshold={threshold:.4f}, "
          f"events={time_idx.size}", flush=True)

    # Composite at each event's fixed longitude, with no wrapping in time.
    lags = np.arange(-lag_days, lag_days + 1)
    composite_filtered = np.full(lags.size, np.nan)
    composite_raw = np.full(lags.size, np.nan)
    events_per_lag = np.zeros(lags.size, dtype=int)

    for i, lag in enumerate(lags):
        sample_time = time_idx + lag
        valid = (sample_time >= 0) & (sample_time < n_time)
        events_per_lag[i] = np.count_nonzero(valid)
        if events_per_lag[i] > 0:
            composite_filtered[i] = kw_recon[sample_time[valid], lon_idx[valid]].mean()
            composite_raw[i] = olr_raw[sample_time[valid], lon_idx[valid]].mean()

    if not make_plots:
        continue

    figure_dir = figures_dir / band_name
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: longitude-time reconstruction for the first 100 days.
    n_plot_days = min(100, n_time)
    fig, ax = plt.subplots(figsize=(7, 9))
    pcm = ax.pcolormesh(lon, np.arange(n_plot_days), kw_recon[:n_plot_days],
                        cmap="RdBu_r", shading="auto")
    ax.set_xlim(lon[0] - longitude_spacing / 2, lon[-1] + longitude_spacing / 2)
    ax.set_xlabel("Longitude [degrees east]")
    ax.set_ylabel(f"Days after {time[0].isoformat()}")
    ax.set_title(f"Kelvin OLR reconstruction ({band_name})")
    fig.colorbar(pcm, ax=ax, label=f"OLR anomaly [{olr_units}]")
    fig.savefig(figure_dir / "reconstruct.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: filtered OLR distribution and event threshold.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(kw_recon.ravel(), bins=50, density=True, color="#0072B2",
            edgecolor="black", alpha=0.75)
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.2f}")
    ax.set_xlabel(f"Kelvin OLR anomaly [{olr_units}]")
    ax.set_ylabel("Density")
    ax.set_title(f"Filtered OLR distribution ({band_name})")
    ax.legend()
    fig.savefig(figure_dir / "histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: filtered and raw OLR around the selected event centers.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(lags, composite_filtered, color="#0072B2", label="filtered")
    ax.plot(lags, composite_raw, color="#D55E00", label="raw")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag [days]")
    ax.set_ylabel(f"Composite OLR anomaly [{olr_units}]")
    ax.set_title(f"Kelvin OLR composite ({time_idx.size} events)")
    ax.legend()
    fig.savefig(figure_dir / "composite.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
