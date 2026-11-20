"""Fit radiative-heating Jacobians for vertical velocity and specific humidity.

Converted from select_valid_column.ipynb. Edit the settings below and run:
    python Code/select_valid_column.py

Event indices refer to the OLR subset between event_start and event_end.
An OLR longitude INDEX is first converted to an OLR longitude VALUE, then
mapped to the nearest ERA5 and CloudSat grid points using cyclic distance.

Fit separate LW/SW Ridge models for w and Q; these are not a joint w+Q fit.
Q stays in its native kg/kg units. Ridge(alpha=1, solver="svd") and the
saved-composite verification follow the notebook. Verification uses
J @ predictor_composite, without the fitted intercept. Results are saved in
separate w/k=... and Q/k=... directories under Files/Jacobian and Figure/Jacobian.
The supplied composite files may represent a different event catalogue;
these diagnostics are not an independent validation on held-out events.
"""

# Match the notebook's single-thread numerical-library settings.
import os
for thread_setting in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[thread_setting] = "1"

import csv
from datetime import datetime
from pathlib import Path

import matplotlib
import netCDF4 as nc
import numpy as np
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator


# ====================================================
# 1. Settings
# ====================================================

project_dir = Path(__file__).resolve().parents[1]
k_min, k_max = 1, 13
band_name = f"k={k_min}~{k_max}"
event_file = project_dir / "Files" / "KW_events" / f"{band_name}.csv"

olr_file = Path("/work/DATA/Satellite/OLR/olr_anomaly.nc")
w_file = Path("/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10/w.nc")
humidity_file = Path("/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10/Q.nc")
heating_file = Path("/data92/b11209013/CloudSat/DATA/QR_gridded_15layer_-5_5.nc")

# These dates must match the OLR subset used to create the event CSV.
event_start = np.datetime64("2006-01-01", "D")
event_end = np.datetime64("2017-12-31", "D")
latitude_bound = 5.0
ridge_alpha = 1.0  # Applied in each predictor's native units; Q is not rescaled.

# Retain the notebook's verification against existing saved composites.
composite_dir = project_dir / "Files"
output_root = project_dir / "Files" / "Jacobian"
figure_root = project_dir / "Figure" / "Jacobian"
comparison_window = (-50.0, 50.0)  # relative longitude, degrees


# ====================================================
# 2. Functions used for more than one dataset
# ====================================================

def calendar_days(time_variable):
    """Read daily dates from either CF time or ERA5's YYYYMMDD.fraction."""
    values = np.asarray(time_variable[:])
    if time_variable.units.startswith("day as %Y%m%d"):
        dates = [datetime.strptime(str(int(value)), "%Y%m%d") for value in values]
    else:
        dates = nc.num2date(values, time_variable.units,
                           calendar=getattr(time_variable, "calendar", "standard"))
    return np.array([date.strftime("%Y-%m-%d") for date in dates], dtype="datetime64[D]")


def nearest_longitude(event_longitudes, target_longitudes):
    """Return nearest grid indices, including across the 0/360-degree boundary."""
    distance = np.abs(
        (target_longitudes[None, :] - event_longitudes[:, None] + 180.0) % 360.0 - 180.0
    )
    return np.argmin(distance, axis=1)


def event_anomalies(variable, time_slice, latitude_slice, event_times, event_longitudes):
    """Remove the time-zonal mean and center each event on the longitude axis.

    Read one pressure level at a time to avoid loading all 12 years and all
    15 levels at once. Return (level, relative longitude, event, latitude),
    the same arrangement as the notebook's selected arrays.
    """
    n_level = variable.shape[1]
    n_lat = latitude_slice.stop - latitude_slice.start
    n_lon = variable.shape[-1]
    selected = np.empty((n_level, n_lon, event_times.size, n_lat), dtype=np.float32)
    for level in range(n_level):
        print(f"Reading {variable.name}: level {level + 1}/{n_level}", flush=True)
        field = np.ma.asarray(variable[time_slice, level, latitude_slice, :],
                              dtype=np.float32).filled(np.nan)
        field -= np.nanmean(field, axis=(0, 2), keepdims=True)
        for event, (time_index, lon_index) in enumerate(zip(event_times, event_longitudes)):
            frame = field[time_index - time_slice.start]
            centered = np.roll(frame, shift=n_lon // 2 - lon_index, axis=-1)
            selected[level, :, event, :] = centered.T
    return selected


def finite_array(values):
    """Keep masked NetCDF values missing when converting to a float array."""
    return np.ma.asarray(values, dtype=float).filled(np.nan)

def calculate_pattern_metrics(prediction, composite):
    """Centered pattern r and standard-deviation ratio on finite paired values."""
    prediction, composite = finite_array(prediction), finite_array(composite)
    if prediction.shape != composite.shape:
        raise ValueError("Prediction and composite must have matching shapes")
    valid = np.isfinite(prediction) & np.isfinite(composite)
    predicted, observed = prediction[valid], composite[valid]
    result = {"r": np.nan, "amplitude_ratio": np.nan, "n_pairs": int(valid.sum())}
    if predicted.size < 2 or np.all(observed == observed[0]):
        return result
    if np.all(predicted == predicted[0]):
        result["amplitude_ratio"] = 0.0
        return result
    predicted_centered = predicted - predicted.mean()
    observed_centered = observed - observed.mean()
    predicted_norm = np.linalg.norm(predicted_centered)
    observed_norm = np.linalg.norm(observed_centered)
    result["amplitude_ratio"] = float(predicted_norm / observed_norm)
    result["r"] = float(np.clip(
        np.dot(predicted_centered / predicted_norm, observed_centered / observed_norm),
        -1.0, 1.0,
    ))
    return result

def calculate_comparison_metrics(prediction, composite):
    """Use identical pairs for pattern correlation, R2, RMSE, and mean bias."""
    result = calculate_pattern_metrics(prediction, composite)
    prediction, composite = finite_array(prediction), finite_array(composite)
    valid = np.isfinite(prediction) & np.isfinite(composite)
    predicted, observed = prediction[valid], composite[valid]
    result.update(r2=np.nan, rmse=np.nan, bias=np.nan)
    if observed.size:
        difference = predicted - observed
        result["rmse"] = float(np.sqrt(np.mean(difference**2)))
        result["bias"] = float(difference.mean())
        if observed.size >= 2 and np.any(observed != observed[0]):
            result["r2"] = float(1.0 - np.sum(difference**2)
                                 / np.sum((observed - observed.mean())**2))
    return result

def metric_label(value, digits=3):
    return f"{value:.{digits}f}" if np.isfinite(value) else "undefined"

def symmetric_limit(*fields):
    """Round the largest finite magnitude up without clipping any data."""
    peak = max((np.max(np.abs(a[np.isfinite(a)]))
                for field in fields
                if np.isfinite(a := finite_array(field)).any()), default=0.0)
    if peak == 0.0:
        return 1.0
    step = 10.0 ** np.floor(np.log10(peak)) / 2.0
    return float(np.ceil(peak / step) * step)

def save_verification_figure(fig, stem):
    for extension in ("png", "pdf"):
        fig.savefig(figure_dir / f"{stem}.{extension}", bbox_inches="tight",
                    facecolor="white")

def plot_prediction_verification(prediction, composite, component):
    """Overlay predicted shading and observed contours, with mean profiles."""
    prediction, composite = finite_array(prediction), finite_array(composite)
    if prediction.shape != composite.shape or prediction.shape != (pressure_hpa.size, relative_lon.size):
        raise ValueError("Heating fields must match the pressure and longitude coordinates")
    longitude = relative_lon[lon_window]
    predicted, observed = prediction[:, lon_window], composite[:, lon_window]
    metrics = calculate_pattern_metrics(predicted, observed)

    fig = plt.figure(figsize=(14.8, 6.3), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.045),
                           width_ratios=(3.6, 1.15), hspace=0.12)
    ax_overlay = fig.add_subplot(grid[0, 0])
    ax_mean = fig.add_subplot(grid[0, 1], sharey=ax_overlay)

    mesh = ax_overlay.pcolormesh(
        longitude, pressure_hpa, np.ma.masked_invalid(predicted),
        cmap=heating_cmap, norm=heating_norm, shading="nearest", rasterized=True,
    )
    observed_for_contours = observed[:, ::3]
    finite_observed = observed_for_contours[np.isfinite(observed_for_contours)]
    observed_limit = symmetric_limit(observed)
    contour_levels = MaxNLocator(nbins=8, symmetric=True, steps=[1, 2, 5, 10]).tick_values(-observed_limit, observed_limit)
    contour_levels = contour_levels[contour_levels != 0.0]
    if finite_observed.size:
        contour_levels = contour_levels[
            (contour_levels > finite_observed.min()) & (contour_levels < finite_observed.max())
        ]
        if contour_levels.size and observed_for_contours.shape[1] >= 2:
            contours = ax_overlay.contour(
                longitude[::3], pressure_hpa, np.ma.masked_invalid(observed_for_contours),
                levels=contour_levels, colors="#202020", linewidths=1.15,
                linestyles=["dashed" if level < 0 else "solid" for level in contour_levels],
            )
            ax_overlay.clabel(contours, inline=True, fontsize=12, fmt="%.2f", inline_spacing=5)
    ax_overlay.axvline(0.0, color="#475467", ls=":", lw=1.0, alpha=0.8)
    ax_overlay.set(xlim=comparison_window, xlabel="Relative longitude (°)", ylabel="Pressure (hPa)")
    ax_overlay.set_xticks([-50, -25, 0, 25, 50])
    ax_overlay.set_title("(a) Prediction (shading), observation (contours)", pad=12, fontsize=16)
    ax_overlay.text(
        0.025, 0.04, f"Pattern r = {metric_label(metrics['r'])}", transform=ax_overlay.transAxes,
        fontsize=15, color="#243447", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#D0D5DD", alpha=0.94),
    )

    # Compare amplitudes using exactly the same finite longitudes at each level.
    valid = np.isfinite(predicted) & np.isfinite(observed)
    count = valid.sum(axis=1)
    predicted_mean = np.divide(np.where(valid, predicted, 0.0).sum(axis=1), count,
                               out=np.full(pressure_hpa.size, np.nan), where=count > 0)
    observed_mean = np.divide(np.where(valid, observed, 0.0).sum(axis=1), count,
                              out=np.full(pressure_hpa.size, np.nan), where=count > 0)
    ax_mean.axvline(0.0, color="#98A2B3", ls=":", lw=1.0)
    ax_mean.fill_betweenx(pressure_hpa, observed_mean, predicted_mean, color="#98A2B3", alpha=0.2)
    ax_mean.plot(observed_mean, pressure_hpa, color=observed_color, lw=2,
                 marker="o", ms=3, label="Observed")
    ax_mean.plot(predicted_mean, pressure_hpa, color=predicted_color, lw=2,
                 ls="--", marker="s", ms=3, label="Predicted")
    ax_mean.set_title("(b) Longitude mean", pad=12, fontsize=16)
    ax_mean.set_xlabel(f"Heating ({heating_units})")
    ax_mean.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax_mean.grid(axis="both", color="#D0D5DD", lw=0.6, alpha=0.55)
    ax_mean.set_axisbelow(True)
    ax_mean.tick_params(labelleft=False)
    for ax in (ax_overlay, ax_mean):
        ax.set_ylim(pressure_hpa.max(), pressure_hpa.min())
        ax.set_yticks(np.arange(100, 1001, 100))

    legend_ax = fig.add_subplot(grid[1, 1])
    legend_ax.set_axis_off()
    handles, labels = ax_mean.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=2, columnspacing=1.0, handlelength=2.1)
    colorbar = fig.colorbar(mesh, cax=fig.add_subplot(grid[1, 0]), orientation="horizontal")
    colorbar.set_label(f"Predicted heating-rate anomaly ({heating_units})", labelpad=5)
    colorbar.locator = MaxNLocator(nbins=7, symmetric=True)
    colorbar.update_ticks()
    colorbar.outline.set_visible(False)
    save_verification_figure(fig, component)
    return fig


# ====================================================
# 3. Read event indices and map them to the physical data grids
# ====================================================

events = np.genfromtxt(event_file, delimiter=",", names=True, dtype=np.int64, ndmin=1)
if events.size == 0 or events.dtype.names != ("time_idx", "lon_idx"):
    raise ValueError("The event CSV must contain time_idx,lon_idx and at least one event.")
olr_time_idx = events["time_idx"]
olr_lon_idx = events["lon_idx"]

with nc.Dataset(olr_file) as dataset:
    olr_days = calendar_days(dataset.variables["time"])
    olr_days = olr_days[(olr_days >= event_start) & (olr_days <= event_end)]
    olr_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
if (np.any(olr_time_idx < 0) or np.any(olr_time_idx >= olr_days.size)
        or np.any(olr_lon_idx < 0) or np.any(olr_lon_idx >= olr_lon.size)):
    raise ValueError("Event indices fall outside the selected OLR dates or longitude grid.")
event_days = olr_days[olr_time_idx]
event_lon_values = olr_lon[olr_lon_idx]  # Never apply OLR indices directly to ERA5!

with nc.Dataset(w_file) as dataset:
    w_days = calendar_days(dataset.variables["time"])
    era5_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
    era5_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
    pressure_levels = np.asarray(dataset.variables["plev"][:], dtype=float)
    if dataset.variables["w"].dimensions != ("time", "plev", "lat", "lon"):
        raise ValueError("Expected w(time, plev, lat, lon).")

# Q is on the same ERA5 grid. Check this before reusing the w event mapping.
with nc.Dataset(humidity_file) as dataset:
    if dataset.variables["Q"].dimensions != ("time", "plev", "lat", "lon"):
        raise ValueError("Expected Q(time, plev, lat, lon).")
    if not np.array_equal(calendar_days(dataset.variables["time"]), w_days):
        raise ValueError("Humidity and w must have the same calendar dates.")
    for coordinate, expected in (("lon", era5_lon), ("lat", era5_lat),
                                 ("plev", pressure_levels)):
        actual = np.asarray(dataset.variables[coordinate][:], dtype=float)
        if actual.shape != expected.shape or not np.allclose(actual, expected):
            raise ValueError(f"Humidity and w must share the {coordinate} grid.")

with nc.Dataset(heating_file) as dataset:
    heating_days = calendar_days(dataset.variables["time"])
    heating_lon = np.asarray(dataset.variables["longitude"][:], dtype=float)
    heating_lat = np.asarray(dataset.variables["latitude"][:], dtype=float)
    heating_levels = np.asarray(dataset.variables["lev"][:], dtype=float)
    for name in ("QLW", "QSW"):
        if dataset.variables[name].dimensions != ("time", "lev", "latitude", "longitude"):
            raise ValueError(f"Expected {name}(time, lev, latitude, longitude).")

# OLR may have a different grid. ERA5 and heating columns must share a grid
# with each other, so each regression pair represents the same atmosphere.
w_lat_indices = np.where(np.abs(era5_lat) <= latitude_bound)[0]
heating_lat_indices = np.where(np.abs(heating_lat) <= latitude_bound)[0]
if w_lat_indices.size == 0 or heating_lat_indices.size == 0:
    raise ValueError("No target latitudes fall inside the requested band.")
if (not np.array_equal(np.diff(w_lat_indices), np.ones(w_lat_indices.size - 1))
        or not np.array_equal(np.diff(heating_lat_indices), np.ones(heating_lat_indices.size - 1))):
    raise ValueError("Selected latitude rows must be contiguous.")
if (era5_lon.shape != heating_lon.shape
        or not np.allclose(era5_lon % 360, heating_lon % 360)
        or w_lat_indices.size != heating_lat_indices.size
        or not np.allclose(era5_lat[w_lat_indices], heating_lat[heating_lat_indices])
        or pressure_levels.shape != heating_levels.shape
        or not np.allclose(pressure_levels, heating_levels)):
    raise ValueError("ERA5 and CloudSat must share longitude, latitude, and pressure grids.")

longitude_step = era5_lon[1] - era5_lon[0]
if (longitude_step <= 0 or not np.allclose(np.diff(era5_lon), longitude_step)
        or not np.isclose(longitude_step * era5_lon.size, 360)):
    raise ValueError("Target longitude must be an ascending, evenly spaced global grid.")

# Match calendar dates separately, rather than assuming identical time origins.
w_period = np.where((w_days >= event_start) & (w_days <= event_end))[0]
heating_period = np.where((heating_days >= event_start) & (heating_days <= event_end))[0]
for label, days, selected_period in (
    ("ERA5", w_days, w_period), ("CloudSat", heating_days, heating_period),
):
    if (selected_period.size != olr_days.size
            or not np.array_equal(days[selected_period], olr_days)
            or not np.all(np.diff(days).astype("timedelta64[D]") > np.timedelta64(0, "D"))):
        raise ValueError(f"{label} must cover the OLR analysis dates in increasing order.")

w_time_idx = np.searchsorted(w_days, event_days)
heating_time_idx = np.searchsorted(heating_days, event_days)
w_lon_idx = nearest_longitude(event_lon_values, era5_lon)
heating_lon_idx = nearest_longitude(event_lon_values, heating_lon)
w_time_slice = slice(int(w_period[0]), int(w_period[-1]) + 1)
heating_time_slice = slice(int(heating_period[0]), int(heating_period[-1]) + 1)
w_lat_slice = slice(int(w_lat_indices[0]), int(w_lat_indices[-1]) + 1)
heating_lat_slice = slice(int(heating_lat_indices[0]), int(heating_lat_indices[-1]) + 1)

print(f"Loaded {events.size} events from {event_file}", flush=True)
print(f"First OLR event: index {olr_lon_idx[0]}, {event_lon_values[0]:g} degrees; "
      f"nearest ERA5 index {w_lon_idx[0]}, {era5_lon[w_lon_idx[0]]:g} degrees", flush=True)


# ====================================================
# 4. Read the shared radiative-heating anomalies
# ====================================================

with nc.Dataset(heating_file) as dataset:
    LW_selected = event_anomalies(dataset.variables["QLW"], heating_time_slice,
                                  heating_lat_slice, heating_time_idx, heating_lon_idx)
    SW_selected = event_anomalies(dataset.variables["QSW"], heating_time_slice,
                                  heating_lat_slice, heating_time_idx, heating_lon_idx)
heating_valid = np.all(np.isfinite(LW_selected) & np.isfinite(SW_selected), axis=0)

LW_composite = np.loadtxt(composite_dir / "LW_composite.txt")
SW_composite = np.loadtxt(composite_dir / "SW_composite.txt")
composite_pressure = np.loadtxt(composite_dir / "plev.txt")
expected_shape = (pressure_levels.size, era5_lon.size)
if (LW_composite.shape != expected_shape or SW_composite.shape != expected_shape
        or composite_pressure.shape != pressure_levels.shape
        or not np.allclose(composite_pressure, pressure_levels)):
    raise ValueError("Saved composites must match the target longitude and pressure grids.")

# Consistent typography, colors, and units for all verification figures.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.edgecolor": "#667085",
    "axes.labelcolor": "#243447",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": "#475467",
    "ytick.color": "#475467",
    "legend.fontsize": 14,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

observed_color, predicted_color = "#243447", "#D55E00"
heating_cmap = plt.get_cmap("RdBu_r").copy()
heating_cmap.set_bad("#E9EDF2")
heating_units = r"K day$^{-1}$"

dlon = float(np.abs(np.diff(era5_lon[:2]))[0])
relative_lon = (np.arange(era5_lon.size) - era5_lon.size // 2) * dlon
pressure_hpa = np.asarray(pressure_levels, dtype=float) / 100.0
lon_window = (relative_lon >= comparison_window[0]) & (relative_lon <= comparison_window[1])
if not lon_window.any():
    raise ValueError("No longitude points fall inside the comparison window")


# ====================================================
# 5. Fit and verify w and Q separately
# ====================================================

for predictor_name, predictor_file in (("w", w_file), ("Q", humidity_file)):
    output_dir = output_root / predictor_name / band_name
    figure_dir = figure_root / predictor_name / band_name
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if predictor_name == "w":
        predictor_label = "vertical-velocity"
        jacobian_label = r"$\partial Q_i / \partial w_j$  [(K day$^{-1}$) / (m s$^{-1}$)]"
    else:
        predictor_label = "specific-humidity"
        jacobian_label = r"$\partial Q_i / \partial q_j$  [(K day$^{-1}$) / (kg kg$^{-1}$)]"

    with (output_dir / "event_grid_mapping.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("date", "olr_time_idx", "olr_lon_idx", "olr_longitude",
                         "era5_time_idx", "era5_lon_idx", "era5_longitude",
                         "cloudsat_time_idx", "cloudsat_lon_idx", "cloudsat_longitude"))
        writer.writerows(zip(event_days.astype(str), olr_time_idx, olr_lon_idx, event_lon_values,
                             w_time_idx, w_lon_idx, era5_lon[w_lon_idx],
                             heating_time_idx, heating_lon_idx, heating_lon[heating_lon_idx]))

    # Read this predictor only; Q uses the verified, identical ERA5 grid.
    with nc.Dataset(predictor_file) as dataset:
        variable = dataset.variables[predictor_name]
        predictor_units = getattr(variable, "units", "m s-1" if predictor_name == "w" else "kg kg-1")
        predictor_selected = event_anomalies(variable, w_time_slice, w_lat_slice,
                                             w_time_idx, w_lon_idx)

    # Keep separate masks: missing Q must not remove any profiles from the w fit.
    valid_columns = heating_valid & np.all(np.isfinite(predictor_selected), axis=0)
    predictor_valid = predictor_selected[:, valid_columns]
    LW_valid = LW_selected[:, valid_columns]
    SW_valid = SW_selected[:, valid_columns]
    del predictor_selected
    if predictor_valid.shape[1] < 2:
        raise ValueError(f"{predictor_name}: fewer than two complete, collocated profiles.")
    profile_count = predictor_valid.shape[1]
    print(f"{predictor_name}: fitting {profile_count:,} profiles at {pressure_levels.size} levels "
          f"in {predictor_units}.", flush=True)

    # These are two separate regressions, not a combined w-and-humidity model.
    LW_model = Ridge(alpha=ridge_alpha, solver="svd", fit_intercept=True)
    SW_model = Ridge(alpha=ridge_alpha, solver="svd", fit_intercept=True)
    LW_model.fit(predictor_valid.T, LW_valid.T)
    SW_model.fit(predictor_valid.T, SW_valid.T)
    J_LW = LW_model.coef_
    J_SW = SW_model.coef_

    np.savetxt(output_dir / "Jacob_LW.txt", J_LW)
    np.savetxt(output_dir / "Jacob_SW.txt", J_SW)
    np.savetxt(output_dir / "plev.txt", pressure_levels)
    np.savetxt(output_dir / "intercept_LW.txt", LW_model.intercept_)
    np.savetxt(output_dir / "intercept_SW.txt", SW_model.intercept_)
    with (output_dir / "fit_info.txt").open("w") as stream:
        stream.write(f"Predictor: {predictor_name}\nUnits: {predictor_units}\n")
        stream.write(f"Ridge alpha: {ridge_alpha}\nComplete profiles: {profile_count}\n")
        stream.write("Composite prediction uses the Jacobian without the fitted intercept.\n")
    del predictor_valid, LW_valid, SW_valid

    # Use the matching saved w_composite.txt or Q_composite.txt for verification.
    predictor_composite = np.loadtxt(composite_dir / f"{predictor_name}_composite.txt")
    if predictor_composite.shape != expected_shape:
        raise ValueError(f"{predictor_name} composite must match the target pressure-longitude grid.")
    LW_prediction = J_LW @ predictor_composite
    SW_prediction = J_SW @ predictor_composite
    np.savetxt(output_dir / "LW_prediction.txt", LW_prediction)
    np.savetxt(output_dir / "SW_prediction.txt", SW_prediction)

    # Use a common prediction scale for LW/SW spatial-pattern overlays.
    # Observed contour labels retain their original heating-rate units.
    heating_limit = symmetric_limit(LW_prediction[:, lon_window], SW_prediction[:, lon_window])
    heating_norm = TwoSlopeNorm(vmin=-heating_limit, vcenter=0.0, vmax=heating_limit)


    for component, prediction, composite in (
        ("Longwave", LW_prediction, LW_composite),
        ("Shortwave", SW_prediction, SW_composite),
    ):
        profile_fig = plot_prediction_verification(prediction, composite, component)
        plt.close(profile_fig)

    # Scatter verification, using the same metrics and plotting window.
    predictions = {"LW": LW_prediction, "SW": SW_prediction}
    composites = {"LW": LW_composite, "SW": SW_composite}
    longitude = relative_lon
    window = comparison_window
    longitude = np.asarray(longitude)
    selected_lon = (longitude >= window[0]) & (longitude <= window[1])
    if not selected_lon.any():
        raise ValueError("No longitude points fall inside the scatter window")
    pairs, metrics = {}, {}
    for component in ("LW", "SW"):
        prediction = finite_array(predictions[component])
        composite = finite_array(composites[component])
        if prediction.ndim != 2 or prediction.shape != composite.shape or prediction.shape[-1] != longitude.size:
            raise ValueError(f"{component}: prediction, composite, and longitude shapes must match")
        predicted, observed = prediction[:, selected_lon], composite[:, selected_lon]
        metrics[component] = calculate_comparison_metrics(predicted, observed)
        valid = np.isfinite(predicted) & np.isfinite(observed)
        pairs[component] = (observed[valid], predicted[valid])

    finite_values = [values for pair in pairs.values() for values in pair if values.size]
    if finite_values:
        lower, upper = min(a.min() for a in finite_values), max(a.max() for a in finite_values)
        padding = 0.06 * (upper - lower) if upper > lower else 0.05 * max(abs(lower), 1.0)
        limits = (lower - padding, upper + padding)
    else:
        limits = (-1.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.9), sharex=True, sharey=True,
                             layout="constrained")
    density_artists = []
    scores = {}
    for ax, component, title in zip(axes, ("LW", "SW"), ("(a) Longwave", "(b) Shortwave")):
        observed, predicted = pairs[component]
        metric = metrics[component]
        if observed.size:
            density = ax.hexbin(observed, predicted, gridsize=46, mincnt=1,
                                extent=(*limits, *limits), cmap="viridis",
                                linewidths=0, rasterized=True)
            density_artists.append(density)
        ax.plot(limits, limits, color="#475467", ls="--", lw=1.2,
                label="1:1 agreement", zorder=3)
        ax.axhline(0.0, color="#98A2B3", lw=0.5, alpha=0.5, zorder=0)
        ax.axvline(0.0, color="#98A2B3", lw=0.5, alpha=0.5, zorder=0)
        ax.set(xlim=limits, ylim=limits, aspect="equal",
               xlabel=f"Observed heating anomaly ({heating_units})")
        ax.set_title(title, pad=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.legend(loc="lower right", fontsize=14)
        annotation = (
            f"Pattern r = {metric_label(metric['r'])}\n"
            f"$R^2$ = {metric_label(metric['r2'])}\n"
            f"RMSE = {metric_label(metric['rmse'])}\n"
            f"Bias = {metric_label(metric['bias'])}\n"
            rf"$\sigma_P / \sigma_O$ = {metric_label(metric['amplitude_ratio'])}"
            f"\nN = {metric['n_pairs']:,}"
        )
        ax.text(0.04, 0.96, annotation, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, linespacing=1.4, color="#243447",
                bbox=dict(boxstyle="round,pad=0.55", fc="white", ec="#D0D5DD", alpha=0.96))
        scores[component] = metric["r2"]
        print(f"{component}: pattern r = {metric_label(metric['r'], 4)}, "
              f"R2 = {metric_label(metric['r2'], 4)}, "
              f"Std(P)/Std(O) = {metric_label(metric['amplitude_ratio'], 4)}, "
              f"RMSE = {metric_label(metric['rmse'], 4)}, "
              f"bias = {metric_label(metric['bias'], 4)}, N = {metric['n_pairs']:,}")

    if density_artists:
        maximum_count = max(float(artist.get_array().max()) for artist in density_artists)
        count_norm = LogNorm(vmin=1.0, vmax=max(2.0, maximum_count))
        for artist in density_artists:
            artist.set_norm(count_norm)
        colorbar = fig.colorbar(density_artists[0], ax=axes, shrink=0.82, pad=0.025, aspect=30)
        colorbar.set_label("Paired grid points per hexagon")
        colorbar.outline.set_visible(False)
    axes[0].set_ylabel(f"Predicted heating anomaly ({heating_units})")
    save_verification_figure(fig, "LW_SW_prediction_scatter")
    plt.close(fig)

    # Numerical summary of the same finite pressure-longitude pairs.
    pattern_window = lon_window
    composite_pattern_metrics = {}
    composite_verification_metrics = {}
    print(f"{predictor_name} composite verification: all pressure levels, −50° to 50°")
    print(f"{'Dataset':<9}{'Pattern r':>12}{'R2':>10}{'Std(P)/Std(O)':>16}"
          f"{'RMSE':>11}{'Bias':>11}{'Pairs':>10}")
    for component, prediction, composite in (
        ("LW", LW_prediction, LW_composite),
        ("SW", SW_prediction, SW_composite),
    ):
        metrics = calculate_comparison_metrics(prediction[:, pattern_window], composite[:, pattern_window])
        composite_verification_metrics[component] = metrics
        composite_pattern_metrics[component] = {
            key: metrics[key] for key in ("r", "amplitude_ratio", "n_pairs")
        }
        print(f"{component:<9}{metrics['r']:>12.4f}{metrics['r2']:>10.4f}"
              f"{metrics['amplitude_ratio']:>16.4f}{metrics['rmse']:>11.4f}"
              f"{metrics['bias']:>11.4f}{metrics['n_pairs']:>10,d}")
    print("RMSE and bias are in K day^-1; bias = predicted − observed.")
    print("NaN denotes an undefined metric. These are composite diagnostics.")

    with (output_dir / "verification_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["component", "r", "r2", "amplitude_ratio",
                                                   "rmse", "bias", "n_pairs"])
        writer.writeheader()
        for component, metrics in composite_verification_metrics.items():
            writer.writerow({"component": component, **metrics})

    # Jacobian matrices: rows are heating levels; columns are predictor levels.
    # Equal-size cells display the discrete regression coefficients without smoothing.
    pressure_labels = [f"{value:g}" for value in pressure_hpa]
    limit = symmetric_limit(J_LW, J_SW)
    jacobian_norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.4), sharey=True, layout="constrained")

    for ax, jacobian, title in zip(axes, (J_LW, J_SW), ("(a) Longwave", "(b) Shortwave")):
        jacobian = finite_array(jacobian)
        if jacobian.shape != (pressure_hpa.size, pressure_hpa.size):
            raise ValueError("Jacobian shape must match the pressure levels")
        mesh = ax.imshow(np.ma.masked_invalid(jacobian), cmap=heating_cmap,
                         norm=jacobian_norm, interpolation="nearest", origin="upper",
                         aspect="equal")
        index = np.arange(pressure_hpa.size)
        ax.set_xlim(pressure_hpa.size - 0.5, -0.5)  # 1000 hPa on the left
        ax.set_ylim(pressure_hpa.size - 0.5, -0.5)  # 1000 hPa at the bottom
        ax.set_xticks(index, pressure_labels, rotation=55, ha="right", rotation_mode="anchor")
        ax.set_yticks(index, pressure_labels)
        ax.tick_params(axis="both", which="major", length=0, labelsize=14)
        ax.set_xticks(np.arange(-0.5, pressure_hpa.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, pressure_hpa.size, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.45, alpha=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.plot(index, index, color="#475467", ls=":", lw=1.1, alpha=0.75)
        ax.set_xlabel(f"Input {predictor_label} pressure (hPa)", labelpad=10)
        ax.set_title(title, pad=12)
        maximum = np.max(np.abs(jacobian[np.isfinite(jacobian)])) if np.isfinite(jacobian).any() else np.nan
        ax.text(0.98, 1.025, f"max |J| = {metric_label(maximum, 2)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=14, color="#667085")

    axes[0].set_ylabel("Heating-response pressure (hPa)", labelpad=10)
    colorbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.73,
                            pad=0.05, fraction=0.045, aspect=42)
    colorbar.set_label(jacobian_label, labelpad=7)
    colorbar.locator = MaxNLocator(nbins=7, symmetric=True)
    colorbar.update_ticks()
    colorbar.outline.set_visible(False)
    save_verification_figure(fig, "Jacobian")
    plt.close(fig)

    print(f"Saved Jacobians and verification data to {output_dir}")
    print(f"Saved verification figures to {figure_dir}")
