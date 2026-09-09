"""Composite ERA5-GRIB vertical velocity and temperature for Kelvin-wave events."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from workflow import (
    COMPOSITE_DIR,
    FIGURE_DIR,
    band_name,
    dataset_coordinates,
    event_files,
    load_event_fields,
    load_events,
    longitude_indices,
    pressure_hpa,
    rolled_composite,
)


ERA5_DIR = Path("/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10")

# Tropical background absolute temperature (K) at the 15 ERA5 pressure levels.
# The GRIB T.nc file contains temperature anomalies, so it cannot be used in
# rho = p/(Rd*T). This profile is from the corresponding absolute ERA5 field.
T_BACKGROUND = np.array([
    192.75532, 205.56882, 219.88644, 232.11625, 242.20571,
    257.62539, 268.06591, 275.64239, 282.78556, 288.63271,
    290.97076, 292.99709, 294.12510, 295.36493, 298.94947,
])


def plot_composite(data: np.ndarray, level: np.ndarray, title: str, unit: str, path: Path) -> None:
    relative_longitude = np.linspace(-180.0, 180.0, data.shape[-1], endpoint=False)
    limit = max(float(np.nanmax(np.abs(data))), np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.contourf(
        relative_longitude, pressure_hpa(level), data,
        levels=np.linspace(-limit, limit, 15), cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), extend="both",
    )
    ax.set(xlim=(-70, 70), ylim=(1000, 100), xlabel="Relative longitude (degrees)",
           ylabel="Pressure (hPa)", title=title)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    fig.colorbar(image, ax=ax, label=unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_band(event_path: Path, save_concat: bool) -> None:
    name = band_name(event_path)
    w_path, t_path = ERA5_DIR / "W.nc", ERA5_DIR / "T.nc"
    coordinates = dataset_coordinates(w_path, "W")
    event_time, event_longitude = load_events(event_path, int(coordinates["n_time"]))

    omega, inverse, coordinates = load_event_fields(w_path, "W", event_time)
    temperature, t_inverse, t_coordinates = load_event_fields(t_path, "T", event_time)
    level = np.asarray(coordinates["level"])
    longitude = np.asarray(coordinates["longitude"])
    if T_BACKGROUND.shape != level.shape:
        raise ValueError("T_BACKGROUND must contain one value per ERA5 pressure level")
    if not np.array_equal(level, np.asarray(t_coordinates["level"])) or not np.array_equal(inverse, t_inverse):
        raise ValueError("ERA5 W and T coordinates do not match")

    density = level / (287.5 * T_BACKGROUND)
    w = -omega / (9.81 * density[None, :, None])
    event_x = longitude_indices(event_longitude, longitude)
    band_dir = COMPOSITE_DIR / name
    concat_dir = band_dir / "concat"
    w_composite = rolled_composite(
        w, inverse, event_x, concat_dir / "w.npy" if save_concat else None
    )
    t_composite = rolled_composite(
        temperature, inverse, event_x, concat_dir / "temperature.npy" if save_concat else None
    )

    output_dir = band_dir / "composite"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "w.npy", w_composite)
    np.save(output_dir / "temperature.npy", t_composite)
    np.savez(COMPOSITE_DIR / "coordinates.npz", pressure=level, longitude=longitude)
    plot_composite(w_composite, level, f"Vertical velocity anomaly: {name}", "m s$^{-1}$",
                   FIGURE_DIR / "composites" / name / "w.png")
    plot_composite(t_composite, level, f"Temperature anomaly: {name}", "K",
                   FIGURE_DIR / "composites" / name / "temperature.png")
    print(f"Finished {name}: {event_time.size:,} events")


def main(save_concat: bool = False) -> None:
    for event_path in event_files():
        process_band(event_path, save_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-concat", action="store_true", help="also save every centered event")
    main(parser.parse_args().save_concat)
