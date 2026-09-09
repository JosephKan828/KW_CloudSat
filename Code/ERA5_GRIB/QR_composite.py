"""Composite CloudSat LW/SW heating using the ERA5-GRIB Kelvin-wave events."""

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


CLOUDSAT_PATH = Path("/data92/b11209013/CloudSat/DATA/QR_gridded_15layer_-5_5.nc")


def plot_heating(lw: np.ndarray, sw: np.ndarray, level: np.ndarray, name: str) -> None:
    relative_longitude = np.linspace(-180.0, 180.0, lw.shape[-1], endpoint=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, field, label in zip(axes, (lw, sw), ("LW", "SW")):
        limit = max(float(np.nanmax(np.abs(field))), np.finfo(float).eps)
        image = ax.contourf(
            relative_longitude, pressure_hpa(level), field,
            levels=np.linspace(-limit, limit, 15), cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), extend="both",
        )
        ax.set(xlim=(-70, 70), ylim=(1000, 100), xlabel="Relative longitude (degrees)",
               title=f"{label} heating anomaly: {name}")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.14, label="K day$^{-1}$")
    axes[0].set_ylabel("Pressure (hPa)")
    output = FIGURE_DIR / "composites" / name / "radiative_heating.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_band(event_path: Path, save_concat: bool) -> None:
    name = band_name(event_path)
    metadata = dataset_coordinates(CLOUDSAT_PATH, "QLW")
    event_time, event_longitude = load_events(event_path, int(metadata["n_time"]))
    lw, inverse, coordinates = load_event_fields(CLOUDSAT_PATH, "QLW", event_time)
    sw, sw_inverse, sw_coordinates = load_event_fields(CLOUDSAT_PATH, "QSW", event_time)
    level = np.asarray(coordinates["level"])
    longitude = np.asarray(coordinates["longitude"])
    if not np.array_equal(inverse, sw_inverse) or not np.array_equal(level, np.asarray(sw_coordinates["level"])):
        raise ValueError("CloudSat QLW and QSW coordinates do not match")

    # Remove the longitude mean at each event time and pressure level. This
    # isolates the wave-related perturbation without loading the full dataset.
    lw -= np.nanmean(lw, axis=-1, keepdims=True)
    sw -= np.nanmean(sw, axis=-1, keepdims=True)
    event_x = longitude_indices(event_longitude, longitude)
    band_dir = COMPOSITE_DIR / name
    concat_dir = band_dir / "concat"
    lw_composite = rolled_composite(
        lw, inverse, event_x, concat_dir / "LW.npy" if save_concat else None
    )
    sw_composite = rolled_composite(
        sw, inverse, event_x, concat_dir / "SW.npy" if save_concat else None
    )

    output_dir = band_dir / "composite"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "LW.npy", lw_composite)
    np.save(output_dir / "SW.npy", sw_composite)
    np.savez(COMPOSITE_DIR / "coordinates.npz", pressure=level, longitude=longitude)
    plot_heating(lw_composite, sw_composite, level, name)
    print(f"Finished {name}: {event_time.size:,} events")


def main(save_concat: bool = False) -> None:
    for event_path in event_files():
        process_band(event_path, save_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-concat", action="store_true", help="also save every centered event")
    main(parser.parse_args().save_concat)
