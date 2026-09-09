"""Shared I/O and compositing helpers for the ERA5-GRIB workflow."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
from scipy.ndimage import convolve1d


ROOT = Path(__file__).resolve().parents[2]
FILES_DIR = ROOT / "Files" / "ERA5_GRIB"
FIGURE_DIR = ROOT / "Figure" / "ERA5_GRIB"
EVENT_DIR = FILES_DIR / "KW_events"
COMPOSITE_DIR = FILES_DIR / "composites"


def event_files() -> list[Path]:
    files = sorted(EVENT_DIR.glob("(*, *).txt"))
    if not files:
        raise FileNotFoundError(f"No ERA5-GRIB event files found in {EVENT_DIR}")
    return files


def band_name(path: Path) -> str:
    """Convert an event stem such as '(1, 3)' to 'k=1~3'."""
    lower, upper = (int(value.strip()) for value in path.stem.strip("()").split(","))
    return f"k={lower}~{upper}"


def load_events(path: Path, n_time: int) -> tuple[np.ndarray, np.ndarray]:
    events = np.atleast_2d(np.loadtxt(path))
    if events.shape[0] != 2:
        raise ValueError(f"Expected two rows (time index, longitude) in {path}")
    time_index = events[0].astype(np.int64)
    longitude = events[1].astype(np.float64) % 360.0
    if np.any((time_index < 0) | (time_index >= n_time)):
        raise IndexError(f"Event time index outside [0, {n_time}) in {path}")
    return time_index, longitude


def dataset_coordinates(path: Path, variable: str) -> dict[str, np.ndarray | str]:
    with nc.Dataset(path) as dataset:
        data = dataset.variables[variable]
        level_name = next(name for name in ("plev", "lev", "level") if name in data.dimensions)
        latitude_name = next(name for name in ("lat", "latitude") if name in data.dimensions)
        longitude_name = next(name for name in ("lon", "longitude") if name in data.dimensions)
        return {
            "level": np.asarray(dataset.variables[level_name][:], dtype=np.float64),
            "latitude": np.asarray(dataset.variables[latitude_name][:], dtype=np.float64),
            "longitude": np.asarray(dataset.variables[longitude_name][:], dtype=np.float64),
            "level_name": level_name,
            "latitude_name": latitude_name,
            "longitude_name": longitude_name,
            "n_time": len(dataset.dimensions["time"]),
        }


def load_event_fields(
    path: Path,
    variable: str,
    event_time: np.ndarray,
    latitude_bounds: tuple[float, float] = (-5.0, 5.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | str]]:
    """Load latitude-mean fields once per unique event time.

    Returns fields indexed by unique time, an inverse index mapping every event
    to a field, and the dataset coordinates. This avoids loading the full 4-D
    dataset or repeatedly reading duplicate event times.
    """
    coordinates = dataset_coordinates(path, variable)
    latitude = np.asarray(coordinates["latitude"])
    latitude_index = np.flatnonzero(
        (latitude >= latitude_bounds[0]) & (latitude <= latitude_bounds[1])
    )
    if latitude_index.size == 0:
        raise ValueError(f"No latitudes in {latitude_bounds} are available in {path}")

    unique_time, inverse = np.unique(event_time, return_inverse=True)
    n_level = np.asarray(coordinates["level"]).size
    n_longitude = np.asarray(coordinates["longitude"]).size
    fields = np.empty((unique_time.size, n_level, n_longitude), dtype=np.float32)

    with nc.Dataset(path) as dataset:
        data = dataset.variables[variable]
        remaining_dimensions = [name for name in data.dimensions if name != "time"]
        level_axis = remaining_dimensions.index(str(coordinates["level_name"]))
        latitude_axis = remaining_dimensions.index(str(coordinates["latitude_name"]))
        longitude_axis = remaining_dimensions.index(str(coordinates["longitude_name"]))

        for output_index, time_index in enumerate(unique_time):
            field = np.asarray(
                np.ma.filled(data[int(time_index)], np.nan), dtype=np.float32
            )
            field = np.moveaxis(
                field,
                (level_axis, latitude_axis, longitude_axis),
                (0, 1, 2),
            )
            fields[output_index] = np.nanmean(field[:, latitude_index, :], axis=1)

    return fields, inverse, coordinates


def longitude_indices(event_longitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    spacing = float(np.median(np.diff(longitude)))
    if not np.allclose(np.diff(longitude), spacing):
        return np.array(
            [np.argmin(np.abs(((longitude - value + 180.0) % 360.0) - 180.0)) for value in event_longitude]
        )
    return np.rint((event_longitude - longitude[0]) / spacing).astype(np.int64) % longitude.size


def rolled_composite(
    fields: np.ndarray,
    inverse_time: np.ndarray,
    event_longitude_index: np.ndarray,
    concat_path: Path | None = None,
    chunk_size: int = 64,
    smoothing_points: int = 33,
) -> np.ndarray:
    """Center event fields on zero lag and calculate a NaN-safe composite."""
    n_event = inverse_time.size
    n_level, n_longitude = fields.shape[1:]
    total = np.zeros((n_level, n_longitude), dtype=np.float64)
    count = np.zeros((n_level, n_longitude), dtype=np.int64)
    concatenated = None
    if concat_path is not None:
        concat_path.parent.mkdir(parents=True, exist_ok=True)
        concatenated = np.lib.format.open_memmap(
            concat_path, mode="w+", dtype=np.float32,
            shape=(n_event, n_level, n_longitude),
        )

    destination = np.arange(n_longitude)
    for start in range(0, n_event, chunk_size):
        stop = min(start + chunk_size, n_event)
        source = fields[inverse_time[start:stop]]
        shifts = n_longitude // 2 - event_longitude_index[start:stop]
        source_index = (destination[None, :] - shifts[:, None]) % n_longitude
        rolled = np.take_along_axis(source, source_index[:, None, :], axis=2)
        finite = np.isfinite(rolled)
        total += np.nansum(rolled, axis=0)
        count += finite.sum(axis=0)
        if concatenated is not None:
            concatenated[start:stop] = rolled

    composite = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    if smoothing_points > 1:
        composite = convolve1d(
            composite, np.ones(smoothing_points) / smoothing_points,
            axis=-1, mode="wrap",
        )
    return composite


def pressure_hpa(level: np.ndarray) -> np.ndarray:
    level = np.asarray(level, dtype=np.float64)
    return level / 100.0 if np.nanmax(level) > 2_000.0 else level
