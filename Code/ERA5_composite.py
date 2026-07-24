# ====================================================
# This script is to composite radiative heating profile
# derived from CloudSat
# ====================================================

# ====================================================
# Environment Setup
# ====================================================

# Limit CPU usage
import os
import sys

# Define target CPU/thread limit
MAX_CPUS = 4

# Set thread limit for major libraries
cpu_limit_str = str(MAX_CPUS)
os.environ["OMP_NUM_THREADS"] = cpu_limit_str
os.environ["MKL_NUM_THREADS"] = cpu_limit_str
os.environ["OPENBLAS_NUM_THREADS"] = cpu_limit_str
os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_limit_str
os.environ["NUMEXPR_NUM_THREADS"] = cpu_limit_str

# Import package
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr

from pathlib import Path
from pprint import pprint
from typing import Tuple, Dict
from scipy.ndimage import convolve1d

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.style.use("~/KW_CloudSat/scientific.mplstyle")

# ====================================================
# Helper functions
# ====================================================

def _get_subset(
        fname: Path
        ) -> tuple[dict, np.ndarray, np.ndarray]:
    
    # Load coordinate metadata with xarray for time decoding
    with xr.open_dataset(fname) as ds_xr:
        time_vals = pd.to_datetime(ds_xr['time'].values)
        lat_vals = ds_xr['lat'].values
        lon_vals = ds_xr['lon'].values
        lev_name = 'plev' if 'plev' in ds_xr.coords else 'level'
        lev_vals = ds_xr[lev_name].values
        
    # Assign coordinate limits
    t_mask = (time_vals >= pd.Timestamp("2006-01-01")) & (time_vals <= pd.Timestamp("2017-12-31"))
    t_indices = np.where(t_mask)[0]
    
    lat_mask = (lat_vals >= -5) & (lat_vals <= 5)
    lat_indices = np.where(lat_mask)[0]
    
    t_start, t_end = t_indices[0], t_indices[-1] + 1
    lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1

    # Load data with netCDF4
    with nc.Dataset(fname, "r") as ds:
        # Load variables
        var_keys = [
                key for key in ds.variables.keys()
                if key not in ds.dimensions.keys()
                ]

        var = ds.variables[var_keys[-1]]

        # Build dynamic slice object based on dimension order
        slices = []
        for dim in var.dimensions:
            if dim == 'time':
                slices.append(slice(t_start, t_end))
            elif dim == 'lat':
                slices.append(slice(lat_start, lat_end))
            else:
                slices.append(slice(None))
                
        # Read exact subset into memory
        subset = var[tuple(slices)]
 
        # Find lat axis and mean over it
        lat_axis = var.dimensions.index('lat')
        var_global = np.nanmean(subset, axis=lat_axis)
        
        # Adjust time axis for removing mean after lat axis is removed
        time_axis = var.dimensions.index('time')
        if time_axis > lat_axis:
            time_axis -= 1

    coords_global = {
        "lon": lon_vals,
        "lev": lev_vals,
        "time": time_vals[t_start:t_end]
    }

    # Remove time mean to create anomalies
    anom_global = (var_global - np.nanmean(var_global, axis=time_axis, keepdims=True))

    return coords_global, var_global, anom_global


# ====================================================
# Main function
# ====================================================

def main(
    k_min: int,
    k_max: int,
    w_anom: np.ndarray,
    t_anom: np.ndarray,
    coords: dict
) -> None :

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    # Load event data
    time_idx, lon_idx = pd.read_csv(
        f"/home/b11209013/KW_CloudSat/Files/KW_events/k={k_min}~{k_max}.csv"
        ).to_numpy().T
    
    pprint(f"Finished: Load event indices for k={k_min}~{k_max}")
    
    # ------------------------------------------------
    # Composite data
    # ------------------------------------------------

    # Roll data according to index (vectorized)
    n_lon = coords["lon"].size
    n_level = coords["lev"].size
    shifts_lon = n_lon // 2 - lon_idx
    lon_idx_2d = (np.arange(n_lon)[None, :] - shifts_lon[:, None]) % n_lon
    
    w_roll = w_anom[time_idx[:, None, None], np.arange(n_level)[None, :, None], lon_idx_2d[:, None, :]]
    t_roll = t_anom[time_idx[:, None, None], np.arange(n_level)[None, :, None], lon_idx_2d[:, None, :]]

    # composite
    w_comp: np.ndarray = np.nanmean(w_roll, axis=0)
    t_comp: np.ndarray = np.nanmean(t_roll, axis=0)

    # concatenate along event
    w_conc: np.ndarray = np.array([w_roll[i] for i in range(w_roll.shape[0])])
    t_conc: np.ndarray = np.array([t_roll[i] for i in range(t_roll.shape[0])])

    # convolve over longitude
    w_comp = convolve1d(w_comp, np.ones(33)/33, axis=-1, mode="reflect")
    t_comp = convolve1d(t_comp, np.ones(33)/33, axis=-1, mode="reflect")

    pprint("Finished: composite")

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # Visualize profile of composited w profile
    lags = np.arange(coords["lon"].size) - coords["lon"].size // 2

    # Create a single-panel figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # w panel
    w_max: float      = np.nanmax(np.abs(w_comp))
    w_lev: np.ndarray = np.linspace(-w_max, w_max, 11)

    w_cf = ax.contourf(
        lags,
        coords["lev"],
        w_comp,
        levels=w_lev,
        cmap="RdBu_r",
    )
    ax.set_xlim(-50, 50)
    ax.set_ylim(1000, 100)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag [degrees]")
    ax.set_ylabel("Pressure [hPa]")
    ax.set_title("w Composite Anomaly")
    fig.colorbar(w_cf, ax=ax, orientation="horizontal", pad=0.12, label="m/s")

    # Save figure
    output_dir = Path("/home/b11209013/KW_CloudSat/Figure/w_composite")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"k={k_min}~{k_max}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create a single-panel figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # w panel
    t_max: float      = np.nanmax(np.abs(t_comp))
    t_lev: np.ndarray = np.linspace(-t_max, t_max, 11)

    t_cf = ax.contourf(
        lags,
        coords["lev"],
        t_comp,
        levels=t_lev,
        cmap="RdBu_r",
    )
    ax.set_xlim(-50, 50)
    ax.set_ylim(1000, 100)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag [degrees]")
    ax.set_ylabel("Pressure [hPa]")
    ax.set_title("T Composite Anomaly")
    fig.colorbar(t_cf, ax=ax, orientation="horizontal", pad=0.12, label="K")

    # Save figure
    output_dir = Path("/home/b11209013/KW_CloudSat/Figure/t_composite")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"k={k_min}~{k_max}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pprint(f"Finished: visualization. Saved to {output_path}")

    # ------------------------------------------------
    # Save Files
    # ------------------------------------------------
    comp_path_w: Path = Path( f"/home/b11209013/KW_CloudSat/Files/w_composite/")
    comp_path_t: Path = Path( f"/home/b11209013/KW_CloudSat/Files/t_composite/")

    conc_path_w: Path = Path(f"/home/b11209013/KW_CloudSat/Files/w_concate/")
    conc_path_t: Path = Path(f"/home/b11209013/KW_CloudSat/Files/t_concate/")

    os.makedirs(comp_path_w, exist_ok=True)
    os.makedirs(comp_path_t, exist_ok=True)
    os.makedirs(conc_path_w, exist_ok=True)
    os.makedirs(conc_path_t, exist_ok=True)

    np.save(comp_path_w / f"k={k_min}~{k_max}.npy", w_comp)
    np.save(comp_path_t / f"k={k_min}~{k_max}.npy", t_comp)
    np.save(conc_path_w / f"k={k_min}~{k_max}.npy", w_conc)
    np.save(conc_path_t / f"k={k_min}~{k_max}.npy", t_conc)

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    k_mins: list[int] = [1, 1, 3, 5, 7, 9, 11]
    k_maxs: list[int] = [13, 3, 5, 7, 9, 11, 13]

    pprint("Loading ERA5 data using netCDF4...")

    # Load metadata of pressure velocity
    input_dir: Path = Path("/data92/b11209013/ERA5/")
    
    # Use the refactored helper function to load and process data
    coords_global, omega_global, omega_anom_global = _get_subset(input_dir / "w" / "w_merged_Itp.nc")
    _, t_global, t_anom_global = _get_subset(input_dir / "t" / "t_merged_Itp.nc")

    # Calculate density via ideal gas law (convert hPa to Pa by multiplying by 100)
    rho: np.ndarray = (coords_global["lev"][None, :, None]) / 287.5 / t_global

    print("maximum of density: ", rho.max())

    w_global     : np.ndarray = -omega_global / 9.81 / rho 
    w_anom_global: np.ndarray = w_global - np.nanmean(w_global, axis=0, keepdims=True)

    print(np.nanmax(w_anom_global))

    # execute composite
    for (k_min, k_max) in zip(k_mins, k_maxs):
        pprint(f"Processing k={k_min}~{k_max}")
        main(
            k_min=k_min, 
            k_max=k_max, 
            w_anom=w_anom_global,
            t_anom=t_anom_global,
            coords=coords_global
        )
