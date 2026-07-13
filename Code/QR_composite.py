# ====================================================
# This script is to composite radiative heating profile
# derived from CloudSat
# ====================================================

# ====================================================
# Environment Setup
# ====================================================

# Limit CPU usage
from optparse import Values
import os
import sys

from numpy.ma.core import shrink_mask

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
import sys
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from pprint import pprint
from typing import List
from scipy.ndimage import convolve1d

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.style.use("~/KW_CloudSat/scientific.mplstyle")

# ====================================================
# Main function
# ====================================================

def main(
    k_min: int,
    k_max: int,
    LW_anom: np.ndarray,
    SW_anom: np.ndarray,
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
    
    LW_roll = LW_anom[time_idx[:, None, None], np.arange(n_level)[None, :, None], lon_idx_2d[:, None, :]]
    SW_roll = SW_anom[time_idx[:, None, None], np.arange(n_level)[None, :, None], lon_idx_2d[:, None, :]]

    # composite
    LW_comp: np.ndarray = np.nanmean(LW_roll, axis=0)
    SW_comp: np.ndarray = np.nanmean(SW_roll, axis=0)

    # convolve over longitude
    LW_comp: np.ndarray = convolve1d(LW_comp, np.ones(33)/33, axis=-1, mode="reflect")
    SW_comp: np.ndarray = convolve1d(SW_comp, np.ones(33)/33, axis=-1, mode="reflect")


    pprint("Finished: composite")

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # Visualize profile of composited LW and SW profile
    lags = np.arange(coords["lon"].size) - coords["lon"].size // 2

    # Create a two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # LW panel
    lw_max: float      = np.nanmax(np.abs(LW_comp))
    lw_lev: np.ndarray = np.linspace(-lw_max, lw_max, 11)

    lw_cf = axes[0].contourf(
        lags,
        coords["lev"],
        LW_comp,
        levels=lw_lev,
        cmap="RdBu_r",
    )
    axes[0].set_xlim(-50, 50)
    axes[0].set_ylim(1000, 100)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Lag [degrees]")
    axes[0].set_ylabel("Pressure [hPa]")
    axes[0].set_title("QLW Composite Anomaly")
    fig.colorbar(lw_cf, ax=axes[0], orientation="horizontal", pad=0.12, label="K/day")

    # SW panel
    sw_max: float      = np.nanmax(np.abs(SW_comp))
    sw_lev: np.ndarray = np.linspace(-sw_max, sw_max, 11)
    sw_cf = axes[1].contourf(
        lags,
        coords["lev"],
        SW_comp,
        levels=sw_lev,
        cmap="RdBu_r",
    )
    axes[1].set_xlim(-50, 50)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Lag [degrees]")
    axes[1].set_title("QSW Composite Anomaly")
    fig.colorbar(sw_cf, ax=axes[1], orientation="horizontal", pad=0.12, label="K/day")

    # Save figure
    output_dir = Path("/home/b11209013/KW_CloudSat/Figure/QR_composite")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"k={k_min}~{k_max}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pprint(f"Finished: visualization. Saved to {output_path}")

    # ------------------------------------------------
    # Save Files
    # ------------------------------------------------
    save_path: Path = Path( f"/home/b11209013/KW_CloudSat/Files/QR_composite/k={k_min}~{k_max}")
    os.makedirs(save_path, exist_ok=True)

    np.save(save_path / f"LW.npy", LW_comp)
    np.save(save_path / f"SW.npy", SW_comp)

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    k_mins: list[int] = [1, 1, 3, 5, 7, 9, 11]
    k_maxs: list[int] = [13, 3, 5, 7, 9, 11, 13]

    pprint("Loading CloudSat data using netCDF4...")
    import netCDF4 as nc

    # First, use xarray strictly to get the metadata/indices quickly
    with xr.open_dataset("/data92/b11209013/CloudSat/DATA/QR_gridded.nc") as ds_tmp:
        time_vals = pd.to_datetime(ds_tmp['time'].values)
        t_mask = (time_vals >= pd.Timestamp("2006-01-01")) & (time_vals <= pd.Timestamp("2017-12-31"))
        t_indices = np.where(t_mask)[0]
        
        lat_vals = ds_tmp['lat'].values
        lat_mask = (lat_vals >= -5) & (lat_vals <= 5)
        lat_indices = np.where(lat_mask)[0]
        
        lon_vals = ds_tmp['lon'].values
        lev_name = 'lev' if 'lev' in ds_tmp.coords else 'level'
        lev_vals = ds_tmp[lev_name].values
        
        t_start, t_end = t_indices[0], t_indices[-1] + 1
        lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1

    # Now use netCDF4 for the heavy lifting (direct contiguous memory read)
    with nc.Dataset("/data92/b11209013/CloudSat/DATA/QR_gridded.nc", "r") as ds:
        qlw_var = ds.variables["QLW"]
        dims = qlw_var.dimensions
        
        # Build dynamic slice object based on dimension order
        slices = []
        for dim in dims:
            if dim == 'time':
                slices.append(slice(t_start, t_end))
            elif dim == 'lat':
                slices.append(slice(lat_start, lat_end))
            else:
                slices.append(slice(None))
                
        # Read exact subset into memory
        LW_subset = qlw_var[tuple(slices)]
        SW_subset = ds.variables["QSW"][tuple(slices)]
        
        # Find lat axis and mean over it
        lat_axis = dims.index('lat')
        LW_global = np.nanmean(LW_subset, axis=lat_axis)
        SW_global = np.nanmean(SW_subset, axis=lat_axis)
        
    coords_global: dict[str, np.ndarray] = {
        "lon": lon_vals,
        "lev": lev_vals,
        "time": time_vals[t_start:t_end]
    }

    # Remove time mean to create anomalies
    LW_anom_global: np.ndarray = (LW_global - np.nanmean(LW_global, axis=(0, ), keepdims=True))
    SW_anom_global: np.ndarray = (SW_global - np.nanmean(SW_global, axis=(0, ), keepdims=True)) 

    pprint("Data loaded and anomalies computed.")

    # execute composite and collect for different wavenumber band
    LW_comp: List = []
    SW_comp: List = []

    for (k_min, k_max) in zip(k_mins, k_maxs):
        pprint(f"Processing k={k_min}~{k_max}")
        main(
            k_min=k_min, 
            k_max=k_max, 
            LW_anom=LW_anom_global, 
            SW_anom=SW_anom_global, 
            coords=coords_global
        )

