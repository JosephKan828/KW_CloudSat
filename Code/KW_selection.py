# ====================================================
# This script is to select KW events using local minima
# ====================================================

# ====================================================
# Environment Setup
# ====================================================

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

# Import packages
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import minimum_filter

from pathlib import Path
from pprint import pprint
from matplotlib import pyplot as plt

plt.style.use("~/KW_CloudSat/scientific.mplstyle")

sys.path.append("/home/b11209013/demo_code/")
import PowerSpec as ps                # pyright: ignore[reportMissingImports]
import SpaceTimeReconstruct as recon  # pyright: ignore[reportMissingImports]

# ====================================================
# Main function
# ====================================================
def main(k_min: int, k_max: int) -> None:

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    olr_data_path: Path = Path("/work/DATA/Satellite/OLR/olr_anomaly.nc")

    with xr.open_dataset(olr_data_path, chunks={}, engine="netcdf4") as olr_ds:
        olr_ds: xr.Dataset = olr_ds.sel(
            time = slice("2006-01-01", "2017-12-31"),
            lat  = slice(-5, 5),
            )
    
        coords: dict[str, np.ndarray] = {
                key : value.values
                for (key, value) in olr_ds.coords.items()
            }

        olr: np.ndarray = olr_ds["olr"].values

    olr -= np.nanmean(olr, axis=(0, 2))[None, :, None]

    pprint("Finished: Load OLR data")

    # ------------------------------------------------
    # Pre-processing
    # ------------------------------------------------
    
    # Symmetrizing OLR data
    olr_symm: np.ndarray = (olr + olr[:, ::-1, :]) / 2

    # Construct wavenumber and frequency coordinate
    wnum: np.ndarray = np.fft.fftfreq(coords["lon"].size, d=1/coords["lon"].size)
    freq: np.ndarray = np.fft.fftfreq(coords["time"].size, d=1)

    wnums, freqs = np.meshgrid(wnum, freq)

    pprint("Finished: Setup frequency domain")

    # ------------------------------------------------
    # Implement KW filter
    # ------------------------------------------------
    
    kw_filter: recon.SpaceTimeFilter = recon.SpaceTimeFilter(
        dispersion = recon.DispersionParams(
            n_planetary_wave=576, rlat=0.0, 
            equivalent_depth=(8, 90),
            s_min=-288, s_max=287
        ),
        bandpass = recon.BandpassParams(
            k_range=(k_min, k_max),
            f_range=(1/30, 1/2.5),
            nan_to_inf=True
        )
    )

    kw_recon, mask = kw_filter.compute(
        data    = olr_symm,
        fr_grid = freqs,
        wn_grid = wnums,
        wave_type="Kelvin",
        return_mask=True,
    )

    pprint("Finished: Reconstruct OLR associated with KW")

    # ------------------------------------------------
    # Composite & Event Selection (The Update)
    # ------------------------------------------------

    olr_avg: float = np.nanmean(kw_recon)
    olr_std: float = np.nanstd(kw_recon) 

    sig_value: float = -2.77 * olr_std
    
    # Define the footprint for the local minimum search
    # t_window = 7 (±3 days), x_window = 31 (±15 degrees assuming 1-deg grid)
    t_window: int = 7
    x_window: int = 31

    # Apply the minimum filter. Mode 'wrap' ensures the footprint correctly 
    # crosses the prime meridian (0 to 360).
    local_min_array: np.ndarray = minimum_filter(
        kw_recon, 
        size=(t_window, x_window), 
        mode='wrap'
    )

 # An event is verified only if it is the local minimum AND exceeds the threshold
    is_local_min: np.ndarray   = (kw_recon == local_min_array)
    is_significant: np.ndarray = (kw_recon <= sig_value)
    
    valid_events: np.ndarray = is_local_min & is_significant
    time_idx, lon_idx = np.where(valid_events)

    # Calculate raw shifted indices without temporal wrapping
    n_time = coords["time"].size
    shifts_time = n_time // 2 - time_idx
    raw_time_idx = np.arange(n_time)[None, :] - shifts_time[:, None]
    
    # Create a boolean mask for valid physical boundaries in the time domain
    valid_mask = (raw_time_idx >= 0) & (raw_time_idx < n_time)
    
    # Clip the indices to prevent IndexError during extraction 
    # (The clipped values will be NaN'd out in the next step)
    safe_time_idx = np.clip(raw_time_idx, 0, n_time - 1)
    
    # Extract the data using safe indices
    kw_recon_roll_filtered = kw_recon[safe_time_idx, lon_idx[:, None]]
    
    olr_symm_zmean = olr_symm.mean(axis=1)
    kw_recon_roll_raw = olr_symm_zmean[safe_time_idx, lon_idx[:, None]]
    
    # Apply NaNs to boundary violations so they do not influence the composite mean
    kw_recon_roll_filtered = np.where(valid_mask, kw_recon_roll_filtered, np.nan)
    kw_recon_roll_raw = np.where(valid_mask, kw_recon_roll_raw, np.nan)
        
    # Compute the composite using nanmean to ignore the masked boundary clippings
    kw_recon_comp_filtered: np.ndarray = np.nanmean(kw_recon_roll_filtered, axis=0)
    kw_recon_comp_raw     : np.ndarray = np.nanmean(kw_recon_roll_raw, axis=0)

    pprint(f"Finished: Event Selection. Found {lon_idx.size} discrete events.")
    # ------------------------------------------------
    # Save files
    # ------------------------------------------------

    csv_path = Path(f"/home/b11209013/KW_CloudSat/Files/KW_events/k={k_min}~{k_max}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"time_idx": time_idx, "lon_idx": lon_idx})
    df.to_csv(csv_path, index=False)

    pprint("Finished: Save indices")

    # ------------------------------------------------
    # Visualize reconstruction
    # ------------------------------------------------

    os.makedirs(f"/home/b11209013/KW_CloudSat/Figure/KW_olr/k={k_min}~{k_max}/", exist_ok=True)

    # Plot Hovmoller diagram
    fig, ax = plt.subplots(1, 1, figsize=(7, 9))

    olr_pcm = ax.pcolormesh(
        coords["lon"], np.arange(100),
        kw_recon[:100, ...],
        cmap="RdBu_r"
    )

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Longitude [$^\circ$]")
    ax.set_ylabel("Day after 2006-01-01")
    ax.set_title("KW Reconstruction (First 100 days)")
    fig.colorbar(olr_pcm, ax=ax, shrink=0.8, aspect=50)

    plt.savefig(
        f"/home/b11209013/KW_CloudSat/Figure/KW_olr/k={k_min}~{k_max}/reconstruct.png",
        dpi=300,
        bbox_inches="tight"
        )
    plt.close(fig)

    # Plot histogram for filtered OLR
    kw_recon_flat = kw_recon.flatten()
    kw_recon_flat = kw_recon_flat[~np.isnan(kw_recon_flat)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(kw_recon_flat, bins=50, edgecolor="black", density=True, alpha=0.75, color="#0072B2")
    ax.axvline(sig_value, ymin=0, ymax=1, color="r", linestyle="--")

    ax.set_xlim(-40, 40)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"KW Reconstructed OLR Anomaly [$W\ m^{-2}$]")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Reconstructed KW OLR Anomaly")

    plt.savefig(
        f"/home/b11209013/KW_CloudSat/Figure/KW_olr/k={k_min}~{k_max}/histogram.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Plot filtered composite OLR
    lags = np.arange(coords["time"].size) - coords["time"].size // 2

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(lags, kw_recon_comp_filtered, color="#0072B2", linewidth=2, label="filtered")
    ax.plot(lags, kw_recon_comp_raw, color="#D55E00", linewidth=2, label="raw")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, label="Lag 0")
    
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Lag [days]")
    ax.set_ylabel(r"Composite OLR Anomaly [$W\ m^{-2}$]")
    ax.set_title("Composite of Reconstructed KW OLR Anomaly")
    ax.legend(loc="upper right")

    plt.savefig(
        f"/home/b11209013/KW_CloudSat/Figure/KW_olr/k={k_min}~{k_max}/composite.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    pprint("Finished: visualization")

# ====================================================
# Execute main function
# ====================================================
if __name__ == "__main__":

    k_mins: list[int] = [1, 1, 3, 5, 7, 9, 11]
    k_maxs: list[int] = [13, 3, 5, 7, 9, 11, 13]

    for (k_min, k_max) in zip(k_mins, k_maxs):
        pprint(f"Processing k={k_min}~{k_max}")
        main(k_min=k_min, k_max=k_max)
