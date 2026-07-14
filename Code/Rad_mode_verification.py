# ====================================================
# This script is to predict radiative heating associated
# with different vertical modes
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

from glob import glob
from pathlib import Path
from pprint import pprint
from typing import List, Dict
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.style.use("~/KW_CloudSat/scientific.mplstyle")

# ====================================================
# Main function
# ====================================================

def main() -> None:

    # ------------------------------------------------
    # Load composite vertical motion
    # ------------------------------------------------

    # roots directory
    root_dir: Path = Path("/home/b11209013/KW_CloudSat/")
    file_dir: Path = root_dir / "Files"

    # Load vertical motion as dictionary
    fname_w: List[str] = list(glob(str(file_dir / "w_composite/*.npy")))

    dx: float = 360.0 / 576.0 

    w: Dict[str, np.ndarray] = {
            fname.split("/")[-1].split(".")[0]: np.load(fname)[:, int(288-100/dx):int(288+100/dx)]
            for fname in fname_w
            }

    nz, nx = w["k=1~3"].shape

    # Load regression coefficient
    reg_coeff: np.ndarray = np.load(file_dir / "Linear_Relation" / "regress_coeff.npy").squeeze()
    
    # Load radiative heating profile
    fname_qr: List[str] = list(glob(str(file_dir / "QR_composite/k*")))

    lw_valid: Dict[str, np.ndarray] = {
            fname.split("/")[-1]: np.load(fname+"/LW.npy")[:, int(288-100/dx):int(288+100/dx)]
            for fname in fname_qr
            }

    sw_valid: Dict[str, np.ndarray] = {
            fname.split("/")[-1]: np.load(fname+"/SW.npy")[:, int(288-100/dx):int(288+100/dx)]
            for fname in fname_qr
            }


    # ------------------------------------------------
    # Construct backgroun information
    # ------------------------------------------------

    z  : np.ndarray = np.linspace(0.0, 14000.0, 71)                   # vertical coordinate

    T  : np.ndarray = 300.0 - 0.0065 * z                              # background temperature profile
    p  : np.ndarray = 1e5 * (1-0.0065*z/300.0) ** (9.81/0.0065/287.5) # pressure profile based on hydrstat

    rho: np.ndarray = p / T / 287.5

    # ------------------------------------------------
    # Calculate vertical motion profile
    # ------------------------------------------------

    # vertical basis
    G1: np.ndarray = np.pi / 2 * np.sin(np.pi*z/z.max())
    G2: np.ndarray = np.pi / 2 * np.sin(2*np.pi*z/z.max())

    Gstack: np.ndarray = np.stack([G1, G2]).T

    # ------------------------------------------------
    # concatenate vertical motion
    # ------------------------------------------------

    keys = sorted(w.keys())
    w_concat: np.ndarray = np.concatenate([
        w[k].T for k in keys
        ])

    lw_concat: np.ndarray = np.concatenate([
        lw_valid[k].T for k in keys
        ])

    sw_concat: np.ndarray = np.concatenate([
        sw_valid[k].T for k in keys
        ])

    # ------------------------------------------------
    # Project data onto basis
    # ------------------------------------------------
    
    # ERA5 pressure coordinate
    p_era5: np.ndarray = np.linspace(1000.0, 100.0, 37)

    # Interpolate vertical motion to z coordinate
    w_z: np.ndarray = interp1d(p_era5, w_concat, axis=1, fill_value="extrapolate")(p/100.0).T*86400.0

    # Projection
    w_coeff: np.ndarray = np.linalg.solve(Gstack.T@Gstack, Gstack.T @ (rho[:, None]*w_z))

    # ------------------------------------------------
    # Predict radiative heating rate
    # ------------------------------------------------

    lw1: np.ndarray = w_coeff[0]*reg_coeff[0, 0] + w_coeff[1]*reg_coeff[1, 0]
    lw2: np.ndarray = w_coeff[0]*reg_coeff[0, 1] + w_coeff[1]*reg_coeff[1, 1]
    sw1: np.ndarray = w_coeff[0]*reg_coeff[2, 0] + w_coeff[1]*reg_coeff[3, 0]
    sw2: np.ndarray = w_coeff[0]*reg_coeff[2, 1] + w_coeff[1]*reg_coeff[3, 1]

    # ------------------------------------------------
    # reconstruct to vertical profile
    # ------------------------------------------------

    lw1_z: np.ndarray = np.einsum("s,z->sz", lw1, G1) * (9.8/1004.5 - 0.0065)
    lw2_z: np.ndarray = np.einsum("s,z->sz", lw2, G2) * (9.8/1004.5 - 0.0065)
    sw1_z: np.ndarray = np.einsum("s,z->sz", sw1, G1) * (9.8/1004.5 - 0.0065)
    sw2_z: np.ndarray = np.einsum("s,z->sz", sw2, G2) * (9.8/1004.5 - 0.0065)

    lw_z: np.ndarray = (lw1_z + lw2_z) / rho[None, :]
    sw_z: np.ndarray = (sw1_z + sw2_z) / rho[None, :]

    # ------------------------------------------------
    # Interpolate to ERA5 space
    # ------------------------------------------------

    lw_era5: np.ndarray = interp1d(p/100.0, lw_z, axis=1, fill_value="extrapolate")(p_era5)
    sw_era5: np.ndarray = interp1d(p/100.0, sw_z, axis=1, fill_value="extrapolate")(p_era5)

    print("Reconstructed shape:", lw_era5.shape)

    # ------------------------------------------------
    # Validation Scatter Plot
    # ------------------------------------------------
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Calculate global min/max for identity line
    lw_min, lw_max = np.nanmin(lw_concat), np.nanmax(lw_concat)
    sw_min, sw_max = np.nanmin(sw_concat), np.nanmax(sw_concat)

    # Flatten the arrays to compute correlation and plot
    lw_era5_flat = lw_era5.flatten()
    lw_concat_flat = lw_concat.flatten()
    sw_era5_flat = sw_era5.flatten()
    sw_concat_flat = sw_concat.flatten()

    axes[0].scatter(lw_era5_flat, lw_concat_flat, alpha=0.1, s=3, color='steelblue')
    axes[0].plot([lw_min, lw_max], [lw_min, lw_max], 'k--', lw=1.5, zorder=3)
    
    axes[1].scatter(sw_era5_flat, sw_concat_flat, alpha=0.1, s=3, color='steelblue')
    axes[1].plot([sw_min, sw_max], [sw_min, sw_max], 'k--', lw=1.5, zorder=3)

    corr_lw = np.corrcoef(lw_concat_flat, lw_era5_flat)[0, 1]
    corr_sw = np.corrcoef(sw_concat_flat, sw_era5_flat)[0, 1]

    axes[0].set_title(r"LW Heating Validation ($R$ = " + f"{corr_lw:.3f})")
    axes[1].set_title(r"SW Heating Validation ($R$ = " + f"{corr_sw:.3f})")

    for i in range(2):
        axes[i].set_xlabel("Reconstructed (K/day)")
        axes[i].set_ylabel("Validation (K/day)")
        axes[i].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig_path = root_dir / "Figure" / "Rad_mode_verification"
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path / "scatter_verify.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {fig_path / 'scatter_verify.png'}")

    # ------------------------------------------------
    # Visualize the last 320 samples (Reconstruction vs Validation)
    # ------------------------------------------------
    
    # Extract the last 320 samples
    nsamples = 320
    lw_recon_last = lw_era5[-nsamples:, :].T
    lw_valid_last = lw_concat[-nsamples:, :].T
    sw_recon_last = sw_era5[-nsamples:, :].T
    sw_valid_last = sw_concat[-nsamples:, :].T
    
    samples = np.arange(nsamples)
    
    # ------------------ LW Overlay ------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    
    pcm = ax[0].pcolormesh(samples, p_era5, lw_recon_last, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0), shading='nearest')
    
    levels = np.linspace(np.min(lw_valid_last), np.max(lw_valid_last), 9)
    linestyles = ['--' if v < 0 else '-' for v in levels]
    cs = ax[0].contour(samples, p_era5, lw_valid_last, levels=levels, colors='black', linewidths=1.0, alpha=0.8, linestyles=linestyles)
    ax[0].clabel(cs, inline=True, fontsize=9, fmt='%.1f')
    
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Pressure (hPa)")
    ax[0].set_xlabel(f"Sample Index (last {nsamples})")
    ax[0].set_title("LW Heating Rate: Recon (Color) vs Valid (Contours)")
    fig.colorbar(pcm, ax=ax[0], label="LW Heating Rate (K/day)", pad=0.02)
    
    # Mean Vertical Profile
    lw_recon_mean = np.mean(lw_recon_last, axis=1)
    lw_valid_mean_1d = np.mean(lw_valid_last, axis=1)
    
    ax[1].plot(lw_valid_mean_1d, p_era5, color='black', label='Validation', linewidth=2.5, zorder=3)
    ax[1].plot(lw_recon_mean, p_era5, color='red', linestyle='--', label='Reconstruction', linewidth=2.5, zorder=4)
    
    ax[1].axvline(0, color='gray', linestyle='--', linewidth=1.5, zorder=2)
    ax[1].set_xlabel("Mean LW Heating (K/day)")
    ax[1].set_title("Mean Vertical Profile")
    ax[1].legend()
    ax[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(fig_path / "lw_reconstruct_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------ SW Overlay ------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    
    pcm_sw = ax[0].pcolormesh(samples, p_era5, sw_recon_last, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0), shading='nearest')
    
    if np.ptp(sw_valid_last) > 0:
        levels_sw = np.linspace(np.min(sw_valid_last), np.max(sw_valid_last), 9)
        linestyles_sw = ['--' if v < 0 else '-' for v in levels_sw]
        cs_sw = ax[0].contour(samples, p_era5, sw_valid_last, levels=levels_sw, colors='black', linewidths=1.0, alpha=0.8, linestyles=linestyles_sw)
        ax[0].clabel(cs_sw, inline=True, fontsize=9, fmt='%.1f')
    
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Pressure (hPa)")
    ax[0].set_xlabel(f"Sample Index (last {nsamples})")
    ax[0].set_title("SW Heating Rate: Recon (Color) vs Valid (Contours)")
    fig.colorbar(pcm_sw, ax=ax[0], label="SW Heating Rate (K/day)", pad=0.02)
    
    # Mean Vertical Profile
    sw_recon_mean = np.mean(sw_recon_last, axis=1)
    sw_valid_mean_1d = np.mean(sw_valid_last, axis=1)
    
    ax[1].plot(sw_valid_mean_1d, p_era5, color='black', label='Validation', linewidth=2.5, zorder=3)
    ax[1].plot(sw_recon_mean, p_era5, color='red', linestyle='--', label='Reconstruction', linewidth=2.5, zorder=4)
    
    ax[1].axvline(0, color='gray', linestyle='--', linewidth=1.5, zorder=2)
    ax[1].set_xlabel("Mean SW Heating (K/day)")
    ax[1].set_title("Mean Vertical Profile")
    ax[1].legend()
    ax[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(fig_path / "sw_reconstruct_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
