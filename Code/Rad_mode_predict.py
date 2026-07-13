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
    # Load Jacobian Matrix
    # ------------------------------------------------

    root_dir : Path = Path("/home/b11209013/KW_CloudSat/")
    input_dir: Path = root_dir / "Files" / "Linear_Relation"

    M_lw: np.ndarray = np.load(input_dir / "M_lw.npy")
    M_sw: np.ndarray = np.load(input_dir / "M_sw.npy")

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

    G_mat: np.ndarray = np.vstack([G1, G2]).T

    # vertical motion profile
    w1: np.ndarray = G1 / rho / 86400.0 # convert from m/day to m/s
    w2: np.ndarray = G2 / rho / 86400.0

    # ------------------------------------------------
    # Interpolate vertical motion profile to ERA5 level
    # ------------------------------------------------

    p_era5 : np.ndarray = np.linspace(1000.0, 100.0, 37) # pressure level in ERA5

    w1_era5: np.ndarray = interp1d(p/100.0, w1, fill_value="extrapolate")(p_era5)
    w2_era5: np.ndarray = interp1d(p/100.0, w2, fill_value="extrapolate")(p_era5)

    # ------------------------------------------------
    # predict radiation through Jacobian Matrix
    # ------------------------------------------------
    
    lw_w1_era5: np.ndarray = M_lw @ w1_era5
    sw_w1_era5: np.ndarray = M_sw @ w1_era5
    
    lw_w2_era5: np.ndarray = M_lw @ w2_era5
    sw_w2_era5: np.ndarray = M_sw @ w2_era5

    # ------------------------------------------------
    # Interpolate radiative heating rate back to Kuang's
    # coordinate
    # ------------------------------------------------

    lw_w1: np.ndarray = interp1d(p_era5, lw_w1_era5, fill_value="extrapolate")(p/100.0)
    sw_w1: np.ndarray = interp1d(p_era5, sw_w1_era5, fill_value="extrapolate")(p/100.0)

    lw_w2: np.ndarray = interp1d(p_era5, lw_w2_era5, fill_value="extrapolate")(p/100.0)
    sw_w2: np.ndarray = interp1d(p_era5, sw_w2_era5, fill_value="extrapolate")(p/100.0)

    # ------------------------------------------------
    # Decompose radiative heating rate
    # ------------------------------------------------

    # basis for J and T
    J_G_mat: np.ndarray = G_mat * (9.81/1004.5 - 0.0065)

    lw_w1_coeff: np.ndarray = np.linalg.solve(J_G_mat.T @ J_G_mat, J_G_mat.T @ (rho*lw_w1)[:, None])
    sw_w1_coeff: np.ndarray = np.linalg.solve(J_G_mat.T @ J_G_mat, J_G_mat.T @ (rho*sw_w1)[:, None])

    lw_w2_coeff: np.ndarray = np.linalg.solve(J_G_mat.T @ J_G_mat, J_G_mat.T @ (rho*lw_w2)[:, None])
    sw_w2_coeff: np.ndarray = np.linalg.solve(J_G_mat.T @ J_G_mat, J_G_mat.T @ (rho*sw_w2)[:, None])

    # ------------------------------------------------
    # Reconstruct radiative heating through regression
    # ------------------------------------------------

    lw_w1_recon: np.ndarray = (J_G_mat @ lw_w1_coeff) / rho[:, None]
    sw_w1_recon: np.ndarray = (J_G_mat @ sw_w1_coeff) / rho[:, None]
    lw_w2_recon: np.ndarray = (J_G_mat @ lw_w2_coeff) / rho[:, None]
    sw_w2_recon: np.ndarray = (J_G_mat @ sw_w2_coeff) / rho[:, None]

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Panel 1: Mode 1
    axes[0].plot(lw_w1, z, 'r-', linewidth=2, label="LW w1 (Orig)")
    axes[0].plot(lw_w1_recon.flatten(), z, 'r--', linewidth=2, label="LW w1 (Recon)")
    axes[0].plot(sw_w1, z, 'b-', linewidth=2, label="SW w1 (Orig)")
    axes[0].plot(sw_w1_recon.flatten(), z, 'b--', linewidth=2, label="SW w1 (Recon)")
    axes[0].set_xlabel("Heating Rate (K/day)")
    axes[0].set_ylabel("Height (m)")
    axes[0].set_title("Mode 1 Radiative Heating")
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Mode 2
    axes[1].plot(lw_w2, z, 'r-', linewidth=2, label="LW w2 (Orig)")
    axes[1].plot(lw_w2_recon.flatten(), z, 'r--', linewidth=2, label="LW w2 (Recon)")
    axes[1].plot(sw_w2, z, 'b-', linewidth=2, label="SW w2 (Orig)")
    axes[1].plot(sw_w2_recon.flatten(), z, 'b--', linewidth=2, label="SW w2 (Recon)")
    axes[1].set_xlabel("Heating Rate (K/day)")
    axes[1].set_title("Mode 2 Radiative Heating")
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    save_path = root_dir / "Figure" / "Rad_mode_predict.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {save_path}")

    # ------------------------------------------------
    # Save coefficient
    # ------------------------------------------------

    # file path
    output_path: Path = root_dir / "Files" / "Linear_Relation"

    np.save(output_path / "regress_coeff.npy", np.array([lw_w1_coeff, lw_w2_coeff, sw_w1_coeff, sw_w2_coeff]))

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
