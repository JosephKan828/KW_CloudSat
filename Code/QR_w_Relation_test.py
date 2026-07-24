# ====================================================
# This script is to test the new algorithm of radiative 
# feedback
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
MAX_CPUS = 1

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
import netCDF4 as nc

from glob import glob
from pathlib import Path
from pprint import pprint
from typing import List, Dict

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.style.use("~/KW_CloudSat/scientific.mplstyle")

# ====================================================
# Main function
# ====================================================

def main() -> None:

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    # path Setup
    root_dir: Path = Path("/home/b11209013/KW_CloudSat/Files/")

    # Load radiative heating
    lw: np.ndarray = np.concatenate([
        np.load(file + "/concat/LW.npy")
        for file in list(glob(str(root_dir / "QR_composite/*")))
        ], axis=0)

    sw: np.ndarray = np.concatenate([
        np.load(file + "/concat/SW.npy")
        for file in list(glob(str(root_dir / "QR_composite/*")))
        ], axis=0)

    # Load vertical motion
    w: np.ndarray = np.concatenate([
        np.load(file)
        for file in list(glob(str(root_dir / "w_concate/*.npy")))
        ], axis=0)

    # ------------------------------------------------
    # concatenate data to form (nz, nsample) data
    # ------------------------------------------------

    lw_concate: np.ndarray = np.concatenate([
        lw[i] for i in range(lw.shape[0])
        ], axis=-1)

    sw_concate: np.ndarray = np.concatenate([
        sw[i] for i in range(lw.shape[0])
        ], axis=-1)

    w_concate: np.ndarray = np.concatenate([
        w[i] for i in range(lw.shape[0])
        ], axis=-1)
    
    # ------------------------------------------------
    # collect not all NAN column
    # ------------------------------------------------

    # fileter   
    lw_non_nan: np.ndarray = np.all(~np.isnan(lw_concate), axis=0)
    sw_non_nan: np.ndarray = np.all(~np.isnan(sw_concate), axis=0)

    # valid columns
    lw_valid: np.ndarray = lw_concate[:, lw_non_nan]
    sw_valid: np.ndarray = sw_concate[:, sw_non_nan]

    w_lw_valid: np.ndarray = w_concate[:, lw_non_nan]
    w_sw_valid: np.ndarray = w_concate[:, sw_non_nan]
    
    # ------------------------------------------------
    # Apply running average along the sample axis
    # ------------------------------------------------
    def apply_running_average(data: np.ndarray, window: int = 50) -> np.ndarray:
        ret = np.cumsum(data, axis=1, dtype=float)
        ret[:, window:] = ret[:, window:] - ret[:, :-window]
        return ret[:, window - 1:] / window

    window_size = 50
    lw_valid = apply_running_average(lw_valid, window_size)
    w_lw_valid = apply_running_average(w_lw_valid, window_size)

    sw_valid = apply_running_average(sw_valid, window_size)
    w_sw_valid = apply_running_average(w_sw_valid, window_size)

    # ------------------------------------------------
    # split data into training and validating set 
    # ------------------------------------------------

    # define the lenght of sampling space
    lw_nz, lw_nsample = lw_valid.shape
    sw_nz, sw_nsample = sw_valid.shape

    lw_train: np.ndarray = lw_valid[:, :-int(lw_nsample/6)]
    lw_test : np.ndarray = lw_valid[:, -int(lw_nsample/6):]

    sw_train: np.ndarray = sw_valid[:, :-int(sw_nsample/6)]
    sw_test : np.ndarray = sw_valid[:, -int(sw_nsample/6):]

    w_lw_train: np.ndarray = w_lw_valid[:, :-int(lw_nsample/6)]
    w_lw_test : np.ndarray = w_lw_valid[:, -int(lw_nsample/6):]

    w_sw_train: np.ndarray = w_sw_valid[:, :-int(sw_nsample/6)]
    w_sw_test : np.ndarray = w_sw_valid[:, -int(sw_nsample/6):]

    print(np.nanmax(w_sw_test))
    print(np.nanmax(sw_test))

    # ------------------------------------------------
    # Calculate relations with partial least square 
    # method
    # ------------------------------------------------

    from sklearn.cross_decomposition import PLSRegression
    
    # Fit PLS models for LW and SW (using 4 components)
    # scale=False means it will center the data but not normalize by standard deviation
    pls_lw = PLSRegression(n_components=5, scale=False)
    pls_sw = PLSRegression(n_components=5, scale=False)
    
    pls_lw.fit(w_lw_train.T, lw_train.T)
    pls_sw.fit(w_sw_train.T, sw_train.T)
    
    # Extract Jacobian Matrices (equivalent to the M_lw and M_sw matrices)
    # PLS prediction is internally: Y_pred = (X - X_mean) @ coef_ + Y_mean
    # Since we want M_lw such that M_lw @ w.T matches this, we transpose coef_
    M_lw: np.ndarray = pls_lw.coef_
    M_sw: np.ndarray = pls_sw.coef_
    
    print(f"M_lw shape: {M_lw.shape}")
    print(f"M_sw shape: {M_sw.shape}")

    # ------------------------------------------------
    # Save Jacobian Matrices
    # ------------------------------------------------
    jacobian_dir: Path = Path("/home/b11209013/KW_CloudSat/Files/Linear_Relation_new/")
    os.makedirs(jacobian_dir, exist_ok=True)
    np.save(jacobian_dir / "M_lw.npy", M_lw)
    np.save(jacobian_dir / "M_sw.npy", M_sw)
    print(f"Jacobian matrices saved to {jacobian_dir}")

    # ------------------------------------------------
    # Testing & Scatter Plot Visualization
    # ------------------------------------------------
    
    # Predict on the testing set
    lw_recon_test = pls_lw.predict(w_lw_test.T)
    sw_recon_test = pls_sw.predict(w_sw_test.T)
    
    # Actual test data transposed to match prediction shape (nsamples, 37)
    lw_actual = lw_test.T
    sw_actual = sw_test.T
    
    # Calculate global min/max for identity line
    lw_min, lw_max = np.min(lw_actual), np.max(lw_actual)
    sw_min, sw_max = np.min(sw_actual), np.max(sw_actual)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].scatter(lw_recon_test.flatten(), lw_actual.flatten(), alpha=0.1, s=3, color='steelblue')
    ax[0].plot([lw_min, lw_max], [lw_min, lw_max], 'k--', lw=1.5, zorder=3)
    ax[1].scatter(sw_recon_test.flatten(), sw_actual.flatten(), alpha=0.1, s=3, color='steelblue')
    ax[1].plot([sw_min, sw_max], [sw_min, sw_max], 'k--', lw=1.5, zorder=3)
    
    corr_lw = np.corrcoef(lw_actual.flatten(), lw_recon_test.flatten())[0, 1]
    corr_sw = np.corrcoef(sw_actual.flatten(), sw_recon_test.flatten())[0, 1]
    
    ax[0].set_title(r"LW Heating Testing ($R$ = " + f"{corr_lw:.3f})")
    ax[1].set_title(r"SW Heating Testing ($R$ = " + f"{corr_sw:.3f})")
    
    for i in range(2):
        ax[i].set_xlabel("Reconstructed (K/day)")
        ax[i].set_ylabel("Testing (K/day)")
        ax[i].grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    
    # Setup figure path
    fig_path: Path = Path("/home/b11209013/KW_CloudSat/Figure/QR_w_Relation_test")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path / "testing_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved to {fig_path / 'testing_scatter.png'}")

    plt.pcolormesh(
            np.linspace(1000, 100, 37),
            np.linspace(1000, 100, 37),
            M_lw
            )
    plt.xlim(1000, 100)
    plt.ylim(1000, 100)
    plt.colorbar()
    plt.show()

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
