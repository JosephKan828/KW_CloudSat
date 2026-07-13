# ====================================================
# This script is to calculate linear relations between 
# vertical motion and radiative heating rate.
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

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.style.use("~/KW_CloudSat/scientific.mplstyle")

# ====================================================
# Helper function
# ====================================================

def _EOF(data):

    # acquire the shape information
    nsample, nz = data.shape

    # 1. Center the data (calculate anomalies)
    data_mean = np.mean(data, axis=0)
    data_anom = data - data_mean

    # 2. Covariance matrix for data array
    cov: np.ndarray = (data_anom.T @ data_anom) / (nsample - 1)

    # 3. Eigenvalue and eigenvector
    # Use eigh for symmetric matrices (guarantees real numbers and better stability)
    eigvals, eigvecs = np.linalg.eigh(cov)

    ## sort eigenvector with the magnitude of eigenvalue (eigh sorts ascending, so reverse)
    eigvals_sort_idx: np.ndarray = np.argsort(eigvals)[::-1]

    exp_var: np.ndarray = eigvals[eigvals_sort_idx] / np.sum(eigvals)

    ## sort eigenvector
    eigvec_sorted: np.ndarray = eigvecs[:, eigvals_sort_idx]

    # construct EOF and PCs
    EOF: np.ndarray = eigvec_sorted
    
    # 4. Orthogonal projection (EOF @ EOF.T is Identity)
    PCs: np.ndarray = EOF.T @ data_anom.T

    return exp_var, EOF, PCs, data_mean

# ====================================================
# Main function
# ====================================================

def main() -> None:
    
    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    input_dir: Path = Path.cwd().parent / "Files"

    # Load vertical motion as dictionary
    fname_w: List[str] = list(glob(str(input_dir / "w_composite/*.npy")))

    w: Dict[str, np.ndarray] = {
            fname.split("/")[-1].split(".")[0]: np.load(fname)
            for fname in fname_w
            }

    # Load radiative heating rate
    fname_qr: List[str] = list(glob(str(input_dir / "QR_composite/k*")))

    lw: Dict[str, np.ndarray] = {
            fname.split("/")[-1]: np.load(fname+"/LW.npy")
            for fname in fname_qr
            }

    sw: Dict[str, np.ndarray] = {
            fname.split("/")[-1]: np.load(fname+"/SW.npy")
            for fname in fname_qr
            }

    # ------------------------------------------------
    # Concatenate data
    # ------------------------------------------------

    w_concat: np.ndarray = np.concatenate([
        val.T for val in w.values()
        ])

    
    lw_concat: np.ndarray = np.concatenate([
        val.T for val in lw.values()
        ])

    sw_concat: np.ndarray = np.concatenate([
        val.T for val in sw.values()
        ])

    # ------------------------------------------------
    # Split data into training and verifying
    # ------------------------------------------------

    w_train : np.ndarray = w_concat[:37*37*2] ; w_valid : np.ndarray = w_concat[37*37*2:]
    lw_train: np.ndarray = lw_concat[:37*37*2]; lw_valid: np.ndarray = lw_concat[37*37*2:]
    sw_train: np.ndarray = sw_concat[:37*37*2]; sw_valid: np.ndarray = sw_concat[37*37*2:]

    # ------------------------------------------------
    # Use Partial Least Squares (PLS) regression
    # ------------------------------------------------

    from sklearn.cross_decomposition import PLSRegression
    
    # Fit PLS models for LW and SW (using 5 components)
    # scale=False means it will center the data but not normalize by standard deviation
    pls_lw = PLSRegression(n_components=5, scale=False)
    pls_sw = PLSRegression(n_components=5, scale=False)
    
    pls_lw.fit(w_train, lw_train)
    pls_sw.fit(w_train, sw_train)
    
    # Extract Jacobian Matrices (equivalent to the M_lw and M_sw matrices)
    # PLS prediction is internally: Y_pred = (X - X_mean) @ coef_ + Y_mean
    # Since we want M_lw such that M_lw @ w.T matches this, we transpose coef_
    M_lw: np.ndarray = pls_lw.coef_
    M_sw: np.ndarray = pls_sw.coef_

    # ------------------------------------------------
    # Verifying
    # ------------------------------------------------
    # 1. Calculate the true physical means of the validation block
    w_valid_mean = np.mean(w_valid, axis=0)
    lw_valid_mean = np.mean(lw_valid, axis=0)
    sw_valid_mean = np.mean(sw_valid, axis=0)

    # 2. Extract strictly kinematic anomalies for the validation block
    w_valid_anom = w_valid - w_valid_mean

    # 3. Project anomalies through your extracted Jacobians
    # w_valid_anom is (nsample, nz). We transpose it for the M @ w operation.
    lw_recon_anom = M_lw @ w_valid_anom.T  # Shape: (nz, nsample)
    sw_recon_anom = M_sw @ w_valid_anom.T  # Shape: (nz, nsample)

    # 4. Add the validation mean back to eliminate the climatological bias
    # np.newaxis ensures the (nz,) mean array broadcasts correctly across nsamples
    lw_recon: np.ndarray = lw_recon_anom + lw_valid_mean[:, np.newaxis]
    sw_recon: np.ndarray = sw_recon_anom + sw_valid_mean[:, np.newaxis]

    
    # ------------------------------------------------
    # Visualize the matrix
    # ------------------------------------------------
    
    # setup vertical coordinate
    lev: np.ndarray = np.linspace(1000.0, 100.0, 37)

    # Setup figure path
    fig_path: Path = Path.cwd().parent / "Figure" / "QR_w_Relation"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    lw_pcm = ax.pcolormesh(
            lev, lev, M_lw,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0)
            )
    ax.set_xlim(1000, 100)
    ax.set_ylim(1000, 100)

    fig.colorbar(lw_pcm, ax=ax)
    plt.savefig(fig_path / "M_lw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sw_pcm = ax.pcolormesh(
            lev, lev, M_sw,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0)
            )
    ax.set_xlim(1000, 100)
    ax.set_ylim(1000, 100)

    fig.colorbar(sw_pcm, ax=ax)
    plt.savefig(fig_path / "M_sw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------
    # Visualize the validation
    # ------------------------------------------------
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(lw_recon.T, lw_valid)
    ax[1].scatter(sw_recon.T, sw_valid)

    corr_lw = np.corrcoef(lw_valid.flatten(), lw_recon.T.flatten())[0, 1]
    corr_sw = np.corrcoef(sw_valid.flatten(), sw_recon.T.flatten())[0, 1]

    ax[0].set_title(r"Correlation ($R$): " + f"{corr_lw:.3f}")
    ax[1].set_title(r"Correlation ($R$): " + f"{corr_sw:.3f}")

    plt.savefig(fig_path / "verify.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------
    # Visualize the Correlation Score Profile
    # ------------------------------------------------
    
    nz = lw_valid.shape[1]
    corr_lw_profile = []
    corr_sw_profile = []
    
    for i in range(nz):
        if np.std(lw_valid[:, i]) > 0 and np.std(lw_recon.T[:, i]) > 0:
            corr_lw_profile.append(np.corrcoef(lw_valid[:, i], lw_recon.T[:, i])[0, 1])
        else:
            corr_lw_profile.append(0.0)
            
        if np.std(sw_valid[:, i]) > 0 and np.std(sw_recon.T[:, i]) > 0:
            corr_sw_profile.append(np.corrcoef(sw_valid[:, i], sw_recon.T[:, i])[0, 1])
        else:
            corr_sw_profile.append(0.0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    # Plot horizontal bar charts (barh) since the y-axis is pressure level
    ax[0].barh(lev, corr_lw_profile, height=20, align='center', color='coral', edgecolor='black', linewidth=0.5)
    ax[1].barh(lev, corr_sw_profile, height=20, align='center', color='skyblue', edgecolor='black', linewidth=0.5)

    ax[0].set_title(r"LW Correlation ($R$) Profile")
    ax[1].set_title(r"SW Correlation ($R$) Profile")
    
    ax[0].set_xlabel(r"Correlation Coefficient ($R$)")
    ax[1].set_xlabel(r"Correlation Coefficient ($R$)")
    ax[0].set_ylabel("Pressure (hPa)")

    # Invert y-axis to have higher pressure (surface) at the bottom
    ax[0].invert_yaxis()
    
    # Add vertical line at 0 for reference
    ax[0].axvline(0, color='black', linestyle='--', linewidth=1.0)
    ax[1].axvline(0, color='black', linestyle='--', linewidth=1.0)

    plt.savefig(fig_path / "corr_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------
    # Visualize the last 676 samples (Reconstruction vs Validation)
    # ------------------------------------------------
    
    # Extract the last 676 samples
    # lw_recon has shape (nz, nsample), so we slice the columns
    lw_recon_last = lw_recon[:, -676:]
    # lw_valid has shape (nsample, nz), so we slice the rows and transpose
    lw_valid_last = lw_valid[-676:, :].T
    
    samples = np.arange(676)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    
    # Panel 1: 2D Cross section
    # Reconstructed data is the colormesh background
    pcm = ax[0].pcolormesh(samples, lev, lw_recon_last, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0), shading='auto')
    
    # Validating data is overlaid as contour lines
    levels = np.linspace(np.min(lw_valid_last), np.max(lw_valid_last), 9)
    cs = ax[0].contour(samples, lev, lw_valid_last, levels=levels, colors='black', linewidths=0.8, alpha=0.8)
    ax[0].clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Pressure (hPa)")
    ax[0].set_xlabel("Sample Index (last 676)")
    ax[0].set_title("LW Heating Rate: Reconstruction (Color) vs Validation (Contours)")
    fig.colorbar(pcm, ax=ax[0], label="LW Heating Rate")
    
    # Panel 2: Mean Vertical Profile (1D)
    lw_recon_mean = np.mean(lw_recon_last, axis=1)
    lw_valid_mean = np.mean(lw_valid_last, axis=1)
    
    ax[1].plot(lw_valid_mean, lev, color='black', label='Validation (Truth)', linewidth=2)
    ax[1].plot(lw_recon_mean, lev, color='red', linestyle='--', label='Reconstruction', linewidth=2)
    
    ax[1].axvline(0, color='gray', linestyle='--', linewidth=1)
    ax[1].set_xlabel("Mean LW Heating Rate")
    ax[1].set_title("Mean Vertical Profile (over 676 samples)")
    ax[1].legend()
    
    plt.savefig(fig_path / "lw_reconstruct_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------
    # Visualize the last 676 samples for SW (Reconstruction vs Validation)
    # ------------------------------------------------
    
    # Extract the last 676 samples for SW
    sw_recon_last = sw_recon[:, -676:]
    sw_valid_last = sw_valid[-676:, :].T
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    
    # Panel 1: 2D Cross section
    pcm_sw = ax[0].pcolormesh(samples, lev, sw_recon_last, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0), shading='auto')
    
    # Validating data is overlaid as contour lines
    # Only contour if valid data has variance to prevent errors
    if np.ptp(sw_valid_last) > 0:
        levels_sw = np.linspace(np.min(sw_valid_last), np.max(sw_valid_last), 9)
        cs_sw = ax[0].contour(samples, lev, sw_valid_last, levels=levels_sw, colors='black', linewidths=0.8, alpha=0.8)
        ax[0].clabel(cs_sw, inline=True, fontsize=8, fmt='%.1f')
    
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Pressure (hPa)")
    ax[0].set_xlabel("Sample Index (last 676)")
    ax[0].set_title("SW Heating Rate: Reconstruction (Color) vs Validation (Contours)")
    fig.colorbar(pcm_sw, ax=ax[0], label="SW Heating Rate")
    
    # Panel 2: Mean Vertical Profile (1D)
    sw_recon_mean = np.mean(sw_recon_last, axis=1)
    sw_valid_mean = np.mean(sw_valid_last, axis=1)
    
    ax[1].plot(sw_valid_mean, lev, color='black', label='Validation (Truth)', linewidth=2)
    ax[1].plot(sw_recon_mean, lev, color='red', linestyle='--', label='Reconstruction', linewidth=2)
    
    ax[1].axvline(0, color='gray', linestyle='--', linewidth=1)
    ax[1].set_xlabel("Mean SW Heating Rate")
    ax[1].set_title("Mean Vertical Profile (over 676 samples)")
    ax[1].legend()
    
    plt.savefig(fig_path / "sw_reconstruct_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------
    # Save file
    # ------------------------------------------------

    # save file
    save_path: Path = Path("/home/b11209013/KW_CloudSat/Files/Linear_Relation/")

    np.save(save_path / "M_lw.npy", M_lw)
    np.save(save_path / "M_sw.npy", M_sw)

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
