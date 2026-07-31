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
import utils

utils.set_matplotlib_style()

# ====================================================
# Main function
# ====================================================

def main(data_type: str) -> None:
    
    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    root_dir: Path = Path("/home/b11209013/KW_CloudSat/")
    input_dir: Path =  root_dir / "Files/"

    # Load vertical motion as dictionary
    match data_type:
        case "concat":
            fname_w: List[str] = list(glob(str(input_dir / "w_concate/*.npy")))
        case "composite":
            fname_w: List[str] = list(glob(str(input_dir / "w_composite/*.npy")))
        case _:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'concat' or 'composite'.")

    w: Dict[str, np.ndarray] = {
            fname.split("/")[-1].split(".")[0]: np.load(fname)[...]
            for fname in fname_w
            }  

    match data_type:
        case "concat":
            _, nz, nx = w["k=1~3"].shape
        case "composite":
            nz, nx = w["k=1~3"].shape
        case _:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'concat' or 'composite'.")

    # Load radiative heating rate
    fname_qr: List[str] = list(glob(str(input_dir / "QR_composite/k*")))

    lw: Dict[str, np.ndarray] = {
            fname.split("/")[-1]: np.load(fname+f"/{data_type}/LW.npy")[...]
            for fname in fname_qr
            }

    sw: Dict[str, np.ndarray] = {
            fname.split("/")[-1]: np.load(fname+f"/{data_type}/SW.npy")[...]
            for fname in fname_qr
            }

    # ------------------------------------------------
    # Concatenate data
    # ------------------------------------------------

    keys = sorted(w.keys())
    
    # if the data type is `concat`, first transform the
    # data shape into nz nsample

    match data_type:
        case "concat":
            w_concat: np.ndarray = np.concatenate([w[k].transpose(0, 2, 1).reshape(-1, nz) for k in keys], axis=0)
            lw_concat: np.ndarray = np.concatenate([lw[k].transpose(0, 2, 1).reshape(-1, nz) for k in keys], axis=0)
            sw_concat: np.ndarray = np.concatenate([sw[k].transpose(0, 2, 1).reshape(-1, nz) for k in keys], axis=0)

            lw_non_nan : np.ndarray = np.all(~np.isnan(lw_concat), axis=1)
            sw_non_nan : np.ndarray = np.all(~np.isnan(sw_concat), axis=1)
            non_nan_idx: np.ndarray = lw_non_nan & sw_non_nan

            # Calculate standard derivation in LW and SW heating
            lw_std: float = float(np.nanstd(lw_concat[non_nan_idx, :]))
            sw_std: float = float(np.nanstd(sw_concat[non_nan_idx, :]))

            lw_norm_idx: np.ndarray = np.all(np.abs(lw_concat[non_nan_idx, :]) < 3 * lw_std, axis=1)
            sw_norm_idx: np.ndarray = np.all(np.abs(sw_concat[non_nan_idx, :]) < 3 * sw_std, axis=1) 
            norm_idx: np.ndarray = lw_norm_idx & sw_norm_idx
            
            w_concat = w_concat[non_nan_idx, :][norm_idx, :]
            lw_concat = lw_concat[non_nan_idx, :][norm_idx, :]
            sw_concat = sw_concat[non_nan_idx, :][norm_idx, :]

            print(w_concat.shape)

        case "composite":
            w_concat: np.ndarray = np.concatenate([w[k].T for k in keys])
            lw_concat: np.ndarray = np.concatenate([lw[k].T for k in keys])
            sw_concat: np.ndarray = np.concatenate([sw[k].T for k in keys])
        case _:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'concat' or 'composite'.")

    # ------------------------------------------------
    # Split data into training and verifying
    # ------------------------------------------------

    # training size: first 5/6 of total samples, verifying size: last 1/6 of total samples
    n_sample_train: int = w_concat.shape[0] * 5 // 6
    n_sample_valid: int = w_concat.shape[0] - n_sample_train

    w_train : np.ndarray = w_concat[:n_sample_train] ; w_valid : np.ndarray = w_concat[n_sample_train:n_sample_train + n_sample_valid]
    lw_train: np.ndarray = lw_concat[:n_sample_train]; lw_valid: np.ndarray = lw_concat[n_sample_train:n_sample_train + n_sample_valid]
    sw_train: np.ndarray = sw_concat[:n_sample_train]; sw_valid: np.ndarray = sw_concat[n_sample_train:n_sample_train + n_sample_valid]

    # ------------------------------------------------
    # Use Partial Least Squares (PLS) regression
    # ------------------------------------------------

    from sklearn.cross_decomposition import PLSRegression
    
    # Fit PLS models for LW and SW (using 4 components)
    # scale=False means it will center the data but not normalize by standard deviation
    n_components: int = 4
    
    pls_lw = PLSRegression(n_components=n_components, scale=False)
    pls_sw = PLSRegression(n_components=n_components, scale=False)
    
    pls_lw.fit(w_train, lw_train)
    pls_sw.fit(w_train, sw_train)
    
    # Extract Jacobian Matrices (equivalent to the M_lw and M_sw matrices)
    # PLS prediction is internally: Y_pred = (X - X_mean) @ coef_ + Y_mean
    # Since we want M_lw such that M_lw @ w.T matches this, we transpose coef_
    M_lw: np.ndarray = pls_lw.coef_.T
    M_sw: np.ndarray = pls_sw.coef_.T

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
    fig_path: Path = root_dir / "Figure" / "QR_w_Relation" / data_type
    os.makedirs(fig_path, exist_ok=True)

    # M_lw
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    lw_pcm = ax.pcolormesh(
            lev, lev, M_lw,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0),
            shading='nearest'
            )
    ax.set_xlim(1000, 100)
    ax.set_ylim(1000, 100)
    ax.set_xlabel("Input w Pressure (hPa)")
    ax.set_ylabel("Output LW Heating Pressure (hPa)")
    ax.set_title("Jacobian Matrix ($M_{LW}$)")
    cbar = fig.colorbar(lw_pcm, ax=ax)
    cbar.set_label("Response Magnitude")
    plt.tight_layout()
    plt.savefig(fig_path / "M_lw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # M_sw
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sw_pcm = ax.pcolormesh(
            lev, lev, M_sw,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0),
            shading='nearest'
            )
    ax.set_xlim(1000, 100)
    ax.set_ylim(1000, 100)
    ax.set_xlabel("Input w Pressure (hPa)")
    ax.set_ylabel("Output SW Heating Pressure (hPa)")
    ax.set_title("Jacobian Matrix ($M_{SW}$)")
    cbar = fig.colorbar(sw_pcm, ax=ax)
    cbar.set_label("Response Magnitude")
    plt.tight_layout()
    plt.savefig(fig_path / "M_sw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------
    # Visualize the validation
    # ------------------------------------------------
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Calculate global min/max for identity line
    lw_min, lw_max = np.min(lw_valid), np.max(lw_valid)
    sw_min, sw_max = np.min(sw_valid), np.max(sw_valid)

    ax[0].scatter(lw_recon.T, lw_valid, alpha=0.1, s=3, color='steelblue')
    ax[0].plot([lw_min, lw_max], [lw_min, lw_max], 'k--', lw=1.5, zorder=3)
    ax[1].scatter(sw_recon.T, sw_valid, alpha=0.1, s=3, color='steelblue')
    ax[1].plot([sw_min, sw_max], [sw_min, sw_max], 'k--', lw=1.5, zorder=3)

    corr_lw = np.corrcoef(lw_valid.flatten(), lw_recon.T.flatten())[0, 1]
    corr_sw = np.corrcoef(sw_valid.flatten(), sw_recon.T.flatten())[0, 1]

    ax[0].set_title(r"LW Heating Validation ($R$ = " + f"{corr_lw:.3f})")
    ax[1].set_title(r"SW Heating Validation ($R$ = " + f"{corr_sw:.3f})")

    for i in range(2):
        ax[i].set_xlabel("Reconstructed (K/day)")
        ax[i].set_ylabel("Validation (K/day)")
        ax[i].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(fig_path / "verify.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------
    # Statistical Verification of Correlation Profile
    # ------------------------------------------------

    def vectorized_col_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Computes column-wise Pearson correlation between two 2D arrays."""
        # Center data
        A_centered = A - A.mean(axis=0, keepdims=True)
        B_centered = B - B.mean(axis=0, keepdims=True)
        
        # Sum of squares of anomalies using einsum for efficiency
        ssA = np.einsum('ij,ij->j', A_centered, A_centered)
        ssB = np.einsum('ij,ij->j', B_centered, B_centered)
        
        # Numerator for correlation
        corr_num = np.einsum('ij,ij->j', A_centered, B_centered)
        
        # Denominator
        corr_den = np.sqrt(ssA * ssB)
        
        # Handle division by zero
        corr = np.divide(corr_num, corr_den, out=np.zeros_like(corr_num), where=(corr_den!=0))
        return corr

    corr_lw_profile = vectorized_col_corr(lw_valid, lw_recon.T)
    corr_sw_profile = vectorized_col_corr(sw_valid, sw_recon.T)

    # Create a pandas DataFrame for easy statistics and saving
    stats_df = pd.DataFrame({
        'pressure_hpa': lev,
        'corr_lw': corr_lw_profile,
        'corr_sw': corr_sw_profile
    })

    # Print summary statistics and save to file
    pprint("\n--- Correlation Profile Statistics ---")
    pprint("LW Correlation Summary:")
    pprint(stats_df['corr_lw'].describe())
    pprint("\nSW Correlation Summary:")
    pprint(stats_df['corr_sw'].describe())
    
    # ------------------------------------------------
    # Visualize the Correlation Score Profile
    # ------------------------------------------------
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    # Use a line plot for vertical atmospheric profiles
    ax[0].plot(corr_lw_profile, lev, marker='o', markersize=5, color='coral', linewidth=2, zorder=3)
    ax[1].plot(corr_sw_profile, lev, marker='o', markersize=5, color='skyblue', linewidth=2, zorder=3)

    ax[0].set_title(r"LW Correlation ($R$) Profile")
    ax[1].set_title(r"SW Correlation ($R$) Profile")
    
    for i in range(2):
        ax[i].set_xlabel(r"Correlation Coefficient ($R$)")
        ax[i].set_xlim(-0.1, 1.0)
        ax[i].axvline(0, color='black', linestyle='--', linewidth=1.5, zorder=2)
        ax[i].grid(True, linestyle=':', alpha=0.6)

    ax[0].set_ylabel("Pressure (hPa)")
    ax[0].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(fig_path / "corr_profile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pprint(f"Correlation profile plot saved to {fig_path / 'corr_profile.png'}")
    
    # ------------------------------------------------
    # Save file
    # ------------------------------------------------

    # save file
    folder_name = "Linear_Relation"
    save_path: Path = root_dir / "Files" / folder_name / data_type
    os.makedirs(save_path, exist_ok=True)

    stats_path = save_path / "correlation_profile_stats.csv"
    stats_df.to_csv(stats_path, index=False, float_format='%.4f')
    pprint(f"\nCorrelation statistics saved to {stats_path}\n")

    np.save(save_path / "M_lw.npy", M_lw)
    np.save(save_path / "M_sw.npy", M_sw)

# ====================================================
# Execute main function
# ====================================================

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True, choices=["concat", "composite"])
    args = parser.parse_args()
    main(args.data_type)
