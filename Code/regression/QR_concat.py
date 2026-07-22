# ====================================================
# This script is to test the new algorithm of radiative 
# feedback
# ====================================================

# ====================================================
# Environment Setup
# ====================================================

# Limit CPU usage
import os
import sys

# Import package
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc

from tqdm import tqdm
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
    
    # Load Radiative heating data
    with nc.Dataset(
            "/data92/b11209013/CloudSat/DATA/QR_gridded.nc",
            "r"
            ) as rad_ds:

        # dimensions
        ## Load overall dimensions
        rad_dims: Dict[str, np.ndarray] = {
                key: rad_ds.variables[key][...]
                for key in rad_ds.variables.keys()
                if key in rad_ds.dimensions.keys()
                }

        ## limit to tropical ocean
        lat_lim: np.ndarray = np.where(
                (rad_dims["lat"] > -15.0) &
                (rad_dims["lat"] < 15.0)
                )[0]

        lon_lim: np.ndarray = np.where(
                (rad_dims["lon"] > 150.0) & 
                (rad_dims["lon"] < 260.0)
                )[0]

        # Create contiguous slice objects for fast block reading
        lat_slice = slice(lat_lim[0], lat_lim[-1] + 1)
        lon_slice = slice(lon_lim[0], lon_lim[-1] + 1)

        # load radiative heating efficiently in a single, unchained read
        lw: np.ndarray = rad_ds.variables["QLW"][..., lat_slice, lon_slice]
        sw: np.ndarray = rad_ds.variables["QSW"][..., lat_slice, lon_slice]

    # ------------------------------------------------
    # Concatenate through different longitude index
    # ------------------------------------------------
    # Pre-allocate list for indices
    lw_valid_idx: Dict[int, Dict[str, np.ndarray]] = {}
    sw_valid_idx: Dict[int, Dict[str, np.ndarray]] = {}

    # use the last vertical level to find valid grid.
    for t in tqdm(range(lw.shape[0])):

        ## Find valid grid at each time slice
        lw_row_idx_tmp, lw_col_idx_tmp = np.where(~np.isnan(lw[t, -1]))
        sw_row_idx_tmp, sw_col_idx_tmp = np.where(~np.isnan(sw[t, -1]))

        ## save into dictionary
        lw_valid_idx[t] = {
                "row": lw_row_idx_tmp,
                "col": lw_row_idx_tmp
                }
        sw_valid_idx[t] = {
                "row": sw_row_idx_tmp,
                "col": sw_col_idx_tmp
                }

    # ------------------------------------------------
    # save valid indices
    # ------------------------------------------------
    with open(
            "/data92/b11209013/CloudSat/DATA/tropical_valid/lw_valid_idx.json",
            "w"
            ) as lw_file:
        json.dump(lw_valid_idx, lw_file, indent=4)

    with open(
            "/data92/b11209013/CloudSat/DATA/tropical_valid/lw_valid_idx.json",
            "w"
            ) as lw_file:
        json.dump(lw_valid_idx, lw_file, indent=4)




# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
