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

        # load radiative heating
        lw: np.ndarray = rad_ds.variables["QLW"][:, :, lat_lim, :][..., lon_lim]

        print(lw.shape)

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
