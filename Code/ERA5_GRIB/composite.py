# ====================================================
# Composite Analysis on Pressure Velocity
# ====================================================

# ====================================================
# Import package
# ====================================================
import os

import numpy as np
import netCDF4 as nc

from glob import glob
from pathlib import Path
from tqdm import tqdm
from typing import Any, List, Dict

from matplotlib import pyplot as plt

# ====================================================
# Helper functions
# ====================================================


# ====================================================
# Main function
# ====================================================

def main(var:str) -> None:
    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    # Path setup
    root_dir: Path = Path("/home/b11209013/KW_CloudSat")

    # load Kelvin waves Events
    event_Files = glob(f"{root_dir}/Files/ERA5_GRIB/KW_events/*.txt")

    # Load omega data using time slice
    with nc.Dataset(f"/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10/{var}.nc") as ds:

        ## Load coordinate information
        coords: Dict = {
            key: ds.variables[key][...]
            for key in ds.dimensions.keys()
        }

        ## Load data
        data: np.ndarray = ds.variables[var][...]

    print("Finished: Load omega fields")

    # ------------------------------------------------
    # Pre-processing
    # ------------------------------------------------
    # remove temporal and zonal mean
    data_anom   : np.ndarray = data - np.nanmean(data, axis=(0, -1), keepdims=True)

    # ------------------------------------------------
    # Composite 
    # ------------------------------------------------

    # Rolling data based on events and bands
    lon_half_idx: int = int(coords["lon"].size//2)

    composite: dict = {}

    for event_file in event_Files:

        print(str(event_file)+ "start")

        # Load events data
        events: np.ndarray = np.loadtxt(event_file)

        # convert the longitude into index
        lon_indices: np.ndarray = np.array([
            np.argmin(np.abs(coords["lon"] - l))
            for l in events[1, :]
        ])

        # Roll data
        tmp_store: List = []

        for e_idx in tqdm(range(events.shape[1])):
            tidx = events[0, e_idx] # time and lon index

            xidx = lon_indices[e_idx]

            tmp_store.append(np.nanmean(np.roll(data_anom[int(tidx)],  int(lon_half_idx) - int(xidx), axis=-1), axis=1))

        # composite
        composite[str(event_file).split("/")[-1].split(".")[0]] = np.nanmean(np.array(tmp_store), axis=0)

    # ------------------------------------------------
    # Output composite proefiles
    # ------------------------------------------------

    # output path
    save_path: Path = root_dir / f"Files/ERA5_GRIB/composite/{var}"

    os.makedirs(save_path, exist_ok=True)

    for key in composite.keys():
        np.savetxt(str(save_path / f"{key}.txt"), composite[key])    

    if var == "W":
        np.savetxt(str(save_path / "plev.txt"), coords["plev"])

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":

    vars: list = ["W", "T", "Z"]

    for var in vars:
        print(f"start processing variable {var}")
        main(var)