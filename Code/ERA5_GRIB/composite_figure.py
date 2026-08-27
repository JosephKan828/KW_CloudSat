# ====================================================
# This script is to visualize the composite structure
# ====================================================

# ====================================================
# Import package
# ====================================================
import os

import numpy as np

from glob import glob
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ====================================================
# Helper function
# ====================================================

def _visual_composite(
        lev : np.ndarray,
        data: np.ndarray,
        var : str,
        cmap: str,
        title: str,
        save_name: str,
) -> None:

    # define the relative longitude
    nz, nx = data.shape

    rel_lon: np.ndarray = np.linspace(-180.0, 180.0, nx, endpoint=False)

    # determine for different variables

    # Figure visualization
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))

    pcm = ax.contourf(
        rel_lon, lev/100,
        data,
        cmap=cmap, norm=TwoSlopeNorm(0)
    )
    ax.minorticks_on()
    ax.set_xticks(np.linspace(-60, 60, 7))
    ax.set_xlim(-70, 70)
    ax.set_ylim(1000, 100)
    ax.set_xlabel(r"Relative Longitude [ $^\circ$ ]")
    ax.set_ylabel(r"Level [ hPa ]")
    ax.set_title(title)

    fig.colorbar(pcm, ax=ax, aspect=50, shrink=0.8)
    plt.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.close(fig)


# ====================================================
# Main function
# ====================================================

def main(var: str) -> None:
    # ------------------------------------------------
    # Load file
    # ------------------------------------------------

    # path setup
    root_dir: Path = Path("/home/b11209013/KW_CloudSat")
    file_dir: Path = root_dir / "Files" / "ERA5_GRIB" / "composite"

    # Load files
    ## Load pressure level
    plev: np.ndarray = np.loadtxt(str(file_dir / "W" / "plev.txt"))

    ## Load composite structure
    files: List = list(glob(str(file_dir / var / "(*).txt")))

    data: Dict[str, np.ndarray] = {
        file.split("/")[-1].split(".")[0]: np.loadtxt(file)
        for file in files
    }

    # ------------------------------------------------
    # Visualize the result of composite
    # ------------------------------------------------

    # save path
    fig_path: Path = root_dir / "Figure" / "ERA5_GRIB" / var

    os.makedirs(fig_path, exist_ok=True)

    # general setup
    plt.rcParams.update({
        "font.family"     : "serif",
        "xtick.direction" : "in",
        "ytick.direction" : "in",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18
    })

    for (key, val) in tqdm(data.items()):

        # determine the colormap
        if   var == "W" : cmap: str = "RdBu_r"; varname = r"$\omega$"
        elif var == "T" : cmap: str = "PiYG_r"; varname = r"$T$"
        elif var == "Z" : cmap: str = "BrBG"  ; varname = r"$Z$"
        else: raise ValueError(f"{var} is not valid")



        _visual_composite(
            plev, val, var, cmap=cmap, title=f"{varname} {key}",
            save_name = str(fig_path / f"{key}.png")
        )

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":

    vars: List[str] = ["W", "T", "Z"]

    for var in vars:

        print(f"start variable {var}")
        main(var)