import numpy as np
from matplotlib import pyplot as plt

def set_matplotlib_style():
    """Sets the global matplotlib style."""
    plt.style.use("~/KW_CloudSat/scientific.mplstyle")

def get_background_profiles():
    """
    Constructs background thermodynamic profiles based on standard atmosphere.
    Returns:
        z   : vertical coordinate in meters
        T   : background temperature profile
        p   : pressure profile based on hydrostatics
        rho : density profile
    """
    z = np.linspace(0.0, 14000.0, 71)
    T = 300.0 - 0.0065 * z
    p = 1e5 * (1 - 0.0065 * z / 300.0) ** (9.81 / 0.0065 / 287.5)
    rho = p / T / 287.5
    return z, T, p, rho

def get_vertical_basis(z):
    """
    Calculates the two primary vertical basis functions for vertical motion.
    Returns:
        G1, G2
    """
    G1 = np.pi / 2 * np.sin(np.pi * z / z.max())
    G2 = np.pi / 2 * np.sin(2 * np.pi * z / z.max())
    return G1, G2
