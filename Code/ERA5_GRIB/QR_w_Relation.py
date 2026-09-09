"""Fit LW/SW linear response matrices to ERA5-GRIB vertical velocity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.cross_decomposition import PLSRegression

from workflow import COMPOSITE_DIR, FIGURE_DIR, FILES_DIR, pressure_hpa


def load_samples(data_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinate_path = COMPOSITE_DIR / "coordinates.npz"
    if not coordinate_path.exists():
        raise FileNotFoundError("Run ERA5_composite.py and QR_composite.py first")
    with np.load(coordinate_path) as coordinates:
        pressure = np.asarray(coordinates["pressure"])

    bands = sorted(path for path in COMPOSITE_DIR.glob("k=*~*") if path.is_dir())
    required = ("w.npy", "LW.npy", "SW.npy")
    bands = [path for path in bands if all((path / data_type / name).exists() for name in required)]
    if not bands:
        raise FileNotFoundError(f"No complete {data_type!r} datasets found in {COMPOSITE_DIR}")

    w_parts, lw_parts, sw_parts = [], [], []
    for band in bands:
        w = np.load(band / data_type / "w.npy", mmap_mode="r")
        lw = np.load(band / data_type / "LW.npy", mmap_mode="r")
        sw = np.load(band / data_type / "SW.npy", mmap_mode="r")
        if data_type == "concat":
            w_parts.append(np.asarray(w).transpose(0, 2, 1).reshape(-1, w.shape[1]))
            lw_parts.append(np.asarray(lw).transpose(0, 2, 1).reshape(-1, lw.shape[1]))
            sw_parts.append(np.asarray(sw).transpose(0, 2, 1).reshape(-1, sw.shape[1]))
        else:
            w_parts.append(np.asarray(w).T)
            lw_parts.append(np.asarray(lw).T)
            sw_parts.append(np.asarray(sw).T)

    X, lw, sw = map(lambda parts: np.concatenate(parts, axis=0), (w_parts, lw_parts, sw_parts))
    valid = np.all(np.isfinite(X) & np.isfinite(lw) & np.isfinite(sw), axis=1)
    return X[valid], lw[valid], sw[valid], pressure


def metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = np.sum((actual - predicted) ** 2)
    total = np.sum((actual - actual.mean(axis=0)) ** 2)
    return float(np.sqrt(np.mean((actual - predicted) ** 2))), float(1.0 - residual / total)


def main(data_type: str, components: int) -> None:
    X, lw, sw, pressure = load_samples(data_type)
    split = int(0.8 * X.shape[0])
    if split <= components or split == X.shape[0]:
        raise ValueError("Not enough samples for the requested model")
    X_train, X_test = X[:split], X[split:]
    lw_train, lw_test = lw[:split], lw[split:]
    sw_train, sw_test = sw[:split], sw[split:]

    models = {
        "LW": PLSRegression(n_components=components, scale=True).fit(X_train, lw_train),
        "SW": PLSRegression(n_components=components, scale=True).fit(X_train, sw_train),
    }
    targets = {"LW": lw_test, "SW": sw_test}
    output_dir = FILES_DIR / "linear_response" / data_type
    figure_dir = FIGURE_DIR / "linear_response" / data_type
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    statistics = []
    p_hpa = pressure_hpa(pressure)
    for name, model in models.items():
        prediction = model.predict(X_test)
        rmse, r2 = metrics(targets[name], prediction)
        # sklearn applies coef_ after standardizing X. Convert it back so the
        # saved matrix maps physical w (m/s) directly to heating (K/day).
        operator = np.asarray(model.coef_) / np.asarray(model._x_std)[None, :]
        intercept = np.asarray(model.intercept_) - operator @ np.asarray(model._x_mean)
        np.save(output_dir / f"M_{name.lower()}.npy", operator)
        np.save(output_dir / f"b_{name.lower()}.npy", intercept)
        statistics.append({"component": name, "rmse": rmse, "r2": r2, "n_components": components})

        limit = max(float(np.nanmax(np.abs(operator))), np.finfo(float).eps)
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.pcolormesh(
            p_hpa, p_hpa, operator, shading="nearest", cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        ax.set(xlabel="Input w pressure (hPa)", ylabel=f"Output {name} pressure (hPa)",
               title=f"{name} response ({components} PLS components)")
        ax.invert_xaxis()
        ax.invert_yaxis()
        fig.colorbar(image, ax=ax, label=f"{name} response per (m s$^{{-1}}$)")
        fig.savefig(figure_dir / f"M_{name.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"{name}: test RMSE={rmse:.4g}, test R²={r2:.4f}")

    pd.DataFrame(statistics).to_csv(output_dir / "metrics.csv", index=False)
    np.save(output_dir / "pressure.npy", pressure)
    print(f"Saved response matrices and metrics to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-type", choices=("composite", "concat"), default="composite")
    parser.add_argument("--components", type=int, default=3)
    arguments = parser.parse_args()
    main(arguments.data_type, arguments.components)
