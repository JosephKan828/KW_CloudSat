import numpy as np
from glob import glob
from pathlib import Path

root_dir = Path("/home/b11209013/KW_CloudSat/Files/")
lw = np.concatenate([np.load(f + "/concat/LW.npy") for f in glob(str(root_dir / "QR_composite/*"))], axis=0)
w = np.concatenate([np.load(f) for f in glob(str(root_dir / "w_concate/*.npy"))], axis=0)

lw_concate = np.concatenate([lw[i] for i in range(lw.shape[0])], axis=-1)
w_concate = np.concatenate([w[i] for i in range(lw.shape[0])], axis=-1)

lw_non_nan = np.all(~np.isnan(lw_concate), axis=0)
lw_valid = lw_concate[:, lw_non_nan]
w_lw_valid = w_concate[:, lw_non_nan]

for p in [0.5, 0.1]:
    lw_low = np.percentile(lw_valid, p, axis=1, keepdims=True)
    lw_high = np.percentile(lw_valid, 100-p, axis=1, keepdims=True)
    w_low = np.percentile(w_lw_valid, p, axis=1, keepdims=True)
    w_high = np.percentile(w_lw_valid, 100-p, axis=1, keepdims=True)
    
    lw_good = ~np.any((lw_valid < lw_low) | (lw_valid > lw_high), axis=0)
    lw_good &= ~np.any((w_lw_valid < w_low) | (w_lw_valid > w_high), axis=0)
    print(f"Per-level Percentile {p}-{100-p}: Kept {np.sum(lw_good)} out of {lw_valid.shape[1]} ({(np.sum(lw_good)/lw_valid.shape[1])*100:.2f}%)")

