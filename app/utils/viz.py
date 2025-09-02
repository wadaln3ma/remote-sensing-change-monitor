from __future__ import annotations

import os
from typing import Optional
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def _normalize_img(arr: np.ndarray, q=(2, 98)) -> np.ndarray:
    """Percentile-based linear stretch to 0..1 for display."""
    if arr.ndim == 3:
        out = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            p2, p98 = np.nanpercentile(arr[i], q)
            out[i] = np.clip((arr[i] - p2) / max(p98 - p2, 1e-6), 0, 1)
        return out
    else:
        p2, p98 = np.nanpercentile(arr, q)
        return np.clip((arr - p2) / max(p98 - p2, 1e-6), 0, 1)


def save_true_color_png(ds: xr.Dataset, path: str, valid_mask: Optional[xr.DataArray] = None):
    r = ds["red"].values
    g = ds["green"].values
    b = ds["blue"].values
    rgb = np.stack([r, g, b], axis=0)
    rgb = _normalize_img(rgb)
    rgb = np.transpose(rgb, (1, 2, 0))  # HWC
    if valid_mask is not None:
        vm = valid_mask.values
        rgb[~vm] = 0.0
    plt.figure(figsize=(6, 6), dpi=150)
    plt.axis("off")
    plt.imshow(rgb)
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_index_png(da: xr.DataArray, path: str, vmin: float = -1.0, vmax: float = 1.0, cmap: str = "RdYlGn"):
    plt.figure(figsize=(6, 6), dpi=150)
    plt.axis("off")
    arr = da.values
    plt.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
