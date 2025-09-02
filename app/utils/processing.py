from __future__ import annotations

from typing import Dict
import numpy as np
import xarray as xr

CLOUD_SCL_CLASSES = [3, 8, 9, 10, 11]  # shadow, cloud prob, high cloud, cirrus, snow


def _safe_ratio(n: xr.DataArray, d: xr.DataArray) -> xr.DataArray:
    eps = 1e-6
    return (n - d) / (n + d + eps)


def cloud_mask_from_scl(scl: xr.DataArray) -> xr.DataArray:
    """
    Returns boolean mask where True = valid (not cloud/shadow/snow/no-data).
    Uses xarray's dask-aware .isin() to avoid apply_ufunc issues.
    """
    s = scl.astype("int16")
    valid = ~s.isin(CLOUD_SCL_CLASSES)
    valid = valid & (s != 0)  # drop nodata = 0
    return valid


def compute_indices(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    """Expects variables: red, green, blue, nir, (optional: swir16), scl"""
    red = ds["red"]
    green = ds["green"]
    nir = ds["nir"]
    swir = ds.get("swir16", None)
    scl = ds.get("scl", None)

    valid = cloud_mask_from_scl(scl) if scl is not None else xr.ones_like(red, dtype=bool)

    ndvi = _safe_ratio(nir, red).where(valid)
    ndwi = _safe_ratio(green, nir).where(valid)  # McFeeters
    ndbi = _safe_ratio(swir, nir).where(valid) if swir is not None else None

    out = {"NDVI": ndvi, "NDWI": ndwi}
    if ndbi is not None:
        out["NDBI"] = ndbi
    return out


def change_map(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    return (b - a).astype("float32")
