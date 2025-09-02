from __future__ import annotations

import os
from typing import List, Dict, Optional, Tuple

import geopandas as gpd
from shapely.geometry import mapping
from pystac_client import Client
import stackstac
import xarray as xr
import numpy as np


STAC_URL = os.environ.get("STAC_URL", "https://earth-search.aws.element84.com/v1")
S2_COLLECTION = "sentinel-2-l2a"


def load_aoi(aoi_file: Optional[str] = None, aoi_geojson: Optional[Dict] = None) -> gpd.GeoDataFrame:
    """Load an AOI from a file (GeoJSON/GPKG/Shapefile) or a GeoJSON dict; return EPSG:4326."""
    if aoi_geojson is not None:
        gdf = gpd.GeoDataFrame.from_features(aoi_geojson["features"], crs="EPSG:4326")
        return gdf.to_crs(4326)
    if aoi_file:
        gdf = gpd.read_file(aoi_file)
        if gdf.crs is None:
            gdf.set_crs(4326, inplace=True)
        return gdf.to_crs(4326)
    raise ValueError("Provide either aoi_file or aoi_geojson")


def aoi_bounds_latlon(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in EPSG:4326."""
    return tuple(gdf.to_crs(4326).total_bounds.tolist())


def _utm_from_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def search_items(
    aoi_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    max_cloud: int = 40,
    limit: int = 50,
) -> List:
    """Search STAC for Sentinel-2 L2A items intersecting the AOI and date window."""
    client = Client.open(STAC_URL)

    geom_obj = aoi_gdf.to_crs(4326).unary_union
    if getattr(geom_obj, "geom_type", "") == "GeometryCollection":
        geom_obj = geom_obj.buffer(0)
    geom = mapping(geom_obj)

    dt = f"{start_date}/{end_date}"
    search = client.search(
        collections=[S2_COLLECTION],
        intersects=geom,
        datetime=dt,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=limit,
    )
    # Use items() to avoid deprecation warnings
    return list(search.items())


def _common_assets(items: List, requested: List[str]) -> List[str]:
    """Keep only assets present across all items; avoids errors on missing bands."""
    if not items:
        return requested
    common = set(requested)
    for it in items:
        common &= set(it.assets.keys())
    # Prefer at least RGB + nir; SCL is expected on L2A, swir16 often present.
    filtered = [a for a in requested if a in common]
    if not filtered:
        # Fallback minimal set
        for candidate in (["red", "green", "blue", "nir", "scl"], ["red", "green", "blue", "nir"]):
            if all(c in common for c in candidate):
                return candidate
        return [a for a in requested if a in set.union(*(set(it.assets.keys()) for it in items))]
    return filtered


def stack_sentinel(
    items: List,
    assets=("red", "green", "blue", "nir", "scl", "swir16"),
    resolution: int = 10,
    epsg: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> xr.DataArray:
    """
    Stack selected assets into an xarray DataArray (time, band, y, x).
    - Force metric CRS (UTM) via epsg to avoid CRS-less items.
    - Disable stackstac rescaling; rescale reflectance later ourselves.
    - dtype=float32 with float32 NaN fill_value.
    - chunksize: int pixels or "auto".
    - Filter assets to those common across all items.
    """
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

    # 1) assets as list + filter to common assets across items
    assets_list = _common_assets(items, list(assets))

    # 2) choose EPSG from item centroid if not provided
    if epsg is None:
        try:
            import shapely.geometry as shg
            center = shg.shape(items[0].geometry).centroid
            lon, lat = float(center.x), float(center.y)
            epsg = _utm_from_lonlat(lon, lat)
        except Exception:
            epsg = 32633  # fallback

    # 3) normalize chunksize
    chunks_param = chunksize if (isinstance(chunksize, int) and chunksize > 0) else "auto"

    da = stackstac.stack(
        items,
        assets=assets_list,
        resolution=resolution,
        epsg=epsg,
        chunksize=chunks_param,             # int or "auto"
        rescale=False,                      # avoid scale/offset casting inside stackstac
        dtype="float32",
        fill_value=np.float32(np.nan),      # dtype-consistent NaN
    )
    return da


def ds_for_time_index(da: xr.DataArray, idx: int) -> xr.Dataset:
    """
    Convert a single time slice to a band-named Dataset.
    Rescale reflectance bands (only) 0..10000 → 0..1 as float32.
    Ensure SCL is an integer array; replace NaNs with 0 (S2 SCL 'no data').
    """
    slice_da = da.isel(time=idx)
    band_names = [b for b in slice_da.band.values]
    ds = slice_da.to_dataset(dim="band").rename_vars({b: str(b) for b in band_names})

    # rioxarray accessor
    try:
        import rioxarray  # noqa: F401
    except Exception:
        pass

    # Fix SCL first: NaNs → 0, then cast to int16
    if "scl" in ds:
        ds["scl"] = ds["scl"].fillna(0).astype("int16")

    # Rescale reflectance bands to 0..1
    for band in ("red", "green", "blue", "nir", "swir16"):
        if band in ds:
            ds[band] = (ds[band] / 10000.0).astype("float32", copy=False)

    return ds