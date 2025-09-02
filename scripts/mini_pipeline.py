# Mini CI pipeline:
# - Load demo AOI
# - Search a small date window
# - Stack one item
# - Compute NDVI
# - Export GeoTIFF + PNG to artifacts/

import os
from datetime import date, timedelta

from app.utils.stac_utils import load_aoi, search_items, stack_sentinel, ds_for_time_index
from app.utils.processing import compute_indices, cloud_mask_from_scl
from app.utils.viz import save_true_color_png, save_index_png

import rioxarray  # noqa: F401


def main():
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    here = os.path.dirname(os.path.dirname(__file__))
    demo_file = os.path.join(here, "data", "demo", "aoi.geojson")
    aoi = load_aoi(aoi_file=demo_file)

    end = date.today()
    start = end - timedelta(days=30)
    items = search_items(aoi, start.isoformat(), end.isoformat(), max_cloud=30, limit=10)
    if not items:
        raise SystemExit("No Sentinel-2 items found in CI window.")

    item = items[0]
    da = stack_sentinel([item], resolution=10, chunksize="auto")
    da = da.rio.write_crs(da.rio.crs or da.rio.estimate_utm_crs())
    da = da.rio.clip(aoi.to_crs(da.rio.crs).geometry, aoi.to_crs(da.rio.crs).crs, drop=True)
    ds = ds_for_time_index(da, 0)

    idx = compute_indices(ds)
    ndvi = idx["NDVI"]

    artifacts = os.path.join(here, "artifacts")
    os.makedirs(artifacts, exist_ok=True)

    ndvi_tif = os.path.join(artifacts, "ndvi.tif")
    ndvi.rio.to_raster(ndvi_tif, compress="deflate")

    rgb_png = os.path.join(artifacts, "truecolor.png")
    valid = cloud_mask_from_scl(ds["scl"]) if "scl" in ds else None
    save_true_color_png(ds, rgb_png, valid)

    ndvi_png = os.path.join(artifacts, "ndvi.png")
    save_index_png(ndvi, ndvi_png)

    print("Artifacts written to:", artifacts)


if __name__ == "__main__":
    main()
