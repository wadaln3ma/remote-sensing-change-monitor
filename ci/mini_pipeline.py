# ci/mini_pipeline.py
# Minimal smoke test: search STAC, stack one Sentinel-2 item, compute NDVI, export GeoTIFF+PNG.

import os
import sys
import json
import tempfile
import warnings
from datetime import datetime, timedelta

# --- ensure repo root is importable (so "app.utils..." works) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ----------------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)

from app.utils.stac_utils import load_aoi, search_items, stack_sentinel, ds_for_time_index
from app.utils.processing import compute_indices
from app.utils.viz import save_index_png

# Tiny AOI around Dubai (polygon bbox)
AOI = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [55.17, 25.13],
                        [55.39, 25.13],
                        [55.39, 25.33],
                        [55.17, 25.33],
                        [55.17, 25.13],
                    ]
                ],
            },
        }
    ],
}

def main() -> None:
    gdf = load_aoi(aoi_geojson=AOI)

    end = datetime.utcnow().date()
    start = end - timedelta(days=180)

    items = search_items(gdf, start.isoformat(), end.isoformat(), max_cloud=30, limit=3)
    assert len(items) > 0, "No STAC items returned in CI search"

    da = stack_sentinel(items[:1], resolution=10, chunksize="auto")
    da = da.rio.write_crs(da.rio.crs or da.rio.estimate_utm_crs())
    da = da.rio.clip(gdf.to_crs(da.rio.crs).geometry, gdf.to_crs(da.rio.crs).crs, drop=True)

    ds = ds_for_time_index(da, 0)
    idx = compute_indices(ds)
    ndvi = idx["NDVI"]

    outdir = tempfile.mkdtemp(prefix="mini_pipeline_")
    tif_path = os.path.join(outdir, "ndvi.tif")
    png_path = os.path.join(outdir, "ndvi.png")

    ndvi.rio.to_raster(tif_path, compress="deflate")
    save_index_png(ndvi, png_path)

    print(json.dumps({"ok": True, "outdir": outdir, "files": [tif_path, png_path]}))

if __name__ == "__main__":
    main()
