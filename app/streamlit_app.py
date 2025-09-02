import os
import sys
import io
import zipfile
import uuid
from datetime import date, timedelta, datetime
from typing import Optional, Dict, List, Tuple

# --- Robust imports even if launched from anywhere --------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------------------------------------------

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import pandas as pd
import geopandas as gpd
import rioxarray  # noqa: F401
import xarray as xr

# Pillow optional (used to read PNG size in summary)
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

from app.utils.stac_utils import (
    load_aoi,
    search_items,
    stack_sentinel,
    ds_for_time_index,
)
from app.utils.processing import compute_indices, cloud_mask_from_scl, change_map
from app.utils.viz import save_true_color_png, save_index_png


# ==================== Small helpers =========================================
def _read_bytes(path: str) -> Optional[bytes]:
    """Safely read a file as bytes; return None if missing/unreadable."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def _list_files(dirpath: str) -> List[str]:
    """List files in a directory (non-recursive), sorted."""
    try:
        return sorted([fn for fn in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, fn))])
    except Exception:
        return []

def _latest_run_dir(base_dirs: List[str]) -> Optional[str]:
    """Find the most recently modified run directory under any of the given bases."""
    candidates: List[Tuple[float, str]] = []
    for base in base_dirs:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            d = os.path.join(base, name)
            if os.path.isdir(d):
                try:
                    candidates.append((os.path.getmtime(d), d))
                except Exception:
                    pass
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]

def _aoi_area_km2(gdf) -> Optional[float]:
    """Compute AOI area in km²; try equal-area, then Web Mercator fallback."""
    if gdf is None:
        return None
    try:
        # World Mollweide equal-area (EPSG:6933)
        return float(gdf.to_crs(6933).area.sum()) / 1e6
    except Exception:
        try:
            # Web Mercator fallback
            return float(gdf.to_crs(3857).area.sum()) / 1e6
        except Exception:
            return None

def _stac_props(items, scene_id) -> Dict:
    """Find properties for a scene id in a list of STAC items."""
    if not items or not scene_id:
        return {}
    for it in items:
        if getattr(it, "id", None) == scene_id:
            return getattr(it, "properties", {}) or {}
    return {}

def render_results_summary(results: Dict,
                           items: Optional[List] = None,
                           aoi_gdf=None,
                           run_dir: Optional[str] = None) -> None:
    """Render a compact metrics summary for current (or latest) run."""
    scene_a = results.get("scene_a")
    scene_b = results.get("scene_b")
    props_a = _stac_props(items, scene_a)
    props_b = _stac_props(items, scene_b)
    date_a = (props_a.get("datetime") or "")[:10] or "–"
    date_b = (props_b.get("datetime") or "")[:10] or "–"
    cloud_a = props_a.get("eo:cloud_cover")
    cloud_b = props_b.get("eo:cloud_cover")
    cloud_a_str = "–" if cloud_a is None else f"{cloud_a:.0f}"
    cloud_b_str = "–" if cloud_b is None else f"{cloud_b:.0f}"

    # AOI area
    area_km2 = _aoi_area_km2(aoi_gdf)
    area_str = "–" if area_km2 is None else f"{area_km2:.2f}"

    # Resolution
    res_str = str(results.get("resolution", "–"))

    # Image size (from PNG quicklook)
    px_str = "–"
    try:
        a_png = results.get("a_rgb_png")
        if Image and a_png and os.path.exists(a_png):
            w, h = Image.open(a_png).size
            px_str = f"{w} × {h}"
    except Exception:
        pass

    # Files & ZIP size
    run_dir = run_dir or (os.path.dirname(results.get("zip_path")) if results.get("zip_path") else None)
    files_count = len(_list_files(run_dir)) if run_dir else 0
    zip_size = None
    if results.get("zip_path") and os.path.exists(results["zip_path"]):
        try:
            zip_size = os.path.getsize(results["zip_path"]) / (1024 * 1024)
        except Exception:
            zip_size = None
    zip_str = "–" if zip_size is None else f"{zip_size:.2f} MB"

    st.markdown("#### 📋 Results summary")
    c = st.columns(6)
    with c[0]:
        st.metric("Scene A date", date_a)
    with c[1]:
        st.metric("Scene A cloud %", cloud_a_str)
    with c[2]:
        st.metric("Scene B date", date_b)
    with c[3]:
        st.metric("Scene B cloud %", cloud_b_str)
    with c[4]:
        st.metric("Resolution (m)", res_str)
    with c[5]:
        st.metric("AOI area (km²)", area_str)

    c2 = st.columns(3)
    with c2[0]:
        st.metric("Image pixels", px_str)
    with c2[1]:
        st.metric("Files in run", f"{files_count}")
    with c2[2]:
        st.metric("ZIP size", zip_str)


# ==================== App setup & state =====================================
st.set_page_config(page_title="Remote Sensing Change Monitor", layout="wide")
st.title("🛰️ Remote Sensing Change Monitor — Sentinel-2")

if "app" not in st.session_state:
    st.session_state.app = {
        "aoi_gdf": None,           # geopandas GeoDataFrame
        "items": None,             # list[pystac.Item]
        "items_df": None,          # pandas DataFrame for table
        "scene_a": None,           # selected scene ID A
        "scene_b": None,           # selected scene ID B
        "results": None,           # dict of output paths & metadata
        "last_error": None,        # store last exception for display
    }

S = st.session_state.app
OUTDIR_BASES = ["data/outputs", "data/output"]  # support both plural & singular if present
for b in OUTDIR_BASES:
    os.makedirs(b, exist_ok=True)

# ===================== 🆘 Quick Downloads (from disk) ========================
with st.expander("🆘 Quick Downloads (from disk, bypass session)", expanded=False):
    latest_dir = _latest_run_dir(OUTDIR_BASES)
    if latest_dir:
        st.write(f"Latest run directory: `{latest_dir}`")
        files = _list_files(latest_dir)

        # --- Mini-summary from disk (fallback) ---
        guess: Dict[str, str] = {}
        a_png = os.path.join(latest_dir, "A_truecolor.png")
        b_png = os.path.join(latest_dir, "B_truecolor.png")
        zipp  = os.path.join(latest_dir, "exports.zip")
        if os.path.exists(a_png): guess["a_rgb_png"] = a_png
        if os.path.exists(b_png): guess["b_rgb_png"] = b_png
        if os.path.exists(zipp):  guess["zip_path"] = zipp
        guess["resolution"] = guess.get("resolution", "–")
        st.markdown("#### 📋 Latest on disk — summary")
        render_results_summary(guess, items=None, aoi_gdf=S.get("aoi_gdf"), run_dir=latest_dir)

        # Offer ZIP first
        if "exports.zip" in files:
            zp = os.path.join(latest_dir, "exports.zip")
            zb = _read_bytes(zp)
            if zb:
                size_mb = os.path.getsize(zp) / (1024 * 1024)
                st.caption(f"ZIP size: {size_mb:.2f} MB  •  {zp}")
                st.download_button("⬇️ Download Latest ZIP", data=zb, file_name="exports.zip",
                                   mime="application/zip", use_container_width=True, key="dl_latest_zip")
        # Then per-file downloads
        st.write("Per-file:")
        for fn in files:
            if fn.lower().endswith((".png", ".tif")):
                fp = os.path.join(latest_dir, fn)
                b = _read_bytes(fp)
                if b:
                    st.download_button(f"⬇️ {fn}", data=b, file_name=fn, mime="application/octet-stream",
                                       use_container_width=True, key=f"dl_latest_{fn}")
    else:
        st.info("No previous outputs found under `data/outputs` or `data/output`.")

# ===================== Sidebar controls =====================================
with st.sidebar:
    st.header("Controls")
    st.caption("Source: Element84 Earth Search → `sentinel-2-l2a`")
    today = date.today()
    start = st.date_input("Start date", value=today - timedelta(days=60), key="start_date")
    end = st.date_input("End date", value=today, key="end_date")
    max_cloud = st.slider("Max cloud cover (%)", 0, 100, 40, step=5, key="max_cloud")
    resolution = st.selectbox("Resolution (m)", [10, 20, 60], index=0, key="resolution")
    use_dask = st.checkbox("Use Dask chunking", value=False, key="use_dask")
    chunk_size = st.number_input("Chunk size (pixels)", value=1024, step=256, key="chunk_size")
    st.divider()
    st.caption("Session")
    if st.button("🔁 Reset session", use_container_width=True):
        st.session_state.app = {
            "aoi_gdf": None, "items": None, "items_df": None,
            "scene_a": None, "scene_b": None, "results": None, "last_error": None
        }
        st.rerun()

# ===================== 0) Show existing results (if any) ====================
if S["results"]:
    st.subheader("Latest Results")

    # NEW: compact summary
    render_results_summary(S["results"], items=S.get("items"), aoi_gdf=S.get("aoi_gdf"))

    # Visual previews
    try:
        from streamlit_image_comparison import image_comparison
        st.markdown("**Before/After Swipe (true color)**")
        image_comparison(
            img1=S["results"]["a_rgb_png"],
            img2=S["results"]["b_rgb_png"],
            label1="A (older)",
            label2="B (newer)",
            key="img_compare",
        )
    except Exception:
        c1, c2 = st.columns(2)
        with c1:
            st.image(S["results"]["a_rgb_png"], caption="A (older)")
        with c2:
            st.image(S["results"]["b_rgb_png"], caption="B (newer)")

    st.markdown("**ΔNDVI Heatmap**")
    if S["results"].get("delta_png") and os.path.exists(S["results"]["delta_png"]):
        st.image(S["results"]["delta_png"], caption="ΔNDVI (B - A)")

    # Downloads
    st.subheader("Downloads")
    zip_bytes = _read_bytes(S["results"]["zip_path"])
    if zip_bytes:
        size_mb = os.path.getsize(S["results"]["zip_path"]) / (1024 * 1024)
        st.caption(f"ZIP size: {size_mb:.2f} MB  •  {S['results']['zip_path']}")
        safe_a = (S["results"].get("scene_a") or "A").replace("/", "_")
        safe_b = (S["results"].get("scene_b") or "B").replace("/", "_")
        st.download_button(
            "⬇️ Download All (ZIP)",
            data=zip_bytes,
            file_name=f"exports_{safe_a}_{safe_b}.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_all",
        )
    else:
        st.error("ZIP file missing. Use the per-file downloads below or recompute.")

    with st.expander("Per-file downloads"):
        files_to_offer: Dict[str, str] = {}
        files_to_offer["A_truecolor.png"] = S["results"]["a_rgb_png"]
        files_to_offer["B_truecolor.png"] = S["results"]["b_rgb_png"]
        if S["results"].get("delta_png"):
            files_to_offer["delta_ndvi.png"] = S["results"]["delta_png"]
        for p in (S["results"].get("indices_a_pngs") or {}).values():
            files_to_offer[os.path.basename(p)] = p
            tif = p.replace("_png", "_tif").replace(".png", ".tif")
            if os.path.exists(tif):
                files_to_offer[os.path.basename(tif)] = tif
        for p in (S["results"].get("indices_b_pngs") or {}).values():
            files_to_offer[os.path.basename(p)] = p
            tif = p.replace("_png", "_tif").replace(".png", ".tif")
            if os.path.exists(tif):
                files_to_offer[os.path.basename(tif)] = tif

        for fname, fpath in sorted(files_to_offer.items()):
            b = _read_bytes(fpath)
            if b:
                st.download_button(f"⬇️ {fname}", data=b, file_name=fname,
                                   mime="application/octet-stream", use_container_width=True, key=f"dl_{fname}")

    if st.button("🔁 Reset results", use_container_width=True):
        S["results"] = None
        st.rerun()

# ===================== 1) Area of Interest (AOI) ============================
st.subheader("1) Area of Interest (AOI)")
col_map, col_upload = st.columns([2, 1], gap="large")

with col_map:
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
    Draw(
        export=False,
        filename="aoi.geojson",
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": True,
            "circle": False,
            "polygon": True,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True},
    ).add_to(m)
    map_out = st_folium(m, width=None, height=520, key="aoi_map")
    drawn = map_out.get("all_drawings", [])
    drawn_geojson = {"type": "FeatureCollection", "features": drawn} if drawn else None

with col_upload:
    st.write("Or upload an AOI file:")
    up = st.file_uploader(
        "GeoJSON / GPKG / Shapefile (.zip)", type=["geojson", "gpkg", "zip"], key="aoi_upload"
    )
    use_demo = st.checkbox("Use demo AOI (Dubai area)", value=(not drawn and up is None), key="use_demo")

    with st.form("aoi_form"):
        set_aoi = st.form_submit_button("✅ Set AOI", use_container_width=True)

# Assign AOI
try:
    if set_aoi or (S["aoi_gdf"] is None and use_demo and not drawn and up is None):
        gdf = None
        if drawn_geojson and len(drawn_geojson["features"]) > 0:
            gdf = load_aoi(aoi_geojson=drawn_geojson)
            st.success("AOI: using drawn polygon/rectangle.")
        elif up is not None:
            if up.name.endswith(".geojson") or up.name.endswith(".gpkg"):
                tmp = f"/tmp/{up.name}"
                with open(tmp, "wb") as f:
                    f.write(up.getbuffer())
                gdf = load_aoi(aoi_file=tmp)
            elif up.name.endswith(".zip"):
                import tempfile, glob, zipfile as _zip
                tmp = f"/tmp/{up.name}"
                with open(tmp, "wb") as f:
                    f.write(up.getbuffer())
                with _zip.ZipFile(tmp) as zf:
                    tdir = tempfile.mkdtemp()
                    zf.extractall(tdir)
                    shp = glob.glob(os.path.join(tdir, "*.shp"))
                    if shp:
                        gdf = load_aoi(aoi_file=shp[0])
                    else:
                        st.error("No .shp found in zip.")
            if gdf is not None:
                st.success("AOI: using uploaded file.")
        elif use_demo:
            demo_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "demo", "aoi.geojson"))
            gdf = load_aoi(aoi_file=demo_file)
            st.info("AOI: loaded demo area (Dubai).")

        if gdf is not None and not gdf.empty:
            S["aoi_gdf"] = gdf
            S["items"], S["items_df"], S["scene_a"], S["scene_b"], S["results"] = None, None, None, None, None
            st.rerun()
except Exception as e:
    S["last_error"] = e
    st.error(f"AOI error: {e}")

aoi_ready = S["aoi_gdf"] is not None and not (S["aoi_gdf"].empty if isinstance(S["aoi_gdf"], gpd.GeoDataFrame) else False)

# ===================== 2) STAC Search =======================================
st.subheader("2) STAC Search")
with st.form("search_form"):
    st.write("Filter by date & cloud, then search.")
    do_search = st.form_submit_button("🔎 Search STAC", use_container_width=True, disabled=not aoi_ready)

if do_search and aoi_ready:
    try:
        with st.spinner("Searching STAC…"):
            items = search_items(S["aoi_gdf"], start.isoformat(), end.isoformat(), max_cloud=max_cloud, limit=40)
        if not items:
            st.warning("No items found. Try expanding the date range or raising cloud cover.")
            S["items"], S["items_df"] = None, None
        else:
            rows = []
            for it in items:
                props = it.properties
                rows.append({
                    "id": it.id,
                    "datetime": props.get("datetime"),
                    "eo:cloud": props.get("eo:cloud_cover"),
                    "platform": props.get("platform"),
                    "constellation": props.get("constellation"),
                })
            df = pd.DataFrame(rows).sort_values("datetime")
            S["items"] = items
            S["items_df"] = df
            S["scene_a"], S["scene_b"], S["results"] = None, None, None
    except Exception as e:
        S["last_error"] = e
        st.error(f"STAC search failed: {e}")

if S["items_df"] is not None:
    st.dataframe(S["items_df"], use_container_width=True, hide_index=True)
else:
    st.info("Search results will appear here.")

# ===================== 3) Choose Two Scenes =================================
st.subheader("3) Choose Two Scenes")
if S["items_df"] is not None and len(S["items_df"]) >= 2:
    ids = list(S["items_df"]["id"])
    cols = st.columns(2)
    with cols[0]:
        S["scene_a"] = st.selectbox("Scene A (older)", ids, index=0, key="scene_a")
    with cols[1]:
        default_b_idx = 1 if len(ids) > 1 else 0
        S["scene_b"] = st.selectbox("Scene B (newer)", ids, index=default_b_idx, key="scene_b")

    if S["scene_a"] == S["scene_b"]:
        st.warning("Pick two *different* scenes.")
else:
    st.info("Search first, then pick two distinct scenes to compare.")

have_two = (
    S["items"] is not None and
    S["scene_a"] is not None and
    S["scene_b"] is not None and
    S["scene_a"] != S["scene_b"]
)

# ===================== 4) Compute & Preview =================================
st.subheader("4) Compute & Preview")
with st.form("compute_form"):
    compute_btn = st.form_submit_button(
        "⚙️ Compute & Preview",
        use_container_width=True,
        disabled=not (aoi_ready and have_two),
    )

if compute_btn and aoi_ready and have_two:
    try:
        # Unique run dir (avoids stale file clashes)
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        base_dir = OUTDIR_BASES[0] if os.path.isdir(OUTDIR_BASES[0]) else OUTDIR_BASES[-1]
        OUTDIR = os.path.join(base_dir, run_id)
        os.makedirs(OUTDIR, exist_ok=True)

        it_map = {it.id: it for it in S["items"]}
        sel_items = [it_map[S["scene_a"]], it_map[S["scene_b"]]]

        with st.spinner("Stacking imagery…"):
            da = stack_sentinel(
                sel_items,
                resolution=st.session_state.resolution,
                chunksize=(int(st.session_state.chunk_size) if st.session_state.use_dask else "auto"),
            )
            da = da.rio.write_crs(da.rio.crs or da.rio.estimate_utm_crs())
            da = da.rio.clip(
                S["aoi_gdf"].to_crs(da.rio.crs).geometry,
                S["aoi_gdf"].to_crs(da.rio.crs).crs,
                drop=True,
            )
        st.toast("Stack ready ✅", icon="✅")

        ds_a = ds_for_time_index(da, 0)
        ds_b = ds_for_time_index(da, 1)

        idx_a = compute_indices(ds_a)
        idx_b = compute_indices(ds_b)
        valid_a = cloud_mask_from_scl(ds_a["scl"]) if "scl" in ds_a else None
        valid_b = cloud_mask_from_scl(ds_b["scl"]) if "scl" in ds_b else None

        a_rgb_png = os.path.join(OUTDIR, "A_truecolor.png")
        b_rgb_png = os.path.join(OUTDIR, "B_truecolor.png")
        save_true_color_png(ds_a, a_rgb_png, valid_a)
        save_true_color_png(ds_b, b_rgb_png, valid_b)

        delta_png = None
        if "NDVI" in idx_a and "NDVI" in idx_b:
            delta = change_map(idx_a["NDVI"], idx_b["NDVI"])
            delta_png = os.path.join(OUTDIR, "delta_ndvi.png")
            save_index_png(delta, delta_png, vmin=-0.5, vmax=0.5, cmap="RdBu")

        def _export_indices(idx_dict, prefix):
            paths = {}
            for name, da_idx in idx_dict.items():
                tif = os.path.join(OUTDIR, f"{prefix}_{name}.tif")
                da_idx.rio.to_raster(tif, compress="deflate")
                png = os.path.join(OUTDIR, f"{prefix}_{name}.png")
                save_index_png(da_idx, png)
                paths[f"{name}_tif"] = tif
                paths[f"{name}_png"] = png
            return paths

        paths_a = _export_indices(idx_a, "A")
        paths_b = _export_indices(idx_b, "B")

        # Build ZIP on disk
        zip_path = os.path.join(OUTDIR, "exports.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in [a_rgb_png, b_rgb_png] + list(paths_a.values()) + list(paths_b.values()):
                if os.path.exists(p):
                    zf.write(p, arcname=os.path.basename(p))
            if delta_png and os.path.exists(delta_png):
                zf.write(delta_png, arcname=os.path.basename(delta_png))

        # Persist results & show a tiny toast
        S["results"] = {
            "a_rgb_png": a_rgb_png,
            "b_rgb_png": b_rgb_png,
            "delta_png": delta_png,
            "zip_path": zip_path,
            "scene_a": S["scene_a"],
            "scene_b": S["scene_b"],
            "resolution": st.session_state.resolution,
            "date_range": (st.session_state.start_date.isoformat(), st.session_state.end_date.isoformat()),
            "indices_a_pngs": {k: v for k, v in paths_a.items() if k.endswith("_png")},
            "indices_b_pngs": {k: v for k, v in paths_b.items() if k.endswith("_png")},
        }
        st.toast(
            f"Ready: {S['scene_a']} → {S['scene_b']} | res {st.session_state.resolution} m | saved to {os.path.dirname(zip_path)}",
            icon="✅"
        )
        st.rerun()

    except Exception as e:
        S["last_error"] = e
        st.error(f"Compute failed: {e}")

# ===================== Batch Export ==========================================
st.subheader("Batch Export")
batch_disabled = not (S["items"] and S["aoi_gdf"] is not None)
colb1, colb2 = st.columns([1, 3])
with colb1:
    run_batch = st.button("📦 Run Batch Export", disabled=batch_disabled, use_container_width=True)
with colb2:
    st.caption("Exports **all** scenes in results table to clipped GeoTIFFs + quicklook PNGs.")

if run_batch and not batch_disabled:
    try:
        base_dir = OUTDIR_BASES[0] if os.path.isdir(OUTDIR_BASES[0]) else OUTDIR_BASES[-1]
        batch_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        batch_dir = os.path.join(base_dir, f"batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)

        with st.spinner("Batch exporting…"):
            for i, it in enumerate(S["items"], 1):
                da = stack_sentinel(
                    [it],
                    resolution=st.session_state.resolution,
                    chunksize=(int(st.session_state.chunk_size) if st.session_state.use_dask else "auto"),
                )
                da = da.rio.write_crs(da.rio.crs or da.rio.estimate_utm_crs())
                da = da.rio.clip(S["aoi_gdf"].to_crs(da.rio.crs).geometry, S["aoi_gdf"].to_crs(da.rio.crs).crs, drop=True)
                ds = ds_for_time_index(da, 0)
                idx = compute_indices(ds)
                valid = cloud_mask_from_scl(ds["scl"]) if "scl" in ds else None
                date_str = (it.properties.get("datetime", "") or "scene").split("T")[0] or f"scene_{i}"
                rgb_png = os.path.join(batch_dir, f"{date_str}_truecolor.png")
                save_true_color_png(ds, rgb_png, valid)
                for name, da_idx in idx.items():
                    tif = os.path.join(batch_dir, f"{date_str}_{name}.tif")
                    da_idx.rio.to_raster(tif, compress="deflate")
                    png = os.path.join(batch_dir, f"{date_str}_{name}.png")
                    save_index_png(da_idx, png)

            batch_zip = os.path.join(batch_dir, "batch_exports.zip")
            with zipfile.ZipFile(batch_zip, "w") as zf:
                for root, _, files in os.walk(batch_dir):
                    for fn in files:
                        if fn.endswith((".tif", ".png")):
                            fp = os.path.join(root, fn)
                            zf.write(fp, arcname=fn)

        zb = _read_bytes(batch_zip)
        if zb:
            size_mb = os.path.getsize(batch_zip) / (1024 * 1024)
            st.caption(f"Batch ZIP size: {size_mb:.2f} MB  •  {batch_zip}")
            st.download_button("⬇️ Download Batch (ZIP)", data=zb, file_name="batch_exports.zip",
                               mime="application/zip", use_container_width=True, key="dl_batch")
        else:
            st.error("Could not read the batch ZIP from disk. Please re-run Batch Export.")
        st.success("Batch export complete.")
    except Exception as e:
        S["last_error"] = e
        st.error(f"Batch export failed: {e}")

# ===================== Error panel (if any) ==================================
if S["last_error"]:
    with st.expander("Debug: Last error", expanded=False):
        st.exception(S["last_error"])
