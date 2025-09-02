
(Keep your existing screenshots section; update with your deployed URL.)

---

# 6) Portfolio blurb (copy/paste)

**Sentinel-2 Construction/Change Monitor — Remote Sensing + GIS**

I built a portfolio-ready, public web app that lets anyone draw or upload an Area of Interest, fetch **Sentinel-2 L2A** imagery from a STAC catalog, **cloud-mask** it, compute classic indices (**NDVI, NDWI, NDBI**), and run simple **change detection** between two dates. The app previews a **before/after swipe** and a **ΔNDVI heatmap**, and exports **clipped GeoTIFFs + quick-look PNGs** as a one-click ZIP.

- **Tech**: Python 3.11, Streamlit, STAC (**pystac-client**, **stackstac**), **xarray/rioxarray**, geopandas, rasterio, folium, matplotlib  
- **Features**: AOI draw/upload, date & cloud filters, index computation, change map, batch export  
- **Infra**: Streamlit Cloud deploy, pinned wheels; GitHub Actions **mini pipeline** sanity-checks STAC access & NDVI compute on every push  
- **Outcome**: <one sentence about a real AOI you tested—e.g., detected new construction/land-use changes between Month X and Y>

> Live demo: _<your Streamlit link>_ • Source: _<GitHub repo link>_

(Attach 2–3 screenshots: AOI map, swipe panel, ΔNDVI panel.)

---

# 7) (Optional) One-click Docker deploy (Render/Railway)

If you prefer container hosting:

### `Dockerfile`
```dockerfile
FROM python:3.11-slim

# System deps for rasterio/pyproj wheels (light)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal32 proj-bin && rm -rf /var/lib/apt/lists/* || true

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
