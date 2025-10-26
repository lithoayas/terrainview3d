# streamlit_app.py
# -------------------------------------------------------------
# Streamlit app: Draw a polygon over Namibia -> fetch DEM -> 3D terrain
# -------------------------------------------------------------
# Requirements:
#   pip install streamlit folium streamlit-folium shapely requests rasterio numpy plotly geopandas pyproj
#
# Run:
#   export OPENTOPO_API_KEY="your-opentopo-key"
#   streamlit run streamlit_app.py

import math
import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from shapely.geometry import shape, Polygon, mapping
from pyproj import Transformer

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

# -------------------------------------------------------------
# Page config
# -------------------------------------------------------------
st.set_page_config(page_title="Namibia 3D Terrain Picker", layout="wide")
st.title("🗺️ Draw a polygon → 3D terrain map")
st.write(
    "Draw a polygon or rectangle *inside Namibia*. We'll fetch a DEM from OpenTopography, "
    "clip it to your AOI, and render a 3D surface."
)

with st.sidebar:
    st.header("Settings")
    demtype = st.selectbox(
        "DEM Source (OpenTopography Global DEM API)",
        ["SRTMGL1_E", "SRTMGL3", "NASADEM", "AW3D30"],
        index=0,
        help="SRTMGL1_E generally works between ~60°N and ~56°S.",
    )
    max_side_px = st.slider(
        "Max grid size (downsample)", 100, 600, 300,
        help="Lower this if large AOIs render slowly."
    )
    export_geotiff = st.checkbox("Save clipped DEM as GeoTIFF", value=True)

# API key from Streamlit secrets or environment
API_KEY = None
try:
    API_KEY = st.secrets.get("OPENTOPO_API_KEY")
except Exception:
    pass
if not API_KEY:
    API_KEY = os.environ.get("OPENTOPO_API_KEY")

# -------------------------------------------------------------
# Namibia focus
# -------------------------------------------------------------
DEFAULT_CENTER = [-22.56, 17.08]  # Windhoek approx
DEFAULT_ZOOM = 6

# Namibia bounding box (approx)
NAMIBIA_BBOX = {
    "south": -28.97,
    "west": 11.73,
    "north": -16.95,
    "east": 25.26,
}

m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles="CartoDB positron")

# Show Namibia extent and fit to it
folium.Rectangle(
    bounds=[
        [NAMIBIA_BBOX["south"], NAMIBIA_BBOX["west"]],
        [NAMIBIA_BBOX["north"], NAMIBIA_BBOX["east"]],
    ],
    color="#1f77b4",
    fill=False,
    weight=2,
    dash_array="5,5",
    tooltip="Namibia extent (approx)"
).add_to(m)

m.fit_bounds([
    [NAMIBIA_BBOX["south"], NAMIBIA_BBOX["west"]],
    [NAMIBIA_BBOX["north"], NAMIBIA_BBOX["east"]],
])

Draw(
    draw_options={
        "polyline": False,
        "rectangle": True,
        "polygon": True,
        "circle": False,
        "circlemarker": False,
        "marker": False,
    },
    edit_options={"edit": True, "remove": True}
).add_to(m)

st.write("**Step 1:** Draw a polygon or rectangle inside the dashed Namibia box, then click **Process**.")
map_data = st_folium(m, height=480, width=None, returned_objects=["last_active_drawing", "all_drawings"])
process = st.button("🚀 Process AOI → Fetch DEM → 3D Render")

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def normalize_lon(lon: float) -> float:
    """Wrap any longitude to [-180, 180]."""
    lon = ((lon + 180.0) % 360.0) - 180.0
    return -180.0 if abs(lon + 180.0) < 1e-9 else lon

def poly_bbox(poly: Polygon):
    minx, miny, maxx, maxy = poly.bounds
    # Return as south, west, north, east
    return miny, minx, maxy, maxx

def lonlat_to_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def _extract_polygon_from_map(map_data_obj):
    # Prefer the most recent drawing
    if map_data_obj and map_data_obj.get("last_active_drawing"):
        feat = map_data_obj["last_active_drawing"]
        if feat and feat.get("geometry"):
            g = feat["geometry"]
            if g.get("type") in ("Polygon", "MultiPolygon"):
                return g
    # Fallback: scan all drawings
    if map_data_obj and map_data_obj.get("all_drawings"):
        for feat in map_data_obj["all_drawings"]:
            g = feat.get("geometry")
            if g and g.get("type") in ("Polygon", "MultiPolygon"):
                return g
    return None

def normalize_leaflet_coords(geometry):
    """
    Detect whether incoming coords are [lat, lng] or [lng, lat], and normalize to [lng, lat].
    Uses simple heuristics + Namibia bbox to decide if a swap is needed.
    """
    def fix_ring(ring):
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        # Heuristic: if x looks like a latitude or y does not look like Namibia lat range, swap.
        needs_swap = any(abs(x) > 90 for x in xs) or (min(ys) > 10 or max(ys) < -35)
        return [[pt[1], pt[0]] for pt in ring] if needs_swap else ring

    gtype = geometry.get("type")
    if gtype == "Polygon":
        coords = [fix_ring(r) for r in geometry["coordinates"]]
        return {"type": "Polygon", "coordinates": coords}
    elif gtype == "MultiPolygon":
        polys = []
        for poly in geometry["coordinates"]:
            rings = [fix_ring(r) for r in poly]
            polys.append(rings)
        return {"type": "MultiPolygon", "coordinates": polys}
    return geometry

def clip_and_downsample(src_dataset, clip_geom, max_side=300):
    # Clip raster to polygon
    out_image, out_transform = mask(src_dataset, [mapping(clip_geom)], crop=True)
    out_image = out_image[0]  # single band expected
    # Replace nodata with nan
    nodata = src_dataset.nodata
    if nodata is not None:
        out_image = np.where(out_image == nodata, np.nan, out_image)
    h, w = out_image.shape
    if h == 0 or w == 0:
        raise ValueError("Empty output after clipping.")
    # Downsample to max_side while preserving aspect (fast stride-based)
    scale = max(h, w) / float(max_side)
    if scale > 1.0:
        new_h = max(2, int(round(h / scale)))
        new_w = max(2, int(round(w / scale)))
        out_image_ds = out_image[::int(math.ceil(h / new_h)), ::int(math.ceil(w / new_w))]
    else:
        out_image_ds = out_image
    return out_image_ds

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if process:
    if not API_KEY:
        st.error("Missing OpenTopography API key. Set OPENTOPO_API_KEY in Streamlit secrets or environment.")
        st.stop()

    # Extract polygon
    raw_geom = _extract_polygon_from_map(map_data)
    if raw_geom is None:
        st.error("Please draw a Polygon or Rectangle inside Namibia, then click Process.")
        st.stop()

    # Convert to [lng, lat] reliably
    geom = shape(normalize_leaflet_coords(raw_geom))
    if geom.geom_type == "MultiPolygon":
        geom = max(list(geom.geoms), key=lambda g: g.area)

    # Intersect with Namibia bbox (restrict to Namibia)
    namibia_poly = Polygon([
        (NAMIBIA_BBOX["west"],  NAMIBIA_BBOX["south"]),
        (NAMIBIA_BBOX["east"],  NAMIBIA_BBOX["south"]),
        (NAMIBIA_BBOX["east"],  NAMIBIA_BBOX["north"]),
        (NAMIBIA_BBOX["west"],  NAMIBIA_BBOX["north"]),
    ])
    intersection_geom = geom.intersection(namibia_poly)
    if intersection_geom.is_empty:
        st.error("Your AOI appears outside Namibia. Please draw inside the dashed box.")
        st.write("AOI bounds (lon/lat):", geom.bounds)
        st.write("Namibia bbox:", NAMIBIA_BBOX)
        st.stop()

    geom = intersection_geom
    if geom.geom_type == "MultiPolygon":
        geom = max(list(geom.geoms), key=lambda g: g.area)

    # Compute bbox (south, west, north, east)
    south, west, north, east = poly_bbox(geom)

    # Normalize and guard against degenerate boxes
    west = normalize_lon(west)
    east = normalize_lon(east)
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))
    if north < south:
        south, north = north, south
    if east < west:
        west, east = east, west

    MIN_DEG = 0.005  # ~500 m at equator
    if (north - south) < MIN_DEG:
        c = 0.5 * (north + south)
        south = max(-90.0, c - MIN_DEG / 2.0)
        north = min(90.0,  c + MIN_DEG / 2.0)
    if (east - west) < MIN_DEG:
        c = 0.5 * (east + west)
        west = normalize_lon(c - MIN_DEG / 2.0)
        east = normalize_lon(c + MIN_DEG / 2.0)
    if west > east:
        west, east = east, west

    st.info(f"AOI bounds: south={south:.5f}, west={west:.5f}, north={north:.5f}, east={east:.5f}")

    # Build API URL
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype={demtype}&south={south}&north={north}&west={west}&east={east}"
        f"&outputFormat=GTiff&API_Key={API_KEY}"
    )

    # Download and validate DEM
    with st.status("Downloading DEM from OpenTopography…", expanded=False):
        r = requests.get(url, timeout=120)
        r.raise_for_status()

        dem_bytes = r.content
        if len(dem_bytes) < 16:
            st.error("DEM download returned empty/short content.")
            st.stop()

        # Validate response is a GeoTIFF by magic bytes or content-type
        magic = dem_bytes[:4]
        ctype = (r.headers.get("Content-Type", "") or "").lower()
        is_tiff_magic = magic in (b"II*\x00", b"MM\x00*")
        looks_like_tiff = ctype.startswith("image/tiff") or ctype.startswith("application/octet-stream")
        if not (is_tiff_magic or looks_like_tiff):
            try:
                preview = dem_bytes[:400].decode("utf-8", errors="ignore")
            except Exception:
                preview = repr(dem_bytes[:100])
            st.error("OpenTopography did not return a GeoTIFF. Check API key, AOI size, and DEM availability.")
            st.code(preview)
            st.stop()

    # Open in-memory GeoTIFF, sanity check overlap, clip, and render
    with MemoryFile(dem_bytes) as memfile:
        with memfile.open(driver="GTiff") as src:
            # Check overlap for clearer error
            src_bounds = src.bounds  # (left, bottom, right, top) lon/lat
            raster_poly = Polygon([
                (src_bounds.left, src_bounds.bottom),
                (src_bounds.right, src_bounds.bottom),
                (src_bounds.right, src_bounds.top),
                (src_bounds.left, src_bounds.top),
            ])
            if not geom.intersects(raster_poly):
                st.error("Your AOI does not overlap the DEM tile. Try a smaller AOI or switch DEM source.")
                st.info(
                    f"DEM bounds lon/lat: left={src_bounds.left:.5f}, right={src_bounds.right:.5f}, "
                    f"bottom={src_bounds.bottom:.5f}, top={src_bounds.top:.5f}"
                )
                st.stop()

            try:
                dem = clip_and_downsample(src, geom, max_side=max_side_px)
            except Exception as e:
                st.error(f"Clipping/downsampling failed: {e}")
                st.stop()

            # Build axes in meters (UTM) for better aspect ratio in 3D
            center_lon = (west + east) / 2.0
            center_lat = (south + north) / 2.0
            utm_epsg = lonlat_to_utm_epsg(center_lon, center_lat)
            to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True).transform

            h, w = dem.shape
            lon_lin = np.linspace(west, east, w)
            lat_lin = np.linspace(north, south, h)  # north -> south so rows match
            lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)
            x_grid, y_grid = to_utm(lon_grid, lat_grid)

            st.subheader("🌄 3D Terrain")
            fig = go.Figure(data=[go.Surface(x=x_grid, y=y_grid, z=dem, showscale=True)])
            fig.update_scenes(
                xaxis_title_text="X (m)",
                yaxis_title_text="Y (m)",
                zaxis_title_text="Elevation (m)"
            )
            fig.update_layout(
                height=700,
                scene_aspectmode="data",
                margin=dict(l=0, r=0, b=0, t=30),
                title=f"{demtype} — 3D surface"
            )
            st.plotly_chart(fig, use_container_width=True)

            if export_geotiff:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                        out_path = tmp.name
                    profile = src.profile.copy()
                    profile.update({
                        "driver": "GTiff",
                        "height": dem.shape[0],
                        "width": dem.shape[1],
                        "count": 1,
                        "dtype": "float32",
                        "crs": "EPSG:4326",
                        "transform": rasterio.transform.from_bounds(
                            west, south, east, north, dem.shape[1], dem.shape[0]
                        ),
                    })
                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(dem.astype("float32"), 1)
                    st.success(f"Saved clipped DEM: {out_path}")
                    with open(out_path, "rb") as fh:
                        st.download_button("⬇️ Download DEM GeoTIFF", data=fh.read(), file_name="clipped_dem.tif")
                except Exception as e:
                    st.warning(f"Could not save GeoTIFF: {e}")

st.caption("Tip: For a web-native globe, consider CesiumJS + Cesium World Terrain. For browser hillshade, MapLibre GL JS + deck.gl TerrainLayer works well.")
