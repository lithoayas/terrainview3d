# streamlit_app.py
# Cloud-safe Namibia terrain picker (AAIGrid). No GDAL/Shapely/pyproj.
# Adds: Bigger map + Clear shapes button (resets folium draw state).

import io
import math
import os
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

st.set_page_config(page_title="Namibia 3D Terrain Picker", layout="wide")

# ---------------- Session state (for map reset) ----------------
if "map_key" not in st.session_state:
    st.session_state.map_key = 0  # increment to force a fresh Folium widget

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
demtype = st.sidebar.selectbox(
    "DEM (OpenTopography Global DEM API)",
    ["SRTMGL1_E", "SRTMGL3", "NASADEM", "AW3D30"],
    index=0,
)
max_side_px = st.sidebar.slider("Max grid size (downsample)", 100, 1000, 600)
export_ascii = st.sidebar.checkbox("Save clipped DEM as ASCII Grid", value=True)
auto_clear = st.sidebar.checkbox("Auto-clear shapes after processing", value=True)

# API key
API_KEY = None
try:
    API_KEY = st.secrets.get("OPENTOPO_API_KEY")
except Exception:
    API_KEY = None
if not API_KEY:
    API_KEY = os.environ.get("OPENTOPO_API_KEY")

# ---------------- Constants ----------------
NAMIBIA_BBOX = {"south": -28.97, "west": 11.73, "north": -16.95, "east": 25.26}
DEFAULT_CENTER = [-22.56, 17.08]  # lat, lon

# ---------------- Header ----------------
st.title("🗺️ Draw a polygon → 3D terrain map (Namibia)")
st.write(
    "Draw one or more **polygons/rectangles** inside Namibia. We’ll fetch DEM from "
    "OpenTopography (ASCII Grid), clip to the **union** of your shapes, and render a 3D surface."
)

# ---------------- Map controls row ----------------
colA, colB, colC = st.columns([1, 1, 6])
with colA:
    if st.button("🧹 Clear shapes", help="Reset the drawing layer and start fresh"):
        st.session_state.map_key += 1
        st.rerun()
with colB:
    zoom_click = st.button("🔎 Zoom to Namibia", help="Refit map to Namibia extent")

# ---------------- Build Folium map ----------------
m = folium.Map(location=DEFAULT_CENTER, zoom_start=6, tiles="CartoDB positron")
folium.Rectangle(
    bounds=[[NAMIBIA_BBOX["south"], NAMIBIA_BBOX["west"]],
            [NAMIBIA_BBOX["north"], NAMIBIA_BBOX["east"]]],
    color="#1f77b4", weight=2, dash_array="6,6", fill=False,
    tooltip="Namibia extent (approx)",
).add_to(m)
if zoom_click:
    m.fit_bounds([[NAMIBIA_BBOX["south"], NAMIBIA_BBOX["west"]],
                  [NAMIBIA_BBOX["north"], NAMIBIA_BBOX["east"]]])

Draw(
    draw_options={
        "polyline": False, "rectangle": True, "polygon": True,
        "circle": False, "circlemarker": False, "marker": False,
    },
    edit_options={"edit": True, "remove": True}
).add_to(m)

st.write("**Step 1:** Finish each shape (click the first point or press **Finish**), then click **Process**.")

# Make the map **bigger**: height=720, width=1200, and unique key
map_data = st_folium(
    m,
    height=720,
    width=1200,  # increases visible canvas; Streamlit will cap at container width
    returned_objects=["last_active_drawing", "all_drawings"],
    key=f"map_{st.session_state.map_key}",
)

process = st.button("🚀 Process AOI → Fetch DEM → 3D Render")

# ---------------- Helpers ----------------
def normalize_lon(lon: float) -> float:
    lon = ((lon + 180.0) % 360.0) - 180.0
    return -180.0 if abs(lon + 180.0) < 1e-9 else lon

def ensure_lnglat_coords(geometry: Dict) -> List[List[List[float]]]:
    """Return list of rings (outer), each [[lon,lat]...]. Fix [lat,lon] if needed."""
    def fix_ring(ring):
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        swap = any(abs(x) > 90 for x in xs) or (min(ys) > 10 or max(ys) < -35)
        return [[p[1], p[0]] for p in ring] if swap else ring

    rings = []
    t = geometry.get("type")
    if t == "Polygon":
        rings.append(fix_ring(geometry["coordinates"][0]))
    elif t == "MultiPolygon":
        for poly in geometry["coordinates"]:
            rings.append(fix_ring(poly[0]))
    return rings

def extract_all_rings(map_obj: Dict) -> List[List[List[float]]]:
    rings: List[List[List[float]]] = []
    items = []
    if map_obj and map_obj.get("last_active_drawing"):
        items.append(map_obj["last_active_drawing"])
    if map_obj and map_obj.get("all_drawings"):
        items.extend(map_obj["all_drawings"])
    for feat in items:
        g = feat.get("geometry")
        if g and g.get("type") in ("Polygon", "MultiPolygon"):
            rings.extend(ensure_lnglat_coords(g))
    return rings

def bounds_of_ring(r):  # -> (south, west, north, east)
    xs = [p[0] for p in r]; ys = [p[1] for p in r]
    return min(ys), min(xs), max(ys), max(xs)

def bounds_of_rings(rings):
    s_list, w_list, n_list, e_list = [], [], [], []
    for r in rings:
        s, w, n, e = bounds_of_ring(r)
        s_list.append(s); w_list.append(w); n_list.append(n); e_list.append(e)
    return min(s_list), min(w_list), max(n_list), max(e_list)

def bbox_intersection(a: Dict, b: Dict) -> Optional[Dict]:
    west = max(a["west"], b["west"])
    east = min(a["east"], b["east"])
    south = max(a["south"], b["south"])
    north = min(a["north"], b["north"])
    if west < east and south < north:
        return {"west": west, "east": east, "south": south, "north": north}
    return None

def points_in_polygon(xs: np.ndarray, ys: np.ndarray, ring) -> np.ndarray:
    X = xs.ravel(); Y = ys.ravel()
    inside = np.zeros_like(X, dtype=bool)
    poly = np.asarray(ring, dtype=float)
    px, py = poly[:, 0], poly[:, 1]
    n = len(poly)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        cond = ((yi > Y) != (yj > Y))
        x_int = (xj - xi) * (Y - yi) / (yj - yi + 1e-16) + xi
        cond &= (X < x_int)
        inside ^= cond
    return inside.reshape(xs.shape)

def parse_aaigrid(text_bytes: bytes):
    s = text_bytes.decode("utf-8", errors="ignore").strip().splitlines()
    header, data_start = {}, 0
    for i, line in enumerate(s[:12]):
        parts = line.split()
        if len(parts) >= 2 and parts[0].lower() in {
            "ncols","nrows","xllcorner","yllcorner","xllcenter","yllcenter","cellsize","nodata_value"
        }:
            key = parts[0].lower()
            val = float(parts[1]) if key not in ("ncols","nrows") else int(parts[1])
            header[key] = val
            data_start = i + 1
        else:
            data_start = i
            break
    ncols = int(header["ncols"]); nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])
    nodata = float(header.get("nodata_value", -9999.0))
    west = float(header.get("xllcorner", header.get("xllcenter"))) - (0 if "xllcorner" in header else 0.5*cellsize)
    south = float(header.get("yllcorner", header.get("yllcenter"))) - (0 if "yllcorner" in header else 0.5*cellsize)
    data_str = s[data_start : data_start + nrows]
    arr = np.loadtxt(io.StringIO("\n".join(data_str)), dtype=float)
    arr = np.where(arr == nodata, np.nan, arr)
    north = south + nrows * cellsize
    east  = west  + ncols * cellsize
    return arr, ncols, nrows, west, south, east, north, cellsize

def make_lonlat_grids(west, south, east, north, ncols, nrows):
    cellsize_x = (east - west) / ncols
    cellsize_y = (north - south) / nrows
    lon_centers = np.linspace(west + 0.5 * cellsize_x, east - 0.5 * cellsize_x, ncols)
    lat_centers = np.linspace(north - 0.5 * cellsize_y, south + 0.5 * cellsize_y, nrows)
    return np.meshgrid(lon_centers, lat_centers)

def downsample(arr, lon, lat, max_side=600):
    h, w = arr.shape
    scale = max(h, w) / float(max_side)
    if scale <= 1.0:
        return arr, lon, lat
    step_h = int(math.ceil(h / max(2, int(round(h / scale)))))
    step_w = int(math.ceil(w / max(2, int(round(w / scale)))))
    return arr[::step_h, ::step_w], lon[::step_h, ::step_w], lat[::step_h, ::step_w]

# ---------------- Main ----------------
if process:
    if not API_KEY:
        st.error("Missing OpenTopography API key. Set OPENTOPO_API_KEY in secrets or env.")
        st.stop()

    rings = extract_all_rings(map_data)
    if not rings:
        st.error("No finished polygon/rectangle found. Close the shape (or click **Finish**) and try again.")
        st.stop()

    s, w, n, e = bounds_of_rings(rings)
    aoi_bbox = {"south": s, "west": w, "north": n, "east": e}
    inter = bbox_intersection(aoi_bbox, NAMIBIA_BBOX)
    if inter is None:
        st.error("Your shapes are outside Namibia. Please draw inside the dashed rectangle.")
        st.stop()

    west = normalize_lon(inter["west"]); east = normalize_lon(inter["east"])
    south = max(-90.0, min(90.0, inter["south"])); north = max(-90.0, min(90.0, inter["north"]))
    if north < south: south, north = north, south
    if east < west:   west, east   = east, west
    MIN_DEG = 0.005
    if (north - south) < MIN_DEG:
        c = 0.5*(north + south); south, north = c - MIN_DEG/2, c + MIN_DEG/2
    if (east - west) < MIN_DEG:
        c = 0.5*(east + west); west, east = c - MIN_DEG/2, c + MIN_DEG/2

    st.info(f"Download bbox: south={south:.5f}, west={west:.5f}, north={north:.5f}, east={east:.5f}")

    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype={demtype}&south={south}&north={north}&west={west}&east={east}"
        f"&outputFormat=AAIGrid&API_Key={API_KEY}"
    )

    with st.status("Downloading DEM (ASCII Grid)…", expanded=False):
        r = requests.get(url, timeout=120)
        try:
            r.raise_for_status()
        except Exception:
            st.error("Failed to fetch DEM from OpenTopography.")
            st.code(r.content[:400].decode("utf-8", errors="ignore") or str(r.status_code))
            st.stop()
        try:
            dem, ncols, nrows, g_west, g_south, g_east, g_north, cellsize = parse_aaigrid(r.content)
        except Exception as e:
            st.error(f"Failed to parse AAIGrid: {e}")
            st.code(r.content[:400].decode("utf-8", errors="ignore"))
            st.stop()

    lon_grid, lat_grid = make_lonlat_grids(g_west, g_south, g_east, g_north, ncols, nrows)
    union_mask = np.zeros_like(dem, dtype=bool)
    for ring in rings:
        union_mask |= points_in_polygon(lon_grid, lat_grid, ring)
    dem_masked = np.where(union_mask, dem, np.nan)

    dem_ds, lon_ds, lat_ds = downsample(dem_masked, lon_grid, lat_grid, max_side=max_side_px)
    if np.all(np.isnan(dem_ds)):
        st.error("DEM fetched, but union mask excluded everything. Try a larger AOI or different DEM.")
        st.stop()

    # Meters (local tangent)
    deg2rad = np.pi / 180.0
    ref_lat = ((south + north) / 2.0) * deg2rad
    m_per_deg_lon = 111320.0 * np.cos(ref_lat)
    m_per_deg_lat = 110540.0
    x0, y0 = float(np.nanmin(lon_ds)), float(np.nanmin(lat_ds))
    X = (lon_ds - x0) * m_per_deg_lon
    Y = (lat_ds - y0) * m_per_deg_lat

    st.subheader("🌄 3D Terrain")
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=dem_ds, showscale=True)])
    fig.update_scenes(xaxis_title_text="X (m)", yaxis_title_text="Y (m)", zaxis_title_text="Elevation (m)")
    fig.update_layout(height=750, scene_aspectmode="data",
                      margin=dict(l=0, r=0, b=0, t=30),
                      title=f"{demtype} — 3D surface (union of shapes)")
    st.plotly_chart(fig, use_container_width=True)

    if export_ascii:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".asc") as tmp:
                out_path = tmp.name
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"ncols         {dem_ds.shape[1]}\n")
                f.write(f"nrows         {dem_ds.shape[0]}\n")
                f.write(f"xllcorner     {float(np.nanmin(lon_ds))}\n")
                f.write(f"yllcorner     {float(np.nanmin(lat_ds))}\n")
                if dem_ds.shape[1] > 1:
                    csx = (np.nanmax(lon_ds) - np.nanmin(lon_ds)) / dem_ds.shape[1]
                else:
                    csx = (g_east - g_west) / max(1, ncols)
                f.write(f"cellsize      {csx}\n")
                f.write("NODATA_value  -9999\n")
                out = np.where(np.isnan(dem_ds), -9999, dem_ds).astype(float)
                for row in out:
                    f.write(" ".join(f"{v:.3f}" for v in row) + "\n")
            st.success(f"Saved clipped ASCII Grid: {out_path}")
            with open(out_path, "rb") as fh:
                st.download_button("⬇️ Download DEM (ASCII Grid)", data=fh.read(), file_name="clipped_dem.asc")
        except Exception as e:
            st.warning(f"Could not save ASCII Grid: {e}")

    # Auto-clear shapes so old polygons don’t reappear
    if auto_clear:
        st.session_state.map_key += 1
        st.toast("Shapes cleared.")
        st.rerun()

st.caption("Tip: If the map looks cramped, widen your browser or hide the sidebar. "
           "Use **🧹 Clear shapes** to truly reset the drawing layer.")
