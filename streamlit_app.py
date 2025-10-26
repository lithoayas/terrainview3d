# streamlit_app.py
# -------------------------------------------------------------
# Namibia terrain picker (cloud-safe): draw AOI -> fetch DEM (AAIGrid) -> 3D surface
# No GDAL/Rasterio, No Shapely, No pyproj. Works on Streamlit Cloud Python 3.13.
# -------------------------------------------------------------
# Local run:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   export OPENTOPO_API_KEY="your-opentopo-key"
#   streamlit run streamlit_app.py

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

# -------------------------------------------------------------
# Page + sidebar
# -------------------------------------------------------------
st.set_page_config(page_title="Namibia 3D Terrain Picker (Cloud-Safe)", layout="wide")
st.title("🗺️ Draw a polygon → 3D terrain map (Namibia)")

st.write(
    "Draw a **polygon or rectangle** inside Namibia. We’ll fetch DEM from OpenTopography "
    "(ASCII Grid), clip to your AOI, and render a 3D surface — no native deps."
)

with st.sidebar:
    st.header("Settings")
    demtype = st.selectbox(
        "DEM Source (OpenTopography Global DEM API)",
        ["SRTMGL1_E", "SRTMGL3", "NASADEM", "AW3D30"],
        index=0,
        help="SRTMGL1_E generally works between ~60°N and ~56°S."
    )
    max_side_px = st.slider(
        "Max grid size (downsample)", 100, 600, 300,
        help="Lower if large AOIs render slowly."
    )
    export_ascii = st.checkbox("Save clipped DEM as ASCII Grid", value=True)

# API key from secrets or environment
API_KEY = None
try:
    API_KEY = st.secrets.get("OPENTOPO_API_KEY")
except Exception:
    API_KEY = None
if not API_KEY:
    API_KEY = os.environ.get("OPENTOPO_API_KEY")

# -------------------------------------------------------------
# Namibia focus & map
# -------------------------------------------------------------
DEFAULT_CENTER = [-22.56, 17.08]  # Windhoek approx (lat, lon)
DEFAULT_ZOOM = 6

# Namibia bounding box (lon/lat)
NAMIBIA_BBOX = {"south": -28.97, "west": 11.73, "north": -16.95, "east": 25.26}

m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles="CartoDB positron")
folium.Rectangle(
    bounds=[[NAMIBIA_BBOX["south"], NAMIBIA_BBOX["west"]],
            [NAMIBIA_BBOX["north"], NAMIBIA_BBOX["east"]]],
    color="#1f77b4", fill=False, weight=2, dash_array="5,5",
    tooltip="Namibia extent (approx)",
).add_to(m)
m.fit_bounds([[NAMIBIA_BBOX["south"], NAMIBIA_BBOX["west"]],
              [NAMIBIA_BBOX["north"], NAMIBIA_BBOX["east"]]])

Draw(
    draw_options={"polyline": False, "rectangle": True, "polygon": True,
                  "circle": False, "circlemarker": False, "marker": False},
    edit_options={"edit": True, "remove": True}
).add_to(m)

st.write("**Step 1:** Draw inside the dashed Namibia box, then click **Process**.")
map_data = st_folium(m, height=480, width=None,
                     returned_objects=["last_active_drawing", "all_drawings"])
process = st.button("🚀 Process AOI → Fetch DEM → 3D Render")

# -------------------------------------------------------------
# Helpers — pure NumPy utils
# -------------------------------------------------------------
def normalize_lon(lon: float) -> float:
    """Wrap lon to [-180, 180]."""
    lon = ((lon + 180.0) % 360.0) - 180.0
    return -180.0 if abs(lon + 180.0) < 1e-9 else lon

def extract_polygon_geojson(map_obj: Dict) -> Optional[Dict]:
    """Get the last drawn Polygon/MultiPolygon geometry from st_folium output."""
    if map_obj and map_obj.get("last_active_drawing"):
        g = map_obj["last_active_drawing"].get("geometry")
        if g and g.get("type") in ("Polygon", "MultiPolygon"):
            return g
    if map_obj and map_obj.get("all_drawings"):
        for feat in map_obj["all_drawings"]:
            g = feat.get("geometry")
            if g and g.get("type") in ("Polygon", "MultiPolygon"):
                return g
    return None

def ensure_lnglat_coords(geometry: Dict) -> Dict:
    """
    Normalize coords to [lng, lat]. Some draw tools can output [lat, lng].
    Heuristic: If 'x' looks like latitude or 'y' outside S. Africa, swap.
    """
    def fix_ring(ring):
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        needs_swap = any(abs(x) > 90 for x in xs) or (min(ys) > 10 or max(ys) < -35)
        return [[pt[1], pt[0]] for pt in ring] if needs_swap else ring

    gtype = geometry.get("type")
    if gtype == "Polygon":
        return {"type": "Polygon", "coordinates": [fix_ring(r) for r in geometry["coordinates"]]}
    if gtype == "MultiPolygon":
        return {"type": "MultiPolygon",
                "coordinates": [[fix_ring(r) for r in poly] for poly in geometry["coordinates"]]}
    return geometry

def pick_largest_polygon(geometry: Dict) -> List[List[float]]:
    """
    Return the exterior ring (list of [lon,lat]) of the largest polygon.
    We ignore holes (Draw rarely provides them).
    """
    def polygon_area(coords):
        # Shoelace on lon/lat (area proxy just for choosing largest)
        x = np.asarray([p[0] for p in coords])
        y = np.asarray([p[1] for p in coords])
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    if geometry["type"] == "Polygon":
        ring = geometry["coordinates"][0]
        return ring
    # MultiPolygon: choose largest by area of outer ring
    best_ring, best_area = None, -1.0
    for poly in geometry["coordinates"]:
        ring = poly[0]
        a = polygon_area(ring)
        if a > best_area:
            best_ring, best_area = ring, a
    return best_ring

def polygon_bounds(ring: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(ys), min(xs), max(ys), max(xs)  # south, west, north, east

def bbox_intersection(a: Dict, b: Dict) -> Optional[Dict]:
    west = max(a["west"], b["west"])
    east = min(a["east"], b["east"])
    south = max(a["south"], b["south"])
    north = min(a["north"], b["north"])
    if west < east and south < north:
        return {"west": west, "east": east, "south": south, "north": north}
    return None

def points_in_polygon(xs: np.ndarray, ys: np.ndarray, ring: List[List[float]]) -> np.ndarray:
    """
    Vectorized ray-casting point-in-polygon for a single ring (no holes).
    xs, ys are same-shaped grids. Returns boolean mask of same shape.
    """
    x = xs.ravel()
    y = ys.ravel()
    inside = np.zeros_like(x, dtype=bool)

    poly = np.asarray(ring, dtype=float)
    px = poly[:, 0]
    py = poly[:, 1]
    n = len(poly)
    # iterate edges (i -> j)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        # edge crosses the horizontal line at y?
        cond = ((yi > y) != (yj > y))
        # x intersection of edge with horizontal line at y
        x_int = (xj - xi) * (y - yi) / (yj - yi + 1e-16) + xi
        cond = cond & (x < x_int)
        inside ^= cond

    return inside.reshape(xs.shape)

def parse_aaigrid(text_bytes: bytes):
    """
    Parse ESRI ASCII Grid (AAIGrid) into (arr, ncols, nrows, west, south, east, north, cellsize).
    Supports xllcorner/xllcenter and yllcorner/yllcenter.
    """
    s = text_bytes.decode("utf-8", errors="ignore").strip().splitlines()
    header = {}
    data_start = 0
    for i, line in enumerate(s[:12]):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].lower() in {
            "ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter",
            "cellsize", "nodata_value"
        }:
            key = parts[0].lower()
            val = float(parts[1]) if key not in ("ncols", "nrows") else int(parts[1])
            header[key] = val
            data_start = i + 1
        else:
            data_start = i
            break

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])
    nodata = float(header.get("nodata_value", -9999.0))

    if "xllcorner" in header:
        west = float(header["xllcorner"])
    elif "xllcenter" in header:
        west = float(header["xllcenter"]) - 0.5 * cellsize
    else:
        raise ValueError("AAIGrid missing xllcorner/xllcenter")

    if "yllcorner" in header:
        south = float(header["yllcorner"])
    elif "yllcenter" in header:
        south = float(header["yllcenter"]) - 0.5 * cellsize
    else:
        raise ValueError("AAIGrid missing yllcorner/yllcenter")

    data_str = s[data_start : data_start + nrows]
    arr = np.loadtxt(io.StringIO("\n".join(data_str)), dtype=float)
    if arr.shape != (nrows, ncols):
        raise ValueError(f"AAIGrid shape mismatch: got {arr.shape}, expected {(nrows, ncols)}")

    arr = np.where(arr == nodata, np.nan, arr)

    north = south + nrows * cellsize
    east = west + ncols * cellsize
    return arr, ncols, nrows, west, south, east, north, cellsize

def clip_and_downsample_ascii(arr, west, south, east, north,
                              ring: List[List[float]], max_side=300):
    """
    Mask array to drawn polygon using cell centers, then downsample.
    """
    nrows, ncols = arr.shape

    cellsize_x = (east - west) / ncols
    cellsize_y = (north - south) / nrows

    lon_centers = np.linspace(west + 0.5 * cellsize_x, east - 0.5 * cellsize_x, ncols)
    lat_centers = np.linspace(north - 0.5 * cellsize_y, south + 0.5 * cellsize_y, nrows)
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    inside = points_in_polygon(lon_grid, lat_grid, ring)
    masked = np.where(inside, arr, np.nan)

    h, w = masked.shape
    scale = max(h, w) / float(max_side)
    if scale > 1.0:
        step_h = int(math.ceil(h / max(2, int(round(h / scale)))))
        step_w = int(math.ceil(w / max(2, int(round(w / scale)))))
        masked = masked[::step_h, ::step_w]
        lon_grid = lon_grid[::step_h, ::step_w]
        lat_grid = lat_grid[::step_h, ::step_w]

    return masked, lon_grid, lat_grid

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if process:
    if not API_KEY:
        st.error("Missing OpenTopography API key. Set OPENTOPO_API_KEY in Streamlit secrets or env.")
        st.stop()

    geom = extract_polygon_geojson(map_data)
    if geom is None:
        st.error("Please draw a Polygon or Rectangle inside Namibia, then click Process.")
        st.stop()

    geom = ensure_lnglat_coords(geom)
    ring = pick_largest_polygon(geom)

    # Compute user AOI bbox
    aoi_s, aoi_w, aoi_n, aoi_e = polygon_bounds(ring)
    aoi_bbox = {"south": aoi_s, "west": aoi_w, "north": aoi_n, "east": aoi_e}

    # Intersect with Namibia bbox for the API call (so we only fetch Namibia data)
    inter = bbox_intersection(aoi_bbox, NAMIBIA_BBOX)
    if inter is None:
        st.error("Your AOI appears **outside Namibia**. Please draw inside the dashed box.")
        st.write("AOI bounds (lon/lat):", aoi_bbox)
        st.write("Namibia bbox:", NAMIBIA_BBOX)
        st.stop()

    # Normalize and ensure non-zero area
    west = normalize_lon(inter["west"])
    east = normalize_lon(inter["east"])
    south = max(-90.0, min(90.0, inter["south"]))
    north = max(-90.0, min(90.0, inter["north"]))
    if north < south:
        south, north = north, south
    if east < west:
        west, east = east, west
    MIN_DEG = 0.005
    if (north - south) < MIN_DEG:
        c = 0.5 * (north + south)
        south, north = c - MIN_DEG / 2, c + MIN_DEG / 2
    if (east - west) < MIN_DEG:
        c = 0.5 * (east + west)
        west, east = c - MIN_DEG / 2, c + MIN_DEG / 2

    st.info(f"AOI bounds used: south={south:.5f}, west={west:.5f}, north={north:.5f}, east={east:.5f}")

    # Build API URL (ASCII Grid)
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype={demtype}&south={south}&north={north}&west={west}&east={east}"
        f"&outputFormat=AAIGrid&API_Key={API_KEY}"
    )

    with st.status("Downloading DEM (ASCII Grid) from OpenTopography…", expanded=False):
        r = requests.get(url, timeout=120)
        try:
            r.raise_for_status()
        except Exception:
            preview = r.content[:400].decode("utf-8", errors="ignore")
            st.error("Failed to fetch DEM from OpenTopography.")
            st.code(preview or str(r.status_code))
            st.stop()

        try:
            dem_arr, ncols, nrows, g_west, g_south, g_east, g_north, cellsize = parse_aaigrid(r.content)
        except Exception as e:
            preview = r.content[:400].decode("utf-8", errors="ignore")
            st.error(f"Failed to parse AAIGrid: {e}")
            st.code(preview)
            st.stop()

    # Clip to drawn polygon (not just bbox) and downsample
    dem_masked, lon_grid, lat_grid = clip_and_downsample_ascii(
        dem_arr, g_west, g_south, g_east, g_north, ring, max_side=max_side_px
    )
    if np.all(np.isnan(dem_masked)):
        st.error("DEM fetched, but AOI mask excluded everything. Try a larger AOI or different DEM.")
        st.stop()

    # Build axes in meters without PROJ (local tangent approximation)
    deg2rad = np.pi / 180.0
    ref_lat = ((south + north) / 2.0) * deg2rad
    m_per_deg_lon = 111320.0 * np.cos(ref_lat)
    m_per_deg_lat = 110540.0
    lon0 = np.nanmin(lon_grid)
    lat0 = np.nanmin(lat_grid)
    x_grid = (lon_grid - lon0) * m_per_deg_lon
    y_grid = (lat_grid - lat0) * m_per_deg_lat

    # 3D surface
    st.subheader("🌄 3D Terrain")
    fig = go.Figure(data=[go.Surface(x=x_grid, y=y_grid, z=dem_masked, showscale=True)])
    fig.update_scenes(xaxis_title_text="X (m)", yaxis_title_text="Y (m)", zaxis_title_text="Elevation (m)")
    fig.update_layout(height=700, scene_aspectmode="data", margin=dict(l=0, r=0, b=0, t=30),
                      title=f"{demtype} — 3D surface (ASCII Grid)")

    st.plotly_chart(fig, use_container_width=True)

    # Optional export
    if export_ascii:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".asc") as tmp:
                out_path = tmp.name
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"ncols         {dem_masked.shape[1]}\n")
                f.write(f"nrows         {dem_masked.shape[0]}\n")
                f.write(f"xllcorner     {float(np.nanmin(lon_grid))}\n")
                f.write(f"yllcorner     {float(np.nanmin(lat_grid))}\n")
                # Assume uniform spacing from grid (approx)
                if dem_masked.shape[1] > 1:
                    csx = (np.nanmax(lon_grid) - np.nanmin(lon_grid)) / dem_masked.shape[1]
                else:
                    csx = (g_east - g_west) / max(1, ncols)
                f.write(f"cellsize      {csx}\n")
                f.write("NODATA_value  -9999\n")
                out = np.where(np.isnan(dem_masked), -9999, dem_masked).astype(float)
                for row in out:
                    f.write(" ".join(f"{v:.3f}" for v in row) + "\n")
            st.success(f"Saved clipped ASCII Grid: {out_path}")
            with open(out_path, "rb") as fh:
                st.download_button("⬇️ Download DEM (ASCII Grid)", data=fh.read(), file_name="clipped_dem.asc")
        except Exception as e:
            st.warning(f"Could not save ASCII Grid: {e}")

st.caption("This build avoids GDAL/pyproj/Shapely for painless deployment. For globe apps, consider CesiumJS; for browser hillshade, MapLibre GL JS + deck.gl TerrainLayer.")
