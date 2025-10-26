# streamlit_app.py
# -------------------------------------------------------------
# Streamlit app: Draw a polygon over Namibia -> fetch DEM (AAIGrid) -> 3D terrain
# No GDAL/Rasterio required. Cloud-safe.
# -------------------------------------------------------------
# Run locally:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   export OPENTOPO_API_KEY="your-opentopo-key"
#   streamlit run streamlit_app.py

import io
import math
import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from shapely.geometry import shape, Polygon, mapping
from shapely import vectorized
from pyproj import Transformer

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

# -------------------------------------------------------------
# Page config
# -------------------------------------------------------------
st.set_page_config(page_title="Namibia 3D Terrain Picker (Cloud-Safe)", layout="wide")
st.title("🗺️ Draw a polygon → 3D terrain map (Namibia, cloud-safe)")

st.write(
    "Draw a polygon or rectangle *inside Namibia*. We'll fetch a DEM from OpenTopography "
    "(ASCII Grid), clip to your AOI, and render a 3D surface — without GDAL/Rasterio."
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
    export_ascii = st.checkbox("Save clipped DEM as ASCII Grid", value=True)

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

# Namibia bounding box (approx lon/lat)
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
# Helpers (no GDAL)
# -------------------------------------------------------------
def normalize_lon(lon: float) -> float:
    """Wrap any longitude to [-180, 180]."""
    lon = ((lon + 180.0) % 360.0) - 180.0
    return -180.0 if abs(lon + 180.0) < 1e-9 else lon

def poly_bbox(poly):
    minx, miny, maxx, maxy = poly.bounds
    return miny, minx, maxy, maxx  # south, west, north, east

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
    Uses simple heuristics + Namibia lat range to decide if a swap is needed.
    """
    def fix_ring(ring):
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        # If x looks like latitude (>|90|) or y way outside southern Africa, swap.
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

def parse_aaigrid(text_bytes: bytes):
    """
    Parse ESRI ASCII Grid (AAIGrid) into (data, ncols, nrows, west, south, cellsize).
    Supports xllcorner/xllcenter and yllcorner/yllcenter.
    """
    s = text_bytes.decode("utf-8", errors="ignore").strip().splitlines()
    header = {}
    data_start = 0
    # Read header lines (first 6 lines typically)
    for i, line in enumerate(s[:10]):
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
            # First non-header encountered
            data_start = i
            break

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])
    nodata = float(header.get("nodata_value", -9999.0))

    # Determine lower-left corner
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

    # Read data rows (top to bottom = north to south)
    data_str = s[data_start : data_start + nrows]
    arr = np.loadtxt(io.StringIO("\n".join(data_str)), dtype=float)
    if arr.shape != (nrows, ncols):
        raise ValueError(f"AAIGrid shape mismatch: got {arr.shape}, expected {(nrows, ncols)}")

    # Replace NODATA with NaN
    arr = np.where(arr == nodata, np.nan, arr)

    # Compute north/east (for sanity)
    north = south + nrows * cellsize
    east = west + ncols * cellsize

    return arr, ncols, nrows, west, south, east, north, cellsize

def clip_and_downsample_ascii(arr, west, south, east, north, polygon, max_side=300):
    """
    Mask array to polygon using cell centers, then downsample by simple striding.
    """
    nrows, ncols = arr.shape

    # Build lon/lat centers (row 0 corresponds to north edge)
    lon_centers = np.linspace(west + 0.5, east - 0.5, ncols)  # add 0.5 * cellsize; cellsize assumed 1 deg unit? No.
    # Correct: use cellsize properly
    # recompute with cellsize from bounds and ncols/nrows
    cellsize_x = (east - west) / ncols
    cellsize_y = (north - south) / nrows
    lon_centers = np.linspace(west + 0.5 * cellsize_x, east - 0.5 * cellsize_x, ncols)
    lat_centers = np.linspace(north - 0.5 * cellsize_y, south + 0.5 * cellsize_y, nrows)

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    # Vectorized point-in-polygon mask
    inside = vectorized.contains(polygon, lon_grid, lat_grid)
    masked = np.where(inside, arr, np.nan)

    # Downsample (keep max_side)
    h, w = masked.shape
    scale = max(h, w) / float(max_side)
    if scale > 1.0:
        new_h = max(2, int(round(h / scale)))
        new_w = max(2, int(round(w / scale)))
        masked = masked[:: int(math.ceil(h / new_h)), :: int(math.ceil(w / new_w))]
        # Rebuild lon/lat grids to match
        lon_grid = lon_grid[:: int(math.ceil(h / new_h)), :: int(math.ceil(w / new_w))]
        lat_grid = lat_grid[:: int(math.ceil(h / new_h)), :: int(math.ceil(w / new_w))]

    return masked, lon_grid, lat_grid

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

    # Compute bbox (south, west, north, east) for API
    south, west, north, east = poly_bbox(geom)

    # Normalize and ensure non-zero area
    west = normalize_lon(west)
    east = normalize_lon(east)
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))
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

    st.info(f"AOI bounds: south={south:.5f}, west={west:.5f}, north={north:.5f}, east={east:.5f}")

    # Build API URL (ASCII Grid)
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype={demtype}&south={south}&north={north}&west={west}&east={east}"
        f"&outputFormat=AAIGrid&API_Key={API_KEY}"
    )

    # Download and parse AAIGrid
    with st.status("Downloading DEM (ASCII Grid) from OpenTopography…", expanded=False):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        # Quick sanity: ensure it's text-ish, not binary or HTML error
        ctype = (r.headers.get("Content-Type", "") or "").lower()
        if "text" not in ctype and "ascii" not in ctype and "plain" not in ctype:
            # still try to parse; if it fails, show preview
            pass
        try:
            dem_arr, ncols, nrows, g_west, g_south, g_east, g_north, cellsize = parse_aaigrid(r.content)
        except Exception as e:
            preview = r.content[:400].decode("utf-8", errors="ignore")
            st.error(f"Failed to parse AAIGrid: {e}")
            st.code(preview)
            st.stop()

    # Clip to polygon (cell-center test) and downsample
    dem_masked, lon_grid, lat_grid = clip_and_downsample_ascii(
        dem_arr, g_west, g_south, g_east, g_north, geom, max_side=max_side_px
    )

    if np.all(np.isnan(dem_masked)):
        st.error("DEM was fetched, but your AOI mask removed everything. Try a larger AOI or different DEM source.")
        st.stop()

    # Build axes in meters (UTM) for better aspect ratio in 3D
    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    utm_epsg = lonlat_to_utm_epsg(center_lon, center_lat)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True).transform
    x_grid, y_grid = to_utm(lon_grid, lat_grid)

    st.subheader("🌄 3D Terrain")
    fig = go.Figure(data=[go.Surface(x=x_grid, y=y_grid, z=dem_masked, showscale=True)])
    fig.update_scenes(
        xaxis_title_text="X (m)",
        yaxis_title_text="Y (m)",
        zaxis_title_text="Elevation (m)"
    )
    fig.update_layout(
        height=700,
        scene_aspectmode="data",
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"{demtype} — 3D surface (ASCII Grid)"
    )
    st.plotly_chart(fig, use_container_width=True)

    if export_ascii:
        # Save a clipped ASCII of just the masked area (keep bbox grid, NaNs outside polygon)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".asc") as tmp:
                out_path = tmp.name
            # Write ESRI ASCII Grid (keeping original bbox/resolution)
            nrows_o, ncols_o = dem_arr.shape
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"ncols         {ncols}\n")
                f.write(f"nrows         {nrows}\n")
                f.write(f"xllcorner     {g_west}\n")
                f.write(f"yllcorner     {g_south}\n")
                f.write(f"cellsize      {cellsize}\n")
                f.write(f"NODATA_value  -9999\n")
                # Data must be north->south
                out = np.where(np.isnan(dem_masked), -9999, dem_masked).astype(float)
                for row in out:
                    f.write(" ".join(f"{v:.3f}" for v in row) + "\n")
            st.success(f"Saved clipped ASCII Grid: {out_path}")
            with open(out_path, "rb") as fh:
                st.download_button("⬇️ Download DEM (ASCII Grid)", data=fh.read(), file_name="clipped_dem.asc")
        except Exception as e:
            st.warning(f"Could not save ASCII Grid: {e}")

st.caption("Tip: This cloud-safe build avoids GDAL/Rasterio. For globe apps, consider CesiumJS; for 2.5D terrain in the browser, MapLibre GL JS + deck.gl TerrainLayer.")
