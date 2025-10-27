# streamlit_app.py
# Namibia terrain picker (no GDAL/Shapely/pyproj).
# Adds: keyboard finish (Enter/F), manual AOI inputs (rectangle or polygon),
# and persists the 3D view across reruns.

import io
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from branca.element import MacroElement, Template

st.set_page_config(page_title="Namibia 3D Terrain Picker", layout="wide")

# ---------- Session state ----------
ss = st.session_state
if "map_key" not in ss:
    ss.map_key = 0
if "settings" not in ss:
    ss.settings = {
        "demtype": "SRTMGL1_E",
        "max_side_px": 600,
        "export_ascii": True,
        "auto_clear": True,
    }
if "manual_rings" not in ss:
    ss.manual_rings: List[List[List[float]]] = []  # list of rings [[ [lon,lat], ... ]]
if "last_fig" not in ss:
    ss.last_fig = None

# ---------- Sidebar: Settings (FORM stops live reruns) ----------
with st.sidebar.form("settings_form", clear_on_submit=False):
    st.header("Settings")
    demtype = st.selectbox(
        "DEM (OpenTopography Global DEM API)",
        ["SRTMGL1_E", "SRTMGL3", "NASADEM", "AW3D30"],
        index=["SRTMGL1_E", "SRTMGL3", "NASADEM", "AW3D30"].index(ss.settings["demtype"]),
        key="sb_demtype",
    )
    max_side_px = st.slider("Max grid size (downsample)", 100, 1000, ss.settings["max_side_px"], key="sb_maxside")
    export_ascii = st.checkbox("Save clipped DEM as ASCII Grid", value=ss.settings["export_ascii"], key="sb_export")
    auto_clear = st.checkbox("Auto-clear shapes after processing", value=ss.settings["auto_clear"], key="sb_autoclear")
    applied = st.form_submit_button("Apply")

if applied:
    ss.settings.update(
        {"demtype": demtype, "max_side_px": max_side_px, "export_ascii": export_ascii, "auto_clear": auto_clear}
    )

# Use the saved settings
demtype = ss.settings["demtype"]
max_side_px = ss.settings["max_side_px"]
export_ascii = ss.settings["export_ascii"]
auto_clear = ss.settings["auto_clear"]

# ---------- Sidebar: Manual AOI entry (RECTANGLE or POLYGON) ----------
with st.sidebar.expander("Manual AOI (no clicking)", expanded=False):
    st.markdown("**Rectangle (South/West/North/East)**")
    c1, c2 = st.columns(2)
    with c1:
        man_s = st.number_input("South (lat)", value=-23.0, step=0.01, key="man_s")
        man_w = st.number_input("West (lon)", value=16.5, step=0.01, key="man_w")
    with c2:
        man_n = st.number_input("North (lat)", value=-22.5, step=0.01, key="man_n")
        man_e = st.number_input("East (lon)", value=17.2, step=0.01, key="man_e")
    add_rect = st.button("➕ Add rectangle AOI", key="btn_add_rect", help="Adds as a ring to the current session")

    st.markdown("**Polygon (one `lat,lon` per line)**")
    poly_text = st.text_area(
        "Example:\n-22.90, 17.00\n-22.70, 17.10\n-22.80, 17.25",
        key="poly_text",
        height=120,
    )
    add_poly = st.button("➕ Add polygon AOI", key="btn_add_poly", help="Adds as a ring to the current session")

    if add_rect:
        # make a closed ring [lon,lat]
        ring = [[man_w, man_s], [man_e, man_s], [man_e, man_n], [man_w, man_n], [man_w, man_s]]
        ss.manual_rings.append(ring)
        st.success("Rectangle added.")
    if add_poly:
        try:
            pts = []
            for line in poly_text.strip().splitlines():
                if not line.strip():
                    continue
                lat_str, lon_str = [p.strip() for p in line.split(",")]
                lat = float(lat_str); lon = float(lon_str)
                pts.append([lon, lat])  # convert to [lon,lat]
            if len(pts) >= 3:
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                ss.manual_rings.append(pts)
                st.success(f"Polygon with {len(pts)-1} vertices added.")
            else:
                st.warning("Need at least 3 points.")
        except Exception as e:
            st.error(f"Could not parse polygon: {e}")

# ---------- API key ----------
API_KEY = None
try:
    API_KEY = st.secrets.get("OPENTOPO_API_KEY")
except Exception:
    API_KEY = None
if not API_KEY:
    API_KEY = os.environ.get("OPENTOPO_API_KEY")

# ---------- Constants ----------
NAMIBIA_BBOX = {"south": -28.97, "west": 11.73, "north": -16.95, "east": 25.26}
DEFAULT_CENTER = [-22.56, 17.08]  # lat, lon

# ---------- Header ----------
st.title("🗺️ Draw a polygon → 3D terrain map (Namibia)")
st.write(
    "Finish drawing by **double-clicking**, **pressing Enter**, or pressing **F** (we capture the key). "
    "Or skip clicking entirely and use **Manual AOI** in the sidebar."
)

# ---------- Controls above map ----------
colA, colB, _ = st.columns([1, 1, 6])
with colA:
    if st.button("🧹 Clear shapes", key="btn_clear", help="Reset the drawing layer and start fresh"):
        ss.map_key += 1
        ss.manual_rings = []
        st.rerun()
with colB:
    zoom_click = st.button("🔎 Zoom to Namibia", key="btn_zoom", help="Refit map to Namibia extent")

# ---------- Build Folium map ----------
m = folium.Map(location=DEFAULT_CENTER, zoom_start=6, tiles="CartoDB positron", control_scale=True)

# Namibia outline
folium.Rectangle(
    bounds=[[NAMIBIA_BBOX["south"], NAMIBIA_BBOX["west"]],
            [NAMIBIA_BBOX["north"], NAMIBIA_BBOX["east"]]],
    color="#1f77b4", weight=2, dash_array="6,6", fill=False, tooltip="Namibia extent (approx)",
).add_to(m)

if zoom_click:
    m.fit_bounds([[NAMIBIA_BBOX["south"], NAMIBIA_BBOX["west"]],
                  [NAMIBIA_BBOX["north"], NAMIBIA_BBOX["east"]]])

# ---- Disable double-click zoom AND bind keyboard finish (Enter/F) ----
finish_macro = Template("""
{% macro script(this, kwargs) %}
    var map = {{this._parent.get_name()}};
    if (map.doubleClickZoom) { map.doubleClickZoom.disable(); }

    function findDrawControl() {
        if (!map._controls) return null;
        for (var i=0; i<map._controls.length; i++) {
            var c = map._controls[i];
            if (typeof L !== 'undefined' && L.Control && L.Control.Draw && (c instanceof L.Control.Draw)) {
                return c;
            }
        }
        return null;
    }
    function finishIfDrawing() {
        var dc = findDrawControl();
        var handler = dc && dc._toolbars && dc._toolbars.draw && dc._toolbars.draw._activeMode && dc._toolbars.draw._activeMode.handler;
        if (handler && handler._finishShape) { handler._finishShape(); return true; }
        return false;
    }
    document.addEventListener('keydown', function(e){
        var k = (e.key || '').toLowerCase();
        if (k === 'enter' || k === 'f') {
            if (finishIfDrawing()) { e.preventDefault(); }
        }
    });
{% endmacro %}
""")
macro = MacroElement(); macro._template = finish_macro
m.get_root().add_child(macro)

# Draw controls (Leaflet.draw)
polygon_opts = {
    "allowIntersection": True,
    "showArea": True,
    "shapeOptions": {"weight": 2},
    "repeatMode": False,
    "finishOnDoubleClick": True,   # harmless if ignored
}
rectangle_opts = {"shapeOptions": {"weight": 2}, "repeatMode": False}

Draw(
    draw_options={
        "polyline": False,
        "rectangle": rectangle_opts,
        "polygon": polygon_opts,
        "circle": False,
        "circlemarker": False,
        "marker": False,
    },
    edit_options={"edit": True, "remove": True}
).add_to(m)

# Also render any MANUAL rings the user added
for ring in ss.manual_rings:
    folium.Polygon(locations=[[lat, lon] for (lon, lat) in ring],  # folium wants [lat, lon]
                   color="#ff7f0e", weight=2, fill=True, fill_opacity=0.15,
                   tooltip="Manual AOI").add_to(m)

# Big canvas; stable key so drawing state persists unless reset
map_data = st_folium(
    m, height=820, use_container_width=True,
    returned_objects=["last_active_drawing", "all_drawings"], key=f"map_{ss.map_key}",
)

process_clicked = st.button("🚀 Process AOI → Fetch DEM → 3D Render", key="btn_process")

# ---------- Helpers ----------
def normalize_lon(lon: float) -> float:
    lon = ((lon + 180.0) % 360.0) - 180.0
    return -180.0 if abs(lon + 180.0) < 1e-9 else lon

def ensure_lnglat_coords(geometry: Dict) -> List[List[List[float]]]:
    def fix_ring(ring):
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        swap = any(abs(x) > 90 for x in xs) or (min(ys) > 10 or max(ys) < -35)
        return [[p[1], p[0]] for p in ring] if swap else ring
    rings: List[List[List[float]]] = []
    t = geometry.get("type")
    if t == "Polygon":
        rings.append(fix_ring(geometry["coordinates"][0]))
    elif t == "MultiPolygon":
        for poly in geometry["coordinates"]:
            rings.append(fix_ring(poly[0]))
    return rings

def extract_all_rings(map_obj: Dict) -> List[List[List[float]]]:
    rings: List[List[List[float]]] = []
    # from drawn geometry
    if map_obj and map_obj.get("last_active_drawing"):
        g = map_obj["last_active_drawing"].get("geometry")
        if g and g.get("type") in ("Polygon", "MultiPolygon"):
            rings.extend(ensure_lnglat_coords(g))
    if map_obj and map_obj.get("all_drawings"):
        for feat in map_obj["all_drawings"]:
            g = feat.get("geometry")
            if g and g.get("type") in ("Polygon", "MultiPolygon"):
                rings.extend(ensure_lnglat_coords(g))
    # include manual rings (already [lon,lat])
    rings.extend(ss.manual_rings)
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
    step_h = max(1, int(round(scale)))
    step_w = max(1, int(round(scale)))
    return arr[::step_h, ::step_w], lon[::step_h, ::step_w], lat[::step_h, ::step_w]

# ---------- Main ----------
if process_clicked:
    if not API_KEY:
        st.error("Missing OpenTopography API key. Set OPENTOPO_API_KEY in secrets or env.")
        st.stop()

    rings = extract_all_rings(map_data)
    if not rings:
        st.error("No AOI found. Use drawing tools or add a Manual AOI in the sidebar.")
        st.stop()

    # Intersect union bbox with Namibia to keep API request valid
    s, w, n, e = bounds_of_rings(rings)
    west = max(w, NAMIBIA_BBOX["west"])
    east = min(e, NAMIBIA_BBOX["east"])
    south = max(s, NAMIBIA_BBOX["south"])
    north = min(n, NAMIBIA_BBOX["north"])
    if not (west < east and south < north):
        st.error("Your shapes are outside Namibia. Please draw inside the dashed rectangle.")
        st.stop()

    west = normalize_lon(west); east = normalize_lon(east)
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

    # Build mask for union of all rings
    lon_grid, lat_grid = make_lonlat_grids(g_west, g_south, g_east, g_north, ncols, nrows)
    union_mask = np.zeros_like(dem, dtype=bool)
    for ring in rings:
        union_mask |= points_in_polygon(lon_grid, lat_grid, ring)
    dem_masked = np.where(union_mask, dem, np.nan)

    dem_ds, lon_ds, lat_ds = downsample(dem_masked, lon_grid, lat_grid, max_side=max_side_px)
    if np.all(np.isnan(dem_ds)):
        st.error("DEM fetched, but union mask excluded everything. Try a larger AOI or different DEM.")
        st.stop()

    # Approx meters for axes
    ref_lat = ((south + north) / 2.0) * (np.pi / 180.0)
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
    ss.last_fig = fig  # <-- persist plot so it survives future reruns

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
            with open(out_path, "rb") as fh:
                st.download_button("⬇️ Download DEM (ASCII Grid)", data=fh.read(), file_name="clipped_dem.asc", key="dl_dem")
        except Exception as e:
            st.warning(f"Could not save ASCII Grid: {e}")

    # Optional auto-clear of map drawings AFTER plotting; figure stays via session_state
    if auto_clear:
        ss.map_key += 1
        ss.manual_rings = []
        st.toast("Shapes cleared (3D view kept).")
        # no st.rerun() → keeps the plot visible

# If a rerun happened (e.g., you adjusted the sidebar), show the last plot
if ss.last_fig is not None and not process_clicked:
    st.subheader("🌄 3D Terrain (last result)")
    st.plotly_chart(ss.last_fig, use_container_width=True)

st.caption(
    "Finishing options: **double-click**, **Enter**, or **F**. "
    "Or use the **Manual AOI** expander to type a rectangle or polygon (lat,lon per line). "
    "We keep the last 3D plot visible even after reruns."
)
