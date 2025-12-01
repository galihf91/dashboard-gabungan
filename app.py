# app.py
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import datetime
import re
import numpy as np
from io import StringIO

st.set_page_config(page_title="Status Tera Ulang Pasar – Kab. Tangerang",
                   page_icon="🏪", layout="wide")

st.title("🏪 Status Tera Ulang Timbangan Pasar – Kabupaten Tangerang")
st.caption("Dinas Perindustrian dan Perdagangan • Bidang Kemetrologian")

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def parse_coord(val):
    """Parse koordinat dari berbagai format"""
    try:
        if pd.isna(val) or val == "":
            return np.nan, np.nan
        
        s = str(val).strip()
        # Handle format: "-6.26435, 106.42592"
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            if len(parts) >= 2:
                lat = float(parts[0])
                lon = float(parts[1])
                # Auto-swap jika format terbalik (lon, lat)
                if abs(lat) > 90 and abs(lon) <= 90:
                    lat, lon = lon, lat
                return lat, lon
    except Exception as e:
        st.warning(f"Gagal parse koordinat: {val}, error: {e}")
    return np.nan, np.nan

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    rename_mapping = {
        'Nama Pasar': 'nama_pasar',
        'Alamat': 'alamat',
        'Kecamatan': 'kecamatan', 
        'Koordinat': 'koordinat',
        'Tahun Tera Ulang': 'tera_ulang_tahun',
        'Total UTTP': 'jumlah_timbangan_tera_ulang',
        'Total Pedagang': 'total_pedagang'
    }
    
    existing_rename = {k: v for k, v in rename_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_rename)
    
    if 'koordinat' in df.columns:
        coords = df['koordinat'].apply(parse_coord)
        df['lat'] = coords.apply(lambda x: x[0])
        df['lon'] = coords.apply(lambda x: x[1])
    
    timbangan_cols = ['Timb. Pegas', 'Timb. Meja', 'Timb. Elektronik', 
                      'Timb. Sentisimal', 'Timb. Bobot Ingsut', 'Neraca']
    
    available_timbangan_cols = [col for col in timbangan_cols if col in df.columns]
    
    if available_timbangan_cols:
        def summarize_timbangan(row):
            parts = []
            for col in available_timbangan_cols:
                try:
                    val = pd.to_numeric(row[col], errors='coerce')
                    if pd.notna(val) and val > 0:
                        label = col.replace('Timb. ', '').replace('Timb.', '')
                        parts.append(f"{label}: {int(val)}")
                except Exception:
                    continue
            return "; ".join(parts) if parts else "Tidak ada data"
        
        df['jenis_timbangan'] = df.apply(summarize_timbangan, axis=1)
    
    return df


@st.cache_data
def load_csv(path: str):
    import csv
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8-sig", errors="ignore")

        lines = text.splitlines()
        # Deteksi apakah mayoritas baris terkutip penuh
        if len(lines) > 1:
            quoted = sum(1 for l in lines[1:] if l.strip().startswith('"') and l.strip().endswith('"'))
            if quoted >= 0.8 * max(1, len(lines[1:])):
                fixed = [lines[0]]
                for line in lines[1:]:
                    s = line.strip()
                    if s.startswith('"') and s.endswith('"'):
                        s = s[1:-1].replace('""','"')
                    fixed.append(s)
                text = "\n".join(fixed)

        # Tentukan delimiter (fallback koma)
        try:
            sep = csv.Sniffer().sniff(lines[0]).delimiter
        except Exception:
            sep = ","

        df = pd.read_csv(StringIO(text), sep=sep)
        df = standardize_columns(df)
        return df, None

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return create_sample_data(), "Menggunakan data sample"


def create_sample_data():
    """Buat data sample jika file asli bermasalah"""
    st.warning("Membuat data sample karena file asli bermasalah")
    
    sample_data = {
        'nama_pasar': ['Cisoka', 'Curug', 'Mauk', 'Cikupa', 'Pasar Kemis'],
        'kecamatan': ['Cisoka', 'Curug', 'Mauk', 'Cikupa', 'Pasar Kemis'],
        'alamat': [
            'Jl. Ps. Cisoka No.44, Cisoka, Kec. Cisoka, Kabupaten Tangerang, Banten 15730',
            'Jl. Raya Curug, Curug Wetan, Kec. Curug, Kabupaten Tangerang, Banten 15810', 
            'East Mauk, Mauk, Tangerang Regency, Banten 15530',
            'Jl. Raya Serang, Cikupa, Kec. Cikupa, Kabupaten Tangerang, Banten 15710',
            'RGPJ+FJX, Jalan Raya, Sukaasih, Pasar Kemis, Tangerang Regency, Banten 15560'
        ],
        'lat': [-6.26435, -6.26100, -6.06044, -6.22907, -6.16365],
        'lon': [106.42592, 106.55858, 106.51129, 106.51981, 106.53155],
        'tera_ulang_tahun': [2025, 2025, 2025, 2025, 2025],
        'jumlah_timbangan_tera_ulang': [195, 251, 161, 257, 174],
        'jenis_timbangan': [
            'Pegas: 77; Meja: 30; Elektronik: 87',
            'Pegas: 60; Meja: 76; Elektronik: 107', 
            'Pegas: 80; Meja: 10; Elektronik: 71',
            'Pegas: 36; Meja: 88; Elektronik: 130',
            'Pegas: 54; Meja: 48; Elektronik: 72'
        ]
    }
    return pd.DataFrame(sample_data)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan tipe data konsisten"""
    if 'tera_ulang_tahun' in df.columns:
        df['tera_ulang_tahun'] = pd.to_numeric(df['tera_ulang_tahun'], errors='coerce').fillna(0).astype(int)
    
    if 'jumlah_timbangan_tera_ulang' in df.columns:
        df['jumlah_timbangan_tera_ulang'] = pd.to_numeric(df['jumlah_timbangan_tera_ulang'], errors='coerce').fillna(0).astype(int)
    
    for col in ['nama_pasar', 'alamat', 'kecamatan', 'jenis_timbangan']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    
    for c in ['lat', 'lon']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df

def clean_str_series(series: pd.Series) -> pd.Series:
    """Bersihkan series string"""
    if series is None:
        return pd.Series([], dtype=str)
    s = series.astype(str).str.strip()
    s = s.str.title()
    mask_bad = s.str.lower().isin(["", "nan", "none", "null", "na", "n/a", "-", "--"])
    return s[~mask_bad]

def uniq_clean(series: pd.Series) -> list:
    """Ambil nilai unik yang sudah dibersihkan"""
    return sorted(clean_str_series(series).unique().tolist())

def marker_color(year: int):
    """Tentukan warna marker berdasarkan tahun"""
    this_year = datetime.now().year
    if year is None or year == 0:
        return "gray"
    if year >= this_year:
        return "green"
    elif year == this_year - 1:
        return "orange"
    else:
        return "red"

# === MAIN APP ===
df, err = load_csv("DATA DASHBOARD PASAR.csv")

if err:
    st.warning(f"Peringatan: {err}")

df = coerce_types(df)

# === SIDEBAR FILTERS ===
with st.sidebar:
    st.header("Filter")
    mode = st.radio(
        "Mode pemilihan",
        options=["Pilih Kecamatan dulu → pilih Pasar", "Langsung pilih Pasar"],
        index=0
    )

    # --- Slider tahun ---
    if 'tera_ulang_tahun' in df.columns and df['tera_ulang_tahun'].notna().any():
        year_min = int(df['tera_ulang_tahun'].min())
        year_max = int(df['tera_ulang_tahun'].max())
    else:
        year_min = datetime.now().year - 5
        year_max = datetime.now().year

    if year_min == year_max:
        year_sel = (year_min, year_max)
        st.info(f"Data hanya punya satu tahun: {year_min}")
    else:
        year_sel = st.slider(
            "Rentang Tahun Tera Ulang",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1
        )

    # --- Daftar kecamatan & pasar ---
    all_kec = uniq_clean(df['kecamatan']) if 'kecamatan' in df.columns else []
    all_pasar = uniq_clean(df['nama_pasar']) if 'nama_pasar' in df.columns else []

    if mode.startswith("Pilih Kecamatan"):
        kec_opsi = ["(Semua)"] + all_kec
        kec = st.selectbox("Kecamatan", kec_opsi, index=0)

        if kec != "(Semua)" and {'kecamatan', 'nama_pasar'}.issubset(df.columns):
            pasar_opsi = ["(Semua)"] + uniq_clean(df.loc[df['kecamatan'] == kec, 'nama_pasar'])
        else:
            pasar_opsi = ["(Semua)"] + all_pasar

        nama_pasar = st.selectbox("Nama Pasar", pasar_opsi, index=0)
    else:
        kec = "(Semua)"
        pasar_opsi = ["(Semua)"] + all_pasar
        nama_pasar = st.selectbox("Nama Pasar", pasar_opsi, index=0)

        # --- Kartu info pasar terpilih (versi ungu elegan) ---
    if ('nama_pasar' in df.columns) and (nama_pasar != "(Semua)"):
        info = df.loc[df['nama_pasar'] == nama_pasar].head(1)
        if not info.empty:
            nama = info['nama_pasar'].iat[0]
            alamat = info['alamat'].iat[0] if 'alamat' in info.columns else "Alamat tidak tersedia"
            kecamatan = info['kecamatan'].iat[0] if 'kecamatan' in info.columns else "–"

            st.markdown("---")
            st.markdown(
                f"""
                <div style="
                    background-color:#f3e8ff;              /* ungu pastel */
                    padding:14px 16px;
                    border-radius:12px;
                    border-left:5px solid #8000FF;         /* garis tepi ungu */
                    box-shadow:0px 1px 4px rgba(0,0,0,0.15);
                    margin-top:10px;
                    ">
                    <h4 style="margin-bottom:6px; color:#4B0082; font-size:16px;">
                        🏪 {nama}
                    </h4>
                    <p style="margin:2px 0; font-size:13px; color:#222;">
                        <b>Kecamatan:</b> {kecamatan}<br>
                        <b>Alamat:</b> {alamat}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    # === Ringkasan Timbangan di Sidebar ===
    st.markdown("---")
    st.subheader("⚖️ Total Timbangan Tera Ulang")

    timb_map = {
        "Pegas":        ["Timb. Pegas", "Timb Pegas", "Pegas"],
        "Meja":         ["Timb. Meja", "Timb Meja", "Meja"],
        "Elektronik":   ["Timb. Elektronik", "Timb Elektronik", "Elektronik"],
        "Sentisimal":   ["Timb. Sentisimal", "Timb Sentisimal", "Sentisimal"],
        "Bobot Ingsut": ["Timb. Bobot Ingsut", "Timb Bobot Ingsut", "Bobot Ingsut"],
        "Neraca":       ["Neraca"]
    }

    def sum_first_existing(df_src: pd.DataFrame, candidates: list) -> int:
        for c in candidates:
            if c in df_src.columns:
                return int(pd.to_numeric(df_src[c], errors="coerce").fillna(0).sum())
        return 0

    # Gunakan df yang sudah terfilter agar sesuai pilihan tahun/kecamatan/pasar
    fdf_sidebar = df.copy()
    if 'tera_ulang_tahun' in fdf_sidebar.columns:
        fdf_sidebar = fdf_sidebar[(fdf_sidebar['tera_ulang_tahun'] >= year_sel[0]) &
                                  (fdf_sidebar['tera_ulang_tahun'] <= year_sel[1])]
    if 'kecamatan' in fdf_sidebar.columns and kec != "(Semua)":
        fdf_sidebar = fdf_sidebar[fdf_sidebar['kecamatan'] == kec]
    if 'nama_pasar' in fdf_sidebar.columns and nama_pasar != "(Semua)":
        fdf_sidebar = fdf_sidebar[fdf_sidebar['nama_pasar'] == nama_pasar]

    totals = {label: sum_first_existing(fdf_sidebar, cands) for label, cands in timb_map.items()}

    if 'jumlah_timbangan_tera_ulang' in fdf_sidebar.columns:
        total_uttp = int(pd.to_numeric(fdf_sidebar['jumlah_timbangan_tera_ulang'],
                                       errors='coerce').fillna(0).sum())
    else:
        total_uttp = sum(totals.values())

    # Tampilkan dengan gaya ringkas
    for label, val in totals.items():
        st.markdown(f"**{label}**: {val:,}")

    st.markdown(f"**Total UTTP (semua jenis):** {total_uttp:,}")
    st.markdown("---")



# === FILTER DATA ===
fdf = df.copy()
if 'tera_ulang_tahun' in fdf.columns:
    fdf = fdf[(fdf['tera_ulang_tahun'] >= year_sel[0]) & (fdf['tera_ulang_tahun'] <= year_sel[1])]
if 'kecamatan' in fdf.columns and kec != "(Semua)":
    fdf = fdf[fdf['kecamatan'] == kec]
if 'nama_pasar' in fdf.columns and nama_pasar != "(Semua)":
    fdf = fdf[fdf['nama_pasar'] == nama_pasar]


# === KPIs ===
if kec == "(Semua)" and nama_pasar == "(Semua)":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_kec = clean_str_series(df['kecamatan']).nunique() if 'kecamatan' in df.columns else 0
        st.metric("Total Kecamatan", total_kec)
    with c2:
        total_pasar = clean_str_series(df['nama_pasar']).nunique() if 'nama_pasar' in df.columns else 0
        st.metric("Total Seluruh Pasar", total_pasar)
    with c3:
        if 'tera_ulang_tahun' in fdf.columns and fdf['tera_ulang_tahun'].notna().any():
            latest_year = int(fdf['tera_ulang_tahun'].max())
        else:
            latest_year = "–"
        st.metric("Tahun Tera Ulang Terbaru", latest_year)
    with c4:
        total_timb = int(fdf['jumlah_timbangan_tera_ulang'].fillna(0).sum()) if 'jumlah_timbangan_tera_ulang' in fdf.columns else 0
        st.metric("Total Timbangan Tera Ulang", total_timb)
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        display_name = kec if kec != "(Semua)" else nama_pasar
        st.metric("Kecamatan Terpilih", display_name)
    with c2:
        if 'tera_ulang_tahun' in fdf.columns and fdf['tera_ulang_tahun'].notna().any():
            latest_year = int(fdf['tera_ulang_tahun'].max())
        else:
            latest_year = "–"
        st.metric("Tahun Tera Ulang Terbaru", latest_year)
    with c3:
        total_timb = int(fdf['jumlah_timbangan_tera_ulang'].fillna(0).sum()) if 'jumlah_timbangan_tera_ulang' in fdf.columns else 0
        st.metric("Total Timbangan Tera Ulang", total_timb)

# === MAP ===
st.subheader("🗺️ Peta Lokasi Pasar")

default_center = [-6.2, 106.55]
default_zoom = 10

has_coords = {'lat','lon'}.issubset(fdf.columns)
coords = None
if has_coords:
    try:
        coords = fdf[['lat','lon']].astype(float).dropna()
        # st.write(f"📍 {len(coords)} lokasi dengan koordinat valid") 
    except Exception as e:
        st.warning(f"Error processing coordinates: {e}")
        coords = None

center_loc = default_center
zoom_start = default_zoom

if 'nama_pasar' in fdf.columns and nama_pasar != "(Semua)" and coords is not None and not coords.empty:
    row_sel = fdf[fdf['nama_pasar'] == nama_pasar]
    if not row_sel.empty:
        try:
            lat0 = float(row_sel['lat'].iloc[0])
            lon0 = float(row_sel['lon'].iloc[0])
            if pd.notna(lat0) and pd.notna(lon0):
                center_loc = [lat0, lon0]
                zoom_start = 16
        except Exception:
            pass
elif coords is not None and len(coords) == 1:
    center_loc = [coords.iloc[0]['lat'], coords.iloc[0]['lon']]
    zoom_start = 14

# Buat peta
m = folium.Map(location=center_loc, zoom_start=zoom_start, control_scale=True, tiles="OpenStreetMap")

# === Pane khusus agar batas di atas marker ===
from folium.map import CustomPane
m.add_child(CustomPane("batas-top", z_index=650))  # markerPane = 600, jadi 650 di atas marker

# Muat batas administrasi dari file GeoJSON
import geopandas as gpd

# Nama file GeoJSON kamu
batas_path = "batas_kecamatan_tangerang.geojson"

batas = gpd.read_file(batas_path)

# Pastikan CRS WGS84 (lat-lon)
if not batas.crs or batas.crs.to_epsg() != 4326:
    batas = batas.to_crs(epsg=4326)

# (opsional) sederhanakan geometri biar ringan
try:
    batas["geometry"] = batas.geometry.simplify(0.0005, preserve_topology=True)
except Exception:
    pass


# (opsional) sederhanakan geometri biar ringan
try:
    batas["geometry"] = batas.geometry.simplify(0.0005, preserve_topology=True)
except Exception:
    pass

# === Gambar Batas Kabupaten / Kecamatan tanpa bayangan ===
kolom_nama = next(
    (c for c in ["NAMOBJ", "WADMKC", "KECAMATAN", "NAMA_KEC", "NAMA_KECAMATAN", "Kecamatan"]
     if c in batas.columns),
    None
)

tooltip_args = {}
if kolom_nama:
    tooltip_args["tooltip"] = folium.GeoJsonTooltip(
        fields=[kolom_nama],
        aliases=["Kecamatan:"]
    )

# Garis merah tebal tanpa bayangan / isi warna
folium.GeoJson(
    batas,
    name="Batas Kecamatan",
    pane="batas-top",
    style_function=lambda x: {
        "color": "#8000FF",   # ungu khas Tangerang
        "weight": 2,          # atur ketebalan
        "opacity": 1.0,       # full opacity
        "fill": False,        # tidak ada isi warna
        "fillOpacity": 0,     # transparan
    },
    **tooltip_args
).add_to(m)

if has_coords and coords is not None and not coords.empty:
    cluster = MarkerCluster(name="Pasar", show=True).add_to(m)
    
    for _, r in fdf.iterrows():
        try:
            lat = float(r.get('lat', float('nan')))
            lon = float(r.get('lon', float('nan')))
        except Exception:
            lat, lon = float("nan"), float("nan")
        
        if pd.isna(lat) or pd.isna(lon):
            continue

        nama = str(r.get('nama_pasar', 'Unknown'))
        alamat = str(r.get('alamat', 'Tidak ada alamat'))
        tahun = r.get('tera_ulang_tahun', None)
        jumlah = r.get('jumlah_timbangan_tera_ulang', None)
        jenis = str(r.get('jenis_timbangan', 'Tidak ada data'))

        # Buat popup content
        html = f"""
        <div style='width: 280px; font-family: Arial, sans-serif;'>
            <h4 style='margin:8px 0; color: #2E86AB;'>{nama}</h4>
            <div style='font-size: 12px; color:#666; margin-bottom:8px;'>{alamat}</div>
            <hr style='margin:6px 0'/>
            <table style='font-size: 12px; width: 100%;'>
                <tr><td><b>Tera Ulang</b></td><td style='padding-left:8px'>: {tahun if pd.notna(tahun) else 'Tidak ada data'}</td></tr>
                <tr><td><b>Jumlah Timbangan</b></td><td style='padding-left:8px'>: {jumlah if pd.notna(jumlah) else 'Tidak ada data'}</td></tr>
                <tr><td><b>Jenis Timbangan</b></td><td style='padding-left:8px'>: {jenis}</td></tr>
            </table>
        </div>
        """
        
        tooltip_text = f"{nama} - {tahun if pd.notna(tahun) else 'Tahun tidak diketahui'}"
        popup = folium.Popup(html, max_width=320)
        tooltip = folium.Tooltip(tooltip_text)

        # Tambahkan marker ke peta
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color=marker_color(int(tahun) if pd.notna(tahun) else None),
            fill=True,
            fill_color=marker_color(int(tahun) if pd.notna(tahun) else None),
            fill_opacity=0.7,
            weight=2,
            tooltip=tooltip,
            popup=popup
        ).add_to(cluster)

    # Auto-fit bounds jika multiple locations
    if not ('nama_pasar' in fdf.columns and nama_pasar != "(Semua)") and len(coords) > 1:
        try:
            sw = [coords['lat'].min(), coords['lon'].min()]
            ne = [coords['lat'].max(), coords['lon'].max()]
            m.fit_bounds([sw, ne], padding=(30, 30))
        except Exception as e:
            st.warning(f"Tidak bisa auto-fit peta: {e}")
else:
    st.warning("⚠️ Tidak ada data koordinat yang valid untuk ditampilkan di peta")

# Tampilkan peta
st_folium(m, height=500, use_container_width=True)