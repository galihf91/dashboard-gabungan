import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")  # gaya bawaan yang lebih modern
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# ========== Helper untuk SIMPAN & MUAT model (h5 + pkl) ==========

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def make_key(komoditas: str, pasar: str, window_size: int) -> str:
    """
    Membuat 'key' sederhana untuk nama file model berdasarkan
    komoditas, pasar, dan window_size.
    Contoh: BAWANG_MERAH_CISOKA_30
    """
    def slug(s: str) -> str:
        return (
            str(s)
            .strip()
            .upper()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )
    return f"{slug(komoditas)}_{slug(pasar)}_{window_size}"

def get_model_paths(komoditas: str, pasar: str, window_size: int):
    key = make_key(komoditas, pasar, window_size)
    model_path = MODEL_DIR / f"{key}.h5"
    scaler_path = MODEL_DIR / f"{key}_scaler.pkl"
    return model_path, scaler_path

def save_model_and_scaler(model, scaler, komoditas, pasar, window_size):
    """
    Simpan model LSTM ke .h5 dan scaler MinMax ke .pkl
    """
    model_path, scaler_path = get_model_paths(komoditas, pasar, window_size)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path

def load_model_and_scaler(komoditas, pasar, window_size):
    model_path, scaler_path = get_model_paths(komoditas, pasar, window_size)
    if model_path.exists() and scaler_path.exists():
        try:
            # ⬇⬇ perhatikan compile=False ⬇⬇
            model = load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.error(f"Gagal memuat model dari file: {e}")
            return None, None
    else:
        return None, None

# =========================================================
# ⚠️ PERHATIAN:
# JANGAN pakai st.set_page_config di file halaman.
# set_page_config SUDAH ada di app.py (dashboard TERA).
# =========================================================

# -------------------------
# CSS & HEADER
# -------------------------
#CSS KATRTU
st.markdown(
    """
    <style>
    .komod-card {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .komod-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.18);
    }
    .komod-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 600;
        color: white;
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# CSS BACKGROUND UNGU (berlaku ke seluruh app saat halaman ini dibuka)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3E5F5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_base64_of_image(image_path: str) -> str:
    """
    Mengubah file gambar lokal menjadi string base64
    agar bisa dipakai di CSS background-image.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

# === Header dengan background foto ===
img_b64 = get_base64_of_image("assets/background_header.jpeg")

if img_b64:
    st.markdown(
        f"""
        <style>
        .header-banner {{
            width: 100%;
            height: 280px;
            background-image: url("data:image/jpeg;base64,{img_b64}");
            background-size: cover;
            background-position: center -300px;
            background-repeat: no-repeat;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        </style>

        <div class="header-banner"></div>
        """,
        unsafe_allow_html=True
    )

# Hilangkan menu & footer Streamlit (opsional)
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Header utama halaman ini
st.markdown(
    """
    <div style='text-align:center; margin-bottom: 15px;'>
        <h1 style='margin-bottom: 0;'>Dashboard Harga Barang & Prediksi</h1>
        <p style='font-size:14px; margin-top:4px; color:#555;'>
            Dinas Perindustrian & Perdagangan Kabupaten Tangerang – Analisis Harga Pasar
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------
# Fungsi Bantu Dataset Harga
# -------------------------
def reshape_price_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah dataset format lebar (TANGGAL, KOMODITI, PASAR CISOKA, PASAR SEPATAN)
    menjadi format long (tanggal, komoditas, pasar, harga).
    Jika sudah format long, akan dikembalikan apa adanya.
    """
    cols = [c.lower() for c in df_raw.columns]

    # Cek apakah sudah long format
    if {"tanggal", "komoditas", "pasar", "harga"}.issubset(set(cols)):
        df = df_raw.copy()
        # Normalkan nama kolom
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "tanggal":
                rename_map[c] = "tanggal"
            elif cl == "komoditas":
                rename_map[c] = "komoditas"
            elif cl == "pasar":
                rename_map[c] = "pasar"
            elif cl == "harga":
                rename_map[c] = "harga"
        df = df.rename(columns=rename_map)
        df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
        df["komoditas"] = df["komoditas"].astype(str).str.strip().str.upper()
        df["pasar"] = df["pasar"].astype(str).str.strip().str.upper()
        df["harga"] = pd.to_numeric(df["harga"], errors="coerce")
        df = df.dropna(subset=["tanggal", "harga"])
        df = df.sort_values(["tanggal", "komoditas", "pasar"])
        return df

    # Asumsi masih format lebar bawaan: TANGGAL, KOMODITI, PASAR CISOKA, PASAR SEPATAN
    df = df_raw.rename(columns={
        "TANGGAL": "tanggal",
        "KOMODITI": "komoditas",
        "PASAR CISOKA": "CISOKA",
        "PASAR SEPATAN": "SEPATAN"
    })

    expected_cols = ["tanggal", "komoditas", "CISOKA", "SEPATAN"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Kolom berikut tidak ditemukan di dataset: {missing}")

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

    df_long = df.melt(
        id_vars=["tanggal", "komoditas"],
        value_vars=["CISOKA", "SEPATAN"],
        var_name="pasar",
        value_name="harga"
    )

    df_long["komoditas"] = df_long["komoditas"].astype(str).str.strip().str.upper()
    df_long["pasar"] = df_long["pasar"].astype(str).str.strip().str.upper()
    df_long["harga"] = pd.to_numeric(df_long["harga"], errors="coerce")

    df_long = df_long.dropna(subset=["tanggal", "harga"])
    df_long = df_long.sort_values(["tanggal", "komoditas", "pasar"])

    return df_long


def create_window_dataset(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)


def train_lstm_for(df, komoditas, pasar, window_size=30, epochs=30):
    # Filter per komoditas & pasar
    df_sub = df[
        (df["komoditas"].str.upper() == komoditas.upper()) &
        (df["pasar"].str.upper() == pasar.upper())
    ].sort_values("tanggal")

    if df_sub.shape[0] <= window_size + 1:
        st.error(f"Data terlalu sedikit untuk {komoditas} - {pasar} (n={df_sub.shape[0]}). "
                 f"Kurangi window_size atau tambahkan data.")
        return None, None, None, None, None

    prices = df_sub["harga"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    X, y = create_window_dataset(prices_scaled, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(32))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    with st.spinner("Melatih model LSTM..."):
        history = model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

    y_pred = model.predict(X, verbose=0)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_true_inv = scaler.inverse_transform(y)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = math.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

    return model, scaler, df_sub, history, (mae, rmse)


def forecast_lstm(model, scaler, df_sub, n_days=30, window_size=30):
    last_window = df_sub["harga"].values[-window_size:].reshape(-1, 1)
    last_window_scaled = scaler.transform(last_window)

    preds_scaled = []
    current = last_window_scaled.copy()

    for _ in range(n_days):
        X_input = current[-window_size:].reshape(1, window_size, 1)
        pred = model.predict(X_input, verbose=0)
        preds_scaled.append(pred[0][0])
        current = np.vstack([current, pred])

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))

    last_date = pd.to_datetime(df_sub["tanggal"]).max()
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days)

    df_pred = pd.DataFrame({
        "tanggal": dates,
        "prediksi": preds.flatten()
    })
    return df_pred


@st.cache_data
def load_data():
    # pastikan file ini ada di folder utama bersama app.py
    df_raw = pd.read_csv("harga_pasar_2024.csv")
    df_clean = reshape_price_dataset(df_raw)
    return df_clean

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal membaca dataset 'harga_pasar_2024.csv': {e}")
    st.stop()

# Init session state untuk model
if "models" not in st.session_state:
    st.session_state["models"] = {}

# -------------------------
# Kategori style kartu komoditas
# -------------------------
def get_komoditas_style(nama: str):
    """
    Mengembalikan (kategori, bg_color, badge_color) berdasarkan nama komoditas.
    """
    n = str(nama).lower()

    # Beras
    if "beras" in n:
        return "BERAS", "#FFF8E1", "#F9A825"

    # Minyak goreng
    if "minyak" in n:
        return "MINYAK", "#FFF3E0", "#FB8C00"

    # Cabe / cabai
    if "cabe" in n or "cabai" in n or "rawit" in n:
        return "CABAI", "#FFEBEE", "#E53935"

    # Bawang
    if "bawang" in n:
        return "BAWANG", "#EDE7F6", "#8E24AA"

    # Tepung / terigu
    if "tepung" in n or "segitiga biru" in n:
        return "TEPUNG", "#E8F5E9", "#43A047"

    # Gula
    if "gula" in n:
        return "GULA", "#F3E5F5", "#7B1FA2"

    # Default / lain-lain
    return "PETERNAKAN", "#F5F5F5", "#757575"

# -------------------------
# Tab Layout
# -------------------------
tab1, tab2 = st.tabs(["📊 Harga Pasar", "🤖 Training & Prediksi"])

# =========================================================
# ========================= TAB 1 =========================
# =========================================================
with tab1:
    st.markdown("### 📊 Harga Komoditas per Pasar")

    pasar_list = sorted(df["pasar"].unique().tolist())
    pasar = st.selectbox("Pilih Pasar", pasar_list, key="pilih_pasar_tab1")

    # Filter data sesuai pasar terpilih
    df_pasar = df[df["pasar"] == pasar].copy()

    if df_pasar.empty:
        st.warning(f"Tidak ada data untuk pasar **{pasar}**.")
    else:
        df_pasar["tanggal"] = pd.to_datetime(df_pasar["tanggal"])
        min_date = df_pasar["tanggal"].min().date()
        max_date = df_pasar["tanggal"].max().date()

        # ---------- Layout utama: kiri (semua komoditas), kanan (detail komoditas) ----------
        col_left, col_right = st.columns([2, 1])

        # ===================== KIRI: semua komoditas per tanggal =====================
        with col_left:
            st.markdown("#### 📅 Pilih Tanggal")

            selected_date = st.date_input(
                "Tanggal harga yang ingin dilihat",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="tgl_pasar_kiri"
            )

            df_hari_ini = df_pasar[df_pasar["tanggal"].dt.date == selected_date].copy()

            if df_hari_ini.empty:
                st.warning(f"Tidak ada data untuk pasar **{pasar}** pada tanggal **{selected_date}**.")
            else:
                df_hari_ini = df_hari_ini.sort_values("komoditas")

                st.markdown(f"#### 💰 Daftar Harga Komoditas di Pasar **{pasar}** pada {selected_date}")

                # TAMPILAN KARTU KOMODITAS
                num_cols = 3
                cols = st.columns(num_cols)

                for i, row in df_hari_ini.iterrows():
                    c = cols[i % num_cols]
                    nama = str(row["komoditas"])
                    harga = row["harga"]
                    kategori, bg_color, badge_color = get_komoditas_style(nama)

                    with c:
                        st.markdown(
                            f"""
                            <div class="komod-card" style="
                                background-color: {bg_color};
                                padding: 14px 16px;
                                border-radius: 14px;
                                margin-bottom: 12px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                border: 1px solid rgba(0,0,0,0.08);
                            ">
                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                                    <div style="font-weight: 700; font-size: 14px;">
                                        {nama.upper()}
                                    </div>
                                    <span class="komod-badge" style="background-color: {badge_color};">
                                        {kategori}
                                    </span>
                                </div>
                                <div style="font-size: 12px; color: #555;">Harga</div>
                                <div style="font-size: 20px; font-weight: 800; color: #1A237E;">
                                    Rp {harga:,.0f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.markdown("#### 📈 Grafik Harga per Komoditas")

                df_plot = df_hari_ini.sort_values("komoditas").copy()

                fig = px.bar(
                    df_plot,
                    x="komoditas",
                    y="harga",
                    text="harga",
                    color="harga",
                    color_continuous_scale="Blues",
                )

                fig.update_traces(
                    texttemplate="Rp %{y:,.0f}",
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Harga: Rp %{y:,.0f}<extra></extra>",
                )

                fig.update_layout(
                    title={
                        "text": f"Harga Komoditas di Pasar {pasar} pada {selected_date}",
                        "x": 0.5,
                        "xanchor": "center",
                        "font": {"size": 16},
                    },
                    xaxis_title="Komoditas",
                    yaxis_title="Harga (Rp)",
                    xaxis_tickangle=-60,
                    template="plotly_white",
                    coloraxis_showscale=False,
                    margin=dict(l=40, r=20, t=60, b=80),
                )
                fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.15)")
                st.plotly_chart(fig, use_container_width=True)

        # ===================== KANAN: detail riwayat satu komoditas =====================
        with col_right:
            st.markdown("#### 🔍 Detail Per Komoditas")

            komoditas_pasar_list = sorted(df_pasar["komoditas"].unique().tolist())

            selected_option = st.selectbox(
                "Pilih komoditas",
                ["— Pilih komoditas —"] + komoditas_pasar_list,
                index=0,
                key=f"detail_{pasar}"
            )

            if selected_option == "— Pilih komoditas —":
                st.info("Pilih komoditas untuk melihat riwayat harganya.")
            else:
                komoditas_detail = selected_option
                df_view = df_pasar[df_pasar["komoditas"] == komoditas_detail].copy().sort_values("tanggal")

                if df_view.empty:
                    st.warning(f"Tidak ada data untuk {komoditas_detail}.")
                else:
                    df_view["tanggal"] = pd.to_datetime(df_view["tanggal"])

                    st.caption(
                        f"Periode: {df_view['tanggal'].min().date()} s.d. {df_view['tanggal'].max().date()}"
                    )

                    st.markdown("#### 📉 Grafik Riwayat Harga Komoditas")

                    df_plot = df_view.copy()
                    df_plot = df_plot.sort_values("tanggal")

                    fig = px.line(
                        df_plot,
                        x="tanggal",
                        y="harga",
                        markers=True,
                        line_shape="spline",
                    )

                    fig.update_traces(
                        line=dict(color="#1A73E8", width=3),
                        marker=dict(size=2, color="#0D47A1"),
                        hovertemplate="<b>%{x}</b><br>Harga: <b>Rp %{y:,.0f}</b><extra></extra>",
                    )

                    fig.update_layout(
                        title={
                            "text": f"Riwayat Harga {komoditas_detail} - Pasar {pasar}",
                            "x": 0.5,
                            "xanchor": "center",
                            "font": {"size": 16},
                        },
                        xaxis_title="Tanggal",
                        yaxis_title="Harga (Rp)",
                        template="plotly_white",
                        hovermode="x unified",
                        margin=dict(l=40, r=20, t=50, b=40),
                        paper_bgcolor="white",
                        plot_bgcolor="rgba(230,242,255,1)"
                    )

                    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.15)")
                    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.10)")
                    st.plotly_chart(fig, use_container_width=True)

# ==================================TAB 2==================
with tab2:
    st.markdown("### 🤖 Training Model & Prediksi Harga + Saran Kebijakan")

    # Filter data berdasarkan PASAR yang dipilih di Tab 1
    df_pasar = df[df["pasar"] == pasar].copy()

    if df_pasar.empty:
        st.warning(f"Tidak ada data untuk pasar **{pasar}**.")
    else:
        # 1. PILIH KOMODITAS
        komoditas_list_pasar = sorted(df_pasar["komoditas"].unique().tolist())
        komoditas = st.selectbox(
            "Pilih Komoditas untuk Dilatih",
            komoditas_list_pasar,
            key="komoditas_pilih_tab2"
        )

        # 2. PARAMETER MODEL
        st.markdown("#### ⚙️ Pengaturan Model LSTM")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            window_size = st.slider("Window size (hari historis)", 7, 60, 30)
        with col_b:
            forecast_days = st.slider("Hari prediksi", 7, 60, 30)
        with col_c:
            epochs = st.slider("Epoch training", 5, 100, 30, step=5)

        # 3. DATA UNTUK KOMBINASI KOMODITAS + PASAR
        df_sub = df[
            (df["komoditas"] == komoditas) &
            (df["pasar"] == pasar)
        ].sort_values("tanggal")

        if df_sub.empty:
            st.warning(f"Tidak ada data untuk komoditas **{komoditas}** di pasar **{pasar}**.")
            st.stop()

        st.markdown(
            f"**Kombinasi aktif:** {komoditas} – {pasar}  \n"
            f"Periode data: {df_sub['tanggal'].min().date()} s.d. {df_sub['tanggal'].max().date()}"
        )

        # 4. COBA MUAT MODEL TERLEBIH DAHULU
        model, scaler = load_model_and_scaler(komoditas, pasar, window_size)
        if model is not None and scaler is not None:
            st.success("✔ Model tersimpan ditemukan, siap digunakan untuk prediksi.")
        else:
            st.info("ℹ Belum ada model tersimpan untuk kombinasi ini. Silakan latih model terlebih dahulu.")

        # 5. TOMBOL TRAINING / UPDATE MODEL
        if st.button("🔁 Latih / Perbarui Model untuk Kombinasi Ini"):
            model, scaler, df_sub_trained, history, (mae, rmse) = train_lstm_for(
                df,
                komoditas=komoditas,
                pasar=pasar,
                window_size=window_size,
                epochs=epochs
            )
            if model is not None:
                # Simpan ke file .h5 dan .pkl
                model_path, scaler_path = save_model_and_scaler(
                    model, scaler, komoditas, pasar, window_size
                )
                st.success(
                    f"✅ Model berhasil dilatih & disimpan:\n\n"
                    f"- `{model_path}`\n"
                    f"- `{scaler_path}`"
                )
                df_sub = df_sub_trained  # pakai df_sub hasil training

        # Setelah tombol (atau kalau sudah punya model tersimpan):
        # cek lagi apakah model & scaler sudah ada
        if model is None or scaler is None:
            st.warning("⚠ Model belum tersedia. Tidak bisa melakukan prediksi.")
            st.stop()

        # 6. PREDIKSI
        df_pred = forecast_lstm(
            model,
            scaler,
            df_sub,
            n_days=forecast_days,
            window_size=window_size
        )

        st.markdown("#### 📋 Tabel Hasil Prediksi")

        df_pred_tampil = df_pred.copy()
        df_pred_tampil["tanggal"] = pd.to_datetime(df_pred_tampil["tanggal"])
        df_pred_tampil = df_pred_tampil.sort_values("tanggal").reset_index(drop=True)
        df_pred_tampil.insert(0, "Hari ke-", df_pred_tampil.index + 1)
        df_pred_tampil["prediksi"] = df_pred_tampil["prediksi"].round(0).astype(int)
        df_pred_tampil["tanggal"] = df_pred_tampil["tanggal"].dt.strftime("%d-%m-%Y")
        df_pred_tampil = df_pred_tampil.rename(columns={
            "tanggal": "Tanggal",
            "prediksi": "Prediksi Harga (Rp)"
        })

        col1, col2, col3 = st.columns([0.25, 0.5, 0.25])
        with col2:
            st.dataframe(
                df_pred_tampil,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("#### 📈 Grafik Aktual vs Prediksi (Plotly Premium)")

        df_sub_plot = df_sub.copy()
        df_sub_plot["tanggal"] = pd.to_datetime(df_sub_plot["tanggal"])

        df_pred_plot = df_pred.copy()
        df_pred_plot["tanggal"] = pd.to_datetime(df_pred_plot["tanggal"])

        last_actual_date = df_sub_plot["tanggal"].max()

        fig = go.Figure()

        # Garis AKTUAL
        fig.add_trace(
            go.Scatter(
                x=df_sub_plot["tanggal"],
                y=df_sub_plot["harga"],
                mode="lines+markers",
                name="Aktual",
                line=dict(color="#1A73E8", width=3),
                marker=dict(size=4),
                hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Aktual: <b>Rp %{y:,.0f}</b><extra></extra>",
            )
        )

        # Garis PREDIKSI
        fig.add_trace(
            go.Scatter(
                x=df_pred_plot["tanggal"],
                y=df_pred_plot["prediksi"],
                mode="lines+markers",
                name="Prediksi",
                line=dict(color="#E53935", width=3, dash="dash"),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor="rgba(229, 57, 53, 0.15)",
                hovertemplate="<b>%{x|%d-%m-%Y}</b><br>Prediksi: <b>Rp %{y:,.0f}</b><extra></extra>",
            )
        )

        fig.add_shape(
            type="line",
            x0=last_actual_date,
            x1=last_actual_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="gray", width=2, dash="dot"),
        )

        fig.add_annotation(
            x=last_actual_date,
            y=1,
            xref="x",
            yref="paper",
            text="Mulai Prediksi",
            showarrow=False,
            font=dict(size=10, color="gray"),
            yshift=10
        )

        fig.update_layout(
            title={
                "text": f"Grafik Aktual vs Prediksi\n{komoditas} – Pasar {pasar}",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 16},
            },
            xaxis_title="Tanggal",
            yaxis_title="Harga (Rp)",
            template="plotly_white",
            hovermode="x unified",
            paper_bgcolor="rgba(250,250,250,1)",
            plot_bgcolor="rgba(250,250,250,1)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=20, t=70, b=40),
        )

        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.15)")
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.06)")

        st.plotly_chart(fig, use_container_width=True)

        # 7. SARAN KEBIJAKAN (pakai fungsi sama seperti sebelumnya)
        st.markdown("#### 📑 Saran Kebijakan")

        def kebijakan_saran(df_hist, df_pred, horizon_analisis: int = 7) -> str:
            if df_hist is None or df_hist.empty or df_pred is None or df_pred.empty:
                return (
                    "Data historis atau data prediksi belum tersedia sehingga "
                    "belum dapat disusun saran kebijakan yang spesifik."
                )

            df_hist = df_hist.sort_values("tanggal").copy()
            df_pred = df_pred.sort_values("tanggal").copy()

            komod = df_hist["komoditas"].iloc[-1] if "komoditas" in df_hist.columns else "-"
            p = df_hist["pasar"].iloc[-1] if "pasar" in df_hist.columns else "-"

            last_actual_price = float(df_hist["harga"].iloc[-1])

            h = min(horizon_analisis, len(df_pred))
            next_pred = df_pred.iloc[:h].copy()
            mean_pred = float(next_pred["prediksi"].mean())

            if last_actual_price > 0:
                change_pct = (mean_pred - last_actual_price) / last_actual_price * 100
            else:
                change_pct = 0.0

            if len(df_pred) > 1:
                pct_changes = df_pred["prediksi"].pct_change().dropna().abs() * 100
                volatility = float(pct_changes.mean())
            else:
                volatility = 0.0

            if change_pct > 10:
                tren = "naik tajam"
            elif change_pct > 5:
                tren = "cenderung naik"
            elif change_pct < -10:
                tren = "turun tajam"
            elif change_pct < -5:
                tren = "cenderung turun"
            else:
                tren = "relatif stabil"

            if volatility > 8:
                vol_text = "sangat bergejolak"
            elif volatility > 4:
                vol_text = "cukup bergejolak"
            else:
                vol_text = "relatif stabil"

            def fmt_rp(x: float) -> str:
                return f"Rp {x:,.0f}"

            change_dir = "lebih tinggi" if change_pct >= 0 else "lebih rendah"

            teks = []
            teks.append(
                f"**Ringkasan Prediksi Harga {komod} – Pasar {p}**"
            )
            teks.append(
                f"- Harga aktual terakhir : **{fmt_rp(last_actual_price)}**"
            )
            teks.append(
                f"- Rata-rata prediksi {h} hari ke depan : **{fmt_rp(mean_pred)}** "
                f"({abs(change_pct):.1f}% {change_dir} dibanding harga terakhir; tren **{tren}**)"
            )
            teks.append(
                f"- Pola pergerakan prediksi dikategorikan sebagai **{vol_text}** "
                f"(volatilitas sekitar {volatility:.1f}% per hari)."
            )

            teks.append("")
            teks.append("**Implikasi Kebijakan yang Disarankan:**")

            if "naik" in tren:
                teks.append(
                    "- **Penguatan pasokan:** koordinasi dengan pemasok/gapoktan untuk "
                    "meningkatkan pasokan ke pasar, terutama pada hari-hari dengan puncak permintaan."
                )
                teks.append(
                    "- **Pantau potensi spekulasi harga:** lakukan sidak lapangan jika kenaikan dirasa "
                    "tidak wajar untuk mencegah penahanan barang (stockpiling)."
                )
                teks.append(
                    "- **Informasi harga ke masyarakat:** perkuat publikasi harga referensi agar konsumen "
                    "memiliki acuan dan pedagang tidak menaikkan harga berlebihan."
                )
            elif "turun" in tren:
                teks.append(
                    "- **Jaga agar penurunan harga tetap wajar:** pastikan penurunan tidak karena "
                    "kualitas barang yang memburuk atau pasokan yang tidak terserap."
                )
                teks.append(
                    "- **Dukung stabilisasi pendapatan pedagang/petani:** bila penurunan sangat tajam, "
                    "dipertimbangkan intervensi seperti promosi pasar, operasi pembelian, atau kerjasama "
                    "penyaluran ke pasar lain."
                )
            else:
                teks.append(
                    "- **Lanjutkan pola distribusi saat ini:** karena harga relatif stabil, pola distribusi "
                    "dan pasokan yang ada dapat dipertahankan dengan pemantauan rutin."
                )
                teks.append(
                    "- **Fokus pada pemeliharaan kualitas dan kontinuitas pasokan** agar stabilitas harga "
                    "dapat dipertahankan dalam jangka menengah."
                )

            if "bergejolak" in vol_text:
                teks.append(
                    "- **Perlu pemantauan lebih sering:** karena harga bergejolak, disarankan pemantauan harian "
                    "dan koordinasi lintas pasar untuk mengantisipasi lonjakan mendadak."
                )

            return "\n".join(teks)

        saran = kebijakan_saran(df_sub, df_pred)
        st.markdown(saran)
