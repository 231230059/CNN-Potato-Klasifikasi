from datetime import datetime
import io
import json
import math
import os
import altair as alt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import streamlit as st
import tensorflow as tf

# ==========================================
# 1. KONFIGURASI HALAMAN & GAYA VISUAL
# ==========================================
st.set_page_config(
    page_title="JagaDaun • Klasifikasi Penyakit Daun Kentang",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .stApp {
            background-image: linear-gradient(rgba(16, 34, 21, 0.86), rgba(11, 24, 15, 0.92)),
                              url('https://images.unsplash.com/photo-1518977676601-b53f82aba655?q=80&w=1920&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        .hero-box {
            background: linear-gradient(135deg, rgba(20, 83, 45, 0.92) 0%, rgba(21, 128, 61, 0.88) 60%, rgba(5, 46, 22, 0.95) 100%);
            padding: 1.7rem 2rem;
            border-radius: 18px;
            color: #ffffff;
            margin-bottom: 1.3rem;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.14);
        }
        .hero-pill {
            display: inline-block;
            background: rgba(220, 252, 231, 0.18);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.55rem;
            color: #dcfce7;
        }
        .hero-box h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
        }
        .hero-box p {
            margin: 0.45rem 0 0 0;
            font-size: 0.98rem;
            color: #dcfce7;
        }

        .result-banner-healthy {
            background: linear-gradient(135deg, rgba(236, 253, 245, 0.96) 0%, rgba(209, 250, 229, 0.96) 100%);
            border-left: 6px solid #059669;
            border-radius: 14px;
            padding: 1.15rem 1.3rem;
            color: #064e3b;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        }
        .result-banner-warn {
            background: linear-gradient(135deg, rgba(255, 251, 235, 0.96) 0%, rgba(254, 243, 199, 0.96) 100%);
            border-left: 6px solid #d97706;
            border-radius: 14px;
            padding: 1.15rem 1.3rem;
            color: #78350f;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        }
        .result-banner-alert {
            background: linear-gradient(135deg, rgba(254, 242, 242, 0.96) 0%, rgba(254, 226, 226, 0.96) 100%);
            border-left: 6px solid #dc2626;
            border-radius: 14px;
            padding: 1.15rem 1.3rem;
            color: #7f1d1d;
            margin-bottom: 1rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        }

        .info-card {
            background-color: rgba(255, 255, 255, 0.95);
            border-left: 5px solid #2d6a4f;
            padding: 1.15rem 1.3rem;
            border-radius: 12px;
            margin-top: 0.75rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.22);
            color: #1e293b;
        }

        .stButton > button {
            background: linear-gradient(90deg, #1b4332 0%, #2d6a4f 55%, #40916c 100%);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 1.4rem;
            font-weight: 700;
            transition: all 0.25s ease;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(64, 145, 108, 0.45);
        }

        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.02rem;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. PENGATURAN MODEL, PATH FOTO & KAMUS PENYAKIT
# ==========================================
IMG_HEIGHT = 150
IMG_WIDTH = 150
MODEL_PATH = "model_klasifikasi_cnn.h5"  # Ganti ke "model_klasifikasi_cnn.h5" jika memakai file .h5
CLASS_INDICES_PATH = "class_indices.json"
FALLBACK_CLASSES = ["Early_Blight", "Healthy", "Late_Blight"]

# Masukkan path foto contoh kamu di sini (kalau kosong/tidak ditemukan, otomatis pakai ilustrasi bawaan)
LEAF_IMAGE_PATHS = {
    "Healthy": r"D:\potato-disease-leaf-datasetpld2\PLD_3_Classes_256\Testing\Healthy\Healthy_1.jpg",       # Contoh: r"D:\Dataset\Testing\Healthy\Healthy_3.jpg"
    "Early_Blight": r"D:\potato-disease-leaf-datasetpld2\PLD_3_Classes_256\Testing\Early_Blight\Early_Blight_2.jpg",  # Contoh: r"D:\Dataset\Testing\Early_Blight\Early_Blight_3.jpg"
    "Late_Blight": r"D:\potato-disease-leaf-datasetpld2\PLD_3_Classes_256\Testing\Late_Blight\Late_Blight_1.jpg",   # Contoh: r"D:\Dataset\Testing\Late_Blight\Late_Blight_1.jpg"
}

disease_info = {
    "Healthy": {
        "nama": "Daun Sehat (Healthy)",
        "ikon": "✅",
        "tag": "KONDISI OPTIMAL",
        "patogen": "Tidak ada (Normal)",
        "color": "#10b981",
        "banner": "result-banner-healthy",
        "gejala": "Warna daun hijau segar merata, struktur jaringan kokoh, tidak terdapat bercak nekrotik, dan permukaan daun bersih dari infeksi.",
        "pencegahan": [
            "Gunakan bibit kentang yang sehat dan berkualitas.",
            "Terapkan pemupukan berimbang dan jaga drainase lahan agar tidak tergenang.",
            "Bersihkan gulma serta sisa tanaman sakit untuk menekan sumber patogen."
        ],
        "penanganan": [
            "Pertahankan perawatan rutin dan pemantauan berkala tiap minggu.",
            "Catat kondisi tanaman secara rutin untuk deteksi dini perubahan visual."
        ],
    },
    "Early_Blight": {
        "nama": "Bercak Awal (Early Blight)",
        "ikon": "⚠️",
        "tag": "PERLU PERHATIAN",
        "patogen": "Jamur Alternaria solani",
        "color": "#f59e0b",
        "banner": "result-banner-warn",
        "gejala": "Muncul bercak cokelat tua hingga kehitaman dengan pola cincin konsentris menyerupai target, umumnya diawali pada daun bagian bawah/tua.",
        "pencegahan": [
            "Lakukan rotasi tanaman dengan tanaman non-Solanaceae.",
            "Atur jarak tanam agar sirkulasi udara lancar dan kelembapan daun tidak berlebih.",
            "Hindari kelebihan pupuk Nitrogen yang memicu jaringan terlalu sukulen."
        ],
        "penanganan": [
            "Pangkas dan musnahkan daun bergejala agar penyebaran spora berkurang.",
            "Hindari penyiraman dari atas menjelang malam dan gunakan fungisida sesuai anjuran penyuluh setempat."
        ],
    },
    "Late_Blight": {
        "nama": "Busuk Daun / Hawar Daun (Late Blight)",
        "ikon": "🚨",
        "tag": "RISIKO TINGGI",
        "patogen": "Oomycete Phytophthora infestans",
        "color": "#ef4444",
        "banner": "result-banner-alert",
        "gejala": "Bercak basah kehijauan pucat yang cepat berubah menjadi cokelat gelap/kehitaman, meluas agresif saat udara sejuk dan lembap.",
        "pencegahan": [
            "Pilih varietas yang lebih tahan terhadap serangan hawar daun.",
            "Pastikan drainase lahan berjalan baik dan hindari genangan air.",
            "Lakukan pemantauan intensif saat cuaca lembap dan lakukan pencegahan sesuai rekomendasi agronomis."
        ],
        "penanganan": [
            "Segera pisahkan/musnahkan bagian tanaman yang terinfeksi berat agar penyebarannya tidak meluas.",
            "Kurangi kelembapan mikro tajuk dan konsultasikan tindakan pengendalian lanjutan dengan ahli/penyuluh pertanian."
        ],
    },
}


@st.cache_data
def build_leaf_illustration(class_key: str) -> Image.Image:
    w, h = 480, 340
    img = Image.new("RGB", (w, h), (228, 232, 229))

    cx, cy = w // 2, h // 2 + 8
    rx, ry = 105, 138
    leaf_pts = []
    for deg in range(0, 360, 4):
        rad = math.radians(deg)
        taper = 0.82 + 0.18 * math.sin(rad)
        px = cx + rx * taper * math.cos(rad)
        py = cy + ry * math.sin(rad)
        leaf_pts.append((px, py))

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(leaf_pts, fill=255)

    if class_key == "Healthy":
        base_rgb = (68, 152, 62)
    elif class_key == "Early_Blight":
        base_rgb = (94, 148, 56)
    else:
        base_rgb = (78, 136, 52)

    leaf_layer = Image.new("RGB", (w, h), base_rgb)
    draw_leaf = ImageDraw.Draw(leaf_layer)

    draw_leaf.line([(cx, cy + ry - 12), (cx, cy - ry + 18)], fill=(122, 184, 96), width=5)
    for y_off in [-70, -35, 0, 35, 70]:
        span = int(rx * 0.65 * (1 - abs(y_off) / (ry * 1.35)))
        draw_leaf.line([(cx, cy + y_off), (cx - span, cy + y_off - 22)], fill=(112, 174, 88), width=2)
        draw_leaf.line([(cx, cy + y_off), (cx + span, cy + y_off - 22)], fill=(112, 174, 88), width=2)

    if class_key == "Early_Blight":
        spots = [
            (cx - 42, cy - 45, 16), (cx + 38, cy - 25, 14), (cx - 25, cy + 28, 18),
            (cx + 45, cy + 40, 15), (cx - 52, cy + 8, 12), (cx + 22, cy - 68, 13),
            (cx + 12, cy + 68, 14), (cx - 12, cy - 10, 11),
        ]
        for sx, sy, sr in spots:
            draw_leaf.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(92, 52, 24), outline=(58, 32, 14), width=2)
            inner = max(3, sr // 2)
            draw_leaf.ellipse([sx - inner, sy - inner, sx + inner, sy + inner], fill=(128, 76, 36))
    elif class_key == "Late_Blight":
        blight_blobs = [
            (cx - 58, cy - 52, 34), (cx - 32, cy - 88, 30), (cx + 52, cy - 38, 32),
            (cx + 25, cy - 92, 28), (cx - 65, cy + 12, 26), (cx + 60, cy + 22, 28),
        ]
        for bx, by, br in blight_blobs:
            draw_leaf.ellipse([bx - br, by - br, bx + br, by + br], fill=(46, 34, 24))
            halo = int(br * 1.25)
            draw_leaf.ellipse([bx - halo, by - halo, bx + halo, by + halo], outline=(138, 148, 64), width=3)

    shadow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(shadow).polygon([(x + 5, y + 7) for x, y in leaf_pts], fill=80)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    img.paste((150, 156, 152), mask=shadow)

    img.paste(leaf_layer, mask=mask)
    return img


def get_leaf_image(class_key: str) -> Image.Image:
    raw_path = LEAF_IMAGE_PATHS.get(class_key, "").strip().strip('"').strip("'")
    if raw_path and os.path.exists(raw_path):
        try:
            return ImageOps.exif_transpose(Image.open(raw_path)).convert("RGB")
        except Exception:
            pass
    return build_leaf_illustration(class_key)


@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        idx_to_class = {int(v): k for k, v in class_indices.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]
    return FALLBACK_CLASSES


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(image).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def generate_gradcam_overlay(model: tf.keras.Model, image: Image.Image, class_idx: int) -> Image.Image | None:
    try:
        if not model.built:
            model(np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32))

        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break
        if last_conv is None:
            return None

        if isinstance(model, tf.keras.Sequential):
            inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
            x = inputs
            conv_out = None
            for layer in model.layers:
                x = layer(x)
                if layer is last_conv:
                    conv_out = x
            if conv_out is None:
                return None
            grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, x])
        else:
            grad_model = tf.keras.Model(inputs=model.inputs, outputs=[last_conv.output, model.output])

        img_tensor = tf.cast(preprocess_image(image), tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            if isinstance(predictions, list):
                predictions = predictions[0]
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0:
            return None
        heatmap = (heatmap / max_val).numpy()

        orig_rgb = ImageOps.exif_transpose(image).convert("RGB")
        hm_resized = np.array(
            Image.fromarray(np.uint8(255 * heatmap)).resize(orig_rgb.size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        r = np.clip(1.5 - np.abs(4.0 * hm_resized - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * hm_resized - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * hm_resized - 1.0), 0.0, 1.0)
        cmap = np.stack([r, g, b], axis=-1) * 255.0

        overlay = np.clip(0.58 * np.array(orig_rgb, dtype=np.float32) + 0.42 * cmap, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception:
        return None


if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    if os.path.exists("Gemini_Generated_Image_m5cb24m5cb24m5cb.jpg"):
        st.image("Gemini_Generated_Image_m5cb24m5cb24m5cb.jpg", use_container_width=True)

    st.title("🌿 JagaDaun")
    st.caption("Klasifikasi Penyakit Daun Kentang berbasis Convolutional Neural Network (CNN)")
    st.markdown("---")

    st.subheader("📌 Cara Penggunaan")
    st.markdown(
        """
1. Pilih mode **Unggah Gambar** atau **Kamera Langsung**.
2. Klik **Analisis & Klasifikasi Sekarang**.
3. Periksa skor probabilitas, fokus area CNN, gejala, serta saran penanganan.
        """
    )

    st.markdown("---")
    st.subheader("📊 Performa Evaluasi Model")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Test Accuracy", "97.04%")
    m_col2.metric("Test Loss", "0.1029")
    st.progress(0.9704, text="Akurasi Uji: 97.04% (405 Citra)")

    st.markdown("---")
    st.subheader("⚙️ Pengaturan Tampilan")
    show_gradcam = st.toggle("Tampilkan Fokus Area CNN (Grad-CAM)", value=True)
    show_progress_bars = st.toggle("Tampilkan progress bar per kelas", value=True)
    min_conf = st.slider("Batas minimum confidence (%)", 50, 99, 75, 1)

    st.markdown("---")
    if st.button("🗑️ Hapus Riwayat Prediksi"):
        st.session_state.history = []
        st.rerun()

# ==========================================
# 4. HERO HEADER
# ==========================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-pill">DEEP LEARNING • CONVOLUTIONAL NEURAL NETWORK</div>
        <h1>🌿 Sistem Deteksi & Klasifikasi Penyakit Daun Kentang</h1>
        <p>Identifikasi kondisi <b>Healthy</b>, <b>Early Blight</b>, dan <b>Late Blight</b> secara cepat dengan visualisasi probabilitas, fokus area model, serta panduan pencegahan & penanganan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_cnn_model()
    class_names = load_class_names()
    model_loaded = True
except Exception as e:
    model_loaded = False
    class_names = FALLBACK_CLASSES
    st.error(f"⚠️ Gagal memuat model (`{MODEL_PATH}`): {e}")

# ==========================================
# 5. TAB UTAMA
# ==========================================
tab1, tab2 = st.tabs(["🔍 Klasifikasi Citra Daun", "📚 Ensiklopedia & Panduan Pencegahan"])

with tab1:
    col_kiri, col_kanan = st.columns([1, 1.35], gap="large")

    selected_image = None
    source_label = ""

    with col_kiri:
        st.subheader("📤 1. Input Gambar Daun")
        mode_input = st.radio(
            "Pilih sumber gambar:",
            ["Unggah File", "Kamera Langsung"],
            horizontal=True,
        )

        if mode_input == "Unggah File":
            uploaded_file = st.file_uploader(
                "Pilih atau seret gambar daun kentang (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file is not None:
                selected_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
                source_label = f"Upload: {uploaded_file.name}"
        else:
            cam_file = st.camera_input("Ambil foto daun kentang")
            if cam_file is not None:
                selected_image = ImageOps.exif_transpose(Image.open(cam_file)).convert("RGB")
                source_label = "Kamera Langsung"

        if selected_image is not None:
            st.image(selected_image, caption=f"Pratinjau ({source_label})", use_container_width=True)
            tombol_prediksi = st.button("🚀 Analisis & Klasifikasi Sekarang")
        else:
            st.info("Silakan unggah atau ambil foto daun kentang terlebih dahulu.")
            tombol_prediksi = False

    with col_kanan:
        st.subheader("📋 2. Hasil Analisis & Rekomendasi")

        if selected_image is not None and tombol_prediksi:
            if not model_loaded:
                st.error("Model CNN belum siap digunakan.")
            else:
                with st.spinner("Menganalisis pola tekstur, warna, dan probabilitas kelas..."):
                    tensor = preprocess_image(selected_image)
                    probs = model.predict(tensor, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx]) * 100.0
                    result_class = class_names[pred_idx]
                    info = disease_info.get(
                        result_class,
                        {
                            "nama": result_class,
                            "ikon": "🔍",
                            "tag": "TERDETEKSI",
                            "patogen": "Tidak spesifik",
                            "color": "#3b82f6",
                            "banner": "result-banner-healthy",
                            "gejala": "Klasifikasi berdasarkan pola fitur visual pada citra daun.",
                            "pencegahan": ["Lakukan pemantauan rutin pada tanaman."],
                            "penanganan": ["Verifikasi kondisi di lapangan."],
                        },
                    )

                st.markdown(
                    f"""
                    <div class="{info['banner']}">
                        <div style="font-size:0.75rem; font-weight:800; letter-spacing:0.05em; opacity:0.85;">{info['tag']}</div>
                        <div style="font-size:1.45rem; font-weight:800; margin:0.15rem 0;">{info['ikon']} Hasil Klasifikasi: {info['nama']}</div>
                        <div style="font-size:0.95rem;">Tingkat keyakinan model: <b>{confidence:.2f}%</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if confidence < min_conf:
                    st.warning(
                        f"Confidence ({confidence:.2f}%) di bawah ambang {min_conf}%. "
                        "Coba gunakan pencahayaan merata dan fokus gambar yang tajam."
                    )

                m1, m2 = st.columns(2)
                m1.metric("Tingkat Keyakinan (Confidence)", f"{confidence:.2f}%")
                m2.metric("Penyebab / Patogen", info["patogen"])

                st.markdown("#### 📊 Distribusi Probabilitas Kelas")
                df_probs = pd.DataFrame({
                    "Kelas": class_names,
                    "Probabilitas (%)": [round(float(p) * 100.0, 2) for p in probs],
                    "Warna": [disease_info.get(c, {}).get("color", "#3b82f6") for c in class_names],
                }).sort_values("Probabilitas (%)", ascending=False)

                chart = (
                    alt.Chart(df_probs)
                    .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, height=32)
                    .encode(
                        x=alt.X("Probabilitas (%):Q", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("Kelas:N", sort="-x", title=None),
                        color=alt.Color("Warna:N", scale=None, legend=None),
                        tooltip=[alt.Tooltip("Kelas:N"), alt.Tooltip("Probabilitas (%):Q", format=".2f")],
                    )
                    .properties(height=155)
                )
                st.altair_chart(chart, use_container_width=True)

                if show_progress_bars:
                    for idx, name in enumerate(class_names):
                        prob = float(probs[idx])
                        st.write(f"**{name}** ({prob * 100:.2f}%)")
                        st.progress(min(max(prob, 0.0), 1.0))

                if show_gradcam:
                    hm_img = generate_gradcam_overlay(model, selected_image, pred_idx)
                    if hm_img is not None:
                        st.markdown("#### 🔥 Fokus Area Model (Grad-CAM)")
                        st.image(
                            hm_img,
                            caption="Area berwarna hangat menunjukkan bagian daun yang paling memengaruhi prediksi.",
                            use_container_width=True,
                        )

                st.markdown(
                    f"""
                    <div class="info-card">
                        <strong>📖 Karakteristik & Gejala:</strong><br>
                        {info['gejala']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### 🛠️ Rekomendasi Penanganan")
                for i, step in enumerate(info["penanganan"], 1):
                    st.markdown(f"**{i}.** {step}")

                st.session_state.history.insert(
                    0,
                    {
                        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Sumber": source_label,
                        "Prediksi": result_class,
                        "Confidence (%)": round(confidence, 2),
                    },
                )

        elif selected_image is not None and not tombol_prediksi:
            st.info("👈 Gambar sudah siap! Klik tombol **'Analisis & Klasifikasi Sekarang'** untuk melihat hasilnya.")
        else:
            st.markdown(
                """
                <div class="info-card">
                    <strong>Menunggu Input Gambar...</strong><br>
                    Hasil identifikasi kelas penyakit, skor probabilitas, fokus area model, gejala, serta panduan penanganan akan muncul di panel ini setelah dianalisis.
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab2:
    st.subheader("📚 Ensiklopedia Lengkap & Strategi Pencegahan Penyakit Kentang")
    st.markdown(
        "Halaman ini menyajikan informasi mendalam mengenai karakteristik patogen, gejala klinis, **cara pencegahan dini**, "
        "serta langkah penanganan untuk menjaga produktivitas tanaman kentang."
    )
    st.markdown("---")

    col_e1, col_e2, col_e3 = st.columns(3, gap="medium")

    with col_e1:
        st.success("### ✅ Daun Sehat (Healthy)")
        st.image(get_leaf_image("Healthy"), use_container_width=True, caption="Contoh Daun Sehat")

        st.markdown(f"**Patogen:** {disease_info['Healthy']['patogen']}")
        st.write(f"**Gejala:** {disease_info['Healthy']['gejala']}")
        st.markdown("---")
        st.markdown("🛡️ **Cara Pencegahan:**")
        for p in disease_info['Healthy']['pencegahan']:
            st.markdown(f"- {p}")

    with col_e2:
        st.warning("### ⚠️ Early Blight")
        st.image(get_leaf_image("Early_Blight"), use_container_width=True, caption="Contoh Gejala Early Blight")

        st.markdown(f"**Patogen:** {disease_info['Early_Blight']['patogen']}")
        st.write(f"**Gejala:** {disease_info['Early_Blight']['gejala']}")
        st.markdown("---")
        st.markdown("🛡️ **Cara Pencegahan:**")
        for p in disease_info['Early_Blight']['pencegahan']:
            st.markdown(f"- {p}")

    with col_e3:
        st.error("### 🚨 Late Blight")
        st.image(get_leaf_image("Late_Blight"), use_container_width=True, caption="Contoh Gejala Late Blight")

        st.markdown(f"**Patogen:** {disease_info['Late_Blight']['patogen']}")
        st.write(f"**Gejala:** {disease_info['Late_Blight']['gejala']}")
        st.markdown("---")
        st.markdown("🛡️ **Cara Pencegahan:**")
        for p in disease_info['Late_Blight']['pencegahan']:
            st.markdown(f"- {p}")

# ==========================================
# 6. RIWAYAT PREDIKSI SESI INI
# ==========================================
st.divider()
h_left, h_right = st.columns([3, 1])
with h_left:
    st.subheader("🕒 Riwayat Prediksi Sesi Ini")
with h_right:
    if st.session_state.history:
        csv_buf = io.StringIO()
        pd.DataFrame(st.session_state.history).to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Unduh Riwayat CSV",
            data=csv_buf.getvalue(),
            file_name="riwayat_prediksi_jagadaun.csv",
            mime="text/csv",
        )

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
else:
    st.caption("Belum ada riwayat prediksi pada sesi ini.")