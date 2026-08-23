"""
Cartoonizer — Streamlit front-end for CartoonGAN.
"""

import io
import os
import glob
import subprocess

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Cartoonizer — CartoonGAN", page_icon="🎨", layout="wide")

PRETRAINED_REPO = "CartoonGan-tensorflow"

# Auto-clone the author's pretrained repository if it's not already downloaded
if not os.path.exists(PRETRAINED_REPO):
    with st.spinner("Downloading pretrained author styles (Shinkai, Paprika, Hayao, Hosoda)..."):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/mnicnc404/CartoonGan-tensorflow.git"],
            check=False
        )

MODEL_PATHS = {
    "My trained CartoonGAN": "exported_models/my_cartoongan_SavedModel",
    "Pretrained · Makoto Shinkai (Your Name)": f"{PRETRAINED_REPO}/exported_models/light_shinkai_SavedModel",
    "Pretrained · Satoshi Kon (Paprika)": f"{PRETRAINED_REPO}/exported_models/light_paprika_SavedModel",
    "Pretrained · Hayao Miyazaki (Ghibli)": f"{PRETRAINED_REPO}/exported_models/light_hayao_SavedModel",
    "Pretrained · Mamoru Hosoda (Mirai)": f"{PRETRAINED_REPO}/exported_models/light_hosoda_SavedModel",
}

# ----------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    if path.endswith((".keras", ".h5")):
        return tf.keras.models.load_model(path, compile=False)
    return tf.saved_model.load(path)

def run_model(model, batch: np.ndarray) -> np.ndarray:
    # 1. Keras Model instance
    if isinstance(model, tf.keras.Model):
        out = model(tf.constant(batch), training=False)
    # 2. SavedModel with signatures (handles both custom & pretrained models)
    elif hasattr(model, "signatures") and len(model.signatures) > 0:
        sig_name = list(model.signatures.keys())[0]
        infer = model.signatures[sig_name]
        out = infer(tf.constant(batch))
    # 3. Direct callable fallback
    else:
        out = model(tf.constant(batch))

    if isinstance(out, dict):
        out = list(out.values())[0]
    return np.array(out)

def cartoonize(pil_img: Image.Image, model, max_dim: int = 512) -> Image.Image:
    img = pil_img.convert("RGB")
    w, h = img.size

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.LANCZOS)

    pad_w, pad_h = (4 - w % 4) % 4, (4 - h % 4) % 4
    arr = np.asarray(img, np.float32) / 127.5 - 1.0
    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")[None, ...]

    out = run_model(model, arr)[0][:h, :w]
    return Image.fromarray(np.clip((out + 1) * 127.5, 0, 255).astype(np.uint8))

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Settings")

available = {name: path for name, path in MODEL_PATHS.items() if os.path.exists(path)}
custom = st.sidebar.text_input("…or a path to your own SavedModel / .keras", "")
if custom:
    available["Custom"] = custom

if not available:
    st.error("No model found. Please ensure the model files exist.")
    st.stop()

choice = st.sidebar.selectbox("Select Style / Model", list(available.keys()))
max_dim = st.sidebar.slider("Max output size (px)", 256, 1024, 512, 64)
st.sidebar.caption("Larger sizes look sharper but take slightly longer to process on CPU.")

# ----------------------------------------------------------------------------- main
st.title("🎨 Cartoonize your photos")
st.caption("CartoonGAN (Chen et al., CVPR 2018) — Domain translation from real photos to anime art.")

upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

samples = sorted(glob.glob("sample_images/*"))
if not upload and samples:
    pick = st.selectbox("…or try a sample", ["—"] + [os.path.basename(p) for p in samples])
    if pick != "—":
        upload = os.path.join("sample_images", pick)

if upload:
    source = Image.open(upload)
    model = load_model(available[choice])

    with st.spinner(f"Applying {choice}…"):
        result = cartoonize(source, model, max_dim)

    left, right = st.columns(2)
    left.subheader("Original Photo")
    left.image(source, use_container_width=True)
    right.subheader(f"Cartoonized ({choice})")
    right.image(result, use_container_width=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button(
        "Download cartoon", buf.getvalue(), "cartoon.png", "image/png", use_container_width=True
    )
else:
    st.info(
        "Upload a photo to get started. You can toggle between your custom trained model "
        "and the official pretrained styles in the sidebar."
    )
