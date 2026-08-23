"""
Cartoonizer — Streamlit front-end for CartoonGAN.

Run:      streamlit run app.py
Expects:  exported_models/my_cartoongan_SavedModel/  (produced by Section 13 of the notebook)
          and/or a clone of https://github.com/mnicnc404/CartoonGan-tensorflow for pretrained styles.
"""

import io
import os
import glob

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Cartoonizer — CartoonGAN", page_icon="🎨", layout="wide")

MODEL_PATHS = {
    "My trained CartoonGAN": "exported_models/my_cartoongan_SavedModel",
    "Pretrained · Shinkai": "CartoonGan-tensorflow/exported_models/light_shinkai_SavedModel",
    "Pretrained · Paprika": "CartoonGan-tensorflow/exported_models/light_paprika_SavedModel",
}


# ----------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    """Loads once per session. Removing the cache decorator makes every click reload TF."""
    if path.endswith((".keras", ".h5")):
        return tf.keras.models.load_model(path, compile=False)
    return tf.saved_model.load(path)


def run_model(model, batch: np.ndarray) -> np.ndarray:
    out = (
        model(tf.constant(batch), training=False)
        if isinstance(model, tf.keras.Model)
        else model(tf.constant(batch))
    )
    if isinstance(out, dict):
        out = list(out.values())[0]
    return np.array(out)


def cartoonize(pil_img: Image.Image, model, max_dim: int = 512) -> Image.Image:
    """Scale down, pad to a multiple of 4, run the generator, crop back."""
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
    st.error(
        "No model found. Export one from the notebook into `exported_models/`, or clone "
        "https://github.com/mnicnc404/CartoonGan-tensorflow next to this file for pretrained styles."
    )
    st.stop()

choice = st.sidebar.selectbox("Model", list(available))
max_dim = st.sidebar.slider("Max output size (px)", 256, 1024, 512, 64)
st.sidebar.caption("Larger sizes look better but take longer on CPU.")

# ----------------------------------------------------------------------------- main
st.title("🎨 Cartoonize your photos")
st.caption("CartoonGAN (Chen et al., CVPR 2018) — photo → cartoon, trained on unpaired data.")

upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

samples = sorted(glob.glob("sample_images/*"))
if not upload and samples:
    pick = st.selectbox("…or try a sample", ["—"] + [os.path.basename(p) for p in samples])
    if pick != "—":
        upload = os.path.join("sample_images", pick)

if upload:
    source = Image.open(upload)
    model = load_model(available[choice])

    with st.spinner("Cartoonizing…"):
        result = cartoonize(source, model, max_dim)

    left, right = st.columns(2)
    left.subheader("Original")
    left.image(source, use_container_width=True)
    right.subheader("Cartoonized")
    right.image(result, use_container_width=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button(
        "Download cartoon", buf.getvalue(), "cartoon.png", "image/png", use_container_width=True
    )
else:
    st.info(
        "Upload a photo to get started. Landscapes and portraits both work; "
        "results follow whatever style the model was trained on."
    )
