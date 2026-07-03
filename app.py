import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Config ──
MODEL_PATH = "models/quantized_model.tflite"
IMG_SIZE = 128
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Emoji for a bit of visual flair per class
CLASS_EMOJI = {
    'cardboard': '📦',
    'glass': '🍾',
    'metal': '🥫',
    'paper': '📄',
    'plastic': '🧴',
    'trash': '🗑️'
}

# ── Load model (cached so it only loads once per session) ──
@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Preprocessing (must match training exactly) ──
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32)
    # NOTE: no manual /255 here — the model's own Rescaling(1./255) layer
    # was baked in during training, so the interpreter expects raw 0-255 floats.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ── Run inference ──
def predict(image: Image.Image):
    img_array = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

# ── UI ──
st.set_page_config(page_title="EcoFriend CNN — Trash Classifier", page_icon="♻️", layout="centered")

st.title("♻️ EcoFriend CNN")
st.caption("A lightweight, quantized CNN that sorts trash into 6 categories — cardboard, glass, metal, paper, plastic, and trash.")

st.divider()

uploaded_file = st.file_uploader("Upload an image of a trash item", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        preds = predict(image)

    pred_idx = int(np.argmax(preds))
    pred_class = CLASSES[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    with col2:
        st.subheader(f"{CLASS_EMOJI[pred_class]} {pred_class.capitalize()}")
        st.metric("Confidence", f"{confidence:.1f}%")

        st.write("**All class probabilities:**")
        for cls, prob in sorted(zip(CLASSES, preds), key=lambda x: -x[1]):
            st.progress(float(prob), text=f"{cls.capitalize()}: {prob*100:.1f}%")

else:
    st.info("👆 Upload an image to get a prediction.")

st.divider()
st.caption("Model: EcoCNN (VGG-style backbone + SE attention blocks), quantized to INT8 — 730 KB, ~97% test accuracy.")
