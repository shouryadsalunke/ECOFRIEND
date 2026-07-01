import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Config ──
IMG_SIZE = 128
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

DISPOSAL_GUIDANCE = {
    'cardboard': "♻️ Recyclable. Flatten boxes and keep dry before placing in the recycling bin.",
    'glass':     "♻️ Recyclable. Rinse out any residue. Check local rules for color-sorted glass.",
    'metal':     "♻️ Recyclable. Rinse cans/tins. Aluminum and steel are both widely accepted.",
    'paper':     "♻️ Recyclable. Keep clean and dry — greasy or food-stained paper should go to trash.",
    'plastic':   "♻️ Recyclable (check resin code). Rinse containers and remove caps if required locally.",
    'trash':     "🚮 Not recyclable. Dispose of in general waste.",
}

RAW_MODEL_PATH = "best_scratch_model.keras"
QUANT_MODEL_PATH = "quantized_model.tflite"
METRICS_PATH = "metrics.json"
DASHBOARD_IMAGE_PATH = "evaluation_dashboard.png"

st.set_page_config(page_title="EcoFriend", page_icon="♻️", layout="centered")


# ── Cached model loaders ──
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model(RAW_MODEL_PATH)


@st.cache_resource
def load_tflite_interpreter():
    interpreter = tf.lite.Interpreter(model_path=QUANT_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize + format image to match model's expected input.
    Model has a Rescaling(1./255) layer built in, so we feed raw 0-255 floats."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)


def predict_keras(model, img_array):
    preds = model.predict(img_array, verbose=0)[0]
    return preds


def predict_tflite(interpreter, img_array):
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    interpreter.set_tensor(in_det[0]["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(out_det[0]["index"])[0]


# ── UI ──
st.title("♻️ EcoFriend")
st.caption("AI-powered smart waste segregation — upload an image to classify it into one of six categories.")

tab_classify, tab_analytics = st.tabs(["🔍 Classify Waste", "📊 Model Analytics"])

with tab_classify:
    uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

    model_choice = st.radio(
        "Model to use for prediction",
        ["Original model (higher accuracy)", "Quantized model (faster, smaller)"],
        horizontal=True,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        img_array = preprocess_image(image)

        with st.spinner("Classifying..."):
            if model_choice.startswith("Original"):
                model = load_keras_model()
                preds = predict_keras(model, img_array)
            else:
                interpreter = load_tflite_interpreter()
                preds = predict_tflite(interpreter, img_array)

        top_idx = int(np.argmax(preds))
        top_class = CLASSES[top_idx]
        confidence = float(preds[top_idx]) * 100

        st.subheader(f"Prediction: **{top_class.capitalize()}**")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.info(DISPOSAL_GUIDANCE[top_class])

        st.write("Confidence by category:")
        st.bar_chart({cls: float(p) for cls, p in zip(CLASSES, preds)})

with tab_analytics:
    st.subheader("Original vs. Quantized Model")

    metrics = load_metrics()
    if metrics:
        raw = metrics["raw_model"]
        quant = metrics["quantized_model"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Model**")
            st.write(f"Test accuracy: {raw['test_accuracy']*100:.2f}%")
            st.write(f"File size: {raw['file_size_mb']} MB")
            st.write(f"Avg latency: {raw['average_latency_ms']} ms")
            st.write(f"Parameters: {raw['trainable_parameters']:,}")
        with col2:
            st.markdown("**Quantized Model**")
            st.write(f"Test accuracy: {quant['test_accuracy']*100:.2f}%")
            st.write(f"File size: {quant['file_size_mb']} MB")
            st.write(f"Avg latency: {quant['average_latency_ms']} ms")
            st.write(f"Parameters: {quant['trainable_parameters']:,}")
    else:
        st.warning("metrics.json not found — analytics unavailable.")

    st.subheader("Evaluation Dashboard")
    try:
        st.image(DASHBOARD_IMAGE_PATH, use_container_width=True)
    except Exception:
        st.warning("evaluation_dashboard.png not found.")
