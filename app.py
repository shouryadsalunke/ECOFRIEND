{\rtf1\ansi\ansicpg1252\cocoartf2869
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from flask import Flask, request, jsonify\
import tensorflow as tf\
import numpy as np\
from PIL import Image\
\
app = Flask(__name__)\
\
# Load the trained model\
model = tf.keras.models.load_model("models/best_scratch_model.keras")\
\
# Class names (same order as training)\
CLASSES = [\
    "cardboard",\
    "glass",\
    "metal",\
    "paper",\
    "plastic",\
    "trash"\
]\
\
# Disposal instructions\
DISPOSAL_GUIDE = \{\
    "cardboard": "Flatten and place in the cardboard recycling bin.",\
    "glass": "Dispose of in the glass recycling bin.",\
    "metal": "Recycle in the metal recycling bin.",\
    "paper": "Recycle with paper waste.",\
    "plastic": "Dispose of in the plastic recycling bin.",\
    "trash": "Dispose of in the general waste bin."\
\}\
\
\
@app.route("/")\
def home():\
    return "EcoFriend API is running!"\
\
\
@app.route("/predict", methods=["POST"])\
def predict():\
\
    if "image" not in request.files:\
        return jsonify(\{"error": "No image uploaded"\}), 400\
\
    file = request.files["image"]\
\
    image = Image.open(file).convert("RGB")\
\
    # Same image size used during training\
    image = image.resize((128, 128))\
\
    image = np.array(image).astype(np.float32)\
\
    image = np.expand_dims(image, axis=0)\
\
    prediction = model.predict(image, verbose=0)\
\
    predicted_index = np.argmax(prediction)\
\
    predicted_class = CLASSES[predicted_index]\
\
    confidence = float(np.max(prediction))\
\
    return jsonify(\{\
        "prediction": predicted_class,\
        "confidence": round(confidence * 100, 2),\
        "disposal": DISPOSAL_GUIDE[predicted_class]\
    \})\
\
\
if __name__ == "__main__":\
    app.run(debug=True)}