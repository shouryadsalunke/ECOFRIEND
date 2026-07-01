# EcoFriend
smart waste segregation system that classifies household waste into six categories using a custom-built Convolutional Neural Network. The application provides real-time predictions, recycling guidance, and a model analytics page comparing the performance of the original and quantized models


# ♻️ EcoFriend CNN – Smart Waste Segregator

EcoFriend CNN is an AI-powered waste classification system that uses a custom Convolutional Neural Network (CNN) to classify household waste into six categories: Cardboard, Glass, Metal, Paper, Plastic, and Trash. The application provides real-time predictions, disposal guidance, and compares the performance of the original and quantized models.

---

## Features

- Image-based waste classification
- Custom CNN built from scratch using TensorFlow/Keras
- Post-Training Quantization (PTQ) using TensorFlow Lite
- Flask backend for inference
- Streamlit frontend for an intuitive user interface
- Model performance comparison (Accuracy, Latency, Model Size)
- Recycling and disposal recommendations

---

## Dataset

- **Dataset:** TrashNet (Kaggle/GitHub)
- **Classes:**
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash

---

## Model

- Custom CNN Architecture
- Image Size: **128 × 128**
- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**
- Data Augmentation
- Class Weight Balancing
- Early Stopping
- Learning Rate Scheduling

---

## Tech Stack

- Python
- TensorFlow / Keras
- TensorFlow Lite
- Flask
- Streamlit
- NumPy
- Scikit-learn
- Matplotlib

---

## Project Structure

```text
EcoFriend-CNN/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
│
├── models/
│   ├── best_scratch_model.keras
│   └── quantized_model.tflite
│
├── app/
│   ├── metrics.json
│   └── evaluation_dashboard.png
│
└── README.md
```

---

## Installation

```bash
git clone <repository-url>

cd EcoFriend-CNN

pip install -r requirements.txt
```

---

## Run the Application

### Start the Flask Backend

```bash
python app.py
```

### Start the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

---

## Model Performance

| Metric | Raw Model | Quantized Model |
|---------|-----------|-----------------|
| Accuracy | XX.XX% | XX.XX% |
| Model Size | XX MB | XX MB |
| Latency | XX ms | XX ms |

> Replace the values with the ones from `metrics.json`.

---

## Future Improvements

- Webcam support
- Cloud deployment (Render/Hugging Face Spaces)
- Additional waste categories
- Mobile-friendly interface

---

## Author

**Shourya Salunke**
