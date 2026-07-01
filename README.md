# ♻️ EcoFriend

EcoFriend is an AI-powered smart waste segregation system that classifies household waste into six categories using a custom Convolutional Neural Network (CNN). Users can upload an image of waste to receive an instant prediction, confidence score, and appropriate recycling or disposal guidance. The project also compares the performance of the original and TensorFlow Lite quantized models to demonstrate efficient edge deployment.

---

## Features

- 🧠 Custom CNN trained from scratch using TensorFlow/Keras
- 📷 Real-time image-based waste classification
- ♻️ Disposal and recycling recommendations
- ⚡ TensorFlow Lite (TFLite) quantized model for optimized inference
- 📊 Model analytics comparing the original and quantized models
- 🌐 Flask backend with a Streamlit frontend

---

## Waste Categories

The model classifies waste into the following categories:

- 📦 Cardboard
- 🥤 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model Optimization | TensorFlow Lite |
| Backend | Flask |
| Frontend | Streamlit |
| Data Processing | NumPy, Pillow |
| Visualization | Matplotlib, Plotly |
| Evaluation | Scikit-learn |

---

## Project Structure

```text
EcoFriend/
│
├── app.py                      # Flask backend
├── streamlit_app.py            # Streamlit frontend
├── requirements.txt
├── README.md
│
├── models/
│   ├── best_scratch_model.keras
│   └── quantized_model.tflite
│
├── app/
│   ├── metrics.json
│   └── evaluation_dashboard.png
│
└── uploads/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/EcoFriend.git
cd EcoFriend
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask backend:

```bash
python app.py
```

In a new terminal, launch the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser.

---

## Model

- **Architecture:** Custom Convolutional Neural Network (CNN)
- **Input Size:** 128 × 128 RGB Images
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Training Techniques:**
  - Data Augmentation
  - Class Weight Balancing
  - Early Stopping
  - Learning Rate Scheduling

---

## Performance

The repository includes both the original TensorFlow model and a TensorFlow Lite quantized model. Performance metrics such as accuracy, model size, and inference latency are available in `metrics.json` and are displayed within the application's analytics page.

---

## Future Improvements

- 📸 Webcam-based real-time waste detection
- ☁️ Cloud deployment (Render/Hugging Face Spaces)
- 📱 Mobile-friendly interface
- ♻️ Support for additional waste categories
- 📍 Nearby recycling center recommendations

---

## Author

**Shourya Salunke**

---

## License

This project is intended for educational purposes
