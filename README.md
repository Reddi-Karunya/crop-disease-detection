# 🌿 Crop Disease Detection System

A modern, multilingual web application for detecting diseases in crops including **Tomato, Potato, Pepper (Bell), and Rice**. This project uses deep learning (CNN) for image classification and state-of-the-art NLP models (Transformers) for multilingual treatment suggestions.

![Project Screenshot](screenshot.png)
*(Replace screenshot.png with an actual screenshot of the app in action!)*

---

## 🚀 Features

- **Wide Crop Coverage**: Detects diseases across 15+ categories for Tomato, Potato, and Pepper.
- **Specialized Rice Model**: Dedicated analysis for Bacterial Leaf Blight, Brown Spot, and Leaf Smut.
- **Multilingual Support**: Real-time translation of treatment advice into **English, Hindi (हिंदी), and Telugu (తెలుగు)**.
- **Modern UI**: Clean, responsive dashboard with drag-and-drop upload and real-time progress tracking.
- **Quick Results**: Optimized model loading for fast predictions.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Flask-CORS
- **Deep Learning**: TensorFlow/Keras (CNN)
- **NLP/Translation**: Hugging Face Transformers (Helsinki-NLP/Meher2006)
- **Frontend**: HTML5, CSS3, JavaScript (Fetch API)
- **Deployment**: Render-ready with Gunicorn and `render.yaml`

---

## 📁 Project Structure

```text
crop-disease-detection/
├── crop-disease-backend/
│   ├── app.py                # Main Flask server
│   ├── requirements.txt      # Python dependencies
│   ├── templates/
│   │   └── index.html        # Modern Frontend UI
│   ├── crop_disease_cnn_model.keras  # Tomato/Potato/Pepper Model
│   └── rice_disease_cnn_model.keras  # Specialized Rice Model
├── notebooks/                # Training scripts and data exploration
├── render.yaml               # Auto-deployment configuration
└── README.md                 # Project Documentation
```

---

## 💻 Local Setup & Running

Follow these steps to run the project on your machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Reddi-Karunya/crop-disease-detection.git
   cd crop-disease-detection
   ```

2. **Navigate to the backend**:
   ```bash
   cd crop-disease-backend
   ```

3. **Set up a Virtual Environment (Recommended)**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows
   source venv/bin/activate      # Linux/macOS
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the server**:
   ```bash
   python app.py
   ```

6. **Access the App**:
   Open your browser and visit: `http://127.0.0.1:5000/`

---

## 📝 Usage

1. Choose the crop category (General Vegetables or Rice).
2. Drag and drop an image of the affected leaf or click to select a file.
3. Click **"Get Disease & Treatment"**.
4. View the diagnosis and treatment steps in English, Hindi, or Telugu.

---

## ✨ Support

Give a ⭐️ if this project helped you!
