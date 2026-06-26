# 🌿 Crop Disease Detection System

A modern, multilingual web application for detecting diseases in crops including Tomato, Potato, Pepper (Bell), and Rice. The app uses a Flask backend with TensorFlow/Keras CNN models for image classification and returns treatment suggestions in English, Hindi, and Telugu.

<img width="1918" height="1000" alt="image" src="https://github.com/user-attachments/assets/24227079-9b8f-44b5-a11a-022e9e53370d" />
<img width="1918" height="983" alt="image" src="https://github.com/user-attachments/assets/3ebb2ae3-bf19-4856-a1d6-07b73474eea7" />
<img width="1916" height="1011" alt="image" src="https://github.com/user-attachments/assets/ff295818-c70d-4516-8b9b-fc254d09b86d" />

## Features

- Detects disease classes for tomato, potato, pepper, and rice
- Uses local TensorFlow/Keras models for inference
- Returns top-3 predictions with confidence scores
- Provides multilingual treatment guidance in English, Hindi, and Telugu
- Includes a simple web UI for image upload and prediction

## Tech stack

- Backend: Python, Flask, Flask-CORS
- Deep learning: TensorFlow/Keras
- Image processing: Pillow, NumPy
- Translation: deep-translator
- Deployment: Vercel-ready with Python entrypoint support

## Project structure

```text
crop-disease-detection/
├── app.py                     # Root Flask entrypoint for Vercel
├── pyproject.toml             # Vercel Python entrypoint configuration
├── requirements.txt           # Root Python dependencies
├── vercel.json                # Vercel routing configuration
├── api/                       # Vercel serverless Python entrypoints
├── crop_disease_backend/      # Import wrapper for the backend app
├── crop-disease-backend/      # Main Flask app and model assets
│   ├── app.py
│   ├── templates/
│   ├── crop_disease_cnn_model.keras
│   └── rice_disease_cnn_model.keras
└── README.md
```

## Local run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000/

## Vercel deployment

1. Push this repository to GitHub.
2. Create a new Vercel project and import the repository.
3. Use these settings:
   - Framework: Other
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
4. Deploy.

> The app uses TensorFlow model files from the backend folder, so the first prediction may take a little longer while the model loads.

## Notes

- The app exposes /health, /predict, and /predict_rice endpoints.
- The default UI is served from the Flask app.

## Usage

1. Choose the crop category (General Vegetables or Rice).
2. Drag and drop an image of the affected leaf or click to select a file.
3. Click Get Disease & Treatment.
4. View the diagnosis and treatment steps in English, Hindi, or Telugu.

## Support

Give a ⭐️ if this project helped you!
