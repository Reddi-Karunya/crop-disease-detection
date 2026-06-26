import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "crop-disease-backend"
MODEL_PATH = BACKEND_DIR / "crop_disease_cnn_model.keras"
RICE_MODEL_PATH = BACKEND_DIR / "rice_disease_cnn_model.keras"

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
]

treatment_solutions = {
    "Pepper__bell___Bacterial_spot": "Spray with copper-based fungicide. Use 2g per liter of water. Avoid spraying in peak sun hours. Repeat every 7 days until controlled.",
    "Pepper__bell___healthy": "The plant is healthy. Maintain regular watering and monitor for pests.",
    "Potato___Early_blight": "Apply fungicide weekly until disease subsides. Remove infected leaves immediately and maintain good air circulation.",
    "Potato___healthy": "The plant is healthy. Continue standard care and ensure proper nutrition.",
    "Potato___Late_blight": "Use a fungicide containing chlorothalonil or metalaxyl. Apply at first sign of disease and repeat every 5-7 days.",
    "Tomato___Target_Spot": "Treat with mancozeb or copper fungicide. Remove affected leaves and improve field hygiene.",
    "Tomato___Tomato_mosaic_virus": "No cure for viral infections. Remove infected plants to prevent spread. Practice crop rotation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly vectors using insecticides. Remove infected plants promptly.",
    "Tomato_Bacterial_spot": "Spray copper-based bactericides early in the season. Avoid working in wet plants to reduce spread.",
    "Tomato_Early_blight": "Apply fungicides like chlorothalonil or copper compounds. Prune lower leaves to improve air flow.",
    "Tomato_healthy": "Plant is healthy. Maintain regular care and watch for early signs of disease.",
    "Tomato_Late_blight": "Use fungicides containing chlorothalonil and protect from wet conditions. Remove infected debris.",
    "Tomato_Leaf_Mold": "Apply fungicides and ensure plants are spaced properly for airflow. Avoid overhead watering.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves and apply fungicides regularly. Rotate crops to reduce disease buildup.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides or insecticidal soap. Regularly spray plants with water to reduce mite populations.",
}

rice_class_names = [
    "Bacterial leaf blight",
    "Brown spot",
    "Leaf smut",
]

rice_treatment_solutions = {
    "Bacterial leaf blight": "Use resistant rice varieties and apply recommended bactericides. Avoid standing water and improve drainage in the field.",
    "Brown spot": "Apply balanced fertilizer, especially nitrogen and potassium. Use recommended fungicide sprays if infection is severe and remove heavily infected leaves.",
    "Leaf smut": "Remove and destroy infected leaves and panicles. Use disease-free seeds, treat seed before sowing, and practice crop rotation with non-host crops.",
}

_model = None
_rice_model = None


def hf_translate(text, target_lang):
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as exc:
        print(f"Translation Error ({target_lang}): {exc}")
        return text


def prepare_image(img):
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def parse_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return prepare_image(img)
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image file. Please upload a valid image.") from exc
    except Exception as exc:
        raise ValueError(f"Unable to read the uploaded file: {exc}") from exc


def get_top_predictions(preds, names, top_k=3):
    scores = np.asarray(preds).reshape(-1)
    indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "disease": names[int(i)],
            "confidence": float(np.round(float(scores[i]) * 100.0, 2)),
        }
        for i in indices
    ]


def get_main_model():
    global _model
    if _model is None:
        _model = load_model(str(MODEL_PATH))
    return _model


def get_rice_model():
    global _rice_model
    if _rice_model is None:
        _rice_model = load_model(str(RICE_MODEL_PATH))
    return _rice_model


def build_prediction_payload(file_bytes, model_getter, labels, treatment_map):
    try:
        prepared_img = parse_image(file_bytes)
    except ValueError as exc:
        return {"error": str(exc)}, 400

    try:
        preds = model_getter().predict(prepared_img)
    except Exception as exc:
        print(f"Prediction error: {exc}")
        return {"error": "Model prediction failed"}, 500

    top_predictions = get_top_predictions(preds, labels)
    pred_index = int(np.argmax(preds, axis=1)[0])
    disease = labels[pred_index]

    solution_en = treatment_map.get(disease, "Treatment information not available for this disease.")
    solution_te = hf_translate(solution_en, "te")
    solution_hi = hf_translate(solution_en, "hi")

    return {
        "disease": disease,
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
        "solution_en": solution_en,
        "solution_te": solution_te,
        "solution_hi": solution_hi,
    }, 200
