from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)  # enable CORS once, on the single app

# ----------------- MAIN (tomato/potato/pepper) MODEL -----------------

model = load_model('crop_disease_cnn_model.keras')

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
    "Tomato_Spider_mites_Two_spotted_spider_mite"
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
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides or insecticidal soap. Regularly spray plants with water to reduce mite populations."
}

# ----------------- RICE MODEL -----------------

rice_model = load_model('rice_disease_cnn_model.keras')

class_names_rice = [
    "Bacterial leaf blight",  # 0
    "Brown spot",             # 1
    "Leaf smut"               # 2
]

rice_treatment_solutions = {
    "Bacterial leaf blight": "Use resistant rice varieties and apply recommended bactericides. Avoid standing water and improve drainage in the field.",
    "Brown spot": "Apply balanced fertilizer, especially nitrogen and potassium. Use recommended fungicide sprays if infection is severe and remove heavily infected leaves.",
    "Leaf smut": "Remove and destroy infected leaves and panicles. Use disease-free seeds, treat seed before sowing, and practice crop rotation with non-host crops."
}

# ----------------- SHARED HELPERS -----------------

def hf_translate(text, target_lang):
    if target_lang == "te":
        model_name = "Meher2006/english-to-telugu-model"
    elif target_lang == "hi":
        model_name = "Helsinki-NLP/opus-mt-en-hi"
    else:
        return text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tmodel = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = tmodel.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def prepare_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------- ROUTES -----------------

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Vegetables: tomato/potato/pepper
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    prepared_img = prepare_image(img)

    preds = model.predict(prepared_img)
    pred_index = np.argmax(preds)
    disease = class_names[pred_index]

    solution_en = treatment_solutions.get(
        disease,
        "Treatment information not available for this disease."
    )
    solution_te = hf_translate(solution_en, "te")
    solution_hi = hf_translate(solution_en, "hi")

    print("Disease:", disease)
    print("English:", solution_en)
    print("Hindi:", solution_hi)
    print("Telugu:", solution_te)

    return jsonify({
        "disease": disease,
        "solution_en": solution_en,
        "solution_te": solution_te,
        "solution_hi": solution_hi
    }), 200

# Rice endpoint
@app.route('/predict_rice', methods=['POST'])
def predict_rice():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    prepared_img = prepare_image(img)

    preds = rice_model.predict(prepared_img)
    pred_index = int(np.argmax(preds))
    disease = class_names_rice[pred_index]

    solution_en = rice_treatment_solutions.get(
        disease,
        "Treatment information not available for this disease."
    )
    solution_te = hf_translate(solution_en, "te")
    solution_hi = hf_translate(solution_en, "hi")

    print("Rice Disease:", disease)
    print("English:", solution_en)
    print("Hindi:", solution_hi)
    print("Telugu:", solution_te)

    return jsonify({
        "disease": disease,
        "solution_en": solution_en,
        "solution_te": solution_te,
        "solution_hi": solution_hi
    }), 200

if __name__ == '__main__':
    print('Preview URL: http://127.0.0.1:5000/', flush=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
