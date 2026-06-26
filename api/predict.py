from flask import Flask, jsonify, request

from api.shared import build_prediction_payload, class_names, get_main_model, treatment_solutions

app = Flask(__name__)


@app.post("/predict")
@app.post("/api/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    payload, status_code = build_prediction_payload(
        file.read(),
        get_main_model,
        class_names,
        treatment_solutions,
    )
    return jsonify(payload), status_code
