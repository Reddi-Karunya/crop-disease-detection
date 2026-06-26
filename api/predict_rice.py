from flask import Flask, jsonify, request

from api.shared import build_prediction_payload, get_rice_model, rice_class_names, rice_treatment_solutions

app = Flask(__name__)


@app.post("/predict_rice")
@app.post("/api/predict_rice")
def predict_rice():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    payload, status_code = build_prediction_payload(
        file.read(),
        get_rice_model,
        rice_class_names,
        rice_treatment_solutions,
    )
    return jsonify(payload), status_code
