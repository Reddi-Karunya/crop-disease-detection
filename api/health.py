from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({"status": "ok"}), 200
