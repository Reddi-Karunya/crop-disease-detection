from pathlib import Path

from flask import Flask, Response

app = Flask(__name__)

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "crop-disease-backend" / "templates" / "index.html"


@app.get("/")
@app.get("/api/index")
def home():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    return Response(html, mimetype="text/html")
