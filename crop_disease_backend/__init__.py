from pathlib import Path
import importlib.util

BACKEND_APP_PATH = Path(__file__).resolve().parent.parent / "crop-disease-backend" / "app.py"
spec = importlib.util.spec_from_file_location("backend_app", BACKEND_APP_PATH)
backend_app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_app_module)

app = backend_app_module.app
