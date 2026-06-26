from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

BACKEND_APP_PATH = Path(__file__).resolve().parent / "crop-disease-backend" / "app.py"

spec = spec_from_file_location("backend_app", BACKEND_APP_PATH)
backend_app_module = module_from_spec(spec)
spec.loader.exec_module(backend_app_module)

app = backend_app_module.app
