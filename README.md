# crop-disease-detection

## Live Demo
When deployed, the app will be accessible at a public URL. Connect this repo to Render and the deployment will auto-provision. The service name is `crop-disease-backend`. The link will appear on the Render dashboard after the first deploy.

Local Preview: http://127.0.0.1:5000/

## Deploy to Render (One-time Setup)
- Ensure this repository is on GitHub.
- In Render, create a new Web Service from this GitHub repo.
- Render detects `render.yaml` and provisions the service automatically.
- First build installs dependencies from [crop-disease-backend/requirements.txt](file:///c:/Users/REDDY%20KARUNYA/crop-disease-detection/crop-disease-backend/requirements.txt).
- Start command runs `gunicorn app:app` in [crop-disease-backend](file:///c:/Users/REDDY%20KARUNYA/crop-disease-detection/crop-disease-backend).

## Project Structure
- Backend Flask app: [app.py](file:///c:/Users/REDDY%20KARUNYA/crop-disease-detection/crop-disease-backend/app.py)
- Frontend UI: [templates/index.html](file:///c:/Users/REDDY%20KARUNYA/crop-disease-detection/crop-disease-backend/templates/index.html)
- Deployment config: [render.yaml](file:///c:/Users/REDDY%20KARUNYA/crop-disease-detection/render.yaml)
