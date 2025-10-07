import mlflow
import os
import json

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "hhttp://127.0.0.1:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 2️⃣ Path to the run_info.json (downloaded via GitHub Actions)
run_info_path = "models/run_info.json"

# 3️⃣ Load run_info.json
with open(run_info_path, "r") as f:
    run_info = json.load(f)

run_id = run_info["run_id"]
print(f"Deploying model from run_id: {run_id}")

# 4️⃣ Register the model in MLflow Model Registry
model_uri = f"runs:/{run_id}/titanic_model"
registered_model_name = "TitanicSurvivalModel"

mlflow.register_model(model_uri=model_uri, name=registered_model_name)

print(f"Model deployed to MLflow Model Registry: TitanicSurvivalModel")
