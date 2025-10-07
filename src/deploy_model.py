import mlflow
import os
import json

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "hhttp://127.0.0.1:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment_name = "Titanic-Classification"
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
run_id = runs.iloc[0]["run_id"]

print(f"Deploying model from run_id: {run_id}")

# 4️⃣ Register the model in MLflow Model Registry
model_uri = f"runs:/{run_id}/titanic_model"
registered_model_name = "TitanicSurvivalModel"

mlflow.register_model(model_uri=model_uri, name=registered_model_name)

print(f"Model deployed to MLflow Model Registry: TitanicSurvivalModel")
