from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Optional
from src.load_data import load_dataset
# from src.preprocess import clean_data
from src.preprocess import build_preprocessor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np

import joblib
import json
import os

def train_model(
        model_params: Optional[Dict[str, Any]] = None,
        save_path: str = "models/titanic_model.pkl"):

    """
    Train Titanic survival prediction model with comprehensive MLflow logging.
    
    Args:
        model_params: Dictionary of hyperparameters for RandomForestClassifier
        save_path: Path to save the trained model
    
    Returns:
        Trained pipeline and run_id for model deployment
    """
    
    # Default hyperparameters
    if model_params is None:
        model_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }
    
    print("Inside Train model")
    df = load_dataset()
    df = df.replace({None: np.nan})
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    preprocessor =build_preprocessor()

    #Build full pipeline (Preprocessing + Model)
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**model_params))
    ])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # If running MLflow locally (UI at http://127.0.0.1:5000)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    # mlflow.set_tracking_uri("http://127.0.0.1:5000"))

    # Optional: organize experiments
    mlflow.set_experiment("Titanic-Classification")

    # Enable autologging for sklearn (logs params, metrics, model automatically)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    #MLFlow start
    with mlflow.start_run(run_name=f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        print(f"🔹 MLflow Run ID: {run_id}")

        print("🔹 Step 4: Training model...")

        #Fit model        
        clf.fit(X_train, y_train)

        #Predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

         # Calculate comprehensive metrics
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "train_f1": f1_score(y_train, y_pred_train),
            "test_f1": f1_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test),
            "test_recall": recall_score(y_test, y_pred_test),
        }

        # Log additional metrics (autolog handles basic ones)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)


        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("total_features", X_train.shape[1])

        # Create and log confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("metrics/confusion_matrix.png")
        mlflow.log_artifact("metrics/confusion_matrix.png")
        plt.close()

        # Log feature importance if available
        if hasattr(clf.named_steps["classifier"], "feature_importances_"):
            feature_importance = clf.named_steps["classifier"].feature_importances_
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_importance)), feature_importance)
            plt.xlabel("Importance")
            plt.ylabel("Feature Index")
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig("metrics/feature_importance.png")
            mlflow.log_artifact("metrics/feature_importance.png")
            plt.close()


        #Log Model
        signature = infer_signature(X_test, y_pred_test)

        mlflow.sklearn.log_model(
            clf,
            artifact_path="titanic_model",
            input_example=X_test.iloc[:5],   # first 5 rows as example
            signature=signature,
            registered_model_name="TitanicSurvivalModel"
        )

        # Save run info for later use
        run_info = {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("models/run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        #Save full pipeline
        joblib.dump(clf, save_path)
        print(f"\n📊 Model Performance:")
        print(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"   Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"   Test F1 Score:  {metrics['test_f1']:.4f}")
        print(f"   Test Precision: {metrics['test_precision']:.4f}")
        print(f"   Test Recall:    {metrics['test_recall']:.4f}")
        print(f"\n✅ Model saved at {save_path}")
        print(f"✅ Run info saved to models/run_info.json")

    return clf

if __name__ == "__main__":
# You can pass custom hyperparameters
    custom_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    train_model(model_params=custom_params)