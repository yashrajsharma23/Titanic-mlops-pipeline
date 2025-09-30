from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Optional
from src.load_data import load_dataset
# from src.preprocess import clean_data
from src.preprocess import build_preprocessor
from sklearn.metrics import accuracy_score
import joblib

def train_model(save_path: str = "models/titanic_model.pkl"):

    df = load_dataset()
        
        # df = clean_data(df)
    # else:
    #     X, y =clean_data(df)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    preprocessor =build_preprocessor()

    #Build full pipeline (Preprocessing + Model)
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    print("🔹 Step 4: Training model...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 Model Accuracy: {acc: .4f}")

    #Save full pipeline
    joblib.dump(clf, save_path)
    print(f"✅ Saved model pipeline at {save_path}")
    return clf

if __name__ == "__main__":
    train_model()
