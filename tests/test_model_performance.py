"""
Unit tests for model performance and functionality.
"""
import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Adjust import paths as needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load_data import load_dataset
from src.preprocess import build_preprocessor

class TestModelPerformance:
    
    @pytest.fixture(scope="class")
    def model(self):
        """Load the trained model."""
        model_path = Path("models/titanic_model.pkl")
        if not model_path.exists():
            pytest.skip("Model file not found")
        return joblib.load(model_path)
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Load sample test data."""
        return pd.DataFrame([
            {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S"
            },
            {
                "Pclass": 1,
                "Sex": "female",
                "Age": 38,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 71.28,
                "Embarked": "C"
            }
        ])
    
    def test_model_exists(self):
        """Test that model file exists."""
        assert Path("models/titanic_model.pkl").exists(), "Model file not found"
    
    def test_model_loads(self, model):
        """Test that model loads successfully."""
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_model_prediction_shape(self, model, sample_data):
        """Test that model produces correct prediction shape."""
        predictions = model.predict(sample_data)
        assert predictions.shape[0] == len(sample_data)
    