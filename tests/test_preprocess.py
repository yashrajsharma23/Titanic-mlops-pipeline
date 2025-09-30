import pandas as pd

import pytest
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import build_preprocessor

# 1. Test that the pipeline runs without errors
def test_preprocessor_runs():

    df = pd.DataFrame({
        "Age": [22, None, 35],
        "Fare": [7.25, 71.83, None],
        "Pclass": [3, 1, 2],
        "SibSp": [1, 0, 0],
        "Parch": [0, 0, 0],
        "Sex": ["male", "female", "male"],
        "Embarked": ["S", None, "C"]
    })

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(df)

    assert X_transformed.shape[0] == df.shape[0]  # rows stay the same

# 2. Test that missing values are handled
def test_missing_values_handled():
    df = pd.DataFrame({
        "Age": [22, None],
        "Fare": [None, 50],
        "Pclass": [3, 1],
        "SibSp": [0, 1],
        "Parch": [0, 0],
        "Sex": ["male", "female"],
        "Embarked": [None, "S"]
    })

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(df)

    # No NaNs should remain
    assert not pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed).isnull().any().any()


# 3. Test that categorical encoding produces the right number of columns
from sklearn.preprocessing import OneHotEncoder

def test_categorical_encoding_shape():
    df = pd.DataFrame({
        "Age": [22, 35],
        "Fare": [7.25, 71.83],
        "Pclass": [3, 1],
        "SibSp": [1, 0],
        "Parch": [0, 0],
        "Sex": ["male", "female"],
        "Embarked": ["S", "C"]
    })

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(df)

    # Expected columns: 5 numeric + (2 for Sex + 3 for Embarked) = 10
    assert X_transformed.shape[1] == 10

# 4. Test feature names (optional but powerful)
def test_feature_names():
    df = pd.DataFrame({
        "Age": [22, 35],
        "Fare": [7.25, 71.83],
        "Pclass": [3, 1],
        "SibSp": [1, 0],
        "Parch": [0, 0],
        "Sex": ["male", "female"],
        "Embarked": ["S", "C"]
    })

    preprocessor = build_preprocessor()
    preprocessor.fit(df)

    feature_names = (
        ["Age", "Fare", "Pclass", "SibSp", "Parch"]
        + list(preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(["Sex", "Embarked"]))
    )

    assert "Sex_male" in feature_names
    assert "Embarked_S" in feature_names

