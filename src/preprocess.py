import pandas as pd
from src.load_data import load_dataset
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor() ->ColumnTransformer:
    """Build Preprocessing pipeline for Numerical and Categorical data"""
  
    # Define numeric and categorical columns
    numeric_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
    categorical_features = ["Sex", "Embarked"]

        # Preprocessing for numeric data
    numeric_transformer =  Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

     # Preprocessing for categorical data
    categorical_transformer  = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(categories=[["female","male"], ["C","Q","S"]],handle_unknown="ignore"))
    ])

    #Combine preprocessing for numeric + Categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     print("🔹 Step 3: Preprocessing...")

#     # Drop irrelevant columns
#     df = df.drop(["Name", "Ticket", "Cabin"], axis=1, errors="ignore")

#     # Separate features and target
#     X = df.drop("Survived", axis=1)
#     y = df["Survived"]

#     # Define numeric and categorical columns
#     numeric_features = ["Age", "Fare", "Pclass", "SibSp", "Parch"]
#     categorical_features = ["Sex", "Embarked"]

#     # Preprocessing for numeric data
#     numeric_transformer =  SimpleImputer(strategy="median")

#      # Preprocessing for categorical data
#     categorical_transformer  = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore"))
#     ])
    
#     #Combine preprocessing for numeric + Categorical
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer,numeric_features),
#             ("cat", categorical_transformer, categorical_features),
#         ]
#     )

    
#     x_processed = preprocessor.fit_transform(X)
    
#     feature_name = (
#         numeric_features
#         + list(preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features))
#     )

#     x_processed = pd.DataFrame(x_processed.toarray() if hasattr(x_processed,"toarray") else x_processed, columns=feature_name)

#     #Returns processed features+target
#     return x_processed, y
             
# if __name__ == "__main__":
#     df = load_dataset()
#     X, y = clean_data(df)
#     print(X.head())
#     print(y.head())

