import joblib
import pandas as pd

def predict(sample: pd.DataFrame, model_path: str = "models/titanic_model.pkl"):
    #Load saved pipeline
    clf = joblib.load(model_path)

    #Predict
    prediction = clf.predict(sample)
    return prediction

if __name__=="__main__":
    #Example new passenger
    sample = pd.DataFrame([{
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }])

    result = predict(sample)
    print("🚢 Survival Prediction:", result[0])