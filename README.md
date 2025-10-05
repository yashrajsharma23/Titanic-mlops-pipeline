# Titanic ML Pipeline 🚢

This project demonstrates a Machine Learning pipeline for predicting passenger survival on the Titanic dataset. It covers the full cycle of data loading, preprocessing, feature engineering, model training, evaluation, and CI/CD testing.

## 📂 Project Structure

```bash
Titanic-ML/
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── load_data.py         # Load dataset
│   ├── eda.py               # Visualizing data
│   ├── preprocess.py        # Build preprocessing pipeline
│   ├── train.py             # Train ML model
│   └── predict.py           # Evaluate model
│
├── tests/                   # Unit tests
│   └── test_preprocess.py   # Tests for preprocessing pipeline
│
├── run_pipeline.py          # Pipeline execution
├── .github/workflows/ci.yml # GitHub Actions CI workflow
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Titanic-ML.git
cd Titanic-ML
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 📊 Workflow
### 1. Load Data
```bash
from src.data import load_data

df = load_data("data/titanic.csv")
```

### 2. Preprocess
```bash
from src.preprocess import build_preprocessor

preprocessor = build_preprocessor()
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_processed = preprocessor.fit_transform(X)
```

### 3. Train Model
```bash
from src.train import train_model

model = train_model(X_processed, y)
```

### 4. Evaluate
```bash
from src.evaluate import evaluate_model

evaluate_model(model, X_processed, y)
```

## 🧪 Testing

We use pytest for unit testing.
Run all tests:
```bash
pytest
```


Run a specific test file:
```bash
pytest tests/test_preprocess.py
```

Run one test function:
```bash
pytest tests/test_preprocess.py::test_missing_values_handled
```

## 🔄 CI/CD with GitHub Actions

This project includes a CI pipeline (.github/workflows/ci.yml) that runs automatically on every push or pull request:

**Lint code with flake8**

**Run unit tests with pytest**

**Ensure reproducibility and reliability**

## 📌 Key Concepts Practiced

**Python Packaging (modular src/ structure)**

**Data Preprocessing with Pipeline + ColumnTransformer**

**Feature Encoding (OneHotEncoder, SimpleImputer)**

**Unit Testing with pytest**

**CI/CD using GitHub Actions**

**Code Quality with linters and formatting**

## 🚀 Next Steps

**Add MLflow tracking for experiments**

**Containerize with Docker**

**Deploy model via FastAPI / Flask**

**Add Spark for scalable pipelines**

**Integrate Terraform for infrastructure automation**

## 🤝 Contributing

1. Fork the repo

2. Create a new branch (feature/my-feature)

3. Commit your changes

4. Push and open a PR

## 📜 License

MIT License – free to use and modify.
