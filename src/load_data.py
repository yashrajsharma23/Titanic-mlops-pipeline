import pandas as pd

def load_dataset(path: str="./dataset/titanic.csv") -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(path)

# if __name__ == "__main__":
#     df = load_dataset("../dataset/titanic.csv")
#     print(df.head())
