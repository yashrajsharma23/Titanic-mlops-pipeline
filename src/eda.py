import seaborn as sns
import matplotlib.pyplot as plt
from src.load_data import load_dataset
from typing import Optional
import pandas as pd

def eda(df: Optional[pd.DataFrame] = None):
    if df is None:
        df = load_dataset()
        # df = clean_data(df)

    print(df.info())
    print(df.describe())

    sns.countplot(x="Survived", data=df)
    plt.show()


# if __name__ == "__main__":
#     eda()
