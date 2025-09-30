from src.load_data import load_dataset
# from src.preprocess import clean_data
from src.train import train_model
from src.eda import eda

def main():
    print("🔹 Step 1: Loading dataset...")
    df = load_dataset()

    print("🔹 Step 2: Running EDA...")
    eda(df)   # optional: if you don’t want plots every time, comment out

    # print("🔹 Step 3: Preprocessing...")
    # clean_df = clean_data(df)

    model = train_model()

    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
