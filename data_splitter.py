# data_splitter.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_DATASET_FILENAME = 'response_data.csv' 
TRAIN_FILENAME = 'train_dataset.csv'
TEST_FILENAME = 'test_dataset.csv'
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42


def split_dataset():
    if not os.path.exists(INPUT_DATASET_FILENAME):
        print(f"--- ERROR: Input file not found: '{INPUT_DATASET_FILENAME}' ---")
        return

    print(f"Loading data from '{INPUT_DATASET_FILENAME}'...")
    df = pd.read_csv(INPUT_DATASET_FILENAME)
    print(f"Successfully loaded {len(df)} rows.")

    print("\nSplitting the data...")
    train_df, test_df = train_test_split(
        df, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    print(f"Training set size: {len(train_df)} rows")
    print(f"Testing set size: {len(test_df)} rows")

    try:
        print(f"\nSaving training data to '{TRAIN_FILENAME}'...")
        train_df.to_csv(TRAIN_FILENAME, index=False)
        
        print(f"Saving testing data to '{TEST_FILENAME}'...")
        test_df.to_csv(TEST_FILENAME, index=False)
        
        print("\n--- Process Complete ---")

    except Exception as e:
        print(f"\nAn error occurred while saving the files: {e}")

if __name__ == '__main__':
    split_dataset()

