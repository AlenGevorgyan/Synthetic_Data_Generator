import pandas as pd
import os

# --- Configuration ---
# The path to your full, original CSV dataset.
SOURCE_CSV_PATH = "sample_data.csv"
# The file where the training portion of the data will be saved.
TRAIN_DEST_PATH = "train_data.csv"
# The file where the test portion of the data will be saved.
TEST_DEST_PATH = "test_data.csv"
# The proportion of data to use for training (e.g., 0.8 means 80%).
TRAIN_SPLIT_RATIO = 0.8


def split_data():
    """
    Loads a source CSV file and splits it into two separate files:
    one for training and one for testing.
    """
    # 1. Check if the source data exists.
    if not os.path.exists(SOURCE_CSV_PATH):
        print(f"Error: Source data file not found at '{SOURCE_CSV_PATH}'.")
        print("Please make sure your data is named correctly and in the same directory.")
        return

    print(f"Loading data from '{SOURCE_CSV_PATH}'...")
    df = pd.read_csv(SOURCE_CSV_PATH)

    # 2. Calculate the split point.
    split_index = int(len(df) * TRAIN_SPLIT_RATIO)

    if split_index == 0 or split_index == len(df):
        print("Error: Train split ratio is too extreme. Results in an empty dataset.")
        return

    # 3. Create the training and testing dataframes.
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # 4. Save the new data files.
    train_df.to_csv(TRAIN_DEST_PATH, index=False)
    print(f"Successfully saved {len(train_df)} rows to '{TRAIN_DEST_PATH}'")

    test_df.to_csv(TEST_DEST_PATH, index=False)
    print(f"Successfully saved {len(test_df)} rows to '{TEST_DEST_PATH}'")

    print("\nData preparation complete.")


if __name__ == "__main__":
    split_data()
