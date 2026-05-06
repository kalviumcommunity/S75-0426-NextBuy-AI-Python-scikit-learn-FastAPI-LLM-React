from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE

def clean_data(df):
    # Simple cleaning (can extend later)
    df = df.dropna()
    return df

def split_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )