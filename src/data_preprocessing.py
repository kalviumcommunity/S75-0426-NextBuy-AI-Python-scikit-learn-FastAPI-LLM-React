# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def clean_data(df):
    return df.dropna()

def split_data(df):
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)