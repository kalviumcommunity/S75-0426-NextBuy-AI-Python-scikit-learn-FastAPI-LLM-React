import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    if df.empty:
        raise ValueError("Dataset is empty")

    return df