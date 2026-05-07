import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data():

    data = pd.read_csv("data/house_price.csv")

    X = data.drop("price", axis=1)
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test