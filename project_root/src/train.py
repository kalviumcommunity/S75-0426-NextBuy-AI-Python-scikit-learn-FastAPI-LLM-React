from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, random_state: int):
    """Fit model on training data and return the fitted model."""
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model