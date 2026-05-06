from src.config import *
from src.data_preprocessing import load_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts # type: ignore

# Execute Workflow
df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = split_data(df, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)

pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
X_train_proc = pipeline.fit_transform(X_train)
X_test_proc = pipeline.transform(X_test)

model = train_model(X_train_proc, y_train, RANDOM_STATE)
metrics = evaluate_model(model, X_test_proc, y_test)

save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
print(f"Training Complete. Metrics: {metrics}")