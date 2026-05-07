# src/config.py

# ==========================================
# 🎯 TARGET COLUMN
# ==========================================
TARGET_COLUMN = "target"


# ==========================================
# 🔢 NUMERICAL FEATURES
# ==========================================
NUMERICAL_FEATURES = [
    "feature1",
    "feature2"
]


# ==========================================
# 🏷️ CATEGORICAL FEATURES
# ==========================================
CATEGORICAL_FEATURES = [
    "category1"
]


# ==========================================
# ❌ EXCLUDED COLUMNS
# ==========================================
EXCLUDED_COLUMNS = [
    "id",
    "timestamp"
]


# ==========================================
# ⚙️ TRAIN TEST CONFIGURATION
# ==========================================
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==========================================
# 💾 ARTIFACT PATHS
# ==========================================
MODEL_PATH = "models/model.pkl"
PIPELINE_PATH = "models/pipeline.pkl"


# ==========================================
# ✅ FINAL FEATURE LIST
# ==========================================
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ==========================================
# 🔒 VALIDATION CHECKS
# ==========================================
assert TARGET_COLUMN not in ALL_FEATURES

for column in EXCLUDED_COLUMNS:
    assert column not in ALL_FEATURES