# src/config.py

# ===============================
# 🎯 TARGET COLUMN
# ===============================
TARGET_COLUMN = "target"   # change if your dataset has a different name


# ===============================
# 🔢 NUMERICAL FEATURES
# ===============================
NUMERICAL_FEATURES = [
    "feature1",
    "feature2"
]


# ===============================
# 🏷️ CATEGORICAL FEATURES
# ===============================
CATEGORICAL_FEATURES = [
    "category1"
]


# ===============================
# ❌ EXCLUDED COLUMNS
# ===============================
EXCLUDED_COLUMNS = [
    "id",          # unique identifier
    "timestamp"    # not available during prediction
]


# ===============================
# ✅ ALL FEATURES (FINAL INPUT)
# ===============================
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES