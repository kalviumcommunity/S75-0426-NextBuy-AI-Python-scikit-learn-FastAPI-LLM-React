# ==========================================
# 🎯 TARGET COLUMN
# ==========================================
TARGET_COLUMN = "target"


# ==========================================
# 🔢 NUMERICAL FEATURES
# ==========================================
# These features represent measurable quantities
# where mathematical operations are meaningful.
NUMERICAL_FEATURES = [
    "feature1",
    "feature2"
]


# ==========================================
# 🏷️ CATEGORICAL FEATURES
# ==========================================
# These features represent labels/categories
# rather than continuous measurable values.
CATEGORICAL_FEATURES = [
    "category1"
]


# ==========================================
# ❌ EXCLUDED COLUMNS
# ==========================================
# Columns excluded because they are:
# - identifiers
# - leakage-prone
# - unavailable during prediction
EXCLUDED_COLUMNS = [
    "id",
    "timestamp"
]


# ==========================================
# ⚙️ TRAIN-TEST SPLIT CONFIGURATION
# ==========================================
TEST_SIZE = 0.2
RANDOM_STATE = 42


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