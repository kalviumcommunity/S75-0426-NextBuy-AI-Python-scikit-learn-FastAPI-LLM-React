import pandas as pd
from sklearn.model_selection import train_test_split
from config import TARGET_COLUMN, ALL_FEATURES

# Load dataset
df = pd.read_csv("data/sample.csv")

# =============================
# ✅ Validation
# =============================
assert TARGET_COLUMN in df.columns, "Target column missing"
assert TARGET_COLUMN not in ALL_FEATURES, "Target leakage in features"

# =============================
# ✅ Separate X and y
# =============================
X = df[ALL_FEATURES]
y = df[TARGET_COLUMN]

print("✅ X shape:", X.shape)
print("✅ y shape:", y.shape)

# =============================
# ✅ Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # remove this if regression
)

# =============================
# ✅ Verification
# =============================
print("\n📊 Training shape:", X_train.shape)
print("📊 Testing shape:", X_test.shape)

print("\n📊 Train distribution:")
print(y_train.value_counts(normalize=True))

print("\n📊 Test distribution:")
print(y_test.value_counts(normalize=True))

# =============================
# ✅ Leakage Check
# =============================
print("\n🚫 No preprocessing applied before split")
print("🚫 No scaling / encoding / resampling done before split")