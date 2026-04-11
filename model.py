"""
Train a RandomForestClassifier on the healthcare dataset.

Feature columns used for training (must match inference exactly):
  - 7 binary symptom flags: fever, cough, headache, fatigue, chest_pain, shortness_of_breath, nausea
  - 7 severity scores: fever_severity, cough_severity, headache_severity, fatigue_severity, chest_pain_severity, sob_severity, nausea_severity
  - 1 aggregate: num_symptoms
  - 1 demographic: age

Total: 16 numeric features → target: disease
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Load data ────────────────────────────────────────────────────────
df = pd.read_csv("clean_dataset.csv")

# ── Define feature columns (must match exactly what inference provides) ──
FEATURE_COLUMNS = [
    # Binary symptom flags
    "fever", "cough", "headache", "fatigue",
    "chest_pain", "shortness_of_breath", "nausea",
    # Severity scores
    "fever_severity", "cough_severity", "headache_severity",
    "fatigue_severity", "chest_pain_severity", "sob_severity",
    "nausea_severity",
    # Aggregates
    "num_symptoms",
    # Demographics
    "age",
]

TARGET_COLUMN = "disease"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# XGBoost requires numeric targets for classification
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── Train/test split for evaluation ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ── Train model ──────────────────────────────────────────────────────
# Using XGBoost with softprob for multi-class confidence scores
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("XGBoost Model trained on", len(X_train), "samples")
print(f"\nClasses: {list(le.classes_)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

# ── Save ─────────────────────────────────────────────────────────────
# We save the model AND the label encoder to ensure consistent inference
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\nModel saved to model.pkl")
print("Label Encoder saved to label_encoder.pkl")
print(f"Feature columns expected: {FEATURE_COLUMNS}")
