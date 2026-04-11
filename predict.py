"""
Predict disease from a symptom vector.

The model expects exactly 16 features in this order:
  fever, cough, headache, fatigue, chest_pain, shortness_of_breath, nausea,
  fever_severity, cough_severity, headache_severity, fatigue_severity,
  chest_pain_severity, sob_severity, nausea_severity,
  num_symptoms, age
"""

import joblib
import numpy as np

# Feature order matching model.py
FEATURE_COLUMNS = [
    "fever", "cough", "headache", "fatigue",
    "chest_pain", "shortness_of_breath", "nausea",
    "fever_severity", "cough_severity", "headache_severity",
    "fatigue_severity", "chest_pain_severity", "sob_severity",
    "nausea_severity",
    "num_symptoms", "age",
]

# Load the model once when the module is imported
model = joblib.load("model.pkl")


def predict_disease(symptoms_dict: dict) -> tuple:
    """
    Predict disease from a dictionary of symptom values.

    Args:
        symptoms_dict: dict mapping feature names to numeric values.
                       Missing keys default to 0.

    Returns:
        (predicted_disease: str, confidence: float)
    """
    # Build feature vector in correct order
    feature_values = [symptoms_dict.get(col, 0) for col in FEATURE_COLUMNS]
    X = np.array([feature_values], dtype=np.float32)

    prediction = model.predict(X)[0]

    # Get confidence score
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = float(np.max(proba))

    return prediction, confidence


def predict_disease_raw(symptoms_list: list) -> str:
    """
    Legacy interface: predict from a flat list of feature values.
    The list must be in the same order as FEATURE_COLUMNS.
    """
    X = np.array([symptoms_list], dtype=np.float32)
    return model.predict(X)[0]
