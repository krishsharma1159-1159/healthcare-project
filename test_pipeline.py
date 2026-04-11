"""Quick pipeline test - reads from local cache."""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from tigergraph_ml_integration import (
    _normalize_records,
    load_config_from_env,
    preprocess_features,
    load_or_create_model,
    run_predictions,
)

# Load cache
cache_path = os.path.join(os.path.dirname(__file__), "tg_features_cache.json")
with open(cache_path, "r") as f:
    cached = json.load(f)

records = cached[0].get("patients", [])
print(f"Loaded {len(records)} patient records from cache")

# Normalize
df = _normalize_records(records)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"First 3 rows:")
print(df.head(3).to_string())

# Load config
cfg = load_config_from_env()
print(f"\nFeature columns: {cfg.feature_columns}")

# Check which features exist
missing = [c for c in cfg.feature_columns if c not in df.columns]
if missing:
    print(f"MISSING columns: {missing}")
    print(f"Available columns: {list(df.columns)}")
else:
    print("All feature columns found!")

    # Preprocess
    ids, X = preprocess_features(
        df=df,
        feature_columns=cfg.feature_columns,
        id_column=cfg.id_column,
        scaler_path=cfg.scaler_path,
    )
    print(f"\nPreprocessed: {len(ids)} records, {X.shape[1]} features")

    # Load model
    model, used_dummy, msg = load_or_create_model(cfg.model_path, cfg.feature_columns)
    print(f"Model: {msg}")

    # Predict
    labels, scores = run_predictions(model, X)
    print(f"\nPredictions ({len(labels)} total):")
    for i, (pid, lbl, sc) in enumerate(zip(ids, labels, scores)):
        if i >= 10:
            break
        print(f"  {pid}: {lbl} (confidence: {sc:.3f})")
    
    # Check variety
    unique = set(labels)
    print(f"\nUnique predictions: {unique}")
    if len(unique) == 1:
        print("WARNING: All predictions are the same!")
    else:
        print("GOOD: Predictions are diverse!")
