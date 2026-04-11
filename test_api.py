"""Test the Flask API endpoints."""
import requests
import json

BASE = "http://127.0.0.1:8000"

# Test predict endpoint
print("=== POST /predict-from-tigergraph ===")
r = requests.post(f"{BASE}/predict-from-tigergraph", json={})
d = r.json()
print(f"Status: {d['status']}")
print(f"Message: {d.get('message', 'N/A')}")
preds = d.get("sample_predictions", [])
print(f"\nPredictions ({len(preds)} patients):")
for p in preds:
    print(f"  {p['id']:>5}: {p['prediction']:<20s} (confidence: {p['score']:.3f})")

# Show unique predictions
unique = set(p["prediction"] for p in preds)
print(f"\nUnique diseases predicted: {unique}")
print(f"Graph nodes: {len(d.get('graph', {}).get('nodes', []))}")
print(f"Graph edges: {len(d.get('graph', {}).get('edges', []))}")
