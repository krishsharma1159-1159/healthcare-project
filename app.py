from __future__ import annotations

from typing import Any, Dict, List

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS  # type: ignore[reportMissingImports]

from tigergraph_ml_integration import (
    AppConfig,
    _normalize_records,
    build_dummy_dataframe,
    connect_tigergraph,
    execute_prediction_pipeline,
    load_config_from_env,
    load_or_create_model,
    preprocess_features,
    run_predictions,
)

app = Flask(__name__)
CORS(app)

# Load TigerGraph + ML configuration from environment variables.
CFG: AppConfig = load_config_from_env()


@app.after_request
def add_cors_headers(response):
    """
    Add permissive CORS headers for local beginner setup.
    If you serve frontend from Flask (recommended), CORS will not be needed.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def home():
    # Serve frontend from backend to avoid URL/CORS mismatch.
    return send_from_directory(".", "index.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return send_from_directory(".", "dashboard.html")


@app.route("/dashboard.html", methods=["GET"])
def dashboard_html():
    return send_from_directory(".", "dashboard.html")


@app.route("/styles.css", methods=["GET"])
def styles_css():
    return send_from_directory(".", "styles.css")


@app.route("/script.js", methods=["GET"])
def script_js():
    return send_from_directory(".", "script.js")


def _color_for_prediction(label: str) -> str:
    """Assign node colors based on prediction label."""
    lower = str(label).lower().replace("_", " ")
    if "covid" in lower:
        return "#e74c3c"  # red — COVID-19
    if "pneumonia" in lower:
        return "#e67e22"  # dark orange — Pneumonia
    if "flu" in lower:
        return "#f39c12"  # orange — Flu
    if "food" in lower or "poisoning" in lower:
        return "#27ae60"  # green — Food Poisoning
    if "migraine" in lower:
        return "#3498db"  # blue — Migraine
    if "healthy" in lower:
        return "#2ecc71"  # bright green
    return "#9b59b6"  # purple fallback


def _build_graph_nodes(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert pipeline sample predictions to visualization nodes.
    Each node has label + color based on prediction.
    """
    nodes: List[Dict[str, Any]] = []
    for idx, row in enumerate(records):
        pred_label = str(row.get("prediction", "Unknown"))
        score = row.get("score")
        node_id = str(row.get("id", idx))
        nodes.append(
            {
                "id": node_id,
                "label": f"{node_id}\n{pred_label}",
                "title": f"Prediction: {pred_label}, Score: {score}",
                "color": _color_for_prediction(pred_label),
            }
        )
    return nodes


def _build_graph_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create simple chain edges so graph is visually connected.
    """
    edges: List[Dict[str, Any]] = []
    for i in range(len(nodes) - 1):
        edges.append({"from": nodes[i]["id"], "to": nodes[i + 1]["id"]})
    return edges


@app.route("/predict-from-tigergraph", methods=["POST"])
def predict_from_tigergraph():
    print("[API] /predict-from-tigergraph called")
    body = request.get_json(silent=True) or {}
    print(f"[API] Request body: {body}")

    # Process custom symptoms from UI
    symptoms = body.get("symptoms", {})
    user_sample = None
    if symptoms and isinstance(symptoms, dict):
        try:
            from tigergraph_ml_integration import load_or_create_model, run_predictions
            import numpy as np
            
            # Map dynamic UI inputs into full 16-feature vector
            fever = symptoms.get("fever", 0)
            cough = symptoms.get("cough", 0)
            headache = symptoms.get("headache", 0)
            fatigue = symptoms.get("fatigue", 0)
            chest_pain = symptoms.get("chest_pain", 0)
            
            f_sev = 80 if fever else 0
            c_sev = 75 if cough else 0

            num_symp = (1 if fever else 0) + (1 if cough else 0) + (1 if headache else 0) + (1 if fatigue else 0) + (1 if chest_pain else 0)

            # [fever, cough, headache, fatigue, chest_pain, sob, nausea, fs, cs, hs, fas, cps, sobs, ns, num, age]
            custom_x = [
                fever, cough, headache, fatigue, chest_pain, 0, 0,
                f_sev, c_sev, 0, 0, 0, 0, 0,
                num_symp, 45
            ]
            custom_X = np.array([custom_x], dtype=np.float32)
            
            model, le, _ = load_or_create_model(CFG.model_path, CFG.feature_columns)
            custom_labels, custom_scores = run_predictions(model, custom_X, le=le)
            
            final_label = str(custom_labels[0])
            user_score = custom_scores[0]
            # Heuristics removed: relying on model prediction
            # Default logic for asymptomatic cases
            if fever == 0 and cough == 0 and headache == 0 and fatigue == 0 and chest_pain == 0:
                final_label = "Healthy"
                user_score = 0.99

            user_sample = {
                "id": "Current Patient",
                "prediction": final_label,
                "score": user_score
            }
            print(f"[API] Custom user prediction: {user_sample}")
        except Exception as e:
            print(f"[API] Error processing custom symptoms: {e}")

    try:
        # 1) Connect TigerGraph
        conn = connect_tigergraph(CFG)
        print("[API] Connected to TigerGraph")

        # 2) Fetch + predict + write-back to TigerGraph
        result = execute_prediction_pipeline(conn, CFG, source_mode=body.get("source_mode"))
        print("[API] Prediction done and write-back attempted")

        sample_predictions = result.get("sample_predictions", [])
        
        if user_sample:
            if sample_predictions:
                sample_predictions[0] = user_sample
            else:
                sample_predictions.append(user_sample)

        nodes = _build_graph_nodes(sample_predictions)
        edges = _build_graph_edges(nodes)

        return jsonify(
            {
                "status": "ok",
                "message": "Prediction done, TigerGraph updated.",
                "pipeline_result": result,
                "sample_predictions": sample_predictions,
                "graph": {"nodes": nodes, "edges": edges},
            }
        ), 200
    except Exception as exc:
        # Fallback: use cached TigerGraph data (extracted via MCP) for predictions
        print(f"[API] TigerGraph pipeline failed: {exc}")
        print("[API] Switching to cached-data prediction flow")
        try:
            import json as _json
            import os as _os
            from tigergraph_ml_integration import (
                _normalize_records,
                load_or_create_model,
                preprocess_features,
                run_predictions,
            )

            cache_path = _os.path.join(_os.path.dirname(__file__) or ".", "tg_features_cache.json")
            if _os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    cached = _json.load(f)
                records = cached[0].get("patients", []) if cached else []
                df = _normalize_records(records)
                data_source = "TigerGraph cache"
            else:
                df = build_dummy_dataframe(CFG.feature_columns, rows=8)
                data_source = "dummy data"

            ids, X = preprocess_features(
                df=df,
                feature_columns=CFG.feature_columns,
                id_column=CFG.id_column,
                scaler_path=CFG.scaler_path,
            )
            model, le, model_msg = load_or_create_model(CFG.model_path, CFG.feature_columns)
            labels, scores = run_predictions(model, X, le=le)

            sample_predictions: List[Dict[str, Any]] = []
            for i, (pid, label, score) in enumerate(zip(ids.tolist(), labels, scores)):
                if i >= 25:
                    break
                sample_predictions.append({"id": pid, "prediction": str(label), "score": score})

            if user_sample:
                if sample_predictions:
                    sample_predictions[0] = user_sample
                else:
                    sample_predictions.append(user_sample)

            nodes = _build_graph_nodes(sample_predictions)
            edges = _build_graph_edges(nodes)
            print(f"[API] Fallback prediction done ({data_source}, {len(sample_predictions)} patients)")

            error_msg = str(exc)
            display_msg = f"Using {data_source}. TigerGraph auth unavailable."
            if "STOPPED" in error_msg or "start" in error_msg.lower():
                display_msg = "TigerGraph WORKSPACE IS STOPPED. Please start it in the Cloud Portal."

            return jsonify(
                {
                    "status": "fallback",
                    "message": display_msg,
                    "error": error_msg,
                    "model_message": model_msg,
                    "sample_predictions": sample_predictions,
                    "graph": {"nodes": nodes, "edges": edges},
                }
            ), 200
        except Exception as fallback_exc:
            print(f"[API] Fallback flow failed: {fallback_exc}")
            return jsonify({"status": "error", "message": str(fallback_exc)}), 500


@app.route("/ml-metrics", methods=["GET"])
def get_ml_metrics():
    """Calculate real-time Accuracy and Precision."""
    import pandas as pd
    import xgboost as xgb
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.preprocessing import LabelEncoder
    import os

    try:
        data_file = "clean_dataset.csv"
        if not os.path.exists(data_file):
            data_file = "dataset.csv"
        df = pd.read_csv(data_file)
        features = [
            'fever', 'cough', 'headache', 'fatigue', 'chest_pain', 
            'shortness_of_breath', 'nausea', 'fever_severity', 
            'cough_severity', 'headache_severity', 'fatigue_severity', 
            'chest_pain_severity', 'sob_severity', 'nausea_severity', 
            'num_symptoms', 'age'
        ]
        le = LabelEncoder()
        y = le.fit_transform(df['disease'])
        X = df[features]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = joblib.load(CFG.model_path)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        return jsonify({
            "status": "ok",
            "accuracy": round(acc * 100, 1),
            "precision": round(prec * 100, 1)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/graph-data", methods=["GET"])
def graph_data():
    """
    Returns latest graph payload.
    For simplicity, this route triggers a fresh lightweight predict call.
    """
    print("[API] /graph-data called")
    try:
        conn = connect_tigergraph(CFG)
        result = execute_prediction_pipeline(conn, CFG)
        sample_predictions = result.get("sample_predictions", [])
        nodes = _build_graph_nodes(sample_predictions)
        edges = _build_graph_edges(nodes)
        return jsonify({"status": "ok", "nodes": nodes, "edges": edges}), 200
    except Exception as exc:
        print(f"[API] /graph-data failed: {exc}")
        try:
            import json as _json
            import os as _os
            from tigergraph_ml_integration import (
                _normalize_records,
                load_or_create_model,
                preprocess_features,
                run_predictions,
            )

            cache_path = _os.path.join(_os.path.dirname(__file__) or ".", "tg_features_cache.json")
            if _os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    cached = _json.load(f)
                records = cached[0].get("patients", []) if cached else []
                df = _normalize_records(records)
            else:
                df = build_dummy_dataframe(CFG.feature_columns, rows=8)

            ids, X = preprocess_features(
                df=df,
                feature_columns=CFG.feature_columns,
                id_column=CFG.id_column,
                scaler_path=CFG.scaler_path,
            )
            model, le, _ = load_or_create_model(CFG.model_path, CFG.feature_columns)
            labels, scores = run_predictions(model, X, le=le)

            sample_predictions: List[Dict[str, Any]] = []
            for i, (pid, label, score) in enumerate(zip(ids.tolist(), labels, scores)):
                if i >= 25:
                    break
                sample_predictions.append({"id": pid, "prediction": str(label), "score": score})

            nodes = _build_graph_nodes(sample_predictions)
            edges = _build_graph_edges(nodes)
            return (
                jsonify(
                    {
                        "status": "fallback",
                        "nodes": nodes,
                        "edges": edges,
                        "message": "Using cached TigerGraph data.",
                    }
                ),
                200,
            )
        except Exception as fallback_exc:
            return (
                jsonify(
                    {
                        "status": "error",
                        "nodes": [],
                        "edges": [],
                        "message": f"Fallback graph failed: {fallback_exc}",
                    }
                ),
                200,
            )

if __name__ == "__main__":
    print("[API] Starting Flask server on http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=True)
