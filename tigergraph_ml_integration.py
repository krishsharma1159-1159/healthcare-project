"""
TigerGraph + ML model integration pipeline.

Install dependencies:
    pip install pyTigerGraph pandas requests joblib numpy scikit-learn
Optional for API mode:
    pip install flask
Optional for Keras/TensorFlow models:
    pip install tensorflow

This script supports:
1) Fetching input rows from TigerGraph (vertex or installed query)
2) Preprocessing rows into model-ready features
3) Loading a trained model (.pkl/.joblib/.h5/.keras)
4) Running predictions
5) Writing predictions back to TigerGraph
6) Running a beginner-friendly terminal workflow via main()
7) (Optional) Triggering the workflow via Flask API
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests as _requests
from sklearn.ensemble import RandomForestClassifier


# ---------------------------
# Logging configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("tg-ml-integration")


@dataclass
class AppConfig:
    """Central configuration for TigerGraph and ML pipeline."""

    # TigerGraph credentials
    host: str = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
    graph_name: str = "HealthcareGraph"
    username: str = "krishsharma1159@gmail.com"
    password: str = ""
    apiToken: str = "Du0A~MWcGdtWXyLc2EbFRmMQmCuKz4J5kLUQh_KO"

    # Data source options
    source_mode: str = "query"  # "vertex" or "query"
    source_vertex_type: str = "Patient"
    source_query_name: str = "extract_ml_features"
    source_query_params: Dict[str, Any] = field(default_factory=dict)
    source_vertex_limit: int = 200

    # Input schema used by model — must match model.py FEATURE_COLUMNS exactly
    id_column: str = "id"
    feature_columns: List[str] = field(
        default_factory=lambda: [
            # 7 binary symptom flags
            "fever", "cough", "headache", "fatigue",
            "chest_pain", "shortness_of_breath", "nausea",
            # 7 severity scores
            "fever_severity", "cough_severity", "headache_severity",
            "fatigue_severity", "chest_pain_severity", "sob_severity",
            "nausea_severity",
            # aggregates & demographics
            "num_symptoms", "age",
        ]
    )

    # Model options
    model_path: str = "model.pkl"
    scaler_path: Optional[str] = None

    # Write-back options
    update_vertex_type: str = "Patient"
    prediction_label_attr: str = "predicted_disease"
    prediction_score_attr: str = "prediction_score"

    # Flask options
    flask_host: str = "127.0.0.1"
    flask_port: int = 8000
    flask_debug: bool = False


def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables with safe defaults."""
    cfg = AppConfig()

    cfg.host = os.getenv("TG_HOST", cfg.host)
    cfg.graph_name = os.getenv("TG_GRAPH_NAME", cfg.graph_name)
    cfg.username = os.getenv("TG_USERNAME", cfg.username)
    cfg.password = os.getenv("TG_PASSWORD", cfg.password)

    cfg.source_mode = os.getenv("SOURCE_MODE", cfg.source_mode)
    cfg.source_vertex_type = os.getenv("SOURCE_VERTEX_TYPE", cfg.source_vertex_type)
    cfg.source_query_name = os.getenv("SOURCE_QUERY_NAME", cfg.source_query_name)
    cfg.source_vertex_limit = int(os.getenv("SOURCE_VERTEX_LIMIT", str(cfg.source_vertex_limit)))

    feature_columns_env = os.getenv("FEATURE_COLUMNS", "")
    if feature_columns_env.strip():
        cfg.feature_columns = [x.strip() for x in feature_columns_env.split(",") if x.strip()]

    cfg.model_path = os.getenv("MODEL_PATH", cfg.model_path)
    cfg.scaler_path = os.getenv("SCALER_PATH", cfg.scaler_path) or None
    cfg.update_vertex_type = os.getenv("UPDATE_VERTEX_TYPE", cfg.update_vertex_type)
    cfg.prediction_label_attr = os.getenv("PREDICTION_LABEL_ATTR", cfg.prediction_label_attr)
    cfg.prediction_score_attr = os.getenv("PREDICTION_SCORE_ATTR", cfg.prediction_score_attr)

    cfg.flask_host = os.getenv("FLASK_HOST", cfg.flask_host)
    cfg.flask_port = int(os.getenv("FLASK_PORT", str(cfg.flask_port)))
    cfg.flask_debug = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}

    query_params_str = os.getenv("SOURCE_QUERY_PARAMS", "")
    if query_params_str.strip():
        try:
            cfg.source_query_params = json.loads(query_params_str)
            if not isinstance(cfg.source_query_params, dict):
                raise ValueError("SOURCE_QUERY_PARAMS must decode to object/dict.")
        except Exception as exc:
            logger.warning("Could not parse SOURCE_QUERY_PARAMS as JSON: %s", exc)

    return cfg


class TGRestConnection:
    """Lightweight REST-based TigerGraph connection (bypasses pyTigerGraph 2.x bugs)."""

    def __init__(self, host: str, graph: str, secret: Optional[str] = None, token: Optional[str] = None):
        self.host = host.rstrip("/")
        self.graph = graph
        self.secret = secret
        self.token = token
        
        if not self.token and self.secret:
            self.token = self._generate_token()
            
        self._headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _generate_token(self) -> str:
        """Exchange secret for a JWT token."""
        if not self.secret:
            raise ValueError("Secret is required to generate a token.")
            
        logger.info("Generating new JWT token for graph=%s", self.graph)
        try:
            # Standard Savanna/restpp token endpoint
            resp = _requests.post(
                f"{self.host}/restpp/requesttoken",
                json={"secret": self.secret, "graph": self.graph},
                timeout=15
            )
            
            if resp.status_code == 200:
                data = resp.json()
                return data.get("token", "")
            
            # If 404 or 500, check if workspace is stopped
            if resp.status_code == 500 and "Auto start is not enabled" in resp.text:
                raise ConnectionError("TigerGraph Workspace is STOPPED. Please start it in the Cloud Portal.")
                
            raise ConnectionError(f"Token generation failed ({resp.status_code}): {resp.text}")
        except _requests.exceptions.RequestException as e:
            raise ConnectionError(f"Network error during token generation: {e}")

    # --- helpers -------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict] = None, timeout: int = 30):
        url = f"{self.host}{path}"
        resp = _requests.get(url, headers=self._headers, params=params, timeout=timeout)
        if resp.status_code == 401:
            logger.warning("Token expired or invalid, attempting to refresh...")
            if self.secret:
                self.token = self._generate_token()
                self._headers["Authorization"] = f"Bearer {self.token}"
                resp = _requests.get(url, headers=self._headers, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json_body: Any = None, timeout: int = 30):
        url = f"{self.host}{path}"
        resp = _requests.post(url, headers=self._headers, json=json_body, timeout=timeout)
        if resp.status_code == 401:
            logger.warning("Token expired or invalid, attempting to refresh...")
            if self.secret:
                self.token = self._generate_token()
                self._headers["Authorization"] = f"Bearer {self.token}"
                resp = _requests.post(url, headers=self._headers, json=json_body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # --- public wrappers -----------------------------------------------
    def getVertices(self, vertex_type: str, limit: int = 200):
        data = self._get(f"/restpp/graph/{self.graph}/vertices/{vertex_type}", params={"limit": limit})
        return data.get("results", [])

    def runInstalledQuery(self, query_name: str, params: Optional[Dict] = None):
        url_params = params or {}
        data = self._get(f"/restpp/query/{self.graph}/{query_name}", params=url_params)
        return data.get("results", [])

    def upsertVertex(self, vertex_type: str, vertex_id: str, attrs: Dict[str, Any]):
        payload = {"vertices": {vertex_type: {vertex_id: {}}}}
        for k, v in attrs.items():
            payload["vertices"][vertex_type][vertex_id][k] = {"value": v}
        return self._post(f"/restpp/graph/{self.graph}", json_body=payload)


def connect_tigergraph(cfg: AppConfig) -> TGRestConnection:
    """Create TigerGraph connection via REST API with JWT auth."""
    logger.info("Connecting to TigerGraph host=%s graph=%s", cfg.host, cfg.graph_name)
    host = cfg.host.rstrip("/")

    # Step 1: Check if server is alive and handle stopped workspace error
    try:
        ping_resp = _requests.get(f"{host}/api/ping", timeout=5)
        if ping_resp.status_code == 500 and "Auto start is not enabled" in ping_resp.text:
            raise ConnectionError("CRITICAL: TigerGraph Workspace is currently STOPPED. Please go to the TigerGraph Cloud portal and start your workspace.")
    except _requests.exceptions.RequestException as e:
        logger.warning("Ping failed, server might be offline: %s", e)

    # Step 2: Preferred method for Savanna - generate token from secret
    if cfg.apiToken:
        try:
            conn = TGRestConnection(host, cfg.graph_name, secret=cfg.apiToken)
            # Quick smoke-test
            conn._get(f"/restpp/graph/{cfg.graph_name}/vertices/Patient", params={"limit": 1})
            logger.info("Connected successfully via Secret -> JWT exchange.")
            return conn
        except Exception as exc:
            logger.warning("Secret exchange failed: %s", exc)
            if "STOPPED" in str(exc):
                raise exc

    # Step 3: Try GSQL auth endpoint
    try:
        resp = _requests.post(
            f"{host}/gsqlserver/gsql/authtoken",
            json={"username": cfg.username, "password": cfg.password},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("token") or data.get("access_token", "")
            if token:
                conn = TGRestConnection(host, cfg.graph_name, token=token)
                logger.info("Connected via GSQL JWT auth.")
                return conn
    except Exception as exc:
        logger.debug("GSQL auth attempt failed: %s", exc)

    # Final Fallback: use secret directly (older systems)
    if cfg.apiToken:
        logger.info("Falling back to using secret directly as bearer token.")
        try:
            conn = TGRestConnection(host, cfg.graph_name, token=cfg.apiToken)
            conn._get(f"/restpp/graph/{cfg.graph_name}/vertices/Patient", params={"limit": 1})
            return conn
        except Exception as exc:
            logger.error("All TigerGraph auth methods failed. Last error: %s", exc)
            raise ConnectionError(
                "Cannot authenticate to TigerGraph. 1) Ensure workspace is STARTED. 2) Check Secret/API Token."
            ) from exc
    
    raise ConnectionError("No credentials (apiToken or username/password) provided for TigerGraph.")


# Sentinel value TigerGraph MaxAccum<INT> uses when no value was accumulated
_TG_INT_MIN = -9223372036854775808


def _normalize_records(raw_records: Iterable[Dict[str, Any]], feature_cols: List[str] = []) -> pd.DataFrame:
    """
    Normalize pyTigerGraph / GSQL query records into a flat DataFrame.

    Handles:
    - GSQL attribute prefix stripping  ("patients.@fever" → "fever")
    - MaxAccum INT_MIN sentinel → 0
    - Accumulator '@' prefix removal    ("@num_symptoms" → "num_symptoms")
    - Feature column defaulting to 0 for missing columns
    """
    rows: List[Dict[str, Any]] = []

    # Handle pyTigerGraph's typical return nesting { 'results': [ ... ] }
    if isinstance(raw_records, dict) and "results" in raw_records:
        items = raw_records["results"]
    elif isinstance(raw_records, list):
        items = raw_records
    else:
        items = [raw_records]

    for rec in items:
        row: Dict[str, Any] = {}
        if not isinstance(rec, dict):
            continue

        # Extract ID
        if "v_id" in rec:
            row["id"] = rec["v_id"]
        elif "id" in rec:
            row["id"] = rec["id"]

        # Extract attributes or values
        attrs = rec.get("attributes", rec)
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                # Strip GSQL prefix, e.g. "patients.@fever" → "@fever"
                clean_k = str(k)
                if "." in clean_k:
                    clean_k = clean_k.split(".", 1)[1]

                # Strip accumulator '@' prefix: "@fever" → "fever"
                clean_k = clean_k.lstrip("@")

                # Rename disease_name to actual_disease (label, not a feature)
                if clean_k == "disease_name":
                    row["actual_disease"] = v if v else "Unknown"
                    continue

                # Clamp TigerGraph's MaxAccum INT_MIN sentinel to 0
                if isinstance(v, int) and v == _TG_INT_MIN:
                    v = 0

                row[clean_k] = v

        # Ensure every expected feature column exists (default 0)
        for col in feature_cols:
            if col not in row:
                row[col] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def fetch_data_from_vertex(
    conn: TGRestConnection,
    vertex_type: str,
    limit: int = 200,
) -> pd.DataFrame:
    """Fetch vertex rows from TigerGraph."""
    logger.info("Fetching data from vertex type=%s limit=%s", vertex_type, limit)
    records = conn.getVertices(vertex_type, limit=limit)
    df = _normalize_records(records)
    logger.info("Fetched %d rows from vertex.", len(df))
    return df


def fetch_data_from_query(
    conn: TGRestConnection,
    query_name: str,
    params: Optional[Dict[str, Any]] = None,
    feature_cols: List[str] = [],
) -> pd.DataFrame:
    """
    Fetch rows from installed TigerGraph query.
    Expects the query to return JSON serializable rows.
    """
    params = params or {}
    logger.info("Running installed query=%s params=%s", query_name, params)
    result = conn.runInstalledQuery(query_name, params=params)
    
    if not result:
        return pd.DataFrame()

    # DEBUG: Print result structure
    logger.info("Raw query result type: %s", type(result))
    if isinstance(result, list) and len(result) > 0:
        logger.info("First element keys: %s", result[0].keys() if isinstance(result[0], dict) else "not a dict")

    # The result usually has a shape like [{'patients': [...]}] or just [...]
    # We want to extract the patient list.
    extracted_records: List[Dict[str, Any]] = []
    if isinstance(result, list):
        for entry in result:
            if isinstance(entry, dict):
                # Check for query output blocks like 'patients'
                for val in entry.values():
                    if isinstance(val, list):
                        extracted_records.extend(val)
                    elif isinstance(val, dict):
                        extracted_records.append(val)
    
    if not extracted_records and isinstance(result, list):
        extracted_records = result

    df = _normalize_records(extracted_records, feature_cols=feature_cols)
    logger.info("Fetched %d rows from query result.", len(df))
    if not df.empty:
        logger.info("DataFrame head:\n%s", df.head().to_string())
    return df


def preprocess_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    id_column: str = "id",
    scaler_path: Optional[str] = None,
) -> Tuple[pd.Series, np.ndarray]:
    """Convert raw TigerGraph rows into model-ready array."""
    if df.empty:
        raise ValueError("No rows available to preprocess.")

    missing_cols = [c for c in feature_columns if c not in df.columns]
    for col in missing_cols:
        # Fill absent features with zeros to keep model input shape stable.
        df[col] = 0

    X_df = df[feature_columns].copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if id_column in df.columns:
        ids = df[id_column].astype(str)
    else:
        ids = pd.Series([str(i) for i in range(len(df))], name=id_column)

    X = X_df.to_numpy(dtype=np.float32)

    if scaler_path:
        scaler_obj = joblib.load(scaler_path)
        X = scaler_obj.transform(X)

    return ids, X


def load_trained_model(model_path: str):
    """Load sklearn or Keras model based on extension."""
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = model_path_obj.suffix.lower()
    if suffix in {".pkl", ".joblib"}:
        return joblib.load(model_path_obj)

    if suffix in {".h5", ".keras"}:
        try:
            from tensorflow.keras.models import load_model  # type: ignore
        except Exception as exc:
            raise ImportError(
                "TensorFlow is required for .h5/.keras models. Install with: pip install tensorflow"
            ) from exc
        return load_model(model_path_obj)

    raise ValueError(f"Unsupported model format: {suffix}")


def build_dummy_dataframe(feature_columns: List[str], rows: int = 8) -> pd.DataFrame:
    """Create a small fallback dataset when TigerGraph is not reachable."""
    rng = np.random.default_rng(42)
    data: Dict[str, Any] = {"id": [f"dummy_{i+1}" for i in range(rows)]}
    for col in feature_columns:
        # Binary symptoms/features (0/1)
        data[col] = rng.integers(0, 2, size=rows)
    return pd.DataFrame(data)


def load_or_create_model(
    model_path: str,
    feature_columns: List[str],
):
    """
    Load an existing model. If missing/invalid, create and persist a simple fallback model.
    """
    try:
        model = load_trained_model(model_path)
        # Try to load label encoder if it exists
        le_path = Path(model_path).parent / "label_encoder.pkl"
        le = None
        if le_path.exists():
            le = joblib.load(le_path)
        return model, le, f"Loaded model from '{model_path}'."
    except Exception as exc:
        logger.warning("Model load failed, creating dummy sklearn model: %s", exc)

    # Synthetic training data for fallback model.
    n_features = len(feature_columns)
    X_train = np.array(
        [
            [0] * n_features,
            [1] + [0] * (n_features - 1),
            [0, 1] + [0] * max(0, n_features - 2),
            [1, 1] + [0] * max(0, n_features - 2),
            [1] * n_features,
        ],
        dtype=np.float32,
    )
    y_train = np.array(["Healthy", "Flu", "Cold", "Flu", "Severe"], dtype=object)

    dummy_model = RandomForestClassifier(n_estimators=60, random_state=42)
    dummy_model.fit(X_train, y_train)

    # Save fallback model so next run does not retrain.
    fallback_path = Path(model_path)
    if fallback_path.suffix.lower() not in {".pkl", ".joblib"}:
        fallback_path = Path("dummy_model.pkl")
    joblib.dump(dummy_model, fallback_path)

    message = (
        f"Model file was missing/invalid. Created fallback model and saved to '{fallback_path}'."
    )
    return dummy_model, True, message


def run_predictions(model: Any, X: np.ndarray, le: Any = None) -> Tuple[List[Any], List[Optional[float]]]:
    """Run model inference and return labels plus confidence score when available."""
    if X.size == 0:
        return [], []

    # Standard prediction
    raw_pred = model.predict(X)
    if isinstance(raw_pred, np.ndarray):
        raw_pred = raw_pred.tolist()

    labels: List[Any] = []
    if raw_pred and isinstance(raw_pred[0], (list, tuple, np.ndarray)):
        # Keras multi-class style output probabilities
        arr = np.array(raw_pred)
        labels = arr.argmax(axis=1).tolist()
    else:
        labels = list(raw_pred)

    scores: List[Optional[float]] = [None] * len(labels)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        probs_arr = np.array(probs)
        if probs_arr.ndim == 2:
            scores = probs_arr.max(axis=1).astype(float).tolist()
    elif raw_pred and isinstance(raw_pred[0], (list, tuple, np.ndarray)):
        arr = np.array(raw_pred)
        if arr.ndim == 2:
            scores = arr.max(axis=1).astype(float).tolist()
    
    # If a label encoder is provided, decode numeric predictions
    if le is not None:
        # Check if the labels are actually numeric before transforms
        if len(labels) > 0 and isinstance(labels[0], (int, np.integer)):
            labels = le.inverse_transform(labels).tolist()

    return labels, scores


def print_terminal_report(
    records_fetched: int,
    sample_predictions: List[Dict[str, Any]],
    data_source_message: str,
    model_message: str,
    write_message: str,
    notes: List[str],
) -> None:
    """Print a clean beginner-friendly terminal summary."""
    print("\n=== TigerGraph + ML Prediction Run ===")
    print(f"Data source: {data_source_message}")
    print(f"Model status: {model_message}")
    print(f"Records fetched: {records_fetched}")
    print(f"Write-back: {write_message}")

    print("\nSample predictions:")
    if not sample_predictions:
        print("- No predictions generated.")
    else:
        for row in sample_predictions:
            print(
                f"- id={row['id']}, prediction={row['prediction']}, score={row['score']}"
            )

    if notes:
        print("\nNotes:")
        for msg in notes:
            print(f"- {msg}")
    print("======================================\n")


def write_predictions_to_tigergraph(
    conn: TGRestConnection,
    vertex_type: str,
    ids: Iterable[str],
    labels: Iterable[Any],
    scores: Iterable[Optional[float]],
    prediction_label_attr: str,
    prediction_score_attr: str,
) -> Dict[str, int]:
    """Upsert prediction fields to existing TigerGraph vertices."""
    success_count = 0
    fail_count = 0

    for item_id, label, score in zip(ids, labels, scores):
        attrs: Dict[str, Any] = {prediction_label_attr: str(label)}
        if score is not None:
            attrs[prediction_score_attr] = float(score)
        try:
            conn.upsertVertex(vertex_type, str(item_id), attrs)
            success_count += 1
        except Exception as exc:
            fail_count += 1
            logger.exception("Failed to upsert prediction for id=%s: %s", item_id, exc)

    logger.info("Write-back complete: success=%d fail=%d", success_count, fail_count)
    return {"success": success_count, "failed": fail_count}


def execute_prediction_pipeline(
    conn: TGRestConnection,
    cfg: AppConfig,
    source_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate full TigerGraph -> Model -> TigerGraph workflow."""
    mode = (source_mode or cfg.source_mode).lower().strip()
    if mode not in {"vertex", "query"}:
        raise ValueError("source_mode must be either 'vertex' or 'query'.")

    if mode == "vertex":
        df = fetch_data_from_vertex(conn, cfg.source_vertex_type, cfg.source_vertex_limit)
    else:
        df = fetch_data_from_query(conn, cfg.source_query_name, cfg.source_query_params, feature_cols=cfg.feature_columns)

    if df.empty:
        return {
            "status": "no_data",
            "message": "No rows returned from TigerGraph source.",
            "records_fetched": 0,
        }

    ids, X = preprocess_features(
        df=df,
        feature_columns=cfg.feature_columns,
        id_column=cfg.id_column,
        scaler_path=cfg.scaler_path,
    )
    model, le, _ = load_or_create_model(cfg.model_path, cfg.feature_columns)
    labels, scores = run_predictions(model, X, le=le)

    write_result = write_predictions_to_tigergraph(
        conn=conn,
        vertex_type=cfg.update_vertex_type,
        ids=ids.tolist(),
        labels=labels,
        scores=scores,
        prediction_label_attr=cfg.prediction_label_attr,
        prediction_score_attr=cfg.prediction_score_attr,
    )

    sample = []
    for i, (item_id, label, score) in enumerate(zip(ids.tolist(), labels, scores)):
        if i >= 5:
            break
        sample.append({"id": item_id, "prediction": label, "score": score})

    return {
        "status": "ok",
        "records_fetched": int(len(df)),
        "records_predicted": int(len(labels)),
        "write_result": write_result,
        "sample_predictions": sample,
    }


def create_app(cfg: AppConfig) -> Any:
    """Create Flask app with a single endpoint to trigger prediction."""
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.route("/predict-from-tigergraph", methods=["POST"])
    def predict_from_tigergraph() -> Any:
        """
        Optional request body:
        {
          "source_mode": "vertex" | "query"
        }
        """
        try:
            body = request.get_json(silent=True) or {}
            source_mode = body.get("source_mode")
            conn = connect_tigergraph(cfg)
            result = execute_prediction_pipeline(conn, cfg, source_mode=source_mode)
            return jsonify(result), 200
        except Exception as exc:
            logger.exception("Prediction pipeline failed: %s", exc)
            return jsonify({"status": "error", "message": str(exc)}), 500

    return app


def example_direct_function_calls(cfg: AppConfig) -> None:
    """Example for non-API usage."""
    try:
        conn = connect_tigergraph(cfg)
        result = execute_prediction_pipeline(conn, cfg)
        logger.info("Direct run result: %s", result)
    except Exception as exc:
        logger.exception("Direct run failed: %s", exc)


def main() -> None:
    """
    Beginner-friendly execution path:
    - Try TigerGraph fetch
    - Fall back to dummy DataFrame when needed
    - Try loading model
    - Fall back to dummy sklearn model when needed
    - Print terminal results directly
    """
    cfg = load_config_from_env()
    notes: List[str] = []

    conn: Optional[TGRestConnection] = None
    df = pd.DataFrame()
    used_dummy_data = False
    data_source_message = "TigerGraph"

    # Step 1: Connect and fetch data, else use fallback sample DataFrame.
    try:
        conn = connect_tigergraph(cfg)
        if cfg.source_mode.lower().strip() == "query":
            df = fetch_data_from_query(conn, cfg.source_query_name, cfg.source_query_params, feature_cols=cfg.feature_columns)
        else:
            df = fetch_data_from_vertex(conn, cfg.source_vertex_type, cfg.source_vertex_limit)

        if df.empty:
            used_dummy_data = True
            df = build_dummy_dataframe(cfg.feature_columns)
            data_source_message = "Fallback dummy data (TigerGraph returned no rows)"
            notes.append("TigerGraph returned no rows, so sample dummy rows were used.")
    except Exception as exc:
        notes.append(f"TigerGraph connection issue: {exc}")

        # Try local cache (populated via MCP or previous successful run)
        cache_path = os.path.join(os.path.dirname(__file__) or ".", "tg_features_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                records = cached[0].get("patients", []) if cached else []
                df = _normalize_records(records)
                if not df.empty:
                    data_source_message = "Local TigerGraph cache (tg_features_cache.json)"
                    notes.append("Using cached TigerGraph data (offline mode).")
                else:
                    raise ValueError("Cache file was empty")
            except Exception as cache_exc:
                notes.append(f"Cache load failed: {cache_exc}")
                used_dummy_data = True
                df = build_dummy_dataframe(cfg.feature_columns)
                data_source_message = "Fallback dummy data (cache + TigerGraph failed)"
        else:
            used_dummy_data = True
            df = build_dummy_dataframe(cfg.feature_columns)
            data_source_message = "Fallback dummy data (TigerGraph connection failed)"

    # Step 2: Preprocess features.
    ids, X = preprocess_features(
        df=df,
        feature_columns=cfg.feature_columns,
        id_column=cfg.id_column,
        scaler_path=cfg.scaler_path,
    )

    # Step 3: Load model, else train a fallback model.
    model, used_dummy_model, model_message = load_or_create_model(
        model_path=cfg.model_path,
        feature_columns=cfg.feature_columns,
    )
    if used_dummy_model:
        notes.append("Used fallback sklearn model for this run.")

    # Step 4: Predict.
    labels, scores = run_predictions(model, X)

    # Step 5: Write back to TigerGraph only when real TigerGraph data is used.
    write_message = "Skipped (running in fallback mode)."
    if conn is not None and not used_dummy_data:
        try:
            write_result = write_predictions_to_tigergraph(
                conn=conn,
                vertex_type=cfg.update_vertex_type,
                ids=ids.tolist(),
                labels=labels,
                scores=scores,
                prediction_label_attr=cfg.prediction_label_attr,
                prediction_score_attr=cfg.prediction_score_attr,
            )
            write_message = (
                f"Updated TigerGraph vertices: success={write_result['success']}, "
                f"failed={write_result['failed']}."
            )
        except Exception as exc:
            write_message = "Failed to write results back to TigerGraph."
            notes.append(f"Write-back issue: {exc}")

    sample_predictions: List[Dict[str, Any]] = []
    for idx, (item_id, label, score) in enumerate(zip(ids.tolist(), labels, scores)):
        if idx >= 5:
            break
        sample_predictions.append({"id": item_id, "prediction": label, "score": score})

    print_terminal_report(
        records_fetched=len(df),
        sample_predictions=sample_predictions,
        data_source_message=data_source_message,
        model_message=model_message,
        write_message=write_message,
        notes=notes,
    )


if __name__ == "__main__":
    # Default behavior: run pipeline directly in terminal (no Postman/API needed).
    main()
