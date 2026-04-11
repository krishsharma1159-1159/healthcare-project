"""
Load patients from dataset.csv into TigerGraph HealthcareGraph.
Maps CSV columns to the graph's relational schema:
  Patient -(HAS_SYMPTOM)-> Symptom  (with severity attribute)
  Patient -(HAS_DISEASE)-> Disease
"""

import csv
import requests

# ── TigerGraph Configuration ──
HOST = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
GRAPH = "HealthcareGraph"
SECRET = "Du0A~MWcGdtWXyLc2EbFRmMQmCuKz4J5kLUQh_KO"

# Symptom name → Symptom vertex ID mapping (matches existing graph)
SYMPTOM_MAP = {
    "fever":                ("S1", "Fever"),
    "cough":                ("S2", "Cough"),
    "headache":             ("S3", "Headache"),
    "fatigue":              ("S4", "Fatigue"),
    "chest_pain":           ("S5", "Chest Pain"),
    "shortness_of_breath":  ("S6", "Shortness of Breath"),
    "nausea":               ("S7", "Nausea"),
}

# Severity column names mapping
SEVERITY_MAP = {
    "fever":                "fever_severity",
    "cough":                "cough_severity",
    "headache":             "headache_severity",
    "fatigue":              "fatigue_severity",
    "chest_pain":           "chest_pain_severity",
    "shortness_of_breath":  "sob_severity",
    "nausea":               "nausea_severity",
}

# Disease name → Disease vertex ID mapping
DISEASE_MAP = {
    "Flu":              "D1",
    "Common Cold":      "D2",
    "Pneumonia":        "D3",
    "Hypertension":     "D4",
    "Diabetes Type 2":  "D5",
    "Bronchitis":       "D6",
    # New diseases from dataset that need new vertices
    "COVID-19":         "D7",
    "Food_Poisoning":   "D8",
    "Migraine":         "D9",
}


def get_token():
    """Get auth token from TigerGraph."""
    url = f"{HOST}/restpp/token?graph={GRAPH}&secret={SECRET}"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if data.get("error"):
        raise Exception(f"Token error: {data.get('message')}")
    return data["token"]


def upsert_vertex(token, vtype, vid, attrs):
    """Upsert a single vertex via REST API."""
    url = f"{HOST}/restpp/graph/{GRAPH}/vertices/{vtype}/{vid}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {k: {"value": v} for k, v in attrs.items()}
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    return resp.json()


def upsert_edge(token, src_type, src_id, edge_type, tgt_type, tgt_id, attrs=None):
    """Upsert a single edge via REST API."""
    url = f"{HOST}/restpp/graph/{GRAPH}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    edge_attrs = {}
    if attrs:
        edge_attrs = {k: {"value": v} for k, v in attrs.items()}
    payload = {
        "edges": {
            src_type: {
                src_id: {
                    edge_type: {
                        tgt_type: {
                            tgt_id: edge_attrs
                        }
                    }
                }
            }
        }
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    return resp.json()


def main():
    token = get_token()
    print(f"Got auth token: {token[:20]}...")

    # First, create the new Disease vertices that don't exist yet
    new_diseases = {
        "D7": "COVID-19",
        "D8": "Food_Poisoning",
        "D9": "Migraine",
    }
    for did, dname in new_diseases.items():
        result = upsert_vertex(token, "Disease", did, {"name": dname})
        print(f"Created Disease {did} ({dname}): {result}")

    # Read CSV and load patients P6..P25 (first 20 new ones)
    with open("dataset.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Skip P1-P5 (already in graph), load P6-P25
    patients_to_load = [r for r in rows if r["patient_id"] in
                        [f"P{i}" for i in range(6, 26)]]

    print(f"\nLoading {len(patients_to_load)} patients...")
    
    for row in patients_to_load:
        pid = row["patient_id"]
        age = int(row["age"])
        gender = row["gender"]
        disease = row["disease"]

        # 1. Create Patient vertex
        result = upsert_vertex(token, "Patient", pid, {
            "name": f"Patient_{pid}",
            "age": age,
            "gender": gender,
        })
        print(f"\n{pid} (age={age}, {gender}): vertex={result.get('accepted_vertices', '?')}")

        # 2. Create HAS_SYMPTOM edges for each present symptom
        symptoms_added = []
        for symptom_col, (sid, sname) in SYMPTOM_MAP.items():
            if int(row.get(symptom_col, 0)) == 1:
                severity_col = SEVERITY_MAP[symptom_col]
                severity = int(row.get(severity_col, 0))
                upsert_edge(token, "Patient", pid, "HAS_SYMPTOM", "Symptom", sid,
                           {"severity": severity})
                symptoms_added.append(f"{sname}(sev={severity})")

        print(f"   Symptoms: {', '.join(symptoms_added) if symptoms_added else 'None'}")

        # 3. Create HAS_DISEASE edge
        disease_id = DISEASE_MAP.get(disease)
        if disease_id:
            upsert_edge(token, "Patient", pid, "HAS_DISEASE", "Disease", disease_id)
            print(f"   Disease: {disease} ({disease_id})")
        else:
            print(f"   Disease: {disease} (NOT MAPPED — skipped)")

    print("\n✅ Done loading patients!")


if __name__ == "__main__":
    main()
