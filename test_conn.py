"""Test JWT auth for TigerGraph Savanna."""
import requests, json

HOST = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
GRAPH = "HealthcareGraph"

# Try GSQL auth with the email/password credentials
# We'll test with empty password first, then need user to provide real password
creds_to_try = [
    {"username": "krishsharma1159@gmail.com", "password": ""},
    {"username": "tigergraph", "password": "tigergraph"},
]

for cred in creds_to_try:
    try:
        r = requests.post(
            f"{HOST}/gsqlserver/gsql/authtoken",
            json=cred,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        print(f"GSQL auth ({cred['username']}): {r.status_code} -> {r.text[:200]}")
    except Exception as e:
        print(f"GSQL auth ({cred['username']}): ERR -> {e}")

# Also test /api/ping to see what's available
try:
    r = requests.get(f"{HOST}/api/ping", timeout=10)
    print(f"\n/api/ping: {r.status_code} -> {r.text[:200]}")
except Exception as e:
    print(f"/api/ping: ERR -> {e}")
