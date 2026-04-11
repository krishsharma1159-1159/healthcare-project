import pyTigerGraph as tg
host = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
graph="HealthcareGraph"
secret="8rlhu62susejkmrlpnpvd1875fcmultr"

try:
    print("Testing connection with the new secret...")
    conn = tg.TigerGraphConnection(host=host, graphname=graph, gsqlSecret=secret)
    token = conn.getToken(secret)
    print("Successfully authenticated!")
except Exception as e:
    print("Connection failed:", e)
