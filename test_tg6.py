import pyTigerGraph as tg
host = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
graph="HealthcareGraph"
secret="8rlhu62susejkmrlpnpvd1875fcmultr"

try:
    print("Testing 0.9.2 connection...")
    conn = tg.TigerGraphConnection(host=host, graphname=graph, gsqlSecret=secret)
    token = conn.getToken(secret)
    print("Got token:", token)
except Exception as e:
    import traceback
    traceback.print_exc()
