import pyTigerGraph as tg
host = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
graph="HealthcareGraph"
secret="8rlhu62susejkmrlpnpvd1875fcmultr"

try:
    print("Testing standard initialization...")
    conn = tg.TigerGraphConnection(host=host, graphname=graph, gsqlSecret=secret)
    token = conn.getToken(secret)
    print("Standard Init Token:", token)
except Exception as e:
    import traceback
    traceback.print_exc()

print("---")

try:
    print("Testing manual init...")
    conn2 = tg.TigerGraphConnection(host=host, graphname=graph)
    token2 = conn2.getToken(secret)
    print("Manual Init Token:", token2)
except Exception as e:
    import traceback
    traceback.print_exc()
