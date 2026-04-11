import pyTigerGraph as tg
host = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
graph="HealthcareGraph"
secret="Du0A~MWcGdtWXyLc2EbFRmMQmCuKz4J5kLUQh_KO"
try:
    print("Testing tgCloud=True with gsqlSecret")
    conn = tg.TigerGraphConnection(host=host, graphname=graph, gsqlSecret=secret, tgCloud=True)
    res = conn.getToken(secret)
    print("Got token!", res)
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
