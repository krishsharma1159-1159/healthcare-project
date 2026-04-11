import pyTigerGraph as tg
host = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
graph="HealthcareGraph"
secret="Du0A~MWcGdtWXyLc2EbFRmMQmCuKz4J5kLUQh_KO"

try:
    print("Testing with both password and secret")
    conn = tg.TigerGraphConnection(host=host, graphname=graph, username="tigergraph", password="tigergraph", apiToken=secret)
    print("Version:", conn.getVersion())
except Exception as e:
    import traceback
    traceback.print_exc()
