import pyTigerGraph as tg
host = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
graph="HealthcareGraph"
secret="Du0A~MWcGdtWXyLc2EbFRmMQmCuKz4J5kLUQh_KO"

try:
    print("Testing init without tokens, then getting token")
    # Use empty password and secret if possible?
    conn = tg.TigerGraphConnection(host=host, graphname=graph)
    conn.apiToken = secret # directly set it?
    # or use conn.getToken(secret)?
    try:
        print("Version:", conn.getVersion())
        print("Got token:", conn.getToken(secret))
    except Exception as e:
        print("Failed getting token:", e)
except Exception as e:
    import traceback
    traceback.print_exc()
