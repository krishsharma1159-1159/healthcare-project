import pyTigerGraph as tg
import os

host = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
graph="HealthcareGraph"
secret="Du0A~MWcGdtWXyLc2EbFRmMQmCuKz4J5kLUQh_KO"

try:
    conn = tg.TigerGraphConnection(host=host, graphname=graph)
    token = conn.getToken(secret, setToken=True)
    print("Successfully connected!")
    print("Vertices count:", conn.getVertexCount("Patient"))
except Exception as e:
    print(f"Connection failed: {e}")
