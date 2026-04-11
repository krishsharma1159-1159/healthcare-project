import pyTigerGraph as tg
host = "https://tg-916d9f11-c704-4456-ae6e-f46bd45d1764.tg-2635877100.i.tgcloud.io"
graph="HealthcareGraph"
secret="Du0A~MWcGdtWXyLc2EbFRmMQmCuKz4J5kLUQh_KO"

print("Connecting...")
conn = tg.TigerGraphConnection(host=host, graphname=graph)
token = conn.getToken(secret)
print("Token:", token)
conn.apiToken = token[0]  # depending on output format
conn.apiToken = token
print(conn.getVertices("Patient", limit=1))
