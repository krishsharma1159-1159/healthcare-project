"""Extract cached MCP data and save as local JSON."""
import json
import os

# Read MCP output file
mcp_file = r"C:\Users\krish\.gemini\antigravity\brain\f55d0614-e138-42e9-abce-ca0a09853126\.system_generated\steps\514\output.txt"

with open(mcp_file, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# Find JSON block
start = content.find("{")
brace_count = 0
end = start
for i in range(start, len(content)):
    if content[i] == "{":
        brace_count += 1
    elif content[i] == "}":
        brace_count -= 1
    if brace_count == 0:
        end = i + 1
        break

json_str = content[start:end]
data = json.loads(json_str)
result = data["data"]["result"]

# Save just the result portion
out_path = os.path.join(os.path.dirname(__file__), "tg_features_cache.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

patients = result[0]["patients"]
print(f"Saved {len(patients)} patient records to tg_features_cache.json")
for p in patients[:5]:
    pid = p["v_id"]
    disease = p["attributes"].get("patients.@disease_name", "?")
    print(f"  {pid}: disease={disease}")
