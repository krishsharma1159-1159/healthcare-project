import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from predict import predict_disease
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes (to allow separate frontend hosting)
CORS(app)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

TG_URL = os.environ.get("TG_URL")
TG_TOKEN = os.environ.get("TG_TOKEN")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # 1. Request Validation
    if not data or "symptoms" not in data:
        return jsonify({"error": "Missing symptoms data in the request"}), 400

    symptoms = data["symptoms"]

    disease = predict_disease(symptoms)

    params = {"diseaseName": disease}
    headers = {
        "Authorization": f"Bearer {TG_TOKEN}"
    }

    try:
        response = requests.get(TG_URL, params=params, headers=headers)
        data = response.json()
        
        # Cleanly extract drug names from the TigerGraph json format
        if not data.get("error") and data.get("results"):
            drugs_raw = data["results"][0].get("drugs", [])
            drugs = [d["attributes"]["name"] for d in drugs_raw]
        else:
            drugs = {"error": "Query returned an error or no results", "details": data}
            
    except Exception as e:
        drugs = {"error": str(e)}

    return jsonify({
        "disease": disease,
        "drugs": drugs
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
