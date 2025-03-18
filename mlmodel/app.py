from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
import time
import subprocess

app = Flask(__name__)
CORS(app)

# File paths
URL_JSON_PATH = "url.json"
RESULTS_JSON_PATH = "analysis.json"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    video_url = data.get("url")

    if not video_url:
        return jsonify({"error": "Missing YouTube URL"}), 400

    # Save URL to JSON file
    with open(URL_JSON_PATH, 'w') as f:
        json.dump({"url": video_url}, f)
    
    # Run the model script as a subprocess
    try:
        subprocess.run(["python", "model.py"], check=True)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Error running analysis model"}), 500
    
    # Wait for results file to be created (with timeout)
    max_wait = 60  # Maximum wait time in seconds
    start_time = time.time()
    while not os.path.exists(RESULTS_JSON_PATH) and time.time() - start_time < max_wait:
        time.sleep(1)
    
    if not os.path.exists(RESULTS_JSON_PATH):
        return jsonify({"error": "Analysis results not found"}), 500
    
    # Read results from JSON file
    try:
        with open(RESULTS_JSON_PATH, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid results data"}), 500
    except Exception as e:
        return jsonify({"error": f"Error reading results: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)