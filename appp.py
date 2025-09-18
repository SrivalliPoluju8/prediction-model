import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS if needed
logging.basicConfig(level=logging.INFO)

# Load your trained model
MODEL_PATH = "linear_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")
    model = None

FEATURE_COUNT = 8

@app.route("/")
def indexx():
    return render_template("indexx.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        payload = request.get_json(force=True)
        features = payload.get("features")

        if features is None:
            return jsonify({"error": "JSON must include 'features' key"}), 400
        if not isinstance(features, list):
            return jsonify({"error": "'features' must be a list"}), 400
        if len(features) != FEATURE_COUNT:
            return jsonify({"error": f"'features' must be of length {FEATURE_COUNT}"}), 400
        if not all(isinstance(x, (int, float)) for x in features):
            return jsonify({"error": "All feature values must be numeric (int or float)"}), 400

        arr = np.array(features).reshape(1, -1)
        prediction = model.predict(arr)

        app.logger.info(f"Prediction input: {features} -> {prediction[0]}")
        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)



