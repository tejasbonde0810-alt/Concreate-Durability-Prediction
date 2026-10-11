
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "models" / "concrete_model.joblib"

if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run 'python train_model.py' first."
    )

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
FEATURES = artifact["features"]
METRICS = {
    "r2_score": None,
    "mse": None,
    "mae": None
}

app = Flask(__name__)

# Broad input bounds for basic data validation, not engineering approval.
MAX_VALUES = {
    "cement": 1200,
    "slag": 1200,
    "fly_ash": 1200,
    "water": 600,
    "superplasticizer": 200,
    "coarse_aggregate": 2000,
    "fine_aggregate": 2000,
    "age": 365,
}

@app.get("/")
def home():
    return render_template("index.html", metrics=METRICS)

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Please submit valid JSON data."}), 400

    values = {}

    for feature in FEATURES:
        try:
            value = float(payload[feature])
        except (KeyError, TypeError, ValueError):
            return jsonify({
                "error": f"Enter a valid number for {feature.replace('_', ' ')}."
            }), 400

        if not pd.notna(value) or value < 0 or value > MAX_VALUES[feature]:
            return jsonify({
                "error": (
                    f"{feature.replace('_', ' ').title()} must be between "
                    f"0 and {MAX_VALUES[feature]}."
                )
            }), 400

        values[feature] = value

    if values["age"] < 1:
        return jsonify({"error": "Curing age must be at least 1 day."}), 400

    if values["cement"] <= 0:
        return jsonify({"error": "Cement must be greater than zero."}), 400

    # The model was trained on kg/m³ for ingredients and days for age.
    input_data = pd.DataFrame([values], columns=FEATURES)
    strength = float(model.predict(input_data)[0])

    return jsonify({
        "predicted_strength_mpa": round(strength, 2),
        "metrics": METRICS,
        "target": "Concrete compressive strength (MPa)",
        "disclaimer": (
            "This is a machine-learning estimate of concrete strength, "
            "not a prediction of house lifespan or structural safety."
        ),
    })

if __name__ == "__main__":
    app.run(debug=True)
