from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import io
from werkzeug.utils import secure_filename
from model_loader import load_model
from utils import preprocess_input  # ✅ FIXED: Removed TRAIN_FEATURES import

app = Flask(__name__)

# Load trained fusion model
fusion_model = load_model()

@app.route('/')
def home():
    """Render the main web UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle JSON input for single or multiple records."""
    try:
        data = request.get_json(force=True)

        # Convert JSON into DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        # Preprocess input
        features = preprocess_input(df)

        # Predictions and probabilities
        probs = fusion_model.predict_proba(features)[:, 1]
        preds = fusion_model.predict(features)

        # Combine results
        results = [
            {"prediction": int(p), "probability": float(prob)}
            for p, prob in zip(preds, probs)
        ]

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_file', methods=['POST'])
def predict_file():
    """Handle CSV upload and batch predictions."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    filename = secure_filename(f.filename)

    try:
        # Read file and preprocess
        df = pd.read_csv(io.BytesIO(f.read()))
        features = preprocess_input(df)

        # Predictions and probabilities
        probs = fusion_model.predict_proba(features)[:, 1]
        preds = fusion_model.predict(features)

        # Combine results
        results = [
            {"prediction": int(p), "probability": float(prob)}
            for p, prob in zip(preds, probs)
        ]

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run Flask app
    app.run(host="0.0.0.0", port=7860, debug=True)