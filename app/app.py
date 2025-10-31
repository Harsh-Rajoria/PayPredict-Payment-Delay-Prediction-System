from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import io
from werkzeug.utils import secure_filename
from model_loader import load_model
from utils import preprocess_input, TRAIN_FEATURES

app = Flask(__name__)
fusion_model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        features = preprocess_input(df)
        probs = fusion_model.predict_proba(features)[:, 1]
        preds = fusion_model.predict(features)
        results = [{"prediction": int(p), "probability": float(prob)} for p, prob in zip(preds, probs)]
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    filename = secure_filename(f.filename)
    try:
        df = pd.read_csv(io.BytesIO(f.read()))
        features = preprocess_input(df)
        probs = fusion_model.predict_proba(features)[:, 1]
        preds = fusion_model.predict(features)
        results = [{"prediction": int(p), "probability": float(prob)} for p, prob in zip(preds, probs)]
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
