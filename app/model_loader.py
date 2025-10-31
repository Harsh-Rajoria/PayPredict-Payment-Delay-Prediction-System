import joblib

MODEL_PATH = "../models/paypredict_fusion_model.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    return model
