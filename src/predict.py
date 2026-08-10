"""
predict.py

Loads the trained model + vectorizer and scores a new piece of job
posting / message text, returning a scam probability and a list of
human-readable red flags that were detected.
"""

import os
import joblib
from scipy.sparse import hstack, csr_matrix

from feature_engineering import build_feature_frame, explain_flags

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ENGINEERED_FEATURE_WEIGHT = 3.0  # must match the value used in train.py


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "scam_classifier.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.joblib"))
    return model, vectorizer, scaler


def predict(text: str, model=None, vectorizer=None, scaler=None) -> dict:
    if model is None or vectorizer is None or scaler is None:
        model, vectorizer, scaler = load_artifacts()

    tfidf = vectorizer.transform([text])
    engineered = build_feature_frame([text])
    engineered_scaled = scaler.transform(engineered.values) * ENGINEERED_FEATURE_WEIGHT
    combined = hstack([tfidf, csr_matrix(engineered_scaled)])

    proba = model.predict_proba(combined)[0]
    scam_probability = float(proba[1])
    label = "scam" if scam_probability >= 0.5 else "legit"

    return {
        "label": label,
        "scam_probability": round(scam_probability, 4),
        "red_flags": explain_flags(text),
    }


if __name__ == "__main__":
    test_message = (
        "Congratulations! You are selected for a Work From Home Data Entry Job. "
        "Pay a small registration fee of $49 via Western Union to activate your account today. "
        "Contact us on WhatsApp immediately!!!"
    )
    result = predict(test_message)
    print(f"Prediction: {result['label']} (scam probability: {result['scam_probability']})")
    print("Red flags detected:")
    for flag in result["red_flags"]:
        print(f"  - {flag}")
