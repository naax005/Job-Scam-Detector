"""
train.py

Trains a Logistic Regression classifier on job posting text to detect
scam vs legit postings. Combines TF-IDF text features with hand-crafted
red-flag features from feature_engineering.py.
"""

import os
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

from feature_engineering import build_feature_frame

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "job_postings.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Engineered features get an explicit weight multiplier after scaling so
# strong, interpretable red flags (payment requests, sensitive info asks)
# aren't drowned out by the much larger TF-IDF vocabulary space, especially
# on a small training set.
ENGINEERED_FEATURE_WEIGHT = 3.0


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["label_bin"] = (df["label"] == "scam").astype(int)
    return df


def build_features(texts, vectorizer, scaler, fit=False):
    if fit:
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    engineered = build_feature_frame(texts)
    if fit:
        engineered_scaled = scaler.fit_transform(engineered.values)
    else:
        engineered_scaled = scaler.transform(engineered.values)

    engineered_sparse = csr_matrix(engineered_scaled * ENGINEERED_FEATURE_WEIGHT)
    combined = hstack([tfidf, engineered_sparse])
    return combined, engineered.columns.tolist()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label_bin"], test_size=0.25, random_state=42, stratify=df["label_bin"]
    )

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    scaler = StandardScaler()

    X_train, feature_names = build_features(X_train_text, vectorizer, scaler, fit=True)
    X_test, _ = build_features(X_test_text, vectorizer, scaler, fit=False)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["legit", "scam"]))

    joblib.dump(model, os.path.join(MODEL_DIR, "scam_classifier.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "feature_scaler.joblib"))
    print(f"\nModel + vectorizer + scaler saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
