"""
app.py

Simple Flask web app: paste a job posting or recruiter message,
get a scam probability score and a breakdown of which red flags fired.

Run with: python app.py
Then visit http://127.0.0.1:5000
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request
from predict import predict, load_artifacts

app = Flask(__name__)

# Load model artifacts once at startup
try:
    MODEL, VECTORIZER, SCALER = load_artifacts()
except FileNotFoundError:
    MODEL, VECTORIZER, SCALER = None, None, None


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message_text = ""

    if request.method == "POST":
        message_text = request.form.get("message_text", "")
        if message_text.strip() and MODEL is not None:
            result = predict(message_text, MODEL, VECTORIZER, SCALER)

    return render_template("index.html", result=result, message_text=message_text)


if __name__ == "__main__":
    if MODEL is None:
        print("WARNING: No trained model found. Run `python src/train.py` first.")
    app.run(debug=True)
