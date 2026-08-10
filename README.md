# 🛡️ Job Scam Detector

> An NLP-based classifier that flags fraudulent job postings and recruiter messages, with an interpretable breakdown of *why* a message looks like a scam — not just a black-box label.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white">
  <img src="https://img.shields.io/badge/status-prototype-yellow?style=flat-square">
</p>

## 🎯 Why this project

Online job scams — fake "work from home" offers, upfront "registration fees," requests for bank details before any real interview — are common and costly. This project was born out of a real experience reporting a job scam, and turns that experience into a tool: paste in a job posting or recruiter message, and get back a scam probability *plus* the specific red flags that triggered it.

## 🧩 Problem Statement

Most people spot scams only after losing money or sharing sensitive information. This tool gives a fast, explainable second opinion before you engage — combining a machine learning classifier with transparent, rule-based red-flag detection so the output is trustworthy and actionable, not just a score.

## ⚙️ Approach

1. **Curated dataset** — hand-collected scam and legitimate job posting examples covering common Indian and international scam patterns (upfront fees, urgency language, requests for bank/Aadhar details, informal-only contact channels) vs. normal hiring language.
2. **Feature engineering** — beyond TF-IDF text vectors, extracts explicit red-flag signals: urgency score, payment-request score, sensitive-info-request score, informal-channel score, "no interview needed" score, exclamation/caps usage.
3. **Model** — Logistic Regression over combined TF-IDF + scaled engineered features, with `class_weight="balanced"` for the two classes.
4. **Explainability layer** — a separate rule-based `explain_flags()` function surfaces *which* red flags fired in plain English, alongside the model's probability score.
5. **Interactive app** — a small Flask UI to paste in any message and get an instant check.

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy (held-out split) | 1.00 |
| F1-score | 1.00 |

⚠️ **Known limitation:** the dataset is small (~49 examples) and cleanly separable, so these numbers are optimistic. In testing, single weak signals (e.g. the word "urgent" in an otherwise normal posting) can still trigger false positives. This is a solid proof-of-concept architecture — the real next step is scaling up the dataset with more diverse, ambiguous, real-world examples (e.g. from public scam-report datasets) to properly stress-test and calibrate it.

## 🛠️ Tech Stack

`Python` `scikit-learn` `pandas` `Flask` `TF-IDF` `Logistic Regression`

## 🚀 How to Run

```bash
git clone https://github.com/naax005/job-scam-detector.git
cd job-scam-detector
pip install -r requirements.txt

# Train the model
python src/train.py

# Run the web app
python app.py
# then visit http://127.0.0.1:5000
```

Or use it directly in Python:

```python
from src.predict import predict

result = predict("Congratulations! Pay a $49 registration fee via Western Union to activate your work from home job today!!!")
print(result["label"], result["scam_probability"])
print(result["red_flags"])
```

## 📁 Project Structure

```
job-scam-detector/
├── data/
│   └── job_postings.csv          # curated scam vs legit examples
├── src/
│   ├── feature_engineering.py    # red-flag feature extraction + explanations
│   ├── train.py                  # trains TF-IDF + Logistic Regression model
│   └── predict.py                # scores new text, returns probability + flags
├── models/                       # saved model, vectorizer, scaler (generated)
├── templates/
│   └── index.html                # Flask UI
├── app.py                        # Flask web app
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

- Scale up the dataset using public scam-report datasets and real scraped job postings for better generalization
- Add a browser extension so people can check messages without leaving LinkedIn/WhatsApp
- Try a transformer-based model (DistilBERT) for better handling of subtle/ambiguous phrasing
- Add domain/URL reputation checks for links included in messages
- Track false positive rate specifically on legitimate postings that use urgent-sounding but normal hiring language

## 👤 Author

**Alinas Ferdavus** — [LinkedIn](https://linkedin.com/in/alinas-ferdavus-567693264)

Built after personally encountering and reporting an online job scam via [cybercrime.gov.in](https://cybercrime.gov.in) — this project is part of a broader interest in applying data science to cybersecurity and consumer protection.
