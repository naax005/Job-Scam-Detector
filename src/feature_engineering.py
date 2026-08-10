"""
feature_engineering.py

Extracts hand-crafted "red flag" features from job posting text,
on top of raw text used for TF-IDF. These features encode common
scam patterns: urgency language, upfront payment requests, requests
for sensitive personal info, informal contact channels, and
excessive punctuation/emphasis.
"""

import re
import pandas as pd

# --- Keyword lexicons -------------------------------------------------

URGENCY_WORDS = [
    "urgent", "urgently", "immediately", "today", "asap", "hurry",
    "limited slots", "act now", "don't miss", "instant", "instantly",
    "guaranteed", "congratulations", "selected", "shortlisted", "won"
]

PAYMENT_WORDS = [
    "pay", "fee", "deposit", "registration fee", "processing fee",
    "activation fee", "joining fee", "security deposit", "gift card",
    "western union", "refundable", "transfer", "starter kit"
]

SENSITIVE_INFO_WORDS = [
    "bank account", "bank details", "aadhar", "ifsc", "passbook",
    "card details", "bank login", "otp", "pin number", "ssn",
    "account number"
]

INFORMAL_CHANNEL_WORDS = [
    "whatsapp", "telegram", "personal number", "text me", "dm me"
]

NO_PROCESS_WORDS = [
    "no interview", "no experience needed", "no resume", "no qualifications",
    "no skills required", "without an interview"
]


def _count_keyword_hits(text: str, keywords: list) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def extract_features(text: str) -> dict:
    """Extract a dict of engineered red-flag features for one job posting."""
    text_str = str(text)
    lower = text_str.lower()

    features = {
        "urgency_score": _count_keyword_hits(lower, URGENCY_WORDS),
        "payment_score": _count_keyword_hits(lower, PAYMENT_WORDS),
        "sensitive_info_score": _count_keyword_hits(lower, SENSITIVE_INFO_WORDS),
        "informal_channel_score": _count_keyword_hits(lower, INFORMAL_CHANNEL_WORDS),
        "no_process_score": _count_keyword_hits(lower, NO_PROCESS_WORDS),
        "exclamation_count": text_str.count("!"),
        "dollar_sign_count": text_str.count("$") + len(re.findall(r"rs\.?\s?\d", lower)),
        "all_caps_word_count": len(re.findall(r"\b[A-Z]{3,}\b", text_str)),
        "has_money_amount": int(bool(re.search(r"(\$\s?\d+|\brs\.?\s?\d+|\d+k\b)", lower))),
        "text_length": len(text_str.split()),
    }
    return features


def build_feature_frame(texts: pd.Series) -> pd.DataFrame:
    """Apply extract_features to a Series of texts and return a DataFrame."""
    records = [extract_features(t) for t in texts]
    return pd.DataFrame(records)


def explain_flags(text: str) -> list:
    """Return a human-readable list of which red flags fired for a given text,
    used to explain a prediction to the end user."""
    lower = str(text).lower()
    flags = []

    if _count_keyword_hits(lower, URGENCY_WORDS) > 0:
        flags.append("Uses urgency/pressure language (e.g. 'urgent', 'guaranteed', 'act now')")
    if _count_keyword_hits(lower, PAYMENT_WORDS) > 0:
        flags.append("Mentions an upfront payment, fee, or deposit")
    if _count_keyword_hits(lower, SENSITIVE_INFO_WORDS) > 0:
        flags.append("Requests sensitive personal/financial information (bank details, ID numbers)")
    if _count_keyword_hits(lower, INFORMAL_CHANNEL_WORDS) > 0:
        flags.append("Directs communication to informal channels (WhatsApp/Telegram) instead of official email")
    if _count_keyword_hits(lower, NO_PROCESS_WORDS) > 0:
        flags.append("Skips a normal hiring process (no interview, no resume, no experience needed)")
    if text.count("!") >= 2:
        flags.append("Excessive exclamation marks / hype language")
    if len(re.findall(r"\b[A-Z]{3,}\b", text)) >= 2:
        flags.append("Excessive use of ALL CAPS for emphasis")

    return flags


if __name__ == "__main__":
    sample = "URGENT!!! Congratulations, you are selected. Pay a $50 registration fee via Western Union to claim your work from home job today."
    print(extract_features(sample))
    print(explain_flags(sample))
