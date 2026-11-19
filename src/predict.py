"""Classify new text with the trained model.

Usage:
    python src/predict.py "your text here"
    python src/predict.py            # type sentences one by one, empty line to quit
"""
import sys

import joblib

from preprocess import clean_text

MODEL_PATH = "models/model.joblib"


def predict(model, text):
    probs = model.predict_proba([clean_text(text)])[0]
    label = "bot" if probs[1] >= 0.5 else "human"
    return label, max(probs)


def main():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        sys.exit("No trained model found. Run: python src/train.py")

    if len(sys.argv) > 1:
        texts = [" ".join(sys.argv[1:])]
    else:
        texts = iter(lambda: input("Text> ").strip(), "")

    for text in texts:
        label, conf = predict(model, text)
        print(f"{label.upper()} ({conf:.0%} confident)")


if __name__ == "__main__":
    main()
