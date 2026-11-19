"""Train and compare two simple models for human vs. bot text.

Usage:
    python src/train.py                          # uses data/sample.csv
    python src/train.py --data data/raw/my.csv   # use your own dataset
"""
import argparse
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import load_data

LABELS = ["human", "bot"]


def build_models():
    """Two classic baselines: TF-IDF features + a linear classifier."""
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Loaded {len(df)} rows ({(df.label == 0).sum()} human, {(df.label == 1).sum()} bot)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    best_name, best_model, best_acc = None, None, -1
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, target_names=LABELS, zero_division=0))
        print("Confusion matrix (rows = true, cols = predicted):")
        print(confusion_matrix(y_test, preds))
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(best_model, args.out)
    print(f"\nBest model: {best_name} ({best_acc:.3f}) saved to {args.out}")


if __name__ == "__main__":
    main()
