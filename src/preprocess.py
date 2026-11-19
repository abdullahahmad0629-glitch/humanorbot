"""Step 1: load, clean and split the data.

Both models use the SAME train/test split (fixed seed), so the comparison is fair.

Usage (prints a summary of the dataset, useful for the report):
    python src/preprocess.py --data data/sample.csv
"""
import argparse
import re

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABEL_COLUMNS = ["label", "generated"]  # different Kaggle datasets use different names
LABEL_NAMES = {0: "human", 1: "ai"}


def clean_text(text):
    """Remove links and extra whitespace. Case and punctuation are kept on purpose:
    DistilBERT handles them itself, and they are useful clues for both models."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(path, max_rows=None):
    """Read a CSV with a 'text' column and a 'label' or 'generated' column
    (0 = human, 1 = AI). Optionally keep a random, class-balanced subset."""
    df = pd.read_csv(path)
    label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)
    if "text" not in df.columns or label_col is None:
        raise ValueError(f"CSV needs a 'text' column and one of {LABEL_COLUMNS}; found {list(df.columns)}")

    df = df[["text", label_col]].rename(columns={label_col: "label"}).dropna()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates("text")
    df["label"] = df["label"].astype(int)

    if max_rows and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, random_state=SEED, stratify=df["label"])
    return df.reset_index(drop=True)


def split_data(df, test_size=0.2):
    """80/20 stratified split, identical every run."""
    train, test = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def describe(df):
    counts = df["label"].map(LABEL_NAMES).value_counts()
    words = df["text"].str.split().str.len()
    print(f"Rows: {len(df)}")
    for name, n in counts.items():
        print(f"  {name}: {n} ({n / len(df):.1%})")
    print(f"Words per text: mean {words.mean():.0f}, median {words.median():.0f}, max {words.max()}")
    for label, name in LABEL_NAMES.items():
        print(f"  avg words ({name}): {words[df['label'] == label].mean():.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    data = load_data(args.data, args.max_rows)
    describe(data)
    train, test = split_data(data)
    print(f"Train: {len(train)}  Test: {len(test)}")
