"""
preprocess.py - loading, cleaning and splitting the data.

Both models (the base model and the fine-tuned model) use the SAME
train/test split, so they are tested on exactly the same texts.

Run directly to print dataset statistics for the report:
    python src/preprocess.py --data data/sample.csv
"""

import argparse
import re

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42          # fixed seed so the split is identical every run
TEST_SIZE = 0.2    # 80% train / 20% held-out test


def clean_text(text):
    """Remove links and extra whitespace.
    Case and punctuation are kept on purpose: they are useful clues."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove links
    text = re.sub(r"\s+", " ", text)                 # collapse whitespace
    return text.strip()


def load_data(path, max_rows=None):
    """Read a CSV with a 'text' column and a 'label' or 'generated' column.
    label: 0 = human, 1 = AI.
    max_rows: optionally take a smaller, class-balanced sample (for quick runs)."""
    df = pd.read_csv(path)

    # The Kaggle dataset calls the label column 'generated'
    if "label" not in df.columns and "generated" in df.columns:
        df = df.rename(columns={"generated": "label"})

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    if max_rows is not None and max_rows < len(df):
        # take the same number of rows from each class
        per_class = max_rows // 2
        parts = []
        for label in [0, 1]:
            rows = df[df["label"] == label]
            parts.append(rows.sample(min(per_class, len(rows)), random_state=SEED))
        df = pd.concat(parts).reset_index(drop=True)
    return df


def split_data(df):
    """Stratified 80/20 split. Same result every run (seed 42)."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    df = load_data(args.data, args.max_rows)
    train_df, test_df = split_data(df)

    print(f"Total rows: {len(df)}")
    print(f"Human (0): {(df['label'] == 0).sum()}   AI (1): {(df['label'] == 1).sum()}")
    print(f"Train: {len(train_df)}   Test: {len(test_df)}")
    words = df["text"].str.split().str.len()
    print(f"Average words - human: {words[df['label'] == 0].mean():.1f}, "
          f"AI: {words[df['label'] == 1].mean():.1f}")
