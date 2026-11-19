"""Loading and cleaning the text data."""
import re

import pandas as pd


def clean_text(text):
    """Lowercase, remove links and extra spaces. Punctuation is kept on purpose:
    bots and humans use it differently, so it helps the model."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(path):
    """Read a CSV with columns 'text' and 'label' (0 = human, 1 = bot)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)
    return df
