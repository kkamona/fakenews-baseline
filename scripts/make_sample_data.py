"""Generate a tiny synthetic dataset (same schema as WELFake) for demos and tests."""
import random

import pandas as pd

random.seed(42)
FAKE = ["shocking", "unbelievable", "exposed", "secret", "hoax", "outrage", "scandal", "banned", "miracle", "conspiracy"]
REAL = ["ministry", "reported", "according", "statement", "officials", "budget", "committee", "agreement", "research"]
COMMON = ["the", "government", "new", "people", "said", "year", "city", "state", "president", "after", "new", "report"]

def make(i, label):
    vocab = FAKE if label == 1 else REAL
    words = [random.choice(vocab if random.random() < 0.35 else COMMON) for _ in range(40)]
    title = " ".join(random.choices(vocab, k=4)).upper() if label else " ".join(random.choices(vocab, k=4))
    return {"title": title, "text": f"Article {i}. " + " ".join(words), "label": label}

rows = [make(i, i % 2) for i in range(300)]
pd.DataFrame(rows).to_csv("data/sample/sample_news.csv", index=False)
