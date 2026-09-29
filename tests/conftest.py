import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    rows = []
    for i in range(100):
        label = i % 2
        word = "hoax scandal" if label else "ministry report"
        rows.append({"Unnamed: 0": i, "title": f"Title {word} {i}",
                     "text": f"Body of article number {i} about {word} and other things", "label": label})
    return pd.DataFrame(rows)
