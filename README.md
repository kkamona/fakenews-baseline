# fakenews-baseline

[![CI](https://github.com/kkamona/fakenews-baseline/actions/workflows/ci.yml/badge.svg)](https://github.com/kkamona/fakenews-baseline/actions)
![License](https://img.shields.io/badge/license-green)

A small, tested Python toolkit extracted from my thesis on **fake-news detection on the WELFake dataset**.
It turns the exploratory notebooks (setup, preprocessing, de-duplication, TF-IDF baseline, metrics,
bootstrap confidence intervals, error analysis) into a reproducible package and command-line pipeline.

## Features
- Cleaning: builds `full_text = title + text`, handles missing values, drops very short rows.
- **Leakage-safe splitting:** exact-duplicate removal (MD5) *before* a stratified 70/15/15 split, plus an overlap check.
  (The raw WELFake data contains ~8.5k exact duplicates; splitting first put ~1.8k identical texts in both train and test.)
- Baseline model: TF-IDF (uni+bigrams) + class-balanced `LinearSVC` as a scikit-learn `Pipeline`.
- Baseline comparison: LogReg, LinearSVC, MultinomialNB and SGD on validation data (notebook 03/03b).
- Explainability: top tokens pushing towards fake/real from LinearSVC weights (notebook 08); helpers to merge BERT
  word-pieces and filter junk tokens in SHAP output (notebook 07).
- Linguistic cue statistics: causal, hedging and sensational cues per class (notebook 03b).
- Figures: confusion matrix and top-token bar charts.
- Metrics: accuracy, balanced accuracy, MCC, macro/weighted F1, precision/recall/F1 for the fake class, ROC-AUC, PR-AUC.
- 95 % bootstrap confidence intervals.
- Error analysis: most confident false positives / false negatives.

## Installation
```bash
git clone https://github.com/kkamona/fakenews-baseline.git
cd fakenews-baseline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage
Demo on the bundled synthetic sample (same schema as WELFake):
```bash
fakenews-baseline --raw data/sample/sample_news.csv --out-dir outputs --n-boot 200
```
On the real data, download `WELFake_Dataset.csv` from Kaggle into `data/raw/` and run:
```bash
fakenews-baseline --raw data/raw/WELFake_Dataset.csv --out-dir outputs
```
As a library:
```python
import pandas as pd
from fakenews_baseline import prepare_dataframe, split_dataset, build_pipeline, compute_metrics

df = prepare_dataframe(pd.read_csv("data/sample/sample_news.csv"))
train, val, test = split_dataset(df)
model = build_pipeline(min_df=1).fit(train["full_text"], train["label"])
pred, score = model.predict(test["full_text"]), model.decision_function(test["full_text"])
print(compute_metrics(test["label"], pred, score, "TFIDF_LinearSVC"))
```
Label convention: `1 = fake`, `0 = real` (as used throughout the thesis).

## Tests
```bash
pytest --cov=fakenews_baseline
ruff check .
```
The CLI also writes baseline comparison, top tokens, cue statistics and figures to `--out-dir`.

Tests cover data cleaning, de-duplication, split sizes/stratification/reproducibility, leakage detection,
metrics on known cases, bootstrap behaviour, error selection and an end-to-end CLI run.

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request: installs the package on
Python 3.10, 3.11 and 3.12, lints with ruff and runs pytest with coverage.

## Project structure
```
src/fakenews_baseline/   data.py, models.py, metrics.py, errors.py, explain.py, cues.py, plotting.py, cli.py
tests/                   pytest suite
data/sample/             small synthetic sample (real data is git-ignored)
notebooks/               original thesis notebooks (reference)
docs/                    technology justification, suggested issues
```

## Technology choices
See [docs/technology_justification.md](docs/technology_justification.md).

## Scope note
BERT fine-tuning (notebook 04), BERT inference (05, 06, 09) and SHAP computation (07) need a GPU, large model
weights and heavy dependencies (`torch`, `transformers`, `shap`). They stay in the notebooks and are deliberately
excluded from the tested package and CI. Their pure-Python post-processing (word-piece merging, junk filtering) and
all metrics are in the package and tested. The optional `bert` extra is reserved for future work.

## License
MIT, see [LICENSE](LICENSE).
