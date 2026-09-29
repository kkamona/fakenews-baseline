"""TF-IDF baselines: the main LinearSVC pipeline and a multi-model comparison (notebooks 03/03b)."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_vectorizer(
    max_features: int = 50000,
    min_df: float = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """TF-IDF settings used throughout the thesis (uni+bigrams, sublinear tf, English stop words)."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )


def build_pipeline(**vectorizer_kwargs) -> Pipeline:
    """Main baseline: TF-IDF followed by a class-balanced LinearSVC."""
    return Pipeline(
        [
            ("tfidf", build_vectorizer(**vectorizer_kwargs)),
            ("svm", LinearSVC(class_weight="balanced")),
        ]
    )


def make_classifiers(seed: int = 42) -> dict:
    """The four classical classifiers compared in notebook 03b."""
    return {
        "TFIDF_LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga"),
        "TFIDF_LinearSVC": LinearSVC(class_weight="balanced"),
        "TFIDF_MultinomialNB": MultinomialNB(),
        "TFIDF_SGDLinear": SGDClassifier(
            loss="hinge", max_iter=2000, class_weight="balanced", random_state=seed
        ),
    }


def compare_baselines(
    X_train, y_train, X_val, y_val, fake_label: int = 1, seed: int = 42, **vectorizer_kwargs
) -> pd.DataFrame:
    """Fit each classifier on a shared TF-IDF matrix; return validation metrics sorted by macro F1."""
    vec = build_vectorizer(**vectorizer_kwargs)
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    rows = []
    for name, clf in make_classifiers(seed).items():
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xva)
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, pred, average="binary", pos_label=fake_label, zero_division=0
        )
        rows.append({
            "model": name, "split": "val",
            "accuracy": accuracy_score(y_val, pred),
            "macro_f1": f1_score(y_val, pred, average="macro"),
            "weighted_f1": f1_score(y_val, pred, average="weighted"),
            "precision_fake(1)": p, "recall_fake(1)": r, "f1_fake(1)": f1,
        })
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
