"""ML model for pharma-relevance classification with active learning support.

TF-IDF + Logistic Regression pipeline with versioned model storage.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import db

MODELS_DIR = Path(__file__).parent / "models"
MIN_TRAINING_SAMPLES = 50


def _ensure_models_dir():
    MODELS_DIR.mkdir(exist_ok=True)


def _get_next_version() -> int:
    _ensure_models_dir()
    existing = list(MODELS_DIR.glob("model_v*.pkl"))
    if not existing:
        return 1
    versions = [int(f.stem.split("_v")[1]) for f in existing]
    return max(versions) + 1


def get_latest_model_path() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    existing = list(MODELS_DIR.glob("model_v*.pkl"))
    if not existing:
        return None
    versions = [(int(f.stem.split("_v")[1]), f) for f in existing]
    return max(versions, key=lambda x: x[0])[1]


def build_pipeline() -> Pipeline:
    """Build the sklearn pipeline: TF-IDF on text + numeric features -> LogisticRegression."""
    text_transformer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", text_transformer, "text"),
            ("num", "passthrough", ["is_candidate", "word_count", "num_keywords"]),
        ]
    )

    pipeline = Pipeline([
        ("features", preprocessor),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
        )),
    ])
    return pipeline


def load_training_data() -> pd.DataFrame:
    """Load labelled rows with features needed for the model."""
    rows = db.export_labelled()
    df = pd.DataFrame(rows)
    df["num_keywords"] = df["matched_keywords"].apply(
        lambda x: len(json.loads(x)) if x else 0
    )
    return df


def train_model() -> dict:
    """Train a new model version with 80/20 stratified split. Returns metrics dict."""
    df = load_training_data()

    if len(df) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAINING_SAMPLES} labelled samples to train, have {len(df)}"
        )

    X = df[["text", "is_candidate", "word_count", "num_keywords"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Save model
    version = _get_next_version()
    _ensure_models_dir()
    model_path = MODELS_DIR / f"model_v{version}.pkl"
    joblib.dump(pipeline, model_path)
    metrics["model_path"] = str(model_path)
    metrics["version"] = version

    # Log to database
    run_id = db.log_model_run(
        num_samples=len(df),
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        model_path=str(model_path),
        confusion_matrix_data=metrics["confusion_matrix"],
    )
    metrics["run_id"] = run_id

    return metrics


def load_latest_model() -> Pipeline | None:
    """Load the latest model version from disk, or None if no model exists."""
    path = get_latest_model_path()
    if path is None:
        return None
    return joblib.load(path)


def get_latest_model_info() -> dict | None:
    """Get metadata about the latest model run from the DB."""
    return db.get_latest_model_run()


def predict_contribution(pipeline: Pipeline, contribution: dict) -> tuple[int, float]:
    """Predict on a single contribution. Returns (prediction, confidence).

    confidence = P(relevant), i.e. probability of class 1.
    """
    matched = contribution.get("matched_keywords", "[]") or "[]"
    num_keywords = len(json.loads(matched))
    row = pd.DataFrame([{
        "text": contribution["text"],
        "is_candidate": contribution["is_candidate"],
        "word_count": contribution["word_count"],
        "num_keywords": num_keywords,
    }])
    prediction = int(pipeline.predict(row)[0])
    confidence = float(pipeline.predict_proba(row)[0, 1])
    return prediction, confidence


def get_uncertainty_queue(pipeline: Pipeline) -> list[dict]:
    """Get unlabelled queue sorted by model uncertainty (closest to 0.5 first)."""
    rows = db.get_unlabelled_contributions()
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["num_keywords"] = df["matched_keywords"].apply(
        lambda x: len(json.loads(x)) if x else 0
    )
    X = df[["text", "is_candidate", "word_count", "num_keywords"]]
    probs = pipeline.predict_proba(X)[:, 1]
    order = np.argsort(np.abs(probs - 0.5))

    return [rows[i] for i in order]


def predict_all_unlabelled(pipeline: Pipeline, progress_callback=None) -> list[dict]:
    """Run model on all unlabelled contributions. Returns list of prediction dicts."""
    rows = db.get_unlabelled_contributions()
    if not rows:
        return []

    predictions = []
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        df = pd.DataFrame(batch)
        df["num_keywords"] = df["matched_keywords"].apply(
            lambda x: len(json.loads(x)) if x else 0
        )
        X = df[["text", "is_candidate", "word_count", "num_keywords"]]
        probs = pipeline.predict_proba(X)[:, 1]
        preds = pipeline.predict(X)

        for j, row in enumerate(batch):
            predictions.append({
                "id": row["id"],
                "prediction": int(preds[j]),
                "confidence": float(probs[j]),
                "house": row["house"],
            })

        if progress_callback:
            progress_callback(min((i + batch_size) / len(rows), 1.0))

    return predictions
