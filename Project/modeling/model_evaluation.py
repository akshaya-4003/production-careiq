"""Scores fitted models several ways, always next to the baseline they must beat."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _round(value):
    return round(float(value), 4)


def probability_of_escalation(model, features):
    """The probability of the positive class for every row, each within [0, 1]."""
    probabilities = model.predict_proba(features)[:, 1]
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def evaluate_classification_model(model, features, target):
    """ROC AUC, PR AUC, accuracy, precision, recall, F1 and the positive rate."""
    truth = np.asarray(target).astype(int)
    probabilities = probability_of_escalation(model, features)
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": _round(roc_auc_score(truth, probabilities)),
        "pr_auc": _round(average_precision_score(truth, probabilities)),
        "accuracy": _round(accuracy_score(truth, predicted)),
        "precision": _round(precision_score(truth, predicted, zero_division=0)),
        "recall": _round(recall_score(truth, predicted, zero_division=0)),
        "f1_score": _round(f1_score(truth, predicted, zero_division=0)),
        "positive_rate": _round(truth.mean()),
        "rows": int(len(truth)),
    }


def evaluate_regression_model(model, features, target, baseline_value=None):
    """R squared, MAE, RMSE, and the MAE of always predicting the mean."""
    truth = np.asarray(target, dtype=float)
    predicted = np.asarray(model.predict(features), dtype=float)
    constant = float(truth.mean()) if baseline_value is None else float(baseline_value)
    return {
        "r2_score": _round(r2_score(truth, predicted)),
        "mean_absolute_error_hours": _round(mean_absolute_error(truth, predicted)),
        "root_mean_squared_error_hours": _round(np.sqrt(mean_squared_error(truth, predicted))),
        "baseline_mean_absolute_error_hours": _round(
            mean_absolute_error(truth, np.full_like(truth, constant))),
        "rows": int(len(truth)),
    }


def _text_scores(labels, predicted):
    labels = list(labels)
    return {
        "accuracy": _round(accuracy_score(labels, predicted)),
        "macro_f1_score": _round(f1_score(labels, predicted, average="macro", zero_division=0)),
        "weighted_f1_score": _round(f1_score(labels, predicted, average="weighted",
                                             zero_division=0)),
        "classes": int(len(set(labels))),
    }


def evaluate_text_classifier(model, document_matrix, labels):
    """Accuracy, macro F1 (the figure the choice is made on), weighted F1 and class count."""
    predicted = model.predict(document_matrix)
    scores = _text_scores(labels, predicted)
    scores["classes"] = int(len(getattr(model, "classes_", set(labels))))
    return scores


def majority_class_baseline(labels):
    """The same scores as if every prediction were the commonest label."""
    series = pd.Series(list(labels))
    commonest = series.value_counts().idxmax()
    scores = _text_scores(series, [commonest] * len(series))
    scores["majority_label"] = str(commonest)
    return scores


def compare_candidate_scores(evaluation_scores, primary_metric):
    """Rank every candidate by the primary metric, best first, and name the winner."""
    ranking = sorted(
        ({"candidate": name, primary_metric: scores.get(primary_metric, float("-inf")),
          "scores": scores} for name, scores in evaluation_scores.items()),
        key=lambda entry: entry[primary_metric], reverse=True)
    winner = ranking[0]["candidate"] if ranking else ""
    return ranking, winner


def save_evaluation_report(metrics, output_path):
    """Write every candidate's scores and the chosen model per task."""
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)
    return output_path
