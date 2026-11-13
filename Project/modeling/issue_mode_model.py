"""The three-way bake-off for "which of the ten issue modes is this message about"."""

import time

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from config import config
from data_pipeline.message_cleaning import clean_message_text
from modeling.model_evaluation import (
    compare_candidate_scores,
    evaluate_text_classifier,
    majority_class_baseline,
)
from modeling.bakeoff import print_ranking
from modeling.model_registry import save_model_artifact


def issue_mode_candidates():
    seed = config.RANDOM_SEED
    return {
        "MultinomialNB": MultinomialNB(alpha=0.3),
        "LogisticRegression": LogisticRegression(max_iter=3000, C=4.0, random_state=seed),
        # Wrapped in calibration so the chosen model always reports its own probability.
        "LinearSVC": CalibratedClassifierCV(LinearSVC(random_state=seed), cv=3),
    }


def build_vectorizer():
    return TfidfVectorizer(**config.TEXT_VECTORIZER_SETTINGS)


def train_issue_mode_model(frame, log=print):
    """Vectorise once, fit three classifiers on that matrix, choose on macro F1, save both files."""
    # Training text is cleaned with exactly the function prediction uses. Running it
    # again over already-clean text is a no-op, and it makes a raw frame safe too.
    texts = frame[config.MESSAGE_TEXT_COLUMN].astype(str).map(clean_message_text)
    labels = frame[config.MESSAGE_TARGET_COLUMN].astype(str)

    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    log("    messages {0}, vocabulary {1}, modes {2}".format(
        matrix.shape[0], len(vectorizer.vocabulary_), labels.nunique()))

    train_x, holdout_x, train_y, holdout_y = train_test_split(
        matrix, labels, test_size=config.VALIDATION_SPLIT_RATIO, stratify=labels,
        random_state=config.RANDOM_SEED)
    folds = StratifiedKFold(n_splits=config.CROSS_VALIDATION_FOLDS, shuffle=True,
                            random_state=config.RANDOM_SEED)
    candidates = issue_mode_candidates()
    scores = {}
    for name, estimator in candidates.items():
        started = time.time()
        cv_values = cross_val_score(clone(estimator), train_x, train_y, cv=folds,
                                    scoring="f1_macro")
        fitted = clone(estimator).fit(train_x, train_y)
        elapsed = time.time() - started
        candidate_scores = {config.ISSUE_MODE_PRIMARY_METRIC: round(float(cv_values.mean()), 4),
                            "fit_seconds": round(elapsed, 2)}
        candidate_scores.update(evaluate_text_classifier(fitted, holdout_x, holdout_y))
        scores[name] = candidate_scores
        log("    trained {0:<30} in {1:>6.1f} seconds  macro F1 {2:.4f}".format(
            name, elapsed, candidate_scores[config.ISSUE_MODE_PRIMARY_METRIC]))
    ranking, winner = compare_candidate_scores(scores, config.ISSUE_MODE_PRIMARY_METRIC)
    print_ranking(ranking, [config.ISSUE_MODE_PRIMARY_METRIC, "accuracy", "macro_f1_score"], log)

    final_model = clone(candidates[winner]).fit(matrix, labels)
    training_scores = evaluate_text_classifier(final_model, matrix, labels)
    baseline = majority_class_baseline(labels)
    save_model_artifact(final_model, config.ISSUE_MODE_MODEL_FILE)
    save_model_artifact(vectorizer, config.TEXT_VECTORIZER_FILE)

    log("    best candidate         : " + winner)
    log("    held out macro F1      : {0:.4f}".format(scores[winner]["macro_f1_score"]))
    log("    training set macro F1  : {0:.4f}".format(training_scores["macro_f1_score"]))
    log("    always answering '{0}' scores macro F1 {1:.4f}".format(
        baseline["majority_label"], baseline["macro_f1_score"]))
    log("    saved to               : " + config.ISSUE_MODE_MODEL_FILE)
    log("    vectorizer saved to    : " + config.TEXT_VECTORIZER_FILE)
    return {"model": final_model, "vectorizer": vectorizer, "winner": winner,
            "scores": scores[winner], "training_scores": training_scores,
            "baseline": baseline, "candidates": scores,
            "ranking": [entry["candidate"] for entry in ranking],
            "vocabulary_size": len(vectorizer.vocabulary_)}
