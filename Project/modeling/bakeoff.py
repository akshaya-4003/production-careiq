"""The bake-off loop the three trainers share.

Every candidate is fitted on the same prepared columns, scored by cross validation
on the training split and then on a held-out split, and every score is kept so
the choice can be questioned afterwards.
"""

import time

from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from modeling.model_evaluation import compare_candidate_scores


def run_bakeoff(candidates, preprocessor, train_features, train_target, holdout_features,
                holdout_target, cv_folds, cv_scoring, cv_metric_name, holdout_scorer,
                log=print):
    """Fit, cross-validate and hold-out-score every candidate; return scores and the winner."""
    scores, fitted = {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("prepare", clone(preprocessor)), ("model", estimator)])
        started = time.time()
        cv_values = cross_val_score(pipeline, train_features, train_target,
                                    cv=cv_folds, scoring=cv_scoring)
        pipeline.fit(train_features, train_target)
        elapsed = time.time() - started
        held_out = holdout_scorer(pipeline, holdout_features, holdout_target)
        candidate_scores = {cv_metric_name: round(float(cv_values.mean()), 4),
                            "fit_seconds": round(elapsed, 2)}
        candidate_scores.update(held_out)
        scores[name] = candidate_scores
        fitted[name] = pipeline
        log("    trained {0:<30} in {1:>6.1f} seconds  {2} {3:.4f}".format(
            name, elapsed, cv_metric_name, candidate_scores[cv_metric_name]))
    ranking, winner = compare_candidate_scores(scores, cv_metric_name)
    return scores, ranking, winner, fitted


def print_ranking(ranking, columns, log=print):
    """Print the ranked bake-off table, best first."""
    header = "    {0:>32}".format("model") + "".join("  {0:>12}".format(c) for c in columns)
    log(header)
    for entry in ranking:
        line = "    {0:>32}".format(entry["candidate"])
        for column in columns:
            value = entry["scores"].get(column, "")
            line += "  {0:>12}".format("{0:.4f}".format(value) if isinstance(value, float)
                                       else str(value))
        log(line)
