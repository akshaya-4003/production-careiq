"""The seven-way bake-off for "will this order raise a complaint within seven days"."""

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import config
from data_pipeline.feature_builder import build_preprocessing_pipeline, prepare_for_modelling
from modeling.bakeoff import print_ranking, run_bakeoff
from modeling.model_evaluation import evaluate_classification_model
from modeling.model_registry import save_feature_schema, save_model_artifact


def escalation_candidates():
    """The seven candidates. XGBoost and LightGBM are included when installed."""
    seed = config.RANDOM_SEED
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=6, min_samples_leaf=40,
                                                         random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                         n_jobs=-1, random_state=seed),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, random_state=seed),
        "CalibratedSVC": CalibratedClassifierCV(LinearSVC(random_state=seed), cv=3),
    }
    try:
        from xgboost import XGBClassifier

        candidates["XGBClassifier"] = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.06, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=-1)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        candidates["LGBMClassifier"] = LGBMClassifier(
            n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=seed,
            verbose=-1)
    except ImportError:
        pass
    return candidates


def train_escalation_model(frame, log=print):
    """Train seven candidates, choose on cross-validated ROC AUC, save winner and schema."""
    features, target, numeric, categorical = prepare_for_modelling(
        frame, config.ESCALATION_IDENTIFIER_COLUMNS, config.ESCALATION_TARGET_COLUMN)
    target = target.astype(int)
    log("    rows {0}, features {1}, positive rate {2:.4f}".format(
        len(features), features.shape[1], target.mean()))

    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, target, test_size=config.VALIDATION_SPLIT_RATIO, stratify=target,
        random_state=config.RANDOM_SEED)
    preprocessor = build_preprocessing_pipeline(numeric, categorical)
    folds = StratifiedKFold(n_splits=config.CROSS_VALIDATION_FOLDS, shuffle=True,
                            random_state=config.RANDOM_SEED)
    candidates = escalation_candidates()
    scores, ranking, winner, _ = run_bakeoff(
        candidates, preprocessor, train_x, train_y, holdout_x, holdout_y, folds,
        "roc_auc", config.ESCALATION_PRIMARY_METRIC, evaluate_classification_model, log)
    print_ranking(ranking, [config.ESCALATION_PRIMARY_METRIC, "roc_auc", "pr_auc", "recall"],
                  log)

    # The winner is refitted on every training row before it is saved.
    final_model = Pipeline([("prepare", clone(preprocessor)),
                            ("model", clone(candidates[winner]))])
    final_model.fit(features, target)
    save_model_artifact(final_model, config.ESCALATION_MODEL_FILE)
    save_feature_schema(list(features.columns), config.ESCALATION_SCHEMA_FILE)

    chosen = scores[winner]
    log("    best candidate         : " + winner)
    settings = {key: value for key, value in candidates[winner].get_params().items()
                if isinstance(value, (int, float, str)) and not isinstance(value, bool)}
    log("    settings               : " + str(dict(list(settings.items())[:6])))
    log("    cross validation ROC AUC: {0:.4f}".format(chosen[config.ESCALATION_PRIMARY_METRIC]))
    log("    held out ROC AUC       : {0:.4f} | PR AUC: {1:.4f} | recall: {2:.4f}".format(
        chosen["roc_auc"], chosen["pr_auc"], chosen["recall"]))
    log("    base rate to beat      : PR AUC {0:.4f} (positive rate)".format(
        chosen["positive_rate"]))
    log("    saved to               : " + config.ESCALATION_MODEL_FILE)
    return {"model": final_model, "winner": winner, "scores": chosen,
            "candidates": scores, "ranking": [entry["candidate"] for entry in ranking],
            "features": list(features.columns)}
