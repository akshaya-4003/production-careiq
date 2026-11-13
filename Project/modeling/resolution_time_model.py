"""The three-way bake-off for "how many hours will this ticket take"."""

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from config import config
from data_pipeline.feature_builder import build_preprocessing_pipeline, prepare_for_modelling
from modeling.bakeoff import print_ranking, run_bakeoff
from modeling.model_evaluation import evaluate_regression_model
from modeling.model_registry import save_feature_schema, save_model_artifact


def resolution_candidates():
    seed = config.RANDOM_SEED
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                       n_jobs=-1, random_state=seed),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, random_state=seed),
    }


def train_resolution_model(frame, log=print):
    """Train three regressors, choose on cross-validated R squared, require beating the mean."""
    features, target, numeric, categorical = prepare_for_modelling(
        frame, config.RESOLUTION_IDENTIFIER_COLUMNS, config.RESOLUTION_TARGET_COLUMN)
    target = target.astype(float)
    log("    rows {0}, features {1}, median hours {2:.2f}".format(
        len(features), features.shape[1], target.median()))

    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features, target, test_size=config.VALIDATION_SPLIT_RATIO,
        random_state=config.RANDOM_SEED)
    training_mean = float(train_y.mean())

    def holdout_scorer(model, x, y):
        # The baseline is the training mean, which is what anybody gets for free.
        return evaluate_regression_model(model, x, y, baseline_value=training_mean)

    preprocessor = build_preprocessing_pipeline(numeric, categorical)
    folds = KFold(n_splits=config.CROSS_VALIDATION_FOLDS, shuffle=True,
                  random_state=config.RANDOM_SEED)
    candidates = resolution_candidates()
    scores, ranking, winner, _ = run_bakeoff(
        candidates, preprocessor, train_x, train_y, holdout_x, holdout_y, folds, "r2",
        config.RESOLUTION_PRIMARY_METRIC, holdout_scorer, log)
    print_ranking(ranking, [config.RESOLUTION_PRIMARY_METRIC, "r2_score",
                            "mean_absolute_error_hours"], log)

    chosen = scores[winner]
    if chosen["mean_absolute_error_hours"] >= chosen["baseline_mean_absolute_error_hours"]:
        raise RuntimeError("The best resolution model " + winner + " does not beat predicting "
                           "the training mean on held-out tickets; refusing to save it.")

    final_model = Pipeline([("prepare", clone(preprocessor)),
                            ("model", clone(candidates[winner]))])
    final_model.fit(features, target)
    save_model_artifact(final_model, config.RESOLUTION_MODEL_FILE)
    save_feature_schema(list(features.columns), config.RESOLUTION_SCHEMA_FILE)

    log("    best candidate         : " + winner)
    log("    cross validation R2    : {0:.4f}".format(chosen[config.RESOLUTION_PRIMARY_METRIC]))
    log("    held out R2            : {0:.4f} | MAE: {1:.4f} hours | predicting the mean "
        "would be: {2:.4f} hours".format(chosen["r2_score"], chosen["mean_absolute_error_hours"],
                                         chosen["baseline_mean_absolute_error_hours"]))
    log("    saved to               : " + config.RESOLUTION_MODEL_FILE)
    return {"model": final_model, "winner": winner, "scores": chosen,
            "candidates": scores, "ranking": [entry["candidate"] for entry in ranking],
            "features": list(features.columns)}
