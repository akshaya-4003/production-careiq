import math
import os
import shutil
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


# ---------------------------------------------------------------- shared fixtures
# A fixture is a piece of setup pytest runs for you and hands to any test that asks
# for it by name. Four are used here, and each exists to stop the same expensive
# work being repeated once per test.
#
# Nothing in this file asks a language model for anything. Every assertion below
# is against a cleaned table, a trained model, the policy index, a written
# business rule, the database or the REST service, all of which answer the same
# way on every run. The agents are exercised only for their structure — how many
# nodes the workflow declares, which key the state carries, which way the review
# gate routes — because the wording a model returns is not a fact a test can pin
# down, and a suite that asserts on generated prose fails on a quota rather than
# on a defect.


@pytest.fixture(scope="session")
def signed_in_api():
    """Start the service once and sign in as each of the four roles.

    Every endpoint except signing in refuses a request that carries no token, so a
    test that calls the service needs one. Building the application loads three
    models, the policy index and the workflow, which takes a few seconds, so
    scope="session" means it happens once for the whole run rather than once per
    test.

    The accounts come from the configuration rather than being typed in here, so
    renaming a seeded account does not silently break the tests.
    """
    from api.routes import create_api_application
    from config import config

    client = TestClient(create_api_application(config.API_TITLE))
    signed_in = {"client": client}
    for role_name in config.ROLE_OPTIONS:
        account = {}
        for seed_account in config.DEMO_USERS:
            if seed_account["role"] == role_name and len(account) == 0:
                account = seed_account
        response = client.post(
            "/auth/login",
            json={"user_name": account["user_name"],
                  "password": account["password"]},
        )
        assert response.status_code == 200, (
            "Signing in as " + account["user_name"] + " answered with status "
            + str(response.status_code) + ". Run python3 run_database_setup.py to "
            "create the accounts before running the tests."
        )
        signed_in[role_name] = {
            "Authorization": "Bearer " + response.json()["access_token"]
        }
    return signed_in


@pytest.fixture(scope="session")
def policy_knowledge():
    """Open the policy index once, paired with the offline embedder.

    The index is opened rather than rebuilt, because rebuilding it is the job of
    run_knowledge_indexing.py and a test that rebuilt it would be testing the
    builder rather than the store. The offline embedder is named explicitly rather
    than left to "auto": the assertions below are about distances between
    questions, and those only hold if the same backend runs on every machine.
    """
    from knowledge.embedding import load_embedder
    from tools.retrieval_tools import open_knowledge

    knowledge = open_knowledge(embedder=load_embedder("offline"))
    assert knowledge.count() > 0, (
        "The policy index is empty. Run python3 run_knowledge_indexing.py before "
        "running the tests."
    )
    return knowledge


@pytest.fixture(scope="session")
def cleaned_orders():
    """The cleaned training table, read once.

    Several tests measure the cleaning by comparing this table against the raw
    file it came from. Reading a few thousand rows repeatedly is wasteful and the
    table is never modified, so one read serves the whole run.
    """
    from config import config
    from data_pipeline.dataset_loader import load_processed_dataset

    assert os.path.exists(config.CLEAN_ORDERS_TRAIN_FILE), (
        "The cleaned tables are missing. Run python3 run_data_preparation.py "
        "before running the tests."
    )
    return load_processed_dataset(config.CLEAN_ORDERS_TRAIN_FILE)


@pytest.fixture(scope="session")
def trained_models():
    """The three chosen models and their feature schemas, loaded once.

    Unpickling three fitted estimators is the slowest setup in the suite. Every
    model test needs at least one of them and none of them mutates anything, so
    they are loaded once and shared.
    """
    from config import config
    from modeling.model_registry import load_feature_schema, load_model_artifact

    assert os.path.exists(config.ESCALATION_MODEL_FILE), (
        "The trained models are missing. Run python3 run_model_training.py before "
        "running the tests."
    )
    return {
        "escalation": load_model_artifact(config.ESCALATION_MODEL_FILE),
        "escalation_schema": load_feature_schema(config.ESCALATION_SCHEMA_FILE),
        "resolution": load_model_artifact(config.RESOLUTION_MODEL_FILE),
        "resolution_schema": load_feature_schema(config.RESOLUTION_SCHEMA_FILE),
        "issue": load_model_artifact(config.ISSUE_MODE_MODEL_FILE),
        "vectorizer": load_model_artifact(config.TEXT_VECTORIZER_FILE),
    }


# ---------------------------------------------------------------- runtime
# ---------------------------------------------------------------- data cleaning
def test_cleaning_reads_every_date_format_the_extracts_use():
    """The source systems write the same date four different ways.

    A date that cannot be read is not a cosmetic problem: every day count the
    escalation model depends on is a difference between two dates, so an
    unparsed date becomes a missing feature and a wrongly parsed one becomes a
    plausible number that is wrong. A value that is not a date at all must come
    back missing rather than as a guessed date.
    """
    import pandas

    from data_pipeline.common_cleaning import parse_date_columns

    written = pandas.DataFrame({"observation_date": [
        "2026-06-15", "15/06/2026", "15-Jun-2026", "2026/06/15", "not a date"]})
    parsed = parse_date_columns(written, ["observation_date"])
    read_back = parsed["observation_date"]

    assert read_back.notna().sum() == 4, (
        "Only " + str(int(read_back.notna().sum())) + " of the four written "
        "formats were read. The extracts contain all four."
    )
    assert pandas.isna(read_back.iloc[4]), (
        "A value that is not a date became a date, which is worse than leaving "
        "it missing because the day count derived from it looks reasonable."
    )
    for position in range(4):
        assert read_back.iloc[position].date().isoformat() == "2026-06-15", (
            "Format " + str(position) + " was read as a different day."
        )



def test_cleaning_removes_duplicate_records(cleaned_orders):
    """The raw extract repeats rows and the cleaned table must not.

    A duplicated order is not a harmless extra row. It weights that order twice in
    training, and if it survives to the serving table the same order appears twice
    on the risk dashboard with the same score, which reads as two orders in
    trouble rather than one.
    """
    from config import config

    raw = pd.read_csv(config.RAW_ORDERS_TRAIN_FILE, dtype=str)
    assert len(cleaned_orders) < len(raw), (
        "The cleaned table has as many rows as the raw file, so no duplicate was "
        "ever removed."
    )
    repeated = int(cleaned_orders.duplicated(subset=["order_id"]).sum())
    assert repeated == 0, (
        str(repeated) + " order_id values still appear more than once after "
        "cleaning."
    )


def test_cleaning_leaves_no_missing_values():
    """Every cleaned table is complete, because the models cannot read a gap.

    Filling gaps is a decision, not a formality: a numeric gap takes the column
    median and a categorical gap takes an explicit unknown value. What matters
    here is that the decision was applied everywhere, since one surviving gap
    fails at fit time in a message that names the estimator rather than the
    column.
    """
    from config import config
    from data_pipeline.dataset_loader import load_processed_dataset

    for path in (config.CLEAN_ORDERS_TRAIN_FILE, config.CLEAN_ORDERS_EVAL_FILE,
                 config.CLEAN_TICKETS_TRAIN_FILE, config.CLEAN_MESSAGES_FILE,
                 config.CLEAN_LIVE_ORDERS_FILE):
        frame = load_processed_dataset(path)
        remaining = int(frame.isna().sum().sum())
        assert remaining == 0, (
            path + " still has " + str(remaining) + " empty cells."
        )


def test_cleaning_normalises_category_spellings(cleaned_orders):
    """The source systems disagree about spelling and the cleaned table must not.

    "cod", "COD" and "cash on delivery" are one payment method written three
    ways. Left alone they become three categories, the model learns each of them
    separately from a third of the data, and a rule written against "COD" silently
    misses two thirds of the orders it was meant to catch.
    """
    from config import config

    couriers = set(cleaned_orders["courier"].unique())
    assert couriers <= {"SwiftLane", "BharatPost", "MetroDash",
                        config.UNKNOWN_CATEGORY_VALUE}, (
        "Unexpected courier spellings survived cleaning: "
        + str(sorted(couriers))
    )

    methods = set(cleaned_orders["payment_method"].unique())
    assert methods <= {"Prepaid", "COD", "EMI", config.UNKNOWN_CATEGORY_VALUE}, (
        "Unexpected payment_method spellings survived cleaning: "
        + str(sorted(methods))
    )


def test_cleaning_repairs_numbers_written_with_units():
    """A number carrying a currency marker or a unit is still a number.

    The extracts contain "1,231.97 INR" and "0.3 hrs" in columns that are
    otherwise numeric. The percentage case is the one worth spelling out: a
    percentage is a hundredth, so 7.58% is 0.0758, and getting that wrong is
    silent — the column still holds numbers, they are just a hundred times too
    large, and the model happily fits them.
    """
    from data_pipeline.common_cleaning import repair_one_number

    assert repair_one_number("4,501.34 INR") == pytest.approx(4501.34)
    assert repair_one_number("0.3 hrs") == pytest.approx(0.3)
    assert repair_one_number("7.58%") == pytest.approx(0.0758)

    beyond_repair = repair_one_number("not a number")
    assert math.isnan(beyond_repair), (
        "A value that cannot be read as a number must come back as missing rather "
        "than as a guess, so the gap-filling step can decide what to do with it."
    )


def test_the_strongest_signal_survives_cleaning(cleaned_orders):
    """An outlier and an error are not the same thing.

    An earlier cleaner ran interquartile blanking across every numeric column and
    removed the orders that had not moved for days — which are precisely the
    orders that escalate. A statistical outlier can be the whole signal, so the
    blanking is applied by column and this column is exempt.
    """
    longest_dwell = float(cleaned_orders["transit_dwell_hours"].max())
    assert longest_dwell > 150, (
        "The longest transit dwell in the cleaned table is only "
        + str(round(longest_dwell, 1)) + " hours, so the orders that have not "
        "moved for days were blanked as outliers by the cleaner."
    )


# ---------------------------------------------------------------- message cleaning
def test_message_cleaning_strips_markup_and_keeps_meaning():
    """A customer message arrives wrapped in things that are not the message.

    Markup, a greeting, a signature and a mail client's footer all describe how
    the message was sent rather than what went wrong. The classifier learns from
    the words, so "Sent from my iPhone" appearing in a tenth of the rows is a
    feature it will happily use and that means nothing.
    """
    from data_pipeline.message_cleaning import clean_message_text, is_usable_message

    cleaned = clean_message_text(
        "<p>Dear team, my TV arrived CRACKED. Ref NRC-123456. Thanks.</p>"
        "\n--\nSent from my iPhone")

    assert "<p>" not in cleaned, "Markup survived cleaning."
    assert "iphone" not in cleaned.lower(), "The mail client footer survived."
    assert "cracked" in cleaned, (
        "The word describing what went wrong was removed, which is the one thing "
        "cleaning must never do."
    )
    assert "nrc-123456" not in cleaned.lower(), (
        "The order reference should be masked rather than learned: a model given "
        "order numbers memorises orders instead of language."
    )

    # Not every row in the extract is a message. "ok" carries no issue mode, and a
    # classifier trained on it learns that the shortest messages belong to
    # whichever mode is most common.
    assert is_usable_message(cleaned)
    assert not is_usable_message("ok")
    assert not is_usable_message("")


def test_the_quality_report_counts_the_dirt_it_found():
    """Cleaning without measuring is an assertion; measuring makes it evidence.

    The report exists so that the claim "the extracts were dirty" can be checked
    rather than believed. Every figure in it is a count of cells or of rows, so
    the same file profiled twice gives the same answer.
    """
    from config import config
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.quality_profiler import build_quality_report

    raw = load_raw_dataset(config.RAW_ORDERS_TRAIN_FILE, config.NULL_TOKEN_LIST)
    report = build_quality_report(
        raw, "orders_train",
        numeric_columns=config.ESCALATION_NUMERIC_COLUMNS,
        category_mappings={"courier": config.COURIER_MAPPING})

    assert report["rows"] == len(raw)
    assert report["columns"] == raw.shape[1]
    assert report["total_cells"] == report["rows"] * report["columns"]
    assert report["missing_cells"] > 0, (
        "The raw extract is documented as carrying gaps and the profiler found "
        "none, so it is not looking at the raw file."
    )
    assert 0.0 <= report["missing_share"] <= 1.0
    assert report["duplicate_rows"] > 0, (
        "The raw extract is documented as carrying duplicates and the profiler "
        "found none."
    )


def test_the_feature_schema_excludes_the_answer_column():
    """A model that can see the answer scores perfectly and predicts nothing.

    escalation_within_7_days is the value the model exists to estimate. If it
    reaches the feature matrix the model learns to copy it, the held-out score
    looks extraordinary, and the first live order — which has no answer column —
    fails or is scored from a column of zeros.
    """
    from config import config
    from modeling.model_registry import load_feature_schema

    schema = load_feature_schema(config.ESCALATION_SCHEMA_FILE)
    assert config.ESCALATION_TARGET_COLUMN not in schema, (
        "The target column is in the feature schema, so the model was trained on "
        "the answer."
    )

    # An identifier is a name, not a measurement. order_id is unique per row, so a
    # tree can split on it, memorise which orders escalated, and generalise to
    # nothing at all because tomorrow's order_id has never been seen.
    for identifier in config.ESCALATION_IDENTIFIER_COLUMNS:
        assert identifier not in schema, (
            identifier + " is an identifier and must not be a model feature."
        )


def test_selected_features_match_the_saved_schema(cleaned_orders):
    """What the training script selects and what the schema records must agree.

    The schema is what serving uses to build a row. If selection and schema drift
    apart, training saw one set of columns and scoring builds another, in the same
    order, with no error anywhere — the numbers are simply wrong.
    """
    from config import config
    from data_pipeline.feature_builder import prepare_for_modelling
    from modeling.model_registry import load_feature_schema

    features, _, _, _ = prepare_for_modelling(
        cleaned_orders, config.ESCALATION_IDENTIFIER_COLUMNS,
        config.ESCALATION_TARGET_COLUMN)
    schema = load_feature_schema(config.ESCALATION_SCHEMA_FILE)

    assert sorted(features.columns) == sorted(schema), (
        "Selection and the saved schema disagree. Only in selection: "
        + str(sorted(set(features.columns) - set(schema)))
        + ". Only in the schema: "
        + str(sorted(set(schema) - set(features.columns))) + "."
    )


def test_the_column_type_split_covers_every_feature_once(cleaned_orders):
    """Every column is prepared exactly once, or a column is silently dropped.

    The preprocessing pipeline scales the numeric columns and one-hot encodes the
    categorical ones. A column in neither list is thrown away without a word; a
    column in both is prepared twice and appears twice in the matrix.
    """
    from config import config
    from data_pipeline.feature_builder import identify_column_types, prepare_for_modelling

    features, _, _, _ = prepare_for_modelling(
        cleaned_orders, config.ESCALATION_IDENTIFIER_COLUMNS,
        config.ESCALATION_TARGET_COLUMN)
    numeric_columns, categorical_columns = identify_column_types(features)

    assert set(numeric_columns) & set(categorical_columns) == set(), (
        "These columns are in both lists and would be prepared twice: "
        + str(sorted(set(numeric_columns) & set(categorical_columns)))
    )
    assert set(numeric_columns) | set(categorical_columns) == set(features.columns), (
        "These columns are in neither list and would be silently dropped: "
        + str(sorted(set(features.columns)
                     - set(numeric_columns) - set(categorical_columns)))
    )


# ---------------------------------------------------------------- escalation model
def test_the_escalation_model_reports_a_probability(trained_models, cleaned_orders):
    """The number the whole platform turns on has to be a probability.

    Every downstream rule reads this figure: the risk band, the queue priority,
    the severity code and what the dashboard shows. A decision function score or a
    percentage would pass through all of them and quietly change every threshold.
    """
    from config import config
    from data_pipeline.feature_builder import prepare_for_modelling
    from modeling.model_evaluation import probability_of_escalation

    features, _, _, _ = prepare_for_modelling(
        cleaned_orders.head(200), config.ESCALATION_IDENTIFIER_COLUMNS,
        config.ESCALATION_TARGET_COLUMN)
    scores = probability_of_escalation(trained_models["escalation"], features)

    assert len(scores) == len(features)
    assert float(min(scores)) >= 0.0, "A probability cannot be negative."
    assert float(max(scores)) <= 1.0, "A probability cannot exceed one."

    # Serving assembles a single row and it must look exactly like a training row.
    # A row built in a different column order is scored without complaint and the
    # answer is meaningless: training and serving skew in its simplest form.
    from modeling.model_registry import build_model_input_row
    from tools.order_data_tools import fetch_order, load_live_orders

    orders = load_live_orders()
    row = build_model_input_row(
        fetch_order(orders, str(orders.iloc[0]["order_id"])),
        trained_models["escalation_schema"])
    assert list(row.columns) == list(trained_models["escalation_schema"])
    assert len(row) == 1, "Scoring one order must produce exactly one row."


def test_the_escalation_model_separates_the_order_book(trained_models):
    """A model that scores every order the same is useless however accurate.

    With an eighteen per cent escalation rate, answering "no" to everything scores
    eighty-two per cent accuracy and tells the desk nothing. What matters is
    spread: the book has to come back ordered, so the worst orders are actually at
    the top of the dashboard.
    """
    from tools.order_data_tools import load_live_orders
    from tools.prediction_tools import score_order_book

    scored = score_order_book(
        trained_models["escalation"], trained_models["escalation_schema"],
        load_live_orders())

    probabilities = scored["escalation_probability"]
    assert float(probabilities.max()) - float(probabilities.min()) > 0.2, (
        "The scored book spans only "
        + str(round(float(probabilities.max() - probabilities.min()), 4))
        + " in probability, so the model is not telling orders apart."
    )
    assert list(probabilities) == sorted(probabilities, reverse=True), (
        "The scored book must arrive worst first, because that is the order the "
        "dashboard shows it in and the dashboard holds no logic of its own."
    )


def test_the_escalation_model_beats_guessing_on_held_out_orders(trained_models):
    """Measured on the evaluation file, which training never saw.

    The training score answers "did it learn these rows". The only question worth
    asking is whether it learned anything that transfers, and a coin flip scores
    0.50 on ROC AUC. The precision-recall figure is checked as well because with
    an eighteen per cent positive rate ROC AUC flatters a model.
    """
    from config import config
    from data_pipeline.dataset_loader import load_processed_dataset
    from data_pipeline.feature_builder import prepare_for_modelling
    from modeling.model_evaluation import evaluate_classification_model

    held_out = load_processed_dataset(config.CLEAN_ORDERS_EVAL_FILE)
    features, target, _, _ = prepare_for_modelling(
        held_out, config.ESCALATION_IDENTIFIER_COLUMNS,
        config.ESCALATION_TARGET_COLUMN)
    scores = evaluate_classification_model(trained_models["escalation"],
                                           features, target)

    assert scores["roc_auc"] > 0.75, (
        "Held-out ROC AUC is " + str(scores["roc_auc"]) + "; a coin flip scores "
        "0.50."
    )
    assert scores["pr_auc"] > scores["positive_rate"] * 2, (
        "Held-out PR AUC is " + str(scores["pr_auc"]) + " against a positive rate "
        "of " + str(scores["positive_rate"]) + ", which is barely better than "
        "guessing at the base rate."
    )


# ---------------------------------------------------------------- resolution model
def test_the_resolution_model_estimates_a_real_duration(trained_models):
    """Hours out of service, which is a positive number of a plausible size.

    This estimate is shown to a manager deciding whether to approve a remedy, so a
    negative duration or a figure of several months is worse than no estimate: it
    is a number somebody may act on.
    """
    from tools.order_data_tools import fetch_order, load_live_orders
    from tools.prediction_tools import (
        build_planned_resolution_record,
        estimate_resolution_hours,
        predict_escalation_probability,
        classify_risk_band,
        resolve_priority,
        severity_for_band,
    )

    orders = load_live_orders()
    order = fetch_order(orders, str(orders.iloc[0]["order_id"]))

    probability = predict_escalation_probability(
        trained_models["escalation"], trained_models["escalation_schema"], order)
    band = classify_risk_band(probability)
    risk_facts = {
        "escalation_probability": probability,
        "risk_band": band,
        "priority": resolve_priority(order.get("customer_tier", ""), band),
        "severity_code": severity_for_band(band),
    }
    planned = build_planned_resolution_record(
        order, risk_facts,
        {"issue_mode": "Damaged in Transit", "remedy_type": "Replacement"})
    hours = estimate_resolution_hours(
        trained_models["resolution"], trained_models["resolution_schema"], planned)

    assert hours > 0.0, "A resolution cannot take a negative number of hours."
    assert hours < 24 * 60, (
        "An estimate of " + str(round(hours, 1)) + " hours is longer than two "
        "months, which is not a plausible ticket."
    )


def test_the_resolution_schema_excludes_answer_and_identifiers():
    """The same two exclusions as the risk model, for the same two reasons.

    actual_resolution_hours is the answer. The ticket and order keys are names.
    Neither belongs in a feature matrix, and this model is separately at risk
    because its planning record is assembled by hand rather than read from a file.
    """
    from config import config
    from modeling.model_registry import load_feature_schema

    schema = load_feature_schema(config.RESOLUTION_SCHEMA_FILE)
    assert config.RESOLUTION_TARGET_COLUMN not in schema, (
        "The resolution model was trained on the answer column."
    )
    for identifier in config.RESOLUTION_IDENTIFIER_COLUMNS:
        assert identifier not in schema, (
            identifier + " is an identifier and must not be a model feature."
        )


def test_the_resolution_model_beats_predicting_the_average(trained_models):
    """Answering "the usual" is the baseline any model has to beat.

    The mean of the training target requires no model at all. An estimator that
    cannot beat it on held-out tickets is worse than a constant, because it costs
    a fit, a pickle and a load, and it looks like it knows something.
    """
    from config import config
    from data_pipeline.dataset_loader import load_processed_dataset
    from data_pipeline.feature_builder import prepare_for_modelling
    from modeling.model_evaluation import evaluate_regression_model

    held_out = load_processed_dataset(config.CLEAN_TICKETS_EVAL_FILE)
    features, target, _, _ = prepare_for_modelling(
        held_out, config.RESOLUTION_IDENTIFIER_COLUMNS,
        config.RESOLUTION_TARGET_COLUMN)
    scores = evaluate_regression_model(trained_models["resolution"],
                                       features, target)

    assert scores["r2_score"] > 0.5, (
        "Held-out R2 is " + str(scores["r2_score"]) + "; predicting the mean "
        "scores 0.0."
    )
    assert scores["mean_absolute_error_hours"] < \
        scores["baseline_mean_absolute_error_hours"], (
        "The model is off by " + str(scores["mean_absolute_error_hours"])
        + " hours and predicting the mean is off by "
        + str(scores["baseline_mean_absolute_error_hours"]) + " hours, so the "
        "model is not earning its place."
    )


# ---------------------------------------------------------------- issue classifier
def test_the_text_vectorizer_matches_its_vocabulary(trained_models):
    """The classifier and the vectorizer are one artefact in two files.

    The model was fitted on a matrix of a fixed width, decided by the vocabulary.
    A vectorizer rebuilt on different text produces a different width, and the
    mismatch surfaces as a shape error at predict time rather than as anything
    that names the real problem.
    """
    vectorizer = trained_models["vectorizer"]
    matrix = vectorizer.transform(["the parcel arrived cracked and shattered"])

    assert matrix.shape[1] == len(vectorizer.vocabulary_), (
        "The transformed width does not match the stored vocabulary size."
    )
    assert matrix.shape[1] == trained_models["issue"].n_features_in_, (
        "The vectorizer produces " + str(matrix.shape[1]) + " features and the "
        "classifier expects " + str(trained_models["issue"].n_features_in_) + "."
    )


def test_issue_mode_labels_come_from_the_configured_set(trained_models):
    """Ten issue modes exist and the classifier may not invent an eleventh.

    Everything downstream is keyed on this label: the query expansion hints, the
    default remedy, the cost rule. A mode outside the catalogue has none of those,
    so it is refused rather than carried forward.
    """
    from config import config
    from tools.prediction_tools import detect_issue_mode

    for message in ("the parcel arrived cracked and the screen is shattered",
                    "my delivery is six days late and still has not arrived",
                    "i was charged twice for this order",
                    "the courier has lost my parcel entirely"):
        found = detect_issue_mode(
            trained_models["issue"], trained_models["vectorizer"], message)
        assert found["issue_mode"] in config.VALID_ISSUE_MODES, (
            "The classifier answered " + repr(found["issue_mode"]) + ", which is "
            "not one of the ten configured issue modes."
        )
        assert 0.0 <= found["confidence"] <= 1.0, (
            "The reported confidence is not a probability."
        )


def test_the_issue_classifier_beats_always_answering_the_same(trained_models):
    """Across ten modes, the majority class is a low bar and it still has to clear it.

    Macro F1 is the measure that matters here rather than accuracy, because it
    weights the rare modes the same as the common ones. Always answering "Late
    Delivery" scores well on accuracy and near zero on macro F1, which is the
    honest description of a model that has learned nothing.
    """
    from config import config
    from data_pipeline.dataset_loader import load_processed_dataset
    from modeling.model_evaluation import evaluate_text_classifier, majority_class_baseline

    messages = load_processed_dataset(config.CLEAN_MESSAGES_FILE)
    matrix = trained_models["vectorizer"].transform(
        messages[config.MESSAGE_TEXT_COLUMN])
    labels = messages[config.MESSAGE_TARGET_COLUMN]

    scores = evaluate_text_classifier(trained_models["issue"], matrix, labels)
    baseline = majority_class_baseline(labels)

    assert scores["macro_f1_score"] > baseline["macro_f1_score"] * 2, (
        "The classifier scores " + str(scores["macro_f1_score"]) + " macro F1 and "
        "always answering the most common mode scores "
        + str(baseline["macro_f1_score"]) + "."
    )
    assert scores["classes"] == len(config.VALID_ISSUE_MODES), (
        "The classifier knows " + str(scores["classes"]) + " modes and the "
        "configuration lists " + str(len(config.VALID_ISSUE_MODES)) + "."
    )


# ---------------------------------------------------------------- knowledge base
def test_every_policy_document_yields_text():
    """Six documents, and a document that yields nothing is a silent hole.

    A PDF that extracts to an empty string still counts as a file on disk. The
    chunker produces nothing from it, the index is short by that document's worth
    of passages, and the only symptom is that questions about that topic are
    answered from whichever document happens to be nearest.
    """
    from config import config
    from knowledge.pdf_extraction import load_policy_documents

    documents = load_policy_documents(config.DOCS_DIRECTORY)
    assert len(documents) == len(config.POLICY_DOCUMENTS), (
        "Loaded " + str(len(documents)) + " documents and the configuration lists "
        + str(len(config.POLICY_DOCUMENTS)) + "."
    )
    for document in documents:
        assert len(document["text"].strip()) > 200, (
            document["document_id"] + " yielded only "
            + str(len(document["text"].strip())) + " characters of text."
        )
        assert document["title"], document["document_id"] + " has no title."
        assert document["topic"], document["document_id"] + " has no topic."


def test_chunking_respects_the_configured_size_and_overlap():
    """A chunk is the unit of retrieval, so its size is a real setting.

    Too large and one chunk covers several topics, so a search matches it for the
    wrong reason. Too small and the sentence answering the question is separated
    from the heading that gives it meaning. The overlap exists so that an answer
    spanning a boundary survives in at least one chunk.
    """
    from config import config
    from knowledge.chunk_splitting import chunk_documents
    from knowledge.pdf_extraction import load_policy_documents

    chunks = chunk_documents(load_policy_documents(config.DOCS_DIRECTORY))
    assert len(chunks) > 0, "Chunking produced nothing from six documents."

    for chunk in chunks:
        assert len(chunk["text"]) <= config.CHUNK_SIZE_CHARACTERS * 2, (
            "Chunk " + chunk["chunk_id"] + " is " + str(len(chunk["text"]))
            + " characters, far beyond the configured "
            + str(config.CHUNK_SIZE_CHARACTERS) + "."
        )
        assert chunk["chunk_id"], "A chunk with no label cannot be cited."
        assert chunk["document_id"], (
            "Chunk " + chunk["chunk_id"] + " cannot be traced to a document."
        )
        assert chunk["heading"], (
            "Chunk " + chunk["chunk_id"] + " lost the heading it sat under, which "
            "is embedded with the body because a question often uses the words of "
            "a heading and none of the words beneath it."
        )

    # What the chunker produced and what the store holds must be the same count. A
    # partial write is the worst outcome, because retrieval still works: it simply
    # cannot find the passages that were never stored.
    from tools.retrieval_tools import open_knowledge
    from knowledge.embedding import load_embedder

    stored = open_knowledge(embedder=load_embedder("offline")).count()
    assert stored == len(chunks), (
        "The chunker produces " + str(len(chunks)) + " chunks and the store holds "
        + str(stored) + ". Re-run python3 run_knowledge_indexing.py."
    )


def test_search_tells_different_questions_apart(policy_knowledge):
    """Retrieval has to return the document that governs the question asked.

    This is the check that separates a working knowledge base from an index that
    returns its first four rows to everything. Each question below is answered by
    a different policy document, so the right document appearing among the
    passages is the whole test.
    """
    expectations = [
        ("how many days do I have to return a pair of shoes", "NRC-RET-004"),
        ("what compensation is due when a delivery is late", "NRC-SHIP-002"),
        ("who may approve a goodwill credit", "NRC-CSC-014"),
    ]
    for question, expected_document in expectations:
        found = policy_knowledge.search(question, 4, use_topic_filter=False)
        assert found, "Nothing came back for " + repr(question)
        documents = [chunk["document_id"] for chunk in found]
        assert expected_document in documents, (
            repr(question) + " returned " + str(documents) + " and should have "
            "included " + expected_document + "."
        )
        for chunk in found:
            assert 0.0 <= chunk["distance"] <= 2.0, (
                "A cosine distance of " + str(chunk["distance"]) + " is outside "
                "the range the collection was pinned to."
            )


# ---------------------------------------------------------------- order lookup
def test_an_order_that_is_not_on_file_is_refused_rather_than_answered():
    """An unknown reference comes back empty, and callers must check.

    This is the first node of the advisory workflow and the first thing the chat
    does with a reference, so it is where an unknown order has to stop. An empty
    record is the signal to stop; a blank row filled with zeroes would let every
    later step run and produce a confident, well-cited draft about an order that
    does not exist.
    """
    from tools.order_data_tools import fetch_order, load_live_orders, order_exists

    orders = load_live_orders()
    real = str(orders.iloc[0]["order_id"])

    found = fetch_order(orders, real)
    assert isinstance(found, dict), "An order must come back as a plain dictionary."
    assert found["order_id"] == real
    assert order_exists(orders, real)

    # Matching ignores case and padding, because the reference arrives typed by a
    # person rather than selected from a list.
    assert fetch_order(orders, "  " + real.lower() + " ")["order_id"] == real

    # An unknown reference comes back empty rather than as a blank row. An agent
    # handed a blank record will describe it to a customer as though it were
    # real, which is the worst of the available failures.
    assert not order_exists(orders, "NRC-000000")
    assert fetch_order(orders, "NRC-000000") == {}, (
        "An unknown order must come back empty, so the caller can tell the "
        "difference between an order with no value and no order at all."
    )


# ---------------------------------------------------------------- business rules
def test_the_risk_band_follows_the_written_thresholds():
    """NRC-POL-001 states the two thresholds and the code applies exactly those.

    The boundaries are the part worth testing. A probability exactly on a
    threshold has to fall on one stated side of it every time, because the band
    decides the queue priority and two orders scored identically must not be
    banded differently.
    """
    from config import config
    from tools.prediction_tools import classify_risk_band, severity_for_band

    assert classify_risk_band(0.95) == config.RISK_BAND_HIGH
    assert classify_risk_band(0.45) == config.RISK_BAND_MEDIUM
    assert classify_risk_band(0.05) == config.RISK_BAND_LOW

    assert classify_risk_band(config.HIGH_RISK_THRESHOLD) == config.RISK_BAND_HIGH, (
        "A probability exactly on the high threshold must band HIGH."
    )
    assert classify_risk_band(
        config.MEDIUM_RISK_THRESHOLD) == config.RISK_BAND_MEDIUM, (
        "A probability exactly on the medium threshold must band MEDIUM."
    )

    # Severity is not a second opinion on the risk. It is the same band written in
    # the vocabulary the desk's reporting uses, so the mapping is one to one.
    for band in config.RISK_BAND_OPTIONS:
        assert severity_for_band(band) == config.SEVERITY_BY_RISK_BAND[band]


def test_the_priority_comes_from_the_written_matrix():
    """Customer tier crossed with risk band, and nothing else.

    Three tiers by three bands is nine cells, all nine of which are written in
    config. Computing a priority any other way — from the probability directly,
    or by an agent's judgement — would make two orders with the same tier and band
    arrive at different priorities.
    """
    from config import config
    from tools.prediction_tools import resolve_priority

    assert len(config.PRIORITY_MATRIX) == 9, (
        "The matrix should hold all three tiers by all three bands."
    )
    for (tier, band), expected in config.PRIORITY_MATRIX.items():
        assert resolve_priority(tier, band) == expected, (
            "A " + tier + " customer at " + band + " risk should be " + expected
            + " and the rule returned " + resolve_priority(tier, band) + "."
        )


def test_late_delivery_compensation_follows_the_published_tiers():
    """NRC-SHIP-002 owes store credit by how late the delivery is, and it is capped.

    This figure is issued without the customer asking for it, which is exactly why
    it is a rule and not a judgement. The cap matters as much as the tiers: an
    uncapped rule on a badly delayed order commits an amount nobody approved.
    """
    from config import config
    from tools.prediction_tools import late_delivery_compensation

    assert late_delivery_compensation(0) == 0.0, (
        "A delivery that is not late owes nothing."
    )
    assert late_delivery_compensation(-3) == 0.0, (
        "A delivery ahead of the promised date owes nothing."
    )

    previous = 0.0
    for days in (1, 2, 3, 5, 8, 15, 40):
        owed = late_delivery_compensation(days)
        assert owed >= previous, (
            "Compensation fell from " + str(previous) + " to " + str(owed)
            + " as the delay grew from the previous tier to " + str(days) + " days."
        )
        assert owed <= config.LATE_DELIVERY_COMPENSATION_CAP_INR, (
            str(days) + " days late owes " + str(owed) + ", above the published "
            "cap of " + str(config.LATE_DELIVERY_COMPENSATION_CAP_INR) + "."
        )
        previous = owed


def test_the_cash_cost_of_a_remedy_follows_the_written_rules():
    """The authority limits are written about cash, so cash is computed by rule.

    A language model choosing this number would be the control answering to the
    thing it is meant to control: the figure decides whether a human is required,
    so it is derived from the order value and the published rule for that remedy.
    A replacement commits no cash at all, which is why it needs no manager.
    """
    from config import config
    from tools.prediction_tools import ToolError, remedy_cash_cost

    order_value = 10000.0

    assert remedy_cash_cost("Full Refund", order_value) == order_value
    assert remedy_cash_cost("Partial Refund", order_value) == round(
        order_value * config.PARTIAL_REFUND_MAXIMUM_SHARE, 2)
    assert remedy_cash_cost("Goodwill Credit", order_value) == \
        config.GOODWILL_CREDIT_DEFAULT_INR
    assert remedy_cash_cost("Replacement", order_value) == 0.0, (
        "A replacement sends goods, not money, so it commits no cash."
    )

    with pytest.raises(ToolError):
        remedy_cash_cost("Free Holiday", order_value)


def test_whether_a_manager_is_needed_follows_the_authority_rules():
    """NRC-CSC-014, and the reason the workflow has a human gate at all.

    Three cases always need a person whatever the amount, and they are listed in
    config so an auditor can read them without reading code. The amount test is
    strictly greater than the limit: a remedy exactly at an associate's authority
    is within it, which is what an authority limit means.
    """
    from config import config
    from security.access_control import needs_manager_approval

    limit = config.ASSOCIATE_AUTHORITY_LIMIT_INR

    needed, reason = needs_manager_approval("Full Refund", limit + 1.0, "Prepaid")
    assert needed and reason, "An amount above the limit must escalate, with a reason."

    needed, _ = needs_manager_approval("Replacement", limit, "Prepaid")
    assert not needed, (
        "A remedy exactly at the associate limit is within it and must not escalate."
    )

    needed, _ = needs_manager_approval("Goodwill Credit", 1.0, "Prepaid")
    assert needed, "A goodwill credit always needs a manager, whatever the amount."

    needed, _ = needs_manager_approval("Full Refund", 1.0, "EMI")
    assert needed, "A full refund on an EMI order always needs a manager."


# ---------------------------------------------------------------- routing
def test_the_router_sends_each_kind_of_message_to_its_own_destination():
    """Deciding who should answer comes before answering.

    An answer built by retrieving policy for "where is my order" is well written,
    correctly cited and completely useless, because the policy book does not know
    where that parcel is. Only the rule and classifier layers are exercised here;
    the language model layer is not consulted, so the result is the same on every
    run.
    """
    from chatbot.intent_router import route_intent
    from config import config
    from modeling.model_registry import load_model_artifact

    classifier = (load_model_artifact(config.ISSUE_MODE_MODEL_FILE),
                  load_model_artifact(config.TEXT_VECTORIZER_FILE))
    expectations = [
        ("how many days do I have to return a pair of shoes",
         config.INTENT_POLICY_QUESTION),
        ("where is my order NRC-900001", config.INTENT_ORDER_STATUS),
        ("it arrived cracked and the box was crushed", config.INTENT_COMPLAINT),
        ("thanks very much", config.INTENT_SMALL_TALK),
        ("what is the price of gold in Mumbai today", config.INTENT_OUT_OF_SCOPE),
    ]
    for message, expected in expectations:
        routed = route_intent(message, {}, None, classifier)
        assert routed["intent"] == expected, (
            repr(message) + " routed to " + routed["intent"] + " and should have "
            "gone to " + expected + ". Basis: " + routed.get("basis", "") + "."
        )
        assert routed["intent"] in config.INTENT_OPTIONS


def test_an_order_reference_is_recognised_and_a_wrong_one_is_named():
    """A reference the platform does not issue is not the same as no reference.

    "TTT-009983" is shaped like an order number and is not one. Treated as no
    reference at all it produced a ticket with an empty order, which no advisory
    run can ever start from. Saying which token was not recognised is the only way
    the customer can correct it.
    """
    from chatbot.intent_router import (
        extract_order_reference,
        is_bare_order_reference,
        looks_like_a_reference,
    )

    assert extract_order_reference("where is NRC-900123") == "NRC-900123"
    assert extract_order_reference("where is my parcel") == ""

    assert is_bare_order_reference("order no. NRC-900123"), (
        "A message that is an order number and filler words is not a new complaint."
    )
    assert not is_bare_order_reference("NRC-900123 arrived smashed to pieces")

    assert looks_like_a_reference("order no. TTT-009983") == "TTT-009983"
    assert looks_like_a_reference("order NRC-900123") == "", (
        "A valid reference must never reach the mistyped path."
    )


# ---------------------------------------------------------------- conversation memory
def test_the_conversation_carries_the_order_forward():
    """A follow-up question names neither the order nor the issue.

    "Am I covered?" means nothing on its own. What makes it answerable is that the
    conversation already settled which order and which issue it is about, so the
    entities are carried forward rather than the customer being asked to repeat
    themselves. An order named in this message always wins over one remembered,
    because a customer switching orders should be followed, not corrected.
    """
    from chatbot.memory import resolve_order_reference

    order_id, basis = resolve_order_reference("", {"order_id": "NRC-900001"})
    assert order_id == "NRC-900001"
    assert "carried" in basis

    order_id, basis = resolve_order_reference("NRC-900002",
                                              {"order_id": "NRC-900001"})
    assert order_id == "NRC-900002", (
        "An order named in this message must override one remembered from earlier."
    )

    order_id, _ = resolve_order_reference("", {})
    assert order_id == "", "With nothing named and nothing remembered, there is no order."

    # A memory that stores whatever it is handed grows without limit, so what a
    # conversation may retain is an allowed list that lives in config.
    from config import config

    assert "order_id" in config.MEMORY_ENTITY_KEYS
    assert "ticket_id" in config.MEMORY_ENTITY_KEYS, (
        "A conversation has to know what it already filed, or a later turn opens "
        "a second ticket for the same problem."
    )


def test_a_password_is_never_stored_in_readable_form():
    """What is stored is a verifier, not the password.

    A stolen table of hashes must not be a stolen table of passwords. Two things
    are checked: the stored value does not contain the password, and hashing the
    same password twice gives different stored values, which is what a per-hash
    salt is for.
    """
    from security.passwords import PasswordVault

    vault = PasswordVault()
    stored = vault.hash_password("associate123")

    assert "associate123" not in stored, "The password is recoverable from the hash."
    assert vault.matches_stored_hash("associate123", stored)
    assert not vault.matches_stored_hash("associate124", stored)
    assert vault.hash_password("associate123") != stored, (
        "Two hashes of the same password are identical, so no salt is being used."
    )


def test_a_token_carries_the_role_and_can_be_read_back():
    """The role travels in the token, so every request can be checked without a lookup.

    The token is signed, so a role cannot be edited into it by whoever holds it.
    Reading it back has to produce the same subject and role that were issued, or
    the permission checks are guarding nothing.
    """
    from config import config
    from security.access_control import AccessControl

    guard = AccessControl()
    token = guard.issue_token("asha", config.ROLE_ASSOCIATE, "Asha Menon")
    claims = guard.read_token("Bearer " + token)

    assert claims["sub"] == "asha"
    assert claims["role"] == config.ROLE_ASSOCIATE


def test_the_four_roles_differ_in_what_they_may_do():
    """Four roles, and the differences between them are the access control.

    A customer may ask and report and nothing else. An associate may not review.
    A manager may not read the audit trail. Every one of these is enforced on the
    server, so a permission list that quietly grew would be a hole nothing else
    would report.
    """
    from config import config
    from security.access_control import role_has_permission

    assert config.ROLE_PERMISSIONS[config.ROLE_CUSTOMER] == ["chat"], (
        "The customer role is one permission wide by design."
    )
    assert not role_has_permission(config.ROLE_CUSTOMER, "order_read")
    assert not role_has_permission(config.ROLE_ASSOCIATE, "review_decide")
    assert role_has_permission(config.ROLE_MANAGER, "review_decide")
    assert not role_has_permission(config.ROLE_MANAGER, "audit_read")
    assert role_has_permission(config.ROLE_ADMIN, "audit_read")


def test_the_authority_limits_are_ordered_and_can_be_sent_over_http():
    """Four roles, four limits, each above the one below it.

    The admin holds the head of customer care authority, so there is nobody above
    it and the buck stops there — an amount no admin may approve would be an
    amount nobody in the platform can ever action. The limits are also checked for
    being finite: an infinite limit is a valid Python float and is not valid JSON,
    and it made the whoami endpoint fail for admins while the sign-in itself
    succeeded.
    """
    import json

    from config import config
    from security.access_control import authority_limit_for

    limits = [authority_limit_for(role) for role in
              (config.ROLE_CUSTOMER, config.ROLE_ASSOCIATE,
               config.ROLE_MANAGER, config.ROLE_ADMIN)]

    assert limits == sorted(limits), (
        "The limits are not in ascending role order: " + str(limits)
    )
    assert limits[0] == 0.0, "A customer commits nothing."
    assert limits[1] == config.ASSOCIATE_AUTHORITY_LIMIT_INR
    assert limits[2] == config.MANAGER_AUTHORITY_LIMIT_INR

    for role in config.ROLE_OPTIONS:
        limit = authority_limit_for(role)
        assert math.isfinite(limit), role + " has a non-finite authority limit."
        json.dumps({"authority_limit_inr": limit})


# ---------------------------------------------------------------- persistence
def test_the_database_holds_the_tables_the_project_writes():
    """Every table the platform writes to has to exist before it writes.

    Creating them is idempotent, so this can run against a database that is
    already set up. What it catches is a model added in code and never reflected
    in the schema, which fails at the first insert rather than at start up.
    """
    from persistence.database import create_all_tables, engine

    created = create_all_tables(engine)
    for table in ("users", "tickets", "agent_runs", "resolutions",
                  "review_decisions", "llm_calls", "chat_sessions",
                  "chat_turns", "ingestion_runs"):
        assert table in created, table + " is missing from the schema."
    assert set(created) <= set(inspect(engine).get_table_names())


def test_a_ticket_survives_a_save_and_load():
    """What is written has to be readable, with the same values in it.

    The repository hands back plain dictionaries rather than ORM rows, because an
    ORM row stops working once its session closes and a row that leaked into a
    response would fail somewhere far from here.
    """
    from persistence.database import open_database_session
    from persistence.repository import (
        find_ticket,
        insert_ticket,
        next_ticket_identifier,
    )

    session = open_database_session()
    try:
        ticket_id = next_ticket_identifier(session)
        insert_ticket(session, {
            "ticket_id": ticket_id,
            "order_id": "NRC-900001",
            "customer_id": "CUS000001",
            "opened_by": "tester",
            "assigned_to": "tester",
            "source_channel": "Chat",
            "message_text": "the parcel arrived cracked",
            "issue_mode": "Damaged in Transit",
            "issue_confidence": 0.97,
            "status": "OPEN",
        })
        reloaded = find_ticket(session, ticket_id)

        assert isinstance(reloaded, dict), (
            "The repository must return plain dictionaries, not ORM rows."
        )
        assert reloaded["order_id"] == "NRC-900001"
        assert reloaded["issue_mode"] == "Damaged in Transit"
        assert reloaded["issue_confidence"] == pytest.approx(0.97)
        assert reloaded["status"] == "OPEN"
        assert isinstance(reloaded["opened_at"], str), (
            "Dates must come back as text so a response can carry them."
        )
    finally:
        session.close()


def test_a_resolution_survives_a_save_and_load():
    """The drafted resolution is the record a manager decides on.

    Every figure on it was produced by a model or a rule, so all of them have to
    read back exactly as written. The risk figures matter most: they are what the
    manager saw when deciding, and the finalizer is forbidden from writing them.
    """
    import uuid

    from config import config
    from persistence.database import open_database_session
    from persistence.repository import (
        fetch_resolution,
        insert_agent_run,
        insert_resolution,
    )

    session = open_database_session()
    try:
        run_id = "RUN-" + uuid.uuid4().hex[:12].upper()
        insert_agent_run(session, {
            "run_id": run_id,
            "thread_id": "THREAD-" + uuid.uuid4().hex[:8].upper(),
            "order_id": "NRC-900001",
            "ticket_id": "",
            "requested_by": "tester",
            "status": config.AGENT_RUN_STATUS_AWAITING_REVIEW,
        })

        resolution_id = "RES-" + uuid.uuid4().hex[:10].upper()
        insert_resolution(session, {
            "resolution_id": resolution_id,
            "run_id": run_id,
            "ticket_id": "",
            "order_id": "NRC-900001",
            "escalation_probability": 0.9934,
            "risk_band": config.RISK_BAND_HIGH,
            "priority": "P1",
            "issue_mode": "Damaged in Transit",
            "customer_tier": "Platinum",
            "estimated_resolution_hours": 81.98,
            "remedy_type": "Replacement",
            "remedy_amount_inr": 0.0,
            "needs_approval": False,
            "approval_reason": "",
            "customer_reply": "We are sending a replacement.",
            "action_steps": "1. Arrange pickup",
            "policy_basis": "NRC-RET-004 standard return window",
            "citations": "CH0023",
            "source_documents": "NRC-RET-004",
            "status": config.RESOLUTION_STATUS_PENDING,
        })

        reloaded = fetch_resolution(session, resolution_id)

        assert isinstance(reloaded, dict), (
            "The repository must return plain dictionaries, not ORM rows: an ORM "
            "row stops working once its session closes."
        )
        assert reloaded["escalation_probability"] == pytest.approx(0.9934)
        assert reloaded["risk_band"] == config.RISK_BAND_HIGH
        assert reloaded["priority"] == "P1"
        assert reloaded["remedy_amount_inr"] == pytest.approx(0.0)
        assert reloaded["needs_approval"] is False, (
            "A replacement commits no cash, so it needs no manager."
        )
        assert reloaded["citations"] == "CH0023"
        assert reloaded["status"] == config.RESOLUTION_STATUS_PENDING
        assert isinstance(reloaded["created_at"], str), (
            "Dates must come back as text so a response can carry them."
        )
    finally:
        session.close()


def test_saved_status_values_come_from_the_configured_sets():
    """A status outside its set is a row nothing will ever pick up again.

    The review queue reads resolutions whose status is exactly the pending value,
    and the audit page counts runs by status. A typo in a status is not a crash;
    it is a record that has quietly left every queue that was looking for it.
    """
    from config import config
    from persistence.database import open_database_session
    from persistence.repository import list_agent_runs, list_resolutions, list_tickets

    session = open_database_session()
    try:
        for resolution in list_resolutions(session, limit=200):
            assert resolution["status"] in config.RESOLUTION_STATUS_OPTIONS, (
                resolution["resolution_id"] + " holds status "
                + repr(resolution["status"]) + "."
            )
        for run in list_agent_runs(session, limit=200):
            assert run["status"] in (config.AGENT_RUN_STATUS_RUNNING,
                                     config.AGENT_RUN_STATUS_AWAITING_REVIEW,
                                     config.AGENT_RUN_STATUS_COMPLETE), (
                run["run_id"] + " holds status " + repr(run["status"]) + "."
            )
        for ticket in list_tickets(session, limit=200):
            assert ticket["status"] in ("OPEN", "CLOSED"), (
                ticket["ticket_id"] + " holds status " + repr(ticket["status"]) + "."
            )
    finally:
        session.close()


# ---------------------------------------------------------------- ingestion
# ---------------------------------------------------------------- workflow structure
def test_the_initial_state_matches_the_declared_state():
    """Every key the workflow reads has to exist before the first node runs.

    A node reading a key that was never initialised fails partway through a run
    that has already spent model calls. Declaring the state and building it in one
    place is what makes that impossible.
    """
    from agentic.graph import build_initial_state
    from agentic.state import AdvisoryState

    state = build_initial_state({
        "order_id": "NRC-900001",
        "ticket_id": "TKT-000001",
        "message_text": "the parcel arrived cracked",
        "request_text": "Assess and resolve",
        "requested_by": "tester",
    }, "THREAD-TEST")

    declared = set(AdvisoryState.__annotations__)
    built = set(state)
    assert built == declared, (
        "Keys declared and never built: " + str(sorted(declared - built))
        + ". Keys built and never declared: " + str(sorted(built - declared)) + "."
    )
    assert state["order_id"] == "NRC-900001"
    assert state["thread_id"] == "THREAD-TEST", (
        "The thread id names the conversation a run belongs to and must differ per "
        "run, or a second advisory on the same order resumes the first."
    )
    assert state["llm_call_log"] == [], "A new run has spent no calls."
    assert not state["refused"]


def test_the_workflow_declares_a_node_for_every_step():
    """Nine steps, each a named node, because the trace is for a person to read.

    In a chain this long "refused at policy" is a different morning from "it did
    not work". Compiling the graph is also the only way to find an edge pointing
    at a node that does not exist, which is a typo rather than a design error and
    fails at build time here rather than mid-run in production.
    """
    from agentic.graph import build_advisory_graph

    graph = build_advisory_graph()
    nodes = set(graph.get_graph().nodes)

    for step in ("load_context", "plan_triage", "assess_risk", "identify_issue",
                 "retrieve_policy", "compose_resolution",
                 "approve_under_own_authority", "request_manager_review",
                 "finalize_resolution", "persist_outcome"):
        assert step in nodes, step + " is not a node in the compiled workflow."


def test_the_review_gate_routes_on_the_written_authority_rule():
    """Whether a person is needed is decided by rule, never by an agent.

    That would be the control answering to the thing it is meant to control. The
    routing function reads the flag that needs_manager_approval already set at
    compose time, and a refusal ends the run wherever it happened rather than
    carrying on and answering anyway.
    """
    from agentic.graph import route_after_compose, route_after_review, still_going
    from config import config

    assert route_after_compose({"needs_approval": True}) == "request_manager_review"
    assert route_after_compose({"needs_approval": False}) == \
        "approve_under_own_authority", (
        "A remedy inside an associate's authority must not stop for a manager, or "
        "the published limit is a number the platform never honours."
    )
    assert route_after_compose({"refused": "no order"}) == "stop"

    assert still_going({}) == "go"
    assert still_going({"refused": "the order is unknown"}) == "stop"

    rejected = {"review_decision": {"decision": config.REVIEW_DECISION_REJECT}}
    assert route_after_review(rejected) == "persist_outcome", (
        "A rejection skips the finalizer, because there is nothing to rewrite. "
        "That is the sixth call saved on every rejection."
    )
    approved = {"review_decision": {"decision": config.REVIEW_DECISION_APPROVE}}
    assert route_after_review(approved) == "finalize_resolution"


# ---------------------------------------------------------------- REST service
def test_every_seeded_account_can_sign_in(signed_in_api):
    """Signing in is not enough; the session it produces has to work.

    An account can be issued a token and still be unable to use the service. The
    whoami endpoint is the first call the dashboard makes after signing in, so a
    failure there presents as a rejected password and sends everybody looking in
    the wrong place.
    """
    from config import config

    client = signed_in_api["client"]
    for role_name in config.ROLE_OPTIONS:
        response = client.get("/auth/me", headers=signed_in_api[role_name])
        assert response.status_code == 200, (
            "Reading back the session for role " + role_name + " answered "
            + str(response.status_code) + ": " + response.text[:200]
        )
        body = response.json()
        assert body["role"] == role_name
        assert isinstance(body["authority_limit_inr"], float)
        assert body["permissions"] == config.ROLE_PERMISSIONS[role_name]

    # The one endpoint that needs no token, because it answers "is it up". It also
    # names the backends, which is the fastest way to find that a key is present
    # and was never read.
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["orders_monitored"] > 0, "The order book is empty."
    assert health.json()["policy_chunks"] > 0, "The policy index is empty."
    assert health.json()["llm_backend"] in ("gemini", "offline")


def test_each_role_reaches_exactly_what_it_should(signed_in_api):
    """Hiding a button is a courtesy; refusing the request is the control.

    Signing in as an associate and calling the review endpoint by hand has to come
    back 403 rather than succeeding because the dashboard would not have offered
    the button. Every row below is a request the service must refuse on its own
    account.
    """
    from config import config

    client = signed_in_api["client"]
    expectations = [
        (config.ROLE_CUSTOMER, "/tickets", 403),
        (config.ROLE_CUSTOMER, "/risk/summary", 403),
        (config.ROLE_ASSOCIATE, "/tickets", 200),
        (config.ROLE_ASSOCIATE, "/risk/summary", 403),
        (config.ROLE_MANAGER, "/risk/summary", 200),
        (config.ROLE_MANAGER, "/runs", 403),
        (config.ROLE_ADMIN, "/runs", 200),
        (config.ROLE_ADMIN, "/evaluation", 200),
    ]
    for role_name, path, expected in expectations:
        answered = client.get(path, headers=signed_in_api[role_name])
        assert answered.status_code == expected, (
            role_name + " requesting " + path + " answered "
            + str(answered.status_code) + " and should have answered "
            + str(expected) + "."
        )

    assert client.get("/risk/summary").status_code in (401, 403), (
        "A request carrying no token at all must be refused."
    )


def test_the_order_endpoint_answers_for_real_orders_only(signed_in_api):
    """An order that is not on file is a 404, not an invented answer.

    The lookup reads the order book, so an unknown reference has nothing behind
    it. Returning an empty record instead would let every later step run on
    zeroes and produce a confident draft about an order that does not exist.
    """
    from config import config
    from tools.order_data_tools import load_live_orders

    client = signed_in_api["client"]
    header = signed_in_api[config.ROLE_ASSOCIATE]
    real_order = str(load_live_orders().iloc[0]["order_id"])

    found = client.get("/orders/" + real_order, headers=header)
    assert found.status_code == 200
    assert found.json()["order_id"] == real_order

    assert client.get("/orders/NRC-000000", headers=header).status_code == 404


def test_the_risk_endpoint_agrees_with_its_own_probability(signed_in_api):
    """The band on the record has to be the band the written rule gives.

    The dashboard shows both the probability and the band, side by side. If they
    are computed in two places they will eventually disagree, and a manager
    deciding on a remedy would be looking at a HIGH label above a low number.
    """
    from config import config
    from tools.order_data_tools import load_live_orders
    from tools.prediction_tools import classify_risk_band

    client = signed_in_api["client"]
    header = signed_in_api[config.ROLE_MANAGER]

    for order_id in [str(value) for value in
                     load_live_orders()["order_id"].head(5)]:
        answered = client.get("/orders/" + order_id + "/risk", headers=header)
        assert answered.status_code == 200
        body = answered.json()
        assert 0.0 <= body["escalation_probability"] <= 1.0
        assert body["risk_band"] == classify_risk_band(
            body["escalation_probability"]), (
            order_id + " is reported as " + body["risk_band"] + " at probability "
            + str(body["escalation_probability"]) + ", which the written "
            "thresholds do not agree with."
        )
        assert body["risk_band"] in config.RISK_BAND_OPTIONS

    # The dashboard holds no rule and reads no file of its own, so ordering and
    # filtering are the service's job. Worst first is what makes the page useful.
    table = client.get("/risk/orders?limit=50", headers=header)
    assert table.status_code == 200
    probabilities = [row["escalation_probability"] for row in table.json()]
    assert probabilities, "The risk table came back empty."
    assert probabilities == sorted(probabilities, reverse=True), (
        "The risk table is not ordered worst first."
    )
    high_only = client.get("/risk/orders?risk_band=HIGH&limit=20", headers=header)
    for row in high_only.json():
        assert row["risk_band"] == config.RISK_BAND_HIGH


def test_an_advisory_request_is_checked_before_any_call_is_made(signed_in_api):
    """A request that cannot be answered must fail before it costs anything.

    An advisory run spends five model calls before it pauses. Validating the
    request afterwards would mean an unknown order, or a request with no order at
    all, costs the same as a real one. The refusal also has to come from the
    service rather than from the schema's own words reaching an associate.
    """
    from config import config

    client = signed_in_api["client"]
    header = signed_in_api[config.ROLE_ASSOCIATE]

    missing_order = client.post("/advisories", headers=header,
                                json={"request_text": "Assess and resolve"})
    assert missing_order.status_code == 422, (
        "A request with no order at all must be refused by validation."
    )

    empty_order = client.post("/advisories", headers=header,
                              json={"order_id": ""})
    assert empty_order.status_code == 422, (
        "An empty order reference must be refused rather than run."
    )

    unknown_order = client.post(
        "/advisories", headers=header,
        json={"order_id": "NRC-000000", "request_text": "Assess and resolve"})
    assert unknown_order.status_code == 422, (
        "An unknown order must be refused at the first node, before any model "
        "call is spent."
    )

    refused_for_customer = client.post(
        "/advisories", headers=signed_in_api[config.ROLE_CUSTOMER],
        json={"order_id": "NRC-900001"})
    assert refused_for_customer.status_code == 403, (
        "A customer may not start an advisory run on their own complaint."
    )


def test_the_review_endpoint_refuses_an_unusable_decision(signed_in_api):
    """Three decisions exist and the graph must never be resumed with a fourth.

    Resuming a paused run with a decision it does not understand is worse than
    refusing: the run continues, the routing falls through to whichever branch was
    written last, and the record shows an outcome nobody chose. The check happens
    before the graph is touched.
    """
    from config import config

    client = signed_in_api["client"]
    header = signed_in_api[config.ROLE_MANAGER]

    assert len(config.REVIEW_DECISION_OPTIONS) == 3

    bad_decision = client.post("/resolutions/RES-DOESNOTEXIST/review",
                               headers=header, json={"decision": "MAYBE"})
    assert bad_decision.status_code == 422, (
        "An invalid decision must be refused before the run is resumed."
    )

    unknown_resolution = client.post(
        "/resolutions/RES-DOESNOTEXIST/review", headers=header,
        json={"decision": config.REVIEW_DECISION_APPROVE})
    assert unknown_resolution.status_code == 404

    as_associate = client.post(
        "/resolutions/RES-DOESNOTEXIST/review",
        headers=signed_in_api[config.ROLE_ASSOCIATE],
        json={"decision": config.REVIEW_DECISION_APPROVE})
    assert as_associate.status_code == 403, (
        "An associate may not review, and the service must say so rather than "
        "relying on the dashboard to hide the page."
    )


def test_the_citation_check_catches_an_invented_label(policy_knowledge):
    """A model asked for citations will produce well-formed ones for passages it never saw.

    Those are the hardest kind of wrong to spot, because they look exactly like
    the right kind. The check is mechanical rather than a judgement: every label
    an answer names is compared against the labels it was actually handed, and
    anything else is reported as invented.
    """
    from tools.retrieval_tools import search_policy_documents, verify_citations

    found = search_policy_documents(
        policy_knowledge, "how many days do I have to return a pair of shoes")
    real_label = found["chunk_ids"][0]

    cited, invented = verify_citations(
        "The window is 15 days [" + real_label + "] and returns are free [CH9999].",
        found["passages"])

    assert real_label in cited
    assert invented == ["CH9999"], (
        "A label the answer was never handed must be reported as invented; got "
        + str(invented) + "."
    )

    cited, invented = verify_citations("No labels at all here.", found["passages"])
    assert cited == [] and invented == []
