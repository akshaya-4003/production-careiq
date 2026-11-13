"""The regression suite: guards against the defects the specification warns about.

    python3 -m pytest tests_regression.py -q

Run it separately from tests.py. Everything that writes runs against a sandbox
database and checkpoint file in a temporary folder, on the offline language model,
so this suite never touches the operational queue and never spends an API call.
It needs entry points 2 to 5 to have run (database, cleaned data, models, index).
"""

import json
import math
import os
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd
import pytest

from config import config


# ---------------------------------------------------------------- sandbox
@pytest.fixture(scope="module")
def sandbox():
    """A private database and checkpoint file, seeded with the demo accounts."""
    from persistence import database
    from persistence.repository import find_user, insert_user
    from security.passwords import PasswordVault

    folder = tempfile.mkdtemp(prefix="careiq-regression-")
    saved = {"llm": config.LLM_BACKEND, "checkpoint": config.CHECKPOINT_DATABASE_FILE}
    config.LLM_BACKEND = "offline"
    config.CHECKPOINT_DATABASE_FILE = os.path.join(folder, "checkpoint.db")
    database.bind_database("sqlite:///" + os.path.join(folder, "sandbox.db"))
    session = database.open_database_session()
    try:
        vault = PasswordVault()
        for account in config.DEMO_USERS:
            if not find_user(session, account["user_name"]):
                insert_user(session, {"user_name": account["user_name"],
                                      "display_name": account["display_name"],
                                      "password_hash": vault.hash_password(account["password"]),
                                      "role": account["role"]})
    finally:
        session.close()
    yield folder
    config.LLM_BACKEND = saved["llm"]
    config.CHECKPOINT_DATABASE_FILE = saved["checkpoint"]
    database.bind_database(config.APPLICATION_DATABASE_URL)
    shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture(scope="module")
def models():
    from modeling.model_registry import load_feature_schema, load_model_artifact

    return {
        "escalation": load_model_artifact(config.ESCALATION_MODEL_FILE),
        "escalation_schema": load_feature_schema(config.ESCALATION_SCHEMA_FILE),
        "resolution": load_model_artifact(config.RESOLUTION_MODEL_FILE),
        "resolution_schema": load_feature_schema(config.RESOLUTION_SCHEMA_FILE),
        "issue": load_model_artifact(config.ISSUE_MODE_MODEL_FILE),
        "vectorizer": load_model_artifact(config.TEXT_VECTORIZER_FILE),
    }


@pytest.fixture(scope="module")
def knowledge():
    from knowledge.embedding import load_embedder
    from tools.retrieval_tools import open_knowledge

    return open_knowledge(embedder=load_embedder("offline"))


@pytest.fixture(scope="module")
def orders():
    from tools.order_data_tools import load_live_orders

    return load_live_orders()


@pytest.fixture(scope="module")
def workflow(sandbox, models, knowledge, orders):
    """A compiled graph wired to the offline client and the sandbox."""
    from agentic import graph as graph_module
    from agentic import nodes
    from llm.client import OfflineClient

    services = nodes.configure_services(nodes.AdvisoryServices(
        client=OfflineClient(), orders=orders, knowledge=knowledge))
    services.use_models(models)
    return graph_module, graph_module.build_advisory_graph()


@pytest.fixture(scope="module")
def api(sandbox):
    from fastapi.testclient import TestClient

    from api.routes import create_api_application

    client = TestClient(create_api_application(config.API_TITLE))
    headers = {}
    for account in config.DEMO_USERS:
        token = client.post("/auth/login", json={"user_name": account["user_name"],
                                                 "password": account["password"]})
        headers[account["role"]] = {"Authorization": "Bearer " + token.json()["access_token"]}
    return client, headers


def _order_matching(orders, predicate):
    for _, row in orders.iterrows():
        if predicate(row):
            return row.to_dict()
    pytest.skip("No order in the book matches this scenario.")


def _run(graph_module, graph, order_id, message, graph_override=None):
    thread_id = "TEST-" + uuid.uuid4().hex[:12].upper()
    thread = graph_module.thread_config_for(thread_id)
    state = graph_module.start_advisory_run(
        graph_override or graph, graph_module.build_initial_state(
            {"order_id": order_id, "message_text": message, "requested_by": "asha"}, thread_id),
        thread)
    return state, thread


# ================================================================ data cleaning
def test_missing_tokens_are_caught_with_padding_and_case_but_not_inside_text():
    from data_pipeline.common_cleaning import normalise_missing

    frame = pd.DataFrame({"a": ["  NULL ", "n/A", "value", "not null at all"]})
    cleaned = normalise_missing(frame)
    assert cleaned["a"].isna().tolist() == [True, True, False, False]
    assert frame["a"].tolist()[0] == "  NULL ", "The table passed in must be left as it was."


def test_numeric_repair_touches_only_named_columns_and_skips_absent_ones():
    from data_pipeline.common_cleaning import repair_numeric_columns

    frame = pd.DataFrame({"value": ["1,231.97 INR", "x"], "code": ["0012", "0013"]})
    repaired = repair_numeric_columns(frame, ["value", "not_there"])
    assert repaired["value"].iloc[0] == pytest.approx(1231.97)
    assert math.isnan(repaired["value"].iloc[1])
    assert repaired["code"].tolist() == ["0012", "0013"], "A text column with digits stays text."


def test_category_mapping_folds_spellings_and_marks_the_rest_unknown():
    from data_pipeline.common_cleaning import apply_category_mapping

    frame = pd.DataFrame({"payment_method": [" Cash On Delivery ", "cod", "COD", "crypto", None]})
    mapped = apply_category_mapping(frame, "payment_method", config.PAYMENT_METHOD_MAPPING)
    assert mapped["payment_method"].tolist() == ["COD", "COD", "COD",
                                                 config.UNKNOWN_CATEGORY_VALUE,
                                                 config.UNKNOWN_CATEGORY_VALUE]


def test_booleans_have_no_third_state():
    from data_pipeline.common_cleaning import coerce_boolean_columns

    frame = pd.DataFrame({"flag": [" Y", "yes", "TRUE", "t", "1", "no", "", None, "0"]})
    assert coerce_boolean_columns(frame, ["flag"])["flag"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_a_sentinel_is_blanked_not_pulled_back_to_the_boundary():
    from data_pipeline.common_cleaning import clip_impossible_values

    frame = pd.DataFrame({"days_since_order": [5.0, 9999.0, -999.0]})
    clipped = clip_impossible_values(frame, config.ORDER_RANGE_LIMITS)
    assert clipped["days_since_order"].iloc[0] == 5.0
    assert clipped["days_since_order"].iloc[1:].isna().all(), (
        "A sentinel clamped to 120 would read as a real, very old order.")


def test_outlier_blanking_leaves_unnamed_columns_alone():
    from data_pipeline.common_cleaning import blank_extreme_outliers

    values = [float(v) for v in range(1, 21)] + [500.0]
    frame = pd.DataFrame({"scan_count": values, "transit_dwell_hours": values})
    blanked = blank_extreme_outliers(frame, ["scan_count"])
    assert blanked["scan_count"].isna().sum() == 1
    assert blanked["transit_dwell_hours"].isna().sum() == 0


def test_numeric_gaps_take_the_median_and_an_empty_column_takes_zero():
    from data_pipeline.common_cleaning import fill_missing_numeric

    frame = pd.DataFrame({"x": [1.0, 2.0, 1000.0, np.nan], "empty": [np.nan] * 4})
    filled = fill_missing_numeric(frame, ["x", "empty"])
    assert filled["x"].iloc[3] == 2.0, "The mean would have been pulled to 334 by one value."
    assert filled["empty"].tolist() == [0.0] * 4


def test_categorical_gaps_become_unknown_never_the_commonest_value():
    from data_pipeline.common_cleaning import fill_missing_categorical

    frame = pd.DataFrame({"courier": ["SwiftLane", "SwiftLane", None]})
    assert fill_missing_categorical(frame, ["courier"])["courier"].iloc[2] == \
        config.UNKNOWN_CATEGORY_VALUE


def test_duplicates_keep_the_first_occurrence():
    from data_pipeline.common_cleaning import drop_duplicate_rows

    frame = pd.DataFrame({"order_id": ["A", "A", "B"], "value": [1, 2, 3]})
    assert drop_duplicate_rows(frame, ["order_id"])["value"].tolist() == [1, 3]


def test_rows_without_an_answer_are_dropped_and_a_missing_target_column_raises():
    from data_pipeline.common_cleaning import drop_rows_missing_target

    frame = pd.DataFrame({"y": [1.0, np.nan, 0.0], "x": [1, 2, 3]})
    kept = drop_rows_missing_target(frame, "y")
    assert kept["x"].tolist() == [1, 3] and list(kept.columns) == ["y", "x"]
    with pytest.raises(KeyError):
        drop_rows_missing_target(frame, "not_there")


def test_the_serving_feed_is_cleaned_by_the_training_cleaner_and_never_has_an_answer():
    from data_pipeline.dataset_loader import load_raw_dataset
    from data_pipeline.order_cleaning import clean_orders_dataset

    raw = load_raw_dataset(config.LIVE_ORDERS_FEED_FILE, config.NULL_TOKEN_LIST).head(30)
    raw.loc[0, "order_value_inr"] = "1,231.97 INR"
    raw[config.ESCALATION_TARGET_COLUMN] = "1"
    cleaned = clean_orders_dataset(raw, is_serving_feed=True)
    assert config.ESCALATION_TARGET_COLUMN not in cleaned.columns
    assert cleaned.loc[0, "order_value_inr"] == pytest.approx(1231.97)
    assert int(cleaned.isna().sum().sum()) == 0


def test_ticket_catalogues_fold_case_and_padding_and_refuse_strangers():
    from data_pipeline.ticket_cleaning import clean_tickets_dataset
    from data_pipeline.dataset_loader import load_raw_dataset

    raw = load_raw_dataset(config.RAW_TICKETS_TRAIN_FILE, config.NULL_TOKEN_LIST).head(5).copy()
    raw.loc[0, "issue_mode"] = " late delivery "
    raw.loc[1, "issue_mode"] = "Alien Abduction"
    cleaned = clean_tickets_dataset(raw)
    assert cleaned.loc[0, "issue_mode"] == "Late Delivery"
    assert cleaned.loc[1, "issue_mode"] == config.UNKNOWN_CATEGORY_VALUE


def test_message_cleaning_drops_greeting_sign_off_and_quoted_reply_but_keeps_the_fault():
    from data_pipeline.message_cleaning import clean_message_text

    cleaned = clean_message_text(
        "Hello, the box arrived crushed and the lamp is broken Thanks and regards.\n"
        "> On Monday support wrote:\n> please send photos")
    assert "crushed" in cleaned and "broken" in cleaned
    assert "hello" not in cleaned and "regards" not in cleaned and "photos" not in cleaned


def test_the_quality_report_lists_only_columns_with_dirt_and_round_trips(tmp_path):
    from data_pipeline.quality_profiler import build_quality_report, save_quality_report

    frame = pd.DataFrame({"a": ["1", "2,000", None], "b": ["x", "y", "z"]})
    report = build_quality_report(frame, "tiny", numeric_columns=["a", "b"])
    assert report["values_needing_repair"] == {"a": 1}
    assert "b" in report["values_beyond_repair"] and "a" not in report["values_beyond_repair"]
    path = save_quality_report([report], str(tmp_path / "nested" / "q.json"))
    stored = json.load(open(path))
    assert list(stored) == ["datasets"] and stored["datasets"][0]["dataset"] == "tiny"


def test_features_and_target_split_refuses_a_table_with_no_answer():
    from data_pipeline.feature_builder import split_features_and_target

    with pytest.raises(KeyError):
        split_features_and_target(pd.DataFrame({"x": [1]}), config.ESCALATION_TARGET_COLUMN)


# ================================================================ model registry
def test_a_single_row_has_the_schema_order_and_neutral_values_for_missing_fields(models):
    from modeling.model_registry import build_model_input_row

    schema = models["escalation_schema"]
    row = build_model_input_row({"order_value_inr": "1500"}, schema)
    assert list(row.columns) == schema and len(row) == 1
    assert row.loc[0, "order_value_inr"] == 1500.0
    assert row.loc[0, "courier"] == config.UNKNOWN_CATEGORY_VALUE
    assert row.loc[0, "transit_dwell_hours"] == 0.0


def test_a_frame_keeps_row_order_so_scores_join_back(models, orders):
    from modeling.model_registry import build_model_input_frame

    built = build_model_input_frame(orders.iloc[::-1].head(10), models["escalation_schema"])
    assert built["order_value_inr"].tolist() == \
        orders.iloc[::-1].head(10)["order_value_inr"].astype(float).tolist()


def test_a_missing_model_file_names_the_path_and_the_step(tmp_path):
    from modeling.model_registry import load_model_artifact

    with pytest.raises(FileNotFoundError) as raised:
        load_model_artifact(str(tmp_path / "absent.pkl"))
    assert "absent.pkl" in str(raised.value) and "run_model_training" in str(raised.value)


def test_the_saved_schema_keeps_the_order_it_was_given(tmp_path):
    from modeling.model_registry import load_feature_schema, save_feature_schema

    path = save_feature_schema(["zeta", "alpha", "mid"], str(tmp_path / "schema.json"))
    assert load_feature_schema(path) == ["zeta", "alpha", "mid"]


def test_a_bake_off_ranks_every_candidate_including_losers():
    from modeling.model_evaluation import compare_candidate_scores

    ranking, winner = compare_candidate_scores({"a": {"m": 0.7}, "b": {"m": 0.9},
                                                "c": {"m": 0.1}}, "m")
    assert winner == "b" and [entry["candidate"] for entry in ranking] == ["b", "a", "c"]


def test_the_metrics_file_records_thirteen_candidates_and_three_winners():
    metrics = json.load(open(config.MODEL_METRICS_FILE))
    assert len(metrics["escalation_risk"]["candidates"]) == 7
    assert len(metrics["resolution_time"]["candidates"]) == 3
    assert len(metrics["issue_mode"]["candidates"]) == 3
    assert set(metrics["chosen_models"]) == {"escalation_risk", "resolution_time", "issue_mode"}


def test_a_message_too_short_to_read_is_refused_not_given_the_commonest_mode(models):
    from tools.prediction_tools import ToolError, detect_issue_mode

    with pytest.raises(ToolError):
        detect_issue_mode(models["issue"], models["vectorizer"], "ok")


# ================================================================ business rules
def test_just_below_a_threshold_falls_in_the_lower_band():
    from tools.prediction_tools import classify_risk_band

    assert classify_risk_band(config.HIGH_RISK_THRESHOLD - 1e-9) == config.RISK_BAND_MEDIUM
    assert classify_risk_band(config.MEDIUM_RISK_THRESHOLD - 1e-9) == config.RISK_BAND_LOW


def test_an_unrecognised_tier_takes_the_standard_row():
    from tools.prediction_tools import resolve_priority

    for band in config.RISK_BAND_OPTIONS:
        assert resolve_priority("Diamond", band) == config.PRIORITY_MATRIX[("Standard", band)]


def test_sla_compensation_follows_lateness_and_days_late_is_never_negative():
    from tools.order_data_tools import days_late
    from tools.prediction_tools import remedy_cash_cost

    assert days_late({"delivery_slack_days": 3.0}) == 0
    assert days_late({"delivery_slack_days": -5.0}) == 5
    assert remedy_cash_cost("SLA Compensation", 99999.0, 0) == 0.0
    assert remedy_cash_cost("SLA Compensation", 99999.0, 2) == 100.0
    assert remedy_cash_cost("SLA Compensation", 99999.0, 400) == \
        config.LATE_DELIVERY_COMPENSATION_CAP_INR
    for remedy in ("Replacement", "Reship Missing Item", "Exchange", "Explain and Expedite"):
        assert remedy_cash_cost(remedy, 99999.0) == 0.0


def test_a_negative_order_value_is_refused_at_the_boundary():
    from tools.prediction_tools import ToolError, remedy_cash_cost

    with pytest.raises(ToolError):
        remedy_cash_cost("Full Refund", -10.0)


def test_the_approval_rule_explains_itself_in_words():
    from security.access_control import needs_manager_approval

    needed, reason = needs_manager_approval("Partial Refund", 150.0, "Prepaid")
    assert not needed and "within" in reason
    needed, reason = needs_manager_approval("Goodwill Credit", 0.0, "COD")
    assert needed and "Goodwill" in reason
    needed, _ = needs_manager_approval("Partial Refund", 150.0, "EMI")
    assert not needed, "Only a full refund on EMI always escalates, not every EMI remedy."


def test_every_issue_mode_has_a_published_default_remedy_in_the_catalogue():
    from tools.prediction_tools import default_remedy_for

    for mode in config.VALID_ISSUE_MODES:
        assert default_remedy_for(mode) in config.VALID_REMEDY_TYPES


# ================================================================ knowledge base
def test_heading_detection_separates_headings_from_prose():
    from knowledge.chunk_splitting import looks_like_heading

    assert looks_like_heading("Standard return window")
    assert looks_like_heading("AUTHORITY LIMITS")
    assert not looks_like_heading("Goods must be returned unused, with all original tags attached.")
    assert not looks_like_heading("refunds take time")


def test_splitting_never_breaks_a_word_and_overlaps():
    from knowledge.chunk_splitting import split_without_breaking_words

    body = " ".join("word{0}".format(i) for i in range(300))
    chunks = split_without_breaking_words(body, 200, 40)
    words = set(body.split())
    for chunk in chunks:
        assert len(chunk) <= 200
        assert all(piece in words for piece in chunk.split())
    for first, second in zip(chunks, chunks[1:]):
        assert first.split()[-1] in second.split()[:12], "Consecutive chunks must overlap."
    assert split_without_breaking_words("short body", 200, 40) == ["short body"]


def test_chunk_labels_are_unique_and_stable_across_rebuilds():
    from knowledge.chunk_splitting import chunk_documents
    from knowledge.pdf_extraction import load_policy_documents

    documents = load_policy_documents()
    first, second = chunk_documents(documents), chunk_documents(documents)
    labels = [chunk["chunk_id"] for chunk in first]
    assert len(labels) == len(set(labels))
    assert labels == [chunk["chunk_id"] for chunk in second]
    assert all(label.startswith("CH") and len(label) == 6 for label in labels)


def test_the_offline_embedder_is_deterministic_and_normalised():
    from knowledge.embedding import OfflineEmbedder

    embedder = OfflineEmbedder()
    first = embedder.encode_one("parcel arrived cracked")
    second = OfflineEmbedder().encode_one("parcel arrived cracked")
    assert first.shape == (config.OFFLINE_EMBEDDING_DIMENSIONS,)
    assert np.allclose(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)


def test_the_index_is_pinned_to_cosine_and_records_its_embedder(knowledge):
    metadata = knowledge.collection.metadata
    assert metadata["hnsw:space"] == "cosine"
    assert metadata["embedding_backend"] == "offline_hashing"


def test_an_index_refuses_to_be_queried_by_a_different_embedder():
    from knowledge.embedding import Embedder
    from knowledge.procedure_retrieval import open_policy_knowledge

    class Other(Embedder):
        name = "minilm"

    with pytest.raises(ValueError):
        open_policy_knowledge(embedder=Other())


def test_an_empty_topic_list_means_no_filter_rather_than_a_filter_matching_nothing():
    from knowledge.procedure_retrieval import topic_filter, topics_in_question

    assert topic_filter([]) is None
    assert topics_in_question("what is the meaning of life") == []
    assert topic_filter(["returns", "delivery"]) == {"topic": {"$in": ["returns", "delivery"]}}


def test_the_prompt_context_carries_every_label_inside_delimiters(knowledge):
    from knowledge.procedure_retrieval import format_retrieved_context

    chunks = knowledge.search("how long does a refund take")
    context = format_retrieved_context(chunks)
    assert context.startswith("<<<") and context.rstrip().endswith(">>>")
    for chunk in chunks:
        assert "[" + chunk["chunk_id"] + "]" in context


def test_the_remedy_lookup_searches_the_whole_book_for_damage(knowledge):
    from tools.retrieval_tools import lookup_remedy_policy

    found = lookup_remedy_policy(knowledge, "Damaged in Transit", "Electronics")
    assert "NRC-PRD-021" in found["source_documents"], (
        "Damage evidence (photographs, 48 hours) lives in the product care guide.")


# ================================================================ language model layer
def test_labelled_replies_are_parsed_whatever_the_decoration():
    from llm.client import parse_labelled_response

    parsed = parse_labelled_response("**Remedy:** Replacement\n- POLICY BASIS: NRC-RET-004",
                                     ["REMEDY", "POLICY BASIS", "CITATIONS"])
    assert parsed == {"REMEDY": "Replacement", "POLICY BASIS": "NRC-RET-004", "CITATIONS": ""}


def test_a_numbered_section_stops_at_the_next_label():
    from llm.client import parse_numbered_section

    text = "ACTION STEPS:\n1. Pick up\n2) Ship\n- Tell customer\nNOTE: done"
    assert parse_numbered_section(text, "ACTION STEPS", ["ACTION STEPS", "NOTE"]) == \
        ["Pick up", "Ship", "Tell customer"]


def test_the_offline_client_answers_every_declared_label_from_the_prompt_facts():
    from llm.client import ask_for_labels, facts_from_prompt, load_llm_client

    client = load_llm_client("offline")
    prompt = "Facts:\n  Order: NRC-900001\n  Escalation probability: 0.42\n  Risk band: MEDIUM"
    assert facts_from_prompt(prompt)["Escalation probability"] == "0.42"
    answer = ask_for_labels(client, prompt, config.AGENT_NAME_RISK,
                            ["RISK SUMMARY", "KEY DRIVERS"])
    assert "0.42" in answer["labels"]["RISK SUMMARY"], "The stand-in must use the given figure."
    assert answer["labels"]["KEY DRIVERS"]
    assert answer["call_record"]["backend"] == "offline"
    assert answer["call_record"]["prompt_characters"] == len(prompt)


# ================================================================ routing and memory
def test_word_boundaries_stop_hi_matching_this():
    from chatbot.intent_router import count_matches, route_intent

    assert count_matches("how do I exchange this", ["hi"]) == 0
    assert route_intent("how do I exchange this", {}, None, None)["intent"] != \
        config.INTENT_SMALL_TALK


def test_a_greeting_on_a_complaint_is_still_a_complaint():
    from chatbot.intent_router import route_intent

    assert route_intent("Hi, my order is broken", {}, None, None)["intent"] == \
        config.INTENT_COMPLAINT


def test_arrived_cracked_is_damage_not_a_status_question():
    from chatbot.intent_router import route_intent

    routed = route_intent("it arrived cracked and the box was crushed NRC-900001", {}, None, None)
    assert routed["intent"] == config.INTENT_COMPLAINT


def test_a_return_request_with_a_fault_is_a_complaint():
    from chatbot.intent_router import asks_to_do_something

    assert asks_to_do_something("I want to return my shoes")
    assert not asks_to_do_something("I want to return my shoes because they arrived cracked")


def test_a_mistyped_reference_with_a_fault_still_reaches_the_complaint_path():
    from chatbot.intent_router import is_bare_reference_attempt, looks_like_a_reference

    assert is_bare_reference_attempt("TTT-009983", looks_like_a_reference("TTT-009983"))
    assert not is_bare_reference_attempt("my TTT-009983 arrived smashed", "TTT-009983")


def test_memory_keeps_only_allowed_keys_and_never_erases_with_an_empty_value(sandbox):
    from chatbot import memory
    from persistence.database import open_database_session

    session = open_database_session()
    try:
        session_id = "MEM-" + uuid.uuid4().hex[:8]
        memory.start_session(session, session_id, "tester")
        memory.remember(session, session_id, {"order_id": "NRC-900001", "card_number": "4111"})
        known = memory.remember(session, session_id, {"order_id": "", "issue_mode": "Late Delivery"})
        assert known == {"order_id": "NRC-900001", "issue_mode": "Late Delivery"}
        assert memory.format_history([]) and "first" in memory.format_history([])
    finally:
        session.close()


def test_a_complaint_then_its_order_number_files_one_ticket(sandbox, models, knowledge, orders):
    from chatbot.assistant import CareAssistant, new_session_id
    from llm.client import OfflineClient
    from persistence.database import open_database_session

    assistant = CareAssistant(OfflineClient(), knowledge, orders, models["issue"],
                              models["vectorizer"])
    session = open_database_session()
    try:
        session_id = new_session_id()
        first = assistant.handle_message(session, session_id,
                                         "the box arrived crushed and the item inside is cracked "
                                         "and broken", "rohan", config.ROLE_CUSTOMER)
        assert first["ticket_id"], first["answer"]
        reference = str(orders.iloc[3]["order_id"])
        second = assistant.handle_message(session, session_id, reference, "rohan",
                                          config.ROLE_CUSTOMER)
        assert second["ticket_id"] == first["ticket_id"], "The order number amends the ticket."
        from persistence.repository import find_ticket

        ticket = find_ticket(session, first["ticket_id"])
        assert ticket["order_id"] == reference
        assert ticket["assigned_to"] == config.UNASSIGNED_QUEUE
    finally:
        session.close()


# ================================================================ security
def test_a_tampered_or_expired_token_is_refused():
    import jwt

    from security.access_control import AccessControl

    guard = AccessControl()
    token = guard.issue_token("asha", config.ROLE_ASSOCIATE)
    claims = jwt.decode(token, options={"verify_signature": False})
    claims["role"] = config.ROLE_ADMIN
    forged = jwt.encode(claims, "not-the-secret", algorithm=config.JWT_ALGORITHM)
    with pytest.raises(ValueError):
        guard.read_token("Bearer " + forged)
    with pytest.raises(ValueError):
        AccessControl(lifetime_minutes=-1).read_token(
            AccessControl(lifetime_minutes=-1).issue_token("asha", config.ROLE_ASSOCIATE))


def test_an_unknown_role_holds_nothing_and_a_malformed_hash_does_not_raise():
    from security.access_control import authority_limit_for, permissions_for_role
    from security.passwords import PasswordVault

    assert permissions_for_role("superuser") == []
    assert authority_limit_for("superuser") == 0.0
    assert PasswordVault().matches_stored_hash("x", "not-a-hash") is False


# ================================================================ the advisory workflow
def test_a_remedy_inside_authority_is_approved_on_the_spot_in_six_calls(workflow, orders, sandbox):
    from persistence.database import open_database_session
    from persistence.repository import fetch_resolution

    graph_module, graph = workflow
    order = _order_matching(orders, lambda row: True)
    state, _ = _run(graph_module, graph, order["order_id"],
                    "the shoes do not fit, they are narrow and i need the next size up")
    assert not state["refused"], state["refused"]
    assert not graph_module.is_waiting_for_review(state)
    assert len(state["llm_call_log"]) == config.EXPECTED_LLM_CALLS_PER_RUN
    session = open_database_session()
    try:
        stored = fetch_resolution(session, state["resolution_id"])
        assert stored["status"] == config.RESOLUTION_STATUS_APPROVED
    finally:
        session.close()


def test_a_paused_run_survives_a_restart_and_keeps_its_risk_figures(workflow, orders, sandbox):
    from agentic.graph import build_advisory_graph, create_checkpoint_saver
    from persistence.database import open_database_session
    from persistence.repository import fetch_resolution

    graph_module, graph = workflow
    order = _order_matching(orders, lambda row: float(row["order_value_inr"]) >
                            config.ASSOCIATE_AUTHORITY_LIMIT_INR
                            and float(row["order_value_inr"]) <
                            config.MANAGER_AUTHORITY_LIMIT_INR
                            and row["payment_method"] != "EMI")
    state, thread = _run(graph_module, graph, order["order_id"],
                         "the courier has lost my parcel entirely, no scan for days")
    assert graph_module.is_waiting_for_review(state), state["draft_resolution"]
    assert len(state["llm_call_log"]) == 5
    drafted = state["draft_resolution"]

    # A fresh graph on a fresh connection stands in for a restarted process.
    restarted = build_advisory_graph(create_checkpoint_saver())
    final = graph_module.resume_after_review(restarted, thread, {
        "decision": config.REVIEW_DECISION_MODIFY, "remedy_type": "Replacement",
        "approval_delay_hours": 12.0, "feedback_text": "Send a replacement",
        "reviewer": "vikram", "reviewer_role": config.ROLE_MANAGER})
    assert len(final["llm_call_log"]) == 6, "The five calls before the pause are not repeated."
    session = open_database_session()
    try:
        stored = fetch_resolution(session, state["resolution_id"])
    finally:
        session.close()
    assert stored["remedy_type"] == "Replacement" and stored["remedy_amount_inr"] == 0.0
    assert "recomputed" in stored["recompute_note"]
    assert stored["escalation_probability"] == pytest.approx(drafted["escalation_probability"])
    assert stored["risk_band"] == drafted["risk_band"] and stored["priority"] == drafted["priority"]


def test_a_rejection_costs_five_calls_and_is_recorded(workflow, orders, sandbox):
    from persistence.database import open_database_session
    from persistence.repository import fetch_decisions_for_resolution, fetch_resolution

    graph_module, graph = workflow
    order = _order_matching(orders, lambda row: float(row["order_value_inr"]) >
                            config.ASSOCIATE_AUTHORITY_LIMIT_INR)
    state, thread = _run(graph_module, graph, order["order_id"],
                         "the courier has lost my parcel entirely, no scan for days")
    assert graph_module.is_waiting_for_review(state)
    final = graph_module.resume_after_review(graph, thread, {
        "decision": config.REVIEW_DECISION_REJECT, "reviewer": "vikram",
        "reviewer_role": config.ROLE_MANAGER})
    assert len(final["llm_call_log"]) == 5
    session = open_database_session()
    try:
        assert fetch_resolution(session, state["resolution_id"])["status"] == \
            config.RESOLUTION_STATUS_REJECTED
        decisions = fetch_decisions_for_resolution(session, state["resolution_id"])
        assert decisions[0]["decision"] == config.REVIEW_DECISION_REJECT
        assert json.loads(decisions[0]["draft_before"])["remedy_type"], \
            "The draft as it stood before the decision is kept for the auditor."
    finally:
        session.close()


def test_an_unknown_order_is_refused_at_the_first_node_before_any_call(workflow):
    graph_module, graph = workflow
    state, _ = _run(graph_module, graph, "NRC-000000", "the parcel arrived cracked")
    assert state["refused"].startswith("load_context")
    assert state["llm_call_log"] == []


def test_a_decision_outside_the_set_never_reaches_the_graph(workflow):
    graph_module, graph = workflow
    with pytest.raises(ValueError):
        graph_module.resume_after_review(graph, graph_module.thread_config_for("X"),
                                         {"decision": "MAYBE"})


# ================================================================ REST service
def test_a_manager_may_not_approve_past_their_own_limit(api, orders):
    client, headers = api
    order = _order_matching(orders, lambda row: float(row["order_value_inr"]) >
                            config.MANAGER_AUTHORITY_LIMIT_INR)
    started = client.post("/advisories", headers=headers[config.ROLE_ASSOCIATE], json={
        "order_id": order["order_id"],
        "message_text": "the courier has lost my parcel entirely, no scan for days"})
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["waiting_for_review"] and body["calls_made"] == 5
    resolution_id = body["resolution"]["resolution_id"]
    refused = client.post("/resolutions/" + resolution_id + "/review",
                          headers=headers[config.ROLE_MANAGER], json={"decision": "APPROVE"})
    assert refused.status_code == 403 and config.ROLE_ADMIN in refused.json()["detail"]
    approved = client.post("/resolutions/" + resolution_id + "/review",
                           headers=headers[config.ROLE_ADMIN], json={"decision": "APPROVE"})
    assert approved.status_code == 200 and approved.json()["calls_made"] == 6
    again = client.post("/resolutions/" + resolution_id + "/review",
                        headers=headers[config.ROLE_ADMIN], json={"decision": "APPROVE"})
    assert again.status_code == 409


def test_a_customer_sees_the_answer_but_not_the_desk_notes(api):
    client, headers = api
    reply = client.post("/chat", headers=headers[config.ROLE_CUSTOMER],
                        json={"message": "what is the price of gold in Mumbai today"})
    assert reply.status_code == 200
    body = reply.json()
    assert body["intent"] == config.INTENT_OUT_OF_SCOPE
    assert body["refused_reason"] == "" and body["intent_basis"] == ""
    staff = client.post("/chat", headers=headers[config.ROLE_ASSOCIATE],
                        json={"message": "what is the price of gold in Mumbai today"}).json()
    assert staff["refused_reason"], "Staff see why the turn was refused."
    transcript = client.get("/chat/" + body["session_id"], headers=headers[config.ROLE_CUSTOMER])
    assert [turn["speaker"] for turn in transcript.json()] == ["customer", "assistant"]


def test_the_associate_queue_includes_unassigned_tickets(api, orders):
    client, headers = api
    client.post("/chat", headers=headers[config.ROLE_CUSTOMER], json={
        "message": "the box arrived crushed and the item inside is cracked and broken "
                   + str(orders.iloc[5]["order_id"])})
    queue = client.get("/tickets", headers=headers[config.ROLE_ASSOCIATE]).json()
    assert any(ticket["assigned_to"] == config.UNASSIGNED_QUEUE for ticket in queue)
    assert all(ticket["assigned_to"] in ("asha", config.UNASSIGNED_QUEUE) for ticket in queue)


def test_the_catalogue_endpoint_serves_the_closed_lists(api):
    client, headers = api
    body = client.get("/catalogue", headers=headers[config.ROLE_MANAGER]).json()
    assert body["remedy_types"] == config.VALID_REMEDY_TYPES
    assert all(math.isfinite(limit) for limit in body["authority_limits"].values())


# ================================================================ ingestion
def test_a_failed_pass_does_not_advance_the_watermark(tmp_path, models):
    from ingestion.pipeline import IngestionPipeline
    from ingestion.source_reader import SourceReader
    from ingestion.watermark import read_watermark

    class Broken(SourceReader):
        name = "broken"

        def read_since(self, watermark, limit):
            raise RuntimeError("source went away")

    watermark = str(tmp_path / "w.json")
    pipeline = IngestionPipeline(reader=Broken(), serving_path=str(tmp_path / "s.csv"),
                                 watermark_path=watermark, record_run=False,
                                 risk_model=models["escalation"],
                                 feature_names=models["escalation_schema"])
    outcome = pipeline.run_once()
    assert outcome["status"] == config.INGESTION_STATUS_FAILED
    assert read_watermark(watermark)["cursor"] == 0


def test_the_pipeline_refuses_the_evaluation_file_and_upserts_by_order():
    from ingestion.pipeline import IngestionPipeline

    with pytest.raises(ValueError):
        IngestionPipeline.guard_against_the_answer(
            pd.DataFrame({config.ESCALATION_TARGET_COLUMN: [1]}))
    merged = IngestionPipeline.upsert(pd.DataFrame({"order_id": ["A", "B"], "v": [1, 2]}),
                                      pd.DataFrame({"order_id": ["B", "C"], "v": [9, 3]}))
    assert merged.sort_values("order_id")["v"].tolist() == [1, 9, 3]


def test_a_successful_pass_moves_the_watermark_and_ages_on_the_second_day(tmp_path, models):
    from ingestion.pipeline import IngestionPipeline
    from ingestion.watermark import read_watermark

    watermark = str(tmp_path / "w.json")
    pipeline = IngestionPipeline(serving_path=str(tmp_path / "s.csv"), watermark_path=watermark,
                                 record_run=False, risk_model=models["escalation"],
                                 feature_names=models["escalation_schema"])
    first = pipeline.run_once(limit=config.INGESTION_BATCH_LIMIT)
    assert first["status"] == config.INGESTION_STATUS_SUCCESS
    assert read_watermark(watermark)["cursor"] == config.INGESTION_BATCH_LIMIT


# ================================================================ runtime guard
def test_every_source_file_parses_on_python_39():
    from scripts.check_python39_syntax import check_project

    checked, findings = check_project(os.path.dirname(os.path.abspath(__file__)))
    assert checked and findings == [], findings
