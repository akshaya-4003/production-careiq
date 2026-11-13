"""Entry point 9: measure the generative half and exit non-zero when a bar is missed.

    python3 run_evaluation.py                     offline backends (deterministic, CI-safe)
    python3 run_evaluation.py --backend auto      use the configured language model

Everything runs against a sandbox database, so the operational queue, run log and
transcripts are never touched.

Read the two routing lines together. A keyword router scores high on the wording it
was built against and far lower on held-out wording; closing that gap is the job of
the language-model routing layer, which the offline backend cannot stand in for. So
the held-out figure is reported and not gated.
"""

import argparse
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

from config import config  # noqa: E402

# (label, metric, threshold key, direction, gated)
CHECKS = [
    ("Retrieval finds the right document", "retrieval_hit_at_k", "retrieval_hit_at_k", ">=", True),
    ("Every citation is real", "citation_validity", "citation_validity", ">=", True),
    ("Declines what it cannot answer", "refusal_accuracy", "refusal_accuracy", ">=", True),
    ("Router covers known phrasing", "intent_macro_f1", "intent_macro_f1", ">=", True),
    ("Router generalises to new phrasing", "intent_macro_f1_heldout", "intent_macro_f1_heldout",
     ">=", False),
    ("A later turn remembers an earlier one", "memory_carry_over", "memory_carry_over", ">=",
     True),
    ("Never offers what policy forbids", "policy_violation_rate", "policy_violation_rate", "<=",
     True),
    ("Every draft cites its policy basis", "missing_policy_basis_rate", "policy_violation_rate",
     "<=", True),
    ("No run breaches its call budget", "call_budget_breach_rate", "policy_violation_rate", "<=",
     True),
]


def main():
    parser = argparse.ArgumentParser(description="CareIQ evaluation harness")
    parser.add_argument("--backend", default="offline", help="offline (default), auto or gemini")
    arguments = parser.parse_args()

    from persistence import database

    sandbox = tempfile.mkdtemp(prefix="careiq-eval-")
    database.bind_database("sqlite:///" + os.path.join(sandbox, "evaluation.db"))

    from agentic import graph as graph_module
    from agentic import nodes
    from evaluation import scorers
    from knowledge.embedding import load_embedder  # noqa: F401
    from llm.client import load_llm_client
    from modeling.model_registry import load_feature_schema, load_model_artifact
    from tools.order_data_tools import load_live_orders
    from tools.retrieval_tools import open_knowledge

    client = load_llm_client(arguments.backend)
    knowledge = open_knowledge()
    orders = load_live_orders()
    models = {
        "escalation": load_model_artifact(config.ESCALATION_MODEL_FILE),
        "escalation_schema": load_feature_schema(config.ESCALATION_SCHEMA_FILE),
        "resolution": load_model_artifact(config.RESOLUTION_MODEL_FILE),
        "resolution_schema": load_feature_schema(config.RESOLUTION_SCHEMA_FILE),
        "issue": load_model_artifact(config.ISSUE_MODE_MODEL_FILE),
        "vectorizer": load_model_artifact(config.TEXT_VECTORIZER_FILE),
    }
    services = nodes.configure_services(nodes.AdvisoryServices(client=client, orders=orders,
                                                               knowledge=knowledge))
    services.use_models(models)
    graph = graph_module.build_advisory_graph(
        graph_module.create_checkpoint_saver(os.path.join(sandbox, "checkpoint.db")))
    assistant = scorers.build_assistant(client, knowledge, orders, models)

    print("Evaluating with language model '{0}' and embeddings '{1}' in sandbox {2}".format(
        client.name, knowledge.backend_name(), sandbox))
    results = {}
    results.update(scorers.score_retrieval(knowledge, client))
    retrieval_details = results.pop("details")
    results.update(scorers.score_refusal(assistant))
    refusal_details = results.pop("details")
    routing = scorers.score_routing((models["issue"], models["vectorizer"]))
    results.update({key: routing[key] for key in ("intent_macro_f1", "intent_macro_f1_heldout")})
    results.update(scorers.score_memory(assistant))
    memory_details = results.pop("details")
    results.update(scorers.score_advisory_runs(graph_module, graph, orders))
    advisory_details = results.pop("details")

    line = "=" * 78
    print(line)
    print("{0:<44} {1:>9} {2:>7}  {3}".format("CHECK", "VALUE", "BAR", "RESULT"))
    print("-" * 78)
    failures, rows = 0, []
    for label, metric, threshold_key, direction, gated in CHECKS:
        value = results[metric]
        bar = config.EVALUATION_THRESHOLDS[threshold_key]
        passed = value >= bar if direction == ">=" else value <= bar
        verdict = "PASS" if passed else ("FAIL" if gated else "BELOW (reported, not gated)")
        if gated and not passed:
            failures += 1
        rows.append({"check": label, "metric": metric, "value": value, "bar": bar,
                     "direction": direction, "gated": gated, "passed": passed})
        print("{0:<44} {1:>9.4f} {2:>7.2f}  {3}".format(label, value, bar, verdict))
    print(line)

    report = {"backend": client.name, "embedding_backend": knowledge.backend_name(),
              "checks": rows, "metrics": results, "answer_rate_note":
              "answer_rate is the share of answerable golden questions answered rather than "
              "declined; the offline stand-in writes an extractive sentence.",
              "details": {"retrieval": retrieval_details, "refusal": refusal_details,
                          "routing_known": routing["known_details"],
                          "routing_heldout": routing["heldout_details"],
                          "memory": memory_details, "advisory": advisory_details}}
    folder = os.path.dirname(config.EVALUATION_REPORT_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(config.EVALUATION_REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print("Report written: " + config.EVALUATION_REPORT_FILE)
    if failures:
        print(str(failures) + " bar(s) missed.")
        return 1
    print("Every bar cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
