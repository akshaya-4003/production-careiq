"""Scorers for the generative half of CareIQ.

Every scorer builds its assistant with file_tickets=False, so replaying a golden set
does not fill the operational queue with the evaluation talking to itself.
"""

import json
import os
import uuid

from sklearn.metrics import f1_score

from chatbot.assistant import CareAssistant, new_session_id
from chatbot.intent_router import route_intent
from chatbot.rag_answer import answer_or_refuse
from config import config
from persistence.database import open_database_session
from security.access_control import needs_manager_approval
from tools.retrieval_tools import verify_citations


def load_golden_set(name):
    path = os.path.join(config.GOLDEN_SET_DIRECTORY, name + ".json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_assistant(client, knowledge, orders, models):
    return CareAssistant(client, knowledge, orders, models["issue"], models["vectorizer"],
                         file_tickets=False)


def score_retrieval(knowledge, client, golden=None):
    """Share of questions whose expected document is retrieved, and whether every citation is real."""
    golden = golden or load_golden_set("retrieval")
    hits, cited_total, cited_real, details = 0, 0, 0, []
    for item in golden["questions"]:
        passages = knowledge.search(item["question"], config.RETRIEVAL_RESULT_COUNT, True)
        documents = [passage["document_id"] for passage in passages]
        hit = item["expected_document"] in documents
        hits += int(hit)
        answered = answer_or_refuse(client, knowledge, item["question"],
                                    "This is the first message of the conversation.",
                                    "Nothing has been settled yet.")
        cited, invented = verify_citations(" ".join("[" + c + "]" for c in answered["citations"]),
                                           answered.get("passages", []))
        cited_total += len(cited) + len(invented)
        cited_real += len(cited)
        details.append({"question": item["question"], "expected": item["expected_document"],
                        "retrieved": documents, "hit": hit,
                        "answered": not answered["refused_reason"],
                        "citations": answered["citations"]})
    count = len(golden["questions"])
    return {"retrieval_hit_at_k": round(hits / float(count), 4),
            "citation_validity": round(cited_real / float(cited_total), 4) if cited_total else 1.0,
            "answer_rate": round(sum(d["answered"] for d in details) / float(count), 4),
            "details": details}


def score_refusal(assistant, golden=None):
    """Share of unanswerable questions correctly declined."""
    golden = golden or load_golden_set("unanswerable")
    correct, details = 0, []
    session = open_database_session()
    try:
        for question in golden["questions"]:
            answered = assistant.handle_message(session, new_session_id(), question, "evaluator",
                                                config.ROLE_CUSTOMER)
            declined = bool(answered["refused_reason"]) or \
                answered["intent"] in (config.INTENT_OUT_OF_SCOPE, config.INTENT_SMALL_TALK)
            correct += int(declined)
            details.append({"question": question, "intent": answered["intent"],
                            "declined": declined, "answer": answered["answer"][:160]})
    finally:
        session.close()
    return {"refusal_accuracy": round(correct / float(len(golden["questions"])), 4),
            "details": details}


def _routing_f1(messages, classifier):
    truth, predicted, details = [], [], []
    for item in messages:
        routed = route_intent(item["message"], {}, None, classifier)
        truth.append(item["intent"])
        predicted.append(routed["intent"])
        details.append({"message": item["message"], "expected": item["intent"],
                        "routed": routed["intent"], "basis": routed["basis"]})
    score = f1_score(truth, predicted, labels=config.INTENT_OPTIONS, average="macro",
                     zero_division=0)
    return round(float(score), 4), details


def score_routing(classifier):
    """Routing macro F1 on the wording it was built for, and on a held-out set."""
    known, known_details = _routing_f1(load_golden_set("routing_known")["messages"], classifier)
    heldout, heldout_details = _routing_f1(load_golden_set("routing_heldout")["messages"],
                                           classifier)
    return {"intent_macro_f1": known, "intent_macro_f1_heldout": heldout,
            "known_details": known_details, "heldout_details": heldout_details}


def score_memory(assistant, golden=None):
    """Whether a later turn resolves the order an earlier turn settled."""
    golden = golden or load_golden_set("memory")
    correct, details = 0, []
    session = open_database_session()
    try:
        for conversation in golden["conversations"]:
            session_id = new_session_id()
            answered = {}
            for turn in conversation["turns"]:
                answered = assistant.handle_message(session, session_id, turn, "evaluator",
                                                    config.ROLE_CUSTOMER)
            ok = answered.get("order_id") == conversation["expect_order"] and \
                conversation["expect_order"] in answered.get("answer", "")
            correct += int(ok)
            details.append({"turns": conversation["turns"], "resolved": answered.get("order_id"),
                            "basis": answered.get("order_basis"), "ok": ok})
    finally:
        session.close()
    return {"memory_carry_over": round(correct / float(len(golden["conversations"])), 4),
            "details": details}


def _draft_breaks_policy(draft, payment_method):
    """True when a draft offers something policy forbids without the gate catching it."""
    if draft.get("remedy_type") not in config.VALID_REMEDY_TYPES:
        return True
    needed, _ = needs_manager_approval(draft["remedy_type"], draft["remedy_amount_inr"],
                                       payment_method)
    return needed and not draft.get("needs_approval")


def score_advisory_runs(graph_module, graph, orders, golden=None):
    """Policy breaches, drafts without a policy basis, and call budget breaches."""
    golden = golden or load_golden_set("advisory")
    breaches, unsupported, over_budget, details = 0, 0, 0, []
    for item in golden["runs"]:
        order = orders.iloc[item["order_index"]]
        thread_id = "EVAL-" + uuid.uuid4().hex[:12].upper()
        thread = graph_module.thread_config_for(thread_id)
        state = graph_module.start_advisory_run(graph, graph_module.build_initial_state({
            "order_id": str(order["order_id"]), "message_text": item["message"],
            "requested_by": "evaluator"}, thread_id), thread)
        if state.get("refused"):
            details.append({"order_id": str(order["order_id"]), "refused": state["refused"]})
            unsupported += 1
            continue
        draft = state["draft_resolution"]
        if graph_module.is_waiting_for_review(state):
            state = graph_module.resume_after_review(graph, thread, {
                "decision": item["decision"], "remedy_type": item.get("remedy_type"),
                "feedback_text": "evaluation decision", "reviewer": "evaluator",
                "reviewer_role": config.ROLE_MANAGER})
        calls = len(state["llm_call_log"])
        breach = _draft_breaks_policy(draft, str(order.get("payment_method", "")))
        no_basis = not draft.get("citations") or not draft.get("policy_basis")
        budget = calls > config.EXPECTED_LLM_CALLS_PER_RUN
        breaches += int(breach)
        unsupported += int(no_basis)
        over_budget += int(budget)
        details.append({"order_id": draft["order_id"], "remedy": draft["remedy_type"],
                        "amount": draft["remedy_amount_inr"],
                        "needs_approval": draft["needs_approval"], "calls": calls,
                        "breach": breach, "no_basis": no_basis, "over_budget": budget})
    count = float(len(golden["runs"]))
    return {"policy_violation_rate": round(breaches / count, 4),
            "missing_policy_basis_rate": round(unsupported / count, 4),
            "call_budget_breach_rate": round(over_budget / count, 4), "details": details}
