"""The ten workflow nodes.

Each node is a named step so the trace reads as a story: "refused at
retrieve_policy" is a different morning from "it did not work". Tools produce the
facts; the agent in each node only explains or composes.
"""

import json
import uuid

from langgraph.types import interrupt

from agentic import agents
from config import config
from llm.client import LanguageModelError, load_llm_client
from modeling.model_registry import load_feature_schema, load_model_artifact
from persistence import repository
from persistence.database import open_database_session
from security.access_control import needs_manager_approval
from tools.order_data_tools import (
    days_late,
    describe_order_for_person,
    fetch_order,
    load_live_orders,
)
from tools.prediction_tools import (
    ToolError,
    build_planned_resolution_record,
    classify_risk_band,
    default_remedy_for,
    detect_issue_mode,
    estimate_resolution_hours,
    predict_escalation_probability,
    remedy_cash_cost,
    resolve_priority,
    severity_for_band,
)
from tools.retrieval_tools import lookup_remedy_policy, open_knowledge, verify_citations


# ---------------------------------------------------------------- shared services
class AdvisoryServices(object):
    """The heavy objects the nodes need, loaded once and only when first used."""

    def __init__(self, client=None, orders=None, knowledge=None):
        self._client = client
        self._orders = orders
        self._knowledge = knowledge
        self._models = None

    @property
    def client(self):
        if self._client is None:
            self._client = load_llm_client()
        return self._client

    @property
    def orders(self):
        if self._orders is None:
            self._orders = load_live_orders()
        return self._orders

    def replace_orders(self, orders):
        self._orders = orders

    @property
    def knowledge(self):
        if self._knowledge is None:
            self._knowledge = open_knowledge()
        return self._knowledge

    @property
    def models(self):
        if self._models is None:
            self._models = {
                "escalation": load_model_artifact(config.ESCALATION_MODEL_FILE),
                "escalation_schema": load_feature_schema(config.ESCALATION_SCHEMA_FILE),
                "resolution": load_model_artifact(config.RESOLUTION_MODEL_FILE),
                "resolution_schema": load_feature_schema(config.RESOLUTION_SCHEMA_FILE),
                "issue": load_model_artifact(config.ISSUE_MODE_MODEL_FILE),
                "vectorizer": load_model_artifact(config.TEXT_VECTORIZER_FILE),
            }
        return self._models

    def use_models(self, models):
        self._models = models


_SERVICES = {"current": None}


def configure_services(services):
    """Hand the nodes the services the caller already loaded (the API does this)."""
    _SERVICES["current"] = services
    return services


def get_services():
    if _SERVICES["current"] is None:
        _SERVICES["current"] = AdvisoryServices()
    return _SERVICES["current"]


# ---------------------------------------------------------------- helpers
def _step(node, status, detail=""):
    return {"node": node, "status": status, "detail": detail}


def _refuse(node, reason):
    return {"refused": node + ": " + reason, "steps": [_step(node, "refused", reason)]}


def _with_database(work):
    session = open_database_session()
    try:
        return work(session)
    finally:
        session.close()


def _agent_update(node, result, key, detail=""):
    return {key: result["labels"] if not result["sections"] else
            dict(result["labels"], **result["sections"]),
            "llm_call_log": [result["call_record"]],
            "steps": [_step(node, "ok", detail)]}


# ---------------------------------------------------------------- the nodes
def load_context_node(state):
    """Tools only. Look the order up and open a run record; an unknown order refuses here."""
    node = "load_context"
    order = fetch_order(get_services().orders, state["order_id"])
    if not order:
        return _refuse(node, "order " + repr(state["order_id"]) + " is not on file")
    summary = describe_order_for_person(order)
    summary["record"] = order
    run_id = "RUN-" + uuid.uuid4().hex[:12].upper()

    def open_run(session):
        repository.insert_agent_run(session, {
            "run_id": run_id, "thread_id": state["thread_id"], "order_id": order["order_id"],
            "ticket_id": state.get("ticket_id", ""), "requested_by": state["requested_by"],
            "status": config.AGENT_RUN_STATUS_RUNNING})

    _with_database(open_run)
    return {"order_summary": summary, "run_id": run_id, "order_id": order["order_id"],
            "resolution_id": "RES-" + uuid.uuid4().hex[:10].upper(),
            "steps": [_step(node, "ok", "order found; run " + run_id + " opened")]}


def plan_triage_node(state):
    """Call one: the triage plan."""
    node = "plan_triage"
    try:
        result = agents.triage_planner_agent(get_services().client, state["order_summary"],
                                             state["request_text"])
    except LanguageModelError as error:
        return _refuse(node, str(error))
    return _agent_update(node, result, "triage_plan")


def assess_risk_node(state):
    """Tools then call two: score the order by model and rule, then explain the score."""
    node = "assess_risk"
    services = get_services()
    order = state["order_summary"]["record"]
    try:
        probability = predict_escalation_probability(
            services.models["escalation"], services.models["escalation_schema"], order)
    except ToolError as error:
        return _refuse(node, str(error))
    band = classify_risk_band(probability)
    risk_facts = {
        "escalation_probability": round(probability, 4),
        "risk_band": band,
        "priority": resolve_priority(order.get("customer_tier", ""), band),
        "severity_code": severity_for_band(band),
    }
    try:
        result = agents.risk_analyst_agent(services.client, state["order_summary"], risk_facts,
                                           state["triage_plan"])
    except LanguageModelError as error:
        return _refuse(node, str(error))
    update = _agent_update(node, result, "risk_assessment",
                           "probability {0} banded {1}".format(risk_facts["escalation_probability"],
                                                               band))
    update["risk_facts"] = risk_facts
    return update


def identify_issue_node(state):
    """Tools then call three: classify the message, then have the analyst judge the reading."""
    node = "identify_issue"
    services = get_services()
    try:
        found = detect_issue_mode(services.models["issue"], services.models["vectorizer"],
                                  state["message_text"])
    except ToolError as error:
        return _refuse(node, str(error))
    issue_facts = dict(found, message_text=state["message_text"])
    try:
        result = agents.issue_analyst_agent(services.client, state["order_summary"], issue_facts)
    except LanguageModelError as error:
        return _refuse(node, str(error))
    update = _agent_update(node, result, "issue_analysis",
                           "{0} at {1}".format(found["issue_mode"], found["confidence"]))
    update["issue_facts"] = issue_facts
    return update


def retrieve_policy_node(state):
    """Tools then call four: retrieve, have the advisor name a remedy, check every citation."""
    node = "retrieve_policy"
    services = get_services()
    issue_mode = state["issue_facts"]["issue_mode"]
    try:
        found = lookup_remedy_policy(services.knowledge, issue_mode,
                                     state["order_summary"].get("product_category", ""))
        result = agents.policy_advisor_agent(services.client, issue_mode, state["order_summary"],
                                             found["context_text"])
    except (ToolError, LanguageModelError) as error:
        return _refuse(node, str(error))

    guidance = dict(result["labels"])
    named = guidance.get("REMEDY", "").strip()
    remedy = named if named in config.VALID_REMEDY_TYPES else default_remedy_for(issue_mode)
    cited, invented = verify_citations(
        guidance.get("CITATIONS", "") + " " + guidance.get("POLICY BASIS", ""), found["passages"])
    cited_chunks = [chunk for chunk in found["passages"] if chunk["chunk_id"] in cited]
    source_documents = []
    for chunk in cited_chunks or found["passages"]:
        if chunk["document_id"] not in source_documents:
            source_documents.append(chunk["document_id"])
    guidance.update({"remedy_type": remedy, "remedy_named_by_advisor": named,
                     "remedy_replaced_by_default": remedy != named,
                     "cited": cited, "invented": invented,
                     "source_documents": source_documents})
    policy_facts = {key: found[key] for key in ("question", "passages", "source_documents",
                                                "chunk_ids")}
    return {"policy_facts": policy_facts, "policy_guidance": guidance,
            "llm_call_log": [result["call_record"]],
            "steps": [_step(node, "ok", "remedy " + remedy + "; cited " + str(cited)
                            + (("; invented " + str(invented)) if invented else ""))]}


def compose_resolution_node(state):
    """Tools then call five: cost by rule, decide by rule whether a manager is needed, draft, save."""
    node = "compose_resolution"
    services = get_services()
    order = state["order_summary"]["record"]
    remedy = state["policy_guidance"]["remedy_type"]
    issue_mode = state["issue_facts"]["issue_mode"]
    try:
        amount = remedy_cash_cost(remedy, float(order.get("order_value_inr") or 0.0),
                                  days_late(order))
        planned = build_planned_resolution_record(
            order, state["risk_facts"], {"issue_mode": issue_mode, "remedy_type": remedy})
        hours = estimate_resolution_hours(services.models["resolution"],
                                          services.models["resolution_schema"], planned)
    except ToolError as error:
        return _refuse(node, str(error))
    approval_needed, reason = needs_manager_approval(remedy, amount,
                                                     order.get("payment_method", ""))
    costed = {"issue_mode": issue_mode, "remedy_type": remedy, "remedy_amount_inr": amount,
              "estimated_resolution_hours": hours, "needs_approval": approval_needed,
              "approval_reason": reason, "assumptions": planned.get("assumptions", [])}
    try:
        result = agents.resolution_composer_agent(
            services.client, state["order_summary"], state["risk_assessment"],
            state["issue_analysis"], state["policy_guidance"], costed)
    except LanguageModelError as error:
        return _refuse(node, str(error))

    guidance = state["policy_guidance"]
    steps_list = result["sections"].get("ACTION STEPS", [])
    draft = {
        "resolution_id": state["resolution_id"], "run_id": state["run_id"],
        "ticket_id": state.get("ticket_id", ""), "order_id": state["order_id"],
        "escalation_probability": state["risk_facts"]["escalation_probability"],
        "risk_band": state["risk_facts"]["risk_band"],
        "priority": state["risk_facts"]["priority"],
        "issue_mode": issue_mode,
        "customer_tier": state["order_summary"].get("customer_tier", ""),
        "estimated_resolution_hours": hours, "remedy_type": remedy, "remedy_amount_inr": amount,
        "needs_approval": approval_needed, "approval_reason": reason,
        "customer_reply": result["labels"].get("CUSTOMER REPLY", ""),
        "action_steps": "\n".join("{0}. {1}".format(i, s) for i, s in enumerate(steps_list, 1)),
        "policy_basis": guidance.get("POLICY BASIS", ""),
        "citations": ",".join(guidance.get("cited", [])),
        "source_documents": ",".join(guidance.get("source_documents", [])),
        "status": config.RESOLUTION_STATUS_PENDING,
    }
    calls_so_far = list(state["llm_call_log"]) + [result["call_record"]]

    def save(session):
        repository.insert_resolution(session, draft)
        for record in calls_so_far:
            repository.insert_llm_call_record(session, state["run_id"], record)
        if approval_needed:
            repository.update_agent_run_status(session, state["run_id"],
                                               config.AGENT_RUN_STATUS_AWAITING_REVIEW)

    _with_database(save)
    return {"costed_facts": costed, "draft_resolution": draft, "needs_approval": approval_needed,
            "persisted_call_count": len(calls_so_far),
            "llm_call_log": [result["call_record"]],
            "steps": [_step(node, "ok", "INR {0} for {1}; {2}".format(amount, remedy, reason))]}


def approve_under_own_authority_node(state):
    """Tools only. Settle a draft inside the associate's authority, on the spot."""
    node = "approve_under_own_authority"
    decision = {
        "decision": config.REVIEW_DECISION_APPROVE,
        "reviewer": state["requested_by"],
        "reviewer_role": config.ROLE_ASSOCIATE,
        "feedback_text": "",
        "authority_basis": "Approved under associate authority: "
                           + state["draft_resolution"].get("approval_reason", ""),
    }
    return {"review_decision": decision,
            "steps": [_step(node, "ok", decision["authority_basis"])]}


def request_manager_review_node(state):
    """The human gate. Pauses here, checkpointed, until a manager answers."""
    node = "request_manager_review"
    draft = state["draft_resolution"]
    decision = interrupt({
        "resolution_id": state["resolution_id"], "run_id": state["run_id"],
        "remedy_type": draft.get("remedy_type"), "remedy_amount_inr": draft.get("remedy_amount_inr"),
        "approval_reason": draft.get("approval_reason"),
    })
    if not isinstance(decision, dict) or decision.get("decision") not in \
            config.REVIEW_DECISION_OPTIONS:
        return _refuse(node, "the review decision " + repr(decision) + " is not one of "
                       + str(config.REVIEW_DECISION_OPTIONS))
    return {"review_decision": dict(decision),
            "steps": [_step(node, "ok", decision["decision"] + " by "
                            + str(decision.get("reviewer", "")))]}


def _recompute(state, decision):
    """The one recompute rule: only facts that actually changed move the estimate."""
    services = get_services()
    draft = state["draft_resolution"]
    order = state["order_summary"]["record"]
    final_remedy = decision.get("remedy_type") or draft["remedy_type"]
    assumed_delay = config.RESOLUTION_PLANNING_ASSUMPTIONS["approval_delay_hours"]
    actual_delay = decision.get("approval_delay_hours")
    remedy_changed = config.RECOMPUTE_ON_REMEDY_CHANGE and final_remedy != draft["remedy_type"]
    delay_changed = config.RECOMPUTE_ON_APPROVAL_DELAY and actual_delay is not None and \
        float(actual_delay) != float(assumed_delay)
    amount = draft["remedy_amount_inr"]
    hours = draft["estimated_resolution_hours"]
    if remedy_changed:
        amount = remedy_cash_cost(final_remedy, float(order.get("order_value_inr") or 0.0),
                                  days_late(order))
    if remedy_changed or delay_changed:
        context = {"issue_mode": draft["issue_mode"], "remedy_type": final_remedy}
        if delay_changed:
            context["approval_delay_hours"] = float(actual_delay)
        planned = build_planned_resolution_record(order, state["risk_facts"], context)
        hours = estimate_resolution_hours(services.models["resolution"],
                                          services.models["resolution_schema"], planned)
        changed = [name for name, flag in (("remedy_type", remedy_changed),
                                           ("approval_delay_hours", delay_changed)) if flag]
        note = "Estimate recomputed because " + " and ".join(changed) + " changed: {0} -> {1} hours." \
            .format(draft["estimated_resolution_hours"], hours)
    else:
        note = "Estimate carried forward unchanged at {0} hours; neither the remedy nor the " \
               "approval delay changed.".format(hours)
    return final_remedy, amount, hours, note


def finalize_resolution_node(state):
    """Call six: rewrite the reply to carry out the decision, holding the recompute rule."""
    node = "finalize_resolution"
    decision = dict(state["review_decision"])
    draft = state["draft_resolution"]
    try:
        final_remedy, amount, hours, note = _recompute(state, decision)
    except ToolError as error:
        return _refuse(node, str(error))
    decision.update({"final_remedy_type": final_remedy, "final_amount_inr": amount,
                     "final_estimated_hours": hours})
    try:
        result = agents.resolution_finalizer_agent(get_services().client, draft, decision)
    except LanguageModelError as error:
        return _refuse(node, str(error))
    final_steps = result["sections"].get("ACTION STEPS", [])
    final = {
        "customer_reply": result["labels"].get("FINAL REPLY", "") or draft["customer_reply"],
        "action_steps": "\n".join("{0}. {1}".format(i, s) for i, s in enumerate(final_steps, 1))
                        or draft["action_steps"],
        "what_changed": result["labels"].get("WHAT CHANGED", ""),
        "remedy_type": final_remedy,
        "remedy_amount_inr": amount,
        "estimated_resolution_hours": hours,
        "recompute_note": note,
        "status": config.RESOLUTION_STATUS_APPROVED,
    }
    return {"final_resolution": final, "llm_call_log": [result["call_record"]],
            "steps": [_step(node, "ok", note)]}


def persist_outcome_node(state):
    """Database only. Write the decision, the remaining calls and the final status."""
    node = "persist_outcome"
    run_id = state.get("run_id", "")
    remaining_calls = list(state.get("llm_call_log", []))[state.get("persisted_call_count", 0):]
    decision = state.get("review_decision") or {}
    final = state.get("final_resolution") or {}

    def write(session):
        for record in remaining_calls:
            repository.insert_llm_call_record(session, run_id, record)
        if state.get("refused"):
            if run_id:
                repository.update_agent_run_status(session, run_id,
                                                   config.AGENT_RUN_STATUS_COMPLETE,
                                                   state["refused"])
            return "refused"
        repository.insert_review_decision(session, {
            "resolution_id": state["resolution_id"], "decision": decision["decision"],
            "reviewer": decision.get("reviewer", ""),
            "reviewer_role": decision.get("reviewer_role", ""),
            "feedback_text": decision.get("feedback_text", "") or "",
            "authority_basis": decision.get("authority_basis", "") or "",
            "draft_before": json.dumps(state["draft_resolution"], default=str),
        })
        if decision["decision"] == config.REVIEW_DECISION_REJECT:
            repository.update_resolution_status(session, state["resolution_id"],
                                                config.RESOLUTION_STATUS_REJECTED)
        else:
            repository.apply_review_outcome(session, state["resolution_id"], final)
            if state.get("ticket_id"):
                repository.update_ticket_status(session, state["ticket_id"],
                                                config.TICKET_STATUS_CLOSED)
        repository.update_agent_run_status(session, run_id, config.AGENT_RUN_STATUS_COMPLETE)
        return decision["decision"]

    outcome = _with_database(write)
    return {"steps": [_step(node, "ok", "outcome " + outcome + "; "
                            + str(len(remaining_calls)) + " further calls recorded")]}
