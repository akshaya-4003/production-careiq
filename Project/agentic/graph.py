"""Builds the advisory workflow and runs it up to the review gate and beyond.

    START -> load_context -> plan_triage -> assess_risk -> identify_issue
          -> retrieve_policy -> compose_resolution
          -> (rule) approve_under_own_authority | request_manager_review (pauses)
          -> finalize_resolution (skipped on REJECT) -> persist_outcome -> END

The edges run in a fixed order rather than in parallel, so six calls spread over
a few seconds stay inside a free tier's per-minute limit.
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agentic import nodes
from agentic.state import AdvisoryState, new_advisory_state
from config import config

GO, STOP = "go", "stop"


def create_checkpoint_saver(checkpoint_path=None):
    """The SQLite file that stores paused runs, shared across threads."""
    path = config.CHECKPOINT_DATABASE_FILE if checkpoint_path is None else checkpoint_path
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)


def still_going(state):
    """'stop' when a node refused, 'go' otherwise."""
    return STOP if state.get("refused") else GO


def route_after_compose(state):
    """Read the approval flag the rule set at compose time; it does not decide it."""
    if state.get("refused"):
        return STOP
    if state.get("needs_approval"):
        return "request_manager_review"
    return "approve_under_own_authority"


def route_after_review(state):
    """A rejection skips the finalizer; approve and modify both go to it."""
    if state.get("refused"):
        return "persist_outcome"
    decision = (state.get("review_decision") or {}).get("decision")
    if decision == config.REVIEW_DECISION_REJECT:
        return "persist_outcome"
    return "finalize_resolution"


_CHAIN = [
    ("load_context", nodes.load_context_node),
    ("plan_triage", nodes.plan_triage_node),
    ("assess_risk", nodes.assess_risk_node),
    ("identify_issue", nodes.identify_issue_node),
    ("retrieve_policy", nodes.retrieve_policy_node),
    ("compose_resolution", nodes.compose_resolution_node),
]


def build_advisory_graph(checkpointer=None):
    """Register all ten nodes, wire the edges, and compile with the checkpointer."""
    builder = StateGraph(AdvisoryState)
    for name, function in _CHAIN:
        builder.add_node(name, function)
    builder.add_node("approve_under_own_authority", nodes.approve_under_own_authority_node)
    builder.add_node("request_manager_review", nodes.request_manager_review_node)
    builder.add_node("finalize_resolution", nodes.finalize_resolution_node)
    builder.add_node("persist_outcome", nodes.persist_outcome_node)

    builder.add_edge(START, "load_context")
    for (name, _), (following, _) in zip(_CHAIN, _CHAIN[1:]):
        builder.add_conditional_edges(name, still_going, {GO: following, STOP: "persist_outcome"})
    builder.add_conditional_edges("compose_resolution", route_after_compose, {
        "request_manager_review": "request_manager_review",
        "approve_under_own_authority": "approve_under_own_authority",
        STOP: "persist_outcome"})
    builder.add_edge("approve_under_own_authority", "finalize_resolution")
    builder.add_conditional_edges("request_manager_review", route_after_review, {
        "finalize_resolution": "finalize_resolution", "persist_outcome": "persist_outcome"})
    builder.add_edge("finalize_resolution", "persist_outcome")
    builder.add_edge("persist_outcome", END)

    saver = create_checkpoint_saver() if checkpointer is None else checkpointer
    return builder.compile(checkpointer=saver)


def build_initial_state(advisory_request, thread_id):
    """The state a new run starts from. The thread id must be new for every run."""
    return new_advisory_state(
        advisory_request.get("ticket_id", ""), advisory_request.get("order_id", ""),
        advisory_request.get("request_text", "") or config.DEFAULT_ADVISORY_REQUEST_TEXT,
        advisory_request.get("message_text", ""), advisory_request.get("requested_by", ""),
        thread_id)


def thread_config_for(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _current_values(compiled_graph, thread_config):
    return dict(compiled_graph.get_state(thread_config).values)


def start_advisory_run(compiled_graph, initial_state, thread_config):
    """Run until the workflow finishes or pauses at the review gate."""
    compiled_graph.invoke(initial_state, thread_config)
    return _current_values(compiled_graph, thread_config)


def resume_after_review(compiled_graph, thread_config, review_decision):
    """Carry on from the gate with the manager's decision; earlier calls are not repeated."""
    if review_decision.get("decision") not in config.REVIEW_DECISION_OPTIONS:
        raise ValueError("A decision outside " + str(config.REVIEW_DECISION_OPTIONS)
                         + " must never reach the graph.")
    compiled_graph.invoke(Command(resume=review_decision), thread_config)
    return _current_values(compiled_graph, thread_config)


def is_waiting_for_review(state):
    """True when the run stopped at the gate rather than finishing."""
    return bool(state.get("needs_approval")) and not state.get("review_decision") \
        and not state.get("refused")
