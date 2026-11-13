"""The advisory workflow's state, declared and built in one place.

A node reading a key that was never initialised would fail partway through a run
that has already spent model calls. Declaring every key here, and building the
initial state from this declaration, makes that impossible.
"""

import operator
from typing import Annotated, Any, Dict, List

from typing_extensions import TypedDict


class AdvisoryState(TypedDict):
    # What was asked, and the identifiers the run is recorded under
    ticket_id: str
    order_id: str
    request_text: str
    message_text: str
    requested_by: str
    thread_id: str
    run_id: str
    resolution_id: str
    # Facts from the tools: everything a model or a rule produced
    order_summary: Dict[str, Any]
    risk_facts: Dict[str, Any]
    issue_facts: Dict[str, Any]
    policy_facts: Dict[str, Any]
    costed_facts: Dict[str, Any]
    # What each agent concluded: interpretation only, never a new figure
    triage_plan: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    issue_analysis: Dict[str, Any]
    policy_guidance: Dict[str, Any]
    # The resolution, before and after the human decision
    draft_resolution: Dict[str, Any]
    review_decision: Dict[str, Any]
    final_resolution: Dict[str, Any]
    # Control: refused is the one key that stops everything
    refused: str
    needs_approval: bool
    # Accounting: these two lists accumulate across nodes rather than being replaced
    llm_call_log: Annotated[List[Dict[str, Any]], operator.add]
    steps: Annotated[List[Dict[str, Any]], operator.add]
    persisted_call_count: int


_EMPTY_BY_TYPE = {str: "", bool: False, int: 0}


def new_advisory_state(ticket_id, order_id, request_text, message_text, requested_by, thread_id):
    """Every declared key present, accumulators empty, nothing refused."""
    supplied = {
        "ticket_id": ticket_id or "",
        "order_id": (order_id or "").strip().upper(),
        "request_text": request_text or "",
        "message_text": message_text or "",
        "requested_by": requested_by or "",
        "thread_id": thread_id,
    }
    state = {}
    for key, annotation in AdvisoryState.__annotations__.items():
        if key in supplied:
            state[key] = supplied[key]
        elif annotation in _EMPTY_BY_TYPE:
            state[key] = _EMPTY_BY_TYPE[annotation]
        elif key in ("llm_call_log", "steps"):
            state[key] = []
        else:
            state[key] = {}
    return state
