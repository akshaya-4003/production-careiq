"""Every read and write the project makes.

Plain dictionaries come back, never ORM rows, and dates are rendered as text. A row
that does not exist comes back as an empty dictionary.
"""

import json
from datetime import datetime

from sqlalchemy import func, or_

from config import config
from persistence.models import (
    AgentRun,
    ChatSession,
    ChatTurn,
    IngestionRun,
    LlmCallRecord,
    Resolution,
    ReviewDecision,
    Ticket,
    User,
    utc_now,
)


def _as_dict(row):
    if row is None:
        return {}
    result = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        if isinstance(value, datetime):
            value = value.isoformat(timespec="seconds")
        result[column.name] = value
    return result


def _insert(session, model_class, record):
    allowed = set(column.name for column in model_class.__table__.columns) - {"id"}
    row = model_class(**{key: value for key, value in record.items() if key in allowed})
    session.add(row)
    session.commit()
    return _as_dict(row)


def _check_status(value, options, kind):
    if value not in options:
        raise ValueError(kind + " status " + repr(value) + " is not one of " + str(options))


# ---------------------------------------------------------------- users
def insert_user(session, user_record):
    return _insert(session, User, user_record)


def find_user(session, user_name):
    row = session.query(User).filter(User.user_name == str(user_name).strip()).first()
    return _as_dict(row)


def list_users(session):
    return [_as_dict(row) for row in session.query(User).order_by(User.id).all()]


def count_users(session):
    return int(session.query(func.count(User.id)).scalar() or 0)


# ---------------------------------------------------------------- tickets
def insert_ticket(session, ticket_record):
    record = dict(ticket_record)
    record.setdefault("status", config.TICKET_STATUS_OPEN)
    _check_status(record["status"], config.TICKET_STATUS_OPTIONS, "Ticket")
    return _insert(session, Ticket, record)


def find_ticket(session, ticket_id):
    return _as_dict(session.query(Ticket).filter(Ticket.ticket_id == ticket_id).first())


def find_open_ticket(session, ticket_id):
    row = session.query(Ticket).filter(Ticket.ticket_id == ticket_id,
                                       Ticket.status == config.TICKET_STATUS_OPEN).first()
    return _as_dict(row)


def amend_ticket(session, ticket_id, changes):
    row = session.query(Ticket).filter(Ticket.ticket_id == ticket_id,
                                       Ticket.status == config.TICKET_STATUS_OPEN).first()
    if row is None:
        return {}
    allowed = {"order_id", "customer_id", "message_text", "issue_mode", "issue_confidence",
               "assigned_to"}
    for key, value in changes.items():
        if key in allowed:
            setattr(row, key, value)
    row.updated_at = utc_now()
    session.commit()
    return _as_dict(row)


def list_tickets(session, status=None, assigned_to=None, limit=100, include_unassigned=False):
    query = session.query(Ticket)
    if status:
        query = query.filter(Ticket.status == status)
    if assigned_to is not None:
        if include_unassigned:
            query = query.filter(or_(Ticket.assigned_to == assigned_to,
                                     Ticket.assigned_to == config.UNASSIGNED_QUEUE))
        else:
            query = query.filter(Ticket.assigned_to == assigned_to)
    rows = query.order_by(Ticket.opened_at.desc(), Ticket.id.desc()).limit(limit).all()
    return [_as_dict(row) for row in rows]


def update_ticket_status(session, ticket_id, status):
    _check_status(status, config.TICKET_STATUS_OPTIONS, "Ticket")
    row = session.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if row is None:
        return {}
    row.status = status
    row.updated_at = utc_now()
    session.commit()
    return _as_dict(row)


def next_ticket_identifier(session):
    """The next TKT- reference, continuing from whatever is stored."""
    highest = 0
    for (ticket_id,) in session.query(Ticket.ticket_id).all():
        digits = "".join(character for character in ticket_id if character.isdigit())
        if digits:
            highest = max(highest, int(digits))
    return "TKT-{0:06d}".format(highest + 1)


# ---------------------------------------------------------------- agent runs
def insert_agent_run(session, run_record):
    record = dict(run_record)
    record.setdefault("status", config.AGENT_RUN_STATUS_RUNNING)
    _check_status(record["status"], config.AGENT_RUN_STATUS_OPTIONS, "Agent run")
    return _insert(session, AgentRun, record)


def fetch_agent_run(session, run_id):
    return _as_dict(session.query(AgentRun).filter(AgentRun.run_id == run_id).first())


def update_agent_run_status(session, run_id, status, refused_reason=None):
    _check_status(status, config.AGENT_RUN_STATUS_OPTIONS, "Agent run")
    row = session.query(AgentRun).filter(AgentRun.run_id == run_id).first()
    if row is None:
        return {}
    row.status = status
    if refused_reason is not None:
        row.refused_reason = refused_reason
    if status == config.AGENT_RUN_STATUS_COMPLETE:
        row.finished_at = utc_now()
    session.commit()
    return _as_dict(row)


def list_agent_runs(session, status=None, limit=100):
    query = session.query(AgentRun)
    if status:
        query = query.filter(AgentRun.status == status)
    rows = query.order_by(AgentRun.started_at.desc(), AgentRun.id.desc()).limit(limit).all()
    return [_as_dict(row) for row in rows]


# ---------------------------------------------------------------- resolutions
def insert_resolution(session, resolution_record):
    record = dict(resolution_record)
    record.setdefault("status", config.RESOLUTION_STATUS_PENDING)
    _check_status(record["status"], config.RESOLUTION_STATUS_OPTIONS, "Resolution")
    return _insert(session, Resolution, record)


def fetch_resolution(session, resolution_id):
    return _as_dict(session.query(Resolution).filter(
        Resolution.resolution_id == resolution_id).first())


def fetch_resolution_for_run(session, run_id):
    return _as_dict(session.query(Resolution).filter(Resolution.run_id == run_id).first())


def list_resolutions(session, status=None, limit=100):
    query = session.query(Resolution)
    if status:
        query = query.filter(Resolution.status == status)
    rows = query.order_by(Resolution.created_at.desc(), Resolution.id.desc()).limit(limit).all()
    return [_as_dict(row) for row in rows]


def update_resolution_status(session, resolution_id, status):
    _check_status(status, config.RESOLUTION_STATUS_OPTIONS, "Resolution")
    row = session.query(Resolution).filter(Resolution.resolution_id == resolution_id).first()
    if row is None:
        return {}
    row.status = status
    session.commit()
    return _as_dict(row)


# The fields the finalizer may write. The risk figures are deliberately absent:
# they are model output, and the number the manager saw must stay on the record.
FINALIZER_WRITABLE_FIELDS = ("customer_reply", "action_steps", "what_changed", "remedy_type",
                             "remedy_amount_inr", "estimated_resolution_hours",
                             "recompute_note", "status")


def apply_review_outcome(session, resolution_id, final_resolution):
    row = session.query(Resolution).filter(Resolution.resolution_id == resolution_id).first()
    if row is None:
        return {}
    for field in FINALIZER_WRITABLE_FIELDS:
        if field in final_resolution:
            if field == "status":
                _check_status(final_resolution[field], config.RESOLUTION_STATUS_OPTIONS,
                              "Resolution")
            setattr(row, field, final_resolution[field])
    session.commit()
    return _as_dict(row)


# ---------------------------------------------------------------- review decisions
def insert_review_decision(session, decision_record):
    record = dict(decision_record)
    if record.get("decision") not in config.REVIEW_DECISION_OPTIONS:
        raise ValueError("Decision " + repr(record.get("decision")) + " is not one of "
                         + str(config.REVIEW_DECISION_OPTIONS))
    if isinstance(record.get("draft_before"), dict):
        record["draft_before"] = json.dumps(record["draft_before"], default=str)
    return _insert(session, ReviewDecision, record)


def fetch_decisions_for_resolution(session, resolution_id):
    rows = session.query(ReviewDecision).filter(
        ReviewDecision.resolution_id == resolution_id).order_by(ReviewDecision.id).all()
    return [_as_dict(row) for row in rows]


# ---------------------------------------------------------------- language model calls
def insert_llm_call_record(session, run_id, call_record):
    record = dict(call_record)
    record["run_id"] = run_id or ""
    record.pop("created_at", None)
    return _insert(session, LlmCallRecord, record)


def fetch_call_records_for_run(session, run_id):
    rows = session.query(LlmCallRecord).filter(LlmCallRecord.run_id == run_id).order_by(
        LlmCallRecord.id).all()
    return [_as_dict(row) for row in rows]


def count_calls_by_run(session):
    rows = session.query(LlmCallRecord.run_id, func.count(LlmCallRecord.id)).group_by(
        LlmCallRecord.run_id).all()
    return {run_id: int(count) for run_id, count in rows}


# ---------------------------------------------------------------- chat
def get_or_create_chat_session(session, session_id, user_name=""):
    row = session.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if row is None:
        row = ChatSession(session_id=session_id, user_name=user_name or "")
        session.add(row)
    else:
        row.last_active_at = utc_now()
    session.commit()
    return _as_dict(row)


def find_chat_session(session, session_id):
    return _as_dict(session.query(ChatSession).filter(
        ChatSession.session_id == session_id).first())


def append_chat_turn(session, turn_record):
    row = session.query(ChatSession).filter(
        ChatSession.session_id == turn_record.get("session_id")).first()
    if row is not None:
        row.last_active_at = utc_now()
    return _insert(session, ChatTurn, turn_record)


def recent_chat_turns(session, session_id, how_many=None):
    query = session.query(ChatTurn).filter(ChatTurn.session_id == session_id).order_by(
        ChatTurn.id.desc())
    if how_many:
        query = query.limit(how_many)
    return [_as_dict(row) for row in reversed(query.all())]


def read_remembered_entities(session, session_id):
    row = session.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if row is None or not row.remembered_entities:
        return {}
    try:
        return json.loads(row.remembered_entities)
    except ValueError:
        return {}


def write_remembered_entities(session, session_id, entities):
    row = session.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if row is None:
        return {}
    row.remembered_entities = json.dumps(entities)
    row.last_active_at = utc_now()
    session.commit()
    return dict(entities)


# ---------------------------------------------------------------- ingestion
def start_ingestion_run(session, source, watermark_before):
    return _insert(session, IngestionRun, {
        "source": source, "watermark_before": json.dumps(watermark_before, default=str),
        "status": config.INGESTION_STATUS_FAILED, "message": "started"})


def finish_ingestion_run(session, ingestion_run_id, outcome):
    row = session.query(IngestionRun).filter(IngestionRun.id == ingestion_run_id).first()
    if row is None:
        return {}
    _check_status(outcome["status"], config.INGESTION_STATUS_OPTIONS, "Ingestion")
    row.rows_read = int(outcome.get("rows_read", 0))
    row.rows_scored = int(outcome.get("rows_scored", 0))
    row.high_risk_found = int(outcome.get("high_risk_found", 0))
    row.watermark_after = json.dumps(outcome.get("watermark_after", {}), default=str)
    row.status = outcome["status"]
    row.message = str(outcome.get("message", ""))
    row.finished_at = utc_now()
    session.commit()
    return _as_dict(row)


def recent_ingestion_runs(session, limit=None):
    how_many = config.INGESTION_RUN_HISTORY_LIMIT if limit is None else limit
    rows = session.query(IngestionRun).order_by(IngestionRun.id.desc()).limit(how_many).all()
    return [_as_dict(row) for row in rows]
