"""The nine tables CareIQ writes to."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String(64), unique=True, nullable=False, index=True)
    display_name = Column(String(128), nullable=False, default="")
    password_hash = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False)
    created_at = Column(DateTime, nullable=False, default=utc_now)


class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(32), unique=True, nullable=False, index=True)
    order_id = Column(String(32), nullable=False, default="")
    customer_id = Column(String(32), nullable=False, default="")
    opened_by = Column(String(64), nullable=False, default="")
    assigned_to = Column(String(64), nullable=False, default="", index=True)
    source_channel = Column(String(32), nullable=False, default="")
    message_text = Column(Text, nullable=False, default="")
    issue_mode = Column(String(64), nullable=False, default="")
    issue_confidence = Column(Float, nullable=False, default=0.0)
    status = Column(String(16), nullable=False, default="OPEN", index=True)
    opened_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(40), unique=True, nullable=False, index=True)
    thread_id = Column(String(64), nullable=False, index=True)
    order_id = Column(String(32), nullable=False, default="")
    ticket_id = Column(String(32), nullable=False, default="")
    requested_by = Column(String(64), nullable=False, default="")
    status = Column(String(32), nullable=False)
    refused_reason = Column(Text, nullable=False, default="")
    started_at = Column(DateTime, nullable=False, default=utc_now)
    finished_at = Column(DateTime, nullable=True)


class Resolution(Base):
    __tablename__ = "resolutions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    resolution_id = Column(String(40), unique=True, nullable=False, index=True)
    run_id = Column(String(40), nullable=False, index=True)
    ticket_id = Column(String(32), nullable=False, default="")
    order_id = Column(String(32), nullable=False, default="")
    # Risk figures as they stood when the draft was made. The finalizer never writes them.
    escalation_probability = Column(Float, nullable=False, default=0.0)
    risk_band = Column(String(16), nullable=False, default="")
    priority = Column(String(8), nullable=False, default="")
    issue_mode = Column(String(64), nullable=False, default="")
    customer_tier = Column(String(32), nullable=False, default="")
    estimated_resolution_hours = Column(Float, nullable=False, default=0.0)
    remedy_type = Column(String(64), nullable=False, default="")
    remedy_amount_inr = Column(Float, nullable=False, default=0.0)
    needs_approval = Column(Boolean, nullable=False, default=False)
    approval_reason = Column(Text, nullable=False, default="")
    customer_reply = Column(Text, nullable=False, default="")
    action_steps = Column(Text, nullable=False, default="")
    policy_basis = Column(Text, nullable=False, default="")
    citations = Column(Text, nullable=False, default="")
    source_documents = Column(Text, nullable=False, default="")
    what_changed = Column(Text, nullable=False, default="")
    recompute_note = Column(Text, nullable=False, default="")
    status = Column(String(32), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=utc_now)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)


class ReviewDecision(Base):
    __tablename__ = "review_decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    resolution_id = Column(String(40), nullable=False, index=True)
    decision = Column(String(16), nullable=False)
    reviewer = Column(String(64), nullable=False, default="")
    reviewer_role = Column(String(32), nullable=False, default="")
    feedback_text = Column(Text, nullable=False, default="")
    authority_basis = Column(Text, nullable=False, default="")
    draft_before = Column(Text, nullable=False, default="")
    decided_at = Column(DateTime, nullable=False, default=utc_now)


class LlmCallRecord(Base):
    __tablename__ = "llm_calls"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(40), nullable=False, default="", index=True)
    agent_name = Column(String(64), nullable=False)
    backend = Column(String(32), nullable=False)
    model_name = Column(String(64), nullable=False, default="")
    prompt_characters = Column(Integer, nullable=False, default=0)
    response_characters = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=utc_now)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    user_name = Column(String(64), nullable=False, default="")
    started_at = Column(DateTime, nullable=False, default=utc_now)
    last_active_at = Column(DateTime, nullable=False, default=utc_now)
    remembered_entities = Column(Text, nullable=False, default="{}")


class ChatTurn(Base):
    __tablename__ = "chat_turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    speaker = Column(String(16), nullable=False)
    content = Column(Text, nullable=False, default="")
    intent = Column(String(32), nullable=False, default="")
    citations = Column(Text, nullable=False, default="")
    refused_reason = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=utc_now)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(64), nullable=False, default="")
    rows_read = Column(Integer, nullable=False, default=0)
    rows_scored = Column(Integer, nullable=False, default=0)
    high_risk_found = Column(Integer, nullable=False, default=0)
    watermark_before = Column(Text, nullable=False, default="")
    watermark_after = Column(Text, nullable=False, default="")
    status = Column(String(16), nullable=False, default="")
    message = Column(Text, nullable=False, default="")
    started_at = Column(DateTime, nullable=False, default=utc_now)
    finished_at = Column(DateTime, nullable=True)
