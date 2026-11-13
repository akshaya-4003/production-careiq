"""Request and response shapes. Every request is validated before it reaches a model."""

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from config import config


class LoginRequest(BaseModel):
    user_name: str = Field(min_length=1)
    password: str = Field(min_length=1)

    @field_validator("user_name", "password")
    @classmethod
    def _not_blank(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TicketCreateRequest(BaseModel):
    order_id: str = Field(min_length=3, max_length=32)
    message_text: str = Field(min_length=config.MINIMUM_MESSAGE_CHARACTERS,
                              max_length=config.MAXIMUM_MESSAGE_CHARACTERS)
    source_channel: str = Field(default=config.STAFF_SOURCE_CHANNEL, max_length=32)


class AdvisoryRequest(BaseModel):
    order_id: str = Field(min_length=3, max_length=32)
    ticket_id: str = Field(default="", max_length=32)
    message_text: str = Field(default="", max_length=config.MAXIMUM_MESSAGE_CHARACTERS)
    request_text: str = Field(default=config.DEFAULT_ADVISORY_REQUEST_TEXT, min_length=3,
                              max_length=config.MAXIMUM_MESSAGE_CHARACTERS)

    @field_validator("order_id")
    @classmethod
    def _reference(cls, value):
        value = value.strip().upper()
        if len(value) < 3:
            raise ValueError("an order reference is required")
        return value


class ReviewRequest(BaseModel):
    decision: str
    feedback_text: str = Field(default="", max_length=config.MAXIMUM_MESSAGE_CHARACTERS)
    remedy_type: Optional[str] = None
    approval_delay_hours: Optional[float] = Field(default=None, ge=0)

    @field_validator("decision")
    @classmethod
    def _known_decision(cls, value):
        value = (value or "").strip().upper()
        if value not in config.REVIEW_DECISION_OPTIONS:
            raise ValueError("decision must be one of " + ", ".join(config.REVIEW_DECISION_OPTIONS))
        return value

    @field_validator("remedy_type")
    @classmethod
    def _known_remedy(cls, value):
        if value in (None, ""):
            return None
        if value not in config.VALID_REMEDY_TYPES:
            raise ValueError("remedy_type must be one of " + ", ".join(config.VALID_REMEDY_TYPES))
        return value


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=config.MAXIMUM_MESSAGE_CHARACTERS)
    session_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value):
        if not value.strip():
            raise ValueError("the message is empty")
        return value


# ---------------------------------------------------------------- responses
class HealthResponse(BaseModel):
    status: str
    orders_monitored: int
    policy_chunks: int
    llm_backend: str
    embedding_backend: str


class WhoAmIResponse(BaseModel):
    user_name: str
    display_name: str
    role: str
    permissions: List[str]
    authority_limit_inr: float

    @field_validator("authority_limit_inr")
    @classmethod
    def _finite(cls, value):
        if not math.isfinite(value):
            raise ValueError("an authority limit must be a finite number")
        return float(value)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_name: str
    display_name: str
    role: str


class OrderResponse(BaseModel):
    order_id: str
    summary: Dict[str, Any]
    record: Dict[str, Any]


class RiskResponse(BaseModel):
    order_id: str
    escalation_probability: float
    risk_band: str
    priority: str
    severity_code: str
    customer_tier: str = ""
    order_value_inr: float = 0.0
    transit_dwell_hours: float = 0.0
    delivery_slack_days: float = 0.0
    courier: str = ""
    product_category: str = ""
    payment_method: str = ""


class RiskSummaryResponse(BaseModel):
    orders_scored: int
    counts_by_band: Dict[str, int]
    counts_by_priority: Dict[str, int]
    high_risk_exposure_inr: float
    scored_at: str
    thresholds: Dict[str, float]


class TicketResponse(BaseModel):
    ticket_id: str
    order_id: str
    customer_id: str
    opened_by: str
    assigned_to: str
    source_channel: str
    message_text: str
    issue_mode: str
    issue_confidence: float
    status: str
    opened_at: str
    has_order: bool = True


class ResolutionResponse(BaseModel):
    resolution: Dict[str, Any]
    run_id: str = ""
    status: str = ""
    waiting_for_review: bool = False
    steps: List[Dict[str, Any]] = []
    calls_made: int = 0
    calls_expected: int = config.EXPECTED_LLM_CALLS_PER_RUN
    reviews: List[Dict[str, Any]] = []


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str
    citations: List[str]
    source_documents: List[str]
    ticket_id: str = ""
    issue_mode: str = ""
    order_id: str = ""
    calls_made: int = 0
    # Desk-only fields; blank for a customer.
    intent_basis: str = ""
    intent_reason: str = ""
    order_basis: str = ""
    refused_reason: str = ""
    remembered: Dict[str, Any] = {}


class ChatTurnResponse(BaseModel):
    speaker: str
    content: str
    intent: str = ""
    citations: str = ""
    refused_reason: str = ""
    created_at: str = ""


class CallBudgetResponse(BaseModel):
    run_id: str
    calls_made: int
    calls_expected: int
    over_budget: bool
    calls: List[Dict[str, Any]]
