"""Runs the three trained models and applies the written business rules.

Every business figure in the platform comes from this module. No language model
is consulted anywhere here.
"""

import math

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import config
from data_pipeline.message_cleaning import clean_message_text, is_usable_message
from modeling.model_registry import build_model_input_frame, build_model_input_row


class ToolError(Exception):
    """A tool could not answer. The workflow refuses at that node and says why."""


# ---------------------------------------------------------------- input validation
class EscalationRequest(BaseModel):
    order_id: str = Field(min_length=3)
    order_value_inr: float = Field(ge=0)


class ResolutionEstimateRequest(BaseModel):
    issue_mode: str
    remedy_type: str
    order_value_inr: float = Field(ge=0)

    @field_validator("issue_mode")
    @classmethod
    def _known_mode(cls, value):
        if value not in config.VALID_ISSUE_MODES:
            raise ValueError("issue_mode " + repr(value) + " is not in the catalogue")
        return value

    @field_validator("remedy_type")
    @classmethod
    def _known_remedy(cls, value):
        if value not in config.VALID_REMEDY_TYPES:
            raise ValueError("remedy_type " + repr(value) + " is not in the catalogue")
        return value


class IssueModeRequest(BaseModel):
    message_text: str = Field(min_length=config.MINIMUM_MESSAGE_CHARACTERS)


class RemedyCostRequest(BaseModel):
    remedy_type: str
    order_value_inr: float = Field(ge=0)
    days_late_count: int = Field(ge=0)

    @field_validator("remedy_type")
    @classmethod
    def _known_remedy(cls, value):
        if value not in config.VALID_REMEDY_TYPES or value not in config.REMEDY_COSTS_CASH:
            raise ValueError("remedy_type " + repr(value) + " has no cost rule")
        return value


def _validate(model_class, **fields):
    try:
        return model_class(**fields)
    except ValidationError as error:
        problems = "; ".join("{0}: {1}".format(".".join(str(p) for p in item["loc"]),
                                               item["msg"]) for item in error.errors())
        raise ToolError(model_class.__name__ + " refused: " + problems)


# ---------------------------------------------------------------- escalation
def predict_escalation_probability(risk_model, feature_names, order_record):
    """The chance an order raises a complaint within seven days, from 0 to 1."""
    if not order_record:
        raise ToolError("No order record was supplied to score.")
    _validate(EscalationRequest, order_id=str(order_record.get("order_id", "")),
              order_value_inr=float(order_record.get("order_value_inr") or 0.0))
    try:
        row = build_model_input_row(order_record, feature_names)
        probability = float(risk_model.predict_proba(row)[0, 1])
    except Exception as error:
        raise ToolError("The escalation model could not score order "
                        + str(order_record.get("order_id")) + ": " + str(error))
    if math.isnan(probability) or not 0.0 <= probability <= 1.0:
        raise ToolError("The escalation model returned " + str(probability)
                        + ", which is not a probability.")
    return probability


def classify_risk_band(escalation_probability, thresholds=None):
    """HIGH at or above the high threshold, MEDIUM at or above the medium one, else LOW."""
    high, medium = (config.HIGH_RISK_THRESHOLD, config.MEDIUM_RISK_THRESHOLD) \
        if thresholds is None else thresholds
    if escalation_probability >= high:
        return config.RISK_BAND_HIGH
    if escalation_probability >= medium:
        return config.RISK_BAND_MEDIUM
    return config.RISK_BAND_LOW


_KNOWN_TIERS = sorted(set(tier for tier, _ in config.PRIORITY_MATRIX))
_FALLBACK_TIER = "Standard"


def resolve_priority(customer_tier, risk_band):
    """Read the priority from the nine-cell matrix; an unknown tier takes the Standard row."""
    tier = customer_tier if customer_tier in _KNOWN_TIERS else _FALLBACK_TIER
    return config.PRIORITY_MATRIX[(tier, risk_band)]


def severity_for_band(risk_band):
    """The severity code for a band, one to one."""
    return config.SEVERITY_BY_RISK_BAND[risk_band]


def score_order_book(risk_model, feature_names, orders_frame):
    """Score, band, prioritise and grade every order in one pass, worst first."""
    if orders_frame is None or orders_frame.empty:
        raise ToolError("The order book is empty; there is nothing to score.")
    matrix = build_model_input_frame(orders_frame, feature_names)
    probabilities = np.clip(risk_model.predict_proba(matrix)[:, 1], 0.0, 1.0)
    scored = orders_frame.reset_index(drop=True).copy()
    scored["escalation_probability"] = np.round(probabilities.astype(float), 4)
    scored["risk_band"] = [classify_risk_band(p) for p in scored["escalation_probability"]]
    tiers = scored["customer_tier"] if "customer_tier" in scored.columns else \
        [config.UNKNOWN_CATEGORY_VALUE] * len(scored)
    scored["priority"] = [resolve_priority(tier, band)
                          for tier, band in zip(tiers, scored["risk_band"])]
    scored["severity_code"] = [severity_for_band(band) for band in scored["risk_band"]]
    return scored.sort_values("escalation_probability", ascending=False,
                              kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- issue mode
def detect_issue_mode(issue_model, vectorizer, message_text):
    """Clean the message, classify it, and return the mode and the model's own confidence."""
    _validate(IssueModeRequest, message_text=str(message_text or ""))
    cleaned = clean_message_text(message_text)
    if not is_usable_message(cleaned):
        raise ToolError("The message is too short to read once cleaned: " + repr(cleaned))
    matrix = vectorizer.transform([cleaned])
    probabilities = issue_model.predict_proba(matrix)[0]
    best = int(np.argmax(probabilities))
    mode = str(issue_model.classes_[best])
    if mode not in config.VALID_ISSUE_MODES:
        raise ToolError("The classifier answered " + repr(mode) + ", outside the catalogue.")
    ranked = sorted(zip(issue_model.classes_, probabilities), key=lambda pair: -pair[1])
    return {
        "issue_mode": mode,
        "confidence": round(float(probabilities[best]), 4),
        "cleaned_text": cleaned,
        "alternatives": [{"issue_mode": str(name), "confidence": round(float(value), 4)}
                         for name, value in ranked[1:3]],
    }


# ---------------------------------------------------------------- resolution estimate
def build_planned_resolution_record(order, risk_facts, job_context):
    """A resolution not yet carried out, carrying every column the resolution model reads."""
    from modeling.model_registry import load_feature_schema

    schema = load_feature_schema(config.RESOLUTION_SCHEMA_FILE)
    sources = [job_context or {}, risk_facts or {}, order or {},
               config.RESOLUTION_PLANNING_ASSUMPTIONS]
    record = {}
    for name in schema:
        for source in sources:
            if name in source and source[name] is not None:
                record[name] = source[name]
                break
    # Every value in RESOLUTION_PLANNING_ASSUMPTIONS is an assumption, not a fact.
    record["assumptions"] = sorted(name for name in config.RESOLUTION_PLANNING_ASSUMPTIONS
                                   if name not in (job_context or {}))
    return record


def estimate_resolution_hours(resolution_model, feature_names, planned_record):
    """Expected hours for the ticket; always a positive number."""
    try:
        row = build_model_input_row(planned_record, feature_names)
        hours = float(resolution_model.predict(row)[0])
    except Exception as error:
        raise ToolError("The resolution model could not score the planned resolution: "
                        + str(error))
    if math.isnan(hours):
        raise ToolError("The resolution model returned no number.")
    lowest = config.TICKET_RANGE_LIMITS[config.RESOLUTION_TARGET_COLUMN][0]
    return round(max(hours, lowest), 2)


# ---------------------------------------------------------------- cash rules
def late_delivery_compensation(days_late_count):
    """NRC-SHIP-002 store credit for a delivery this late, never above the cap."""
    try:
        days = int(days_late_count)
    except (TypeError, ValueError):
        return 0.0
    if days <= 0:
        return 0.0
    owed = 0.0
    for start, end, credit in config.LATE_DELIVERY_COMPENSATION_TIERS:
        if start <= days <= end:
            owed = float(credit)
        elif days > end:
            owed = max(owed, float(credit))
    return min(owed, config.LATE_DELIVERY_COMPENSATION_CAP_INR)


def remedy_cash_cost(remedy_type, order_value_inr, days_late_count=0):
    """The cash a remedy commits to the customer, by the rule config names for it."""
    request = _validate(RemedyCostRequest, remedy_type=str(remedy_type),
                        order_value_inr=float(order_value_inr),
                        days_late_count=max(0, int(days_late_count or 0)))
    rule = config.REMEDY_COSTS_CASH[request.remedy_type]
    if rule == "order_value":
        return round(request.order_value_inr, 2)
    if rule == "partial_share":
        return round(request.order_value_inr * config.PARTIAL_REFUND_MAXIMUM_SHARE, 2)
    if rule == "lateness_tier":
        return late_delivery_compensation(request.days_late_count)
    if rule == "goodwill_default":
        return float(config.GOODWILL_CREDIT_DEFAULT_INR)
    if rule == "none":
        return 0.0
    raise ToolError("Remedy " + request.remedy_type + " has an unrecognised cost rule " + rule)


def default_remedy_for(issue_mode):
    """The published default remedy for an issue mode, always inside the catalogue."""
    remedy = config.DEFAULT_REMEDY_BY_ISSUE.get(issue_mode)
    if remedy not in config.VALID_REMEDY_TYPES:
        raise ToolError("No published default remedy for issue mode " + repr(issue_mode))
    return remedy
