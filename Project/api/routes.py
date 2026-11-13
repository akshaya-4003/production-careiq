"""The CareIQ REST service. Every figure the dashboard shows comes from here.

Access control is enforced here, per endpoint, by permission. The heavy objects
(three models, the policy index, the graph) are built once when the application
is created, and the order book is scored once at start up and on rescore.
"""

import json
import os
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from agentic import nodes
from agentic.graph import (
    build_advisory_graph,
    build_initial_state,
    is_waiting_for_review,
    resume_after_review,
    start_advisory_run,
    thread_config_for,
)
from api import schemas
from chatbot.assistant import CareAssistant, new_session_id
from config import config
from llm.client import load_llm_client
from modeling.model_registry import load_feature_schema, load_model_artifact
from persistence import repository
from persistence.database import get_database_session
from security.access_control import (
    AccessControl,
    authority_limit_for,
    current_user,
    lowest_role_that_may_commit,
    require_permission,
)
from security.passwords import PasswordVault
from tools.order_data_tools import (
    days_late,
    describe_order_for_person,
    fetch_order,
    load_live_orders,
    order_book_records,
)
from tools.prediction_tools import ToolError, detect_issue_mode, remedy_cash_cost, score_order_book
from tools.retrieval_tools import open_knowledge


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ServiceContext(object):
    """Everything loaded once for the life of the service."""

    def __init__(self):
        self.models = {
            "escalation": load_model_artifact(config.ESCALATION_MODEL_FILE),
            "escalation_schema": load_feature_schema(config.ESCALATION_SCHEMA_FILE),
            "resolution": load_model_artifact(config.RESOLUTION_MODEL_FILE),
            "resolution_schema": load_feature_schema(config.RESOLUTION_SCHEMA_FILE),
            "issue": load_model_artifact(config.ISSUE_MODE_MODEL_FILE),
            "vectorizer": load_model_artifact(config.TEXT_VECTORIZER_FILE),
        }
        self.knowledge = open_knowledge()
        self.client = load_llm_client()
        self.graph = build_advisory_graph()
        self.advisory_services = nodes.configure_services(
            nodes.AdvisoryServices(client=self.client, knowledge=self.knowledge))
        self.advisory_services.use_models(self.models)
        self.assistant = CareAssistant(self.client, self.knowledge, None, self.models["issue"],
                                       self.models["vectorizer"], file_tickets=True)
        self.rescore()

    def rescore(self):
        """Score the whole book in one call; the risk pages are then table reads."""
        self.orders = load_live_orders()
        self.scored = score_order_book(self.models["escalation"],
                                       self.models["escalation_schema"], self.orders)
        self.scored_by_order = {row["order_id"]: row for row in order_book_records(self.scored)}
        self.scored_at = _now()
        self.advisory_services.replace_orders(self.orders)
        self.assistant.orders = self.orders
        return len(self.scored)


def _risk_row(row):
    return {key: row.get(key) for key in schemas.RiskResponse.model_fields if key in row}


def _read_json_report(path, command):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"available": False, "path": path,
                "message": "No report yet. Run " + command + " to produce it."}
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    return {"available": True, "path": path, "report": report}


def _ticket_view(ticket):
    view = dict(ticket)
    view["has_order"] = bool(ticket.get("order_id"))
    return view


def create_api_application(application_title=None):
    """Build the application and attach every endpoint."""
    application = FastAPI(title=application_title or config.API_TITLE,
                          description="CareIQ - customer experience and order risk platform "
                                      "for Northlane Retail Co. Sign in with POST /auth/login "
                                      "and send the token as 'Authorization: Bearer <token>'.")
    application.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS,
                               allow_methods=["*"], allow_headers=["*"])
    context = ServiceContext()
    application.state.context = context
    guard, vault = AccessControl(), PasswordVault()

    # ------------------------------------------------------------ health and identity
    @application.get("/health", response_model=schemas.HealthResponse, tags=["service"])
    def health():
        return {"status": "ok", "orders_monitored": int(len(context.orders)),
                "policy_chunks": context.knowledge.count(), "llm_backend": context.client.name,
                "embedding_backend": context.knowledge.backend_name()}

    @application.post("/auth/login", response_model=schemas.TokenResponse, tags=["auth"])
    def login(request: schemas.LoginRequest, session=Depends(get_database_session)):
        user = repository.find_user(session, request.user_name)
        if not user or not vault.matches_stored_hash(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Unrecognised user name or password.")
        token = guard.issue_token(user["user_name"], user["role"], user["display_name"])
        return {"access_token": token, "token_type": "bearer", "user_name": user["user_name"],
                "display_name": user["display_name"], "role": user["role"]}

    @application.get("/auth/me", response_model=schemas.WhoAmIResponse, tags=["auth"])
    def who_am_i(user=Depends(current_user)):
        return dict(user, authority_limit_inr=authority_limit_for(user["role"]))

    @application.get("/catalogue", tags=["service"])
    def catalogue(user=Depends(current_user)):
        """The closed lists the dashboard offers in its controls, so it holds none itself."""
        return {"remedy_types": config.VALID_REMEDY_TYPES,
                "issue_modes": config.VALID_ISSUE_MODES,
                "review_decisions": config.REVIEW_DECISION_OPTIONS,
                "risk_bands": config.RISK_BAND_OPTIONS,
                "risk_band_colours": config.RISK_BAND_BADGE_COLOURS,
                "review_decision_colours": config.REVIEW_DECISION_BADGE_COLOURS,
                "resolution_statuses": config.RESOLUTION_STATUS_OPTIONS,
                "ticket_statuses": config.TICKET_STATUS_OPTIONS,
                "default_request_text": config.DEFAULT_ADVISORY_REQUEST_TEXT,
                "planning_assumptions": config.RESOLUTION_PLANNING_ASSUMPTIONS,
                "expected_calls_per_run": config.EXPECTED_LLM_CALLS_PER_RUN,
                "order_reference_format": config.ORDER_REFERENCE_FORMAT,
                "authority_limits": {role: authority_limit_for(role)
                                     for role in config.ROLE_OPTIONS}}

    # ------------------------------------------------------------ orders and risk
    @application.get("/orders/{order_id}", response_model=schemas.OrderResponse, tags=["orders"])
    def get_order(order_id: str, user=Depends(require_permission("order_read"))):
        order = fetch_order(context.orders, order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Order " + order_id + " is not on file.")
        return {"order_id": order["order_id"], "summary": describe_order_for_person(order),
                "record": order}

    @application.get("/orders/{order_id}/risk", response_model=schemas.RiskResponse,
                     tags=["orders"])
    def get_order_risk(order_id: str, user=Depends(require_permission("order_read"))):
        order = fetch_order(context.orders, order_id)
        if not order or order["order_id"] not in context.scored_by_order:
            raise HTTPException(status_code=404, detail="Order " + order_id + " is not on file.")
        return _risk_row(context.scored_by_order[order["order_id"]])

    @application.get("/risk/summary", response_model=schemas.RiskSummaryResponse, tags=["risk"])
    def risk_summary(user=Depends(require_permission("risk_dashboard"))):
        scored = context.scored
        high = scored[scored["risk_band"] == config.RISK_BAND_HIGH]
        return {
            "orders_scored": int(len(scored)),
            "counts_by_band": {band: int((scored["risk_band"] == band).sum())
                               for band in config.RISK_BAND_OPTIONS},
            "counts_by_priority": {str(k): int(v) for k, v in
                                   scored["priority"].value_counts().sort_index().items()},
            "high_risk_exposure_inr": round(float(high["order_value_inr"].sum()), 2),
            "scored_at": context.scored_at,
            "thresholds": {"high": config.HIGH_RISK_THRESHOLD,
                           "medium": config.MEDIUM_RISK_THRESHOLD},
        }

    @application.get("/risk/orders", tags=["risk"])
    def risk_orders(risk_band: str = Query(default=""), limit: int = Query(default=100, ge=1,
                                                                           le=5000),
                    user=Depends(require_permission("risk_dashboard"))):
        rows = order_book_records(context.scored)
        if risk_band:
            wanted = risk_band.strip().upper()
            if wanted not in config.RISK_BAND_OPTIONS:
                raise HTTPException(status_code=422, detail="risk_band must be one of "
                                    + ", ".join(config.RISK_BAND_OPTIONS))
            rows = [row for row in rows if row["risk_band"] == wanted]
        return [_risk_row(row) for row in rows[:limit]]

    # ------------------------------------------------------------ tickets
    @application.post("/tickets", response_model=schemas.TicketResponse, tags=["tickets"])
    def create_ticket(request: schemas.TicketCreateRequest,
                      user=Depends(require_permission("own_queue")),
                      session=Depends(get_database_session)):
        order = fetch_order(context.orders, request.order_id)
        if not order:
            raise HTTPException(status_code=404,
                                detail="Order " + request.order_id + " is not on file.")
        try:
            found = detect_issue_mode(context.models["issue"], context.models["vectorizer"],
                                      request.message_text)
        except ToolError as error:
            raise HTTPException(status_code=422, detail=str(error))
        ticket = repository.insert_ticket(session, {
            "ticket_id": repository.next_ticket_identifier(session),
            "order_id": order["order_id"], "customer_id": order.get("customer_id", ""),
            "opened_by": user["user_name"], "assigned_to": user["user_name"],
            "source_channel": request.source_channel, "message_text": request.message_text,
            "issue_mode": found["issue_mode"], "issue_confidence": found["confidence"],
            "status": config.TICKET_STATUS_OPEN})
        return _ticket_view(ticket)

    @application.get("/tickets", tags=["tickets"])
    def list_tickets(status: str = Query(default=""), all_queues: bool = Query(default=False),
                     limit: int = Query(default=config.API_DEFAULT_LIST_LIMIT, ge=1, le=1000),
                     user=Depends(require_permission("own_queue")),
                     session=Depends(get_database_session)):
        # The filter is applied here, never trusted from the request.
        if all_queues and "all_queues" in user["permissions"]:
            rows = repository.list_tickets(session, status=status or None, limit=limit)
        else:
            rows = repository.list_tickets(session, status=status or None,
                                           assigned_to=user["user_name"], limit=limit,
                                           include_unassigned=True)
        return [_ticket_view(row) for row in rows]

    # ------------------------------------------------------------ advisories
    def _resolution_response(session, resolution_id, state=None):
        resolution = repository.fetch_resolution(session, resolution_id)
        run = repository.fetch_agent_run(session, resolution.get("run_id", "")) if resolution \
            else {}
        calls = repository.fetch_call_records_for_run(session, resolution.get("run_id", "")) \
            if resolution else []
        return {"resolution": resolution, "run_id": resolution.get("run_id", ""),
                "status": run.get("status", ""),
                "waiting_for_review": resolution.get("status") == config.RESOLUTION_STATUS_PENDING,
                "steps": (state or {}).get("steps", []), "calls_made": len(calls),
                "calls_expected": config.EXPECTED_LLM_CALLS_PER_RUN,
                "reviews": repository.fetch_decisions_for_resolution(session, resolution_id)}

    @application.post("/advisories", response_model=schemas.ResolutionResponse,
                      tags=["advisories"])
    def start_advisory(request: schemas.AdvisoryRequest,
                       user=Depends(require_permission("advisory_start")),
                       session=Depends(get_database_session)):
        # Checked before the graph starts, so a request that cannot run costs nothing.
        if not fetch_order(context.orders, request.order_id):
            raise HTTPException(status_code=422, detail={
                "refused_at": "load_context",
                "reason": "order " + request.order_id + " is not on file"})
        message = request.message_text
        if request.ticket_id:
            ticket = repository.find_ticket(session, request.ticket_id)
            if not ticket:
                raise HTTPException(status_code=422, detail="Ticket " + request.ticket_id
                                    + " does not exist.")
            message = message or ticket.get("message_text", "")
        if len(message.strip()) < config.MINIMUM_MESSAGE_CHARACTERS:
            raise HTTPException(status_code=422, detail={
                "refused_at": "identify_issue",
                "reason": "an advisory needs the customer's message to read the issue from"})
        thread_id = "THREAD-" + uuid.uuid4().hex[:16].upper()
        state = start_advisory_run(context.graph, build_initial_state({
            "order_id": request.order_id, "ticket_id": request.ticket_id,
            "message_text": message, "request_text": request.request_text,
            "requested_by": user["user_name"]}, thread_id), thread_config_for(thread_id))
        if state.get("refused"):
            node, _, reason = state["refused"].partition(": ")
            raise HTTPException(status_code=422, detail={"refused_at": node, "reason": reason,
                                                         "steps": state.get("steps", [])})
        response = _resolution_response(session, state["resolution_id"], state)
        response["waiting_for_review"] = is_waiting_for_review(state)
        return response

    @application.get("/resolutions", tags=["advisories"])
    def list_resolutions(status: str = Query(default=""),
                         limit: int = Query(default=config.API_DEFAULT_LIST_LIMIT, ge=1, le=1000),
                         user=Depends(require_permission("own_queue")),
                         session=Depends(get_database_session)):
        return repository.list_resolutions(session, status=status or None, limit=limit)

    @application.get("/resolutions/{resolution_id}", response_model=schemas.ResolutionResponse,
                     tags=["advisories"])
    def get_resolution(resolution_id: str, user=Depends(require_permission("own_queue")),
                       session=Depends(get_database_session)):
        if not repository.fetch_resolution(session, resolution_id):
            raise HTTPException(status_code=404, detail="Resolution " + resolution_id
                                + " does not exist.")
        return _resolution_response(session, resolution_id)

    @application.post("/resolutions/{resolution_id}/review",
                      response_model=schemas.ResolutionResponse, tags=["advisories"])
    def review_resolution(resolution_id: str, request: schemas.ReviewRequest,
                          user=Depends(require_permission("review_decide")),
                          session=Depends(get_database_session)):
        resolution = repository.fetch_resolution(session, resolution_id)
        if not resolution:
            raise HTTPException(status_code=404, detail="Resolution " + resolution_id
                                + " does not exist.")
        if resolution["status"] != config.RESOLUTION_STATUS_PENDING:
            raise HTTPException(status_code=409, detail="Resolution " + resolution_id
                                + " has already been reviewed (" + resolution["status"] + ").")
        if request.decision != config.REVIEW_DECISION_REJECT:
            remedy = request.remedy_type or resolution["remedy_type"]
            order = fetch_order(context.orders, resolution["order_id"])
            amount = remedy_cash_cost(remedy, float(order.get("order_value_inr") or 0.0),
                                      days_late(order)) if order else \
                resolution["remedy_amount_inr"]
            limit = authority_limit_for(user["role"])
            if amount > limit:
                raise HTTPException(status_code=403, detail=(
                    "INR {0:,.2f} is above the {1} authority limit of INR {2:,.2f}; it needs "
                    "the {3} role.").format(amount, user["role"], limit,
                                            lowest_role_that_may_commit(amount) or "nobody"))
        run = repository.fetch_agent_run(session, resolution["run_id"])
        thread = thread_config_for(run.get("thread_id", ""))
        snapshot = context.graph.get_state(thread)
        if not snapshot.next:
            raise HTTPException(status_code=409, detail="The run behind " + resolution_id
                                + " is not waiting for a review.")
        decision = {"decision": request.decision, "feedback_text": request.feedback_text,
                    "remedy_type": request.remedy_type,
                    "approval_delay_hours": request.approval_delay_hours,
                    "reviewer": user["user_name"], "reviewer_role": user["role"],
                    "authority_basis": "Decided under the {0} authority limit of INR {1:,.2f}"
                    .format(user["role"], authority_limit_for(user["role"]))}
        state = resume_after_review(context.graph, thread, decision)
        if state.get("refused"):
            node, _, reason = state["refused"].partition(": ")
            raise HTTPException(status_code=422, detail={"refused_at": node, "reason": reason})
        session.expire_all()
        return _resolution_response(session, resolution_id, state)

    @application.get("/resolutions/{resolution_id}/reviews", tags=["advisories"])
    def resolution_reviews(resolution_id: str, user=Depends(require_permission("own_queue")),
                           session=Depends(get_database_session)):
        return repository.fetch_decisions_for_resolution(session, resolution_id)

    # ------------------------------------------------------------ chat
    @application.post("/chat", response_model=schemas.ChatResponse, tags=["chat"])
    def chat(request: schemas.ChatRequest, user=Depends(require_permission("chat")),
             session=Depends(get_database_session)):
        session_id = request.session_id or new_session_id()
        existing = repository.find_chat_session(session, session_id)
        if existing and existing.get("user_name") not in ("", user["user_name"]):
            raise HTTPException(status_code=403, detail="That conversation belongs to someone else.")
        answered = context.assistant.handle_message(session, session_id, request.message,
                                                    user["user_name"], user["role"])
        response = dict(answered, session_id=session_id,
                        calls_made=len(answered["call_records"]))
        if user["role"] == config.ROLE_CUSTOMER:
            # Desk notes describe how the desk reasons; a customer sees the answer only.
            for field in ("intent_basis", "intent_reason", "order_basis", "refused_reason"):
                response[field] = ""
            response["remembered"] = {}
        return response

    @application.get("/chat/{session_id}", tags=["chat"])
    def chat_transcript(session_id: str, user=Depends(require_permission("chat")),
                        session=Depends(get_database_session)):
        existing = repository.find_chat_session(session, session_id)
        if not existing:
            raise HTTPException(status_code=404, detail="No conversation " + session_id + ".")
        if existing.get("user_name") not in ("", user["user_name"]) and \
                "all_queues" not in user["permissions"]:
            raise HTTPException(status_code=403, detail="That conversation belongs to someone else.")
        turns = repository.recent_chat_turns(session, session_id)
        if user["role"] == config.ROLE_CUSTOMER:
            for turn in turns:
                turn["refused_reason"] = ""
        return turns

    # ------------------------------------------------------------ audit
    @application.get("/runs", tags=["audit"])
    def list_runs(limit: int = Query(default=config.API_DEFAULT_LIST_LIMIT, ge=1, le=1000),
                  user=Depends(require_permission("audit_read")),
                  session=Depends(get_database_session)):
        counts = repository.count_calls_by_run(session)
        runs = repository.list_agent_runs(session, limit=limit)
        for run in runs:
            run["calls_made"] = counts.get(run["run_id"], 0)
            run["calls_expected"] = config.EXPECTED_LLM_CALLS_PER_RUN
            run["over_budget"] = run["calls_made"] > config.EXPECTED_LLM_CALLS_PER_RUN
            resolution = repository.fetch_resolution_for_run(session, run["run_id"])
            run["resolution_id"] = resolution.get("resolution_id", "")
            run["resolution_status"] = resolution.get("status", "")
        return runs

    @application.get("/runs/{run_id}/calls", response_model=schemas.CallBudgetResponse,
                     tags=["audit"])
    def run_calls(run_id: str, user=Depends(require_permission("audit_read")),
                  session=Depends(get_database_session)):
        if not repository.fetch_agent_run(session, run_id):
            raise HTTPException(status_code=404, detail="Run " + run_id + " does not exist.")
        calls = repository.fetch_call_records_for_run(session, run_id)
        return {"run_id": run_id, "calls_made": len(calls),
                "calls_expected": config.EXPECTED_LLM_CALLS_PER_RUN,
                "over_budget": len(calls) > config.EXPECTED_LLM_CALLS_PER_RUN, "calls": calls}

    # ------------------------------------------------------------ pipeline
    @application.get("/pipeline/runs", tags=["pipeline"])
    def pipeline_runs(user=Depends(require_permission("pipeline_read")),
                      session=Depends(get_database_session)):
        return repository.recent_ingestion_runs(session)

    @application.post("/pipeline/rescore", tags=["pipeline"])
    def pipeline_rescore(user=Depends(require_permission("pipeline_read"))):
        count = context.rescore()
        return {"orders_scored": count, "scored_at": context.scored_at}

    # ------------------------------------------------------------ evaluation
    @application.get("/evaluation", tags=["evaluation"])
    def evaluation(user=Depends(require_permission("evaluation_read"))):
        return _read_json_report(config.EVALUATION_REPORT_FILE, "python3 run_evaluation.py")

    @application.get("/metrics/models", tags=["evaluation"])
    def model_metrics(user=Depends(require_permission("evaluation_read"))):
        return _read_json_report(config.MODEL_METRICS_FILE, "python3 run_model_training.py")

    @application.get("/metrics/data-quality", tags=["evaluation"])
    def data_quality(user=Depends(require_permission("evaluation_read"))):
        return _read_json_report(config.DATA_QUALITY_REPORT_FILE,
                                 "python3 run_data_preparation.py")

    return application
