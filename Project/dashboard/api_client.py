"""One function per REST endpoint. The dashboard holds no rule and reads no file of its own.

Every function takes the token and returns the parsed body. A refusal from the
service is raised as ApiError carrying the status and the service's own message,
so a page can show why rather than showing nothing.
"""

import requests

from config import config


class ApiError(Exception):
    def __init__(self, status_code, message):
        Exception.__init__(self, message)
        self.status_code = status_code
        self.message = message


class ServiceUnavailable(ApiError):
    pass


def _message_from(response):
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or response.reason
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, list):  # a validation error list
        return "; ".join("{0}: {1}".format(".".join(str(p) for p in item.get("loc", [])[1:]),
                                           item.get("msg", "")) for item in detail)
    if isinstance(detail, dict) and "refused_at" in detail:
        return "Refused at " + str(detail["refused_at"]) + ": " + str(detail.get("reason", ""))
    return str(detail)


def _call(method, path, token=None, base_url=None, **kwargs):
    url = (base_url or config.API_BASE_URL).rstrip("/") + path
    headers = {"Authorization": "Bearer " + token} if token else {}
    try:
        response = requests.request(method, url, headers=headers,
                                    timeout=config.API_REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as error:
        raise ServiceUnavailable(0, "The CareIQ service at " + (base_url or config.API_BASE_URL)
                                 + " is not reachable (" + type(error).__name__
                                 + "). Start it with: python3 run_api_service.py")
    if response.status_code >= 400:
        raise ApiError(response.status_code, _message_from(response))
    return response.json()


# ---------------------------------------------------------------- service and identity
def health():
    return _call("GET", "/health")


def login(user_name, password):
    return _call("POST", "/auth/login", json={"user_name": user_name, "password": password})


def who_am_i(token):
    return _call("GET", "/auth/me", token)


def catalogue(token):
    return _call("GET", "/catalogue", token)


# ---------------------------------------------------------------- orders and risk
def get_order(token, order_id):
    return _call("GET", "/orders/" + order_id.strip(), token)


def get_order_risk(token, order_id):
    return _call("GET", "/orders/" + order_id.strip() + "/risk", token)


def risk_summary(token):
    return _call("GET", "/risk/summary", token)


def risk_orders(token, risk_band="", limit=200):
    return _call("GET", "/risk/orders", token, params={"risk_band": risk_band, "limit": limit})


# ---------------------------------------------------------------- tickets and advisories
def create_ticket(token, order_id, message_text, source_channel=None):
    body = {"order_id": order_id, "message_text": message_text}
    if source_channel:
        body["source_channel"] = source_channel
    return _call("POST", "/tickets", token, json=body)


def list_tickets(token, status="", all_queues=False, limit=200):
    return _call("GET", "/tickets", token, params={"status": status, "all_queues": all_queues,
                                                    "limit": limit})


def start_advisory(token, order_id, ticket_id="", message_text="", request_text=None):
    body = {"order_id": order_id, "ticket_id": ticket_id, "message_text": message_text}
    if request_text:
        body["request_text"] = request_text
    return _call("POST", "/advisories", token, json=body)


def list_resolutions(token, status="", limit=200):
    return _call("GET", "/resolutions", token, params={"status": status, "limit": limit})


def get_resolution(token, resolution_id):
    return _call("GET", "/resolutions/" + resolution_id, token)


def review_resolution(token, resolution_id, decision, feedback_text="", remedy_type=None,
                      approval_delay_hours=None):
    body = {"decision": decision, "feedback_text": feedback_text}
    if remedy_type:
        body["remedy_type"] = remedy_type
    if approval_delay_hours is not None:
        body["approval_delay_hours"] = approval_delay_hours
    return _call("POST", "/resolutions/" + resolution_id + "/review", token, json=body)


def resolution_reviews(token, resolution_id):
    return _call("GET", "/resolutions/" + resolution_id + "/reviews", token)


# ---------------------------------------------------------------- chat
def chat(token, message, session_id=None):
    return _call("POST", "/chat", token, json={"message": message, "session_id": session_id})


def chat_transcript(token, session_id):
    return _call("GET", "/chat/" + session_id, token)


# ---------------------------------------------------------------- audit, pipeline, evaluation
def list_runs(token, limit=200):
    return _call("GET", "/runs", token, params={"limit": limit})


def run_calls(token, run_id):
    return _call("GET", "/runs/" + run_id + "/calls", token)


def pipeline_runs(token):
    return _call("GET", "/pipeline/runs", token)


def pipeline_rescore(token):
    return _call("POST", "/pipeline/rescore", token)


def evaluation_report(token):
    return _call("GET", "/evaluation", token)


def model_metrics(token):
    return _call("GET", "/metrics/models", token)


def data_quality(token):
    return _call("GET", "/metrics/data-quality", token)
