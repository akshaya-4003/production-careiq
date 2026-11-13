"""Every fixed value the CareIQ platform runs on.
 
Each value here is a contract: a published Northlane policy, a platform setting, or a
wording the service is required to produce. Modules import from here rather than
retyping a figure, so one change reaches every module that depends on it.
 
An auditor should be able to read this file and check the business rules without
reading any other Python.
 
Target runtime: Python 3.9.1. Nothing in this project may use syntax newer than 3.9
(no `X | Y` unions, no `match`, no `datetime.UTC`).
"""
 
import os
 
# ---------------------------------------------------------------- .env
# Every setting below reads os.environ at import time, and this module is the
# first thing every other module imports. So .env has to be loaded here, before
# the first os.environ.get runs, or the file is read too late to matter and the
# platform silently falls back to the offline backends with a key sitting in .env.
#
# The path is anchored to this file rather than the working directory, so the
# same .env is found whether a script is started from the project root, from the
# test runner, or from uvicorn's reloader.
#
# python-dotenv is a pinned dependency, but a missing one must not stop CareIQ
# booting: the defaults below are all working values.
try:
    from dotenv import load_dotenv
 
    load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass
 
# ---------------------------------------------------------------- folders
DATA_DIRECTORY = "data"
DOCS_DIRECTORY = "data/docs"
PROCESSED_DIRECTORY = "processed"
DATABASE_DIRECTORY = "database"
OUTPUT_DIRECTORY = "output"
MODEL_DIRECTORY = "output/models"
GOLDEN_SET_DIRECTORY = "evaluation/golden_sets"
 
# ---------------------------------------------------------------- raw input files
# Seven tables, as they would arrive from the warehouse. Deliberately dirty.
RAW_ORDERS_TRAIN_FILE = "data/orders_train.csv"
RAW_ORDERS_EVAL_FILE = "data/orders_eval.csv"
RAW_TICKETS_TRAIN_FILE = "data/tickets_train.csv"
RAW_TICKETS_EVAL_FILE = "data/tickets_eval.csv"
RAW_MESSAGES_FILE = "data/messages.csv"
RAW_ORDER_EVENTS_FILE = "data/order_events.csv"
RAW_SHIPMENTS_FILE = "data/shipments.csv"
RAW_CUSTOMERS_FILE = "data/customers.csv"
RAW_PRODUCTS_FILE = "data/products.csv"
 
# ---------------------------------------------------------------- live serving feed
# One row per in-flight order, as a condition-monitoring system would show this
# morning. Deliberately NOT the evaluation set: the eval file carries the outcome
# column, and an application must never be handed the answer it is meant to predict.
LIVE_ORDERS_FEED_FILE = "data/input/live_orders_feed.csv"
 
# ---------------------------------------------------------------- cleaned output files
CLEAN_ORDERS_TRAIN_FILE = "processed/orders_train_clean.csv"
CLEAN_ORDERS_EVAL_FILE = "processed/orders_eval_clean.csv"
CLEAN_TICKETS_TRAIN_FILE = "processed/tickets_train_clean.csv"
CLEAN_TICKETS_EVAL_FILE = "processed/tickets_eval_clean.csv"
CLEAN_MESSAGES_FILE = "processed/messages_clean.csv"
# The live feed cleaned the same way the training data was. Serving a raw feed to
# a model trained on cleaned data is the classic training/serving skew bug: the
# model was fitted on the number 1231.97 and would be handed the text
# "1,231.97 INR", which fails loudly if you are lucky and silently if you are not.
CLEAN_LIVE_ORDERS_FILE = "processed/live_orders_clean.csv"
 
# ---------------------------------------------------------------- saved model files
ESCALATION_MODEL_FILE = "output/models/escalation_risk_model.pkl"
RESOLUTION_MODEL_FILE = "output/models/resolution_time_model.pkl"
ISSUE_MODE_MODEL_FILE = "output/models/issue_mode_model.pkl"
TEXT_VECTORIZER_FILE = "output/models/text_vectorizer.pkl"
ESCALATION_SCHEMA_FILE = "output/models/escalation_features.json"
RESOLUTION_SCHEMA_FILE = "output/models/resolution_features.json"
 
# ---------------------------------------------------------------- report files
DATA_QUALITY_REPORT_FILE = "output/data_quality_report.json"
MODEL_METRICS_FILE = "output/model_metrics.json"
EVALUATION_REPORT_FILE = "output/evaluation_report.json"
INGESTION_LOG_FILE = "output/ingestion_runs.json"
 
# ---------------------------------------------------------------- databases
APPLICATION_DATABASE_FILE = "database/careiq.db"
APPLICATION_DATABASE_URL = "sqlite:///./database/careiq.db"
CHECKPOINT_DATABASE_FILE = "database/checkpoint.db"
VECTOR_STORE_DIRECTORY = "database/chroma_store"
VECTOR_COLLECTION_NAME = "northlane_policy"
SERVING_STORE_FILE = "database/serving_orders.csv"
WATERMARK_FILE = "database/ingestion_watermark.json"
 
# ---------------------------------------------------------------- missing value tokens
# Every way "missing" is written in the raw extracts.
NULL_TOKEN_LIST = [
    "", " ", "  ", "   ", "NA", "N/A", "n/a", "null", "NULL",
    "-", "?", "NaN", "nan", "none", "None", "#N/A", "\\N",
]
 
# ---------------------------------------------------------------- spelling corrections
# Messy value in lower case, mapped to the value the platform uses.
PRODUCT_CATEGORY_MAPPING = {
    "electronics": "Electronics", "electronic": "Electronics", "elec": "Electronics",
    "home goods": "Home Goods", "home-goods": "Home Goods", "homegoods": "Home Goods",
    "apparel": "Apparel", "clothing": "Apparel", "appar": "Apparel",
    "footwear": "Footwear", "shoes": "Footwear",
    "kitchen": "Kitchen", "kitchenware": "Kitchen",
}
 
CHANNEL_MAPPING = {
    "web": "Web", "website": "Web", "desktop": "Web",
    "app": "App", "mobile app": "App", "mobile": "App",
    "store": "Store", "in-store": "Store", "retail": "Store",
}
 
PAYMENT_METHOD_MAPPING = {
    "prepaid": "Prepaid", "pre-paid": "Prepaid", "card": "Prepaid", "upi": "Prepaid",
    "cod": "COD", "cash on delivery": "COD", "cash-on-delivery": "COD",
    "emi": "EMI", "e.m.i": "EMI", "instalment": "EMI",
}
 
CUSTOMER_TIER_MAPPING = {
    "platinum": "Platinum", "plat": "Platinum", "tier-1": "Platinum",
    "gold": "Gold", "gld": "Gold", "tier-2": "Gold",
    "standard": "Standard", "std": "Standard", "tier-3": "Standard", "basic": "Standard",
}
 
COURIER_MAPPING = {
    "swiftlane": "SwiftLane", "swift lane": "SwiftLane", "swift-lane": "SwiftLane",
    "bharatpost": "BharatPost", "bharat post": "BharatPost", "b-post": "BharatPost",
    "metrodash": "MetroDash", "metro dash": "MetroDash", "m-dash": "MetroDash",
}
 
WAREHOUSE_MAPPING = {
    "pune": "Pune", "pune-w1": "Pune",
    "chennai": "Chennai", "chennai-w2": "Chennai",
    "delhi": "Delhi", "delhi-w3": "Delhi",
    "kolkata": "Kolkata", "kolkata-w4": "Kolkata",
}
 
# Values that count as True. Everything else is False.
BOOLEAN_TRUE_TOKENS = ["y", "yes", "true", "t", "1"]
 
# ---------------------------------------------------------------- issue modes
# The closed catalogue of things that actually go wrong. A complaint that does not
# fit one of these is escalated to a person rather than guessed at.
VALID_ISSUE_MODES = [
    "Late Delivery",
    "Courier Lost Parcel",
    "Damaged in Transit",
    "Wrong Item Shipped",
    "Short Shipment",
    "Faulty on Arrival",
    "Not As Described",
    "Sizing or Fit",
    "Billing Error",
    "Refund Delay",
]
 
# What the desk is allowed to offer. The policy advisor names one of these.
VALID_REMEDY_TYPES = [
    "Replacement",
    "Full Refund",
    "Partial Refund",
    "Reship Missing Item",
    "Exchange",
    "Goodwill Credit",
    "SLA Compensation",
    "Explain and Expedite",
]
 
# ---------------------------------------------------------------- number repair rules
# Numbers stored as text with a unit or a currency marker attached.
UNIT_SUFFIX_PATTERN = r"\s*(INR|Rs\.?|₹|hrs?|hours?|days?|kg|km|%)\s*"
THOUSANDS_AND_PERCENT_PATTERN = r"[,%]"
 
# Interquartile range multiplier used to identify an impossible spike.
OUTLIER_IQR_MULTIPLIER = 3.0
 
# Values outside these limits are impossible, including placeholder codes.
ORDER_RANGE_LIMITS = {
    "order_value_inr": (1.0, 500000.0),
    "item_count": (1.0, 40.0),
    "delivery_slack_days": (-60.0, 60.0),
    "transit_dwell_hours": (0.0, 1000.0),
    "route_failure_rate_30d": (0.0, 1.0),
    "category_return_rate": (0.0, 1.0),
    "prior_orders": (0.0, 500.0),
    "prior_tickets": (0.0, 200.0),
    "prior_returns": (0.0, 200.0),
    "days_since_order": (0.0, 120.0),
    "scan_count": (0.0, 60.0),
}
 
TICKET_RANGE_LIMITS = {
    "resolution_hours": (0.25, 720.0),
    "order_value_inr": (1.0, 500000.0),
    "item_count": (1.0, 40.0),
    "prior_tickets": (0.0, 200.0),
    "warehouse_lead_time_days": (0.0, 60.0),
    "approval_delay_hours": (0.0, 200.0),
    "replacement_unit_cost_inr": (0.0, 400000.0),
    "days_since_order": (0.0, 120.0),
}
 
MESSAGE_RANGE_LIMITS = {
    "message_length_chars": (8.0, 4000.0),
}
 
MINIMUM_MESSAGE_LENGTH = 8
MAXIMUM_MESSAGE_CHARACTERS = 4000
 
# Written into a text column when the original cell was empty.
UNKNOWN_CATEGORY_VALUE = "Unknown"
 
# Leftover markup and encoding damage in customer message text.
MARKUP_PATTERN = r"<[^>]+>"
ENCODING_ARTIFACT_LIST = ["Â", "&amp;", "&nbsp;", "â€™", "Ã©"]
 
# ---------------------------------------------------------------- escalation model columns
ESCALATION_TARGET_COLUMN = "escalation_within_7_days"
 
# Identifiers. Never model inputs.
ESCALATION_IDENTIFIER_COLUMNS = [
    "snapshot_id",
    "order_id",
    "customer_id",
    "shipment_id",
    "sku",
    "order_date",
    "promised_delivery_date",
    "dispatch_date",
    "last_scan_at",
    "observation_date",
]
 
ESCALATION_DATE_COLUMNS = [
    "order_date",
    "promised_delivery_date",
    "dispatch_date",
    "last_scan_at",
    "observation_date",
]
 
ESCALATION_CATEGORY_COLUMNS = [
    "channel",
    "payment_method",
    "product_category",
    "warehouse",
    "courier",
    "customer_tier",
    "city_tier",
]
 
ESCALATION_BOOLEAN_COLUMNS = [
    "is_split_shipment",
    "is_first_order",
    "first_attempt_failed",
    "is_fragile",
]
 
ESCALATION_NUMERIC_COLUMNS = [
    "order_value_inr",
    "item_count",
    "delivery_slack_days",
    "transit_dwell_hours",
    "route_failure_rate_30d",
    "category_return_rate",
    "prior_orders",
    "prior_tickets",
    "prior_returns",
    "days_since_order",
    "scan_count",
    "unit_price_max_inr",
    "warranty_months",
    "escalation_within_7_days",
]
 
# ---------------------------------------------------------------- resolution model columns
RESOLUTION_TARGET_COLUMN = "resolution_hours"
 
RESOLUTION_IDENTIFIER_COLUMNS = [
    "ticket_id",
    "order_id",
    "customer_id",
    "opened_at",
    "closed_at",
    "handled_by",
]
 
RESOLUTION_DATE_COLUMNS = ["opened_at", "closed_at"]
 
RESOLUTION_CATEGORY_COLUMNS = [
    "issue_mode",
    "remedy_type",
    "product_category",
    "customer_tier",
    "payment_method",
    "channel",
    "priority",
    "courier",
    "severity_code",
]
 
RESOLUTION_BOOLEAN_COLUMNS = [
    "requires_pickup",
    "requires_finance_reversal",
    "replacement_in_stock",
    "is_weekend_open",
]
 
RESOLUTION_NUMERIC_COLUMNS = [
    "order_value_inr",
    "item_count",
    "prior_tickets",
    "warehouse_lead_time_days",
    "approval_delay_hours",
    "replacement_unit_cost_inr",
    "days_since_order",
    "resolution_hours",
]
 
# ---------------------------------------------------------------- message model columns
MESSAGE_TEXT_COLUMN = "message_text"
MESSAGE_TARGET_COLUMN = "issue_mode"
MESSAGE_IDENTIFIER_COLUMN = "message_id"
MESSAGE_DATE_COLUMNS = ["sent_at"]
 
# ---------------------------------------------------------------- business rules
# NRC-POL-001. Escalation probability decides the band.
HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30
RISK_BAND_HIGH = "HIGH"
RISK_BAND_MEDIUM = "MEDIUM"
RISK_BAND_LOW = "LOW"
RISK_BAND_OPTIONS = [RISK_BAND_HIGH, RISK_BAND_MEDIUM, RISK_BAND_LOW]
 
# NRC-POL-001. Customer tier crossed with risk band decides the queue priority.
PRIORITY_MATRIX = {
    ("Platinum", "HIGH"): "P1", ("Platinum", "MEDIUM"): "P2", ("Platinum", "LOW"): "P3",
    ("Gold", "HIGH"): "P2", ("Gold", "MEDIUM"): "P3", ("Gold", "LOW"): "P3",
    ("Standard", "HIGH"): "P3", ("Standard", "MEDIUM"): "P4", ("Standard", "LOW"): "P4",
}
 
# Severity recorded against each risk band.
SEVERITY_BY_RISK_BAND = {"HIGH": "S1", "MEDIUM": "S2", "LOW": "S3"}
 
# NRC-CSC-014. The cash value each role may commit without escalating, in rupees.
ASSOCIATE_AUTHORITY_LIMIT_INR = 2000.0
MANAGER_AUTHORITY_LIMIT_INR = 25000.0
# NRC-CSC-014 sends anything above the manager limit to the head of customer care.
# The admin role holds that post, and there is nobody above it, so the buck stops
# there: a remedy no admin may approve would be a remedy nobody in the platform
# can ever action, which is a dead end rather than a control.
#
# A stated figure rather than an unlimited one, for two reasons. An auditor
# reading this file should see the ceiling the platform actually enforces, and
# "no ceiling at all" is not a control anybody signed off. And infinity is not
# JSON: float("inf") here made /auth/me return 500 for every admin, which looked
# exactly like a broken password.
#
# One crore is far above any single retail remedy in this business - the largest
# order in the book is a few tens of thousands of rupees - so it never blocks a
# legitimate decision, while still being a real, auditable number.
HEAD_OF_CARE_AUTHORITY_LIMIT_INR = 10000000.0
 
# NRC-CSC-014. These always need a manager whatever the amount.
ALWAYS_NEEDS_APPROVAL_REMEDIES = ["Goodwill Credit"]
ALWAYS_NEEDS_APPROVAL_PAYMENT_METHODS = ["EMI"]
 
# NRC-SHIP-002. The promise made at checkout, in business days, by channel.
STANDARD_DELIVERY_DAYS = 5
 
# NRC-SHIP-002. Late shipment compensation, issued without the customer asking.
# Each entry is (days late from, days late to, store credit in rupees).
LATE_DELIVERY_COMPENSATION_TIERS = [
    (1, 3, 100.0),
    (4, 7, 250.0),
    (8, 100000, 500.0),
]
LATE_DELIVERY_COMPENSATION_CAP_INR = 500.0
 
# NRC-RET-004. A partial refund is capped at this share of the item value.
PARTIAL_REFUND_MAXIMUM_SHARE = 0.30
 
# NRC-CSC-014. The standing goodwill credit the desk offers when it offers one.
GOODWILL_CREDIT_DEFAULT_INR = 500.0
 
# The remedy the desk usually offers for each issue mode. A starting point, not a
# decision: the policy advisor may move away from it after reading the retrieved
# passages. It lives here rather than in code so that the offline backend and the
# tools layer cannot drift apart on what "usual" means.
DEFAULT_REMEDY_BY_ISSUE = {
    "Late Delivery": "SLA Compensation",
    "Courier Lost Parcel": "Full Refund",
    "Damaged in Transit": "Replacement",
    "Wrong Item Shipped": "Replacement",
    "Short Shipment": "Reship Missing Item",
    "Faulty on Arrival": "Replacement",
    "Not As Described": "Full Refund",
    "Sizing or Fit": "Exchange",
    "Billing Error": "Partial Refund",
    "Refund Delay": "Explain and Expedite",
}
 
# What each remedy actually costs Northlane in cash to the customer. A remedy
# that costs nothing in cash still costs the business money in handling, but the
# authority limits in NRC-CSC-014 are written about cash, so that is what is
# measured here.
REMEDY_COSTS_CASH = {
    "Full Refund": "order_value",
    "Partial Refund": "partial_share",
    "SLA Compensation": "lateness_tier",
    "Goodwill Credit": "goodwill_default",
    "Replacement": "none",
    "Reship Missing Item": "none",
    "Exchange": "none",
    "Explain and Expedite": "none",
}
 
# Stand-in values for a resolution that has not been carried out yet. The manager
# confirms them at review. Every one of these is an assumption, not a fact, and the
# finalizer replaces the ones that have become facts.
RESOLUTION_PLANNING_ASSUMPTIONS = {
    "approval_delay_hours": 4.0,
    "warehouse_lead_time_days": 2.0,
    "replacement_in_stock": 1,
    "requires_pickup": 1,
    "requires_finance_reversal": 0,
    "is_weekend_open": 0,
    "replacement_unit_cost_inr": 1500.0,
}
 
# The finalizer recomputes the resolution estimate only when one of these changed.
RECOMPUTE_ON_REMEDY_CHANGE = True
RECOMPUTE_ON_APPROVAL_DELAY = True
 
# ---------------------------------------------------------------- review workflow
REVIEW_DECISION_APPROVE = "APPROVE"
REVIEW_DECISION_MODIFY = "MODIFY"
REVIEW_DECISION_REJECT = "REJECT"
REVIEW_DECISION_OPTIONS = [
    REVIEW_DECISION_APPROVE,
    REVIEW_DECISION_MODIFY,
    REVIEW_DECISION_REJECT,
]
 
RESOLUTION_STATUS_PENDING = "PENDING_REVIEW"
RESOLUTION_STATUS_APPROVED = "APPROVED"
RESOLUTION_STATUS_REJECTED = "REJECTED"
RESOLUTION_STATUS_OPTIONS = [
    RESOLUTION_STATUS_PENDING,
    RESOLUTION_STATUS_APPROVED,
    RESOLUTION_STATUS_REJECTED,
]
 
AGENT_RUN_STATUS_RUNNING = "RUNNING"
AGENT_RUN_STATUS_AWAITING_REVIEW = "AWAITING_REVIEW"
AGENT_RUN_STATUS_COMPLETE = "COMPLETE"
 
# ---------------------------------------------------------------- access control
ROLE_CUSTOMER = "customer"
ROLE_ASSOCIATE = "associate"
ROLE_MANAGER = "manager"
ROLE_ADMIN = "admin"
ROLE_OPTIONS = [ROLE_CUSTOMER, ROLE_ASSOCIATE, ROLE_MANAGER, ROLE_ADMIN]
 
# Which roles may reach each area of the service. Enforced server side.
#
# The customer holds "chat" and nothing else, and that single line is the whole
# difference between a care desk tool and a self service one. A customer may ask
# about the policy book and report that something has gone wrong; they may not
# read the order book, see anyone's queue, start an advisory run, or approve a
# remedy for themselves. Committing money to a customer stays a decision made by
# somebody who does not receive it.
ROLE_PERMISSIONS = {
    ROLE_CUSTOMER: [
        "chat",
    ],
    ROLE_ASSOCIATE: [
        "chat", "own_queue", "order_read", "advisory_start",
    ],
    ROLE_MANAGER: [
        "chat", "own_queue", "order_read", "advisory_start",
        "risk_dashboard", "review_decide", "all_queues",
    ],
    ROLE_ADMIN: [
        "chat", "own_queue", "order_read", "advisory_start",
        "risk_dashboard", "review_decide", "all_queues",
        "audit_read", "evaluation_read", "pipeline_read", "user_manage",
    ],
}
 
JWT_SECRET_KEY = os.environ.get(
    "CAREIQ_JWT_SECRET", "development-only-secret-replace-before-any-deployment"
)
JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME_MINUTES = 60
 
# Seeded on first database setup so the demo has someone to sign in as.
DEMO_USERS = [
    {"user_name": "rohan", "password": "customer123", "role": ROLE_CUSTOMER,
     "display_name": "Rohan Iyer"},
    {"user_name": "asha", "password": "associate123", "role": ROLE_ASSOCIATE,
     "display_name": "Asha Menon"},
    {"user_name": "vikram", "password": "manager123", "role": ROLE_MANAGER,
     "display_name": "Vikram Rao"},
    {"user_name": "divya", "password": "admin123", "role": ROLE_ADMIN,
     "display_name": "Divya Krishnan"},
]
 
# ---------------------------------------------------------------- document retrieval
CHUNK_SIZE_CHARACTERS = 700
CHUNK_OVERLAP_CHARACTERS = 120
RETRIEVAL_RESULT_COUNT = 4
 
# Downloaded and cached on first use. When it cannot be reached, the deterministic
# offline embedder is used instead so the index still builds and the tests still run.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
OFFLINE_EMBEDDING_DIMENSIONS = 384
EMBEDDING_BACKEND = os.environ.get("CAREIQ_EMBEDDING_BACKEND", "auto")
 
# The six documents the chatbot and the policy advisor are restricted to.
POLICY_DOCUMENTS = [
    {"document_id": "NRC-RET-004", "title": "Returns and Refunds Policy",
     "topic": "returns"},
    {"document_id": "NRC-SHIP-002", "title": "Delivery SLA and Late Shipment Compensation",
     "topic": "delivery"},
    {"document_id": "NRC-WAR-011", "title": "Warranty and Repair Procedures",
     "topic": "warranty"},
    {"document_id": "NRC-FRD-006", "title": "Chargeback and Fraud Escalation Guidelines",
     "topic": "fraud"},
    {"document_id": "NRC-CSC-014", "title": "Customer Service Tone and Goodwill Authority",
     "topic": "service"},
    {"document_id": "NRC-PRD-021", "title": "Product Care and Handling Guide",
     "topic": "product"},
]
 
# ---------------------------------------------------------------- language model
LLM_BACKEND = os.environ.get("CAREIQ_LLM_BACKEND", "auto")
GEMINI_API_KEY_VARIABLE = "GEMINI_API_KEY"
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_MAX_OUTPUT_TOKENS = 900
LLM_TEMPERATURE = 0.2
FREE_TIER_WAIT_SECONDS = 60
LLM_CACHE_FILE = "output/llm_cache.json"
 
AGENT_NAME_TRIAGE = "triage_planner"
AGENT_NAME_RISK = "risk_analyst"
AGENT_NAME_ISSUE = "issue_analyst"
AGENT_NAME_POLICY = "policy_advisor"
AGENT_NAME_COMPOSER = "resolution_composer"
AGENT_NAME_FINALIZER = "resolution_finalizer"
AGENT_NAME_ROUTER = "intent_router"
AGENT_NAME_CHAT = "chat_responder"
 
# Language model calls in one full advisory run. A rejection costs one fewer,
# because the finalizer never runs.
EXPECTED_LLM_CALLS_PER_RUN = 6
 
# ---------------------------------------------------------------- chatbot
INTENT_POLICY_QUESTION = "policy_question"
INTENT_ORDER_STATUS = "order_status"
INTENT_COMPLAINT = "complaint"
INTENT_SMALL_TALK = "small_talk"
INTENT_OUT_OF_SCOPE = "out_of_scope"
INTENT_OPTIONS = [
    INTENT_POLICY_QUESTION,
    INTENT_ORDER_STATUS,
    INTENT_COMPLAINT,
    INTENT_SMALL_TALK,
    INTENT_OUT_OF_SCOPE,
]
 
# Short term memory. Session scoped, never a permanent history.
MEMORY_MAX_TURNS = 6
# "ticket_id" is here so a conversation knows what it has already filed, and adds
# to that ticket rather than opening a second one for the same problem.
MEMORY_ENTITY_KEYS = ["order_id", "issue_mode", "customer_tier", "remedy_type",
                      "ticket_id"]
CHAT_SESSION_IDLE_MINUTES = 60
 
# ---------------------------------------------------------------- ticket routing
# A ticket nobody is named on. A customer reporting their own problem cannot be
# the person who works it, so their ticket is opened unassigned and every
# associate sees it alongside their own queue until somebody picks it up.
UNASSIGNED_QUEUE = ""
 
# Below this, the issue mode model is guessing rather than reading. A guess must
# not become a classification on the record, so the desk asks instead of filing.
MINIMUM_ISSUE_CONFIDENCE = 0.60
 
# What an order reference looks like, and what merely looks like one. The second
# pattern exists so that "TTT-009983" is answered with "that is not a reference we
# recognise" rather than silently treated as no reference at all.
ORDER_REFERENCE_FORMAT = "NRC-" + "0" * 6
LOOKS_LIKE_A_REFERENCE_PATTERN = r"\b[A-Za-z]{2,5}[-\s]?\d{4,8}\b"
 
UNRECOGNISED_REFERENCE_REPLY = (
    "I could not find an order with the reference {0}. A Northlane order number "
    "looks like {1}. Could you check it and send it again?")
 
ISSUE_UNCLEAR_REPLY = (
    "I want to get this to the right place. Could you tell me a little more about "
    "what went wrong? For example whether the parcel arrived damaged, arrived "
    "late, was the wrong item, or has not arrived at all.")
 
# What a customer is shown when nothing can be stood behind. The internal reason
# names what tripped and is never sent to the person who asked.
# Nothing is forwarded when this fires: no ticket is opened and nobody is
# notified. Saying "I have passed this to a colleague" was untrue, and an untrue
# reassurance is worse than an honest dead end, because the customer stops asking.
LOW_CONFIDENCE_REFUSAL = (
    "I could not answer that from Northlane's published policies, so I would rather "
    "not guess. If you can tell me what went wrong with your order, I can log it for "
    "the care desk."
)
OUT_OF_SCOPE_REFUSAL = (
    "That is outside what I can help with. I can answer questions about Northlane "
    "orders, deliveries, returns, refunds and warranties."
)
MINIMUM_ANSWER_CONFIDENCE = 0.6
 
# ---------------------------------------------------------------- ingestion
INGESTION_SOURCE = os.environ.get("CAREIQ_INGESTION_SOURCE", "local_csv")
INGESTION_INTERVAL_MINUTES = 10
# Deliberately smaller than the bundled feed. A batch limit larger than the whole
# source means the first run swallows everything and the pipeline never
# demonstrates the thing it exists to demonstrate: arriving a piece at a time.
INGESTION_BATCH_LIMIT = 150
INGESTION_RUN_HISTORY_LIMIT = 50
 
# ---------------------------------------------------------------- services
API_HOST = "0.0.0.0"
API_PORT = 8080
API_TITLE = "CareIQ API"
API_BASE_URL = os.environ.get("CAREIQ_API_BASE_URL", "http://localhost:8080")
API_REQUEST_TIMEOUT_SECONDS = 180
MAXIMUM_REQUESTS_PER_MINUTE = 120
ALLOWED_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]
REQUEST_LOG_FILE = "output/request.log"
 
DASHBOARD_TITLE = "CareIQ"
DASHBOARD_SUBTITLE = "Northlane Retail Co. - Customer Care Intelligence"
 
RISK_BAND_BADGE_COLOURS = {
    RISK_BAND_HIGH: "red",
    RISK_BAND_MEDIUM: "orange",
    RISK_BAND_LOW: "green",
}
REVIEW_DECISION_BADGE_COLOURS = {
    REVIEW_DECISION_APPROVE: "green",
    REVIEW_DECISION_MODIFY: "blue",
    REVIEW_DECISION_REJECT: "red",
}
 
# ---------------------------------------------------------------- evaluation bars
# Every one of these is asserted by the evaluation harness.
EVALUATION_THRESHOLDS = {
    "retrieval_hit_at_k": 0.85,
    "citation_validity": 1.00,
    "answer_relevance_mean": 4.0,
    "refusal_accuracy": 0.90,
    "intent_macro_f1": 0.85,
    # Generalisation is reported but not gated, and the reason is written down in
    # run_evaluation.py: a keyword router scores ~0.99 on the wording it was built
    # against and ~0.32 on wording it has never seen. Closing that gap is the job
    # of the language model routing layer, which the offline backend cannot stand
    # in for. Gating on it would only ever mean "no API key", which is not a
    # quality signal.
    "intent_macro_f1_heldout": 0.25,
    "memory_carry_over": 0.90,
    "policy_violation_rate": 0.00,
    "draft_acceptance": 0.60,
}
 
# ---------------------------------------------------------------- reproducibility
RANDOM_SEED = 42
VALIDATION_SPLIT_RATIO = 0.2
CROSS_VALIDATION_FOLDS = 3
 
 
# ==================================================================================
# EXTENSIONS
# Constants the specification refers to by name that the supplied file did not
# carry yet. They live here, with the rest, so that no module types a figure or a
# word list of its own. Where a value can be derived from a constant above it is
# derived rather than retyped, so the two can never disagree.
# ==================================================================================

# ---------------------------------------------------------------- cleaning aliases
# The specification calls these by these names.
MINIMUM_MESSAGE_CHARACTERS = MINIMUM_MESSAGE_LENGTH
# A cleaned message needs at least this many words to be classifiable.
MINIMUM_MESSAGE_WORDS = 2
VALIDATION_RANGES = dict(ORDER_RANGE_LIMITS)
VALIDATION_RANGES.update(TICKET_RANGE_LIMITS)

# The date shapes the source systems write. Tried in order; the first that reads a
# value wins, and a value no format reads stays missing rather than guessed.
DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d",
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y",
    "%d-%B-%Y", "%d %B %Y", "%B %d, %Y",
]

# City tier as the raw extracts spell it. "Metro" is kept as its own geography.
CITY_TIER_MAPPING = {
    "metro": "Metro",
    "tier1": "Tier-1", "tier-1": "Tier-1", "tier 1": "Tier-1",
    "tier2": "Tier-2", "tier-2": "Tier-2", "tier 2": "Tier-2",
    "tier3": "Tier-3", "tier-3": "Tier-3", "tier 3": "Tier-3",
}

# Catalogue spellings derived from the catalogues themselves.
ISSUE_MODE_MAPPING = {mode.lower(): mode for mode in VALID_ISSUE_MODES}
REMEDY_TYPE_MAPPING = {remedy.lower(): remedy for remedy in VALID_REMEDY_TYPES}
PRIORITY_MAPPING = {value.lower(): value for value in sorted(set(PRIORITY_MATRIX.values()))}
SEVERITY_MAPPING = {value.lower(): value for value in SEVERITY_BY_RISK_BAND.values()}

ORDER_CATEGORY_MAPPINGS = {
    "channel": CHANNEL_MAPPING,
    "payment_method": PAYMENT_METHOD_MAPPING,
    "product_category": PRODUCT_CATEGORY_MAPPING,
    "warehouse": WAREHOUSE_MAPPING,
    "courier": COURIER_MAPPING,
    "customer_tier": CUSTOMER_TIER_MAPPING,
    "city_tier": CITY_TIER_MAPPING,
}
TICKET_CATEGORY_MAPPINGS = {
    "issue_mode": ISSUE_MODE_MAPPING,
    "remedy_type": REMEDY_TYPE_MAPPING,
    "product_category": PRODUCT_CATEGORY_MAPPING,
    "customer_tier": CUSTOMER_TIER_MAPPING,
    "payment_method": PAYMENT_METHOD_MAPPING,
    "channel": CHANNEL_MAPPING,
    "priority": PRIORITY_MAPPING,
    "courier": COURIER_MAPPING,
    "severity_code": SEVERITY_MAPPING,
}

# Outlier blanking is applied to these columns only. transit_dwell_hours is never
# here: the orders that have not moved for days are the escalation signal.
ESCALATION_OUTLIER_COLUMNS = ["scan_count", "item_count"]
RESOLUTION_OUTLIER_COLUMNS = ["warehouse_lead_time_days"]

# ---------------------------------------------------------------- message cleaning
MAIL_FOOTER_PATTERNS = [
    r"sent from my (iphone|ipad|android|samsung[\w ]*|galaxy[\w ]*|phone|mobile)",
    r"get outlook for (ios|android)",
    r"sent via [\w ]+ app",
]
GREETING_PATTERNS = [
    r"^\s*(dear|respected|hello|hi|hey|good (morning|afternoon|evening))"
    r"(\s+(sir|madam|team|support|northlane|all|there))*\s*[,.!:-]?",
]
SIGN_OFF_PATTERNS = [
    r"(thanks and regards|thanks in advance|thank you in advance|awaiting your response"
    r"|kind regards|best regards|warm regards|regards|thanks|thank you|cheers)\s*[.,!]*\s*$",
]
# A line matching one of these starts the signature block; it and everything
# after it describe the sender, not the problem.
SIGNATURE_START_PATTERNS = [
    r"^\s*--+\s*$", r"^\s*(regards|kind regards|best regards|thanks|thank you|cheers|best),?\s*$",
]
# Lines that are a signature on their own, wherever they sit.
SIGNATURE_LINE_PATTERNS = [
    r"^\s*(customer|northlane shopper|northlane customer)\s*$",
]
QUOTED_REPLY_PATTERNS = [r"^\s*>", r"^\s*on .{4,80} wrote:\s*$", r"^-+\s*original message\s*-+$"]
ORDER_REFERENCE_PATTERN = r"\bNRC-\d{6}\b"
ORDER_REFERENCE_MASK = "orderref"
ENCODING_REPAIR_MAP = {"â€™": "'", "Ã©": "e", "Â": "", "&amp;": "&", "&nbsp;": " "}

# ---------------------------------------------------------------- ticket statuses
TICKET_STATUS_OPEN = "OPEN"
TICKET_STATUS_CLOSED = "CLOSED"
TICKET_STATUS_OPTIONS = [TICKET_STATUS_OPEN, TICKET_STATUS_CLOSED]
AGENT_RUN_STATUS_OPTIONS = [AGENT_RUN_STATUS_RUNNING, AGENT_RUN_STATUS_AWAITING_REVIEW,
                            AGENT_RUN_STATUS_COMPLETE]
INGESTION_STATUS_SUCCESS = "SUCCESS"
INGESTION_STATUS_FAILED = "FAILED"
INGESTION_STATUS_NO_NEW_ROWS = "NO_NEW_ROWS"
INGESTION_STATUS_OPTIONS = [INGESTION_STATUS_SUCCESS, INGESTION_STATUS_FAILED,
                            INGESTION_STATUS_NO_NEW_ROWS]

# Ingestion ages rows it has already sent: each pass through the feed is a day.
INGESTION_AGEING_PER_GENERATION = {
    "transit_dwell_hours": 24.0,
    "days_since_order": 1.0,
    "delivery_slack_days": -1.0,
}

# ---------------------------------------------------------------- modelling
TEXT_VECTORIZER_SETTINGS = {"ngram_range": (1, 2), "min_df": 3, "max_df": 0.9,
                            "sublinear_tf": True}
ESCALATION_PRIMARY_METRIC = "cv_roc_auc"
RESOLUTION_PRIMARY_METRIC = "cv_r2_score"
ISSUE_MODE_PRIMARY_METRIC = "cv_macro_f1_score"

# ---------------------------------------------------------------- retrieval vocabulary
# Words that place a question in one document's topic. Search hints only.
TOPIC_VOCABULARY = {
    "returns": ["return", "returns", "refund", "refunds", "exchange", "rma",
                "return window", "money back", "send back", "send it back"],
    "delivery": ["delivery", "late", "courier", "shipping", "shipment", "parcel",
                 "tracking", "compensation", "in transit", "not arrived", "dispatch"],
    "warranty": ["warranty", "repair", "faulty", "defective", "service centre",
                 "manufacturer", "turnaround"],
    "fraud": ["chargeback", "fraud", "dispute", "charged twice", "duplicate charge",
              "card issuer", "investigation"],
    "service": ["goodwill", "authority", "approve", "approval", "manager", "escalate",
                "first contact", "complain"],
    "product": ["fragile", "photograph", "photo", "handling", "care", "charger",
                "sizing", "size chart", "wash", "appliance"],
}

# Search vocabulary per issue mode. These only enrich the query; nothing here
# reaches a customer or decides a remedy.
ISSUE_QUERY_HINTS = {
    "Late Delivery": "late delivery promised date compensation store credit tiers cap",
    "Courier Lost Parcel": "parcel no courier scan trace declared lost full refund reshipment",
    "Damaged in Transit": "damage reported within 48 hours photographs outer packaging fragile claim courier",
    "Wrong Item Shipped": "wrong item photographs label return pickup replacement",
    "Short Shipment": "missing item order line reship replacement",
    "Faulty on Arrival": "faulty on arrival does not function first use 7 calendar days replacement full refund",
    "Not As Described": "return window condition of returned goods refund original payment method",
    "Sizing or Fit": "size exchange apparel footwear size chart return window",
    "Billing Error": "duplicate charge reversed 3 business days charge higher than checkout total",
    "Refund Delay": "refund timing business days quality check finance trace",
}
AUTHORITY_QUERY_TEMPLATE = "authority limits who may approve {0} associate manager approval"

# ---------------------------------------------------------------- routing
MODEL_ROUTING_THRESHOLD = 0.5
CLASSIFIER_ROUTING_THRESHOLD = 0.75
INTENT_VOCABULARY = {
    INTENT_POLICY_QUESTION: [
        "policy", "how many days", "how long", "return window", "can i return",
        "how do i return", "refund policy", "warranty", "am i covered", "am i eligible",
        "allowed", "exchange", "how does", "do you offer", "rules",
        "how much compensation", "compensation", "goodwill", "return", "refund",
        "photograph", "chargeback", "fragile", "shipping charge", "cancel",
    ],
    INTENT_ORDER_STATUS: [
        "where is my order", "where is my parcel", "where is my package", "track my order",
        "tracking", "order status", "status of my order", "when will my order arrive",
        "when will it arrive", "when will my parcel arrive", "has it shipped",
        "has my order shipped", "delivery date", "expected delivery", "where is it",
        "is my order on the way", "out for delivery",
    ],
    INTENT_COMPLAINT: [
        "broken", "cracked", "crushed", "damaged", "damage", "smashed", "shattered",
        "dented", "defective", "faulty", "not working", "stopped working", "dead",
        "will not power", "wont turn on", "wrong item", "wrong colour", "wrong size",
        "wrong model", "missing", "not received", "never arrived", "never received",
        "lost", "charged twice", "double charged", "overcharged", "refund not received",
        "still not credited", "not credited", "not been credited", "not been refunded",
        "still waiting", "has not arrived", "not arrived", "too small", "too big", "does not fit",
        "doesn't fit", "too tight", "too loose", "late", "delayed", "not as described",
        "different from", "poor service", "complaint", "leaking", "torn", "scratched",
    ],
    INTENT_SMALL_TALK: [
        "hi", "hello", "hey", "thanks", "thank you", "thanks very much", "good morning",
        "good evening", "good afternoon", "bye", "goodbye", "ok", "okay", "great",
        "cheers", "how are you", "much appreciated", "very much", "you too",
    ],
    INTENT_OUT_OF_SCOPE: [
        "price of gold", "weather", "cricket", "football", "stock market", "share price",
        "recipe", "movie", "song", "joke", "news", "election", "bitcoin", "horoscope",
        "capital of", "translate", "homework", "poem", "today's date",
    ],
}
# Phrases that make a message a request to do something rather than a fault report.
REQUEST_OPENERS = ["i want to", "i would like to", "i'd like to", "how do i", "how can i",
                   "can i", "could i", "may i", "is it possible to", "i need to", "please help me"]
REQUEST_ACTIONS = ["return", "exchange", "cancel", "refund", "change", "swap"]
# Words that always mean something is wrong, whatever else the message says.
STRONG_FAULT_WORDS = ["broken", "cracked", "crushed", "damaged", "smashed", "shattered",
                      "defective", "faulty", "not working", "wrong item", "missing",
                      "never arrived", "charged twice", "dead", "torn", "leaking"]
QUESTION_OPENERS = ["what", "how", "when", "why", "which", "who", "where", "can", "could",
                    "do", "does", "is", "are", "will", "would", "should", "may", "am"]
# Words that make a message about Northlane at all. A question with none of them
# is outside what the policy book covers.
DOMAIN_VOCABULARY = sorted(set(
    [word for words in TOPIC_VOCABULARY.values() for word in words]
    + INTENT_VOCABULARY[INTENT_COMPLAINT] + INTENT_VOCABULARY[INTENT_ORDER_STATUS]
    + ["order", "item", "product", "northlane", "package", "pickup", "store credit",
       "emi", "cod", "prepaid", "payment", "shoes", "tv", "phone", "delivered"]))
BARE_REFERENCE_FILLER_WORDS = ["order", "no", "number", "ref", "reference", "my", "is",
                               "it", "its", "the", "here", "id", "please", "sure",
                               "this", "yes", "num"]

SMALL_TALK_REPLY = ("Happy to help. You can ask me about returns, refunds, deliveries "
                    "and warranties, check where an order is, or tell me what went wrong "
                    "with an order.")
ASK_FOR_ORDER_REPLY = ("I can check that for you. Please send your order number; a "
                       "Northlane order number looks like {0}.")
ORDER_NOT_ON_FILE_REPLY = ("I could not find order {0} on file. Please check the number "
                           "and send it again; it looks like {1}.")
TICKET_OPENED_REPLY = ("I have logged this as case {0} ({1}) and it is open on the care "
                       "desk queue.{2}")
TICKET_AMENDED_REPLY = ("I have added this to your open case {0} ({1}), which is on the care "
                        "desk queue.{2}")
TICKET_NEEDS_ORDER_SUFFIX = " To go further I will need your order number, which looks like {0}."
TICKET_NOT_FILED_REPLY = ("This reads as a {0} complaint. Ticket filing is switched off in "
                          "this session, so nothing has been logged.")

# ---------------------------------------------------------------- language model prompts
# The offline stand-in's wording. Every figure it writes comes from the prompt's facts.
OFFLINE_MODEL_NAME = "offline-deterministic-v1"
GEMINI_RETRY_ATTEMPTS = 3

# ---------------------------------------------------------------- service
DEFAULT_ADVISORY_REQUEST_TEXT = "Assess this order and draft a resolution."
CHAT_SOURCE_CHANNEL = "Chat"
STAFF_SOURCE_CHANNEL = "Desk"
API_DEFAULT_LIST_LIMIT = 100
SCRIPTS_SKIPPED_DIRECTORIES = ["venv", ".venv", "env", "__pycache__", ".pytest_cache",
                               ".git", "data", "output", "processed", "database",
                               "node_modules"]

# ---------------------------------------------------------------- chunking and embedding
# A heading is a short line with no terminal punctuation.
HEADING_MAXIMUM_CHARACTERS = 60
HEADING_MAXIMUM_WORDS = 8
HEADING_TERMINAL_PUNCTUATION = ".,;:!?"
# The offline embedder ignores words that carry no topic. Kept short on purpose.
OFFLINE_EMBEDDING_STOPWORDS = [
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "be", "by", "with", "it", "this", "that", "as", "at", "from", "i", "my", "me", "do",
    "does", "have", "has", "can", "what", "who", "how", "when", "which", "if", "there",
    "their", "they", "we", "you", "your", "our", "will", "would", "should", "may", "am",
    "any", "not", "no", "so", "than", "then", "its", "into", "up", "out", "about",
]
OFFLINE_BIGRAM_WEIGHT = 0.6
EMBEDDING_DISTANCE_DECIMALS = 4

# The remedies that refund the whole order value, derived from the cost rules. The
# EMI rule in NRC-CSC-014 is written about these.
FULL_REFUND_REMEDIES = [remedy for remedy, rule in REMEDY_COSTS_CASH.items()
                        if rule == "order_value"]

# ---------------------------------------------------------------- answer confidence (offline)
# Words that carry no subject in a question. The offline stand-in ignores them when
# it measures how much of a question a passage actually covers.
OFFLINE_CONFIDENCE_IGNORED_WORDS = [
    "many", "much", "long", "pair", "get", "got", "need", "want", "please", "tell",
    "know", "happen", "happens", "days", "day", "time", "ok", "okay", "also", "still",
    "just", "like", "something", "thing", "things", "anything", "help", "order", "orders",
]
# Everyday words mapped onto the vocabulary the policy book uses.
QUESTION_SYNONYMS = {
    "shoes": "footwear", "shoe": "footwear", "sneakers": "footwear", "trainers": "footwear",
    "clothes": "apparel", "shirt": "apparel", "dress": "apparel", "jeans": "apparel",
    "tv": "electronics", "television": "electronics", "phone": "electronics",
    "laptop": "electronics", "money": "refund", "broken": "damage", "cracked": "damage",
    "smashed": "damage", "parcel": "parcel", "package": "parcel", "lost": "lost",
    "late": "late", "delayed": "late", "fix": "repair", "send": "return",
}
