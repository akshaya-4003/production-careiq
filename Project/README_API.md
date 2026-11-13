# CareIQ REST API

Base URL: `http://localhost:8080`. Interactive docs, where you can try every call:
<http://localhost:8080/docs>

Start it with `python3 run_api_service.py` after entry points 2 to 5 have run
(see [README.md](README.md)).

## Signing in

Every endpoint except `/health` and `/auth/login` needs a token.

```bash
TOKEN=$(curl -s -X POST localhost:8080/auth/login \
  -H 'content-type: application/json' \
  -d '{"user_name":"asha","password":"associate123"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -s localhost:8080/auth/me -H "Authorization: Bearer $TOKEN"
```

```json
{"user_name":"asha","display_name":"Asha Menon","role":"associate",
 "permissions":["chat","own_queue","order_read","advisory_start"],
 "authority_limit_inr":2000.0}
```

Tokens last `TOKEN_LIFETIME_MINUTES` (60). A wrong password, or a user name that
doesn't exist, both return `401 Unrecognised user name or password.`, without
saying which half was wrong.

## Who may call what

Access is checked **on the server**, by permission. The permission lists live in
`config.ROLE_PERMISSIONS`.

| Permission | customer | associate | manager | admin |
|---|:-:|:-:|:-:|:-:|
| `chat` | ✅ | ✅ | ✅ | ✅ |
| `own_queue`, `order_read`, `advisory_start` | | ✅ | ✅ | ✅ |
| `risk_dashboard`, `review_decide`, `all_queues` | | | ✅ | ✅ |
| `audit_read`, `evaluation_read`, `pipeline_read` | | | | ✅ |

A missing or invalid token returns **401**. A valid token without the permission
returns **403**, naming the role and the permission.

## Endpoints

| Method & path | Permission | LLM calls | What it does |
|---|---|:-:|---|
| `GET /health` | none | 0 | Status, orders monitored, policy chunks, which LLM and embedding backends are running |
| `POST /auth/login` | none | 0 | `{user_name, password}` → `{access_token, role, ...}` |
| `GET /auth/me` | token | 0 | Role, permissions, authority limit (always a finite number) |
| `GET /catalogue` | token | 0 | Remedy types, issue modes, decisions, bands, colours, limits (so the dashboard holds no lists) |
| `GET /orders/{order_id}` | `order_read` | 0 | Order record plus a readable summary. **404** if not on file |
| `GET /orders/{order_id}/risk` | `order_read` | 0 | Probability, band, priority, severity |
| `GET /risk/summary` | `risk_dashboard` | 0 | Counts by band and priority, HIGH-risk exposure, when it was scored |
| `GET /risk/orders?risk_band=HIGH&limit=50` | `risk_dashboard` | 0 | Scored book, **worst first** |
| `POST /tickets` | `own_queue` | 0 | `{order_id, message_text}` → classify and open a ticket (unknown order → 404) |
| `GET /tickets?status=OPEN&all_queues=false` | `own_queue` | 0 | Your tickets plus unassigned ones; `all_queues=true` needs `all_queues` |
| `POST /advisories` | `advisory_start` | **5 or 6** | Run the workflow. Inside authority it is approved on the spot (6); otherwise it pauses (5) |
| `GET /resolutions?status=PENDING_REVIEW` | `own_queue` | 0 | The review queue |
| `GET /resolutions/{id}` | `own_queue` | 0 | One resolution, its run status, call count and reviews |
| `POST /resolutions/{id}/review` | `review_decide` | **1** (0 on REJECT) | `{decision, feedback_text, remedy_type?, approval_delay_hours?}` |
| `GET /resolutions/{id}/reviews` | `own_queue` | 0 | Every decision on a resolution, with the draft as it stood before |
| `POST /chat` | `chat` | **0 or 1** | `{message, session_id?}` → answer, intent, citations, ticket |
| `GET /chat/{session_id}` | `chat` | 0 | The transcript, oldest first (your own conversations only) |
| `GET /runs` | `audit_read` | 0 | Advisory runs with calls made against calls expected |
| `GET /runs/{run_id}/calls` | `audit_read` | 0 | The per-call budget for one run |
| `GET /pipeline/runs` | `pipeline_read` | 0 | Ingestion log |
| `POST /pipeline/rescore` | `pipeline_read` | 0 | Re-read the serving store and re-score the whole book |
| `GET /evaluation` | `evaluation_read` | 0 | `output/evaluation_report.json`, or a message naming the command that produces it |
| `GET /metrics/models` | `evaluation_read` | 0 | `output/model_metrics.json` (every bake-off candidate) |
| `GET /metrics/data-quality` | `evaluation_read` | 0 | `output/data_quality_report.json` |

## Worked examples

### Look up an order and its risk

```bash
curl -s localhost:8080/orders/NRC-900358/risk -H "Authorization: Bearer $TOKEN"
```
```json
{"order_id":"NRC-900358","escalation_probability":0.9723,"risk_band":"HIGH",
 "priority":"P3","severity_code":"S1","customer_tier":"Standard",
 "order_value_inr":3028.6,"transit_dwell_hours":139.43,"delivery_slack_days":-14.0,...}
```

### Start an advisory run

```bash
curl -s -X POST localhost:8080/advisories -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"order_id":"NRC-900001","message_text":"the courier has lost my parcel, no scan for days"}'
```

The response, trimmed. A ₹2,215 full refund is above the ₹2,000 associate limit,
so the run pauses:

```json
{"run_id":"RUN-…","status":"AWAITING_REVIEW","waiting_for_review":true,"calls_made":5,
 "resolution":{"resolution_id":"RES-…","remedy_type":"Full Refund","remedy_amount_inr":2215.35,
   "needs_approval":true,
   "approval_reason":"INR 2,215.35 is above the associate authority limit of INR 2,000.00 under NRC-CSC-014.",
   "escalation_probability":0.0725,"risk_band":"LOW","priority":"P4",
   "citations":"CH0014,CH0011","source_documents":"NRC-SHIP-002",
   "customer_reply":"We are sorry about the problem with order NRC-900001. …",
   "status":"PENDING_REVIEW", …},
 "steps":[{"node":"load_context","status":"ok",…}, …]}
```

Refusals come back as **422** and name the node:

```json
{"detail":{"refused_at":"load_context","reason":"order NRC-000000 is not on file"}}
```

### Review it as a manager

```bash
MTOKEN=…   # sign in as vikram / manager123
curl -s -X POST localhost:8080/resolutions/RES-…/review -H "Authorization: Bearer $MTOKEN" \
  -H 'content-type: application/json' \
  -d '{"decision":"MODIFY","remedy_type":"Replacement","feedback_text":"Offer a replacement first","approval_delay_hours":10}'
```

| Situation | Status |
|---|---|
| `decision` not one of APPROVE, MODIFY, REJECT | **422**, and the graph is never touched |
| Resolution does not exist | **404** |
| Already reviewed | **409** |
| Amount above *your own* limit (for example ₹35,662 for a manager) | **403** `… it needs the admin role.` |
| Caller is an associate | **403** |

### Chat

```bash
curl -s -X POST localhost:8080/chat -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"message":"how many days do I have to return a pair of shoes"}'
```
```json
{"session_id":"CHAT-…","intent":"policy_question",
 "answer":"Apparel and Footwear carry an extended return window of 30 calendar days from the date of delivery. [CH0003]",
 "citations":["CH0003"],"source_documents":["NRC-RET-004"],"calls_made":1,
 "intent_basis":"rules","order_basis":"no order named or remembered","refused_reason":"", …}
```

Send the returned `session_id` with the next message to continue the same
conversation. For a **customer**, the desk-only fields (`intent_basis`,
`intent_reason`, `order_basis`, `refused_reason`, `remembered`) come back blank.

## Validation rules (`api/schemas.py`)

| Request | Rule |
|---|---|
| `LoginRequest` | both fields present and not blank |
| `TicketCreateRequest` | `order_id` ≥ 3 chars; message 8–4000 chars |
| `AdvisoryRequest` | `order_id` ≥ 3 chars (so an empty reference never spends 5 calls); `request_text` ≥ 3 chars, with a default |
| `ReviewRequest` | `decision` in `REVIEW_DECISION_OPTIONS`; `remedy_type`, if given, in `VALID_REMEDY_TYPES`; delay ≥ 0 |
| `ChatRequest` | message 1–4000 chars, not blank; no `session_id` starts a new conversation |
