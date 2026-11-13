# CareIQ Architecture

This page explains how the pieces fit together and why. Read it before changing code.

## The one rule everything follows

> **A model or a written rule produces every fact. A language model only explains
> or composes.**

No language model ever computes an escalation probability, a priority, the cash
cost of a remedy, or whether a manager is needed. Those come from trained models
(`modeling/`) and business rules (`tools/`, `security/`), and they reach the
language model as *facts in its prompt*.

## The big picture

```
                         ┌──────────────────────────────────────────────┐
  Dashboard (Streamlit)  │  dashboard/pages.py  →  dashboard/api_client │   holds no rule,
  port 8501              └──────────────────────┬───────────────────────┘   reads no file
                                                │ HTTP + Bearer token
                         ┌──────────────────────▼───────────────────────┐
  REST service (FastAPI) │ api/routes.py   require_permission(...) on   │   access control
  port 8080              │                 every endpoint               │   enforced here
                         └───┬──────────────┬──────────────┬────────────┘
                             │              │              │
               ┌─────────────▼───┐  ┌───────▼───────┐  ┌───▼──────────────┐
               │ agentic/        │  │ chatbot/      │  │ ingestion/       │
               │ LangGraph       │  │ router →      │  │ reader → clean → │
               │ 6 agents +      │  │ order record /│  │ score → serving  │
               │ human gate      │  │ RAG / ticket  │  │ store, watermark │
               └───┬─────────┬───┘  └──┬─────────┬──┘  └────────┬─────────┘
                   │         │         │         │              │
        ┌──────────▼──┐ ┌────▼─────────▼──┐ ┌────▼──────────────▼───────┐
        │ llm/client  │ │ tools/          │ │ persistence/repository    │
        │ Offline |   │ │ order_data      │ │ 9 SQLite tables           │
        │ Gemini      │ │ prediction      │ │ (plain dicts out)         │
        └─────────────┘ │ retrieval       │ └───────────────────────────┘
                        └──┬───────────┬──┘
              ┌────────────▼──┐   ┌────▼───────────────┐
              │ modeling/     │   │ knowledge/         │
              │ 3 saved models│   │ Chroma index,      │
              │ + schemas     │   │ cosine, 43 chunks  │
              └──────┬────────┘   └────┬───────────────┘
              ┌──────▼─────────────────▼──────┐
              │ data_pipeline/  ←  data/*.csv │   + data/docs/*.pdf
              └───────────────────────────────┘
                  config/config.py is read by every layer
```

## Layers, bottom to top

### 1. `config/config.py`: the single source of truth
Every path, spelling correction, validation range, threshold, authority limit,
word list and reply wording lives here. Modules import constants; they never type
a figure. An auditor can check the business rules by reading this one file.

### 2. `data_pipeline/`: cleaning the raw extracts
The raw CSVs disagree about spelling (`cod`, `Cash On Delivery`), units
(`"1,231.97 INR"`, `"7.58%"`), missing values (`NULL`, `n/a`, `?`) and dates.

| Module | Job |
|---|---|
| `dataset_loader.py` | The only module touching CSV. Raw files load as **all text**; cleaned files load typed |
| `common_cleaning.py` | One repair per function: missing tokens, numbers, categories, booleans, sentinels, outliers, dates, duplicates, gap filling |
| `order_cleaning.py` / `ticket_cleaning.py` | The fixed sequence of repairs. **The live feed uses the same function as training**, so there is no training/serving skew |
| `message_cleaning.py` | Strips markup, greetings, sign-offs, signatures, mail footers and quoted replies; **masks order numbers**; keeps the fault words |
| `quality_profiler.py` | Counts the dirt before cleaning → `output/data_quality_report.json` |
| `feature_builder.py` | Drops identifiers and dates, separates the target, splits numeric and categorical, builds the scaler and one-hot encoder |

Two deliberate choices:
- **`transit_dwell_hours` is never outlier-blanked.** Orders stuck for days
  escalate about 87% of the time against an 18.6% base rate; that "outlier" *is*
  the signal.
- A sentinel such as `-999` or `9999` is **blanked, not clamped**, then filled with
  the median.

### 3. `modeling/`: three bake-offs, thirteen candidates

| Model | Candidates | Chosen on | Typical winner |
|---|---|---|---|
| Escalation risk (will it complain in 7 days?) | LogReg, Tree, RF, HistGB, calibrated SVC, XGBoost, LightGBM | CV ROC AUC | LogisticRegression (held-out AUC ≈ 0.82) |
| Resolution time (hours) | Linear, RF, HistGB | CV R², must beat the mean | HistGradientBoosting (R² ≈ 0.85) |
| Issue mode (10 classes, from text) | MultinomialNB, LogReg, LinearSVC | CV macro F1 | LinearSVC (macro F1 ≈ 0.84) |

Each winner is saved **with its preprocessing** (a scikit-learn `Pipeline`) and its
**ordered feature schema** (`output/models/*_features.json`). Serving builds rows
from the schema, so the column order always matches training. Every candidate's
scores, losers included, go to `output/model_metrics.json`.

### 4. `knowledge/`: the policy index
`PDF → text → sections (split at the document's own headings) → chunks of ≤700
chars with 120-char overlap → embeddings → Chroma`.

- Chunks are labelled `CH0001…CH0043`, stable across rebuilds, and every answer
  cites them.
- The collection is **pinned to cosine distance** and **deleted before each
  rebuild**.
- The collection **records which embedder built it**; querying it with a
  different embedder raises an error rather than returning nonsense.
- `PolicyKnowledge` keeps the collection and its embedder together.

### 5. `tools/`: deterministic fact producers
- `order_data_tools.py`: reads the order book. An unknown order returns `{}`;
  every caller refuses on it.
- `prediction_tools.py`: runs the models and applies the rules:
  `classify_risk_band`, `resolve_priority` (the 3×3 matrix), `severity_for_band`,
  `late_delivery_compensation`, `remedy_cash_cost`, `default_remedy_for`. Inputs
  are validated; a bad input raises `ToolError`.
- `retrieval_tools.py`: searches the policy book and `verify_citations` checks
  that every `[CH####]` an answer cites was actually handed to it.

### 6. `llm/client.py`: one interface, two backends
`generate(prompt, agent_name)` returns `(text, call_record)`. The call record holds
the agent, backend, model, prompt and response sizes, and latency. It never holds
the text itself.
- `OfflineClient` answers from the prompt's `  Label: value` facts in the declared
  format, so parsers and call budgets are genuinely exercised.
- `GeminiClient` uses the free tier, caches replies by prompt hash, and on a
  rate limit waits 60 seconds and retries three times.

### 7. `agentic/`: the advisory workflow
Six agents, one call each, run in a fixed order, with a human gate. See
[README_LANGGRAPH.md](README_LANGGRAPH.md).

### 8. `chatbot/`: cheap and fast, at most one call per turn
`route → answer from the right source → remember → record`.

| Intent | Answered from |
|---|---|
| `order_status` | the order record (never a model) |
| `complaint` | the issue classifier; opens or amends **one** ticket per conversation |
| `policy_question` | retrieved passages only, with checked citations; refuses below 0.6 confidence |
| `small_talk` / `out_of_scope` | fixed replies from config |

Routing uses four layers in order of cost: rules, the issue classifier (for
statements only), question grammar, and finally the language model.

### 9. `persistence/`: nine tables, one query module
`users, tickets, agent_runs, resolutions, review_decisions, llm_calls,
chat_sessions, chat_turns, ingestion_runs`. Only `repository.py` builds queries,
and it always returns plain dictionaries with dates as text. Every status value is
checked against its configured set.

### 10. `security/`: four roles, enforced on the server
- Passwords are hashed with Argon2id, salted per hash.
- Tokens are JWT HS256 carrying `sub`, `role` and `exp`. A tampered or expired
  token is refused.
- Permissions come from `config.ROLE_PERMISSIONS`. An unknown role holds nothing.
- Authority limits ascend: customer ₹0, associate ₹2,000, manager ₹25,000, admin
  (head of care) ₹1,00,00,000. All are finite.
- `needs_manager_approval` is a rule, never a model. It escalates when the amount
  is **strictly above** ₹2,000, for any Goodwill Credit, or for a Full Refund on
  an EMI order.

### 11. `ingestion/`: keeping the book fresh
`SourceReader.read_since(watermark, limit)` → clean (same cleaner as training) →
guard against the answer column → score → **upsert** into
`database/serving_orders.csv` → **then** advance the watermark. A failed pass
never moves the watermark, so no batch is ever lost. `LocalCsvReader` replays the
feed 150 rows at a time and ages rows on each pass (dwell +24 h, slack −1 day).

### 12. `api/` and `dashboard/`
The API loads the models, index and graph once and scores the whole book at
start-up (and on `POST /pipeline/rescore`). The dashboard is only a client: it
shows what the API returns and offers what the API allows.

## Where each figure comes from

| Figure | Produced by | Never by |
|---|---|---|
| Escalation probability | escalation model | an agent |
| Risk band, priority, severity | `prediction_tools` + config matrix | an agent, the dashboard |
| Issue mode + confidence | issue classifier | an agent (it may only disagree) |
| Remedy | policy advisor, checked against the catalogue (else the published default) | free text |
| Cash committed | `remedy_cash_cost` rule | an agent |
| Needs manager? | `needs_manager_approval` rule | an agent, the caller |
| Resolution hours | resolution model (recomputed only if remedy or delay changed) | an agent |
| Customer reply, action steps | composer and finalizer agents | — |
