# The Advisory Workflow (LangGraph)

The advisory workflow drafts a resolution for one order and decides, **by rule**,
whether it can be settled on the spot or must wait for a manager. It is built with
LangGraph in `agentic/`.

## At a glance

- **10 nodes**: 6 agents (one language model call each) plus 4 tool or database steps.
- **A fixed order**, never in parallel, so 6 calls stay inside a free tier's
  per-minute limit.
- **One human gate** that pauses the run, saves it to SQLite, and can resume days
  later, even after a restart.
- **A refusal anywhere ends the run** at that node, with the reason recorded.

## The graph

```
START
  │
  ▼
load_context ............ tools only: look up the order, open the run record
  │  (unknown order → refused here, 0 calls spent)
  ▼
plan_triage ............. call 1  triage planner: how to look into this order
  ▼
assess_risk ............. model → band/priority/severity by rule, then call 2: risk analyst explains it
  ▼
identify_issue .......... classifier reads the message, then call 3: issue analyst judges the reading
  ▼
retrieve_policy ......... retrieve passages, then call 4: policy advisor names remedy + basis + [CH####]
  │                       remedy outside the catalogue → published default; invented citations recorded
  ▼
compose_resolution ...... cost by rule, needs_manager_approval by rule, estimate hours,
  │                       call 5: composer writes reply + steps; DRAFT SAVED with calls so far
  │
  ├── needs_approval = False ──► approve_under_own_authority ... tools only (associate authority)
  │                                        │
  └── needs_approval = True ───► request_manager_review ........ ⏸ PAUSES (interrupt + checkpoint)
                                           │  manager answers via POST /resolutions/{id}/review
                                           ├── REJECT ──────────────────────────┐
                                           ▼                                    │
                                 finalize_resolution ... call 6: rewrite reply  │
                                           │  (holds the recompute rule)        │
                                           ▼                                    ▼
                                 persist_outcome ....... database: decision, remaining calls, status
                                           ▼
                                          END
```

Every node that can refuse is followed by the same condition, `still_going(state)`.
It returns `"stop"` when `refused` is set and sends the run to `persist_outcome`,
which records the refusal and closes the run.

### Call budget

| Ending | Calls | Why |
|---|---|---|
| Inside associate authority | **6** | five to draft, plus the finalizer, approved on the spot |
| Manager APPROVE or MODIFY | **6** | five before the pause, one after (the first five are not repeated) |
| Manager REJECT | **5** | nothing to rewrite, so the finalizer is skipped |
| Unknown order | **0** | refused at `load_context` |

`config.EXPECTED_LLM_CALLS_PER_RUN = 6`; the audit page flags any run above it.

## The state (`agentic/state.py`)

`AdvisoryState` declares every key, and `new_advisory_state` builds all of them, so
no node can read a key that was never set.

| Group | Keys |
|---|---|
| The request | `ticket_id, order_id, request_text, message_text, requested_by, thread_id, run_id, resolution_id` |
| Facts from tools | `order_summary, risk_facts, issue_facts, policy_facts, costed_facts` |
| Agent conclusions | `triage_plan, risk_assessment, issue_analysis, policy_guidance` |
| The resolution | `draft_resolution, review_decision, final_resolution` |
| Control | `refused` (stops everything), `needs_approval` |
| Accounting | `llm_call_log`, `steps` (both **accumulate** via `operator.add`), `persisted_call_count` |

## The six agents (`agentic/agents.py`)

Every prompt lists its facts as indented `Label: value` lines and asks for a
labelled reply such as `REMEDY: …`. Replies are parsed by
`llm.client.ask_for_labels`.

| Agent | Reads | Returns |
|---|---|---|
| triage_planner | order summary, request | `PLAN`, `FOCUS` |
| risk_analyst | order + probability/band/priority (told **not** to recalculate) | `RISK SUMMARY`, `KEY DRIVERS` |
| issue_analyst | message + classifier's mode and confidence | `AGREES`, `ISSUE MODE`, `REASONING` |
| policy_advisor | retrieved passages only | `REMEDY`, `POLICY BASIS`, `CITATIONS` |
| resolution_composer | all of the above + cost + estimate | `CUSTOMER REPLY`, `ACTION STEPS` |
| resolution_finalizer | draft + manager decision | `FINAL REPLY`, `WHAT CHANGED`, `ACTION STEPS` |

## Pause and resume: how the human gate works

1. `request_manager_review_node` calls LangGraph's `interrupt(...)`. The graph
   stops and the full state is checkpointed by `SqliteSaver` into
   `database/checkpoint.db` (`check_same_thread=False`, because the web server
   and the dashboard reach it from several threads).
2. The `agent_runs` row stores the `thread_id`, and the resolution is saved with
   status `PENDING_REVIEW`.
3. A manager calls `POST /resolutions/{id}/review`. Before the graph is touched,
   the API:
   - rejects a decision outside `APPROVE | MODIFY | REJECT` (422),
   - rejects a resolution that was already reviewed (409),
   - rejects an amount above **the reviewer's own** authority, naming the role that
     can approve it (403).
4. `resume_after_review` calls `graph.invoke(Command(resume=decision), thread)`.
   The checkpoint is restored and the run continues from the gate. Nothing before
   it runs again.

A brand-new process can resume the run, because the state lives on disk.
`tests_regression.py::test_a_paused_run_survives_a_restart_and_keeps_its_risk_figures`
proves it with a fresh graph and a fresh connection.

## The recompute rule (in `finalize_resolution`)

At draft time two inputs to the resolution-time model are assumptions:
`approval_delay_hours` (assumed to be 4 h) and `remedy_type` (the advisor's
choice).

- If the manager **changed the remedy** or reported a **different approval
  delay**, the estimate is recomputed. A changed remedy is also re-costed by rule.
- Otherwise the draft's estimate is carried forward, and `recompute_note` says so.
- The **escalation probability, band and priority are never recomputed**.
  `repository.apply_review_outcome` only writes a whitelist of fields, and the risk
  figures are not on it.

## Running it from Python

```python
import uuid
from agentic.graph import (build_advisory_graph, build_initial_state, start_advisory_run,
                           resume_after_review, is_waiting_for_review, thread_config_for)

graph = build_advisory_graph()                       # compiled, with the SQLite checkpointer
thread_id = "THREAD-" + uuid.uuid4().hex[:8]         # must be new for every run
thread = thread_config_for(thread_id)

state = start_advisory_run(graph, build_initial_state({
    "order_id": "NRC-900358",
    "message_text": "the courier has lost my parcel, no scan for days",
    "requested_by": "asha"}, thread_id), thread)

if is_waiting_for_review(state):
    state = resume_after_review(graph, thread, {
        "decision": "MODIFY", "remedy_type": "Replacement",
        "feedback_text": "Offer a replacement first", "approval_delay_hours": 10,
        "reviewer": "vikram", "reviewer_role": "manager"})

print(state["final_resolution"], len(state["llm_call_log"]))
for step in state["steps"]:
    print(step)                                      # the readable trace, node by node
```

The nodes load the models, index and client lazily through
`agentic.nodes.get_services()`. The API injects its already-loaded objects with
`nodes.configure_services(...)`.
