"""The CareIQ dashboard: seven pages, each naming the permission it needs.

The dashboard is a pure client of the REST service. It computes no figure and
applies no rule: every number, list and decision it shows arrives from the API,
and every action it offers is a request the API may refuse.
"""

import pandas as pd
import streamlit as st

from config import config
from dashboard import api_client as api

# ---------------------------------------------------------------- session helpers
def _streamlit_version():
    parts = []
    for piece in st.__version__.split(".")[:2]:
        digits = "".join(character for character in piece if character.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


# Full-width layout keyword: width="stretch" from 1.50, use_container_width before it.
STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else \
    {"use_container_width": True}


def _token():
    return st.session_state.get("token", "")


def _me():
    return st.session_state.get("me", {})


def _has(permission):
    return permission in _me().get("permissions", [])


def _show_error(error):
    if isinstance(error, api.ServiceUnavailable):
        st.error(error.message)
    elif isinstance(error, api.ApiError):
        st.error("The service refused this ({0}): {1}".format(error.status_code, error.message))
    else:
        st.error(str(error))


def _catalogue():
    if "catalogue" not in st.session_state:
        st.session_state["catalogue"] = api.catalogue(_token())
    return st.session_state["catalogue"]


def _band_badge(band):
    colour = _catalogue().get("risk_band_colours", {}).get(band, "gray")
    return ":{0}[**{1}**]".format(colour, band)


def _decision_badge(decision):
    colour = _catalogue().get("review_decision_colours", {}).get(decision, "gray")
    return ":{0}[**{1}**]".format(colour, decision)


def _money(value):
    try:
        return "INR {0:,.2f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _authority_in_words(limit, role):
    if not limit:
        return "None. A remedy is never authorised by the person who receives it."
    if role == config.ROLE_ADMIN:
        return _money(limit) + " (head of customer care; nobody above)"
    return _money(limit)


def _frame(rows, columns=None):
    frame = pd.DataFrame(rows)
    if columns and not frame.empty:
        frame = frame[[column for column in columns if column in frame.columns]]
    return frame


def _selected_row(event, frame):
    try:
        rows = event.selection.rows
    except AttributeError:
        rows = []
    if not rows or frame.empty:
        return None
    return frame.iloc[rows[0]].to_dict()


# ---------------------------------------------------------------- sign in
def render_sign_in():
    st.title(config.DASHBOARD_TITLE)
    st.caption(config.DASHBOARD_SUBTITLE)
    try:
        health = api.health()
    except api.ApiError as error:
        st.error(error.message)
        st.info("The dashboard is a client of the CareIQ REST service, so it shows nothing "
                "until the service is running. In another terminal run:\n\n"
                "    python3 run_api_service.py")
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.form("sign-in"):
            st.subheader("Sign in")
            user_name = st.text_input("User name")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary",
                                              **STRETCH)
        if submitted:
            try:
                token = api.login(user_name.strip(), password)["access_token"]
                st.session_state["token"] = token
                st.session_state["me"] = api.who_am_i(token)
                st.session_state.pop("catalogue", None)
                st.rerun()
            except api.ApiError as error:
                _show_error(error)
    with right:
        with st.container(border=True):
            st.markdown("**Service status**")
            st.write("Orders monitored: **{0}**".format(health["orders_monitored"]))
            st.write("Policy passages indexed: **{0}**".format(health["policy_chunks"]))
            st.write("Language model: **{0}** | Embeddings: **{1}**".format(
                health["llm_backend"], health["embedding_backend"]))
        with st.expander("Demo accounts"):
            st.dataframe(pd.DataFrame([{"user": a["user_name"], "password": a["password"],
                                        "role": a["role"]} for a in config.DEMO_USERS]),
                         hide_index=True, **STRETCH)


# ---------------------------------------------------------------- 1. chat
def render_chat():
    me = _me()
    is_customer = me.get("role") == config.ROLE_CUSTOMER
    st.header("Ask CareIQ")
    st.caption("Answers come only from Northlane's six published policies or from the order "
               "record. A complaint is logged as a case on the care desk queue.")
    history = st.session_state.setdefault("chat_history", [])
    top = st.columns([4, 1])
    with top[1]:
        if st.button("New conversation", **STRETCH):
            st.session_state["chat_history"] = []
            st.session_state.pop("chat_session", None)
            st.rerun()
    if not history:
        st.markdown("**Try one of these:**")
        examples = ["How many days do I have to return a pair of shoes?",
                    "Where is my order NRC-900001?",
                    "My parcel arrived and the box was crushed and the item is cracked",
                    "What compensation is due when a delivery is late?"]
        columns = st.columns(len(examples))
        for column, example in zip(columns, examples):
            if column.button(example, **STRETCH):
                st.session_state["chat_pending"] = example
                st.rerun()

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("meta"):
                _render_chat_meta(turn["meta"], is_customer)

    message = st.chat_input("Ask about a return, an order, or tell us what went wrong")
    message = message or st.session_state.pop("chat_pending", None)
    if message:
        history.append({"role": "user", "content": message})
        try:
            with st.spinner("Finding the answer..."):
                reply = api.chat(_token(), message, st.session_state.get("chat_session"))
            st.session_state["chat_session"] = reply["session_id"]
            history.append({"role": "assistant", "content": reply["answer"], "meta": reply})
        except api.ApiError as error:
            history.append({"role": "assistant", "content": "Sorry, that failed: "
                            + error.message})
        st.rerun()


def _render_chat_meta(reply, is_customer):
    if reply.get("citations"):
        st.caption("Sources: " + ", ".join(reply.get("source_documents", []))
                   + "  |  passages " + ", ".join(reply["citations"]))
    if reply.get("ticket_id"):
        st.caption("Case reference: " + reply["ticket_id"])
    if is_customer:
        return  # desk notes describe how the desk reasons; a customer sees the answer only
    parts = ["route: **{0}** ({1})".format(reply.get("intent", ""), reply.get("intent_basis", ""))]
    if reply.get("order_id"):
        parts.append("order: **{0}** ({1})".format(reply["order_id"], reply.get("order_basis", "")))
    if reply.get("issue_mode"):
        parts.append("issue: **{0}**".format(reply["issue_mode"]))
    parts.append("model calls: {0}".format(reply.get("calls_made", 0)))
    st.caption(" | ".join(parts))
    if reply.get("refused_reason"):
        st.caption(":orange[Internal note: " + reply["refused_reason"] + "]")


# ---------------------------------------------------------------- 2. own queue
def render_queue():
    st.header("My queue")
    st.caption("Your tickets and every unassigned ticket a customer has opened. Select a ticket "
               "to run the advisory workflow on it.")
    controls = st.columns([2, 2, 3])
    status = controls[0].selectbox("Status", ["OPEN", "CLOSED", "All"], index=0)
    all_queues = False
    if _has("all_queues"):
        all_queues = controls[1].toggle("Show every queue", value=False)
    try:
        tickets = api.list_tickets(_token(), "" if status == "All" else status, all_queues)
    except api.ApiError as error:
        _show_error(error)
        return

    columns = ["ticket_id", "order_id", "issue_mode", "issue_confidence", "assigned_to",
               "opened_by", "source_channel", "status", "opened_at"]
    frame = _frame(tickets, columns)
    if frame.empty:
        st.info("No tickets in this view. Open one below, or have a customer report a problem "
                "in the chat.")
    else:
        frame = frame.assign(assigned_to=frame["assigned_to"].replace("", "(unassigned)"),
                             order_id=frame["order_id"].replace("", "(no order yet)"))
        event = st.dataframe(frame, hide_index=True, **STRETCH,
                             on_select="rerun", selection_mode="single-row",
                             column_config={"issue_confidence": st.column_config.ProgressColumn(
                                 "confidence", min_value=0.0, max_value=1.0, format="%.2f")})
        chosen = _selected_row(event, frame)
        if chosen:
            ticket = next(t for t in tickets if t["ticket_id"] == chosen["ticket_id"])
            _render_ticket(ticket)

    with st.expander("Open a ticket for a customer"):
        with st.form("new-ticket"):
            order_id = st.text_input("Order reference", placeholder="NRC-900001")
            message = st.text_area("What the customer said")
            if st.form_submit_button("Open ticket", type="primary"):
                try:
                    created = api.create_ticket(_token(), order_id, message)
                    st.success("Opened {0}, read as {1} ({2:.0%} confidence).".format(
                        created["ticket_id"], created["issue_mode"], created["issue_confidence"]))
                except api.ApiError as error:
                    _show_error(error)

    if _has("order_read"):
        with st.expander("Look up an order"):
            reference = st.text_input("Order reference", key="lookup-order",
                                      placeholder="NRC-900001")
            if reference:
                _render_order(reference)


def _render_ticket(ticket):
    with st.container(border=True):
        st.subheader(ticket["ticket_id"])
        st.write("**Customer message:** " + ticket["message_text"])
        st.write("**Issue mode:** {0} ({1:.0%} confidence) | **Status:** {2}".format(
            ticket["issue_mode"], float(ticket["issue_confidence"]), ticket["status"]))
        if not ticket["has_order"]:
            st.warning("This ticket has no order reference yet. Every workflow step starts "
                       "from the order record, so the advisory cannot run until the customer "
                       "supplies one.")
        if ticket["has_order"] and _has("order_read"):
            _render_order(ticket["order_id"])
        disabled = (not ticket["has_order"] or ticket["status"] != "OPEN"
                    or not _has("advisory_start"))
        if st.button("Run advisory workflow", type="primary", disabled=disabled,
                     key="advise-" + ticket["ticket_id"]):
            try:
                with st.spinner("Running the advisory workflow: triage, risk, issue, policy and "
                                "drafting (five language model calls)..."):
                    result = api.start_advisory(_token(), ticket["order_id"], ticket["ticket_id"])
                st.session_state["last_advisory"] = result
            except api.ApiError as error:
                _show_error(error)
        last = st.session_state.get("last_advisory")
        if last and last["resolution"].get("ticket_id") == ticket["ticket_id"]:
            _render_advisory_result(last)


def _render_order(order_id):
    try:
        order = api.get_order(_token(), order_id)
        risk = api.get_order_risk(_token(), order_id)
    except api.ApiError as error:
        _show_error(error)
        return
    summary = order["summary"]
    cells = st.columns(5)
    cells[0].metric("Escalation risk", "{0:.0%}".format(risk["escalation_probability"]),
                    border=True)
    cells[1].metric("Band / priority", "{0} / {1}".format(risk["risk_band"], risk["priority"]),
                    border=True)
    cells[2].metric("Order value", _money(summary["order_value_inr"]), border=True)
    cells[3].metric("Since last scan", "{0} h".format(summary["hours_since_last_scan"]),
                    border=True)
    cells[4].metric("Customer tier", summary["customer_tier"], border=True)
    st.caption("{0}: {1} x {2} via {3}, paid by {4}, {5}.".format(
        summary["order_id"], summary["item_count"], summary["product_category"],
        summary["courier"], summary["payment_method"], summary["delivery_position"]))


def _render_advisory_result(result):
    resolution = result["resolution"]
    if result["waiting_for_review"]:
        st.warning("Drafted and **waiting for a care manager**: " + resolution["approval_reason"])
    else:
        st.success("Approved on the spot under associate authority: "
                   + resolution["approval_reason"])
    _render_resolution_card(resolution)
    st.caption("Language model calls: {0} of {1} budgeted".format(result["calls_made"],
                                                                  result["calls_expected"]))
    if result.get("steps"):
        with st.expander("Workflow trace"):
            st.dataframe(pd.DataFrame(result["steps"]), hide_index=True,
                         **STRETCH)


def _render_resolution_card(resolution):
    cells = st.columns(5)
    cells[0].metric("Remedy", resolution["remedy_type"], border=True)
    cells[1].metric("Cash committed", _money(resolution["remedy_amount_inr"]), border=True)
    cells[2].metric("Estimate", "{0:.1f} h".format(resolution["estimated_resolution_hours"]),
                    border=True)
    cells[3].metric("Escalation risk", "{0:.0%}".format(resolution["escalation_probability"]),
                    border=True)
    cells[4].metric("Priority", resolution["priority"], border=True)
    st.markdown("**Risk band:** " + _band_badge(resolution["risk_band"]) + " | **Issue mode:** "
                + resolution["issue_mode"] + " | **Status:** " + resolution["status"])
    st.markdown("**Reply to the customer**")
    st.info(resolution["customer_reply"] or "(no reply drafted)")
    st.markdown("**Action steps**")
    st.text(resolution["action_steps"] or "(none)")
    st.markdown("**Policy basis:** " + (resolution["policy_basis"] or "(none)"))
    st.caption("Cited passages: " + (resolution["citations"] or "none") + " | Documents: "
               + (resolution["source_documents"] or "none"))


# ---------------------------------------------------------------- 3. risk dashboard
def render_risk():
    st.header("Order risk")
    st.caption("Every in-flight order, scored by the escalation model and banded by policy "
               "NRC-POL-001. Worst first.")
    try:
        summary = api.risk_summary(_token())
    except api.ApiError as error:
        _show_error(error)
        return
    counts = summary["counts_by_band"]
    cells = st.columns(5)
    cells[0].metric("Orders scored", summary["orders_scored"], border=True)
    cells[1].metric("HIGH risk", counts.get("HIGH", 0), border=True)
    cells[2].metric("MEDIUM risk", counts.get("MEDIUM", 0), border=True)
    cells[3].metric("LOW risk", counts.get("LOW", 0), border=True)
    cells[4].metric("Exposure at HIGH risk", _money(summary["high_risk_exposure_inr"]),
                    border=True)
    st.caption("Scored at {0}. HIGH at or above {1:.0%}, MEDIUM at or above {2:.0%}.".format(
        summary["scored_at"], summary["thresholds"]["high"], summary["thresholds"]["medium"]))

    charts = st.columns(2)
    with charts[0]:
        st.markdown("**Orders by risk band**")
        st.bar_chart(pd.DataFrame({"band": list(counts), "orders": list(counts.values())}),
                     x="band", y="orders")
    with charts[1]:
        st.markdown("**Orders by queue priority**")
        priorities = summary["counts_by_priority"]
        st.bar_chart(pd.DataFrame({"priority": list(priorities),
                                   "orders": list(priorities.values())}),
                     x="priority", y="orders")

    controls = st.columns([2, 2, 4])
    band = controls[0].selectbox("Band", ["All"] + _catalogue()["risk_bands"])
    limit = controls[1].number_input("Rows", min_value=10, max_value=1000, value=100, step=10)
    try:
        rows = api.risk_orders(_token(), "" if band == "All" else band, int(limit))
    except api.ApiError as error:
        _show_error(error)
        return
    frame = _frame(rows, ["order_id", "escalation_probability", "risk_band", "priority",
                          "severity_code", "customer_tier", "order_value_inr",
                          "transit_dwell_hours", "delivery_slack_days", "courier",
                          "product_category", "payment_method"])
    event = st.dataframe(frame, hide_index=True, **STRETCH, on_select="rerun",
                         selection_mode="single-row", column_config={
                             "escalation_probability": st.column_config.ProgressColumn(
                                 "probability", min_value=0.0, max_value=1.0, format="%.3f"),
                             "order_value_inr": st.column_config.NumberColumn(
                                 "value (INR)", format="%.2f")})
    chosen = _selected_row(event, frame)
    if chosen and _has("order_read"):
        st.subheader(chosen["order_id"])
        _render_order(chosen["order_id"])


# ---------------------------------------------------------------- 4. review queue
def render_review():
    st.header("Review queue")
    st.caption("Drafts that commit money above an associate's authority, any goodwill credit and "
               "any full refund on an EMI order wait here for a decision.")
    me = _me()
    st.info("Your authority limit: " + _authority_in_words(me.get("authority_limit_inr"),
                                                            me.get("role")))
    try:
        pending = api.list_resolutions(_token(), config.RESOLUTION_STATUS_PENDING)
    except api.ApiError as error:
        _show_error(error)
        return
    if not pending:
        st.success("Nothing is waiting for a decision.")
    else:
        frame = _frame(pending, ["resolution_id", "order_id", "ticket_id", "remedy_type",
                                 "remedy_amount_inr", "risk_band", "priority", "issue_mode",
                                 "customer_tier", "created_at"])
        event = st.dataframe(frame, hide_index=True, **STRETCH,
                             on_select="rerun", selection_mode="single-row")
        chosen = _selected_row(event, frame)
        if chosen:
            resolution = next(r for r in pending if r["resolution_id"] == chosen["resolution_id"])
            _render_review_form(resolution)

    outcome = st.session_state.get("last_review")
    if outcome:
        _render_review_outcome(outcome)


def _render_review_form(resolution):
    with st.container(border=True):
        st.subheader(resolution["resolution_id"])
        st.warning("Why this needs you: " + resolution["approval_reason"])
        _render_resolution_card(resolution)
        catalogue = _catalogue()
        with st.form("review-" + resolution["resolution_id"]):
            decision = st.radio("Decision", catalogue["review_decisions"], horizontal=True)
            remedies = catalogue["remedy_types"]
            remedy = st.selectbox("Remedy (used when you modify)", remedies,
                                  index=remedies.index(resolution["remedy_type"])
                                  if resolution["remedy_type"] in remedies else 0)
            assumed = catalogue["planning_assumptions"].get("approval_delay_hours", 0.0)
            delay = st.number_input("Hours this draft actually waited for approval",
                                    min_value=0.0, value=float(assumed), step=0.5,
                                    help="The estimate assumed this figure; if it changed, the "
                                         "estimate is recomputed.")
            feedback = st.text_area("Feedback for the reply (optional)")
            submitted = st.form_submit_button("Submit decision", type="primary")
        if submitted:
            try:
                with st.spinner("Recording the decision and rewriting the reply (one language "
                                "model call; none on reject)..."):
                    result = api.review_resolution(
                        _token(), resolution["resolution_id"], decision, feedback,
                        remedy if decision == "MODIFY" else None, delay)
                st.session_state["last_review"] = {"before": resolution, "after": result,
                                                   "decision": decision}
                st.rerun()
            except api.ApiError as error:
                _show_error(error)


def _render_review_outcome(outcome):
    before, after = outcome["before"], outcome["after"]["resolution"]
    st.subheader("Decision recorded: " + before["resolution_id"])
    st.markdown("Decision " + _decision_badge(outcome["decision"]) + " | final status **"
                + after["status"] + "**")
    left, right = st.columns(2)
    with left:
        st.markdown("**Drafted reply**")
        st.info(before["customer_reply"])
        st.caption("Remedy {0}, {1}, estimate {2:.1f} h".format(
            before["remedy_type"], _money(before["remedy_amount_inr"]),
            before["estimated_resolution_hours"]))
    with right:
        st.markdown("**Final reply**")
        if outcome["decision"] == "REJECT":
            st.error("Rejected. Nothing is sent to the customer.")
        else:
            st.success(after["customer_reply"])
        st.caption("Remedy {0}, {1}, estimate {2:.1f} h".format(
            after["remedy_type"], _money(after["remedy_amount_inr"]),
            after["estimated_resolution_hours"]))
    if after.get("what_changed"):
        st.markdown("**What changed:** " + after["what_changed"])
    if after.get("recompute_note"):
        st.caption(after["recompute_note"])


# ---------------------------------------------------------------- 5. audit trail
def render_audit():
    st.header("Audit trail")
    st.caption("Every advisory run, its call budget, and every decision recorded against it.")
    try:
        runs = api.list_runs(_token())
    except api.ApiError as error:
        _show_error(error)
        return
    if not runs:
        st.info("No advisory runs yet.")
        return
    frame = _frame(runs, ["run_id", "order_id", "ticket_id", "requested_by", "status",
                          "calls_made", "calls_expected", "over_budget", "resolution_status",
                          "refused_reason", "started_at", "finished_at"])
    over = int(frame["over_budget"].sum()) if "over_budget" in frame else 0
    cells = st.columns(3)
    cells[0].metric("Runs", len(frame), border=True)
    cells[1].metric("Language model calls", int(frame["calls_made"].sum()), border=True)
    cells[2].metric("Runs over budget", over, border=True)
    event = st.dataframe(frame, hide_index=True, **STRETCH, on_select="rerun",
                         selection_mode="single-row")
    chosen = _selected_row(event, frame)
    if chosen:
        run = next(r for r in runs if r["run_id"] == chosen["run_id"])
        budget = api.run_calls(_token(), run["run_id"])
        st.subheader(run["run_id"])
        st.write("Calls made **{0}** of **{1}** expected{2}".format(
            budget["calls_made"], budget["calls_expected"],
            " - :red[over budget]" if budget["over_budget"] else ""))
        st.dataframe(_frame(budget["calls"], ["agent_name", "backend", "model_name",
                                              "prompt_characters", "response_characters",
                                              "latency_ms", "created_at"]),
                     hide_index=True, **STRETCH)
        if run.get("resolution_id"):
            reviews = api.resolution_reviews(_token(), run["resolution_id"])
            st.markdown("**Decisions on " + run["resolution_id"] + "**")
            if reviews:
                st.dataframe(_frame(reviews, ["decision", "reviewer", "reviewer_role",
                                              "authority_basis", "feedback_text",
                                              "decided_at"]),
                             hide_index=True, **STRETCH)
            else:
                st.caption("No decision yet.")


# ---------------------------------------------------------------- 6. evaluation
def render_evaluation():
    st.header("Evaluation and model metrics")
    tabs = st.tabs(["Generative evaluation", "Model bake-offs", "Data quality"])
    with tabs[0]:
        report = api.evaluation_report(_token())
        if not report["available"]:
            st.info(report["message"])
        else:
            checks = pd.DataFrame(report["report"]["checks"])
            passed = int(checks["passed"].sum())
            st.metric("Checks passed", "{0} of {1}".format(passed, len(checks)), border=True)
            st.caption("Language model: {0} | embeddings: {1}".format(
                report["report"]["backend"], report["report"]["embedding_backend"]))
            st.dataframe(checks[["check", "value", "bar", "direction", "gated", "passed"]],
                         hide_index=True, **STRETCH)
            st.caption("The held-out routing figure is reported and not gated: a keyword router "
                       "always scores far lower on wording it was never tuned on.")
    with tabs[1]:
        metrics = api.model_metrics(_token())
        if not metrics["available"]:
            st.info(metrics["message"])
        else:
            data = metrics["report"]
            chosen = data["chosen_models"]
            cells = st.columns(3)
            cells[0].metric("Escalation risk", chosen["escalation_risk"], border=True)
            cells[1].metric("Resolution time", chosen["resolution_time"], border=True)
            cells[2].metric("Issue mode", chosen["issue_mode"], border=True)
            for key, title in (("escalation_risk", "Escalation risk: 7 candidates"),
                               ("resolution_time", "Resolution time: 3 candidates"),
                               ("issue_mode", "Issue mode: 3 candidates")):
                st.markdown("**" + title + "** (ranked, best first)")
                table = pd.DataFrame(data[key]["candidates"]).T
                table = table.loc[[name for name in data[key]["ranking"] if name in table.index]]
                st.dataframe(table, **STRETCH)
    with tabs[2]:
        quality = api.data_quality(_token())
        if not quality["available"]:
            st.info(quality["message"])
        else:
            rows = quality["report"]["datasets"]
            st.dataframe(_frame(rows, ["dataset", "rows", "columns", "missing_cells",
                                       "missing_share", "duplicate_rows", "total_needing_repair",
                                       "total_out_of_range"]),
                         hide_index=True, **STRETCH)


# ---------------------------------------------------------------- 7. pipeline
def render_pipeline():
    st.header("Ingestion pipeline")
    st.caption("Ingestion writes the serving store on a timer (python3 run_ingestion_service.py). "
               "The service re-scores at start up and when you press Re-score now, so that "
               "button is what moves the risk page.")
    if st.button("Re-score now", type="primary"):
        try:
            with st.spinner("Scoring the whole order book..."):
                result = api.pipeline_rescore(_token())
            st.success("Scored {0} orders at {1}.".format(result["orders_scored"],
                                                         result["scored_at"]))
        except api.ApiError as error:
            _show_error(error)
    try:
        runs = api.pipeline_runs(_token())
    except api.ApiError as error:
        _show_error(error)
        return
    if not runs:
        st.info("No ingestion pass has run yet. Try: python3 run_ingestion_service.py --once")
        return
    frame = _frame(runs, ["id", "source", "status", "rows_read", "rows_scored",
                          "high_risk_found", "message", "started_at", "finished_at"])
    st.dataframe(frame, hide_index=True, **STRETCH)
    st.bar_chart(frame.iloc[::-1], x="id", y="high_risk_found")


# ---------------------------------------------------------------- navigation
# (label, permission needed, renderer). Adding a role is a data change in config.
PAGES = [
    ("Chat", "chat", render_chat),
    ("My queue", "own_queue", render_queue),
    ("Order risk", "risk_dashboard", render_risk),
    ("Review queue", "review_decide", render_review),
    ("Audit trail", "audit_read", render_audit),
    ("Evaluation", "evaluation_read", render_evaluation),
    ("Pipeline", "pipeline_read", render_pipeline),
]


def pages_for(permissions):
    return [page for page in PAGES if page[1] in permissions]


def main():
    st.set_page_config(page_title=config.DASHBOARD_TITLE, layout="wide")
    if not _token():
        render_sign_in()
        return
    me = _me()
    try:
        health = api.health()
    except api.ApiError as error:
        st.error(error.message)
        return
    with st.sidebar:
        st.title(config.DASHBOARD_TITLE)
        st.caption(config.DASHBOARD_SUBTITLE)
        with st.container(border=True):
            st.markdown("**{0}**".format(me.get("display_name") or me.get("user_name")))
            st.caption("Role: " + me.get("role", ""))
            st.caption("Authority: " + _authority_in_words(me.get("authority_limit_inr"),
                                                           me.get("role")))
        allowed = pages_for(me.get("permissions", []))
        labels = [label for label, _, _ in allowed]
        choice = st.radio("Pages", labels, label_visibility="collapsed")
        st.divider()
        st.caption("Model: {0} | Embeddings: {1}".format(health["llm_backend"],
                                                         health["embedding_backend"]))
        st.caption("{0} orders | {1} policy passages".format(health["orders_monitored"],
                                                             health["policy_chunks"]))
        if st.button("Sign out", **STRETCH):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    for label, _, renderer in allowed:
        if label == choice:
            try:
                renderer()
            except api.ApiError as error:
                _show_error(error)
                if error.status_code == 401:
                    st.session_state.pop("token", None)
                    st.button("Sign in again")
