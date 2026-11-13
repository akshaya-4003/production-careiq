# Using the CareIQ Dashboard

The dashboard runs at <http://localhost:8501>. It is a client of the REST service,
so start the service first:

```bash
python3 run_api_service.py                  # terminal 1
python3 -m streamlit run run_dashboard.py   # terminal 2
```

If the service is down, the sign-in page says so and shows the command that starts
it.

## Signing in

The sign-in page also shows the service status and a **Demo accounts** list.

| Sign in as | Password | You will see |
|---|---|---|
| `rohan` (customer) | `customer123` | Chat |
| `asha` (associate) | `associate123` | Chat, My queue |
| `vikram` (manager) | `manager123` | Chat, My queue, Order risk, Review queue |
| `divya` (admin) | `admin123` | all seven pages |

The **sidebar** shows who you are, your role, your authority limit (in words for a
customer, who may commit nothing), the running backends, and **Sign out**. The page
list contains only the pages your role holds; the service would refuse the others
anyway.

---

## 1. Chat (everyone)

Ask about policy, ask where an order is, or say what went wrong.

- Click an example prompt, or type in the box at the bottom.
- **Policy questions** are answered only from the six documents, with sources such
  as `NRC-RET-004 | passages CH0003` under the answer. If the documents don't
  cover a question, the assistant says so rather than guessing.
- **Order questions** ("where is my order NRC-900001") are answered from the order
  record.
- **Complaints** open a case and show its reference, for example `TKT-000004`. If
  you didn't give an order number, send it as your next message; it is added to the
  same case rather than opening a new one.
- **New conversation** clears the memory of the current one.

Staff also see a grey line under each answer showing the route and how it was
decided, the order and whether it was named or remembered, the issue mode, the
model calls, and an orange internal note when a turn was refused. Customers never
see these notes.

## 2. My queue (associate, manager, admin)

1. The table lists your tickets plus every **unassigned** ticket a customer opened.
   Filter by status; managers and admins can switch on **Show every queue**.
2. **Click a row** to open the ticket. It shows the customer's message, the issue
   mode and confidence, and the order's risk tiles.
3. Press **Run advisory workflow**. A spinner names what is happening (five model
   calls). The result is one of:
   - a green **Approved on the spot under associate authority** (remedy within
     ₹2,000), or
   - an amber **waiting for a care manager**, with the reason.

   Either way you see the remedy, the cash committed, the estimate, the reply, the
   action steps, the policy basis and the cited passages. **Workflow trace** shows
   every node.
4. A ticket with **no order** is marked and the button is disabled, because every
   workflow step starts from the order record.
5. **Open a ticket for a customer** takes an order reference and what the customer
   said. **Look up an order** shows any order's risk and delivery position.

## 3. Order risk (manager, admin)

- Tiles show orders scored, HIGH, MEDIUM and LOW counts, and the order value at
  HIGH risk.
- Two bar charts show orders by band and by queue priority.
- The table lists orders **worst first**, with a probability bar. Filter by band and
  choose how many rows to show.
- Click a row to see that order's tiles.
- Figures change only when the service re-scores (see Pipeline).

## 4. Review queue (manager, admin)

1. Your authority limit is shown at the top.
2. Click a pending resolution. You see **Why this needs you** (the rule that
   escalated it) and the full draft.
3. Choose **APPROVE**, **MODIFY** (pick a different remedy) or **REJECT**.
   Optionally enter the hours the draft actually waited and feedback for the reply.
4. **Submit decision.** Afterwards the page shows the **drafted reply and the final
   reply side by side**, what changed, and whether the time estimate was recomputed.
5. If the amount is above your own limit, the service refuses and names the role
   that can approve it. For example, ₹35,662 needs an admin.

## 5. Audit trail (admin)

Every advisory run, with **calls made against calls expected (6)**, flagged when over
budget. Click a run to see each language model call (agent, backend, model, sizes,
latency; never the text) and every decision recorded on its resolution.

## 6. Evaluation (admin)

- **Generative evaluation**: the checks from `run_evaluation.py` with value, bar
  and pass or fail.
- **Model bake-offs**: the chosen model for each task and every candidate's
  scores, best first.
- **Data quality**: how dirty each raw extract was before cleaning.

A tab with no report yet names the command that produces it.

## 7. Pipeline (admin)

The ingestion log (rows read and scored, HIGH risk found per pass) and a
**Re-score now** button. Ingestion (`python3 run_ingestion_service.py --fast 1`)
changes the serving file; re-scoring is what moves the Order risk page.

---

## A 5-minute demo script

1. **rohan**: Chat → "My parcel arrived and the box was crushed and the item is
   cracked" → a case opens → send "NRC-900358" → the same case is updated.
2. **asha**: My queue → click that ticket → **Run advisory workflow** → it pauses
   for a manager (₹3,028 full refund) or is approved on the spot, depending on the
   remedy.
3. **vikram**: Review queue → click it → **MODIFY** to *Replacement* → see the
   side-by-side and the recomputed estimate.
4. **divya**: Audit trail → the run shows 6 of 6 calls → Evaluation → every bar
   passes.
