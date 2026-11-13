# Sample Data: Backend and Frontend

## Part 1: the datasets (backend)

All raw files are under `data/` and are **deliberately dirty**. Cleaned copies are
written to `processed/` by `python3 run_data_preparation.py`.

| File | Rows (raw → clean) | One row is | Used for |
|---|---|---|---|
| `orders_train.csv` | 14,210 → 14,000 | an order snapshot with outcome | training the escalation model |
| `orders_eval.csv` | 3,552 → 3,500 | the same, held out | testing the escalation model |
| `input/live_orders_feed.csv` | 900 → 900 | an in-flight order, **no outcome** | the live order book that gets scored |
| `tickets_train.csv` | 9,135 → 9,000 | a closed care ticket | training the resolution-time model |
| `tickets_eval.csv` | 2,200 → 2,200 | the same, held out | testing it |
| `messages.csv` | 9,093 → 8,806 | a customer's message + issue mode | training the issue classifier |
| `order_events.csv`, `shipments.csv`, `customers.csv`, `products.csv` | 29,091 / 4,900 / 4,000 / 400 | reference data | profiled; customers and products are looked up |
| `docs/*.pdf` | 6 documents → 43 chunks | a published policy | the only source the chatbot answers from |

### What "dirty" looks like, and what cleaning does

| Raw value | Cleaned | Rule |
|---|---|---|
| `"4,501.34 INR"`, `"0.3 hrs"` | `4501.34`, `0.3` | thousands separators and units stripped |
| `"14.87%"` | `0.1487` | a percentage is a hundredth |
| `"NULL"`, `" n/a "`, `"?"`, `"#N/A"` | missing → median or `Unknown` | `NULL_TOKEN_LIST` |
| `cod`, `Cash On Delivery`, `cash-on-delivery` | `COD` | `PAYMENT_METHOD_MAPPING` |
| `M-DASH`, `metro dash` | `MetroDash` | `COURIER_MAPPING` |
| `PUNE-W1` | `Pune` | `WAREHOUSE_MAPPING` |
| `Tier2`, `Metro` | `Tier-2`, `Metro` | `CITY_TIER_MAPPING` |
| `days_since_order = 9999`, `prior_orders = -999` | missing → median | `ORDER_RANGE_LIMITS` (blank, never clamp) |
| `transit_dwell_hours = 271` | **kept** | the strongest signal is never outlier-blanked |
| duplicate `order_id` (210 rows) | dropped, first kept | |

Message cleaning, before and after:

```
RAW:   "Respected team, the size is too small, i need a larger size Thanks in advance.\n--\nSent from my iPhone Ref NRC-109106."
CLEAN: "the size is too small i need a larger size"
```

### Key order columns

`order_id` (NRC-######), `order_value_inr`, `delivery_slack_days` (negative =
late), `transit_dwell_hours` (hours since the last scan), `route_failure_rate_30d`,
`customer_tier` (Platinum/Gold/Standard), `payment_method` (Prepaid/COD/EMI),
`courier`, `product_category`, and `escalation_within_7_days` (the target; about
18.6% are 1).

### The ten issue modes and their default remedies

| Issue mode | Default remedy | Cash committed |
|---|---|---|
| Late Delivery | SLA Compensation | ₹100 / ₹250 / ₹500 by days late (cap ₹500) |
| Courier Lost Parcel | Full Refund | the order value |
| Damaged in Transit | Replacement | ₹0 |
| Wrong Item Shipped | Replacement | ₹0 |
| Short Shipment | Reship Missing Item | ₹0 |
| Faulty on Arrival | Replacement | ₹0 |
| Not As Described | Full Refund | the order value |
| Sizing or Fit | Exchange | ₹0 |
| Billing Error | Partial Refund | 30% of the order value |
| Refund Delay | Explain and Expedite | ₹0 |

Goodwill Credit (₹500) always needs a manager.

---

## Part 2: orders to try

These orders come from the live feed (`data/input/live_orders_feed.csv`). Risk
figures can move after ingestion re-ages the book.

| Order | Value | Payment | Why it is useful |
|---|---|---|---|
| `NRC-900358` | ₹3,028.60 | COD | **HIGH risk** (≈0.97), 139 h since last scan, 14 days late |
| `NRC-900306` | ₹856.01 | Prepaid | HIGH risk, small value: a refund stays **inside** associate authority |
| `NRC-900001` | ₹2,215.35 | Prepaid | a full refund is just **above ₹2,000**, so it pauses for a manager |
| `NRC-900749` | ₹35,662.08 | Prepaid | a full refund is above the **manager** limit, so only an admin may approve it |
| `NRC-900582` | ₹946.86 | EMI | any **full refund on EMI** needs a manager, whatever the amount |
| `NRC-000000` | — | — | not on file: refused everywhere (404 or 422) |
| `TTT-009983` | — | — | a mistyped reference; the chat names it and asks again |

## Part 3: backend sample requests

Python, against the running service:

```python
import requests
B = "http://localhost:8080"
tok = requests.post(B + "/auth/login", json={"user_name": "asha", "password": "associate123"}).json()["access_token"]
H = {"Authorization": "Bearer " + tok}

requests.get(B + "/orders/NRC-900358/risk", headers=H).json()
requests.post(B + "/tickets", headers=H, json={
    "order_id": "NRC-900358", "message_text": "the courier has lost my parcel, no scan for days"}).json()
requests.post(B + "/advisories", headers=H, json={
    "order_id": "NRC-900001", "message_text": "the courier has lost my parcel, no scan for days"}).json()
```

Messages that classify cleanly to each issue mode:

| Message | Expected mode |
|---|---|
| "the parcel arrived cracked and the screen is shattered" | Damaged in Transit |
| "my delivery is six days late and still has not arrived" | Late Delivery |
| "the courier has lost my parcel entirely, no scan for days" | Courier Lost Parcel |
| "i was charged twice for this order" | Billing Error |
| "the shoes do not fit, they are narrow and i need the next size up" | Sizing or Fit |
| "the device is dead on arrival and will not power up" | Faulty on Arrival |
| "i ordered 3 items but only received part of them" | Short Shipment |
| "you sent me the wrong colour and wrong model" | Wrong Item Shipped |
| "the specifications do not match what was described" | Not As Described |
| "the return was picked up weeks ago but my refund has not been credited" | Refund Delay |

## Part 4: frontend sample inputs (chat)

| Type this | What happens |
|---|---|
| How many days do I have to return a pair of shoes? | cited answer: 30 days for Apparel and Footwear [CH0003] |
| What compensation is due when a delivery is late? | cited answer from NRC-SHIP-002 |
| Who may approve a goodwill credit? | cited answer from NRC-CSC-014 |
| Where is my order NRC-900001? | the delivery position from the order record |
| when will it arrive *(next turn)* | remembers NRC-900001 ("carried from earlier") |
| My parcel arrived and the box was crushed and the item is cracked | opens a case, then asks for the order number |
| NRC-900358 *(next turn)* | the order is added to **the same** case |
| TTT-009983 | "I could not find an order with the reference TTT-009983…" |
| What is the price of gold in Mumbai today? | politely out of scope |
| Do you price match competitors? | declined: nothing in the policy book covers it (out of scope) |
| thanks very much | small talk |

## Part 5: golden sets (evaluation)

`evaluation/golden_sets/` holds the replayable sets used by `run_evaluation.py`:
`retrieval.json` (25 questions with their expected document),
`unanswerable.json` (12), `routing_known.json` (27), `routing_heldout.json` (20,
reported but not gated), `memory.json` (6 conversations) and `advisory.json` (8
runs). Add a line to any of them to extend the evaluation; no code change is
needed.
