"""Reads the order book, the customer list and the catalogue.

The only place the serving data is touched. An unknown key always comes back as
an empty dictionary, and every caller checks for that.
"""

import math
import os

import pandas as pd

from config import config
from data_pipeline.dataset_loader import load_processed_dataset, load_raw_dataset
from data_pipeline.order_cleaning import clean_orders_dataset


class OrderNotFound(Exception):
    """The named exception a stricter lookup would raise.

    The platform keeps the empty-dictionary contract everywhere instead: fetch_order
    returns {} for an unknown order and every caller refuses on it.
    """


def _without_answer(frame):
    if config.ESCALATION_TARGET_COLUMN in frame.columns:
        frame = frame.drop(columns=[config.ESCALATION_TARGET_COLUMN])
    return frame.reset_index(drop=True)


def load_live_orders(feed_path=None, serving_path=None, cleaned_path=None):
    """The order book being served: the serving store, else the cleaned feed, else the raw feed."""
    feed = config.LIVE_ORDERS_FEED_FILE if feed_path is None else feed_path
    serving = config.SERVING_STORE_FILE if serving_path is None else serving_path
    cleaned = config.CLEAN_LIVE_ORDERS_FILE if cleaned_path is None else cleaned_path
    for path in (serving, cleaned):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return _without_answer(load_processed_dataset(path))
    if os.path.exists(feed):
        raw = load_raw_dataset(feed, config.NULL_TOKEN_LIST)
        return _without_answer(clean_orders_dataset(raw, is_serving_feed=True))
    raise FileNotFoundError("No order book is available: neither " + serving + ", " + cleaned
                            + " nor " + feed + " exists. Run python3 run_data_preparation.py.")


def _plain(value):
    """A value with numpy and pandas types turned into plain Python."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _record_from(frame, column, key):
    if frame is None or column not in frame.columns or key is None:
        return {}
    wanted = str(key).strip().upper()
    matches = frame[frame[column].astype(str).str.strip().str.upper() == wanted]
    if matches.empty:
        return {}
    return {name: _plain(value) for name, value in matches.iloc[0].to_dict().items()}


def fetch_order(orders_frame, order_id):
    """One order as a plain dictionary, matched ignoring case and padding; {} when unknown."""
    return _record_from(orders_frame, "order_id", order_id)


def order_exists(orders_frame, order_id):
    """Whether a reference is on file, without raising."""
    return bool(fetch_order(orders_frame, order_id))


def days_late(order):
    """Whole days past the promised date; never negative."""
    slack = order.get("delivery_slack_days")
    try:
        slack = float(slack)
    except (TypeError, ValueError):
        return 0
    if math.isnan(slack) or slack >= 0:
        return 0
    return int(round(-slack))


def delivery_position_phrase(order):
    """The delivery position stated as a fact, never inferred."""
    try:
        slack = float(order.get("delivery_slack_days"))
    except (TypeError, ValueError):
        return "delivery position not on record"
    late = days_late(order)
    if late > 0:
        return "running {0} day{1} past the promised date".format(late, "" if late == 1 else "s")
    remaining = int(round(slack))
    if remaining == 0:
        return "due today"
    return "due in {0} day{1}".format(remaining, "" if remaining == 1 else "s")


def describe_order_for_person(order):
    """The summary a prompt and a chat reply are both built from."""
    if not order:
        return {}
    return {
        "order_id": order.get("order_id", ""),
        "customer_id": order.get("customer_id", ""),
        "product_category": order.get("product_category", ""),
        "item_count": int(float(order.get("item_count") or 0)),
        "order_value_inr": round(float(order.get("order_value_inr") or 0.0), 2),
        "courier": order.get("courier", ""),
        "customer_tier": order.get("customer_tier", ""),
        "payment_method": order.get("payment_method", ""),
        "warehouse": order.get("warehouse", ""),
        "promised_delivery_date": order.get("promised_delivery_date", ""),
        "delivery_position": delivery_position_phrase(order),
        "days_late": days_late(order),
        "hours_since_last_scan": round(float(order.get("transit_dwell_hours") or 0.0), 1),
    }


def load_customers(path=None):
    """The customer list, cleaned of its spelling variants."""
    from data_pipeline.common_cleaning import apply_category_mapping, normalise_missing

    frame = load_raw_dataset(config.RAW_CUSTOMERS_FILE if path is None else path,
                             config.NULL_TOKEN_LIST)
    frame = normalise_missing(frame, config.NULL_TOKEN_LIST)
    frame = apply_category_mapping(frame, "customer_tier", config.CUSTOMER_TIER_MAPPING)
    return apply_category_mapping(frame, "city_tier", config.CITY_TIER_MAPPING)


def load_products(path=None):
    """The product catalogue, cleaned of its spelling variants."""
    from data_pipeline.common_cleaning import apply_category_mapping, normalise_missing

    frame = load_raw_dataset(config.RAW_PRODUCTS_FILE if path is None else path,
                             config.NULL_TOKEN_LIST)
    frame = normalise_missing(frame, config.NULL_TOKEN_LIST)
    return apply_category_mapping(frame, "product_category", config.PRODUCT_CATEGORY_MAPPING)


def fetch_customer(customers_frame, customer_id):
    """One customer as a plain dictionary; {} when unknown."""
    return _record_from(customers_frame, "customer_id", customer_id)


def fetch_product(products_frame, sku):
    """One product as a plain dictionary; {} when unknown."""
    return _record_from(products_frame, "sku", sku)


def order_book_records(orders_frame):
    """Every order as a plain dictionary, for a response body."""
    return [{name: _plain(value) for name, value in row.items()}
            for row in orders_frame.to_dict(orient="records")]


def frame_from_records(records):
    return pd.DataFrame(records)
