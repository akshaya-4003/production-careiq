"""Entry point 3: profile the dirt in the raw extracts, then clean them.

    python3 run_data_preparation.py
"""

import os
import sys

from config import config
from data_pipeline.dataset_loader import load_raw_dataset, save_dataset
from data_pipeline.message_cleaning import clean_message_frame
from data_pipeline.order_cleaning import clean_orders_dataset
from data_pipeline.quality_profiler import build_quality_report, save_quality_report, summarise_report
from data_pipeline.ticket_cleaning import clean_tickets_dataset

ORDER_NUMERIC = [c for c in config.ESCALATION_NUMERIC_COLUMNS]
TICKET_NUMERIC = list(config.RESOLUTION_NUMERIC_COLUMNS)

# name, raw file, numeric columns, range limits, category mappings
PROFILES = [
    ("orders_train", config.RAW_ORDERS_TRAIN_FILE, ORDER_NUMERIC, config.ORDER_RANGE_LIMITS,
     config.ORDER_CATEGORY_MAPPINGS),
    ("orders_eval", config.RAW_ORDERS_EVAL_FILE, ORDER_NUMERIC, config.ORDER_RANGE_LIMITS,
     config.ORDER_CATEGORY_MAPPINGS),
    ("tickets_train", config.RAW_TICKETS_TRAIN_FILE, TICKET_NUMERIC, config.TICKET_RANGE_LIMITS,
     config.TICKET_CATEGORY_MAPPINGS),
    ("tickets_eval", config.RAW_TICKETS_EVAL_FILE, TICKET_NUMERIC, config.TICKET_RANGE_LIMITS,
     config.TICKET_CATEGORY_MAPPINGS),
    ("messages", config.RAW_MESSAGES_FILE, [], {},
     {"issue_mode": config.ISSUE_MODE_MAPPING, "channel": config.CHANNEL_MAPPING}),
    ("order_events", config.RAW_ORDER_EVENTS_FILE, [], {}, {}),
    ("shipments", config.RAW_SHIPMENTS_FILE, ["route_failure_rate_30d"], {},
     {"courier": config.COURIER_MAPPING}),
    ("customers", config.RAW_CUSTOMERS_FILE, ["prior_orders", "prior_tickets", "prior_returns"],
     {}, {"customer_tier": config.CUSTOMER_TIER_MAPPING, "city_tier": config.CITY_TIER_MAPPING}),
    ("products", config.RAW_PRODUCTS_FILE, ["unit_price_inr", "return_rate_30d", "warranty_months"],
     {}, {"product_category": config.PRODUCT_CATEGORY_MAPPING}),
    ("live_orders_feed", config.LIVE_ORDERS_FEED_FILE,
     [c for c in ORDER_NUMERIC if c != config.ESCALATION_TARGET_COLUMN],
     config.ORDER_RANGE_LIMITS, config.ORDER_CATEGORY_MAPPINGS),
]

# name, raw file, cleaner, cleaned file
CLEANING = [
    ("orders_train", config.RAW_ORDERS_TRAIN_FILE, clean_orders_dataset,
     config.CLEAN_ORDERS_TRAIN_FILE),
    ("orders_eval", config.RAW_ORDERS_EVAL_FILE, clean_orders_dataset,
     config.CLEAN_ORDERS_EVAL_FILE),
    ("tickets_train", config.RAW_TICKETS_TRAIN_FILE, clean_tickets_dataset,
     config.CLEAN_TICKETS_TRAIN_FILE),
    ("tickets_eval", config.RAW_TICKETS_EVAL_FILE, clean_tickets_dataset,
     config.CLEAN_TICKETS_EVAL_FILE),
    ("messages", config.RAW_MESSAGES_FILE, clean_message_frame, config.CLEAN_MESSAGES_FILE),
    ("live_orders_feed", config.LIVE_ORDERS_FEED_FILE,
     lambda frame: clean_orders_dataset(frame, is_serving_feed=True),
     config.CLEAN_LIVE_ORDERS_FILE),
]


def main():
    missing = [path for _, path, _, _, _ in PROFILES if not os.path.exists(path)]
    if missing:
        print("These raw files are missing: " + ", ".join(missing))
        print("Place the raw extracts under data/ first (see README_SAMPLE_DATA.md).")
        return 1

    print("Profiling the raw extracts")
    reports = []
    for name, path, numeric, ranges, mappings in PROFILES:
        raw = load_raw_dataset(path, config.NULL_TOKEN_LIST)
        report = build_quality_report(raw, name, numeric_columns=numeric,
                                      range_limits=ranges, category_mappings=mappings)
        reports.append(report)
        print(summarise_report(report))
    written = save_quality_report(reports, config.DATA_QUALITY_REPORT_FILE)
    print("    quality report written: " + written)

    print("Cleaning")
    cleaned_tables = {}
    for name, path, cleaner, output_path in CLEANING:
        cleaned = cleaner(load_raw_dataset(path, config.NULL_TOKEN_LIST))
        save_dataset(cleaned, output_path)
        cleaned_tables[name] = cleaned
        print("    {0:<18} {1}".format(name, str(cleaned.shape)))

    orders = cleaned_tables["orders_train"]
    tickets = cleaned_tables["tickets_train"]
    messages = cleaned_tables["messages"]
    gaps = sum(int(table.isna().sum().sum()) for table in cleaned_tables.values())
    print("Escalation rate in training data : {0:.4f}".format(
        orders[config.ESCALATION_TARGET_COLUMN].mean()))
    print("Median resolution hours          : {0:.2f}".format(
        tickets[config.RESOLUTION_TARGET_COLUMN].median()))
    print("Issue modes in the corpus        : {0}".format(
        messages[config.MESSAGE_TARGET_COLUMN].nunique()))
    print("Gaps left anywhere               : {0}".format(gaps))
    if gaps:
        print("Cleaning left gaps; a model cannot be fitted on them.")
        return 1
    print("Done. Next: python3 run_model_training.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
