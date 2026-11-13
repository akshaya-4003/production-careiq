"""Entry point 1: confirm the nine datasets and six policy documents are in place.

    python3 run_data_generation.py

The specification describes a generator that synthesises the Northlane extracts
from config.RANDOM_SEED. This repository ships the generated files themselves
under data/, so this entry point checks them instead: every file with its row
count, and every document with its section and paragraph counts. If anything is
missing it says which file, so the operator knows what to restore.
"""

import os
import sys

from config import config
from data_pipeline.dataset_loader import load_raw_dataset

DATASETS = [config.RAW_ORDERS_TRAIN_FILE, config.RAW_ORDERS_EVAL_FILE,
            config.RAW_TICKETS_TRAIN_FILE, config.RAW_TICKETS_EVAL_FILE,
            config.RAW_MESSAGES_FILE, config.RAW_ORDER_EVENTS_FILE, config.RAW_SHIPMENTS_FILE,
            config.RAW_CUSTOMERS_FILE, config.RAW_PRODUCTS_FILE, config.LIVE_ORDERS_FEED_FILE]


def main():
    missing = []
    print("Checking the Northlane datasets")
    for path in DATASETS:
        if not os.path.exists(path):
            missing.append(path)
            print("    {0:<45} MISSING".format(path))
            continue
        rows = len(load_raw_dataset(path, config.NULL_TOKEN_LIST))
        print("    {0:<45} {1:>7} rows".format(path, rows))

    print("Checking the policy documents the chatbot is restricted to")
    from knowledge.chunk_splitting import split_into_sections
    from knowledge.pdf_extraction import read_document_text

    for entry in config.POLICY_DOCUMENTS:
        path = os.path.join(config.DOCS_DIRECTORY, entry["document_id"] + ".pdf")
        if not os.path.exists(path):
            missing.append(path)
            print("    {0:<45} MISSING".format(path))
            continue
        sections = split_into_sections(read_document_text(path))
        paragraphs = sum(len([s for s in section["body"].split(". ") if s.strip()])
                         for section in sections)
        print("    {0:<45} {1:>3} sections {2:>3} sentences".format(path, len(sections),
                                                                    paragraphs))
    if missing:
        print("Missing " + str(len(missing)) + " file(s). Restore them under data/ first.")
        return 1
    print("Done. Next: python3 run_database_setup.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
