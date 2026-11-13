"""Measures how dirty a raw table was, so that cleaning is evidence rather than assertion.

Every figure is a count of cells or of rows, so the same file profiled twice
gives the same answer. Nothing here repairs anything.
"""

import json
import os
import re

import pandas as pd

from data_pipeline.common_cleaning import repair_one_number

_PLAIN_NUMBER = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")


def build_quality_report(frame, dataset_name, numeric_columns=None, range_limits=None,
                         category_mappings=None):
    """Return one dictionary of dirt counts for a raw table."""
    rows, columns = int(frame.shape[0]), int(frame.shape[1])
    total_cells = rows * columns
    missing_cells = int(frame.isna().sum().sum())

    needs_repair, beyond_repair, out_of_range, unmapped = {}, {}, {}, {}

    for column in (numeric_columns or []):
        if column not in frame.columns:
            continue
        present = frame[column].dropna().astype(str)
        not_plain = present[~present.map(lambda value: bool(_PLAIN_NUMBER.match(value)))]
        repaired = not_plain.map(repair_one_number)
        repairable = int(repaired.notna().sum())
        unrepairable = int(repaired.isna().sum())
        if repairable:
            needs_repair[column] = repairable
        if unrepairable:
            beyond_repair[column] = unrepairable

    for column, (lowest, highest) in (range_limits or {}).items():
        if column not in frame.columns:
            continue
        values = frame[column].map(repair_one_number)
        count = int(((values < lowest) | (values > highest)).sum())
        if count:
            out_of_range[column] = count

    for column, mapping in (category_mappings or {}).items():
        if column not in frame.columns:
            continue
        known = set(str(key).strip().lower() for key in mapping)
        known |= set(str(value).strip().lower() for value in mapping.values())
        present = frame[column].dropna().astype(str).str.strip().str.lower()
        count = int((~present.isin(known)).sum())
        if count:
            unmapped[column] = count

    return {
        "dataset": dataset_name,
        "rows": rows,
        "columns": columns,
        "total_cells": total_cells,
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": missing_cells,
        "missing_share": round(missing_cells / total_cells, 4) if total_cells else 0.0,
        "values_needing_repair": needs_repair,
        "values_beyond_repair": beyond_repair,
        "values_out_of_range": out_of_range,
        "unmapped_category_values": unmapped,
        "total_needing_repair": int(sum(needs_repair.values())),
        "total_out_of_range": int(sum(out_of_range.values())),
    }


def save_quality_report(reports, output_path):
    """Write every report under one top-level key and return the path."""
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"datasets": list(reports)}, handle, indent=2)
    return output_path


def summarise_report(report):
    """One aligned console line for a report."""
    return "    {0:<20} {1:>7} rows  {2:>7} missing ({3:>6.2%})  {4:>5} dupes  {5:>6} need repair  {6:>5} out of range".format(
        report["dataset"], report["rows"], report["missing_cells"], report["missing_share"],
        report["duplicate_rows"], report["total_needing_repair"], report["total_out_of_range"])
