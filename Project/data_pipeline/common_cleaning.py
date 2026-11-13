"""Repair steps shared by more than one dataset.

Each function fixes exactly one kind of problem and returns a new table; none
modifies the table it was handed.
"""

import math
import re
from datetime import datetime

import numpy as np
import pandas as pd

from config import config


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalise_missing(frame, null_token_list=None):
    """Blank every cell whose whole content is a missing token, whatever its padding or case."""
    tokens = config.NULL_TOKEN_LIST if null_token_list is None else null_token_list
    lowered = set(str(token).strip().lower() for token in tokens)
    result = frame.copy()
    for column in result.columns:
        series = result[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        as_text = series.astype("object")
        stripped = as_text.map(lambda value: value.strip().lower()
                               if isinstance(value, str) else value)
        mask = stripped.map(lambda value: isinstance(value, str) and value in lowered)
        if mask.any():
            as_text = as_text.where(~mask, np.nan)
        result[column] = as_text
    return result


_UNIT_PATTERN = re.compile(config.UNIT_SUFFIX_PATTERN, flags=re.IGNORECASE)


def repair_one_number(raw_value):
    """Read one value written as text with separators, a currency marker or a unit."""
    if _is_missing(raw_value):
        return float("nan")
    if isinstance(raw_value, (int, float, np.integer, np.floating)) and not isinstance(
            raw_value, bool):
        return float(raw_value)
    text = str(raw_value).strip()
    if text == "":
        return float("nan")
    is_percentage = "%" in text
    text = _UNIT_PATTERN.sub("", text)
    text = re.sub(config.THOUSANDS_AND_PERCENT_PATTERN, "", text).strip()
    try:
        number = float(text)
    except ValueError:
        return float("nan")
    if math.isinf(number):
        return float("nan")
    return number / 100.0 if is_percentage else number


def repair_numeric_columns(frame, columns):
    """Turn the named columns into numbers; unreadable values are left missing."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        result[column] = pd.to_numeric(result[column].map(repair_one_number),
                                       errors="coerce").astype(float)
    return result


def apply_category_mapping(frame, column, mapping, unknown_value=None):
    """Fold every spelling onto the platform's one spelling; unknown values become unknown."""
    unknown = config.UNKNOWN_CATEGORY_VALUE if unknown_value is None else unknown_value
    result = frame.copy()
    if column not in result.columns:
        return result
    lookup = {str(key).strip().lower(): value for key, value in mapping.items()}
    # A value already written in the platform spelling is recognised too.
    for value in mapping.values():
        lookup.setdefault(str(value).strip().lower(), value)

    def fold(value):
        if _is_missing(value):
            return unknown
        return lookup.get(str(value).strip().lower(), unknown)

    result[column] = result[column].map(fold).astype(object)
    return result


def coerce_boolean_columns(frame, columns, true_tokens=None):
    """Turn yes-or-no columns into 0 or 1; anything not a true token is 0."""
    tokens = config.BOOLEAN_TRUE_TOKENS if true_tokens is None else true_tokens
    truthy = set(str(token).strip().lower() for token in tokens)
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue

        def to_flag(value):
            if _is_missing(value):
                return 0
            if isinstance(value, (int, float, np.integer, np.floating)):
                text = str(int(value)) if float(value).is_integer() else str(value)
            else:
                text = str(value)
            return 1 if text.strip().lower() in truthy else 0

        result[column] = result[column].map(to_flag).astype(int)
    return result


def clip_impossible_values(frame, range_limits):
    """Blank values outside the permitted range; they are sentinels, not measurements."""
    result = frame.copy()
    for column, (lowest, highest) in range_limits.items():
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        outside = (values < lowest) | (values > highest)
        result[column] = values.where(~outside, np.nan)
    return result


def blank_extreme_outliers(frame, columns, multiplier=None):
    """Blank values beyond the quartiles by `multiplier` interquartile ranges, named columns only."""
    factor = config.OUTLIER_IQR_MULTIPLIER if multiplier is None else multiplier
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        first, third = values.quantile(0.25), values.quantile(0.75)
        spread = third - first
        if spread <= 0:
            continue
        lowest, highest = first - factor * spread, third + factor * spread
        extreme = (values < lowest) | (values > highest)
        result[column] = values.where(~extreme, np.nan)
    return result


def _parse_one_date(value, formats):
    if _is_missing(value):
        return pd.NaT
    text = str(value).strip()
    for date_format in formats:
        try:
            return pd.Timestamp(datetime.strptime(text, date_format))
        except ValueError:
            continue
    return pd.NaT


def parse_date_columns(frame, columns):
    """Read every date format the extracts use into one date type; non-dates stay missing."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        parsed = result[column].map(lambda value: _parse_one_date(value, config.DATE_FORMATS))
        result[column] = pd.to_datetime(parsed, errors="coerce")
    return result


def drop_duplicate_rows(frame, subset=None):
    """Remove repeated rows, keeping the first occurrence."""
    if subset is not None:
        subset = [column for column in subset if column in frame.columns]
    result = frame.drop_duplicates(subset=subset or None, keep="first")
    return result.reset_index(drop=True)


def fill_missing_numeric(frame, columns):
    """Fill numeric gaps with the column median, or zero for an entirely empty column."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce").astype(float)
        median = values.median()
        result[column] = values.fillna(0.0 if pd.isna(median) else float(median))
    return result


def fill_missing_categorical(frame, columns, unknown_value=None):
    """Fill categorical gaps with the explicit unknown value, never the commonest value."""
    unknown = config.UNKNOWN_CATEGORY_VALUE if unknown_value is None else unknown_value
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        result[column] = result[column].astype(object).where(result[column].notna(), unknown)
    return result


def drop_rows_missing_target(frame, target_column):
    """Remove rows with no answer. Never run on the serving feed."""
    if target_column not in frame.columns:
        raise KeyError("Target column " + target_column + " is not in the table.")
    return frame[frame[target_column].notna()].reset_index(drop=True)


def render_dates_as_text(frame, columns, unknown_value=None):
    """Write parsed dates back as ISO text so the cleaned file has no gaps.

    The date columns are identifiers in this project, never model inputs, so an
    unreadable date is recorded as the explicit unknown value rather than guessed.
    """
    unknown = config.UNKNOWN_CATEGORY_VALUE if unknown_value is None else unknown_value
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        parsed = pd.to_datetime(result[column], errors="coerce")
        has_time = bool(((parsed.dt.hour != 0) | (parsed.dt.minute != 0)
                         | (parsed.dt.second != 0)).any())
        pattern = "%Y-%m-%dT%H:%M:%S" if has_time else "%Y-%m-%d"
        text = parsed.dt.strftime(pattern).astype(object)
        result[column] = text.where(parsed.notna(), unknown)
    return result
