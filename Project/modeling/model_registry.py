"""Saves and loads the chosen models and the ordered column list each was fitted on."""

import json
import os

import joblib
import pandas as pd

from config import config

# Columns a model reads as numbers. Anything else is categorical. Derived from the
# configured column lists, so a neutral fill value is never guessed.
NUMERIC_FEATURE_NAMES = set(
    config.ESCALATION_NUMERIC_COLUMNS + config.ESCALATION_BOOLEAN_COLUMNS
    + config.RESOLUTION_NUMERIC_COLUMNS + config.RESOLUTION_BOOLEAN_COLUMNS)


def _ensure_folder(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def save_model_artifact(model, output_path):
    """Write the whole fitted object, preprocessing included."""
    _ensure_folder(output_path)
    joblib.dump(model, output_path)
    return output_path


def load_model_artifact(model_path):
    """Load a fitted model; a missing file raises naming the path and the step to run."""
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        raise FileNotFoundError(
            "Model file not found: " + str(model_path)
            + ". Run python3 run_model_training.py to produce it.")
    return joblib.load(model_path)


def save_feature_schema(feature_names, output_path):
    """Write the column list as readable JSON, in the order given."""
    _ensure_folder(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"features": list(feature_names)}, handle, indent=2)
    return output_path


def load_feature_schema(schema_path):
    """Read the column list back in the saved order."""
    if not os.path.exists(schema_path) or os.path.getsize(schema_path) == 0:
        raise FileNotFoundError(
            "Feature schema not found: " + str(schema_path)
            + ". Run python3 run_model_training.py to produce it.")
    with open(schema_path, "r", encoding="utf-8") as handle:
        stored = json.load(handle)
    names = stored["features"] if isinstance(stored, dict) else stored
    return [str(name) for name in names]


def neutral_value_for(feature_name):
    """The value written for a field a record does not carry."""
    if feature_name in NUMERIC_FEATURE_NAMES:
        return 0.0
    return config.UNKNOWN_CATEGORY_VALUE


def _coerce(feature_name, value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return neutral_value_for(feature_name)
    if feature_name in NUMERIC_FEATURE_NAMES:
        try:
            return float(value)
        except (TypeError, ValueError):
            return neutral_value_for(feature_name)
    return str(value)


def build_model_input_row(record, feature_names):
    """Exactly one row, in the schema's columns and order, missing fields made neutral."""
    row = {name: _coerce(name, record.get(name)) for name in feature_names}
    return pd.DataFrame([row], columns=list(feature_names))


def build_model_input_frame(frame, feature_names):
    """Many rows in the schema's columns and order, row order preserved."""
    built = pd.DataFrame(index=frame.index)
    for name in feature_names:
        if name in frame.columns:
            built[name] = [_coerce(name, value) for value in frame[name]]
        else:
            built[name] = neutral_value_for(name)
    return built[list(feature_names)].reset_index(drop=True)
