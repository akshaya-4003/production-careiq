"""The ingestion watermark: a cursor, a generation, and when it last ran."""

import json
import os
from datetime import datetime, timezone

from config import config


def _fresh():
    return {"cursor": 0, "generation": 0, "last_run_at": ""}


def read_watermark(path=None):
    """The stored watermark, or a fresh start when the file is missing or unreadable."""
    target = config.WATERMARK_FILE if path is None else path
    if not os.path.exists(target):
        return _fresh()
    try:
        with open(target, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except (ValueError, OSError):
        return _fresh()
    watermark = _fresh()
    watermark.update({key: stored[key] for key in watermark if key in stored})
    return watermark


def write_watermark(watermark, path=None):
    target = config.WATERMARK_FILE if path is None else path
    folder = os.path.dirname(target)
    if folder:
        os.makedirs(folder, exist_ok=True)
    stored = dict(watermark)
    stored["last_run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(stored, handle, indent=2)
    return stored


def reset_watermark(path=None):
    return write_watermark(_fresh(), path)


def describe_watermark(watermark):
    return "cursor {0}, generation {1} (day {2} of the replay), last run {3}".format(
        watermark.get("cursor", 0), watermark.get("generation", 0),
        int(watermark.get("generation", 0)) + 1, watermark.get("last_run_at") or "never")
