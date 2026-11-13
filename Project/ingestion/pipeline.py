"""Read what is new, clean it as training data was cleaned, score it, write the serving store.

The watermark is advanced only after the write succeeded; advancing it first is
the bug that loses a batch permanently and silently.
"""

import os

import pandas as pd

from config import config
from data_pipeline.dataset_loader import load_processed_dataset, save_dataset
from data_pipeline.order_cleaning import clean_orders_dataset
from ingestion.source_reader import load_source_reader
from ingestion.watermark import read_watermark, write_watermark
from modeling.model_registry import load_feature_schema, load_model_artifact
from persistence import repository
from persistence.database import open_database_session
from tools.prediction_tools import score_order_book


class IngestionPipeline(object):
    def __init__(self, reader=None, serving_path=None, watermark_path=None, record_run=True,
                 risk_model=None, feature_names=None):
        self.reader = load_source_reader() if reader is None else reader
        self.serving_path = config.SERVING_STORE_FILE if serving_path is None else serving_path
        self.watermark_path = config.WATERMARK_FILE if watermark_path is None else watermark_path
        self.record_run = record_run
        self._risk_model = risk_model
        self._feature_names = feature_names

    def _model(self):
        if self._risk_model is None:
            self._risk_model = load_model_artifact(config.ESCALATION_MODEL_FILE)
            self._feature_names = load_feature_schema(config.ESCALATION_SCHEMA_FILE)
        return self._risk_model, self._feature_names

    @staticmethod
    def guard_against_the_answer(frame):
        """Refuse a batch that carries the outcome column; it is the evaluation file, not a feed."""
        if config.ESCALATION_TARGET_COLUMN in frame.columns:
            raise ValueError("The batch carries " + config.ESCALATION_TARGET_COLUMN
                             + "; that is the evaluation file, not the live feed.")
        return frame

    @staticmethod
    def upsert(existing, incoming):
        """Merge a batch into the store by order reference, the newer row winning."""
        if existing is None or existing.empty:
            return incoming.reset_index(drop=True)
        combined = pd.concat([existing, incoming], ignore_index=True)
        return combined.drop_duplicates(subset=["order_id"], keep="last").reset_index(drop=True)

    def _existing(self):
        # The first pass merges onto the cleaned feed, so the served book never
        # shrinks to one batch while the replay is still on its first day.
        for path in (self.serving_path, config.CLEAN_LIVE_ORDERS_FILE):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return load_processed_dataset(path)
        return None

    def run_once(self, limit=None):
        batch_limit = config.INGESTION_BATCH_LIMIT if limit is None else limit
        before = read_watermark(self.watermark_path)
        outcome = {"source": self.reader.name, "rows_read": 0, "rows_scored": 0,
                   "high_risk_found": 0, "watermark_before": before, "watermark_after": before,
                   "status": config.INGESTION_STATUS_FAILED, "message": ""}
        logged = None
        if self.record_run:
            session = open_database_session()
            try:
                logged = repository.start_ingestion_run(session, self.reader.name, before)
            finally:
                session.close()
        try:
            raw, cursor, generation = self.reader.read_since(before, batch_limit)
            outcome["rows_read"] = int(len(raw))
            if raw.empty:
                outcome.update({"status": config.INGESTION_STATUS_NO_NEW_ROWS,
                                "message": "nothing new to read"})
            else:
                cleaned = clean_orders_dataset(self.guard_against_the_answer(raw),
                                               is_serving_feed=True)
                model, features = self._model()
                scored = score_order_book(model, features, cleaned)
                save_dataset(self.upsert(self._existing(), cleaned), self.serving_path)
                # Only now, with the write done, may the watermark move.
                after = write_watermark({"cursor": cursor, "generation": generation},
                                        self.watermark_path)
                outcome.update({
                    "rows_scored": int(len(scored)),
                    "high_risk_found": int((scored["risk_band"] == config.RISK_BAND_HIGH).sum()),
                    "watermark_after": after, "status": config.INGESTION_STATUS_SUCCESS,
                    "message": "{0} rows merged into the serving store".format(len(cleaned))})
        except Exception as error:
            outcome.update({"status": config.INGESTION_STATUS_FAILED,
                            "message": type(error).__name__ + ": " + str(error),
                            "watermark_after": before})
        if logged is not None:
            session = open_database_session()
            try:
                repository.finish_ingestion_run(session, logged["id"], outcome)
            finally:
                session.close()
        return outcome
