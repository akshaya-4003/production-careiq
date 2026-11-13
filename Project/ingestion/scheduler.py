"""Runs the ingestion pipeline on a timer; a failed pass never stops the schedule."""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

from config import config
from ingestion.watermark import describe_watermark


def _report(outcome):
    print("[ingestion] {0}: read {1}, scored {2}, high risk {3}; {4}; {5}".format(
        outcome["status"], outcome["rows_read"], outcome["rows_scored"],
        outcome["high_risk_found"], describe_watermark(outcome["watermark_after"]),
        outcome["message"]))


def build_scheduler(pipeline, interval_minutes=None, blocking=False):
    minutes = config.INGESTION_INTERVAL_MINUTES if interval_minutes is None else interval_minutes
    scheduler = BlockingScheduler() if blocking else BackgroundScheduler()

    def tick():
        try:
            _report(pipeline.run_once())
        except Exception as error:  # a failed pass must not stop the schedule
            print("[ingestion] pass failed and the schedule continues: " + str(error))

    scheduler.add_job(tick, "interval", minutes=minutes, id="careiq-ingestion",
                      max_instances=1, coalesce=True)
    scheduler.tick = tick
    return scheduler
