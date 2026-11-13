"""Entry point 8: keep the order book fresh on a timer.

    python3 run_ingestion_service.py            every config.INGESTION_INTERVAL_MINUTES
    python3 run_ingestion_service.py --once     one pass, then stop
    python3 run_ingestion_service.py --reset    start the replay again from the beginning
    python3 run_ingestion_service.py --fast 1   every minute, for a demonstration

Ingestion changes the serving file. The dashboard moves when the service
re-scores (at start up, or with the Re-score now button / POST /pipeline/rescore).
"""

import argparse
import sys

from config import config
from ingestion.pipeline import IngestionPipeline
from ingestion.scheduler import _report, build_scheduler
from ingestion.watermark import describe_watermark, read_watermark, reset_watermark


def main():
    parser = argparse.ArgumentParser(description="CareIQ ingestion service")
    parser.add_argument("--once", action="store_true", help="run one pass and stop")
    parser.add_argument("--reset", action="store_true", help="reset the watermark first")
    parser.add_argument("--fast", type=float, default=None, help="interval in minutes")
    arguments = parser.parse_args()

    if arguments.reset:
        print("Watermark reset: " + describe_watermark(reset_watermark()))
    pipeline = IngestionPipeline()
    print("Ingestion source : " + str(pipeline.reader.describe()))
    print("Watermark        : " + describe_watermark(read_watermark()))
    if arguments.once:
        outcome = pipeline.run_once()
        _report(outcome)
        return 0 if outcome["status"] != config.INGESTION_STATUS_FAILED else 1

    minutes = arguments.fast if arguments.fast else config.INGESTION_INTERVAL_MINUTES
    scheduler = build_scheduler(pipeline, minutes, blocking=True)
    print("Running every {0} minute(s). Ctrl+C to stop.".format(minutes))
    scheduler.tick()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Ingestion stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
