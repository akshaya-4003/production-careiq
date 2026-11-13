"""Where new order rows come from. Everything above this module asks only for rows newer than a watermark."""

from config import config
from data_pipeline.common_cleaning import repair_one_number
from data_pipeline.dataset_loader import load_raw_dataset


class SourceReader(object):
    """The interface: read_since(watermark, limit) -> (rows, new_cursor, new_generation)."""

    name = "base"

    def read_since(self, watermark, limit):
        raise NotImplementedError

    def describe(self):
        return {"source": self.name}


class LocalCsvReader(SourceReader):
    """Reveals the bundled feed a batch at a time, then keeps ageing what it already sent.

    Each full pass through the feed is one day: parcels that have not moved dwell
    longer and the promised date comes closer, so risk genuinely climbs between
    refreshes.
    """

    name = "local_csv"

    def __init__(self, feed_path=None):
        self.feed_path = config.LIVE_ORDERS_FEED_FILE if feed_path is None else feed_path

    def _age(self, frame, generation):
        if generation <= 0:
            return frame
        aged = frame.copy()
        for column, change in config.INGESTION_AGEING_PER_GENERATION.items():
            if column in aged.columns:
                aged[column] = [
                    "" if value != value else str(round(repair_one_number(value)
                                                        + change * generation, 4))
                    for value in aged[column].map(repair_one_number)]
        return aged

    def read_since(self, watermark, limit):
        feed = load_raw_dataset(self.feed_path, config.NULL_TOKEN_LIST)
        cursor = int(watermark.get("cursor", 0))
        generation = int(watermark.get("generation", 0))
        if cursor >= len(feed):
            cursor, generation = 0, generation + 1
        batch = feed.iloc[cursor:cursor + int(limit)].reset_index(drop=True)
        return self._age(batch, generation), cursor + len(batch), generation

    def describe(self):
        return {"source": self.name, "feed": self.feed_path,
                "batch_limit": config.INGESTION_BATCH_LIMIT}


class SnowflakeReader(SourceReader):
    """A documented outline of the same shape for a warehouse deployment.

    A real implementation would run
        SELECT * FROM LIVE_ORDERS WHERE UPDATED_AT > :cursor ORDER BY UPDATED_AT LIMIT :limit
    with snowflake-connector-python, returning the rows, the newest UPDATED_AT as
    the new cursor, and the unchanged generation. Swapping the source is then one
    line in .env (CAREIQ_INGESTION_SOURCE=snowflake), with nothing above this module
    changing.
    """

    name = "snowflake"

    def read_since(self, watermark, limit):
        raise NotImplementedError("The Snowflake reader is an outline; set "
                                  "CAREIQ_INGESTION_SOURCE=local_csv to run the bundled feed.")


def load_source_reader(source=None):
    """The reader config.INGESTION_SOURCE names."""
    choice = (config.INGESTION_SOURCE if source is None else source).strip().lower()
    readers = {LocalCsvReader.name: LocalCsvReader, SnowflakeReader.name: SnowflakeReader}
    if choice not in readers:
        raise ValueError("Unknown ingestion source " + repr(choice) + "; use one of "
                         + str(sorted(readers)))
    return readers[choice]()
