"""The ticket-specific cleaning sequence for the resolution time model."""

from config import config
from data_pipeline import common_cleaning as steps


def clean_tickets_dataset(frame):
    """Clean a ticket table: catalogues folded, numbers repaired, target required."""
    numeric_columns = list(config.RESOLUTION_NUMERIC_COLUMNS)
    target = config.RESOLUTION_TARGET_COLUMN

    cleaned = steps.normalise_missing(frame, config.NULL_TOKEN_LIST)
    cleaned = steps.parse_date_columns(cleaned, config.RESOLUTION_DATE_COLUMNS)
    cleaned = steps.repair_numeric_columns(cleaned, numeric_columns)
    for column, mapping in config.TICKET_CATEGORY_MAPPINGS.items():
        cleaned = steps.apply_category_mapping(cleaned, column, mapping,
                                               config.UNKNOWN_CATEGORY_VALUE)
    cleaned = steps.coerce_boolean_columns(cleaned, config.RESOLUTION_BOOLEAN_COLUMNS,
                                           config.BOOLEAN_TRUE_TOKENS)
    cleaned = steps.clip_impossible_values(cleaned, config.TICKET_RANGE_LIMITS)
    cleaned = steps.blank_extreme_outliers(cleaned, config.RESOLUTION_OUTLIER_COLUMNS,
                                           config.OUTLIER_IQR_MULTIPLIER)
    feature_numeric = [column for column in numeric_columns if column != target]
    cleaned = steps.fill_missing_numeric(cleaned, feature_numeric)
    text_columns = [column for column in cleaned.columns
                    if column not in numeric_columns
                    and column not in config.RESOLUTION_BOOLEAN_COLUMNS
                    and column not in config.RESOLUTION_DATE_COLUMNS]
    cleaned = steps.fill_missing_categorical(cleaned, text_columns,
                                             config.UNKNOWN_CATEGORY_VALUE)
    cleaned = steps.render_dates_as_text(cleaned, config.RESOLUTION_DATE_COLUMNS,
                                         config.UNKNOWN_CATEGORY_VALUE)
    cleaned = steps.drop_duplicate_rows(cleaned, ["ticket_id"])
    cleaned = steps.drop_rows_missing_target(cleaned, target)
    return cleaned
