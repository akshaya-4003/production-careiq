"""The order-specific cleaning sequence. No repair logic lives here, only the order of steps."""

from config import config
from data_pipeline import common_cleaning as steps


def clean_orders_dataset(frame, is_serving_feed=False):
    """Clean an order table for training or for serving, with identical repairs.

    The sequence is fixed: normalise missing, parse dates, repair numbers, map
    categories, coerce booleans, clip impossible values, blank outliers in the
    chosen columns only, fill numeric gaps, fill categorical gaps, drop duplicates
    on order_id, and finally drop rows with no target (never for the serving feed).
    """
    numeric_columns = list(config.ESCALATION_NUMERIC_COLUMNS)
    if is_serving_feed:
        numeric_columns = [column for column in numeric_columns
                           if column != config.ESCALATION_TARGET_COLUMN]

    cleaned = steps.normalise_missing(frame, config.NULL_TOKEN_LIST)
    cleaned = steps.parse_date_columns(cleaned, config.ESCALATION_DATE_COLUMNS)
    cleaned = steps.repair_numeric_columns(cleaned, numeric_columns)
    for column, mapping in config.ORDER_CATEGORY_MAPPINGS.items():
        cleaned = steps.apply_category_mapping(cleaned, column, mapping,
                                               config.UNKNOWN_CATEGORY_VALUE)
    cleaned = steps.coerce_boolean_columns(cleaned, config.ESCALATION_BOOLEAN_COLUMNS,
                                           config.BOOLEAN_TRUE_TOKENS)
    cleaned = steps.clip_impossible_values(cleaned, config.ORDER_RANGE_LIMITS)
    outlier_columns = [column for column in config.ESCALATION_OUTLIER_COLUMNS
                       if column != "transit_dwell_hours"]
    cleaned = steps.blank_extreme_outliers(cleaned, outlier_columns,
                                           config.OUTLIER_IQR_MULTIPLIER)
    feature_numeric = [column for column in numeric_columns
                       if column != config.ESCALATION_TARGET_COLUMN]
    cleaned = steps.fill_missing_numeric(cleaned, feature_numeric)
    text_columns = [column for column in cleaned.columns
                    if column not in numeric_columns
                    and column not in config.ESCALATION_BOOLEAN_COLUMNS
                    and column not in config.ESCALATION_DATE_COLUMNS]
    cleaned = steps.fill_missing_categorical(cleaned, text_columns,
                                             config.UNKNOWN_CATEGORY_VALUE)
    cleaned = steps.render_dates_as_text(cleaned, config.ESCALATION_DATE_COLUMNS,
                                         config.UNKNOWN_CATEGORY_VALUE)
    cleaned = steps.drop_duplicate_rows(cleaned, ["order_id"])

    if not is_serving_feed:
        cleaned = steps.drop_rows_missing_target(cleaned, config.ESCALATION_TARGET_COLUMN)
        cleaned[config.ESCALATION_TARGET_COLUMN] = cleaned[
            config.ESCALATION_TARGET_COLUMN].astype(int)
    elif config.ESCALATION_TARGET_COLUMN in cleaned.columns:
        # The serving feed must never be handed an answer column.
        cleaned = cleaned.drop(columns=[config.ESCALATION_TARGET_COLUMN])
    return cleaned
