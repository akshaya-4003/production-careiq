"""Turns a cleaned table into the matrix a model is fitted on.

The two most expensive tabular mistakes are prevented here: giving the model the
answer, and giving it an identifier.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_model_features(frame, excluded_columns):
    """Drop the identifier columns that are present; nothing else is removed."""
    present = [column for column in excluded_columns if column in frame.columns]
    return frame.drop(columns=present)


def split_features_and_target(frame, target_column):
    """Separate the answer from the questions. A table with no target raises."""
    if target_column not in frame.columns:
        raise KeyError("The table has no target column " + repr(target_column)
                       + "; features cannot be split from an answer that is not there.")
    features = frame.drop(columns=[target_column])
    target = frame[target_column]
    return features, target


def identify_column_types(frame):
    """Split columns into numeric and categorical, each column exactly once."""
    numeric_columns, categorical_columns = [], []
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]) and \
                not pd.api.types.is_bool_dtype(frame[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def build_preprocessing_pipeline(numeric_columns, categorical_columns):
    """Scale numeric columns and one-hot encode categorical ones; unseen categories are ignored."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric_columns)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical_columns)),
        ],
        remainder="drop",
    )


def prepare_for_modelling(frame, identifier_columns, target_column):
    """Drop identifiers, separate the target and split column types, in one call."""
    selected = select_model_features(frame, identifier_columns)
    features, target = split_features_and_target(selected, target_column)
    numeric_columns, categorical_columns = identify_column_types(features)
    for column in categorical_columns:
        features[column] = features[column].astype(str)
    return features, target, numeric_columns, categorical_columns
