"""Reads tables from disk and writes them back.

This is the only module in the data preparation section that touches CSV files.
Raw files are read as text with nothing guessed; cleaned files are read with
their types, because the cleaning modules already made them trustworthy.
"""

import os

import pandas as pd


def load_raw_dataset(file_path, null_token_list):
    """Read one raw CSV exactly as written: every column text, missing tokens empty."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Raw file not found: " + str(file_path)
            + ". The raw extracts live under data/; generate or copy them first.")
    return pd.read_csv(
        file_path,
        dtype=str,
        na_values=list(null_token_list),
        keep_default_na=True,
    )


def load_processed_dataset(file_path):
    """Read one cleaned table with its column types inferred from the contents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Cleaned file not found: " + str(file_path)
            + ". Run python3 run_data_preparation.py first.")
    # keep_default_na=False: the cleaned file holds no gaps, so a value such as the
    # text "None" (never written by cleaning) must not be turned back into a gap.
    return pd.read_csv(file_path, keep_default_na=False, na_values=[""])


def load_processed_dataset_if_present(file_path):
    """Read a cleaned table when it exists, or return None when it does not."""
    if not os.path.exists(file_path):
        return None
    return load_processed_dataset(file_path)


def save_dataset(frame, file_path):
    """Write a table to CSV, creating the folder when it is absent."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    frame.to_csv(file_path, index=False)
    return file_path
