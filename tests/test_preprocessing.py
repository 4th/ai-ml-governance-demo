# tests/test_preprocessing.py
from __future__ import annotations

import pandas as pd

from src.ml.preprocessing import basic_clean


def test_basic_clean_drops_all_na_rows():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 5]})
    cleaned = basic_clean(df)
    assert cleaned.shape[0] == 2
