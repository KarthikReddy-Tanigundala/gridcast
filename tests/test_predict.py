import pandas as pd

from gridcast.predict import choose_origin


def test_origin_is_noon_once_noon_is_observed():
    now = pd.Timestamp("2026-10-20 12:35")
    assert choose_origin(pd.Timestamp("2026-10-20 12:30"), now, 12) == pd.Timestamp(
        "2026-10-20 12:00"
    )


def test_origin_falls_back_to_latest_actual():
    now = pd.Timestamp("2026-10-20 09:00")
    assert choose_origin(pd.Timestamp("2026-10-20 00:00"), now, 12) == pd.Timestamp(
        "2026-10-20 00:00"
    )
