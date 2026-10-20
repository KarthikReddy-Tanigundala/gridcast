"""Metric correctness against hand-computed values."""

import numpy as np
import pytest

from gridcast.evaluation import metrics as m


def test_mae_rmse():
    y, f = [1, 2, 3, 4], [2, 2, 2, 2]
    assert m.mae(y, f) == pytest.approx(1.0)
    assert m.rmse(y, f) == pytest.approx(np.sqrt(1.5))


def test_seasonal_naive_scale_and_mase():
    hist = [10, 20, 12, 22, 14]  # season 2: diffs |12-10|,|22-20|,|14-12| = 2,2,2
    scale = m.seasonal_naive_scale(hist, season=2)
    assert scale == pytest.approx(2.0)
    assert m.mase([5, 5], [6, 4], scale) == pytest.approx(0.5)


def test_mase_rejects_zero_scale():
    with pytest.raises(ValueError):
        m.mase([1], [1], 0.0)


def test_pinball_hand_computed():
    # tau=0.9: under-forecast by 10 costs 9, over-forecast by 10 costs 1.
    assert m.pinball([10], [0], 0.9) == pytest.approx(9.0)
    assert m.pinball([0], [10], 0.9) == pytest.approx(1.0)
    # Median pinball is half the absolute error.
    assert m.pinball([1, 2, 3], [2, 2, 2], 0.5) == pytest.approx(m.mae([1, 2, 3], [2, 2, 2]) / 2)


def test_coverage_and_width():
    y = [1, 5, 10, -3]
    assert m.coverage(y, [0, 0, 0, 0], [6, 6, 6, 6]) == pytest.approx(0.5)
    assert m.interval_width([0, 2], [4, 4]) == pytest.approx(3.0)


def test_smape_bounds():
    assert m.smape([100], [100]) == 0.0
    assert m.smape([100], [0]) == pytest.approx(200.0)


def test_interval_pinball_uses_tail_quantiles():
    # 80% interval -> quantiles 0.1 / 0.9
    expected = 0.5 * (m.pinball([5], [0], 0.1) + m.pinball([5], [4], 0.9))
    assert m.interval_pinball([5], [0], [4], 80) == pytest.approx(expected)
