from datetime import date

from gridcast import config


def test_settings_load_and_are_consistent():
    s = config.get_settings()
    assert s.region == "VIC1"
    assert s.steps_per_day == 48
    bt = s.backtest
    # The scored window must be exactly the next calendar day after a 12:00 origin.
    assert bt.horizon_steps - bt.scored_from_step + 1 == s.steps_per_day
    assert (24 - bt.origin_hour) * 2 == bt.scored_from_step - 1
    # Headline evaluation only uses forecast-as-known weather.
    assert bt.eval_start >= s.weather.historical_forecast_start
    assert s.data.history_start < bt.eval_start < bt.eval_end < s.data.history_end
    assert s.data.five_min_settlement_start == date(2021, 10, 1)
