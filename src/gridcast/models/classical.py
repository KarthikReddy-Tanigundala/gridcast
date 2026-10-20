"""Classical statistical models via ``statsforecast`` (§8.1 step 2).

* ``AutoETS``      — exponential smoothing with daily seasonality (48).
* ``MSTL-ETS``     — MSTL decomposition (daily 48 + weekly 336), AutoETS on the deseasonalised series.
* ``MSTL-ARIMA``   — same decomposition, AutoARIMA on the deseasonalised series.
* ``MSTL-ARIMAX``  — the SARIMAX equivalent: AutoARIMA with temperature / HDD / CDD regressors on the
                     deseasonalised series.

Why not a plain seasonal AutoARIMA (season 48)? Measured at ~175 s per fit on 56 days of
half-hourly data (vs ~2 s for MSTL-ARIMAX): seasonal differencing at lag 48 makes the order search
explode. Deseasonalising first with MSTL is the standard treatment for multi-seasonal load (errata §16).

Each backtest origin becomes its own series (``unique_id = origin``) holding only its trailing
training window, so many origins fit in one parallel ``StatsForecast.forecast`` call and no window
can see data after its own origin.
"""

from __future__ import annotations

import logging

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, AutoETS

logger = logging.getLogger(__name__)

EXOG = ["temperature", "hdd", "cdd"]
#: Kept for callers that pass the exogenous column list explicitly.
SARIMAX_EXOG = EXOG


def models(season: int = 48) -> dict[str, list]:
    seasons = [season, 7 * season]
    return {
        "univariate": [
            AutoETS(season_length=season, alias="AutoETS"),
            MSTL(season_length=seasons, trend_forecaster=AutoETS(model="ZZN"), alias="MSTL-ETS"),
            MSTL(
                season_length=seasons,
                trend_forecaster=AutoARIMA(max_p=2, max_q=2),
                alias="MSTL-ARIMA",
            ),
        ],
        "exog": [
            MSTL(
                season_length=seasons,
                trend_forecaster=AutoARIMA(max_p=2, max_q=2),
                alias="MSTL-ARIMAX",
            )
        ],
    }


def forecast_windows(
    train: pd.DataFrame, future_x: pd.DataFrame, horizon: int, season: int = 48, n_jobs: int = -1
) -> pd.DataFrame:
    """Forecast every window in ``train`` (one ``unique_id`` per origin).

    ``train``: ``unique_id, ds, y`` + ``EXOG``. ``future_x``: ``unique_id, ds`` + ``EXOG`` for the
    ``horizon`` steps after each window. Returns ``unique_id, ds`` + one column per model.
    """
    ms = models(season)
    uni = StatsForecast(models=ms["univariate"], freq="30min", n_jobs=n_jobs)
    out = uni.forecast(df=train[["unique_id", "ds", "y"]], h=horizon)
    ex = StatsForecast(models=ms["exog"], freq="30min", n_jobs=n_jobs)
    out_x = ex.forecast(
        df=train[["unique_id", "ds", "y", *EXOG]],
        h=horizon,
        X_df=future_x[["unique_id", "ds", *EXOG]],
    )
    out = out.reset_index() if "unique_id" not in out.columns else out
    out_x = out_x.reset_index() if "unique_id" not in out_x.columns else out_x
    return out.merge(out_x, on=["unique_id", "ds"])
