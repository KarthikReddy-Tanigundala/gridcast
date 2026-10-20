"""GridCast dashboard — VIC1 day-ahead demand & price forecasting, benchmarked against AEMO.

Reads only committed artifacts (see ``components/data.py``). Run: ``streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts, data

st.set_page_config(page_title="GridCast — VIC1 forecasting", page_icon="⚡", layout="wide")

STYLE = """
<style>
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,0.25); border-radius: 8px; padding: 10px 14px;}
.small-note {font-size: 0.85rem; opacity: 0.75;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

st.title("⚡ GridCast")
st.caption(
    "Day-ahead forecasts of Victorian (VIC1) operational demand and spot price on the National "
    "Electricity Market, with calibrated intervals and a head-to-head against AEMO's own "
    "PREDISPATCH forecast. All times AEST (UTC+10, no daylight saving), interval-ending."
)

bt = data.backtest_results()
ab = data.aemo_results()
fc, meta = data.latest_forecast()

tabs = st.tabs(
    [
        "Tomorrow's forecast",
        "Historical actuals",
        "Backtest & leaderboard",
        "GridCast vs AEMO",
        "What-if: temperature",
    ]
)

# ------------------------------------------------------------------------------- 1. forecast
with tabs[0]:
    if fc is None:
        st.info("No live forecast yet — run `python -m gridcast.predict`.")
    else:
        st.subheader(f"Forecast for {pd.Timestamp(meta['target_day']):%A %d %B %Y}")
        st.markdown(
            f"<span class='small-note'>Issued {meta['issued_at']} AEST from data to "
            f"{meta['origin']} · weather forecast issued {meta['weather_issued_at']} · "
            f"AEMO PD run {meta.get('aemo_run_time') or 'n/a'}</span>",
            unsafe_allow_html=True,
        )
        c = st.columns(4)
        c[0].metric(
            "Peak demand",
            f"{fc['demand_yhat'].max():,.0f} MW",
            help="Highest half-hour of operational demand (excludes rooftop PV)",
        )
        c[1].metric(
            "Minimum demand",
            f"{fc['demand_yhat'].min():,.0f} MW",
            help="Midday minimum: rooftop solar suppresses operational demand",
        )
        c[2].metric("Average price", f"${fc['price_yhat'].mean():,.0f}/MWh")
        c[3].metric("Max P(price > $300)", f"{fc['p_spike'].max():.0%}")
        hist = data.forecast_history()
        act = data.actuals()
        realised = act[act["ds"].isin(fc["ds"])].rename(columns={"demand_mw": "y"})
        st.markdown("**Operational demand (MW)**")
        st.plotly_chart(
            charts.forecast_chart(fc, "demand", "MW", realised[["ds", "y"]]),
            use_container_width=True,
        )
        st.markdown("**Spot price ($/MWh)**")
        rp = act[act["ds"].isin(fc["ds"])].rename(columns={"rrp": "y"})
        st.plotly_chart(
            charts.forecast_chart(fc, "price", "$/MWh", rp[["ds", "y"]]), use_container_width=True
        )
        st.markdown("**Probability of a price spike (> $300/MWh)**")
        st.plotly_chart(
            charts.bar_chart(fc["ds"], fc["p_spike"], "P(spike)", charts.AEMO, 260, labels=False),
            use_container_width=True,
        )
        if hist is not None and hist["origin"].nunique() > 1:
            st.markdown(
                "**Live track record** — forecasts issued by the daily job, scored once actuals land"
            )
            h = hist.merge(act, on="ds")
            if len(h):
                daily = h.groupby("origin").apply(
                    lambda g: pd.Series(
                        {
                            "days": 1,
                            "demand MAE (MW)": (g["demand_mw"] - g["demand_yhat"]).abs().mean(),
                            "price MAE ($/MWh)": (g["rrp"] - g["price_yhat"]).abs().mean(),
                            "80% coverage (demand)": (
                                (g["demand_mw"] >= g["demand_lo80"])
                                & (g["demand_mw"] <= g["demand_hi80"])
                            ).mean(),
                        }
                    ),
                    include_groups=False,
                )
                st.dataframe(
                    daily.drop(columns="days").sort_index(ascending=False).round(2),
                    use_container_width=True,
                )

# ------------------------------------------------------------------------------- 2. actuals
with tabs[1]:
    act = data.actuals()
    lo, hi = act["ds"].min().date(), act["ds"].max().date()
    c1, c2 = st.columns([3, 1])
    rng = c1.slider(
        "Date range", min_value=lo, max_value=hi, value=(max(lo, hi - pd.Timedelta(days=28)), hi)
    )
    res = c2.selectbox("Resolution", ["30 min", "Daily mean"], index=0)
    view = act[(act["ds"].dt.date >= rng[0]) & (act["ds"].dt.date <= rng[1])]
    if res == "Daily mean":
        view = view.set_index("ds").resample("D", label="left").mean().reset_index()
    st.markdown("**Operational demand (MW)**")
    st.plotly_chart(charts.history_chart(view, "demand_mw", "MW"), use_container_width=True)
    st.markdown("**Spot price ($/MWh)** — note the negative prices from excess solar")
    st.plotly_chart(
        charts.history_chart(view, "rrp", "$/MWh", charts.AEMO), use_container_width=True
    )
    yearly = (
        act.assign(year=act["ds"].dt.year).groupby("year")["rrp"].apply(lambda s: (s < 0).mean())
    )
    st.markdown("**Share of half-hours with a negative price, by year**")
    st.plotly_chart(
        charts.bar_chart(yearly.index.astype(str), yearly.values, "share of intervals"),
        use_container_width=True,
    )
    st.caption(
        "Source: AEMO Aggregated Price and Demand data (VIC1). TOTALDEMAND is operational demand: "
        "it excludes rooftop PV, which is why midday demand keeps falling year on year."
    )

# ------------------------------------------------------------------------------- 3. backtest
with tabs[2]:
    if bt is None:
        st.info("Run `python -m gridcast.evaluation.backtest` to populate this tab.")
    else:
        target = st.radio(
            "Target", ["demand", "price"], horizontal=True, format_func=str.capitalize
        )
        r = bt["targets"].get(target)
        if r is None:
            st.info(f"No {target} backtest yet.")
        else:
            st.markdown(
                f"Walk-forward, **{r['n_origins']} daily forecasts** issued at 12:00 AEST for the next day, "
                f"{r['eval_start'][:10]} → {r['eval_end'][:10]}. Weather inputs are the forecasts issued the "
                "day before. **MASE < 1** beats a same-half-hour-yesterday rule."
            )
            lb = pd.DataFrame(r["leaderboard_daily"]).set_index("model")
            cols = [
                c
                for c in [
                    "MASE",
                    "MAE",
                    "RMSE",
                    "sMAPE",
                    "pinball80",
                    "coverage80",
                    "coverage95",
                    "width80",
                ]
                if c in lb
            ]
            shown = lb[cols].astype(float).map(lambda v: "—" if pd.isna(v) else f"{v:,.3f}")
            st.dataframe(shown, use_container_width=True)
            if r.get("leaderboard_weekly"):
                with st.expander(
                    f"All models incl. classical ({r['n_classical_origins']} weekly origins)"
                ):
                    st.dataframe(
                        pd.DataFrame(r["leaderboard_weekly"])
                        .set_index("model")[["MASE", "MAE", "RMSE"]]
                        .style.format("{:,.3f}"),
                        use_container_width=True,
                    )
            byh = pd.DataFrame(r["mae_by_hour"])
            series = {
                m: (m, col)
                for m, col in [
                    ("LightGBM", charts.GRIDCAST),
                    ("Seasonal-naive (week)", charts.ACTUAL),
                ]
                if m in byh
            }
            st.markdown("**Error by hour of the target day**")
            st.plotly_chart(
                charts.lines_chart(
                    byh, "hour", series, f"MAE ({r['unit']})", x_title="hour (AEST)"
                ),
                use_container_width=True,
            )
            preds = data.backtest_predictions(target)
            if preds is not None:
                lgbm = preds[preds["model"] == "LightGBM"]
                days = sorted(lgbm["origin"].dt.date.unique())
                pick = st.select_slider(
                    "Inspect a backtest day (forecast origin)", options=days, value=days[-1]
                )
                day = lgbm[lgbm["origin"].dt.date == pick]
                st.plotly_chart(charts.backtest_day_chart(day, r["unit"]), use_container_width=True)
            if target == "price":
                sp = data.spike_results()
                if sp:
                    st.markdown(
                        f"**Spike classifier — P(RRP > ${sp['threshold_usd']:.0f})**, base rate "
                        f"{sp['base_rate']:.2%} of half-hours ({sp['n_spikes']} spikes)"
                    )
                    tbl = pd.DataFrame(
                        {"GridCast classifier": sp["model"], "Climatology": sp["climatology"]}
                    ).T
                    st.dataframe(tbl.style.format("{:.3f}"), use_container_width=True)

# ------------------------------------------------------------------------------- 4. vs AEMO
with tabs[3]:
    if ab is None:
        st.info("Run `python -m gridcast.evaluation.aemo_benchmark` to populate this tab.")
    else:
        st.markdown(
            "Same target intervals, same metrics. GridCast is issued at 12:00; AEMO's comparator run is "
            "published ≈12:33 with dispatch-level information (bids, outages, network limits) that no "
            "public model can see. **Where AEMO wins, we say so.**"
        )
        target = st.radio(
            "Target ", ["demand", "price"], horizontal=True, format_func=str.capitalize
        )
        r = ab.get(target)
        if r:
            c = st.columns(3)
            t = {row["model"]: row for row in r["table"]}
            c[0].metric("GridCast MAE", f"{t['GridCast (LightGBM)']['MAE']:,.1f} {r['unit']}")
            c[1].metric("AEMO PD MAE", f"{t['AEMO PREDISPATCH']['MAE']:,.1f} {r['unit']}")
            c[2].metric(
                "Days GridCast wins",
                f"{r['gridcast_win_rate']:.0%}",
                help=f"{r['n_days']} days compared",
            )
            st.dataframe(
                pd.DataFrame(r["table"]).set_index("model").style.format("{:,.2f}"),
                use_container_width=True,
            )
            byh = pd.DataFrame(r["mae_by_hour"])
            st.markdown("**Error by hour of the target day**")
            st.plotly_chart(
                charts.lines_chart(
                    byh,
                    "hour",
                    {
                        "gridcast": ("GridCast", charts.GRIDCAST),
                        "aemo": ("AEMO PREDISPATCH", charts.AEMO),
                    },
                    f"MAE ({r['unit']})",
                    x_title="hour (AEST)",
                ),
                use_container_width=True,
            )
            bym = pd.DataFrame(r["mae_by_month"])
            st.markdown("**Error by month**")
            st.plotly_chart(
                charts.lines_chart(
                    bym,
                    "month",
                    {
                        "gridcast": ("GridCast", charts.GRIDCAST),
                        "aemo": ("AEMO PREDISPATCH", charts.AEMO),
                    },
                    f"MAE ({r['unit']})",
                ),
                use_container_width=True,
            )
        conv = pd.DataFrame(ab["convergence"])
        col = "demand_mae" if target == "demand" else "price_mae"
        st.markdown("**AEMO forecast convergence** — AEMO's error vs lead time across every PD run")
        fig = charts.lines_chart(
            conv,
            "lead_h",
            {col: ("AEMO PREDISPATCH", charts.AEMO)},
            "MAE",
            x_title="lead time (hours before the interval)",
        )
        fig.add_vrect(
            x0=12.5,
            x1=36,
            fillcolor="rgba(42,120,214,0.08)",
            line_width=0,
            annotation_text="GridCast day-ahead lead times",
            annotation_position="top left",
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------- 5. what-if
with tabs[4]:
    inputs, model = data.whatif_inputs(), data.demand_model()
    if inputs is None or model is None or meta is None:
        st.info(
            "Needs the committed demand model and a live forecast (`python -m gridcast.predict`)."
        )
    else:
        st.markdown(
            "Shift tomorrow's forecast temperature and re-run the committed demand model. Melbourne's "
            "demand is U-shaped in temperature: heating load in winter, air-conditioning above ~25 °C."
        )
        shift = st.slider("Temperature shift (°C)", -8.0, 8.0, 3.0, 1.0)
        from gridcast.features.build_features import exog_columns, model_frame
        from gridcast.models.ml_lightgbm import HISTORY_ROWS, predict_origins

        origin = pd.Timestamp(meta["origin"])

        @st.cache_data(ttl=600)
        def scenario(delta: float) -> pd.DataFrame:
            df = inputs.copy()
            df.loc[df["ds"] > origin, "temperature"] += delta
            frame = model_frame(df, "demand_mw")
            hist = frame[frame["ds"] <= origin].tail(HISTORY_ROWS)
            fut = frame[frame["ds"] > origin]
            p = predict_origins(model, hist, fut[["unique_id", "ds", *exog_columns()]], len(fut))
            tomorrow = origin.normalize() + pd.Timedelta(days=1)
            p = p.rename(columns={"LightGBM": "yhat"})
            return p[p["ds"] > tomorrow]

        base, scen = scenario(0.0), scenario(shift)
        c = st.columns(3)
        c[0].metric("Peak (as issued)", f"{base['yhat'].max():,.0f} MW")
        c[1].metric(
            "Peak (scenario)",
            f"{scen['yhat'].max():,.0f} MW",
            f"{scen['yhat'].max() - base['yhat'].max():+,.0f} MW",
        )
        c[2].metric(
            "Daily energy change",
            f"{(scen['yhat'].sum() - base['yhat'].sum()) / 2 / 1000:+,.1f} GWh",
        )
        st.plotly_chart(charts.scenario_chart(base, scen, shift), use_container_width=True)
        st.caption(
            "A sensitivity tool, not a causal model: it moves temperature alone, holding humidity, cloud "
            "and solar fixed, and extrapolates poorly beyond temperatures seen in training."
        )

st.divider()
st.markdown(
    "<span class='small-note'>Data: AEMO (NEMweb, Aggregated Price & Demand, PREDISPATCH) · weather: "
    "Open-Meteo (CC-BY 4.0) · built by Karthik Reddy Tanigundala · "
    "[source on GitHub](https://github.com/KarthikReddy-Tanigundala/gridcast)</span>",
    unsafe_allow_html=True,
)
