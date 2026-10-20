"""Plotly figure builders. One palette, fixed role -> colour mapping (never cycled):
GridCast = blue, AEMO = orange, scenario = aqua, actuals = neutral ink. Uncertainty bands are the
GridCast hue at low opacity. Single y-axis per chart (no dual axes)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

GRIDCAST = "#2a78d6"
AEMO = "#eb6834"
SCENARIO = "#1baf7a"
ACTUAL = "#52514e"
BAND_80 = "rgba(42,120,214,0.22)"
BAND_95 = "rgba(42,120,214,0.10)"
GRID = "rgba(128,128,128,0.18)"


def _layout(fig: go.Figure, y_title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title=y_title, gridcolor=GRID, zeroline=False),
        xaxis=dict(gridcolor=GRID, showgrid=False),
    )
    return fig


def band(fig: go.Figure, x, lo, hi, fill: str, name: str) -> None:
    fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lo,
            line=dict(width=0),
            fill="tonexty",
            fillcolor=fill,
            name=name,
            hoverinfo="skip",
        )
    )


def forecast_chart(
    fc: pd.DataFrame, target: str, unit: str, actual: pd.DataFrame | None = None
) -> go.Figure:
    fig = go.Figure()
    if f"{target}_lo95" in fc:
        band(fig, fc["ds"], fc[f"{target}_lo95"], fc[f"{target}_hi95"], BAND_95, "95% interval")
        band(fig, fc["ds"], fc[f"{target}_lo80"], fc[f"{target}_hi80"], BAND_80, "80% interval")
    fig.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc[f"{target}_yhat"],
            name="GridCast",
            line=dict(color=GRIDCAST, width=2.5),
        )
    )
    aemo_col = {"demand": "aemo_demand_mw", "price": "aemo_rrp"}[target]
    if aemo_col in fc and fc[aemo_col].notna().any():
        fig.add_trace(
            go.Scatter(
                x=fc["ds"],
                y=fc[aemo_col],
                name="AEMO PREDISPATCH",
                line=dict(color=AEMO, width=2, dash="dash"),
            )
        )
    if actual is not None and len(actual):
        fig.add_trace(
            go.Scatter(
                x=actual["ds"], y=actual["y"], name="Actual", line=dict(color=ACTUAL, width=2)
            )
        )
    return _layout(fig, unit)


def backtest_day_chart(day: pd.DataFrame, unit: str, aemo: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    if "lo95" in day and day["lo95"].notna().any():
        band(fig, day["ds"], day["lo95"], day["hi95"], BAND_95, "95% interval")
        band(fig, day["ds"], day["lo80"], day["hi80"], BAND_80, "80% interval")
    fig.add_trace(
        go.Scatter(
            x=day["ds"], y=day["yhat"], name="GridCast", line=dict(color=GRIDCAST, width=2.5)
        )
    )
    if aemo is not None and len(aemo):
        fig.add_trace(
            go.Scatter(
                x=aemo["ds"],
                y=aemo["aemo"],
                name="AEMO PREDISPATCH",
                line=dict(color=AEMO, width=2, dash="dash"),
            )
        )
    fig.add_trace(
        go.Scatter(x=day["ds"], y=day["y"], name="Actual", line=dict(color=ACTUAL, width=2))
    )
    return _layout(fig, unit)


def lines_chart(
    df: pd.DataFrame,
    x: str,
    series: dict[str, tuple[str, str]],
    y_title: str,
    height: int = 360,
    x_title: str | None = None,
) -> go.Figure:
    """``series``: column -> (label, colour)."""
    fig = go.Figure()
    for col, (label, color) in series.items():
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[col],
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            )
        )
    fig = _layout(fig, y_title, height)
    if x_title:
        fig.update_xaxes(title=x_title)
    return fig


def history_chart(df: pd.DataFrame, col: str, unit: str, color: str = GRIDCAST) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df["ds"], y=df[col], line=dict(color=color, width=1.5), name=col))
    fig = _layout(fig, unit, 340)
    fig.update_layout(showlegend=False)
    return fig


def bar_chart(
    x, y, y_title: str, color: str = GRIDCAST, height: int = 300, labels: bool = True
) -> go.Figure:
    """Percentage bars. ``labels`` prints values on bars — only for a handful of bars."""
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            marker=dict(color=color, cornerradius=4),
            text=[f"{v:.0%}" for v in y] if labels else None,
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    fig = _layout(fig, y_title, height)
    fig.update_layout(showlegend=False, hovermode="closest", yaxis_tickformat=".0%")
    return fig


def scenario_chart(base: pd.DataFrame, scen: pd.DataFrame, shift: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base["ds"],
            y=base["yhat"],
            name="Forecast (as issued)",
            line=dict(color=GRIDCAST, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scen["ds"],
            y=scen["yhat"],
            name=f"Scenario ({shift:+.0f} °C)",
            line=dict(color=SCENARIO, width=2.5, dash="dot"),
        )
    )
    return _layout(fig, "MW")
