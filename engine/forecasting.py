"""Forecasting logic for the public Sales Forecasting Engine portfolio demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FORECAST_HORIZON = 12


@dataclass
class BacktestResult:
    model: str
    wape: float


def _monthly_summary(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw transactions to monthly business metrics."""

    df = sales.copy()

    df["sale_date"] = pd.to_datetime(
        df["sale_date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["sale_date"]
    )

    df["month"] = (
        df["sale_date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        df.groupby(
            "month",
            as_index=False,
        )
        .agg(
            units=("units", "sum"),
            revenue=("sale_value", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    monthly["asp"] = np.where(
        monthly["units"] > 0,
        monthly["revenue"] / monthly["units"],
        np.nan,
    )

    return monthly


def _wape(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Weighted absolute percentage error."""

    actual = pd.to_numeric(
        actual,
        errors="coerce",
    ).fillna(0)

    forecast = pd.to_numeric(
        forecast,
        errors="coerce",
    ).fillna(0)

    denominator = actual.abs().sum()

    if denominator == 0:
        return float("nan")

    return float(
        (actual - forecast).abs().sum()
        / denominator
    )


def _moving_average_forecast(
    history: pd.Series,
    horizon: int,
    window: int = 6,
) -> pd.Series:
    """Simple moving-average forecast."""

    values = (
        pd.to_numeric(
            history,
            errors="coerce",
        )
        .dropna()
        .astype(float)
        .tolist()
    )

    predictions = []

    for _ in range(horizon):
        if not values:
            prediction = 0.0
        else:
            prediction = float(
                np.mean(
                    values[-window:]
                )
            )

        predictions.append(
            prediction
        )

        values.append(
            prediction
        )

    return pd.Series(
        predictions,
        dtype=float,
    )


def _seasonal_naive_forecast(
    history: pd.Series,
    horizon: int,
    season_length: int = 12,
) -> pd.Series:
    """Repeat the most recent observed seasonal cycle."""

    values = (
        pd.to_numeric(
            history,
            errors="coerce",
        )
        .fillna(0)
        .astype(float)
        .tolist()
    )

    if not values:
        return pd.Series(
            [0.0] * horizon
        )

    predictions = []

    for step in range(horizon):
        if len(values) >= season_length:
            prediction = values[
                -season_length
            ]
        else:
            prediction = float(
                np.mean(values)
            )

        predictions.append(
            float(prediction)
        )

        values.append(
            float(prediction)
        )

    return pd.Series(
        predictions,
        dtype=float,
    )


def _trend_forecast(
    history: pd.Series,
    horizon: int,
) -> pd.Series:
    """Linear trend forecast with a non-negative floor."""

    y = (
        pd.to_numeric(
            history,
            errors="coerce",
        )
        .dropna()
        .astype(float)
        .to_numpy()
    )

    if len(y) == 0:
        return pd.Series(
            [0.0] * horizon
        )

    if len(y) == 1:
        return pd.Series(
            [max(y[0], 0.0)] * horizon
        )

    x = np.arange(
        len(y),
        dtype=float,
    )

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    future_x = np.arange(
        len(y),
        len(y) + horizon,
        dtype=float,
    )

    forecast = (
        intercept
        + slope * future_x
    )

    forecast = np.maximum(
        forecast,
        0,
    )

    return pd.Series(
        forecast,
        dtype=float,
    )


def _seasonal_trend_forecast(
    history: pd.Series,
    horizon: int,
    season_length: int = 12,
) -> pd.Series:
    """
    Forecast using a simple long-term trend combined
    with monthly seasonal indices.
    """

    values = (
        pd.to_numeric(
            history,
            errors="coerce",
        )
        .dropna()
        .astype(float)
        .reset_index(drop=True)
    )

    if len(values) < season_length * 2:
        return _moving_average_forecast(
            values,
            horizon,
        )

    x = np.arange(
        len(values),
        dtype=float,
    )

    slope, intercept = np.polyfit(
        x,
        values.to_numpy(),
        1,
    )

    trend = (
        intercept
        + slope * x
    )

    safe_trend = np.where(
        trend <= 0,
        np.nan,
        trend,
    )

    seasonal_ratio = (
        values.to_numpy()
        / safe_trend
    )

    month_numbers = (
        np.arange(len(values))
        % season_length
    )

    seasonal_indices = {}

    for month_number in range(
        season_length
    ):
        month_values = seasonal_ratio[
            month_numbers == month_number
        ]

        month_values = month_values[
            np.isfinite(month_values)
        ]

        if len(month_values) == 0:
            seasonal_indices[
                month_number
            ] = 1.0
        else:
            seasonal_indices[
                month_number
            ] = float(
                np.mean(month_values)
            )

    avg_index = np.mean(
        list(
            seasonal_indices.values()
        )
    )

    if avg_index:
        seasonal_indices = {
            key: value / avg_index
            for key, value
            in seasonal_indices.items()
        }

    predictions = []

    for step in range(horizon):
        position = (
            len(values)
            + step
        )

        base = (
            intercept
            + slope * position
        )

        seasonal = seasonal_indices[
            position % season_length
        ]

        prediction = max(
            base * seasonal,
            0,
        )

        predictions.append(
            prediction
        )

    return pd.Series(
        predictions,
        dtype=float,
    )


def _forecast_series(
    history: pd.Series,
    model: str,
    horizon: int,
) -> pd.Series:
    """Dispatch a forecasting strategy."""

    if model == "MOVING_AVERAGE":
        return _moving_average_forecast(
            history,
            horizon,
        )

    if model == "SEASONAL_NAIVE":
        return _seasonal_naive_forecast(
            history,
            horizon,
        )

    if model == "TREND":
        return _trend_forecast(
            history,
            horizon,
        )

    if model == "SEASONAL_TREND":
        return _seasonal_trend_forecast(
            history,
            horizon,
        )

    raise ValueError(
        f"Unknown model: {model}"
    )


def backtest_models(
    history: pd.Series,
    holdout: int = 12,
) -> pd.DataFrame:
    """
    Backtest multiple forecasting strategies
    using the most recent holdout period.
    """

    series = (
        pd.to_numeric(
            history,
            errors="coerce",
        )
        .dropna()
        .reset_index(drop=True)
    )

    if len(series) <= holdout:
        raise ValueError(
            "Not enough history for backtesting."
        )

    train = series.iloc[
        :-holdout
    ]

    actual = series.iloc[
        -holdout:
    ].reset_index(drop=True)

    candidates = [
        "MOVING_AVERAGE",
        "SEASONAL_NAIVE",
        "TREND",
        "SEASONAL_TREND",
    ]

    rows = []

    for model in candidates:
        forecast = _forecast_series(
            train,
            model,
            holdout,
        )

        score = _wape(
            actual,
            forecast,
        )

        rows.append(
            {
                "model": model,
                "wape": score,
            }
        )

    results = pd.DataFrame(
        rows
    )

    return (
        results.sort_values(
            "wape",
            ascending=True,
        )
        .reset_index(drop=True)
    )


def _unit_x_asp_backtest(
    monthly: pd.DataFrame,
    holdout: int = 12,
) -> float:
    """
    Backtest a unit × ASP strategy.

    Unit volume and ASP are independently
    forecast, then multiplied into revenue.
    """

    if len(monthly) <= holdout:
        return float("nan")

    train = monthly.iloc[
        :-holdout
    ].copy()

    actual = monthly.iloc[
        -holdout:
    ].reset_index(drop=True)

    unit_results = backtest_models(
        train["units"],
        holdout=min(
            12,
            max(
                3,
                len(train) // 4,
            ),
        ),
    )

    asp_results = backtest_models(
        train["asp"],
        holdout=min(
            12,
            max(
                3,
                len(train) // 4,
            ),
        ),
    )

    unit_model = unit_results.iloc[
        0
    ]["model"]

    asp_model = asp_results.iloc[
        0
    ]["model"]

    unit_forecast = _forecast_series(
        train["units"],
        unit_model,
        holdout,
    )

    asp_forecast = _forecast_series(
        train["asp"],
        asp_model,
        holdout,
    )

    revenue_forecast = (
        unit_forecast
        * asp_forecast
    )

    return _wape(
        actual["revenue"],
        revenue_forecast,
    )


def build_forecast(
    sales: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> dict:
    """
    Build a 12-month revenue forecast and
    automatically select the strongest strategy.
    """

    monthly = _monthly_summary(
        sales
    )

    if len(monthly) < 24:
        raise ValueError(
            "At least 24 months of history are required."
        )

    holdout = min(
        12,
        max(
            3,
            len(monthly) // 4,
        ),
    )

    revenue_backtest = backtest_models(
        monthly["revenue"],
        holdout=holdout,
    )

    direct_model = revenue_backtest.iloc[
        0
    ]["model"]

    direct_wape = float(
        revenue_backtest.iloc[
            0
        ]["wape"]
    )

    unit_backtest = backtest_models(
        monthly["units"],
        holdout=holdout,
    )

    asp_backtest = backtest_models(
        monthly["asp"],
        holdout=holdout,
    )

    unit_model = unit_backtest.iloc[
        0
    ]["model"]

    asp_model = asp_backtest.iloc[
        0
    ]["model"]

    unit_x_asp_wape = _unit_x_asp_backtest(
        monthly,
        holdout=holdout,
    )

    strategy_rows = [
        {
            "strategy": "DIRECT_REVENUE",
            "wape": direct_wape,
        },
        {
            "strategy": "UNIT_X_ASP",
            "wape": unit_x_asp_wape,
        },
    ]

    strategy_results = (
        pd.DataFrame(
            strategy_rows
        )
        .sort_values(
            "wape",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    selected_strategy = (
        strategy_results.iloc[
            0
        ]["strategy"]
    )

    future_units = _forecast_series(
        monthly["units"],
        unit_model,
        horizon,
    )

    future_asp = _forecast_series(
        monthly["asp"],
        asp_model,
        horizon,
    )

    direct_revenue = _forecast_series(
        monthly["revenue"],
        direct_model,
        horizon,
    )

    if selected_strategy == "UNIT_X_ASP":
        future_revenue = (
            future_units
            * future_asp
        )
    else:
        future_revenue = direct_revenue

    last_month = monthly[
        "month"
    ].max()

    future_months = pd.date_range(
        last_month
        + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    forecast = pd.DataFrame(
        {
            "month": future_months,
            "forecast_units": (
                future_units
                .clip(lower=0)
                .to_numpy()
            ),
            "forecast_asp": (
                future_asp
                .clip(lower=0)
                .to_numpy()
            ),
            "forecast_revenue": (
                future_revenue
                .clip(lower=0)
                .to_numpy()
            ),
        }
    )

    forecast[
        "forecast_units"
    ] = forecast[
        "forecast_units"
    ].round(1)

    forecast[
        "forecast_asp"
    ] = forecast[
        "forecast_asp"
    ].round(2)

    forecast[
        "forecast_revenue"
    ] = forecast[
        "forecast_revenue"
    ].round(2)

    return {
        "monthly_history": monthly,
        "forecast": forecast,
        "strategy_backtest": strategy_results,
        "revenue_model_backtest": revenue_backtest,
        "unit_model_backtest": unit_backtest,
        "asp_model_backtest": asp_backtest,
        "selected_strategy": selected_strategy,
        "revenue_model": direct_model,
        "unit_model": unit_model,
        "asp_model": asp_model,
    }


def forecast_summary(
    result: dict,
) -> dict:
    """Return executive summary metrics."""

    forecast = result[
        "forecast"
    ]

    history = result[
        "monthly_history"
    ]

    forecast_revenue = float(
        forecast[
            "forecast_revenue"
        ].sum()
    )

    forecast_units = float(
        forecast[
            "forecast_units"
        ].sum()
    )

    weighted_asp = (
        forecast_revenue
        / forecast_units
        if forecast_units
        else 0
    )

    recent_12 = history.tail(
        12
    )

    historical_revenue = float(
        recent_12[
            "revenue"
        ].sum()
    )

    yoy_change = (
        forecast_revenue
        / historical_revenue
        - 1
        if historical_revenue
        else 0
    )

    selected_wape = float(
        result[
            "strategy_backtest"
        ]
        .iloc[0]["wape"]
    )

    return {
        "forecast_revenue": (
            forecast_revenue
        ),
        "forecast_units": (
            forecast_units
        ),
        "forecast_asp": (
            weighted_asp
        ),
        "prior_12_month_revenue": (
            historical_revenue
        ),
        "forecast_vs_prior_12": (
            yoy_change
        ),
        "selected_strategy": (
            result[
                "selected_strategy"
            ]
        ),
        "backtest_wape": (
            selected_wape
        ),
    }
