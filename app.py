from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import build_demo_sales, build_forecast, forecast_summary

st.set_page_config(page_title="Sales Forecasting Engine", page_icon="📈", layout="wide")


def money(value: float) -> str:
    return f"${value:,.0f}"


def percent(value: float) -> str:
    return f"{value:.1%}"


def pretty_name(value: str) -> str:
    return value.replace("_", " ").title()


st.title("📈 Sales Forecasting Engine")
st.markdown(
    """
    **Backtested revenue and unit-demand forecasting for equipment dealerships**

    This public portfolio demo compares multiple forecasting approaches,
    selects the strongest strategy using holdout backtesting, and produces
    a forward 12-month executive forecast.
    """
)
st.caption(
    "Portfolio edition • 100% synthetic data • No employer, customer, pricing, "
    "inventory, or proprietary sales records are included."
)

with st.sidebar:
    st.header("Forecast Controls")
    history_window = st.selectbox(
        "Historical chart window",
        options=[24, 36, 48, "All"],
        index=1,
    )
    st.divider()
    st.subheader("Forecast Design")
    st.markdown(
        """
        **Forecast horizon:** 12 months

        **Validation:** Recent-history holdout

        **Error metric:** WAPE

        **Candidate models:**
        - Moving Average
        - Seasonal Naive
        - Linear Trend
        - Seasonal Trend

        **Revenue strategies:**
        - Direct Revenue
        - Unit × ASP
        """
    )
    st.warning("Forecasts are decision-support estimates, not guaranteed future results.")

sales = build_demo_sales()
result = build_forecast(sales, horizon=12)
summary = forecast_summary(result)

history = result["monthly_history"].copy()
forecast = result["forecast"].copy()
strategy_backtest = result["strategy_backtest"].copy()
revenue_backtest = result["revenue_model_backtest"].copy()
unit_backtest = result["unit_model_backtest"].copy()
asp_backtest = result["asp_model_backtest"].copy()

with st.expander("💼 Business Problem & Approach", expanded=False):
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            ### Business problem
            Historical sales exports describe what already happened, but management still needs to answer:
            - What revenue level should we plan around?
            - How many units are likely to sell?
            - Is growth coming from volume, ASP, or both?
            - Which forecasting method performs best on recent history?
            - How much confidence should we place in the selected model?
            """
        )
    with right:
        st.markdown(
            """
            ### Analytical approach
            1. Aggregate transactions to monthly revenue, units, and ASP.
            2. Hold out recent months for validation.
            3. Backtest multiple candidate forecasting models.
            4. Compare error using WAPE.
            5. Compare Direct Revenue vs Unit × ASP strategies.
            6. Select the lower-error strategy.
            7. Refit against full history and forecast the next 12 months.
            """
        )

st.subheader("Executive Forecast")
k1, k2, k3, k4 = st.columns(4)
k1.metric("12-Month Forecast Revenue", money(summary["forecast_revenue"]))
k2.metric("Forecast Units", f'{summary["forecast_units"]:,.0f}')
k3.metric("Forecast ASP", money(summary["forecast_asp"]))
k4.metric("Backtest WAPE", percent(summary["backtest_wape"]))

k5, k6, k7 = st.columns(3)
k5.metric("Prior 12-Month Revenue", money(summary["prior_12_month_revenue"]))
k6.metric("Forecast vs Prior 12 Months", percent(summary["forecast_vs_prior_12"]))
k7.metric("Selected Strategy", pretty_name(summary["selected_strategy"]))

st.info(
    "The engine selects the strategy with the lowest holdout WAPE. "
    "Lower WAPE indicates lower weighted absolute forecast error on the validation period."
)

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Backtesting", "Business Mix", "Monthly Forecast"])

with tab1:
    st.subheader("Historical Revenue vs 12-Month Forecast")
    history_chart = history[["month", "revenue"]].copy()
    if history_window != "All":
        history_chart = history_chart.tail(int(history_window))
    history_chart = history_chart.rename(columns={"revenue": "Actual Revenue"}).set_index("month")
    forecast_chart = forecast[["month", "forecast_revenue"]].rename(
        columns={"forecast_revenue": "Forecast Revenue"}
    ).set_index("month")
    chart_data = history_chart.join(forecast_chart, how="outer").sort_index()
    st.line_chart(chart_data)

    left, right = st.columns(2)
    with left:
        st.subheader("Forecast Unit Volume")
        st.bar_chart(forecast[["month", "forecast_units"]].set_index("month"))
    with right:
        st.subheader("Forecast ASP")
        st.line_chart(forecast[["month", "forecast_asp"]].set_index("month"))

with tab2:
    st.subheader("Revenue Strategy Comparison")
    strategy_display = strategy_backtest.copy()
    strategy_display["strategy"] = strategy_display["strategy"].map(pretty_name)
    strategy_display["wape"] = strategy_display["wape"] * 100
    st.dataframe(
        strategy_display.rename(columns={"strategy": "Strategy", "wape": "WAPE (%)"}),
        use_container_width=True,
        hide_index=True,
        column_config={"WAPE (%)": st.column_config.NumberColumn(format="%.2f%%")},
    )
    st.markdown(f'**Selected strategy:** `{pretty_name(result["selected_strategy"])}`')
    st.divider()

    for column, title, data in zip(
        st.columns(3),
        ["Revenue Models", "Unit Models", "ASP Models"],
        [revenue_backtest, unit_backtest, asp_backtest],
    ):
        with column:
            st.markdown(f"### {title}")
            display = data.copy()
            display["model"] = display["model"].map(pretty_name)
            display["wape"] = display["wape"] * 100
            st.dataframe(
                display.rename(columns={"model": "Model", "wape": "WAPE (%)"}),
                use_container_width=True,
                hide_index=True,
                column_config={"WAPE (%)": st.column_config.NumberColumn(format="%.2f%%")},
            )

with tab3:
    st.subheader("Historical Business Mix")
    manufacturer = (
        sales.groupby("manufacturer", as_index=False)
        .agg(units=("units", "sum"), revenue=("sale_value", "sum"))
        .sort_values("revenue", ascending=False)
    )
    category = (
        sales.groupby("category", as_index=False)
        .agg(units=("units", "sum"), revenue=("sale_value", "sum"))
        .sort_values("revenue", ascending=False)
    )
    left, right = st.columns(2)
    with left:
        st.markdown("### Manufacturer Revenue")
        st.bar_chart(manufacturer.set_index("manufacturer")["revenue"])
        st.dataframe(
            manufacturer.rename(columns={"manufacturer": "Manufacturer", "units": "Units", "revenue": "Revenue"}),
            use_container_width=True,
            hide_index=True,
            column_config={"Revenue": st.column_config.NumberColumn(format="$%d")},
        )
    with right:
        st.markdown("### Category Revenue")
        st.bar_chart(category.set_index("category")["revenue"])
        st.dataframe(
            category.rename(columns={"category": "Category", "units": "Units", "revenue": "Revenue"}),
            use_container_width=True,
            hide_index=True,
            column_config={"Revenue": st.column_config.NumberColumn(format="$%d")},
        )

with tab4:
    st.subheader("12-Month Forecast Detail")
    forecast_display = forecast.copy()
    forecast_display["month"] = forecast_display["month"].dt.strftime("%b %Y")
    forecast_display = forecast_display.rename(
        columns={
            "month": "Month",
            "forecast_units": "Forecast Units",
            "forecast_asp": "Forecast ASP",
            "forecast_revenue": "Forecast Revenue",
        }
    )
    st.dataframe(
        forecast_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Forecast Units": st.column_config.NumberColumn(format="%.1f"),
            "Forecast ASP": st.column_config.NumberColumn(format="$%.0f"),
            "Forecast Revenue": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

st.divider()
st.subheader("Download Forecast")
download_df = forecast.copy()
download_df["month"] = download_df["month"].dt.strftime("%Y-%m")
csv_data = download_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download 12-Month Forecast as CSV",
    data=csv_data,
    file_name="sales_forecasting_demo.csv",
    mime="text/csv",
)

with st.expander("Methodology & Guardrails"):
    st.markdown(
        """
        ### What the engine does
        1. Aggregates synthetic transactions to monthly revenue, units, and ASP.
        2. Holds out recent history to simulate an unseen forecast period.
        3. Backtests Moving Average, Seasonal Naive, Trend, and Seasonal Trend models.
        4. Measures forecast error using WAPE.
        5. Compares a Direct Revenue strategy with a Unit × ASP strategy.
        6. Selects the lower-error strategy.
        7. Rebuilds the forecast using the available historical series.
        8. Produces a forward 12-month forecast.

        ### What the forecast does not claim
        - Forecast values are **not guaranteed sales results**.
        - Historical patterns may change because of promotions, weather, economic conditions, OEM programs, product availability, or new models.
        - Backtest performance describes historical validation performance, not certainty about the future.
        - Aggregate forecasts should not be interpreted as exact SKU-level demand.

        ### Data safety
        Every transaction in this portfolio demo is synthetic and generated deterministically in `engine/demo_data.py`.
        """
    )

st.caption(
    "Portfolio project by Alex Monday • Python • pandas • Streamlit • "
    "forecasting • backtesting • WAPE • executive decision support"
)
