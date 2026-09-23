📈 Sales Forecasting Engine

Backtested revenue and unit-demand forecasting for equipment dealerships

🚀 Launch the Live Demo

A Python and Streamlit portfolio project that compares multiple forecasting methods, validates them against held-out history, and automatically selects a revenue strategy for a forward 12-month forecast.

Portfolio edition: This repository uses 100% synthetic data. It contains no employer, customer, pricing, inventory, or proprietary sales records.

The Business Problem

Historical ERP and dealership-management-system exports describe what already happened, but management still has to answer forward-looking questions:

What revenue level should we plan around?

How many units are likely to sell over the next year?

Is expected growth coming from unit volume, average selling price, or both?

Which forecasting method performs best on recent history?

How much forecast error did the selected method produce during validation?

The Sales Forecasting Engine turns those questions into a repeatable, testable forecasting workflow.

🚀 Live Demo

Launch the Sales Forecasting Engine →

The dashboard includes:

12-month revenue forecast

forecast unit volume

forecast average selling price (ASP)

prior-12-month comparison

selected forecasting strategy

holdout WAPE

historical vs forecast visualization

revenue-model backtesting

unit-model backtesting

ASP-model backtesting

manufacturer and category mix views

downloadable monthly forecast data

How the Engine Works

Synthetic Sales Transactions
            |
            v
     Monthly Aggregation
            |
            +--> Revenue
            +--> Units
            +--> ASP
            |
            v
      Holdout Validation
            |
            v
   Candidate Model Backtests
      |       |       |       |
      v       v       v       v
 Moving   Seasonal   Trend   Seasonal
 Average    Naive              Trend
            |
            v
         WAPE
            |
            v
 Revenue Strategy Comparison
       /                 \
      v                   v
Direct Revenue        Unit × ASP
       \                 /
        v               v
       Lowest Validation Error
                |
                v
       12-Month Forecast

Forecasting Strategies

The engine evaluates two revenue approaches.

Direct Revenue

Monthly revenue is forecast directly from the historical revenue series.

Historical Revenue → Forecast Model → Future Revenue

Unit × ASP

Unit volume and average selling price are modeled independently and then recombined.

Forecast Units × Forecast ASP = Forecast Revenue

This makes the forecast more interpretable because future revenue can be decomposed into expected volume and price effects.

Candidate Forecast Models

The public engine compares four lightweight forecasting methods:

Moving Average

Uses recent observations to estimate the next period and provides a simple baseline for relatively stable series.

Seasonal Naive

Repeats the corresponding period from the most recent seasonal cycle.

Linear Trend

Fits a linear time trend and extends it into future periods.

Seasonal Trend

Combines a long-term trend with estimated recurring seasonal patterns.

Rather than assuming one model is always best, the engine evaluates performance on historical holdout data.

Backtesting and WAPE

Forecast selection is based on Weighted Absolute Percentage Error (WAPE).

WAPE = Sum(|Actual - Forecast|) / Sum(|Actual|)

Lower WAPE indicates lower total weighted forecast error during the validation period.

The engine:

separates recent history as a holdout period,

trains candidate models on the earlier observations,

forecasts the unseen holdout months,

compares those forecasts with actual historical values,

calculates WAPE,

and selects the strongest-performing approach.

This prevents the dashboard from selecting a model solely because it fits the historical series attractively.

Executive Outputs

The dashboard surfaces several management-facing metrics:

12-Month Forecast Revenue

The total projected revenue across the forward forecast horizon.

Forecast Units

Expected total unit sales during the forecast period.

Forecast ASP

Forecast revenue divided by forecast units, providing a directional average selling-price view.

Forecast vs Prior 12 Months

Compares the forward forecast with the most recent twelve months of observed history.

Backtest WAPE

Shows the validation error associated with the selected revenue strategy.

Selected Strategy

Identifies whether Direct Revenue or Unit × ASP produced the stronger historical validation result.

Synthetic Data Design

The public demo intentionally includes realistic analytical features without using real company data:

multiple years of history

monthly seasonality

category-specific seasonality

modest long-term growth

manufacturer differences

model-level demand differences

store variation

changing average selling prices

random demand noise

Every transaction is generated deterministically in:

engine/demo_data.py

Technology

Python

pandas

NumPy

Streamlit

time-series aggregation

holdout backtesting

WAPE validation

automatic model selection

scenario and executive dashboard design

Repository Structure

sales-forecasting-engine/
├── app.py
├── engine/
│   ├── __init__.py
│   ├── forecasting.py
│   └── demo_data.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

Run Locally

Clone the repository:

git clone https://github.com/amonday3-ai/sales-forecasting-engine.git
cd sales-forecasting-engine

Create a virtual environment:

python -m venv .venv

Windows PowerShell

.\.venv\Scripts\Activate.ps1

Linux / macOS

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Launch the dashboard:

streamlit run app.py

Methodology Guardrails

This engine is designed for decision support, not certainty.

Forecast values are not guaranteed future sales.

Historical forecast accuracy does not guarantee future accuracy.

Promotions, weather, OEM programs, economic conditions, product availability, new models, and business-policy changes can alter future demand.

Aggregate forecasts should not be interpreted as exact SKU-level demand.

WAPE measures historical validation performance, not confidence probability.

Data Safety

This repository contains no real employer data.

It does not include:

customer records,

actual dealership transactions,

proprietary pricing,

internal revenue figures,

confidential manufacturer data,

or company financial results.

All portfolio data is fictional and generated programmatically.

What This Project Demonstrates

This project demonstrates the ability to:

translate operational sales records into forecasting datasets,

separate unit demand from selling-price effects,

build and compare multiple forecasting approaches,

validate models against unseen historical periods,

measure forecast error quantitatively,

select strategies based on evidence rather than appearance,

communicate uncertainty and analytical limitations,

and deliver forecasting results through an executive-facing application.

Potential Production Enhancements

A production implementation could add:

rolling-origin backtesting

prediction intervals

store-level hierarchical forecasting

manufacturer and category forecasts

weather signals

promotional calendars

inventory availability constraints

incoming purchase-order data

external economic indicators

model monitoring and forecast-drift alerts

automated monthly retraining

Author

Alex Monday

Business Intelligence • ERP/Data Administration • Python Automation • Operational Analytics

Live Demo
GitHub Repository
