"""Deterministic synthetic sales data for the public forecasting portfolio demo."""

from __future__ import annotations

from datetime import date
import math
import random

import pandas as pd


STORES = [
    "Grand River",
    "Lakeview",
    "Northfield",
    "Pine Ridge",
]

CATALOG = {
    "Apex Motors": {
        "UTV": [
            ("Ranger 900", 13500),
            ("Ranger 1000", 16800),
            ("WorkPro 1000", 18100),
        ],
        "ATV": [
            ("Trail 570", 8300),
            ("Trail 850", 10900),
        ],
    },
    "FieldPro": {
        "Zero Turn": [
            ("ZT 42", 4900),
            ("ZT 54", 6500),
            ("ZT 60", 7800),
            ("ProTurn 52", 8900),
            ("ProTurn 60", 10400),
        ],
    },
    "IronPeak": {
        "Compact Equipment": [
            ("MX 26", 18200),
            ("MX 30", 22400),
            ("RX 48", 27900),
            ("RX 60", 33500),
        ],
        "Utility Equipment": [
            ("Utility 24", 14900),
        ],
    },
    "TrailWorks": {
        "UTV": [
            ("Defender 700", 12600),
            ("Defender 1000", 17400),
            ("Scout 500", 9800),
            ("Scout 800", 14300),
            ("Summit 1000", 19400),
        ],
    },
    "Northland": {
        "Lawn Equipment": [
            ("Lawn 42", 3100),
            ("Lawn 46", 3700),
            ("Lawn 54", 4400),
        ],
        "Snow Equipment": [
            ("Snow 24", 1900),
            ("Snow 30", 2600),
        ],
    },
}


MONTH_SEASONALITY = {
    1: 0.68,
    2: 0.72,
    3: 0.92,
    4: 1.18,
    5: 1.35,
    6: 1.28,
    7: 1.16,
    8: 1.08,
    9: 0.98,
    10: 0.90,
    11: 0.82,
    12: 0.76,
}


CATEGORY_SEASONALITY = {
    "Zero Turn": {
        1: 0.25,
        2: 0.30,
        3: 0.70,
        4: 1.45,
        5: 1.70,
        6: 1.55,
        7: 1.25,
        8: 1.00,
        9: 0.75,
        10: 0.50,
        11: 0.30,
        12: 0.20,
    },
    "Lawn Equipment": {
        1: 0.30,
        2: 0.35,
        3: 0.75,
        4: 1.40,
        5: 1.60,
        6: 1.45,
        7: 1.20,
        8: 1.00,
        9: 0.75,
        10: 0.55,
        11: 0.35,
        12: 0.25,
    },
    "Snow Equipment": {
        1: 1.50,
        2: 1.35,
        3: 0.80,
        4: 0.35,
        5: 0.20,
        6: 0.15,
        7: 0.15,
        8: 0.20,
        9: 0.45,
        10: 0.90,
        11: 1.35,
        12: 1.65,
    },
}


def _month_range(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
):
    """Yield first-of-month dates across an inclusive month range."""

    year = start_year
    month = start_month

    while (year, month) <= (end_year, end_month):
        yield date(year, month, 1)

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def build_demo_sales(
    seed: int = 4712,
    start_year: int = 2022,
    start_month: int = 1,
    end_year: int = 2026,
    end_month: int = 8,
) -> pd.DataFrame:
    """
    Generate deterministic synthetic monthly sales transactions.

    The data includes:
    - multi-year history,
    - seasonal demand,
    - modest long-term growth,
    - manufacturer and category differences,
    - store variation,
    - changing ASP,
    - and random demand noise.

    No employer or proprietary business data is represented.
    """

    rng = random.Random(seed)

    rows = []
    transaction_id = 100000

    months = list(
        _month_range(
            start_year,
            start_month,
            end_year,
            end_month,
        )
    )

    for month_index, month_start in enumerate(months):
        year_growth = 1 + (month_index / 12) * 0.035

        general_seasonality = MONTH_SEASONALITY[
            month_start.month
        ]

        for brand_index, (
            manufacturer,
            categories,
        ) in enumerate(CATALOG.items()):

            brand_factor = (
                0.85
                + brand_index * 0.08
            )

            for category, products in categories.items():

                category_curve = (
                    CATEGORY_SEASONALITY
                    .get(category, {})
                    .get(
                        month_start.month,
                        general_seasonality,
                    )
                )

                for model_index, (
                    model,
                    base_price,
                ) in enumerate(products):

                    product_factor = (
                        0.85
                        + model_index * 0.10
                    )

                    base_units = (
                        2.2
                        * brand_factor
                        * product_factor
                    )

                    demand = (
                        base_units
                        * category_curve
                        * year_growth
                    )

                    noise = rng.uniform(
                        0.72,
                        1.28,
                    )

                    expected_units = max(
                        demand * noise,
                        0,
                    )

                    whole_units = int(
                        math.floor(expected_units)
                    )

                    fractional = (
                        expected_units
                        - whole_units
                    )

                    units = whole_units

                    if rng.random() < fractional:
                        units += 1

                    for unit_number in range(units):

                        transaction_id += 1

                        store = STORES[
                            (
                                brand_index
                                + model_index
                                + unit_number
                                + month_index
                            )
                            % len(STORES)
                        ]

                        annual_price_growth = (
                            1
                            + (month_start.year - start_year)
                            * 0.025
                        )

                        sale_price = (
                            base_price
                            * annual_price_growth
                            * rng.uniform(
                                0.94,
                                1.07,
                            )
                        )

                        day = min(
                            3
                            + (
                                transaction_id
                                % 24
                            ),
                            28,
                        )

                        sale_date = date(
                            month_start.year,
                            month_start.month,
                            day,
                        )

                        rows.append(
                            {
                                "transaction_id": (
                                    f"DEMO-{transaction_id}"
                                ),
                                "sale_date": (
                                    sale_date.isoformat()
                                ),
                                "store": store,
                                "manufacturer": manufacturer,
                                "category": category,
                                "model": model,
                                "condition": (
                                    "New"
                                    if rng.random() > 0.12
                                    else "Used"
                                ),
                                "units": 1,
                                "sale_value": round(
                                    sale_price,
                                    2,
                                ),
                            }
                        )

    sales = pd.DataFrame(rows)

    sales["sale_date"] = pd.to_datetime(
        sales["sale_date"]
    )

    sales = sales.sort_values(
        "sale_date"
    ).reset_index(drop=True)

    return sales
