"""
Smart Product Bundling for the ADPULSE AI dashboard.

Pairs a strong "anchor" product with a lagging one so the anchor's demand
lifts the laggard and raises order value. Works at product level (variants
combined) over the last 14 days of data/dataset.csv.

The dataset has no basket / co-purchase data, so the attach rate is a
*target* derived from category fit and relative price — labelled as such
in the UI. Everything else (orders, ROAS, margin, price, stock) is measured.
"""

import pandas as pd


WINDOW_DAYS = 14
MAX_BUNDLES = 5

BASE_ATTACH = 0.10           # target share of anchor orders that add the laggard
SAME_CATEGORY_ATTACH = 0.06
CHEAP_ADDON_ATTACH = 0.04    # laggard costs <= 60% of the anchor
MAX_DISCOUNT = 0.15
MIN_DISCOUNT = 0.05
MARGIN_BUFFER = 0.10         # keep at least 10 points of margin after discount
LOW_STOCK_DAYS = 3

# Products that sell well together even across categories
COMPLEMENTS = {
    frozenset({"Skin", "Body"}): 0.8,
    frozenset({"Skin", "Lips"}): 0.7,
    frozenset({"Body", "Lips"}): 0.6,
    frozenset({"Hair", "Body"}): 0.5,
    frozenset({"Hair", "Skin"}): 0.5,
}


def product_table(df):
    """Product-level metrics over the last WINDOW_DAYS."""

    end = df["date"].max()
    window = df[df["date"] > end - pd.Timedelta(days=WINDOW_DAYS)]
    last = df.sort_values("date").groupby("sku").tail(1)

    g = window.groupby("product")

    p = pd.DataFrame({
        "category": g["category"].first(),
        "variants": g["sku"].nunique(),
        "orders": g["conversions"].sum(),
        "units": g["sales_volume"].sum(),
        "revenue": g["revenue_inr"].sum(),
        "spend": g["ad_spend_inr"].sum(),
        "profit": g["profit_inr"].sum(),
    })

    p["roas"] = p["revenue"] / p["spend"]
    p["price"] = p["revenue"] / p["units"]
    p["margin"] = (window["revenue_inr"] * window["margin_pct"]).groupby(window["product"]).sum() / p["revenue"] / 100
    p["stock"] = last.groupby("product")["stock_on_hand"].sum()
    p["velocity"] = last.groupby("product")["inventory_velocity_units_per_day"].sum()
    p["days_left"] = p["stock"] / p["velocity"].where(p["velocity"] > 0)

    return p.reset_index()


def _affinity(a, b):

    if a == b:
        return 1.0

    return COMPLEMENTS.get(frozenset({a, b}), 0.4)


def smart_bundles(df):

    p = product_table(df)

    # A product that is already a bundle is not re-bundled
    p = p[p["category"] != "Combos"]

    # Only products whose economics are known can be bundled
    p = p.dropna(subset=["roas", "margin", "price", "orders"])

    if len(p) < 2:
        return []

    roas_median = p["roas"].median()
    orders_median = p["orders"].median()
    margin_median = p["margin"].median()

    anchors = p[(p["roas"] >= roas_median) & (p["orders"] >= orders_median * 0.8)]
    laggards = p[
        (p["roas"] < roas_median)
        & ((p["roas"] <= p["roas"].quantile(0.4)) | (p["margin"] < margin_median * 0.8))
    ]

    candidates = []

    for _, a in anchors.iterrows():
        for _, l in laggards.iterrows():

            if a["product"] == l["product"]:
                continue

            affinity = _affinity(a["category"], l["category"])
            strength = (a["roas"] / roas_median) * min(1.0, a["orders"] / orders_median)
            need = roas_median / max(l["roas"], 0.1)

            candidates.append((affinity * strength * need, a, l, affinity))

    candidates.sort(key=lambda c: c[0], reverse=True)

    bundles, used = [], set()

    for score, a, l, affinity in candidates:

        if a["product"] in used or l["product"] in used:
            continue

        used.update({a["product"], l["product"]})
        bundles.append(_describe(a, l, affinity, margin_median))

        if len(bundles) >= MAX_BUNDLES:
            break

    return bundles


def _describe(a, l, affinity, margin_median):

    discount = max(MIN_DISCOUNT, min(MAX_DISCOUNT, l["margin"] - MARGIN_BUFFER))
    discount = round(discount * 20) / 20          # nearest 5%

    attach = BASE_ATTACH
    attach += SAME_CATEGORY_ATTACH if a["category"] == l["category"] else 0
    attach += CHEAP_ADDON_ATTACH if l["price"] <= 0.6 * a["price"] else 0

    extra_units = a["orders"] * attach
    bundle_price = l["price"] * (1 - discount)
    aov_lift = bundle_price / a["price"]
    extra_margin = extra_units * bundle_price * max(0.0, l["margin"] - discount)

    weakness = []

    if l["roas"] < a["roas"]:
        weakness.append(f"ROAS {l['roas']:.1f}×")

    weakness.append(f"margin {l['margin'] * 100:.1f}%")

    if pd.notna(l["days_left"]) and l["days_left"] >= 7:
        weakness.append(f"{l['days_left']:.0f} days of stock")

    cautions = []

    if pd.isna(a["days_left"]) or a["days_left"] < LOW_STOCK_DAYS:
        cautions.append(f"{a['product']} has {0 if pd.isna(a['days_left']) else a['days_left']:.1f} days of stock — restock before promoting this bundle.")

    laggard_short = l["stock"] <= 0 or pd.isna(l["days_left"]) or l["days_left"] < LOW_STOCK_DAYS

    if laggard_short:
        cautions.append(f"{l['product']} has {int(l['stock'])} units left — restock it before launching.")

    if l["margin"] - discount < MARGIN_BUFFER:
        cautions.append(f"{l['product']} margin is thin — discount capped at {discount:.0%}.")

    confidence = (
        55
        + 15 * affinity
        + 10 * min(1.0, a["orders"] / 800)
        + (0 if laggard_short else 8)
        + (7 if not pd.isna(a["days_left"]) and a["days_left"] >= LOW_STOCK_DAYS else 0)
    )

    same = a["category"] == l["category"]
    cheap = l["price"] <= 0.6 * a["price"]

    # Plain-language working for every number on the card (shown on hover)
    calc = {
        "units": [
            f"Target attach rate {attach:.0%} = {BASE_ATTACH:.0%} base"
            + (f" + {SAME_CATEGORY_ATTACH:.0%} same category" if same else "")
            + (f" + {CHEAP_ADDON_ATTACH:.0%} cheap add-on" if cheap else ""),
            f"{int(a['orders']):,} {a['product']} orders (14d) × {attach:.0%} = ~{extra_units:,.0f} extra {l['product']} units",
        ],
        "aov": [
            f"{l['product']} avg price ₹{l['price']:,.0f} × (1 − {discount:.0%} discount) = ₹{bundle_price:,.0f}",
            f"₹{bundle_price:,.0f} ÷ {a['product']} avg price ₹{a['price']:,.0f} = +{aov_lift:.0%} order value",
        ],
        "margin": [
            f"{extra_units:,.0f} units × ₹{bundle_price:,.0f} bundle price × "
            f"({l['margin']:.0%} margin − {discount:.0%} discount) = ₹{extra_margin:,.0f}",
        ],
        "discount": [
            f"{l['product']} margin {l['margin']:.0%} − {MARGIN_BUFFER:.0%} safety buffer, "
            f"capped {MIN_DISCOUNT:.0%}–{MAX_DISCOUNT:.0%}, rounded to 5% = {discount:.0%}",
        ],
        "confidence": [
            f"55 base + 15 × category fit {affinity:.1f} + 10 × anchor demand ({min(1.0, a['orders'] / 800):.2f})"
            + (" + 8 (partner has stock)" if not laggard_short else "")
            + (" + 7 (anchor has stock)" if not pd.isna(a["days_left"]) and a["days_left"] >= LOW_STOCK_DAYS else ""),
        ],
    }

    return {
        "calc": calc,
        "anchor": a["product"],
        "laggard": l["product"],
        "name": f"{a['product']} + {l['product']}",
        "same_category": a["category"] == l["category"],
        "reason": (
            f"{a['product']} is a top performer (ROAS {a['roas']:.1f}×, {int(a['orders']):,} orders) "
            f"while {l['product']} lags ({', '.join(weakness)})."
        ),
        "offer": (
            f"Bundle at {discount:.0%} off {l['product']}; target {attach:.0%} attach rate on {a['product']} orders."
        ),
        "expected": (
            f"~{extra_units:,.0f} extra {l['product']} units per {WINDOW_DAYS} days and "
            f"~{aov_lift:.0%} higher order value on bundled orders."
        ),
        "extra_units": extra_units,
        "extra_margin": extra_margin,
        "aov_lift": aov_lift,
        "discount": discount,
        "attach": attach,
        "cautions": cautions,
        "confidence": int(round(min(95, max(50, confidence)))),
        "anchor_roas": a["roas"],
        "laggard_roas": l["roas"],
    }
