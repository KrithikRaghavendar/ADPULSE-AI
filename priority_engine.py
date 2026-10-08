"""
Signal engine + AI Priorities for the ADPULSE AI dashboard.

Each detector looks for ONE specific, evidenced problem or opportunity.
A product only receives a recommendation when a detector fires, so the
categories differ per product and a healthy product receives none.

Every number comes from data/dataset.csv (via product_insights) or the
ML pipeline's final report. Nothing here trains or changes the ML models.

Not generated because the dataset has no such fields:
competitor pricing, discount history, seasonality (only 28 days).
"""

import math

import pandas as pd

from product_insights import (
    CRITICAL_DAYS,
    CLEARANCE_MIN_DAYS,
    PAYMENT_FAIL_HIGH,
    SUPPLIER_LEAD_DAYS,
    TARGET_COVER_DAYS,
    creative_table,
)


# ============================================================
# THRESHOLDS
# ============================================================

HORIZON_DAYS = 14            # impact is projected over this window

STRONG_CONTRIB_ROAS = 3.0    # ROAS x margin: worth restocking first / scaling
BREAKEVEN_BUFFER = 1.2       # contribution ROAS below this = (near) loss-making
ROAS_DROP = 0.20             # week-on-week ROAS decline to flag
CVR_DROP = 0.15              # week-on-week conversion-rate decline to flag
DEMAND_SHIFT = 0.25          # week-on-week unit change to flag a trend
CREATIVE_GAP = 1.5           # best / worst ad-format ROAS ratio to flag
CREATIVE_MIN_DAYS = 3        # days each format must have run
CREATIVE_SHIFT = 0.30        # share of the weak format's budget to move
SCALE_STEP = 0.20            # budget increase for scale-ready SKUs
SCALE_EFFICIENCY = 0.7       # new spend assumed to return 70% of current ROAS
CART_RECOVERY = 0.10         # share of abandoned carts a deal can win back
PRICE_TEST = 0.05            # price increase tested on thin-margin SKUs
CLEARANCE_RISK = 0.15        # share of tied-up stock value counted as ageing risk

TIER_HIGH = 50_000           # priority score (₹-weighted) thresholds
TIER_MEDIUM = 15_000

URGENCY = {"now": 1.5, "soon": 1.25, "opportunity": 1.0}
REVERSIBILITY = {"easy": 1.1, "normal": 1.0, "hard": 0.9}

CATEGORY_TAB = {
    "INVENTORY": "Inventory",
    "MARKETING": "Marketing",
    "BUDGET": "Marketing",
    "PROFITABILITY": "Marketing",
    "PRICING": "Marketing",
    "CREATIVE": "Ad creative",
    "FUNNEL": "Abandoned carts",
    "CUSTOMER": "Abandoned carts",
    "TREND": "Sales trend",
    "AFFILIATE": "Affiliate",
}


# ============================================================
# HELPERS
# ============================================================

def _pct(a, b):

    return (a - b) / b if b else 0.0


def _clamp(value, low=55, high=97):

    return int(round(max(low, min(high, value))))


def _price(row):

    return row["revenue"] / row["units"] if row["units"] else 0.0


def window_metrics(df):
    """Last 7 days vs the 7 before, plus last-14-day totals, per SKU."""

    end = df["date"].max()
    last7 = df["date"] > end - pd.Timedelta(days=7)
    prev7 = (df["date"] <= end - pd.Timedelta(days=7)) & (df["date"] > end - pd.Timedelta(days=14))
    last14 = df["date"] > end - pd.Timedelta(days=14)

    def totals(frame):
        g = frame.groupby("sku")
        return pd.DataFrame({
            "spend": g["ad_spend_inr"].sum(),
            "revenue": g["revenue_inr"].sum(),
            "profit": g["profit_inr"].sum(),
            "clicks": g["clicks"].sum(),
            "impressions": g["impressions"].sum(),
            "conversions": g["conversions"].sum(),
            "units": g["sales_volume"].sum(),
            "refund_pct": g["refund_rate_pct"].mean(),
        })

    a, b, c = totals(df[last7]), totals(df[prev7]), totals(df[last14])

    zero = df[last14 & (df["stock_on_hand"] <= 0)].groupby("sku").agg(
        zero_days=("date", "count"), zero_spend=("ad_spend_inr", "sum")
    )

    m = pd.DataFrame(index=c.index)
    m["roas_now"] = a["revenue"] / a["spend"]
    m["roas_before"] = b["revenue"] / b["spend"]
    m["cvr_now"] = a["conversions"] / a["clicks"] * 100
    m["cvr_before"] = b["conversions"] / b["clicks"] * 100
    m["cpm_now"] = a["spend"] / a["impressions"] * 1000
    m["cpm_before"] = b["spend"] / b["impressions"] * 1000
    m["cpa_now"] = a["spend"] / a["conversions"]
    m["clicks_now"] = a["clicks"]
    m["units_now"] = a["units"]
    m["units_before"] = b["units"]
    m["spend_7d"] = a["spend"]
    m["spend_14d"] = c["spend"]
    m["revenue_14d"] = c["revenue"]
    m["profit_14d"] = c["profit"]
    m["profit_now"] = a["profit"]
    m["profit_before"] = b["profit"]
    m["refund_pct"] = c["refund_pct"]

    return m.join(zero).fillna({"zero_days": 0, "zero_spend": 0.0})


def _rec(row, category, kind, title, headline, evidence, cause, action,
         impact, impact_label, confidence, urgency, reversibility, calc=None):
    """calc = (impact working lines, confidence working lines) shown on hover in the app."""

    return {
        "sku": row["sku"],
        "label": row["label"],
        "category": category,
        "kind": kind,
        "title": title,
        "headline": headline,
        "evidence": evidence[:4],
        "cause": cause,
        "action": action,
        "impact": max(0.0, float(impact)) if impact is not None else None,
        "impact_label": impact_label,
        "confidence": confidence,
        "urgency": urgency,
        "reversibility": reversibility,
        "tab": CATEGORY_TAB.get(category, "Marketing"),
        "impact_calc": list(calc[0]) if calc else [],
        "conf_calc": list(calc[1]) if calc else [],
    }


def _ml_agrees(row, action):

    return str(row.get("current_action", "")).upper() == action


def _stock_critical(row):

    return row["stock"] <= 0 or row["days_left"] < CRITICAL_DAYS


# ============================================================
# DETECTORS — each returns [] when nothing is wrong
# ============================================================

def _detect_stock(row, m):

    if not _stock_critical(row):
        return []

    contrib = m["roas_now"] * row["margin"]
    unit_contrib = _price(row) * row["margin"]
    velocity = max(row["velocity"], m["units_now"] / 7)
    units = int(math.ceil(max(0.0, velocity * (TARGET_COVER_DAYS + SUPPLIER_LEAD_DAYS) - row["stock"]) / 10.0) * 10)
    zero_days = int(m["zero_days"])
    ml_bonus = 4 if _ml_agrees(row, "PROTECT_INVENTORY") else 0

    stock_line = f"Stock {row['stock']:,.0f} units · {row['days_left']:.1f} days left"
    velocity_line = f"Velocity {velocity:.0f} units/day"

    # Strong unit economics: the fix is stock, not less advertising
    if contrib >= STRONG_CONTRIB_ROAS:

        at_risk = velocity * unit_contrib * SUPPLIER_LEAD_DAYS

        return [_rec(
            row, "INVENTORY", "restock_before_scaling", "Restock before scaling",
            f"Demand ({velocity:.0f}/day) is outrunning stock ({row['days_left']:.1f} days left)",
            [stock_line, velocity_line, f"Contribution ROAS {contrib:.1f}×", f"Zero stock on {zero_days} of last 14 days"],
            "Demand is outrunning stock on a highly profitable SKU.",
            f"Reorder ~{units:,} units now; scale ads only once stock lands.",
            at_risk, f"₹{at_risk:,.0f} contribution at risk over {SUPPLIER_LEAD_DAYS}-day lead time",
            _clamp(78 + 2 * min(zero_days, 7) + ml_bonus),
            "now", "normal",
            calc=(
                [f"Sales velocity {velocity:.1f} units/day × margin per unit ₹{unit_contrib:,.0f} (price ₹{_price(row):,.0f} × {row['margin'] * 100:.0f}%)",
                 f"× {SUPPLIER_LEAD_DAYS}-day supplier lead time = ₹{at_risk:,.0f} of contribution at risk"],
                [f"78 base + 2 × {min(zero_days, 7)} zero-stock days (max 7)" + (" + 4 because the ML engine also chose Protect Inventory" if ml_bonus else ""),
                 "Capped to the 55–97% range"],
            ),
        )]

    # Weaker economics: stop paying for demand that cannot be served
    at_risk = m["spend_7d"] / 7 * SUPPLIER_LEAD_DAYS

    return [_rec(
        row, "INVENTORY", "pause_spend_at_zero_stock", "Stop spend at zero stock",
        f"Ads still running with {row['stock']:,.0f} units in stock",
        [stock_line, velocity_line, f"₹{m['zero_spend']:,.0f} spent on zero-stock days (14d)"],
        f"Ads keep buying demand stock can't serve; contribution ROAS only {contrib:.1f}×.",
        f"Pause or cut ads until restocked; reorder ~{units:,} units.",
        at_risk, f"₹{at_risk:,.0f} ad spend at risk over {SUPPLIER_LEAD_DAYS} days",
        _clamp(80 + 2 * min(zero_days, 7) + ml_bonus),
        "now", "easy",
        calc=(
        [f"Ad spend last 7 days ₹{m['spend_7d']:,.0f} ÷ 7 = ₹{m['spend_7d'] / 7:,.0f}/day",
         f"× {SUPPLIER_LEAD_DAYS} days until new stock lands = ₹{at_risk:,.0f} at risk"],
        [f"80 base + 2 × {min(zero_days, 7)} zero-stock days (max 7)" + (" + 4 because the ML engine also chose Protect Inventory" if ml_bonus else ""),
         "Capped to the 55–97% range"],
        ),
    )]


def _detect_loss(row, m):

    contrib = m["roas_now"] * row["margin"]

    if contrib >= BREAKEVEN_BUFFER and m["profit_14d"] > 0:
        return []

    breakeven = 1 / row["margin"] if row["margin"] else float("nan")
    cut = 0.5 if contrib < 1 else 0.3
    marginal_saving = m["spend_14d"] * cut * max(0.05, 1 - contrib)
    actual_loss = max(0.0, -m["profit_14d"])
    both_weeks = m["profit_now"] <= 0 and m["profit_before"] <= 0

    if actual_loss > marginal_saving:
        saving, impact_label = actual_loss, f"₹{actual_loss:,.0f} lost in the last {HORIZON_DAYS} days"
    else:
        saving, impact_label = marginal_saving, f"₹{marginal_saving:,.0f} saved / {HORIZON_DAYS} days"

    return [_rec(
        row, "PROFITABILITY", "loss_making_spend", "Cut loss-making spend",
        f"Ads return less margin than they cost (ROAS {m['roas_now']:.1f}× vs {breakeven:.1f}× break-even)",
        [
            f"ROAS {m['roas_now']:.2f}× vs break-even {breakeven:.2f}×",
            f"14-day profit ₹{m['profit_14d']:,.0f}",
            f"Margin {row['margin'] * 100:.0f}%",
        ],
        "Each extra ₹ of ads returns less margin than it costs.",
        f"Cut spend ~{int(cut * 100)}% and keep only the best ad set.",
        saving, impact_label,
        _clamp(72 + 25 * min(1.0, (BREAKEVEN_BUFFER - contrib) / 0.6) + (5 if both_weeks else 0)),
        "now", "easy",
        calc=(
        [f"Contribution ROAS = ROAS {m['roas_now']:.2f} × margin {row['margin'] * 100:.0f}% = {contrib:.2f} (below 1.0 loses money)",
         (f"Actual profit over the last 14 days: ₹{m['profit_14d']:,.0f} → ₹{actual_loss:,.0f} lost"
          if actual_loss > marginal_saving else
          f"₹{m['spend_14d']:,.0f} spend (14d) × {int(cut * 100)}% cut × (1 − {contrib:.2f}) = ₹{marginal_saving:,.0f} saved")],
        [f"72 base + 25 × how far contribution ROAS ({contrib:.2f}) sits below the 1.2 safety line" + (" + 5 (lost money both weeks)" if both_weeks else ""),
         "Capped to the 55–97% range"],
        ),
    )]


def _detect_clearance(row, m):

    falling = _pct(m["units_now"], m["units_before"]) <= -DEMAND_SHIFT
    weak = m["roas_now"] * row["margin"] < BREAKEVEN_BUFFER

    if row["stock"] <= 0 or row["days_left"] < CLEARANCE_MIN_DAYS or not (falling or weak):
        return []

    stock_value = row["stock"] * _price(row)

    evidence = [
        f"{row['stock']:,.0f} units · {row['days_left']:.0f} days of cover",
        f"Units {m['units_before']:.0f} → {m['units_now']:.0f} week-on-week",
    ]

    if weak:
        evidence.append(f"Contribution ROAS {m['roas_now'] * row['margin']:.2f}×")

    return [_rec(
        row, "PRICING", "clearance", "Clear slow stock",
        f"{row['days_left']:.0f} days of stock while demand weakens",
        evidence,
        "Stock is building while demand or ad efficiency weakens.",
        "Run a 15–20% markdown or bundle with a bestseller.",
        stock_value * CLEARANCE_RISK, f"₹{stock_value:,.0f} of stock tied up",
        _clamp(62 + (12 if falling else 0) + (12 if weak else 0)),
        "soon", "normal",
        calc=(
        [f"{row['stock']:,.0f} units × avg price ₹{_price(row):,.0f} = ₹{stock_value:,.0f} of stock tied up",
         f"Ranked on {int(CLEARANCE_RISK * 100)}% ageing risk = ₹{stock_value * CLEARANCE_RISK:,.0f}"],
        ["62 base" + (" + 12 (units fell ≥25% week-on-week)" if falling else "") + (" + 12 (ads below break-even)" if weak else "")],
        ),
    )]


def _detect_pricing(row, m, portfolio, clearing):

    median_margin = portfolio["margin"].median()
    contrib = m["roas_now"] * row["margin"]

    # A markdown is already recommended -> don't also suggest a price rise
    if clearing or row["margin"] >= 0.75 * median_margin or contrib >= 2.5:
        return []

    gain = m["revenue_14d"] * PRICE_TEST

    return [_rec(
        row, "PRICING", "thin_margin", "Fix thin margin",
        f"{row['margin'] * 100:.0f}% margin is too thin for paid ads",
        [
            f"Margin {row['margin'] * 100:.0f}% vs portfolio {median_margin * 100:.0f}%",
            f"Avg price ₹{_price(row):,.0f}",
            f"Contribution ROAS {contrib:.1f}×",
        ],
        "Margin is too thin for paid ads to earn much.",
        f"Test a {int(PRICE_TEST * 100)}% price increase or reduce unit cost.",
        gain, f"~₹{gain:,.0f} margin / {HORIZON_DAYS} days if volume holds",
        _clamp(60 + 40 * min(1.0, (0.75 * median_margin - row["margin"]) / 0.2)),
        "opportunity", "hard",
        calc=(
        [f"Revenue last 14 days ₹{m['revenue_14d']:,.0f} × {PRICE_TEST:.0%} price increase = ₹{gain:,.0f}",
         "Assumes order volume holds after the price change"],
        [f"60 + 40 × how far margin {row['margin'] * 100:.0f}% sits below {0.75 * median_margin * 100:.0f}% (75% of the portfolio median)"],
        ),
    )]


def _detect_roas_decline(row, m):

    drop = -_pct(m["roas_now"], m["roas_before"])

    # During a stock-out the decline is supply-driven; inventory covers it
    if drop < ROAS_DROP or _stock_critical(row):
        return []

    cpm_change = _pct(m["cpm_now"], m["cpm_before"])
    cvr_change = _pct(m["cvr_now"], m["cvr_before"])

    if cpm_change >= 0.10:
        cause, action = "Rising auction costs (CPM) are eating returns.", "Narrow targeting and refresh creative to bring CPM down."
    elif cvr_change <= -0.10:
        cause, action = "Fewer clicks are converting to orders.", "Check product page, price and offer; A/B test the landing page."
    else:
        cause, action = "Revenue per click fell.", "Refresh creative and audiences; re-check pricing."

    lost = (m["roas_before"] - m["roas_now"]) * m["spend_7d"] * 2

    return [_rec(
        row, "MARKETING", "roas_decline", "Reverse ROAS decline",
        f"ROAS fell {drop:.0%} week-on-week",
        [
            f"ROAS {m['roas_before']:.1f}× → {m['roas_now']:.1f}× ({-drop:+.0%})",
            f"CPM {cpm_change:+.0%} · CVR {cvr_change:+.0%}",
            f"7-day spend ₹{m['spend_7d']:,.0f}",
        ],
        cause, action,
        lost, f"~₹{lost:,.0f} revenue / {HORIZON_DAYS} days vs the prior ROAS",
        _clamp(60 + 25 * min(1.0, drop / 0.4) + (8 if abs(cpm_change) >= 0.1 or cvr_change <= -0.1 else 0)),
        "soon", "easy",
        calc=(
        [f"ROAS gap {m['roas_before']:.2f} − {m['roas_now']:.2f} = {round(m['roas_before'], 2) - round(m['roas_now'], 2):.2f}",
         f"× ₹{m['spend_7d']:,.0f} weekly spend × 2 weeks = ₹{lost:,.0f} revenue at the old ROAS"],
        [f"60 + 25 × drop size ({drop:.0%} of a 40% max)" + (" + 8 (explained by CPM or CVR change)" if abs(cpm_change) >= 0.1 or cvr_change <= -0.1 else "")],
        ),
    )]


def _detect_cvr_decline(row, m, roas_flagged):

    change = _pct(m["cvr_now"], m["cvr_before"])

    # Already explained inside the ROAS-decline diagnosis
    if roas_flagged or change > -CVR_DROP or _stock_critical(row):
        return []

    aov = row["aov"] if pd.notna(row["aov"]) else 0.0
    lost = (m["cvr_before"] - m["cvr_now"]) / 100 * m["clicks_now"] * aov * 2

    return [_rec(
        row, "FUNNEL", "cvr_decline", "Investigate conversion drop",
        f"Conversion rate fell {-change:.0%} week-on-week",
        [
            f"CVR {m['cvr_before']:.2f}% → {m['cvr_now']:.2f}% ({change:+.0%})",
            f"CPA ₹{m['cpa_now']:,.0f}",
            f"Cart abandonment {row['cart_abandon']:.0%}",
        ],
        "Traffic holds but fewer visitors buy.",
        "Audit product page, reviews, price and checkout for this SKU.",
        lost, f"~₹{lost:,.0f} revenue / {HORIZON_DAYS} days vs the prior CVR",
        _clamp(62 + 25 * min(1.0, -change / 0.35)),
        "soon", "easy",
        calc=(
        [f"CVR gap {m['cvr_before']:.2f}% − {m['cvr_now']:.2f}% × {m['clicks_now']:,.0f} weekly clicks",
         f"× avg order ₹{aov:,.0f} × 2 weeks = ₹{lost:,.0f}"],
        [f"62 + 25 × drop size ({-change:.0%} of a 35% max)"],
        ),
    )]


def _detect_creative(row, df):

    table = creative_table(df, row["sku"])
    table = table[table["runs"] >= CREATIVE_MIN_DAYS].dropna(subset=["roas"])

    if len(table) < 2:
        return []

    best, worst = table.iloc[0], table.iloc[-1]

    if worst["roas"] <= 0 or best["roas"] / worst["roas"] < CREATIVE_GAP:
        return []

    ratio = best["roas"] / worst["roas"]
    worst_spend_14d = worst["spend"] * HORIZON_DAYS / row["days"]
    gain = CREATIVE_SHIFT * worst_spend_14d * (best["roas"] - worst["roas"])
    min_days = int(min(best["runs"], worst["runs"]))

    return [_rec(
        row, "CREATIVE", "creative_shift", "Shift creative budget",
        f"{worst['ad_type']} underperforms {best['ad_type']}",
        [
            f"{best['ad_type']} ROAS {best['roas']:.1f}× ({int(best['runs'])} days)",
            f"{worst['ad_type']} ROAS {worst['roas']:.1f}× ({int(worst['runs'])} days)",
            f"{worst['ad_type']} spend ₹{worst['spend']:,.0f} (28d)",
        ],
        f"{best['ad_type']} converts better than {worst['ad_type']} for this product.",
        f"Shift ~{int(CREATIVE_SHIFT * 100)}% of {worst['ad_type']} budget to {best['ad_type']}.",
        gain, f"~+₹{gain:,.0f} revenue / {HORIZON_DAYS} days",
        _clamp(55 + 3 * min(min_days, 8) + 15 * min(1.0, (ratio - CREATIVE_GAP) / 1.5)),
        "opportunity", "easy",
        calc=(
        [f"{worst['ad_type']} spend scaled to 14 days: ₹{worst_spend_14d:,.0f} × {CREATIVE_SHIFT:.0%} moved",
         f"× ROAS gap ({best['roas']:.1f} − {worst['roas']:.1f}) = ₹{gain:,.0f} extra revenue"],
        [f"55 + 3 × {min(min_days, 8)} days of evidence per format + 15 × gap strength ({ratio:.1f}× vs the 1.5× trigger)"],
        ),
    )]


def _detect_scale(row, m):

    contrib = m["roas_now"] * row["margin"]
    declining = _pct(m["roas_now"], m["roas_before"]) <= -ROAS_DROP

    if contrib < STRONG_CONTRIB_ROAS or row["days_left"] < 10 or declining or m["profit_14d"] <= 0:
        return []

    extra = m["spend_14d"] * SCALE_STEP
    gain = extra * (contrib * SCALE_EFFICIENCY - 1)

    return [_rec(
        row, "BUDGET", "scale_winner", "Scale a profitable winner",
        f"Profitable at {contrib:.1f}× contribution ROAS with {row['days_left']:.0f} days of stock",
        [
            f"ROAS {m['roas_now']:.1f}× · contribution {contrib:.1f}×",
            f"{row['days_left']:.0f} days of stock",
            f"14-day profit ₹{m['profit_14d']:,.0f}",
        ],
        "Strong unit economics and enough stock to absorb more demand.",
        f"Raise budget {int(SCALE_STEP * 100)}% in steps; watch ROAS for 3 days.",
        gain, f"~+₹{gain:,.0f} profit / {HORIZON_DAYS} days (at {int(SCALE_EFFICIENCY * 100)}% efficiency)",
        _clamp(70 + 4 * min(contrib - STRONG_CONTRIB_ROAS, 5) + (5 if _ml_agrees(row, "INCREASE_BUDGET") else 0)),
        "opportunity", "easy",
        calc=(
        [f"Extra spend ₹{m['spend_14d']:,.0f} × {SCALE_STEP:.0%} = ₹{extra:,.0f}",
         f"× (contribution ROAS {contrib:.1f} × {SCALE_EFFICIENCY} efficiency − 1) = ₹{gain:,.0f} profit"],
        [f"70 + 4 × contribution ROAS above 3× ({min(contrib - STRONG_CONTRIB_ROAS, 5):.1f})" + (" + 5 (ML engine also says increase)" if _ml_agrees(row, "INCREASE_BUDGET") else "")],
        ),
    )]


def _detect_demand(row, m):

    # During stock-outs the movement is supply-driven; inventory handles it
    if _stock_critical(row):
        return []

    change = _pct(m["units_now"], m["units_before"])

    if abs(change) < DEMAND_SHIFT:
        return []

    delta_rev = abs(m["units_now"] - m["units_before"]) * _price(row) * 2

    if change > 0:

        cover = row["stock"] / (m["units_now"] / 7) if m["units_now"] else float("inf")

        return [_rec(
            row, "TREND", "demand_surge", "Ride the demand surge",
            f"Units up {change:.0%} week-on-week",
            [
                f"Units {m['units_before']:.0f} → {m['units_now']:.0f} ({change:+.0%} WoW)",
                f"Cover at the new pace: {cover:.0f} days",
            ],
            "Demand is accelerating.",
            "Reorder at the new run-rate; lift budget while ROAS holds.",
            delta_rev, f"~₹{delta_rev:,.0f} revenue / {HORIZON_DAYS} days if the pace holds",
            _clamp(60 + 30 * min(1.0, change / 0.6)),
            "opportunity", "normal",
            calc=(
                [f"|{m['units_now']:.0f} − {m['units_before']:.0f}| units/week × avg price ₹{_price(row):,.0f} × 2 weeks = ₹{delta_rev:,.0f}"],
                [f"60 + 30 × change size ({change:.0%} of a 60% max)"],
            ),
        )]

    return [_rec(
        row, "TREND", "demand_drop", "Demand slowing",
        f"Units down {-change:.0%} week-on-week",
        [
            f"Units {m['units_before']:.0f} → {m['units_now']:.0f} ({change:+.0%} WoW)",
            f"{row['days_left']:.0f} days of stock on hand",
        ],
        "Fewer orders while stock is available.",
        "Refresh creative or run a limited offer before stock ages.",
        delta_rev, f"~₹{delta_rev:,.0f} revenue / {HORIZON_DAYS} days at risk",
        _clamp(60 + 30 * min(1.0, -change / 0.6)),
        "soon", "easy",
        calc=(
        [f"|{m['units_now']:.0f} − {m['units_before']:.0f}| units/week × avg price ₹{_price(row):,.0f} × 2 weeks = ₹{delta_rev:,.0f}"],
        [f"60 + 30 × change size ({-change:.0%} of a 60% max)"],
        ),
    )]


def _detect_checkout(row, portfolio):

    # Abandonment during a stock-out is caused by the stock-out
    if _stock_critical(row):
        return []

    flags = []

    if row["cart_abandon"] >= portfolio["cart_abandon"].quantile(0.80):
        flags.append(("cart", f"Cart abandonment {row['cart_abandon']:.0%} (top 20%)"))

    if row["checkout_drop"] >= portfolio["checkout_drop"].quantile(0.80):
        flags.append(("checkout", f"Checkout→payment drop {row['checkout_drop']:.0%} (top 20%)"))

    if row["payment_fail_rate"] >= max(PAYMENT_FAIL_HIGH, portfolio["payment_fail_rate"].quantile(0.80)):
        flags.append(("payment", f"Payment failures {row['payment_fail_rate']:.1%} (top 20%)"))

    kinds = {k for k, _ in flags}

    # Need the headline metric or two independent funnel signals
    if "cart" not in kinds and len(flags) < 2:
        return []
    abandoned = row["add_to_cart"] - row["conversions"]
    gain = abandoned * HORIZON_DAYS / row["days"] * CART_RECOVERY * (row["aov"] or 0)

    if "payment" in kinds:
        cause, action = "Payments fail more often than on other SKUs.", "Add UPI/COD fallback and auto-retry failed payments."
    elif "checkout" in kinds:
        cause, action = "Shoppers leave once the final price appears.", "Check shipping cost at checkout; test a free-shipping threshold."
    else:
        cause, action = "Shoppers add to cart but hesitate to buy.", "Send a 24-hour 10%-off deal to cart abandoners."

    return [_rec(
        row, "FUNNEL", "cart_abandonment", "Reduce cart abandonment",
        f"{row['cart_abandon']:.0%} of carts abandoned",
        [text for _, text in flags] + [f"{int(abandoned):,} carts not converted (28d)"],
        cause, action,
        gain, f"~₹{gain:,.0f} revenue / {HORIZON_DAYS} days at {int(CART_RECOVERY * 100)}% recovery",
        _clamp(60 + 8 * len(flags)),
        "soon", "easy",
        calc=(
        [f"{int(abandoned):,} unconverted carts × 14/{int(row['days'])} days × {CART_RECOVERY:.0%} recovered",
         f"× avg order ₹{(row['aov'] or 0):,.0f} = ₹{gain:,.0f}"],
        [f"60 + 8 × {len(flags)} funnel signal(s) in the worst 20% of products"],
        ),
    )]


def _detect_refunds(row, m, portfolio):

    median = portfolio["refund_pct"].median()
    spread = portfolio["refund_pct"].std()

    if not spread or (m["refund_pct"] - median) / spread < 1.5:
        return []

    excess = (m["refund_pct"] - median) / 100 * m["revenue_14d"]

    return [_rec(
        row, "CUSTOMER", "high_refunds", "High refunds eroding profit",
        f"Refunds at {m['refund_pct']:.1f}% vs {median:.1f}% portfolio",
        [
            f"Refund rate {m['refund_pct']:.1f}% vs portfolio {median:.1f}%",
            f"14-day revenue ₹{m['revenue_14d']:,.0f}",
        ],
        "More buyers than usual are unhappy with the product.",
        "Review complaints; fix any description or expectation mismatch.",
        excess, f"~₹{excess:,.0f} refunds above normal / {HORIZON_DAYS} days",
        _clamp(60 + 10 * min(3.0, (m["refund_pct"] - median) / spread)),
        "soon", "normal",
        calc=(
        [f"({m['refund_pct']:.1f}% − {median:.1f}% median) × ₹{m['revenue_14d']:,.0f} revenue (14d) = ₹{excess:,.0f}"],
        [f"60 + 10 × standard deviations above the median ({(m['refund_pct'] - median) / spread:.1f}, max 3)"],
        ),
    )]


def _detect_affiliate(row, portfolio):

    if not row["affiliate"]:
        return []

    # Orders per interaction is price-neutral (cheap SKUs aren't penalised)
    upi = portfolio["units"] / portfolio["affiliate"].where(portfolio["affiliate"] > 0)
    own = row["units"] / row["affiliate"]
    median_upi = upi.median()

    busy = row["affiliate"] >= portfolio["affiliate"].median()
    weak = own <= upi.quantile(0.20)

    if not (busy and weak) or not median_upi:
        return []

    gap = median_upi - own
    gain = gap * row["affiliate"] * HORIZON_DAYS / row["days"] * 0.25 * _price(row)

    return [_rec(
        row, "AFFILIATE", "affiliate_leak", "Fix affiliate conversion",
        "Affiliate traffic converts below the portfolio",
        [
            f"{int(row['affiliate']):,} affiliate interactions (above median)",
            f"{own:.2f} units per interaction vs {median_upi:.2f} median",
        ],
        "Affiliates send traffic that doesn't buy.",
        "Give affiliates exclusive codes and a dedicated landing page.",
        gain, f"~₹{gain:,.0f} revenue / {HORIZON_DAYS} days closing 25% of the gap",
        _clamp(58 + 20 * min(1.0, gap / median_upi)),
        "opportunity", "normal",
        calc=(
        [f"Gap {median_upi:.2f} − {own:.2f} units per interaction × {int(row['affiliate']):,} interactions",
         f"× 14/{int(row['days'])} days × 25% of gap closed × ₹{_price(row):,.0f} = ₹{gain:,.0f}"],
        [f"58 + 20 × relative gap to the median ({gap / median_upi:.0%})"],
        ),
    )]


# ============================================================
# PUBLIC API
# ============================================================

PLATFORM_GAP = 1.4            # best / worst platform ROAS ratio to flag
PLATFORM_MIN_SPEND = 1000     # ₹ per platform over 14 days to count
PLATFORM_SHIFT = 0.25         # share of the weaker platform's budget to move
LEARNING_WEIGHT = 20          # max confidence points added/removed by measured outcomes


def _detect_platform_shift(row, platforms):
    """Cross-platform decision: move budget from the weaker ad platform to the stronger one."""

    if platforms is None or platforms.empty or _stock_critical(row):
        return []

    p = platforms[platforms["entity"] == row["sku"]]

    if p.empty:
        return []

    end = p["date"].max()
    p = p[p["date"] > end - pd.Timedelta(days=HORIZON_DAYS)]
    by = p.groupby("platform")[["ad_spend_inr", "ad_revenue"]].sum()
    by = by[by["ad_spend_inr"] >= PLATFORM_MIN_SPEND]

    if len(by) < 2 or by["ad_revenue"].isna().all():
        return []

    by["roas"] = by["ad_revenue"] / by["ad_spend_inr"]
    by = by.sort_values("roas", ascending=False)
    best, worst = by.index[0], by.index[-1]
    rb, rw = by.loc[best, "roas"], by.loc[worst, "roas"]

    if rw <= 0 or rb / rw < PLATFORM_GAP:
        return []

    moved = by.loc[worst, "ad_spend_inr"] * PLATFORM_SHIFT
    gain = moved * (rb * SCALE_EFFICIENCY - rw)

    return [_rec(
        row, "BUDGET", "platform_shift", "Shift budget across platforms",
        f"{worst} returns {rw:.1f}× vs {best} {rb:.1f}× on this product",
        [f"{best} ROAS {rb:.1f}× (₹{by.loc[best, 'ad_spend_inr']:,.0f} spend)",
         f"{worst} ROAS {rw:.1f}× (₹{by.loc[worst, 'ad_spend_inr']:,.0f} spend)",
         "Last 14 days, attributed revenue"],
        f"The same product converts far better on {best} than on {worst}.",
        f"Move ~{int(PLATFORM_SHIFT * 100)}% of the {worst} budget to {best}; keep total spend flat.",
        gain, f"Est. +₹{gain:,.0f} revenue / {HORIZON_DAYS} days",
        _clamp(60 + 20 * min(1.0, (rb / rw - PLATFORM_GAP) / 1.5) + (8 if len(p["date"].unique()) >= 10 else 0)),
        "opportunity", "easy",
        calc=(
            [f"₹{by.loc[worst, 'ad_spend_inr']:,.0f} {worst} spend × {PLATFORM_SHIFT:.0%} moved = ₹{moved:,.0f}",
             f"× ({rb:.2f} {best} ROAS × {SCALE_EFFICIENCY} efficiency − {rw:.2f} {worst} ROAS) = ₹{gain:,.0f}"],
            [f"60 + 20 × gap strength ({rb / rw:.1f}× vs the {PLATFORM_GAP}× trigger) + 8 if ≥ 10 days of data"],
        ),
    )]


def detect_anomalies(df, recent_days=3, threshold=3.0):
    """Unusual moves in the last few days vs each SKU's own history (robust z-score on median / MAD)."""

    metrics = {"roas": "ROAS", "ad_spend_inr": "Ad spend", "revenue_inr": "Revenue", "sales_volume": "Units sold"}
    end = df["date"].max()
    cut = end - pd.Timedelta(days=recent_days)
    rows = []

    for sku, g in df.groupby("sku"):

        before, recent = g[g["date"] <= cut], g[g["date"] > cut]

        if len(before) < 7 or recent.empty:
            continue

        for column, label in metrics.items():

            if column not in g or g[column].notna().sum() < 10:
                continue

            history = before[column].dropna()
            median = history.median()
            mad = (history - median).abs().median() * 1.4826
            now = recent[column].mean()

            if not mad or pd.isna(now) or not median:
                continue

            z = (now - median) / mad
            change = (now - median) / abs(median)

            if abs(z) >= threshold and abs(change) >= 0.25:
                rows.append({
                    "sku": sku, "product": g["product"].iloc[0], "metric": label,
                    "direction": "up" if z > 0 else "down", "change": change, "z": z,
                    "now": now, "usual": median,
                })

    return pd.DataFrame(rows).sort_values("z", key=abs, ascending=False) if rows else pd.DataFrame(
        columns=["sku", "product", "metric", "direction", "change", "z", "now", "usual"])


def detect_recommendations(df, summary, platforms=None, learning=None):
    """Every evidenced recommendation across the portfolio, scored and tiered.

    A check only runs when the data it needs is present — uploads without
    inventory get no inventory advice instead of a guess.
    learning: {kind: (worked, measured)} from approved actions whose outcome was measured.
    """

    windows = window_metrics(df)

    portfolio = summary.copy()
    portfolio["margin"] = portfolio["sku"].map(df.groupby("sku")["margin_pct"].mean() / 100)
    portfolio["refund_pct"] = portfolio["sku"].map(windows["refund_pct"])

    recs = []

    for _, row in portfolio.iterrows():

        m = windows.loc[row["sku"]]

        def has(*names):
            return all(pd.notna(row.get(n)) if n in row else pd.notna(m.get(n)) for n in names)

        stock_known = has("stock", "days_left")
        ads_known = has("roas_now", "spend_14d")
        margin_known = has("margin") and ads_known and has("profit_14d")

        clearance = _detect_clearance(row, m) if stock_known and has("units_now", "units_before") else []
        roas = _detect_roas_decline(row, m) if has("roas_now", "roas_before", "cpm_now", "cpm_before") else []

        recs += _detect_stock(row, m) if stock_known else []
        recs += _detect_loss(row, m) if margin_known else []
        recs += clearance
        recs += _detect_pricing(row, m, portfolio, bool(clearance)) if margin_known else []
        recs += roas
        recs += _detect_cvr_decline(row, m, bool(roas)) if has("cvr_now", "cvr_before") else []
        recs += _detect_creative(row, df)
        recs += _detect_scale(row, m) if margin_known and stock_known else []
        recs += _detect_demand(row, m) if stock_known and has("units_now", "units_before") else []
        recs += _detect_checkout(row, portfolio) if has("cart_abandon") else []
        recs += _detect_refunds(row, m, portfolio) if has("refund_pct") else []
        recs += _detect_affiliate(row, portfolio) if has("affiliate") else []
        recs += _detect_platform_shift(row, platforms)

    recs = pd.DataFrame(recs)

    if recs.empty:
        return recs

    # Safety net: never show a recommendation built on missing numbers
    recs = recs[recs["impact"].notna()].copy()

    # Clearance evidence (stock vs demand) lives in the Inventory tab
    recs.loc[recs["kind"] == "clearance", "tab"] = "Inventory"

    # Closed loop: measured outcomes of approved actions move confidence up or down
    recs["learning_note"] = ""

    if learning:
        for i, r in recs.iterrows():
            worked, measured = learning.get(r["kind"], (0, 0))
            if measured:
                rate = worked / measured
                shift = int(round((rate - 0.5) * 2 * LEARNING_WEIGHT * min(1.0, measured / 3)))
                new = _clamp(r["confidence"] + shift)
                note = f"{shift:+d} from your {measured} measured outcome(s) of this action ({worked} worked)"
                recs.at[i, "confidence"] = new
                recs.at[i, "conf_calc"] = list(r["conf_calc"]) + [note]
                recs.at[i, "learning_note"] = note

    # Priority = ₹ impact x confidence x urgency x ease of action.
    # Impact dominates: ₹50k at 85% outranks ₹500 at 99%.
    recs["score"] = (
        recs["impact"].fillna(0)
        * recs["confidence"] / 100
        * recs["urgency"].map(URGENCY)
        * recs["reversibility"].map(REVERSIBILITY)
    )

    recs["score_calc"] = recs.apply(
        lambda r: [
            f"₹{(r['impact'] or 0):,.0f} impact × {r['confidence']}% confidence × urgency {URGENCY[r['urgency']]} "
            f"({r['urgency']}) × ease {REVERSIBILITY[r['reversibility']]} ({r['reversibility']}) = score ₹{r['score']:,.0f}",
            f"HIGH ≥ ₹{TIER_HIGH:,} · MEDIUM ≥ ₹{TIER_MEDIUM:,} · otherwise LOW",
        ],
        axis=1,
    )

    recs["tier"] = recs["score"].map(
        lambda s: "HIGH" if s >= TIER_HIGH else "MEDIUM" if s >= TIER_MEDIUM else "LOW"
    )

    return recs.sort_values("score", ascending=False).reset_index(drop=True)


def select_priorities(recs, limit=6, per_sku=2, per_kind=2):
    """Top recommendations with variety — no single product or issue type dominates.

    Recommendations skipped because their issue type is already on screen are
    rolled up into a '+N more products' note on that type's last card.
    """

    if recs.empty:
        return []

    chosen, picked, sku_count, kind_count = [], set(), {}, {}

    for index, rec in recs.iterrows():

        if len(chosen) >= limit:
            break

        if kind_count.get(rec["kind"], 0) >= per_kind or sku_count.get(rec["sku"], 0) >= per_sku:
            continue

        chosen.append(rec.to_dict())
        picked.add(index)
        sku_count[rec["sku"]] = sku_count.get(rec["sku"], 0) + 1
        kind_count[rec["kind"]] = kind_count.get(rec["kind"], 0) + 1

    last_of_kind = {card["kind"]: i for i, card in enumerate(chosen)}

    for i, card in enumerate(chosen):

        extra = recs[(recs["kind"] == card["kind"]) & (~recs.index.isin(picked))]

        if last_of_kind[card["kind"]] != i:
            extra = extra.iloc[0:0]

        card["more_count"] = len(extra)
        card["more_impact"] = float(extra["impact"].fillna(0).sum())

    return chosen


def product_recommendations(recs, sku):

    if recs.empty:
        return recs

    return recs[recs["sku"] == sku]
