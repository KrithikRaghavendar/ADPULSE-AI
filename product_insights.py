"""
Product-level suggestion layer for the ADPULSE AI dashboard.

Reads the raw campaign dataset (data/dataset.csv) and turns it into
per-SKU, section-wise suggestions:

    marketing  - increase / decrease ad budget
    inventory  - days remaining, restock-before-promoting, frequent
                 out-of-stock, clearance sale
    creative   - video vs poster (image / carousel) performance
    checkout   - abandoned purchases (shipping cost, limited deals)
    combo      - popular + slow product bundles
    affiliate  - affiliate link performance
    sales      - product sales trend

This module is read-only. It does not train, modify or call the ML
pipeline; it only reads its optional output (final report) to show
the engine's decision next to the rule-based suggestions.
"""

import math

import numpy as np
import pandas as pd


# ============================================================
# THRESHOLDS
# ============================================================

RECENT_DAYS = 7

ROAS_INCREASE = 8.0          # 7-day ROAS at or above -> scale ads
ROAS_DECREASE = 4.0          # 7-day ROAS below -> cut ads
BUDGET_INCREASE_PCT = 20
BUDGET_DECREASE_PCT = -30

CRITICAL_DAYS = 3            # days of inventory
LOW_DAYS = 7
OVERSTOCK_DAYS = 30
TARGET_COVER_DAYS = 21       # stock to hold after restock
SUPPLIER_LEAD_DAYS = 7
FREQUENT_OOS_DAYS = 5        # out-of-stock days in the window

CLEARANCE_MIN_DAYS = 10

VIDEO_COMPLETION_LOW = 0.15  # 100% watched / 3-sec views
PAYMENT_FAIL_HIGH = 0.05

VIDEO_TYPES = {"Video", "Influencer Video"}
POSTER_TYPES = {"Image", "Carousel"}


# ============================================================
# LOADING
# ============================================================

def load_dataset(path):

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "margin_pct" in df.columns:
        df["margin_pct"] = pd.to_numeric(
            df["margin_pct"].astype(str).str.rstrip("%"),
            errors="coerce"
        )

    return df.sort_values(["sku", "date"]).reset_index(drop=True)


def _safe_div(a, b):

    return float(a) / float(b) if b else 0.0


def _variant_label(row):

    variant = str(row.get("variant", "") or "")

    return f"{row['product']} {variant}".strip()


# ============================================================
# PER-SKU SUMMARY
# ============================================================

def build_summary(df, final_df=None):
    """One row per SKU with every metric the suggestions need."""

    end = df["date"].max()
    recent_start = end - pd.Timedelta(days=RECENT_DAYS - 1)
    prev_start = recent_start - pd.Timedelta(days=RECENT_DAYS)

    rows = []

    for sku, g in df.groupby("sku"):

        g = g.sort_values("date")
        last = g.iloc[-1]
        recent = g[g["date"] >= recent_start]
        previous = g[(g["date"] >= prev_start) & (g["date"] < recent_start)]

        video = g[g["ad_type"].isin(VIDEO_TYPES)]

        rows.append({
            "sku": sku,
            "product": last["product"],
            "variant": last.get("variant", ""),
            "label": _variant_label(last),
            "category": last.get("category", ""),
            "days": g["date"].nunique(),

            "roas_avg": g["roas"].mean(),
            "roas_7d": recent["roas"].mean(),
            "roas_prev_7d": previous["roas"].mean(),
            "revenue": g["revenue_inr"].sum(min_count=1),
            "ad_spend": g["ad_spend_inr"].sum(min_count=1),
            "profit": g["profit_inr"].sum(min_count=1),
            "profit_7d": recent["profit_inr"].sum(min_count=1),
            "margin_pct": g["margin_pct"].mean() if "margin_pct" in g else float("nan"),

            "units": g["sales_volume"].sum(min_count=1),
            "units_7d": recent["sales_volume"].sum(min_count=1),
            "units_prev_7d": previous["sales_volume"].sum(min_count=1),
            "velocity": g["inventory_velocity_units_per_day"].mean(),
            "sales_growth_pct": g["sales_growth_pct"].mean(),
            "trend": last.get("product_sales_trend", ""),

            "stock": float(last["stock_on_hand"]),
            "days_left": float(last["days_of_inventory_remaining"]),
            "oos_days": int((g["stock_on_hand"] <= 0).sum(min_count=1)),
            "stockout_flag_days": g["stockout_frequency_30d"].sum(min_count=1),

            "product_clicks": g["product_clicks"].sum(min_count=1),
            "add_to_cart": g["add_to_cart"].sum(min_count=1),
            "checkout_started": g["checkout_started"].sum(min_count=1),
            "payment_initiated": g["payment_initiated"].sum(min_count=1),
            "payment_failed": g["payment_failed"].sum(min_count=1),
            "conversions": g["conversions"].sum(min_count=1),

            "ctr": g["click_rate_pct"].mean(),
            "purchase_rate": g["purchase_rate_pct"].mean(),
            "engagement": g["engagement_rate_pct"].mean(),
            "video_3s": video["video_3sec_views"].sum(min_count=1),
            "video_100": video["video_100pct_watched"].sum(min_count=1),

            "affiliate": g["affiliate_link_interactions"].sum(min_count=1),
        })

    summary = pd.DataFrame(rows)

    summary["aov"] = summary["revenue"] / summary["conversions"].where(summary["conversions"] > 0)
    summary["cart_abandon"] = 1 - summary["conversions"] / summary["add_to_cart"].where(summary["add_to_cart"] > 0)
    summary["checkout_drop"] = 1 - summary["payment_initiated"] / summary["checkout_started"].where(summary["checkout_started"] > 0)
    summary["cart_to_checkout_drop"] = 1 - summary["checkout_started"] / summary["add_to_cart"].where(summary["add_to_cart"] > 0)
    summary["payment_fail_rate"] = summary["payment_failed"] / summary["payment_initiated"].where(summary["payment_initiated"] > 0)
    summary["video_completion"] = summary["video_100"] / summary["video_3s"].where(summary["video_3s"] > 0)
    summary["rev_per_affiliate"] = summary["revenue"] / summary["affiliate"].where(summary["affiliate"] > 0)

    # Sales popularity terciles (units sold per day)
    if summary["velocity"].notna().sum() >= 3:
        summary["popularity"] = pd.qcut(
            summary["velocity"].rank(method="first"),
            3,
            labels=["Slow", "Steady", "Popular"]
        ).astype(str).replace("nan", "Steady")
    else:
        summary["popularity"] = "Steady"

    # Optional: decision from the ML pipeline's final report
    if final_df is not None and not final_df.empty and "sku" in final_df.columns:

        keep = [
            c for c in [
                "sku", "current_action", "predicted_next_day_roas",
                "predicted_next_day_profit_inr", "root_cause",
                "learned_decision_confidence"
            ] if c in final_df.columns
        ]

        summary = summary.merge(final_df[keep], on="sku", how="left")

    for section, fn in [
        ("marketing", marketing_advice),
        ("inventory", inventory_advice),
    ]:
        advice = summary.apply(fn, axis=1)
        summary[f"{section}_verdict"] = advice.map(lambda a: a["verdict"])

    summary["restock_first"] = summary.apply(lambda r: inventory_advice(r)["restock_first"], axis=1)
    summary["frequent_oos"] = summary.apply(lambda r: inventory_advice(r)["frequent_oos"], axis=1)
    summary["clearance"] = summary.apply(lambda r: inventory_advice(r)["clearance"], axis=1)

    return summary


# ============================================================
# ADVICE HELPERS
# ============================================================

def _advice(verdict, tone, headline, reasons, actions, **extra):
    """tone: good | warn | bad | info  (drives the colour in the UI)"""

    return {
        "verdict": verdict,
        "tone": tone,
        "headline": headline,
        "reasons": reasons,
        "actions": actions,
        **extra
    }


def inventory_status(row):

    if pd.isna(row["stock"]) or pd.isna(row["days_left"]):
        return "Unknown"

    if row["stock"] <= 0:
        return "Out of stock"

    if row["days_left"] < CRITICAL_DAYS:
        return "Critical"

    if row["days_left"] < LOW_DAYS:
        return "Low"

    if row["days_left"] > OVERSTOCK_DAYS:
        return "Overstock"

    return "Healthy"


def restock_units(row):

    if pd.isna(row["stock"]) or pd.isna(row["velocity"]):
        return 0

    need = row["velocity"] * (TARGET_COVER_DAYS + SUPPLIER_LEAD_DAYS) - row["stock"]

    return max(0, int(math.ceil(need / 10.0) * 10))


# ------------------------------------------------------------
# Marketing: increase or decrease
# ------------------------------------------------------------

def marketing_advice(row):

    if pd.isna(row["roas_7d"]):
        return _advice('No ad data', "info", 'Ad data not uploaded', ['Marketing advice needs ad spend and revenue for this product.'], ['Upload an ad-platform export (Meta, Google, Amazon or TikTok) to unlock budget advice.'], change_pct=0, stock_blocked=False)

    roas7 = row["roas_7d"]
    prev = row["roas_prev_7d"]
    profit = row["profit"]
    status = inventory_status(row)
    stock_blocked = status in ("Out of stock", "Critical")

    momentum = ""

    if pd.notna(prev) and prev > 0:
        change = (roas7 - prev) / prev * 100
        momentum = f"ROAS moved {change:+.0f}% vs the previous 7 days."

    reasons = [
        f"7-day ROAS is {roas7:.2f}× (scale ≥ {ROAS_INCREASE:.0f}×, cut < {ROAS_DECREASE:.0f}×).",
        f"28-day profit is ₹{profit:,.0f} on ₹{row['ad_spend']:,.0f} ad spend.",
    ]

    if momentum:
        reasons.append(momentum)

    if roas7 < ROAS_DECREASE or profit <= 0:

        verdict, pct, tone = "Decrease", BUDGET_DECREASE_PCT, "bad"
        headline = f"Decrease ad budget by {abs(pct)}%"
        actions = [
            f"Cut daily spend by {abs(pct)}% and pause the weakest ad sets.",
            "Move the freed budget to SKUs with ROAS above 8×.",
            "Re-test with a new creative before scaling back up.",
        ]

    elif roas7 >= ROAS_INCREASE and profit > 0:

        verdict, pct, tone = "Increase", BUDGET_INCREASE_PCT, "good"
        headline = f"Increase ad budget by {pct}%"
        actions = [
            f"Raise daily spend by {pct}% in steps, watching ROAS for 3 days.",
            "Duplicate the best-performing ad set to new audiences.",
        ]

    else:

        verdict, pct, tone = "Maintain", 0, "info"
        headline = "Maintain the current ad budget"
        actions = [
            "Keep spend steady and refresh creatives to lift ROAS above 8×.",
            "Shift budget toward the best ad format (see Creative).",
        ]

    if stock_blocked:

        reasons.append(f"Inventory is {status.lower()} — promoting now would waste spend on unavailable stock.")
        actions.insert(0, f"Restock first (~{restock_units(row):,} units) before running promotions.")

        if verdict == "Increase":
            headline = f"Increase budget by {pct}% after restocking"
            tone = "warn"

    if "current_action" in row and pd.notna(row.get("current_action")):
        reasons.append(f"ML decision engine action: {str(row['current_action']).replace('_', ' ').title()}.")

    return _advice(verdict, tone, headline, reasons, actions, change_pct=pct, stock_blocked=stock_blocked)


# ------------------------------------------------------------
# Inventory: days remaining, restock, frequent OOS, clearance
# ------------------------------------------------------------

def inventory_advice(row):

    if pd.isna(row["stock"]) or pd.isna(row["days_left"]):
        return _advice('Unknown', "info", 'Inventory data not uploaded', ['Inventory-based recommendations are unavailable without stock levels.'], ['Upload an inventory / stock export to unlock restock and clearance advice.'], status="Unknown", restock_units=0, frequent_oos=False, oos_days=0, clearance=False, restock_first=False)

    status = inventory_status(row)
    units = restock_units(row)
    oos = max(row["oos_days"], row["stockout_flag_days"])
    frequent_oos = oos >= FREQUENT_OOS_DAYS

    clearance = (
        row["days_left"] >= CLEARANCE_MIN_DAYS
        and (row["profit"] <= 0 or row["roas_7d"] < ROAS_DECREASE or str(row["trend"]) == "Falling")
    ) or status == "Overstock"

    restock_first = status in ("Out of stock", "Critical", "Low")

    reasons = [
        f"Stock on hand: {row['stock']:,.0f} units · selling ~{row['velocity']:.1f} units/day.",
        f"Days of inventory remaining: {row['days_left']:.1f}.",
        f"Out of stock on {oos} of the last {row['days']} days.",
    ]

    actions = []

    if restock_first:

        tone = "bad" if status != "Low" else "warn"
        verdict = "Restock"
        headline = f"{status} — restock ~{units:,} units before promoting ads"
        actions.append(
            f"Order ~{units:,} units to cover {TARGET_COVER_DAYS} days plus a {SUPPLIER_LEAD_DAYS}-day supplier lead time."
        )
        actions.append("Hold or reduce ad spend until stock lands.")

    elif clearance:

        tone = "warn"
        verdict = "Clearance"
        headline = "Slow-moving stock — run a clearance sale"

    else:

        tone = "good"
        verdict = "Healthy"
        headline = f"Healthy stock — {row['days_left']:.0f} days of cover"
        actions.append(f"Reorder when cover drops below {LOW_DAYS} days.")

    if frequent_oos:
        actions.append(
            f"Frequent out-of-stock ({oos} days): raise the safety stock and set an automatic reorder point at "
            f"{int(math.ceil(row['velocity'] * (SUPPLIER_LEAD_DAYS + 3)))} units."
        )

    if clearance:
        actions.append("Clearance: 15–25% off or bundle with a popular product (see Combos).")
        actions.append("Stop prospecting ads; retarget past visitors with the sale price.")
        if verdict != "Clearance":
            reasons.append("Low ROAS / weak profit with spare stock makes this a clearance candidate.")

    return _advice(
        verdict, tone, headline, reasons, actions,
        status=status, restock_units=units, frequent_oos=frequent_oos,
        oos_days=oos, clearance=clearance, restock_first=restock_first
    )


# ------------------------------------------------------------
# Creative: video vs poster
# ------------------------------------------------------------

def creative_table(df, sku):

    g = df[df["sku"] == sku]

    table = (
        g.groupby("ad_type")
        .agg(
            runs=("date", "count"),
            spend=("ad_spend_inr", "sum"),
            revenue=("revenue_inr", "sum"),
            ctr=("click_rate_pct", "mean"),
            engagement=("engagement_rate_pct", "mean"),
            purchase_rate=("purchase_rate_pct", "mean"),
        )
        .reset_index()
    )

    table["roas"] = table["revenue"] / table["spend"].where(table["spend"] > 0)
    table["format"] = table["ad_type"].map(lambda t: "Video" if t in VIDEO_TYPES else "Poster")

    return table.sort_values("roas", ascending=False)


def creative_advice(row, df, portfolio):

    if df.loc[df["sku"] == row["sku"], "ad_type"].dropna().empty:
        return _advice('No creative data', "info", 'Creative format data not uploaded', ['Creative advice needs the ad format (video, image, carousel…) for each day.'], ['Include a format / creative type column in the ad export.'], table=pd.DataFrame(columns=['ad_type', 'runs', 'spend', 'revenue', 'ctr', 'engagement', 'purchase_rate', 'roas', 'format']))

    table = creative_table(df, row["sku"])
    eligible = table[table["runs"] >= 2]

    if eligible.empty:
        eligible = table

    best = eligible.iloc[0]
    worst = eligible.iloc[-1]

    video = table[table["format"] == "Video"]
    poster = table[table["format"] == "Poster"]

    video_roas = _safe_div(video["revenue"].sum(), video["spend"].sum())
    poster_roas = _safe_div(poster["revenue"].sum(), poster["spend"].sum())

    winner = "Video" if video_roas >= poster_roas else "Poster"

    reasons = [
        f"Best format: {best['ad_type']} at {best['roas']:.2f}× ROAS over {int(best['runs'])} days.",
        f"Video ROAS {video_roas:.2f}× vs poster (image/carousel) ROAS {poster_roas:.2f}×.",
    ]

    actions = [
        f"Shift budget toward {best['ad_type']} creatives"
        + (f" and away from {worst['ad_type']} ({worst['roas']:.2f}×)." if worst["ad_type"] != best["ad_type"] else "."),
    ]

    completion = row["video_completion"]

    if pd.notna(completion):

        reasons.append(f"Video completion rate (100% watched / 3-sec views): {completion:.0%}.")

        if completion < VIDEO_COMPLETION_LOW:
            actions.append("Videos lose viewers early — show the product and offer in the first 3 seconds; keep it under 15 s.")

    mismatch = row["ctr"] >= portfolio["ctr"].median() and row["purchase_rate"] < portfolio["purchase_rate"].median()

    if mismatch:
        reasons.append("Click rate is above average but purchase rate is below — the ad promises something the product page doesn't deliver.")
        actions.append("Align the creative with the product page: same visuals, price and claim.")

    if winner == "Video":
        actions.append("Next creative: a short demo or influencer video (before/after, texture, results).")
    else:
        actions.append("Next creative: a clean poster/carousel — product shot, key benefit, price and offer badge.")

    tone = "good" if best["roas"] >= ROAS_INCREASE else "info"

    return _advice(
        f"{winner} wins", tone,
        f"Use more {winner.lower()} ads — {best['ad_type']} performs best",
        reasons, actions, table=table
    )


# ------------------------------------------------------------
# Checkout: abandoned purchases
# ------------------------------------------------------------

def checkout_advice(row, portfolio):

    if pd.isna(row["cart_abandon"]):
        return _advice('No funnel data', "info", 'Funnel data not uploaded', ['Abandoned-cart analysis needs add-to-cart and purchase counts.'], ['Include add-to-cart / checkout columns in the sales or store export.'], abandoned=0)

    abandoned = int(row["add_to_cart"] - row["conversions"])
    cart_abandon = row["cart_abandon"]
    checkout_drop = row["checkout_drop"]
    fail = row["payment_fail_rate"]

    reasons = [
        f"{abandoned:,} of {int(row['add_to_cart']):,} carts ({cart_abandon:.0%}) did not convert.",
        f"{checkout_drop:.0%} of shoppers left between checkout and payment (portfolio median "
        f"{portfolio['checkout_drop'].median():.0%}).",
        f"Payment failure rate: {fail:.1%}.",
    ]

    actions = []
    issues = 0

    if checkout_drop >= portfolio["checkout_drop"].median():

        issues += 1
        threshold = int(math.ceil((row["aov"] or 0) * 1.2 / 50.0) * 50) if pd.notna(row["aov"]) else 0
        actions.append(
            "Check shipping cost: shoppers drop after seeing the final price at checkout. "
            + (f"Test free shipping on orders above ₹{threshold:,} (≈1.2× the ₹{row['aov']:,.0f} average order)." if threshold else "")
        )

    if cart_abandon >= portfolio["cart_abandon"].median():

        issues += 1
        actions.append("Offer a limited-time deal to cart abandoners — e.g. 10% off valid for 24 hours via retargeting / email / WhatsApp.")

    if fail >= PAYMENT_FAIL_HIGH:

        issues += 1
        actions.append("Fix payment friction: add UPI / COD fallback and retry on failure.")

    if not actions:
        actions.append("Checkout is healthier than average — keep monitoring weekly.")

    tone = "bad" if issues >= 2 else "warn" if issues == 1 else "good"
    verdict = "High abandonment" if issues >= 2 else "Some leakage" if issues == 1 else "Healthy"

    return _advice(
        verdict, tone,
        f"{abandoned:,} abandoned carts — {cart_abandon:.0%} abandonment",
        reasons, actions, abandoned=abandoned
    )


# ------------------------------------------------------------
# Combos: popular + slow
# ------------------------------------------------------------

def combo_advice(row, portfolio):

    others = portfolio[
        (portfolio["product"] != row["product"])
        & (~portfolio["sku"].str.contains("COMBO", case=False))
    ]

    popularity = row["popularity"]

    def pick(pool, ascending):

        same = pool[pool["category"] == row["category"]].sort_values("velocity", ascending=ascending)
        rest = pool[pool["category"] != row["category"]].sort_values("velocity", ascending=ascending)

        return pd.concat([same, rest]).drop_duplicates("product").head(2)

    if popularity == "Popular":
        partners = pick(others[others["popularity"] == "Slow"], ascending=True)
        role = "anchor"
        headline = "Popular product — use it to move slow stock"
    else:
        partners = pick(others[others["popularity"] == "Popular"], ascending=False)
        role = "booster"
        headline = (
            "Slow mover — bundle with a popular product"
            if popularity == "Slow"
            else "Steady seller — a bundle can lift order value"
        )

    combos = []

    for _, p in partners.iterrows():

        note = "partner out of stock — launch after restock" if p["stock"] <= 0 else "both ready" if row["stock"] > 0 else "restock this SKU first"

        combos.append({
            "name": f"{row['label']} + {p['label']}",
            "partner_sku": p["sku"],
            "partner_popularity": p["popularity"],
            "partner_velocity": p["velocity"],
            "discount": "10–15% off the bundle",
            "note": note,
        })

    reasons = [
        f"This SKU sells ~{row['velocity']:.1f} units/day — ranked {popularity.lower()} in the portfolio.",
        "Pairing a popular item with a slow one lifts the slow item's sales without discounting the bestseller alone.",
    ]

    actions = [f"Create bundle: {c['name']} ({c['discount']}; {c['note']})." for c in combos]

    if not actions:
        actions = ["No suitable partner found in the current portfolio."]

    return _advice(
        popularity, "info" if role == "anchor" else "warn" if popularity == "Slow" else "good",
        headline, reasons, actions, combos=combos
    )


# ------------------------------------------------------------
# Affiliate performance
# ------------------------------------------------------------

def affiliate_advice(row, portfolio):

    if pd.isna(row["affiliate"]):
        return _advice('No affiliate data', "info", 'Affiliate data not uploaded', ['Affiliate analysis needs affiliate link clicks.'], ['Upload an affiliate report to unlock this section.'])

    interactions = row["affiliate"]
    rpi = row["rev_per_affiliate"]

    low_reach = interactions <= portfolio["affiliate"].quantile(0.33)
    low_value = pd.notna(rpi) and rpi <= portfolio["rev_per_affiliate"].quantile(0.33)
    high_reach = interactions >= portfolio["affiliate"].quantile(0.67)

    reasons = [
        f"{int(interactions):,} affiliate link interactions in {row['days']} days "
        f"(portfolio median {portfolio['affiliate'].median():,.0f}).",
        f"Revenue per affiliate interaction: ₹{(rpi or 0):,.0f} "
        f"(median ₹{portfolio['rev_per_affiliate'].median():,.0f}).",
    ]

    actions = []

    if low_reach:
        actions.append("Low affiliate reach — onboard 3–5 niche micro-influencers with a trackable code.")

    if low_value:
        actions.append("Clicks aren't turning into revenue — give affiliates exclusive codes and a dedicated landing page.")

    if high_reach and not low_value:
        actions.append("Strong affiliate channel — move top affiliates to a higher commission tier and scale.")

    if not actions:
        actions.append("Affiliate performance is average — test a limited-time affiliate-only offer.")

    tone = "good" if high_reach and not low_value else "warn" if (low_reach or low_value) else "info"
    verdict = "Strong" if tone == "good" else "Needs work" if tone == "warn" else "Average"

    return _advice(verdict, tone, f"Affiliate channel: {verdict.lower()}", reasons, actions)


# ------------------------------------------------------------
# Sales trend
# ------------------------------------------------------------

def sales_advice(row):

    if pd.isna(row["units_7d"]) or pd.isna(row["units_prev_7d"]):
        return _advice('No sales data', "info", 'Not enough sales history', ['Sales trend needs at least 14 days of unit sales.'], ['Upload a longer sales history.'], wow_change=0)

    prev = row["units_prev_7d"]
    change = (row["units_7d"] - prev) / prev * 100 if prev else 0.0

    reasons = [
        f"Last 7 days: {int(row['units_7d']):,} units vs {int(prev):,} the week before ({change:+.0f}%).",
        f"Average daily sales growth: {row['sales_growth_pct']:+.1f}% · dataset trend label: {row['trend']}.",
    ]

    if change >= 5:
        verdict, tone, headline = "Rising", "good", f"Sales rising {change:+.0f}% week-on-week"
        actions = ["Make sure stock can keep up before scaling ads."]
    elif change <= -5:
        verdict, tone, headline = "Falling", "bad", f"Sales falling {change:+.0f}% week-on-week"
        actions = ["Refresh creative and check price vs competitors.", "Consider a combo or limited deal to restart demand."]
    else:
        verdict, tone, headline = "Stable", "info", "Sales are stable week-on-week"
        actions = ["Keep the current plan; test one new creative to find upside."]

    if row["stock"] <= 0:
        reasons.append("Note: the SKU is currently out of stock, which caps recent sales.")

    return _advice(verdict, tone, headline, reasons, actions, wow_change=change)


# ============================================================
# ONE CALL FOR THE DIALOG
# ============================================================

def product_report(sku, df, summary):

    row = summary[summary["sku"] == sku].iloc[0]

    return {
        "row": row,
        "marketing": marketing_advice(row),
        "inventory": inventory_advice(row),
        "creative": creative_advice(row, df, summary),
        "checkout": checkout_advice(row, summary),
        "combo": combo_advice(row, summary),
        "affiliate": affiliate_advice(row, summary),
        "sales": sales_advice(row),
    }
