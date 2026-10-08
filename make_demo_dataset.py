"""
Create a SIMULATED second company so the dataset switcher can be demonstrated.

"UrbanBrew Coffee" is a fictional D2C brand. Its 28-day dataset is generated
from the Glowroots dataset's structure: every SKU gets a new product identity,
its own price / demand / ad-spend levels, a shifted timeline and fresh noise,
and every derived column is recomputed with the dataset's own definitions
(ROAS = revenue / spend, CTR = clicks / impressions, next-day columns = the
following day's values, action success = next-day ROAS went up, ...).

It is clearly labelled as simulated in the app. Replace it with a real
company's CSV through the Datasets panel whenever you have one.

    python make_demo_dataset.py
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "dataset.csv"
TARGET_DIR = ROOT / "companies" / "urbanbrew"
SEED = 7

# Glowroots product -> (new product, category, [variants], SKU code)
CATALOG = {
    "Body Lotion": ("Cold Brew Concentrate", "Coffee", ["500 ml", "1 L"], "COLDBREW"),
    "Glow Combo": ("Brew Starter Kit", "Gifts", ["Mini kit", "Full kit"], "KIT"),
    "Ubtan Face Wash": ("Arabica Whole Beans", "Coffee", ["250 g", "500 g"], "ARABICA"),
    "Onion Hair Oil": ("South Indian Filter Coffee", "Coffee", ["250 g", "500 g"], "FILTER"),
    "Lip Balm": ("Instant Coffee Sachets", "Instant", ["Hazelnut", "Classic", "Mocha"], "SACHET"),
    "Aloe Moisturizer": ("Masala Chai Blend", "Tea", ["100 g", "250 g"], "CHAI"),
    "Retinol Serum": ("Pour-Over Kit", "Equipment", ["Single", "Double"], "POUROVER"),
    "Onion Shampoo": ("French Press", "Equipment", ["350 ml", "1 L"], "PRESS"),
    "Sunscreen SPF 50": ("Ceramic Mug Set", "Gifts", ["Set of 2", "Set of 4"], "MUG"),
    "Vitamin C Serum": ("Espresso Roast Beans", "Coffee", ["250 g", "1 kg"], "ESPRESSO"),
}

COUNTS = ["impressions", "clicks", "product_page_views_from_ads", "product_clicks",
          "video_3sec_views", "video_25pct_watched", "video_50pct_watched",
          "video_75pct_watched", "video_100pct_watched", "likes", "comments", "shares", "saves"]

FUNNEL = ["add_to_cart", "checkout_started", "payment_initiated", "conversions"]


def build():

    rng = np.random.default_rng(SEED)
    src = pd.read_csv(SOURCE)
    src["margin"] = src["margin_pct"].str.rstrip("%").astype(float)

    frames = []

    for sku, g in src.groupby("sku", sort=False):

        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)

        product, category, variants, code = CATALOG[g["product"].iloc[0]]
        siblings = sorted(src.loc[src["product"] == g["product"].iloc[0], "sku"].unique())
        variant = variants[siblings.index(sku) % len(variants)]

        # Per-SKU personality: different economics from the source SKU
        spend_f = rng.uniform(0.6, 1.5)
        demand_f = rng.uniform(0.6, 1.4)
        price_f = rng.uniform(0.5, 1.6)
        margin_shift = rng.uniform(-10, 10)
        shift = int(rng.integers(0, n))                 # different timeline position

        def series(column):
            return np.roll(g[column].to_numpy(dtype=float), shift)

        out = pd.DataFrame({"date": g["date"]})
        out["sku"] = f"UB-{code}-{variant.upper().replace(' ', '')}"
        out["product_code"] = f"UB-{code}"
        out["product"] = product
        out["category"] = category
        out["variant"] = variant
        out["ad_type"] = np.roll(g["ad_type"].to_numpy(), shift)

        noise = lambda: rng.normal(1.0, 0.08, n).clip(0.75, 1.3)

        for column in COUNTS:
            out[column] = np.round(series(column) * spend_f * noise()).astype(int)

        out["clicks"] = np.maximum(out["clicks"], 1)
        out["product_clicks"] = out["clicks"]
        out["ad_spend_inr"] = (series("ad_spend_inr") * spend_f * noise()).round(2)

        demand = demand_f * noise()
        for column in FUNNEL:
            out[column] = np.round(series(column) * spend_f * demand).astype(int)

        out["payment_failed"] = np.minimum(np.round(series("payment_failed") * spend_f).astype(int), out["payment_initiated"])
        out["sales_volume"] = out["conversions"]
        out["repeat_purchaser_count"] = np.round(series("repeat_purchaser_count") * demand_f).astype(int)

        unit_price = (series("revenue_inr") / np.maximum(series("sales_volume"), 1)) * price_f
        margin = np.clip(g["margin"].iloc[0] + margin_shift, 15, 65)

        out["revenue_inr"] = (out["sales_volume"] * unit_price).round(2)
        out["margin_pct"] = f"{margin:.0f}%"
        out["profit_inr"] = (out["revenue_inr"] * margin / 100 - out["ad_spend_inr"]).round(2)
        out["roas"] = (out["revenue_inr"] / out["ad_spend_inr"]).round(3)

        out["click_rate_pct"] = (out["clicks"] / out["impressions"] * 100).round(3)
        out["add_to_cart_rate_pct"] = (out["add_to_cart"] / out["product_page_views_from_ads"].clip(lower=1) * 100).round(3)
        out["purchase_rate_pct"] = (out["conversions"] / out["clicks"] * 100).round(3)
        out["payment_failed_rate_pct"] = (out["payment_failed"] / out["payment_initiated"].clip(lower=1) * 100).round(3)
        out["engagement_rate_pct"] = ((out["likes"] + out["comments"] + out["shares"] + out["saves"]) / out["impressions"] * 100).round(3)
        out["affiliate_link_interactions"] = np.round(series("affiliate_link_interactions") * rng.uniform(0.5, 1.6)).astype(int)
        out["return_rate_pct"] = (series("return_rate_pct") * rng.uniform(0.7, 1.4)).round(3)
        out["refund_rate_pct"] = (series("refund_rate_pct") * rng.uniform(0.7, 1.4)).round(3)
        out["sales_growth_pct"] = series("sales_growth_pct").round(2)
        out["product_sales_trend"] = np.roll(g["product_sales_trend"].to_numpy(), shift)

        # Inventory: some SKUs get deep stock, so not every product is short
        out["inventory_velocity_units_per_day"] = (out["sales_volume"] * rng.normal(1.22, 0.1, n).clip(0.95, 1.6)).round(2)
        stock_boost = rng.choice([0, 0, 1], p=[0.45, 0.2, 0.35])
        stock = series("stock_on_hand") * demand_f + stock_boost * rng.uniform(15, 40) * out["inventory_velocity_units_per_day"]
        out["stock_on_hand"] = np.round(stock).astype(int)
        out["days_of_inventory_remaining"] = (out["stock_on_hand"] / out["inventory_velocity_units_per_day"].clip(lower=0.1)).round(2)
        out["stockout_frequency_30d"] = (out["stock_on_hand"] <= 0).astype(int)

        # Logged historical decisions travel with their rows
        for column in ["recommended_action", "budget_change_pct", "decision_confidence_pct",
                       "opportunity_score", "decision_reason"]:
            out[column] = np.roll(g[column].to_numpy(), shift)

        out["budget_before_inr"] = out["ad_spend_inr"]
        out["budget_after_inr"] = (out["ad_spend_inr"] * (1 + out["budget_change_pct"] / 100)).round(2)

        # Next-day outcomes = the following day's values
        out["next_day_roas"] = out["roas"].shift(-1)
        out["next_day_profit_inr"] = out["profit_inr"].shift(-1)
        out["next_day_revenue_inr"] = out["revenue_inr"].shift(-1)
        out["next_day_sales_volume"] = out["sales_volume"].shift(-1)
        out["next_day_inventory_days"] = out["days_of_inventory_remaining"].shift(-1)
        out["next_day_stockout"] = out["stockout_frequency_30d"].shift(-1)
        out["baseline_roas"] = out["roas"]
        out["baseline_profit_inr"] = out["profit_inr"]
        out["roas_change_pct"] = ((out["next_day_roas"] - out["roas"]) / out["roas"] * 100).round(2)
        out["profit_change_pct"] = ((out["next_day_profit_inr"] - out["profit_inr"]) / out["profit_inr"].abs() * 100).round(2)
        out["action_success"] = np.where(out["next_day_roas"].isna(), np.nan, (out["roas_change_pct"] > 0).astype(float))

        frames.append(out)

    demo = pd.concat(frames, ignore_index=True)

    return demo[list(pd.read_csv(SOURCE, nrows=0).columns)]


if __name__ == "__main__":

    (TARGET_DIR / "data").mkdir(parents=True, exist_ok=True)

    build().to_csv(TARGET_DIR / "data" / "dataset.csv", index=False)

    (TARGET_DIR / "company.json").write_text(json.dumps({
        "name": "UrbanBrew Coffee",
        "description": "Simulated demo brand · coffee & tea D2C · generated by make_demo_dataset.py",
        "simulated": True,
        "created": date.today().isoformat(),
    }, indent=2), encoding="utf-8")

    print("Wrote", TARGET_DIR / "data" / "dataset.csv")
