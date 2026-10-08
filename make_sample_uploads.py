"""
Create SIMULATED sample CSV exports for trying the drag-and-drop upload.

They are derived from the simulated Glowroots dataset and use the column
names each tool exports, so the column-mapping step has real work to do:

    sample_uploads/meta_ads.csv        Meta Ads Manager style
    sample_uploads/google_ads.csv      Google Ads style
    sample_uploads/shopify_sales.csv   Shopify sales report style
    sample_uploads/inventory.csv       warehouse / stock snapshot style

Ad spend and attributed revenue are split between Meta and Google with a
different efficiency per product, so cross-platform budget decisions can be
demonstrated. These are NOT real platform exports.

    python make_sample_uploads.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sample_uploads"
SEED = 11


def main():

    rng = np.random.default_rng(SEED)
    d = pd.read_csv(ROOT / "data" / "dataset.csv")
    d["margin"] = d["margin_pct"].str.rstrip("%").astype(float) / 100

    meta_rows, google_rows = [], []

    for sku, g in d.groupby("sku"):

        meta_share = rng.uniform(0.45, 0.75)                 # share of spend on Meta
        meta_eff = rng.uniform(0.7, 1.5)                     # relative efficiency of Meta vs Google
        label = f"{g['product'].iloc[0]} {g['variant'].iloc[0]}"

        for _, r in g.iterrows():

            spend_m = r["ad_spend_inr"] * meta_share
            spend_g = r["ad_spend_inr"] - spend_m
            weight_m = spend_m * meta_eff
            rev_share = weight_m / (weight_m + spend_g)
            attributed = r["revenue_inr"] * 0.8               # ads get credit for 80% of sales

            meta_rows.append({
                "Day": r["date"],
                "Campaign name": f"GR | {label} | Prospecting",
                "Ad set name": f"{label} - Broad",
                "Format": r["ad_type"],
                "Amount spent (INR)": round(spend_m, 2),
                "Impressions": int(r["impressions"] * meta_share),
                "Link clicks": int(r["clicks"] * meta_share),
                "Purchases": int(round(r["conversions"] * rev_share)),
                "Purchases conversion value": round(attributed * rev_share, 2),
            })

            google_rows.append({
                "Day": r["date"],
                "Campaign": f"GR {label} - Search",
                "Cost": round(spend_g, 2),
                "Impr.": int(r["impressions"] * (1 - meta_share)),
                "Clicks": int(r["clicks"] * (1 - meta_share)),
                "Conversions": round(r["conversions"] * (1 - rev_share), 1),
                "Conv. value": round(attributed * (1 - rev_share), 2),
            })

    sales = pd.DataFrame({
        "Day": d["date"],
        "Product title": d["product"],
        "Variant title": d["variant"],
        "Variant SKU": d["sku"],
        "Product type": d["category"],
        "Net quantity": d["sales_volume"],
        "Net sales": d["revenue_inr"].round(2),
        "Cost of goods sold": (d["revenue_inr"] * (1 - d["margin"])).round(2),
        "Added to cart": d["add_to_cart"],
        "Reached checkout": d["checkout_started"],
    })

    inventory = pd.DataFrame({
        "Snapshot date": d["date"],
        "SKU": d["sku"],
        "Available": d["stock_on_hand"],
    })

    files = {
        "meta_ads.csv": (pd.DataFrame(meta_rows), "Day"),
        "google_ads.csv": (pd.DataFrame(google_rows), "Day"),
        "shopify_sales.csv": (sales, "Day"),
        "inventory.csv": (inventory, "Snapshot date"),
    }

    # Weeks 1-3 to analyse first; week 4 "arrives later" to measure approved actions
    split = "2026-09-21"

    for folder, later in [("weeks_1-3", False), ("week_4_followup", True)]:
        target = OUT / folder
        target.mkdir(parents=True, exist_ok=True)
        for name, (frame, date_col) in files.items():
            dates = frame[date_col].astype(str)
            frame[(dates > split) if later else (dates <= split)].to_csv(target / name, index=False)

    (OUT / "README.txt").write_text(
        "SIMULATED sample exports for the ADPULSE AI upload demo.\n"
        "Derived from the simulated Glowroots dataset; not real platform data.\n\n"
        "1. Drop the files in weeks_1-3/ on the Upload page -> analysis + recommendations.\n"
        "2. Approve a few recommendations.\n"
        "3. Add the files in week_4_followup/ to the same company -> outcomes are measured\n"
        "   and the AI's confidence learns from them.\n",
        encoding="utf-8",
    )

    print("Wrote", ", ".join(str(p.relative_to(OUT)) for p in sorted(OUT.rglob("*.csv"))))


if __name__ == "__main__":
    main()
