"""
CSV ingestion for ADPULSE AI: any marketing / sales / inventory export ->
the unified schema the decision engine already uses.

    detect source  ->  map columns  ->  validate  ->  reconcile files  ->  unified dataset

Nothing here fabricates data. Fields that cannot be derived from the uploaded
files stay empty (NaN), and `capabilities()` reports which analyses are
therefore available and which are not.
"""

import difflib
import re

import numpy as np
import pandas as pd


# ============================================================
# CANONICAL FIELDS + SYNONYMS
# ============================================================

FIELDS = {
    # keys
    "date": ["date", "day", "reporting_starts", "report_date", "order_date", "created_at", "snapshot_date", "week"],
    "sku": ["sku", "variant_sku", "product_sku", "item_sku", "asin", "seller_sku", "sku_code", "product_id", "item_id"],
    "product": ["product", "product_name", "product_title", "title", "item_name", "sku_name", "lineitem_name", "product_type_name"],
    "category": ["category", "product_category", "product_type", "collection"],
    "variant": ["variant", "variant_title", "variant_name", "size", "option"],
    "campaign": ["campaign", "campaign_name", "ad_campaign", "campaign_title"],
    "platform": ["platform", "channel", "source", "network", "publisher_platform", "ad_platform"],
    "ad_type": ["ad_type", "creative_type", "format", "ad_format", "creative_format", "media_type"],
    "creative": ["creative", "ad_name", "creative_name", "ad"],
    # ad metrics
    "impressions": ["impressions", "impr", "impr.", "views", "impression"],
    "clicks": ["clicks", "link_clicks", "clicks_all", "outbound_clicks"],
    "ad_spend_inr": ["ad_spend", "spend", "amount_spent", "amount_spent_inr", "cost", "ad_cost", "spend_inr", "total_spend", "spend_(inr)", "media_cost"],
    "conversions": ["conversions", "purchases", "orders_from_ads", "results", "conv.", "conversion", "website_purchases", "attributed_orders"],
    "ad_revenue": ["purchase_value", "purchases_conversion_value", "conv._value", "conversion_value", "attributed_sales", "ad_revenue", "website_purchases_conversion_value", "total_conversion_value", "sales_attributed"],
    # sales
    "orders": ["orders", "order_count", "total_orders"],
    "sales_volume": ["units", "quantity", "units_sold", "net_quantity", "qty", "sales_volume", "items_sold", "ordered_units"],
    "revenue_inr": ["revenue", "sales", "net_sales", "gross_sales", "total_sales", "revenue_inr", "sales_amount", "ordered_product_sales"],
    "cogs": ["cogs", "cost_of_goods", "cost_of_goods_sold", "product_cost", "unit_cost_total"],
    "profit_inr": ["profit", "gross_profit", "net_profit", "profit_inr"],
    "margin_pct": ["margin", "margin_pct", "gross_margin", "margin_%"],
    "price": ["price", "unit_price", "selling_price", "avg_price", "mrp"],
    "competitor_price": ["competitor_price", "competitor_avg_price", "market_price", "rival_price"],
    "discount_pct": ["discount", "discount_pct", "discount_%", "promo_discount"],
    # inventory
    "stock_on_hand": ["stock", "stock_on_hand", "on_hand", "available", "inventory", "quantity_available", "inventory_on_hand", "available_quantity", "closing_stock"],
    # funnel / customer
    "product_page_views_from_ads": ["product_page_views", "page_views", "landing_page_views", "sessions", "product_views"],
    "add_to_cart": ["add_to_cart", "adds_to_cart", "add_to_carts", "atc", "added_to_cart"],
    "checkout_started": ["checkout_started", "checkouts_initiated", "initiated_checkout", "checkouts", "reached_checkout"],
    "payment_initiated": ["payment_initiated", "payment_info_added", "payments_initiated", "adds_of_payment_info"],
    "payment_failed": ["payment_failed", "failed_payments", "payment_failures"],
    "return_rate_pct": ["return_rate", "return_rate_pct", "returns_%"],
    "refund_rate_pct": ["refund_rate", "refund_rate_pct", "refunds_%"],
    "repeat_purchaser_count": ["repeat_customers", "returning_customers", "repeat_purchasers", "repeat_purchaser_count"],
    # creative / affiliate
    "video_3sec_views": ["video_3_sec_views", "3_second_video_plays", "video_plays", "video_views_3s"],
    "video_100pct_watched": ["video_100%_watched", "video_plays_at_100%", "video_completions", "video_100pct_watched"],
    "likes": ["likes", "reactions", "post_reactions"],
    "comments": ["comments", "post_comments"],
    "shares": ["shares", "post_shares"],
    "saves": ["saves", "post_saves"],
    "affiliate_link_interactions": ["affiliate_clicks", "affiliate_link_clicks", "affiliate_link_interactions", "affiliate_visits"],
}

NUMERIC = set(FIELDS) - {"date", "sku", "product", "category", "variant", "campaign", "platform", "ad_type", "creative"}

# Header fingerprints -> source type (and ad platform)
SOURCES = {
    "Meta Ads": ["amount_spent", "link_clicks", "purchases_conversion_value", "reach", "ad_set_name", "results", "cpm_(cost_per_1,000_impressions)"],
    "Google Ads": ["cost", "conv._value", "impr.", "conversions", "avg._cpc", "campaign_type", "search_impr._share"],
    "Amazon Ads": ["acos", "asin", "attributed_sales", "sponsored", "roas", "advertised_sku", "7_day_total_sales"],
    "TikTok Ads": ["tiktok", "video_views_at_100%", "cost_per_conversion", "2-second_video_views", "ad_group_name"],
    "Sales": ["orders", "net_sales", "gross_sales", "quantity", "units_sold", "net_quantity", "order_id", "lineitem_quantity"],
    "Inventory": ["stock", "on_hand", "available", "inventory", "warehouse", "reorder_point", "closing_stock"],
    "Customers / funnel": ["add_to_cart", "checkout", "sessions", "payment", "abandoned", "returning_customers"],
    "Pricing": ["price", "competitor_price", "mrp", "discount", "market_price"],
    "Creative performance": ["creative", "ad_name", "thumb_stop", "hook_rate", "video_plays", "format"],
    "Affiliate": ["affiliate", "commission", "publisher", "partner"],
}

AD_PLATFORMS = {"Meta Ads": "Meta", "Google Ads": "Google", "Amazon Ads": "Amazon", "TikTok Ads": "TikTok"}


def _norm(name):

    text = str(name).strip().lower()
    text = re.sub(r"\s*\((?:inr|₹|rs\.?|usd|\$)\)", "", text)
    text = re.sub(r"[^a-z0-9%.&]+", "_", text).strip("_")

    return text


# ============================================================
# 1 · DETECT SOURCE
# ============================================================

def detect_source(df, filename=""):
    """Return (source_type, platform or None, confidence 0-1)."""

    headers = {_norm(c) for c in df.columns}
    blob = " ".join(headers) + " " + _norm(filename)
    scores = {}

    for source, hints in SOURCES.items():
        hits = sum(1 for h in hints if h in headers or h in blob)
        scores[source] = hits / max(3, len(hints) * 0.6)

    # Platform named in the file name or a platform column wins for ad files
    for source, platform in AD_PLATFORMS.items():
        if platform.lower() in _norm(filename):
            scores[source] = scores.get(source, 0) + 1.0

    has_spend = any(h in headers for h in FIELDS["ad_spend_inr"])

    if not has_spend:
        for source in AD_PLATFORMS:
            scores[source] *= 0.3

    best = max(scores, key=scores.get)
    confidence = min(0.99, 0.45 + scores[best] * 0.5) if scores[best] > 0 else 0.3

    if best not in AD_PLATFORMS and has_spend and scores[best] < 0.6:
        best, confidence = "Ad platform (generic)", 0.6

    return best, AD_PLATFORMS.get(best), round(confidence, 2)


# ============================================================
# 2 · MAP COLUMNS
# ============================================================

def map_columns(df):
    """List of {column, field, confidence} — field None when nothing fits."""

    lookup = {}

    for field, synonyms in FIELDS.items():
        for synonym in [field] + synonyms:
            lookup.setdefault(_norm(synonym), field)

    mapping, used = [], set()

    for column in df.columns:

        key = _norm(column)
        field, confidence = None, 0.0

        if key in lookup:
            field, confidence = lookup[key], 1.0
        else:
            # token containment: "Website purchases conversion value" -> ad_revenue
            for synonym, candidate in sorted(lookup.items(), key=lambda kv: -len(kv[0])):
                if len(synonym) >= 4 and (synonym in key or key in synonym):
                    field, confidence = candidate, 0.85
                    break

            if field is None:
                close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.78)
                if close:
                    field = lookup[close[0]]
                    confidence = round(difflib.SequenceMatcher(None, key, close[0]).ratio(), 2)

        if field in used:
            field, confidence = None, 0.0     # first column wins; duplicates left unmapped

        if field:
            used.add(field)

        mapping.append({"column": column, "field": field, "confidence": confidence})

    return mapping


def _to_number(series):

    cleaned = (
        series.astype(str)
        .str.replace(r"[₹$,%\s]", "", regex=True)
        .str.replace(r"^(--|-|nan|None|)$", "", regex=True)
    )

    return pd.to_numeric(cleaned, errors="coerce")


# ============================================================
# 3 · NORMALISE + VALIDATE ONE FILE
# ============================================================

def normalise(df, mapping, platform=None):
    """Apply a mapping. Returns (frame with canonical columns, list of findings)."""

    out = pd.DataFrame(index=df.index)
    findings = []

    for item in mapping:
        if item["field"]:
            out[item["field"]] = df[item["column"]]

    findings.append(("ok", f"{len(df):,} rows read"))

    if "date" in out:
        parsed = pd.to_datetime(out["date"], errors="coerce", dayfirst=False)
        bad = int(parsed.isna().sum())
        out["date"] = parsed.dt.normalize()
        findings.append(("ok" if bad == 0 else "warn",
                         "Date detected" + (f" — {bad} rows have an unreadable date and are skipped" if bad else
                                            f" ({out['date'].min():%d %b} → {out['date'].max():%d %b %Y}, {out['date'].nunique()} days)")))
        out = out[out["date"].notna()]
    else:
        findings.append(("error", "No date column found — daily analysis isn't possible for this file"))

    for field in NUMERIC & set(out.columns):
        raw = out[field]
        out[field] = _to_number(raw)
        bad = int(out[field].isna().sum() - raw.isna().sum())
        if bad > 0:
            findings.append(("warn", f"{bad} non-numeric values in '{field}' treated as missing"))

    if "margin_pct" in out and out["margin_pct"].dropna().between(0, 1).all() and out["margin_pct"].notna().any():
        out["margin_pct"] = out["margin_pct"] * 100      # 0.46 -> 46

    for key in ["sku", "product", "campaign"]:
        if key in out:
            missing = int(out[key].isna().sum() + (out[key].astype(str).str.strip() == "").sum())
            if missing:
                findings.append(("warn", f"{missing} rows have no {key} — product-level insights may be incomplete"))

    dupes = int(out.duplicated().sum())
    if dupes:
        out = out.drop_duplicates()
        findings.append(("warn", f"{dupes} exact duplicate rows removed"))

    if platform:
        out["platform"] = out["platform"].fillna(platform) if "platform" in out else platform

    for name in ["revenue_inr", "ad_spend_inr", "stock_on_hand", "sku", "product", "campaign"]:
        if name in out:
            findings.append(("ok", f"{name.replace('_inr', '').replace('_', ' ').capitalize()} detected"))

    return out, findings


# ============================================================
# 4 · RECONCILE FILES INTO ONE VIEW
# ============================================================

def _entity(frame, catalog):
    """Key each row by SKU; fall back to product name, then to a product found in the campaign name."""

    key = pd.Series(np.nan, index=frame.index, dtype=object)

    if "sku" in frame:
        key = frame["sku"].astype(str).str.strip().where(frame["sku"].notna())

    if "product" in frame:
        by_name = frame["product"].astype(str).str.strip().str.lower().map(catalog.get("by_name", {}))
        key = key.fillna(by_name).fillna(frame["product"].astype(str).str.strip().where(frame["product"].notna()))

    if "campaign" in frame and catalog.get("names"):
        def from_campaign(text):
            text = str(text).lower()
            for name in sorted(catalog["names"], key=len, reverse=True):
                if name in text:
                    return catalog["by_name"][name]
            return np.nan
        key = key.fillna(frame["campaign"].map(from_campaign))

    if "campaign" in frame:
        key = key.fillna(frame["campaign"].astype(str))

    return key


def reconcile(frames):
    """frames: list of (source, normalised frame). Returns (unified daily frame, platform frame, notes)."""

    notes = []

    # Product catalogue from any file that has SKU + product name
    catalog = {"by_name": {}, "names": []}
    info = []

    for _, frame in frames:
        cols = [c for c in ["sku", "product", "category", "variant"] if c in frame]
        if "product" in cols:
            info.append(frame[cols].dropna(subset=["product"]).drop_duplicates())

    if info:
        info = pd.concat(info, ignore_index=True)
        has_sku = "sku" in info
        variants_per_product = info.groupby(info["product"].astype(str).str.strip().str.lower())["product"].transform("size") if has_sku else None

        for i, r in info.iterrows():
            product = str(r["product"]).strip()
            sku = str(r["sku"]).strip() if has_sku and pd.notna(r.get("sku")) else None

            # "Body Lotion 200 ml" -> that SKU (most specific, matched first)
            if sku and "variant" in r and pd.notna(r.get("variant")) and str(r["variant"]).strip():
                catalog["by_name"].setdefault(f"{product} {str(r['variant']).strip()}".lower(), sku)

            # "Body Lotion" alone -> the SKU only when the product has a single variant
            if sku and variants_per_product is not None and variants_per_product.loc[i] == 1:
                catalog["by_name"].setdefault(product.lower(), sku)
            elif not sku:
                catalog["by_name"].setdefault(product.lower(), product)

        catalog["names"] = list(catalog["by_name"])
        info = info.assign(entity=info.get("sku", info["product"]).astype(str).str.strip()).drop_duplicates("entity")
    else:
        info = pd.DataFrame(columns=["entity"])

    additive = sorted(NUMERIC - {"stock_on_hand", "margin_pct", "price", "competitor_price", "discount_pct",
                                 "return_rate_pct", "refund_rate_pct"})
    averaged = ["margin_pct", "price", "competitor_price", "discount_pct", "return_rate_pct", "refund_rate_pct"]

    parts, platform_parts = [], []

    for source, frame in frames:

        if "date" not in frame:
            notes.append(f"{source}: skipped (no date)")
            continue

        frame = frame.copy()
        frame["entity"] = _entity(frame, catalog)
        frame = frame[frame["entity"].notna()]

        agg = {c: "sum" for c in additive if c in frame}
        agg.update({c: "mean" for c in averaged if c in frame})
        if "stock_on_hand" in frame:
            agg["stock_on_hand"] = "last"
        for text in ["ad_type"]:
            if text in frame:
                agg[text] = lambda s: s.mode().iloc[0] if s.notna().any() else np.nan

        if not agg:
            continue

        daily = frame.groupby(["date", "entity"]).agg(agg)
        parts.append((source, daily))

        if "platform" in frame and "ad_spend_inr" in frame:
            cols = [c for c in ["ad_spend_inr", "ad_revenue", "impressions", "clicks", "conversions"] if c in frame]
            platform_parts.append(frame.groupby(["date", "entity", "platform"])[cols].sum(min_count=1).reset_index())

    if not parts:
        return pd.DataFrame(), pd.DataFrame(), notes + ["No usable rows after reconciliation"]

    # Combine: additive metrics from several ad files add up (e.g. Meta + Google spend)
    unified = None

    for source, daily in parts:
        if unified is None:
            unified = daily
            continue
        overlap = [c for c in daily.columns if c in unified.columns]
        new = [c for c in daily.columns if c not in unified.columns]
        unified = unified.join(daily[new], how="outer") if new else unified.reindex(unified.index.union(daily.index))
        for c in overlap:
            if c in additive:
                unified[c] = unified[c].add(daily[c].reindex(unified.index), fill_value=0) if c in daily else unified[c]
            else:
                unified[c] = unified[c].combine_first(daily[c].reindex(unified.index))

    unified = unified.reset_index()

    keys_per_file = {source: set(daily.index.get_level_values("entity")) for source, daily in parts}
    if len(parts) > 1:
        shared = set.intersection(*keys_per_file.values())
        notes.append(f"{len(shared)} products/SKUs matched across {len(parts)} files")

    platforms = pd.concat(platform_parts, ignore_index=True) if platform_parts else pd.DataFrame()

    unified = unified.merge(
        info.drop(columns=[c for c in ["sku"] if c in info]).rename(columns={}), on="entity", how="left"
    ) if len(info) else unified

    return unified, platforms, notes


# ============================================================
# 5 · BUILD THE UNIFIED DATASET (same schema as Glowroots)
# ============================================================

def build_dataset(unified, schema):
    """Derive every schema column that the uploaded data supports; leave the rest empty."""

    u = unified.sort_values(["entity", "date"]).copy()
    out = pd.DataFrame({"date": u["date"].dt.strftime("%Y-%m-%d"), "sku": u["entity"].astype(str)})

    product = u["product"] if "product" in u else u["entity"]
    out["product"] = product.fillna(u["entity"]).astype(str)
    out["product_code"] = out["sku"]
    out["category"] = u["category"].fillna("Uncategorised") if "category" in u else "Uncategorised"
    out["variant"] = u["variant"].fillna("") if "variant" in u else ""
    out["ad_type"] = u["ad_type"] if "ad_type" in u else np.nan

    def col(name):
        return u[name] if name in u else pd.Series(np.nan, index=u.index)

    spend = col("ad_spend_inr")
    units = col("sales_volume").fillna(col("orders"))
    revenue = col("revenue_inr").fillna(col("ad_revenue"))
    # Orders: ad conversions, else store orders, else units sold (1 unit ≈ 1 order — a stated approximation)
    conversions = col("conversions").fillna(col("orders")).fillna(col("sales_volume"))

    out["impressions"] = col("impressions")
    out["clicks"] = col("clicks")
    out["product_clicks"] = col("clicks")
    out["click_rate_pct"] = col("clicks") / col("impressions").where(col("impressions") > 0) * 100
    out["product_page_views_from_ads"] = col("product_page_views_from_ads")
    out["ad_spend_inr"] = spend
    out["conversions"] = conversions
    out["sales_volume"] = units.fillna(conversions)
    out["revenue_inr"] = revenue
    out["roas"] = revenue / spend.where(spend > 0)

    margin = col("margin_pct")
    if "cogs" in u:
        margin = margin.fillna((revenue - col("cogs")) / revenue.where(revenue > 0) * 100)
    out["margin_pct"] = margin.round(1).map(lambda v: f"{v:.0f}%" if pd.notna(v) else np.nan)

    profit = col("profit_inr")
    profit = profit.fillna(revenue * margin / 100 - spend.fillna(0))
    out["profit_inr"] = profit

    for name in ["add_to_cart", "checkout_started", "payment_initiated", "payment_failed",
                 "return_rate_pct", "refund_rate_pct", "repeat_purchaser_count", "video_3sec_views",
                 "video_100pct_watched", "likes", "comments", "shares", "saves", "affiliate_link_interactions"]:
        out[name] = col(name)

    out["add_to_cart_rate_pct"] = out["add_to_cart"] / out["product_page_views_from_ads"].where(out["product_page_views_from_ads"] > 0) * 100
    out["purchase_rate_pct"] = conversions / col("clicks").where(col("clicks") > 0) * 100
    out["payment_failed_rate_pct"] = out["payment_failed"] / out["payment_initiated"].where(out["payment_initiated"] > 0) * 100
    engagement = col("likes").add(col("comments"), fill_value=0).add(col("shares"), fill_value=0).add(col("saves"), fill_value=0)
    out["engagement_rate_pct"] = engagement.where(col("likes").notna()) / col("impressions").where(col("impressions") > 0) * 100

    # Inventory: velocity = trailing 7-day average units, cover = stock / velocity
    stock = col("stock_on_hand")
    velocity = out.groupby("sku")["sales_volume"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["stock_on_hand"] = stock
    out["inventory_velocity_units_per_day"] = velocity
    out["days_of_inventory_remaining"] = (stock / velocity.where(velocity > 0)).where(stock.notna())
    out["stockout_frequency_30d"] = (stock <= 0).astype(float).where(stock.notna())

    out["sales_growth_pct"] = out.groupby("sku")["sales_volume"].pct_change().mul(100).replace([np.inf, -np.inf], np.nan)
    trend = out.groupby("sku")["sales_volume"].transform(lambda s: s.rolling(7, min_periods=3).mean().pct_change(3))
    out["product_sales_trend"] = np.select([trend > 0.05, trend < -0.05], ["Rising", "Falling"], "Stable")

    # Next-day outcomes (the forecasting target) — the following day's values
    group = out.groupby("sku")
    out["next_day_roas"] = group["roas"].shift(-1)
    out["next_day_profit_inr"] = group["profit_inr"].shift(-1)
    out["next_day_revenue_inr"] = group["revenue_inr"].shift(-1)
    out["next_day_sales_volume"] = group["sales_volume"].shift(-1)
    out["next_day_inventory_days"] = group["days_of_inventory_remaining"].shift(-1)
    out["next_day_stockout"] = group["stockout_frequency_30d"].shift(-1)
    out["baseline_roas"] = out["roas"]
    out["baseline_profit_inr"] = out["profit_inr"]
    out["roas_change_pct"] = (out["next_day_roas"] - out["roas"]) / out["roas"].where(out["roas"] > 0) * 100
    out["profit_change_pct"] = (out["next_day_profit_inr"] - out["profit_inr"]) / out["profit_inr"].abs().where(out["profit_inr"] != 0) * 100
    out["budget_before_inr"] = spend

    for name in schema:
        if name not in out:
            out[name] = np.nan                # e.g. historical decision logs: not in uploads

    extra = [c for c in ["competitor_price", "price", "discount_pct"] if c in u]
    for name in extra:
        out[name] = u[name].values

    return out[list(schema) + extra].replace([np.inf, -np.inf], np.nan)


# ============================================================
# 6 · WHAT CAN BE ANALYSED?
# ============================================================

def capabilities(dataset, platforms=None):
    """analysis -> (available, reason). Drives honest 'unavailable' messages in the app."""

    def has(*cols, share=0.3):
        return all(c in dataset and dataset[c].notna().mean() >= share for c in cols)

    days = dataset["date"].nunique() if "date" in dataset else 0
    n_platforms = platforms["platform"].nunique() if platforms is not None and len(platforms) else 0

    return {
        "Ad performance (ROAS)": (has("ad_spend_inr", "revenue_inr"), "needs ad spend and revenue"),
        "Forecasting (next-day ROAS)": (has("roas", "ad_spend_inr") and days >= 10, f"needs ad spend + revenue for ≥ 10 days (found {days})"),
        "Cross-platform budget shifts": (n_platforms >= 2, f"needs ad data from ≥ 2 platforms (found {n_platforms})"),
        "Profitability": (has("margin_pct") or has("profit_inr"), "needs margin, COGS or profit"),
        "Inventory": (has("stock_on_hand"), "inventory data not detected"),
        "Funnel / cart abandonment": (has("add_to_cart", "conversions"), "needs add-to-cart and purchase counts"),
        "Creative performance": (has("ad_type"), "needs ad format / creative type"),
        "Affiliate": (has("affiliate_link_interactions"), "affiliate data not detected"),
        "Competitor pricing": (has("competitor_price") if "competitor_price" in dataset else False, "competitor prices not detected"),
        "Learning from past decisions": (has("recommended_action", "action_success"), "needs a history of past decisions and outcomes"),
    }


def is_complete(dataset, schema):
    """True when every schema column has data — the full ML decision pipeline can run."""

    return all(c in dataset and dataset[c].notna().any() for c in schema)
