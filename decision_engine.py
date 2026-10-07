import pandas as pd
import numpy as np
import joblib


# ============================================================
# ADPULSE AI - DECISION ENGINE
# ============================================================

print("=" * 60)
print("ADPULSE AI - AUTONOMOUS D2C DECISION ENGINE")
print("=" * 60)


# ============================================================
# STEP 1 - LOAD DATASET
# ============================================================

df = pd.read_csv("data/dataset.csv")

print("\nDataset loaded.")
print("Rows:", len(df))


# ============================================================
# STEP 2 - LOAD TRAINED ROAS MODEL
# ============================================================

model_package = joblib.load(
    "models/roas_predictor.pkl"
)

model = model_package["model"]
preprocessor = model_package["preprocessor"]
features = model_package["features"]

print("ROAS model loaded successfully.")


# ============================================================
# STEP 3 - DATE PROCESSING
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month


# ============================================================
# STEP 4 - CATEGORICAL / NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "sku",
    "product",
    "category",
    "variant",
    "ad_type",
    "product_sales_trend"
]

numerical_features = [
    feature
    for feature in features
    if feature not in categorical_features
]


# ============================================================
# STEP 5 - CLEAN NUMERICAL COLUMNS
# ============================================================

for column in numerical_features:

    df[column] = (
        df[column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# STEP 6 - CREATE INPUT DATA
# ============================================================

X = df[features].copy()


# ============================================================
# STEP 7 - GENERATE ROAS PREDICTION
# ============================================================

print("\nGenerating next-day ROAS predictions...")

X_processed = preprocessor.transform(X)

df["predicted_next_day_roas"] = model.predict(
    X_processed
)

# ROAS cannot realistically be negative
df["predicted_next_day_roas"] = (
    df["predicted_next_day_roas"]
    .clip(lower=0)
)


# ============================================================
# STEP 8 - PREDICTED REVENUE
# ============================================================

df["predicted_next_day_revenue_inr"] = (
    df["predicted_next_day_roas"]
    * df["ad_spend_inr"]
)


# ============================================================
# STEP 9 - PREDICTED PROFIT
# ============================================================

# Gross profit = Revenue × Margin
#
# Advertising profit approximation:
#
# Predicted Profit =
# Predicted Revenue × Margin %
#                   - Ad Spend

df["predicted_next_day_profit_inr"] = (
    (
        df["predicted_next_day_revenue_inr"]
        * df["margin_pct"]
        / 100
    )
    - df["ad_spend_inr"]
)


# ============================================================
# STEP 10 - INVENTORY HEALTH
# ============================================================

def inventory_status(days):

    if days <= 3:
        return "CRITICAL"

    elif days <= 7:
        return "LOW"

    elif days <= 15:
        return "HEALTHY"

    else:
        return "STRONG"


df["inventory_status"] = (
    df["days_of_inventory_remaining"]
    .apply(inventory_status)
)


# ============================================================
# STEP 11 - OPPORTUNITY SCORE
# ============================================================

# ROAS score
roas_score = (
    df["predicted_next_day_roas"] / 10 * 100
).clip(0, 100)


# Margin score
margin_score = (
    df["margin_pct"] / 40 * 100
).clip(0, 100)


# Sales growth score
sales_growth_score = (
    (df["sales_growth_pct"] + 20) / 40 * 100
).clip(0, 100)


# Inventory score
inventory_score = (
    df["days_of_inventory_remaining"] / 30 * 100
).clip(0, 100)


# Conversion score
conversion_score = (
    df["purchase_rate_pct"] / 10 * 100
).clip(0, 100)


# Weighted opportunity score
df["opportunity_score_calculated"] = (
    roas_score * 0.35
    + margin_score * 0.20
    + sales_growth_score * 0.15
    + inventory_score * 0.20
    + conversion_score * 0.10
)


df["opportunity_score_calculated"] = (
    df["opportunity_score_calculated"]
    .clip(0, 100)
    .round(2)
)


# ============================================================
# STEP 12 - DECISION ENGINE
# ============================================================

def make_decision(row):

    predicted_roas = row["predicted_next_day_roas"]

    predicted_profit = (
        row["predicted_next_day_profit_inr"]
    )

    inventory_days = (
        row["days_of_inventory_remaining"]
    )

    payment_failure = (
        row["payment_failed_rate_pct"]
    )

    purchase_rate = (
        row["purchase_rate_pct"]
    )

    add_to_cart_rate = (
        row["add_to_cart_rate_pct"]
    )

    click_rate = (
        row["click_rate_pct"]
    )

    sales_growth = (
        row["sales_growth_pct"]
    )


    # --------------------------------------------------------
    # PRIORITY 1 - INVENTORY PROTECTION
    # --------------------------------------------------------

    if inventory_days <= 3:

        return "PROTECT_INVENTORY"


    # --------------------------------------------------------
    # PRIORITY 2 - PAYMENT PROBLEM
    # --------------------------------------------------------

    if payment_failure >= 10:

        return "FIX_PAYMENT"


    # --------------------------------------------------------
    # PRIORITY 3 - HIGH CART BUT LOW PURCHASE
    # --------------------------------------------------------

    if (
        add_to_cart_rate >= 8
        and purchase_rate < 3
    ):

        return "RETARGET"


    # --------------------------------------------------------
    # PRIORITY 4 - GOOD CTR BUT WEAK PURCHASE
    # --------------------------------------------------------

    if (
        click_rate >= 2.5
        and purchase_rate < 3
    ):

        return "CHANGE_CREATIVE"


    # --------------------------------------------------------
    # PRIORITY 5 - SCALE WINNING CAMPAIGN
    # --------------------------------------------------------

    if (
        predicted_roas >= 8
        and predicted_profit > 0
        and sales_growth > 0
        and inventory_days > 7
    ):

        return "INCREASE_BUDGET"


    # --------------------------------------------------------
    # PRIORITY 6 - REDUCE LOSS-MAKING CAMPAIGN
    # --------------------------------------------------------

    if (
        predicted_roas < 4
        or predicted_profit < 0
    ):

        return "DECREASE_BUDGET"


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "MAINTAIN"


df["recommended_action_calculated"] = (
    df.apply(
        make_decision,
        axis=1
    )
)


# ============================================================
# STEP 13 - BUDGET CHANGE
# ============================================================

def budget_change(action):

    if action == "INCREASE_BUDGET":
        return 15

    elif action == "DECREASE_BUDGET":
        return -15

    else:
        return 0


df["recommended_budget_change_pct"] = (
    df["recommended_action_calculated"]
    .apply(budget_change)
)


# ============================================================
# STEP 14 - RECOMMENDED DAILY BUDGET
# ============================================================

df["current_daily_ad_spend_inr"] = (
    df["ad_spend_inr"]
)


df["recommended_daily_budget_inr"] = (
    df["current_daily_ad_spend_inr"]
    * (
        1
        + df["recommended_budget_change_pct"]
        / 100
    )
)


df["recommended_daily_budget_inr"] = (
    df["recommended_daily_budget_inr"]
    .round(2)
)


# ============================================================
# STEP 15 - DECISION REASON
# ============================================================

def decision_reason(row):

    action = row[
        "recommended_action_calculated"
    ]

    predicted_roas = round(
        row["predicted_next_day_roas"],
        2
    )

    predicted_profit = round(
        row["predicted_next_day_profit_inr"],
        2
    )

    inventory_days = round(
        row["days_of_inventory_remaining"],
        1
    )

    sales_growth = round(
        row["sales_growth_pct"],
        1
    )


    if action == "PROTECT_INVENTORY":

        return (
            f"Inventory is critical with "
            f"{inventory_days} days remaining. "
            f"Protect stock before scaling ads."
        )


    if action == "FIX_PAYMENT":

        return (
            f"Payment failure rate is "
            f"{row['payment_failed_rate_pct']:.1f}%. "
            f"Fix payment issues before increasing spend."
        )


    if action == "RETARGET":

        return (
            f"High add-to-cart activity but weak purchase "
            f"conversion. Retarget high-intent users."
        )


    if action == "CHANGE_CREATIVE":

        return (
            f"CTR is strong but purchase conversion is weak. "
            f"Test new creatives or landing-page messaging."
        )


    if action == "INCREASE_BUDGET":

        return (
            f"Predicted ROAS is {predicted_roas} with "
            f"predicted profit of ₹{predicted_profit}. "
            f"Sales growth is {sales_growth}%. "
            f"Inventory is sufficient. Scale budget."
        )


    if action == "DECREASE_BUDGET":

        return (
            f"Predicted ROAS is {predicted_roas} and "
            f"predicted profit is ₹{predicted_profit}. "
            f"Reduce inefficient ad spend."
        )


    return (
        f"Predicted ROAS is {predicted_roas} and "
        f"predicted profit is ₹{predicted_profit}. "
        f"Signals are within acceptable range. "
        f"Maintain current budget."
    )


df["decision_reason_calculated"] = (
    df.apply(
        decision_reason,
        axis=1
    )
)


# ============================================================
# STEP 16 - DECISION CONFIDENCE
# ============================================================

def confidence_score(row):

    score = 50

    predicted_roas = (
        row["predicted_next_day_roas"]
    )

    inventory_days = (
        row["days_of_inventory_remaining"]
    )

    opportunity = (
        row["opportunity_score_calculated"]
    )

    # Strong opportunity signal
    if predicted_roas >= 8:
        score += 15

    # Healthy inventory
    if inventory_days >= 15:
        score += 10

    # Strong opportunity score
    if opportunity >= 70:
        score += 15

    # Critical inventory reduces confidence
    if inventory_days <= 3:
        score -= 20

    return np.clip(
        score,
        0,
        100
    )


df["decision_confidence_pct"] = (
    df.apply(
        confidence_score,
        axis=1
    )
)


# ============================================================
# STEP 17 - CREATE OUTPUT
# ============================================================

output_columns = [

    "date",

    "sku",
    "product",
    "category",
    "variant",

    "ad_type",

    "ad_spend_inr",

    "roas",

    "sales_volume",
    "sales_growth_pct",

    "stock_on_hand",
    "days_of_inventory_remaining",

    "click_rate_pct",
    "add_to_cart_rate_pct",
    "purchase_rate_pct",

    "payment_failed_rate_pct",

    "margin_pct",

    "predicted_next_day_roas",

    "predicted_next_day_revenue_inr",

    "predicted_next_day_profit_inr",

    "inventory_status",

    "opportunity_score_calculated",

    "recommended_action_calculated",

    "recommended_budget_change_pct",

    "current_daily_ad_spend_inr",

    "recommended_daily_budget_inr",

    "decision_confidence_pct",

    "decision_reason_calculated"
]


decision_output = df[output_columns].copy()


# ============================================================
# STEP 18 - SAVE ALL DECISIONS
# ============================================================

decision_output.to_csv(
    "models/decision_engine_results.csv",
    index=False
)


# ============================================================
# STEP 19 - LATEST DECISION PER SKU
# ============================================================

latest_date = (
    decision_output["date"]
    .max()
)


latest_decisions = (
    decision_output[
        decision_output["date"] == latest_date
    ]
    .copy()
)


latest_decisions = latest_decisions.sort_values(
    "opportunity_score_calculated",
    ascending=False
)


latest_decisions.to_csv(
    "models/latest_decisions.csv",
    index=False
)


# ============================================================
# STEP 20 - DISPLAY DECISION SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("DECISION ENGINE RESULTS")
print("=" * 60)

print(
    "\nLatest data date:",
    latest_date.date()
)

print(
    "\nLatest SKU decisions:"
)


display_columns = [
    "sku",
    "product",
    "predicted_next_day_roas",
    "predicted_next_day_profit_inr",
    "days_of_inventory_remaining",
    "opportunity_score_calculated",
    "recommended_action_calculated",
    "recommended_budget_change_pct"
]


print(
    latest_decisions[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# STEP 21 - ACTION SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("ACTION SUMMARY")
print("=" * 60)

action_counts = (
    latest_decisions[
        "recommended_action_calculated"
    ]
    .value_counts()
)


print(action_counts.to_string())


# ============================================================
# STEP 22 - TOP OPPORTUNITIES
# ============================================================

print("\n")
print("=" * 60)
print("TOP 5 OPPORTUNITIES")
print("=" * 60)

top_opportunities = (
    latest_decisions
    .head(5)
)


for _, row in top_opportunities.iterrows():

    print("\nProduct:", row["product"])
    print("SKU:", row["sku"])

    print(
        "Predicted ROAS:",
        round(
            row["predicted_next_day_roas"],
            2
        )
    )

    print(
        "Predicted Profit: ₹",
        round(
            row["predicted_next_day_profit_inr"],
            2
        )
    )

    print(
        "Opportunity Score:",
        row["opportunity_score_calculated"]
    )

    print(
        "Decision:",
        row["recommended_action_calculated"]
    )

    print(
        "Budget Change:",
        str(
            row["recommended_budget_change_pct"]
        ) + "%"
    )

    print(
        "Reason:",
        row["decision_reason_calculated"]
    )


# ============================================================
# STEP 23 - FINAL FILES
# ============================================================

print("\n")
print("=" * 60)
print("STEP 9 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles created:")

print(
    "1. models/decision_engine_results.csv"
)

print(
    "2. models/latest_decisions.csv"
)

print("\n")
print("ADPULSE AI decision engine is now operational.")