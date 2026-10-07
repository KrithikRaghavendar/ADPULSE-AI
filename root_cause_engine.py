import pandas as pd
import numpy as np
import joblib


# ============================================================
# ADPULSE AI - ROOT CAUSE INTELLIGENCE ENGINE
# ============================================================

print("=" * 65)
print("ADPULSE AI - ROOT CAUSE INTELLIGENCE ENGINE")
print("=" * 65)


# ============================================================
# STEP 1 - LOAD DATA
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

print("ROAS prediction model loaded.")


# ============================================================
# STEP 3 - DATE PROCESSING
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month


# ============================================================
# STEP 4 - CLEAN NUMERICAL FEATURES
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
# STEP 5 - ROAS PREDICTION
# ============================================================

print("\nGenerating ROAS predictions...")

X = df[features].copy()

X_processed = preprocessor.transform(X)

df["predicted_next_day_roas"] = model.predict(
    X_processed
)

df["predicted_next_day_roas"] = (
    df["predicted_next_day_roas"]
    .clip(lower=0)
)


# ============================================================
# STEP 6 - PREDICTED REVENUE
# ============================================================

df["predicted_next_day_revenue_inr"] = (
    df["predicted_next_day_roas"]
    * df["ad_spend_inr"]
)


# ============================================================
# STEP 7 - PREDICTED PROFIT
# ============================================================

df["predicted_next_day_profit_inr"] = (
    (
        df["predicted_next_day_revenue_inr"]
        * df["margin_pct"]
        / 100
    )
    - df["ad_spend_inr"]
)


# ============================================================
# STEP 8 - INVENTORY RISK
# ============================================================

def inventory_risk(days):

    if days <= 0:
        return "CRITICAL"

    elif days <= 3:
        return "HIGH"

    elif days <= 7:
        return "MEDIUM"

    elif days <= 15:
        return "LOW"

    else:
        return "HEALTHY"


df["inventory_risk"] = (
    df["days_of_inventory_remaining"]
    .apply(inventory_risk)
)


# ============================================================
# STEP 9 - FUNNEL DIAGNOSTICS
# ============================================================

def funnel_status(row):

    click_rate = row["click_rate_pct"]

    add_to_cart = row["add_to_cart_rate_pct"]

    purchase_rate = row["purchase_rate_pct"]

    page_views = row["product_page_views_from_ads"]

    clicks = row["clicks"]


    # Very weak CTR
    if click_rate < 1:

        return "LOW_AD_ENGAGEMENT"


    # Good CTR but weak ATC
    if (
        click_rate >= 1
        and add_to_cart < 3
    ):

        return "LOW_PRODUCT_INTEREST"


    # Good ATC but weak purchase
    if (
        add_to_cart >= 5
        and purchase_rate < 3
    ):

        return "CHECKOUT_CONVERSION"


    # Good purchase conversion
    if purchase_rate >= 5:

        return "HEALTHY_CONVERSION"


    return "NORMAL"


df["funnel_status"] = (
    df.apply(
        funnel_status,
        axis=1
    )
)


# ============================================================
# STEP 10 - PAYMENT DIAGNOSTICS
# ============================================================

def payment_status(rate):

    if rate >= 10:
        return "CRITICAL"

    elif rate >= 5:
        return "HIGH"

    elif rate >= 2:
        return "MEDIUM"

    else:
        return "NORMAL"


df["payment_status"] = (
    df["payment_failed_rate_pct"]
    .apply(payment_status)
)


# ============================================================
# STEP 11 - CREATIVE DIAGNOSTICS
# ============================================================

def creative_status(row):

    ctr = row["click_rate_pct"]

    engagement = row["engagement_rate_pct"]

    purchase = row["purchase_rate_pct"]


    if (
        ctr < 1
        and engagement < 2
    ):

        return "WEAK_CREATIVE"


    if (
        ctr >= 2.5
        and purchase < 3
    ):

        return "CREATIVE_TO_CONVERSION_GAP"


    if engagement >= 5:

        return "STRONG_CREATIVE"


    return "NORMAL"


df["creative_status"] = (
    df.apply(
        creative_status,
        axis=1
    )
)


# ============================================================
# STEP 12 - SALES DIAGNOSTICS
# ============================================================

def sales_status(row):

    growth = row["sales_growth_pct"]

    trend = str(
        row["product_sales_trend"]
    ).lower()


    if growth >= 10:

        return "STRONG_GROWTH"


    if growth <= -10:

        return "DECLINING"


    if "declin" in trend:

        return "DECLINING"


    if "grow" in trend:

        return "GROWING"


    return "STABLE"


df["sales_status"] = (
    df.apply(
        sales_status,
        axis=1
    )
)


# ============================================================
# STEP 13 - PROFITABILITY DIAGNOSTICS
# ============================================================

def profitability_status(row):

    predicted_profit = (
        row["predicted_next_day_profit_inr"]
    )

    margin = row["margin_pct"]

    predicted_roas = (
        row["predicted_next_day_roas"]
    )


    if predicted_profit < 0:

        return "LOSS_RISK"


    if margin < 15:

        return "LOW_MARGIN"


    if predicted_roas >= 8 and predicted_profit > 0:

        return "HIGHLY_PROFITABLE"


    return "PROFITABLE"


df["profitability_status"] = (
    df.apply(
        profitability_status,
        axis=1
    )
)


# ============================================================
# STEP 14 - ROOT CAUSE ENGINE
# ============================================================

def detect_root_cause(row):

    inventory = row["inventory_risk"]

    payment = row["payment_status"]

    funnel = row["funnel_status"]

    creative = row["creative_status"]

    sales = row["sales_status"]

    profitability = row["profitability_status"]

    predicted_roas = (
        row["predicted_next_day_roas"]
    )


    # --------------------------------------------------------
    # PRIORITY 1
    # INVENTORY
    # --------------------------------------------------------

    if inventory == "CRITICAL":

        return (
            "INVENTORY_SHORTAGE",
            "Stock is exhausted or expected to run out immediately."
        )


    if inventory == "HIGH":

        return (
            "INVENTORY_RISK",
            "Inventory is very low and scaling ads may cause stockout."
        )


    # --------------------------------------------------------
    # PRIORITY 2
    # PAYMENT
    # --------------------------------------------------------

    if payment == "CRITICAL":

        return (
            "PAYMENT_FAILURE",
            "High payment failure rate is blocking completed purchases."
        )


    if payment == "HIGH":

        return (
            "PAYMENT_FRICTION",
            "Elevated payment failures may be reducing completed orders."
        )


    # --------------------------------------------------------
    # PRIORITY 3
    # CHECKOUT / CONVERSION
    # --------------------------------------------------------

    if funnel == "CHECKOUT_CONVERSION":

        return (
            "FUNNEL_DROP_OFF",
            "Users show purchase intent but conversion remains weak."
        )


    # --------------------------------------------------------
    # PRIORITY 4
    # CREATIVE
    # --------------------------------------------------------

    if creative == "WEAK_CREATIVE":

        return (
            "CREATIVE_WEAKNESS",
            "Low engagement and click-through indicate weak ad creative."
        )


    if creative == "CREATIVE_TO_CONVERSION_GAP":

        return (
            "CREATIVE_CONVERSION_GAP",
            "Ads attract clicks but those clicks are not converting."
        )


    # --------------------------------------------------------
    # PRIORITY 5
    # SALES DECLINE
    # --------------------------------------------------------

    if sales == "DECLINING":

        return (
            "DEMAND_DECLINE",
            "Sales signals indicate declining product demand."
        )


    # --------------------------------------------------------
    # PRIORITY 6
    # PROFITABILITY
    # --------------------------------------------------------

    if profitability == "LOSS_RISK":

        return (
            "PROFITABILITY_RISK",
            "Predicted revenue does not generate positive advertising profit."
        )


    if profitability == "LOW_MARGIN":

        return (
            "MARGIN_PRESSURE",
            "Low product margin limits profitable advertising scale."
        )


    # --------------------------------------------------------
    # PRIORITY 7
    # HIGH PERFORMER
    # --------------------------------------------------------

    if predicted_roas >= 8:

        return (
            "HIGH_PERFORMER",
            "Strong predicted ROAS and healthy business signals indicate a scaling opportunity."
        )


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return (
        "NORMAL_PERFORMANCE",
        "No major negative signal detected."
    )


root_cause_results = (
    df.apply(
        detect_root_cause,
        axis=1
    )
)


df["root_cause"] = [
    result[0]
    for result in root_cause_results
]


df["root_cause_explanation"] = [
    result[1]
    for result in root_cause_results
]


# ============================================================
# STEP 15 - AUTONOMOUS ACTION
# ============================================================

def decide_action(row):

    root_cause = row["root_cause"]

    predicted_roas = (
        row["predicted_next_day_roas"]
    )

    predicted_profit = (
        row["predicted_next_day_profit_inr"]
    )

    inventory = row["inventory_risk"]


    # Inventory always wins
    if root_cause in [
        "INVENTORY_SHORTAGE",
        "INVENTORY_RISK"
    ]:

        return "PROTECT_INVENTORY"


    # Payment problems
    if root_cause in [
        "PAYMENT_FAILURE",
        "PAYMENT_FRICTION"
    ]:

        return "FIX_PAYMENT"


    # Funnel problem
    if root_cause == "FUNNEL_DROP_OFF":

        return "RETARGET"


    # Creative problem
    if root_cause in [
        "CREATIVE_WEAKNESS",
        "CREATIVE_CONVERSION_GAP"
    ]:

        return "CHANGE_CREATIVE"


    # Demand decline
    if root_cause == "DEMAND_DECLINE":

        return "DECREASE_BUDGET"


    # Profitability issue
    if root_cause in [
        "PROFITABILITY_RISK",
        "MARGIN_PRESSURE"
    ]:

        return "DECREASE_BUDGET"


    # Strong performer
    if (
        root_cause == "HIGH_PERFORMER"
        and predicted_roas >= 8
        and predicted_profit > 0
        and inventory == "HEALTHY"
    ):

        return "INCREASE_BUDGET"


    # Moderate performance
    if (
        predicted_roas >= 5
        and predicted_profit > 0
    ):

        return "MAINTAIN"


    return "DECREASE_BUDGET"


df["autonomous_action"] = (
    df.apply(
        decide_action,
        axis=1
    )
)


# ============================================================
# STEP 16 - PRIORITY SCORE
# ============================================================

def priority_score(row):

    action = row["autonomous_action"]

    opportunity = (
        row["predicted_next_day_roas"]
    )

    score = 50


    if action == "INCREASE_BUDGET":
        score += 35

    elif action == "PROTECT_INVENTORY":
        score += 30

    elif action == "FIX_PAYMENT":
        score += 30

    elif action == "DECREASE_BUDGET":
        score += 25

    elif action == "CHANGE_CREATIVE":
        score += 20

    elif action == "RETARGET":
        score += 20


    if opportunity >= 10:
        score += 10

    elif opportunity < 4:
        score += 5


    return min(score, 100)


df["decision_priority_score"] = (
    df.apply(
        priority_score,
        axis=1
    )
)


# ============================================================
# STEP 17 - CONFIDENCE
# ============================================================

def confidence(row):

    confidence = 60

    root = row["root_cause"]

    roas = row["predicted_next_day_roas"]

    inventory = row["days_of_inventory_remaining"]


    if roas >= 8:
        confidence += 15

    if roas < 4:
        confidence += 10

    if root in [
        "INVENTORY_SHORTAGE",
        "INVENTORY_RISK",
        "PAYMENT_FAILURE",
        "PAYMENT_FRICTION"
    ]:
        confidence += 10

    if inventory <= 0:
        confidence += 5


    return min(
        confidence,
        100
    )


df["decision_confidence_pct"] = (
    df.apply(
        confidence,
        axis=1
    )
)


# ============================================================
# STEP 18 - FINAL DECISION TABLE
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

    "predicted_next_day_roas",

    "predicted_next_day_revenue_inr",

    "predicted_next_day_profit_inr",

    "margin_pct",

    "sales_volume",

    "sales_growth_pct",

    "stock_on_hand",

    "days_of_inventory_remaining",

    "inventory_risk",

    "click_rate_pct",

    "add_to_cart_rate_pct",

    "purchase_rate_pct",

    "payment_failed_rate_pct",

    "funnel_status",

    "payment_status",

    "creative_status",

    "sales_status",

    "profitability_status",

    "root_cause",

    "root_cause_explanation",

    "autonomous_action",

    "decision_priority_score",

    "decision_confidence_pct"
]


result = df[output_columns].copy()


# ============================================================
# STEP 19 - SAVE COMPLETE RESULTS
# ============================================================

result.to_csv(
    "models/root_cause_decisions.csv",
    index=False
)


# ============================================================
# STEP 20 - LATEST DAY
# ============================================================

latest_date = result["date"].max()

latest = result[
    result["date"] == latest_date
].copy()


latest = latest.sort_values(
    "decision_priority_score",
    ascending=False
)


latest.to_csv(
    "models/latest_root_cause_decisions.csv",
    index=False
)


# ============================================================
# STEP 21 - PRINT LATEST DECISIONS
# ============================================================

print("\n")
print("=" * 65)
print("LATEST ROOT CAUSE ANALYSIS")
print("=" * 65)

print(
    "\nDate:",
    latest_date.date()
)


display_columns = [

    "sku",

    "product",

    "predicted_next_day_roas",

    "days_of_inventory_remaining",

    "root_cause",

    "autonomous_action",

    "decision_confidence_pct"

]


print(
    latest[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# STEP 22 - ROOT CAUSE SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("ROOT CAUSE SUMMARY")
print("=" * 65)

root_summary = (
    latest["root_cause"]
    .value_counts()
)

print(
    root_summary.to_string()
)


# ============================================================
# STEP 23 - ACTION SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("AUTONOMOUS ACTION SUMMARY")
print("=" * 65)

action_summary = (
    latest["autonomous_action"]
    .value_counts()
)

print(
    action_summary.to_string()
)


# ============================================================
# STEP 24 - HIGH PRIORITY DECISIONS
# ============================================================

print("\n")
print("=" * 65)
print("HIGH PRIORITY DECISIONS")
print("=" * 65)


high_priority = latest[
    latest["decision_priority_score"] >= 75
]


for _, row in high_priority.iterrows():

    print("\n----------------------------------------")

    print(
        "Product:",
        row["product"]
    )

    print(
        "SKU:",
        row["sku"]
    )

    print(
        "Predicted ROAS:",
        round(
            row["predicted_next_day_roas"],
            2
        )
    )

    print(
        "Root Cause:",
        row["root_cause"]
    )

    print(
        "Why:",
        row["root_cause_explanation"]
    )

    print(
        "Action:",
        row["autonomous_action"]
    )

    print(
        "Confidence:",
        str(
            row["decision_confidence_pct"]
        ) + "%"
    )


# ============================================================
# STEP 25 - FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 65)
print("STEP 10 COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nFiles created:")

print(
    "1. models/root_cause_decisions.csv"
)

print(
    "2. models/latest_root_cause_decisions.csv"
)

print("\n")
print("ADPULSE AI now has:")
print("ROAS Prediction")
print("Root Cause Diagnosis")
print("Autonomous Decisioning")