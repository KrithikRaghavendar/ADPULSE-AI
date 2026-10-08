"""
STEPS 1-7 — TRAIN THE NEXT-DAY ROAS MODEL

The training stage that produces the files train_roas.py (step 8) and the
decision engines read. It reproduces the format of the original Glowroots
model package exactly, so the rest of the pipeline runs unchanged:

    models/roas_predictor.pkl          {model, preprocessor, features, ...}
    models/roas_predictions.csv        date, actual_roas, baseline_prediction, random_forest_prediction
    models/roas_feature_importance.csv feature, importance
    models/baseline_roas_predictions.csv

Run it from a company folder that contains data/dataset.csv:

    cd companies/<company> && python ../../train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "next_day_roas"
TEST_DAYS = 6

CATEGORICAL_FEATURES = [
    "sku", "product", "category", "variant", "ad_type", "product_sales_trend",
]

NUMERICAL_FEATURES = [
    "impressions", "clicks", "click_rate_pct", "product_page_views_from_ads",
    "ad_spend_inr", "conversions", "roas", "sales_volume", "sales_growth_pct",
    "repeat_purchaser_count", "stock_on_hand", "days_of_inventory_remaining",
    "stockout_frequency_30d", "inventory_velocity_units_per_day", "revenue_inr",
    "profit_inr", "affiliate_link_interactions", "video_3sec_views",
    "video_25pct_watched", "video_50pct_watched", "video_75pct_watched",
    "video_100pct_watched", "engagement_rate_pct", "likes", "comments", "shares",
    "saves", "product_clicks", "add_to_cart", "add_to_cart_rate_pct",
    "checkout_started", "payment_initiated", "payment_failed",
    "payment_failed_rate_pct", "purchase_rate_pct", "return_rate_pct",
    "refund_rate_pct", "margin_pct", "day_of_week", "day_of_month", "month",
]


# ==========================================
# STEP 1 - LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/dataset.csv")

print("Rows:", len(df))


# ==========================================
# STEP 2 - FEATURE ENGINEERING
# ==========================================

df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month

for column in NUMERICAL_FEATURES + [TARGET]:
    df[column] = pd.to_numeric(
        df[column].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )

labelled = df.dropna(subset=[TARGET]).sort_values(["date", "sku"])


# ==========================================
# STEP 3 - TIME-BASED TRAIN / TEST SPLIT
# ==========================================

dates = sorted(labelled["date"].unique())

if len(dates) <= TEST_DAYS + 3:
    raise SystemExit(f"Need more than {TEST_DAYS + 3} days of data to train; found {len(dates)}.")

test_start = dates[-TEST_DAYS]
train = labelled[labelled["date"] < test_start]
test = labelled[labelled["date"] >= test_start]

print(f"Train rows: {len(train)}  ·  Test rows: {len(test)} (last {TEST_DAYS} days)")


# ==========================================
# STEP 4 - PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer(transformers=[
    ("numerical", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERICAL_FEATURES),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_FEATURES),
])

features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# ==========================================
# STEP 5 - TRAIN RANDOM FOREST
# ==========================================

model = RandomForestRegressor(
    n_estimators=300, max_depth=12, min_samples_split=4,
    min_samples_leaf=2, n_jobs=-1, random_state=42
)

X_train = preprocessor.fit_transform(train[features])
model.fit(X_train, train[TARGET])

print("Random Forest trained.")


# ==========================================
# STEP 6 - PREDICT TEST WINDOW + BASELINE
# ==========================================

predictions = pd.DataFrame({
    "date": test["date"].dt.strftime("%Y-%m-%d").values,
    "actual_roas": test[TARGET].values,
    "baseline_prediction": test["roas"].values,          # today's ROAS carried forward
    "random_forest_prediction": model.predict(preprocessor.transform(test[features])),
})

importance = pd.DataFrame({
    "feature": preprocessor.get_feature_names_out(),
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)


# ==========================================
# STEP 7 - SAVE MODEL PACKAGE + OUTPUTS
# ==========================================

Path("models").mkdir(exist_ok=True)

joblib.dump({
    "model": model,
    "preprocessor": preprocessor,
    "features": features,
    "categorical_features": CATEGORICAL_FEATURES,
    "numerical_features": NUMERICAL_FEATURES,
    "target": TARGET,
}, "models/roas_predictor.pkl")

predictions.to_csv("models/roas_predictions.csv", index=False)
importance.to_csv("models/roas_feature_importance.csv", index=False)
predictions[["date", "actual_roas", "baseline_prediction"]].rename(
    columns={"baseline_prediction": "baseline_predicted_roas"}
).to_csv("models/baseline_roas_predictions.csv", index=False)

mae = np.mean(np.abs(predictions["actual_roas"] - predictions["random_forest_prediction"]))
base = np.mean(np.abs(predictions["actual_roas"] - predictions["baseline_prediction"]))

print(f"Test MAE — Random Forest: {mae:.3f}  ·  baseline: {base:.3f}")
print("Saved models/roas_predictor.pkl and prediction outputs.")
