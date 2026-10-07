import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# STEP 8 - LOAD MODEL PREDICTIONS
# ==========================================

print("Loading model predictions...")

predictions = pd.read_csv(
    "models/roas_predictions.csv"
)

feature_importance = pd.read_csv(
    "models/roas_feature_importance.csv"
)


# ==========================================
# STEP 8.1 - CHECK DATA
# ==========================================

print("\n========================================")
print("PREDICTION DATA")
print("========================================")

print("Rows:", len(predictions))

print("\nColumns:")
print(predictions.columns.tolist())

print("\nFirst 10 predictions:")
print(
    predictions.head(10).to_string(index=False)
)


# ==========================================
# STEP 8.2 - EXTRACT VALUES
# ==========================================

actual = predictions["actual_roas"]

baseline = predictions["baseline_prediction"]

random_forest = predictions[
    "random_forest_prediction"
]


# ==========================================
# STEP 8.3 - BASELINE METRICS
# ==========================================

baseline_mae = mean_absolute_error(
    actual,
    baseline
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual,
        baseline
    )
)

baseline_r2 = r2_score(
    actual,
    baseline
)


# ==========================================
# STEP 8.4 - RANDOM FOREST METRICS
# ==========================================

rf_mae = mean_absolute_error(
    actual,
    random_forest
)

rf_rmse = np.sqrt(
    mean_squared_error(
        actual,
        random_forest
    )
)

rf_r2 = r2_score(
    actual,
    random_forest
)


# ==========================================
# STEP 8.5 - IMPROVEMENT CALCULATION
# ==========================================

if baseline_mae != 0:

    mae_improvement = (
        (baseline_mae - rf_mae)
        / baseline_mae
    ) * 100

else:

    mae_improvement = 0


if baseline_rmse != 0:

    rmse_improvement = (
        (baseline_rmse - rf_rmse)
        / baseline_rmse
    ) * 100

else:

    rmse_improvement = 0


# ==========================================
# STEP 8.6 - MODEL COMPARISON
# ==========================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(
    "\nMetric                  Baseline       Random Forest"
)

print(
    "------------------------------------------------------"
)

print(
    f"MAE                     {baseline_mae:.4f}         {rf_mae:.4f}"
)

print(
    f"RMSE                    {baseline_rmse:.4f}         {rf_rmse:.4f}"
)

print(
    f"R²                      {baseline_r2:.4f}         {rf_r2:.4f}"
)


# ==========================================
# STEP 8.7 - IMPROVEMENT
# ==========================================

print("\n========================================")
print("MODEL IMPROVEMENT")
print("========================================")

print(
    f"MAE improvement : {mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: {rmse_improvement:.2f}%"
)


# ==========================================
# STEP 8.8 - DETERMINE MODEL QUALITY
# ==========================================

print("\n========================================")
print("MODEL VERDICT")
print("========================================")

if rf_mae < baseline_mae:

    print(
        "Random Forest performs better than the baseline."
    )

    print(
        "The ML model is learning useful patterns."
    )

    winner = "Random Forest"

elif rf_mae > baseline_mae:

    print(
        "Baseline performs better than Random Forest."
    )

    print(
        "The current dataset may need further feature engineering."
    )

    winner = "Baseline"

else:

    print(
        "Both models have similar performance."
    )

    winner = "Tie"


print(
    f"\nBest model based on MAE: {winner}"
)


# ==========================================
# STEP 8.9 - TOP FEATURES
# ==========================================

print("\n========================================")
print("TOP ROAS DRIVERS")
print("========================================")

top_features = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .head(20)
)


for index, row in top_features.iterrows():

    print(
        f"{row['feature']:<55} "
        f"{row['importance']:.6f}"
    )


# ==========================================
# STEP 8.10 - SAVE TOP FEATURES
# ==========================================

top_features.to_csv(
    "models/top_roas_drivers.csv",
    index=False
)


# ==========================================
# STEP 8.11 - PREDICTION ERROR
# ==========================================

predictions["prediction_error"] = (
    predictions["actual_roas"]
    - predictions["random_forest_prediction"]
)

predictions["absolute_error"] = (
    predictions["prediction_error"]
    .abs()
)


# ==========================================
# STEP 8.12 - BEST AND WORST PREDICTIONS
# ==========================================

best_predictions = (
    predictions
    .sort_values(
        "absolute_error"
    )
    .head(10)
)

worst_predictions = (
    predictions
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(10)
)


print("\n========================================")
print("BEST PREDICTIONS")
print("========================================")

print(
    best_predictions[
        [
            "date",
            "actual_roas",
            "random_forest_prediction",
            "absolute_error"
        ]
    ].to_string(index=False)
)


print("\n========================================")
print("WORST PREDICTIONS")
print("========================================")

print(
    worst_predictions[
        [
            "date",
            "actual_roas",
            "random_forest_prediction",
            "absolute_error"
        ]
    ].to_string(index=False)
)


# ==========================================
# STEP 8.13 - SAVE EVALUATED PREDICTIONS
# ==========================================

predictions.to_csv(
    "models/evaluated_roas_predictions.csv",
    index=False
)


# ==========================================
# STEP 8.14 - SAVE EVALUATION SUMMARY
# ==========================================

evaluation_summary = pd.DataFrame({
    "metric": [
        "MAE",
        "RMSE",
        "R2",
        "MAE_Improvement_Percent",
        "RMSE_Improvement_Percent"
    ],

    "baseline": [
        baseline_mae,
        baseline_rmse,
        baseline_r2,
        0,
        0
    ],

    "random_forest": [
        rf_mae,
        rf_rmse,
        rf_r2,
        mae_improvement,
        rmse_improvement
    ]
})


evaluation_summary.to_csv(
    "models/model_evaluation.csv",
    index=False
)


# ==========================================
# STEP 8.15 - FINAL OUTPUT
# ==========================================

print("\n========================================")
print("STEP 8 COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nFiles created:")

print(
    "1. models/evaluated_roas_predictions.csv"
)

print(
    "2. models/top_roas_drivers.csv"
)

print(
    "3. models/model_evaluation.csv"
)

print("\nModel evaluation completed.")