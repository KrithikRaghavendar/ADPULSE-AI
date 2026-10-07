import pandas as pd
import numpy as np


# ============================================================
# ADPULSE AI - CLOSED LOOP FEEDBACK & LEARNING ENGINE
# ============================================================

print("=" * 70)
print("ADPULSE AI - CLOSED LOOP FEEDBACK & LEARNING ENGINE")
print("=" * 70)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

dataset = pd.read_csv(
    "data/dataset.csv"
)

decisions = pd.read_csv(
    "models/root_cause_decisions.csv"
)

print("\nDataset loaded.")
print("Historical rows:", len(dataset))
print("Decision rows:", len(decisions))


# ============================================================
# STEP 2 - DATE PROCESSING
# ============================================================

dataset["date"] = pd.to_datetime(
    dataset["date"]
)

decisions["date"] = pd.to_datetime(
    decisions["date"]
)


# ============================================================
# STEP 3 - SELECT OUTCOME DATA
# ============================================================

outcome_columns = [
    "date",
    "sku",
    "recommended_action",
    "next_day_roas",
    "next_day_profit_inr",
    "next_day_revenue_inr",
    "next_day_sales_volume",
    "next_day_inventory_days",
    "next_day_stockout",
    "baseline_roas",
    "baseline_profit_inr",
    "roas_change_pct",
    "profit_change_pct",
    "action_success",
    "decision_confidence_pct",
    "opportunity_score"
]

outcomes = dataset[
    outcome_columns
].copy()


# ============================================================
# STEP 4 - KEEP ONLY ROWS WITH FUTURE OUTCOME
# ============================================================

outcomes = outcomes[
    outcomes["next_day_roas"].notna()
].copy()

print("\nHistorical decision outcomes available:")
print(len(outcomes))


# ============================================================
# STEP 5 - FIX ACTION SUCCESS VALUES
# ============================================================

print("\nOriginal action_success values:")
print(
    outcomes["action_success"]
    .value_counts(dropna=False)
)


def convert_success(value):

    if pd.isna(value):
        return np.nan

    # Numeric values
    try:

        numeric_value = float(value)

        if numeric_value == 1:
            return 1

        if numeric_value == 0:
            return 0

    except:

        pass


    # Text values
    text = str(value).strip().lower()

    if text in [
        "true",
        "yes",
        "success",
        "successful",
        "1",
        "1.0"
    ]:

        return 1


    if text in [
        "false",
        "no",
        "failure",
        "failed",
        "0",
        "0.0"
    ]:

        return 0


    return np.nan


outcomes["action_success"] = (
    outcomes["action_success"]
    .apply(convert_success)
)


print("\nCleaned action_success values:")
print(
    outcomes["action_success"]
    .value_counts(dropna=False)
)


# ============================================================
# STEP 6 - CALCULATE PERFORMANCE IMPROVEMENT
# ============================================================

outcomes["roas_improvement"] = (
    outcomes["next_day_roas"]
    -
    outcomes["baseline_roas"]
)


outcomes["profit_improvement"] = (
    outcomes["next_day_profit_inr"]
    -
    outcomes["baseline_profit_inr"]
)


# ============================================================
# STEP 7 - RE-CALCULATE SUCCESS IF NECESSARY
# ============================================================

# If action_success is missing, derive success from actual
# business outcome.

missing_success = (
    outcomes["action_success"].isna()
)


for index in outcomes[missing_success].index:

    roas_change = outcomes.loc[
        index,
        "roas_change_pct"
    ]

    profit_change = outcomes.loc[
        index,
        "profit_change_pct"
    ]


    if (
        pd.notna(roas_change)
        and
        pd.notna(profit_change)
    ):

        if (
            roas_change > 0
            and
            profit_change > 0
        ):

            outcomes.loc[
                index,
                "action_success"
            ] = 1

        else:

            outcomes.loc[
                index,
                "action_success"
            ] = 0


print("\nFinal success values:")
print(
    outcomes["action_success"]
    .value_counts(dropna=False)
)


# ============================================================
# STEP 8 - FEEDBACK STATUS
# ============================================================

def feedback_status(row):

    success = row["action_success"]

    roas_change = row["roas_change_pct"]

    profit_change = row["profit_change_pct"]


    if success == 1:

        return "SUCCESS"


    if (
        pd.notna(roas_change)
        and
        pd.notna(profit_change)
        and
        roas_change > 0
        and
        profit_change > 0
    ):

        return "POSITIVE_OUTCOME"


    if (
        pd.notna(roas_change)
        and
        pd.notna(profit_change)
        and
        roas_change < 0
        and
        profit_change < 0
    ):

        return "NEGATIVE_OUTCOME"


    return "MIXED_OUTCOME"


outcomes["feedback_status"] = (
    outcomes.apply(
        feedback_status,
        axis=1
    )
)


# ============================================================
# STEP 9 - ACTION PERFORMANCE
# ============================================================

action_performance = (
    outcomes
    .groupby("recommended_action")
    .agg(

        decisions=(
            "recommended_action",
            "count"
        ),

        successful_decisions=(
            "action_success",
            "sum"
        ),

        success_rate=(
            "action_success",
            "mean"
        ),

        average_next_day_roas=(
            "next_day_roas",
            "mean"
        ),

        average_roas_change=(
            "roas_change_pct",
            "mean"
        ),

        average_profit_change=(
            "profit_change_pct",
            "mean"
        ),

        average_next_day_profit=(
            "next_day_profit_inr",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# STEP 10 - CONVERT SUCCESS RATE TO %
# ============================================================

action_performance[
    "success_rate"
] = (
    action_performance[
        "success_rate"
    ]
    * 100
)


action_performance[
    "success_rate"
] = (
    action_performance[
        "success_rate"
    ]
    .round(2)
)


# ============================================================
# STEP 11 - LEARNING SCORE
# ============================================================

def calculate_learning_score(row):

    score = 50.0


    # Success rate
    if pd.notna(row["success_rate"]):

        score += (
            row["success_rate"] - 50
        ) * 0.5


    # ROAS improvement
    if (
        pd.notna(row["average_roas_change"])
    ):

        if row["average_roas_change"] > 0:

            score += 10

        elif row["average_roas_change"] < 0:

            score -= 10


    # Profit improvement
    if (
        pd.notna(row["average_profit_change"])
    ):

        if row["average_profit_change"] > 0:

            score += 15

        elif row["average_profit_change"] < 0:

            score -= 15


    return np.clip(
        score,
        0,
        100
    )


action_performance[
    "learning_score"
] = (
    action_performance
    .apply(
        calculate_learning_score,
        axis=1
    )
    .round(2)
)


# ============================================================
# STEP 12 - ACTION QUALITY
# ============================================================

def action_quality(score):

    if score >= 75:

        return "STRONG"

    elif score >= 55:

        return "GOOD"

    elif score >= 40:

        return "WEAK"

    else:

        return "POOR"


action_performance[
    "action_quality"
] = (
    action_performance[
        "learning_score"
    ]
    .apply(action_quality)
)


# ============================================================
# STEP 13 - PRINT HISTORICAL PERFORMANCE
# ============================================================

print("\n")
print("=" * 70)
print("HISTORICAL ACTION PERFORMANCE")
print("=" * 70)

print(
    action_performance[
        [
            "recommended_action",
            "decisions",
            "successful_decisions",
            "success_rate",
            "average_next_day_roas",
            "average_roas_change",
            "average_profit_change",
            "learning_score",
            "action_quality"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 14 - GET LATEST DECISIONS
# ============================================================

latest_date = decisions["date"].max()

latest_decisions = decisions[
    decisions["date"] == latest_date
].copy()


# ============================================================
# STEP 15 - CURRENT ACTION
# ============================================================

if "autonomous_action" in latest_decisions.columns:

    latest_decisions[
        "current_action"
    ] = latest_decisions[
        "autonomous_action"
    ]

else:

    latest_decisions[
        "current_action"
    ] = latest_decisions[
        "recommended_action"
    ]


# ============================================================
# STEP 16 - MERGE HISTORICAL LEARNING
# ============================================================

learning_columns = [
    "recommended_action",
    "success_rate",
    "learning_score",
    "action_quality"
]


latest_decisions = latest_decisions.merge(

    action_performance[
        learning_columns
    ],

    left_on="current_action",

    right_on="recommended_action",

    how="left"
)


# ============================================================
# STEP 17 - HISTORICAL SUCCESS RATE
# ============================================================

latest_decisions[
    "historical_action_success_rate"
] = (
    latest_decisions[
        "success_rate"
    ]
    .fillna(50)
    .round(2)
)


# ============================================================
# STEP 18 - LEARNED CONFIDENCE
# ============================================================

if "decision_confidence_pct" in latest_decisions.columns:

    latest_decisions[
        "learned_decision_confidence"
    ] = (

        latest_decisions[
            "decision_confidence_pct"
        ].fillna(50)

        * 0.60

        +

        latest_decisions[
            "historical_action_success_rate"
        ] * 0.40

    ).round(2)

else:

    latest_decisions[
        "learned_decision_confidence"
    ] = (
        latest_decisions[
            "historical_action_success_rate"
        ]
    )


# ============================================================
# STEP 19 - LEARNING ADJUSTMENT
# ============================================================

def learning_adjustment(row):

    success_rate = (
        row["historical_action_success_rate"]
    )

    action = row["current_action"]


    if success_rate >= 75:

        return "INCREASE_CONFIDENCE"


    if success_rate <= 40:

        return "REVIEW_ACTION_POLICY"


    if action == "PROTECT_INVENTORY":

        return "MAINTAIN_PROTECTIVE_POLICY"


    return "MAINTAIN_POLICY"


latest_decisions[
    "learning_adjustment"
] = (
    latest_decisions
    .apply(
        learning_adjustment,
        axis=1
    )
)


# ============================================================
# STEP 20 - POLICY RECOMMENDATION
# ============================================================

def policy_recommendation(row):

    success_rate = (
        row["historical_action_success_rate"]
    )

    action = row["current_action"]


    if success_rate >= 75:

        return (
            f"Historical {action} decisions "
            f"show strong outcomes "
            f"({success_rate:.1f}% success). "
            f"Keep this policy."
        )


    if success_rate <= 40:

        return (
            f"Historical {action} decisions "
            f"have weak outcomes "
            f"({success_rate:.1f}% success). "
            f"Review or retrain this policy."
        )


    return (
        f"Historical {action} decisions "
        f"have moderate outcomes "
        f"({success_rate:.1f}% success). "
        f"Continue monitoring."
    )


latest_decisions[
    "policy_recommendation"
] = (
    latest_decisions
    .apply(
        policy_recommendation,
        axis=1
    )
)


# ============================================================
# STEP 21 - SAVE FEEDBACK HISTORY
# ============================================================

outcomes.to_csv(
    "models/decision_feedback_history.csv",
    index=False
)


# ============================================================
# STEP 22 - SAVE ACTION LEARNING
# ============================================================

action_performance.to_csv(
    "models/action_learning_performance.csv",
    index=False
)


# ============================================================
# STEP 23 - SAVE LEARNED DECISIONS
# ============================================================

latest_decisions.to_csv(
    "models/latest_learned_decisions.csv",
    index=False
)


# ============================================================
# STEP 24 - PRINT CURRENT LEARNED DECISIONS
# ============================================================

print("\n")
print("=" * 70)
print("CURRENT DECISIONS + HISTORICAL LEARNING")
print("=" * 70)

print(
    "\nLatest date:",
    latest_date.date()
)


display_columns = [
    "sku",
    "product",
    "current_action"
]


if "predicted_next_day_roas" in latest_decisions.columns:

    display_columns.append(
        "predicted_next_day_roas"
    )


if "root_cause" in latest_decisions.columns:

    display_columns.append(
        "root_cause"
    )


display_columns.extend([
    "historical_action_success_rate",
    "learned_decision_confidence",
    "learning_adjustment"
])


print(
    latest_decisions[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# STEP 25 - BEST ACTION POLICY
# ============================================================

print("\n")
print("=" * 70)
print("BEST PERFORMING ACTION POLICY")
print("=" * 70)


best_action = (
    action_performance
    .sort_values(
        [
            "learning_score",
            "success_rate"
        ],
        ascending=False
    )
    .iloc[0]
)


print(
    "\nAction:",
    best_action[
        "recommended_action"
    ]
)


print(
    "Success Rate:",
    f"{best_action['success_rate']:.2f}%"
)


print(
    "Learning Score:",
    best_action[
        "learning_score"
    ]
)


print(
    "Quality:",
    best_action[
        "action_quality"
    ]
)


# ============================================================
# STEP 26 - POLICIES NEEDING REVIEW
# ============================================================

print("\n")
print("=" * 70)
print("POLICIES THAT NEED REVIEW")
print("=" * 70)


review_actions = action_performance[
    action_performance[
        "action_quality"
    ] == "POOR"
]


if len(review_actions) == 0:

    print(
        "\nNo action policy currently requires urgent review."
    )

else:

    print(
        review_actions[
            [
                "recommended_action",
                "success_rate",
                "learning_score"
            ]
        ].to_string(index=False)
    )


# ============================================================
# STEP 27 - SYSTEM STATUS
# ============================================================

print("\n")
print("=" * 70)
print("CLOSED LOOP STATUS")
print("=" * 70)

print("\nPrediction       : COMPLETE")
print("Diagnosis        : COMPLETE")
print("Decision         : COMPLETE")
print("Outcome Tracking : COMPLETE")
print("Feedback         : COMPLETE")
print("Policy Learning  : COMPLETE")


# ============================================================
# STEP 28 - FILES
# ============================================================

print("\n")
print("=" * 70)
print("STEP 11 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nFiles created:")

print(
    "1. models/decision_feedback_history.csv"
)

print(
    "2. models/action_learning_performance.csv"
)

print(
    "3. models/latest_learned_decisions.csv"
)

print("\nADPULSE AI now supports CLOSED-LOOP LEARNING.")