import pandas as pd
import numpy as np


# ============================================================
# ADPULSE AI - FINAL INTELLIGENCE REPORT
# ============================================================

print("=" * 70)
print("ADPULSE AI - FINAL AUTONOMOUS INTELLIGENCE REPORT")
print("=" * 70)


# ============================================================
# STEP 1 - LOAD FINAL DATA
# ============================================================

decisions = pd.read_csv(
    "models/latest_learned_decisions.csv"
)

feedback = pd.read_csv(
    "models/action_learning_performance.csv"
)

print("\nFinal decision data loaded.")
print("Decision rows:", len(decisions))
print("Action policies:", len(feedback))


# ============================================================
# STEP 2 - IDENTIFY LATEST DATE
# ============================================================

if "date" in decisions.columns:

    decisions["date"] = pd.to_datetime(
        decisions["date"]
    )

    latest_date = decisions["date"].max()

else:

    latest_date = None


# ============================================================
# STEP 3 - TOTAL PRODUCTS
# ============================================================

total_products = len(decisions)


# ============================================================
# STEP 4 - ACTION DISTRIBUTION
# ============================================================

action_distribution = (
    decisions[
        "current_action"
    ]
    .value_counts()
)


# ============================================================
# STEP 5 - TOP OPPORTUNITIES
# ============================================================

if "opportunity_score" in decisions.columns:

    top_opportunities = (
        decisions
        .sort_values(
            "opportunity_score",
            ascending=False
        )
        .head(5)
    )

else:

    top_opportunities = decisions.head(5)


# ============================================================
# STEP 6 - TOP PREDICTED ROAS
# ============================================================

if "predicted_next_day_roas" in decisions.columns:

    top_roas = (
        decisions
        .sort_values(
            "predicted_next_day_roas",
            ascending=False
        )
        .head(5)
    )

else:

    top_roas = decisions.head(5)


# ============================================================
# STEP 7 - HIGH CONFIDENCE DECISIONS
# ============================================================

high_confidence = decisions[
    decisions[
        "learned_decision_confidence"
    ] >= 70
]


# ============================================================
# STEP 8 - LOW CONFIDENCE DECISIONS
# ============================================================

low_confidence = decisions[
    decisions[
        "learned_decision_confidence"
    ] < 50
]


# ============================================================
# STEP 9 - POLICIES REQUIRING REVIEW
# ============================================================

review_policies = feedback[
    feedback[
        "action_quality"
    ] == "POOR"
]


# ============================================================
# STEP 10 - STRONG POLICIES
# ============================================================

strong_policies = feedback[
    feedback[
        "action_quality"
    ] == "STRONG"
]


# ============================================================
# STEP 11 - PRINT SYSTEM OVERVIEW
# ============================================================

print("\n")
print("=" * 70)
print("SYSTEM OVERVIEW")
print("=" * 70)

print(
    "\nLatest data date:",
    latest_date.date()
    if latest_date is not None
    else "Unknown"
)

print(
    "Products evaluated:",
    total_products
)

print(
    "High-confidence decisions:",
    len(high_confidence)
)

print(
    "Low-confidence decisions:",
    len(low_confidence)
)


# ============================================================
# STEP 12 - ACTION DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("AUTONOMOUS ACTION DISTRIBUTION")
print("=" * 70)


for action, count in action_distribution.items():

    percentage = (
        count
        /
        total_products
        *
        100
    )

    print(
        f"{action:<25} "
        f"{count:>3} products "
        f"({percentage:.1f}%)"
    )


# ============================================================
# STEP 13 - TOP OPPORTUNITIES
# ============================================================

print("\n")
print("=" * 70)
print("TOP 5 BUSINESS OPPORTUNITIES")
print("=" * 70)


for index, row in top_opportunities.iterrows():

    print("\nProduct:", row["product"])
    print("SKU:", row["sku"])

    if "predicted_next_day_roas" in row:

        print(
            "Predicted ROAS:",
            round(
                row[
                    "predicted_next_day_roas"
                ],
                2
            )
        )

    if "opportunity_score" in row:

        print(
            "Opportunity Score:",
            round(
                row[
                    "opportunity_score"
                ],
                2
            )
        )

    print(
        "Action:",
        row["current_action"]
    )

    print(
        "Confidence:",
        f"{row['learned_decision_confidence']:.2f}%"
    )

    if "root_cause" in row:

        print(
            "Root Cause:",
            row["root_cause"]
        )


# ============================================================
# STEP 14 - TOP ROAS PRODUCTS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 5 PREDICTED ROAS PRODUCTS")
print("=" * 70)


for index, row in top_roas.iterrows():

    print(
        f"{row['sku']:<22}"
        f" ROAS: "
        f"{row['predicted_next_day_roas']:.2f}"
    )


# ============================================================
# STEP 15 - HIGH CONFIDENCE DECISIONS
# ============================================================

print("\n")
print("=" * 70)
print("HIGH CONFIDENCE DECISIONS")
print("=" * 70)


if len(high_confidence) == 0:

    print(
        "\nNo decisions currently exceed 70% confidence."
    )

else:

    for index, row in high_confidence.iterrows():

        print(
            f"{row['sku']:<22}"
            f" {row['current_action']:<22}"
            f" {row['learned_decision_confidence']:.2f}%"
        )


# ============================================================
# STEP 16 - LOW CONFIDENCE DECISIONS
# ============================================================

print("\n")
print("=" * 70)
print("LOW CONFIDENCE DECISIONS")
print("=" * 70)


if len(low_confidence) == 0:

    print(
        "\nNo low-confidence decisions."
    )

else:

    for index, row in low_confidence.iterrows():

        print(
            f"{row['sku']:<22}"
            f" {row['current_action']:<22}"
            f" {row['learned_decision_confidence']:.2f}%"
        )


# ============================================================
# STEP 17 - STRONG ACTION POLICIES
# ============================================================

print("\n")
print("=" * 70)
print("STRONG ACTION POLICIES")
print("=" * 70)


if len(strong_policies) == 0:

    print(
        "\nNo strong policies."
    )

else:

    for index, row in strong_policies.iterrows():

        print(
            f"{row['recommended_action']:<25}"
            f" Success: "
            f"{row['success_rate']:.2f}%"
            f" | Learning Score: "
            f"{row['learning_score']:.2f}"
        )


# ============================================================
# STEP 18 - POLICIES NEEDING REVIEW
# ============================================================

print("\n")
print("=" * 70)
print("POLICIES NEEDING REVIEW")
print("=" * 70)


if len(review_policies) == 0:

    print(
        "\nNo policies require immediate review."
    )

else:

    for index, row in review_policies.iterrows():

        print(
            f"{row['recommended_action']:<25}"
            f" Success: "
            f"{row['success_rate']:.2f}%"
            f" | Learning Score: "
            f"{row['learning_score']:.2f}"
        )


# ============================================================
# STEP 19 - ROOT CAUSE DISTRIBUTION
# ============================================================

if "root_cause" in decisions.columns:

    print("\n")
    print("=" * 70)
    print("ROOT CAUSE DISTRIBUTION")
    print("=" * 70)

    root_causes = (
        decisions[
            "root_cause"
        ]
        .value_counts()
    )

    for cause, count in root_causes.items():

        print(
            f"{cause:<30}"
            f"{count:>3}"
        )


# ============================================================
# STEP 20 - BUSINESS RISK FLAGS
# ============================================================

print("\n")
print("=" * 70)
print("BUSINESS RISK FLAGS")
print("=" * 70)


risk_count = 0


if "root_cause" in decisions.columns:

    inventory_risk = decisions[
        decisions[
            "root_cause"
        ].astype(str).str.contains(
            "INVENTORY",
            case=False,
            na=False
        )
    ]

    payment_risk = decisions[
        decisions[
            "root_cause"
        ].astype(str).str.contains(
            "PAYMENT",
            case=False,
            na=False
        )
    ]

    funnel_risk = decisions[
        decisions[
            "root_cause"
        ].astype(str).str.contains(
            "FUNNEL",
            case=False,
            na=False
        )
    ]


    print(
        "Inventory risk products:",
        len(inventory_risk)
    )

    print(
        "Payment friction products:",
        len(payment_risk)
    )

    print(
        "Funnel drop-off products:",
        len(funnel_risk)
    )


    risk_count = (
        len(inventory_risk)
        +
        len(payment_risk)
        +
        len(funnel_risk)
    )


# ============================================================
# STEP 21 - EXECUTIVE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EXECUTIVE AI SUMMARY")
print("=" * 70)


best_policy = (
    feedback
    .sort_values(
        "learning_score",
        ascending=False
    )
    .iloc[0]
)


print("\nADPULSE AI evaluated")
print(
    f"{total_products} products/SKUs."
)


print(
    "\nThe strongest historical decision policy was:"
)


print(
    f"{best_policy['recommended_action']} "
    f"with "
    f"{best_policy['success_rate']:.2f}% "
    f"success rate."
)


print(
    "\nThe system identified"
    f" {len(high_confidence)} "
    "high-confidence decisions."
)


print(
    f"{len(review_policies)} "
    "decision policies currently require review."
)


# ============================================================
# STEP 22 - SAVE FINAL REPORT
# ============================================================

report_columns = [
    "sku",
    "product",
    "current_action"
]


for column in [
    "predicted_next_day_roas",
    "predicted_next_day_profit_inr",
    "opportunity_score",
    "root_cause",
    "historical_action_success_rate",
    "learned_decision_confidence",
    "learning_adjustment",
    "policy_recommendation"
]:

    if column in decisions.columns:

        report_columns.append(
            column
        )


final_report = decisions[
    report_columns
].copy()


final_report.to_csv(
    "models/final_ai_decision_report.csv",
    index=False
)


# ============================================================
# STEP 23 - FINAL STATUS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 12 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print(
    "\nFinal report created:"
)

print(
    "models/final_ai_decision_report.csv"
)


print("\nADPULSE AI PIPELINE STATUS")

print("\n1. Data Ingestion       : COMPLETE")
print("2. Data Reconciliation : COMPLETE")
print("3. ROAS Prediction     : COMPLETE")
print("4. Root Cause Analysis : COMPLETE")
print("5. Opportunity Scoring : COMPLETE")
print("6. Decision Engine     : COMPLETE")
print("7. Outcome Tracking    : COMPLETE")
print("8. Feedback Learning   : COMPLETE")
print("9. Final Intelligence  : COMPLETE")

print("\nADPULSE AI IS READY FOR DEMONSTRATION.")