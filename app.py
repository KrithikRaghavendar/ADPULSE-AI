import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ADPULSE AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FINAL_REPORT = MODELS_DIR / "final_ai_decision_report.csv"
ACTION_PERFORMANCE = MODELS_DIR / "action_learning_performance.csv"
LEARNED_DECISIONS = MODELS_DIR / "latest_learned_decisions.csv"
ROOT_CAUSE = MODELS_DIR / "latest_root_cause_decisions.csv"
DECISION_RESULTS = MODELS_DIR / "decision_engine_results.csv"
ROAS_PREDICTIONS = MODELS_DIR / "roas_predictions.csv"
MODEL_EVALUATION = MODELS_DIR / "model_evaluation.csv"
FEATURE_IMPORTANCE = MODELS_DIR / "roas_feature_importance.csv"


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data
def load_csv(path):

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


final_df = load_csv(FINAL_REPORT)
action_df = load_csv(ACTION_PERFORMANCE)
learned_df = load_csv(LEARNED_DECISIONS)
root_df = load_csv(ROOT_CAUSE)
decision_df = load_csv(DECISION_RESULTS)
prediction_df = load_csv(ROAS_PREDICTIONS)
evaluation_df = load_csv(MODEL_EVALUATION)
feature_df = load_csv(FEATURE_IMPORTANCE)


# ============================================================
# SAFETY CHECK
# ============================================================

if final_df.empty:

    st.error(
        "Could not load models/final_ai_decision_report.csv"
    )

    st.stop()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "predicted_next_day_roas",
    "predicted_next_day_profit_inr",
    "opportunity_score",
    "learned_decision_confidence",
    "historical_action_success_rate",
    "next_day_roas",
    "roas",
    "profit_inr",
    "revenue_inr",
    "ad_spend_inr",
    "success_rate_pct",
    "learning_score",
    "decision_count",
    "success_count"
]


for df in [
    final_df,
    action_df,
    learned_df,
    root_df,
    decision_df,
    prediction_df,
    evaluation_df,
    feature_df
]:

    if df.empty:
        continue

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    if df.empty:
        return None

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def find_success_rate_column(df):

    if df.empty:
        return None

    exact_candidates = [
        "success_rate_pct",
        "success_rate",
        "action_success_rate",
        "historical_action_success_rate",
        "historical_success_rate",
        "success_percentage",
        "success_percent"
    ]

    column = find_column(
        df,
        exact_candidates
    )

    if column is not None:
        return column

    for column in df.columns:

        name = str(column).lower()

        if (
            "success" in name
            and (
                "rate" in name
                or "percent" in name
                or "%" in name
            )
        ):

            return column

    return None


def find_action_column(df):

    if df.empty:
        return None

    candidates = [
        "action",
        "recommended_action",
        "current_action",
        "autonomous_action",
        "policy_action"
    ]

    column = find_column(
        df,
        candidates
    )

    if column is not None:
        return column

    for column in df.columns:

        name = str(column).lower()

        if "action" in name:

            return column

    return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ ADPULSE AI")

st.sidebar.write(
    "Autonomous D2C Advertising "
    "Intelligence & Decision Engine"
)

st.sidebar.divider()

st.sidebar.subheader("Dashboard Controls")


# ============================================================
# ACTION FILTER
# ============================================================

action_filter_column = find_column(
    final_df,
    [
        "current_action",
        "recommended_action",
        "autonomous_action"
    ]
)


action_options = ["All"]


if action_filter_column is not None:

    action_options += sorted(
        final_df[
            action_filter_column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


selected_action = st.sidebar.selectbox(
    "Autonomous Action",
    action_options
)


# ============================================================
# ROOT CAUSE FILTER
# ============================================================

root_filter_column = find_column(
    final_df,
    [
        "root_cause",
        "primary_root_cause"
    ]
)


root_options = ["All"]


if root_filter_column is not None:

    root_options += sorted(
        final_df[
            root_filter_column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


selected_root = st.sidebar.selectbox(
    "Root Cause",
    root_options
)


# ============================================================
# CONFIDENCE FILTER
# ============================================================

confidence_column = find_column(
    final_df,
    [
        "learned_decision_confidence",
        "decision_confidence_pct",
        "decision_confidence",
        "confidence_pct"
    ]
)


selected_confidence = st.sidebar.selectbox(
    "Decision Confidence",
    [
        "All",
        "High Confidence",
        "Low Confidence"
    ]
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = final_df.copy()


if (
    selected_action != "All"
    and action_filter_column is not None
):

    filtered_df = filtered_df[
        filtered_df[
            action_filter_column
        ].astype(str)
        == selected_action
    ]


if (
    selected_root != "All"
    and root_filter_column is not None
):

    filtered_df = filtered_df[
        filtered_df[
            root_filter_column
        ].astype(str)
        == selected_root
    ]


if (
    selected_confidence == "High Confidence"
    and confidence_column is not None
):

    filtered_df = filtered_df[
        filtered_df[
            confidence_column
        ] >= 70
    ]


if (
    selected_confidence == "Low Confidence"
    and confidence_column is not None
):

    filtered_df = filtered_df[
        filtered_df[
            confidence_column
        ] < 50
    ]


# ============================================================
# HEADER
# ============================================================

st.title("⚡ ADPULSE AI")

st.subheader(
    "Next-Generation Autonomous D2C "
    "Advertising Intelligence & Decision Engine"
)

st.write(
    "Predict → Diagnose → Decide → Learn"
)

st.divider()

st.info(
    "⚠️ Hackathon prototype using simulated scenario data. "
    "Metrics demonstrate the AI decision workflow and are "
    "not production-measured business results."
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_skus = len(final_df)


high_confidence = 0

if confidence_column is not None:

    high_confidence = int(
        (
            final_df[
                confidence_column
            ] >= 70
        ).sum()
    )


avg_roas = 0

roas_column = find_column(
    final_df,
    [
        "predicted_next_day_roas",
        "predicted_roas"
    ]
)


if roas_column is not None:

    avg_roas = final_df[
        roas_column
    ].mean()


profit_column = find_column(
    final_df,
    [
        "predicted_next_day_profit_inr",
        "predicted_profit_inr",
        "predicted_profit"
    ]
)


total_profit = 0


if profit_column is not None:

    total_profit = final_df[
        profit_column
    ].sum()


risk_alerts = 0


if root_filter_column is not None:

    risk_alerts = int(
        final_df[
            root_filter_column
        ]
        .astype(str)
        .str.contains(
            "INVENTORY|PAYMENT|FUNNEL",
            case=False,
            regex=True
        )
        .sum()
    )


# ============================================================
# EXECUTIVE KPIs
# ============================================================

st.subheader("📊 Executive KPIs")


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.metric(
        "SKUs Evaluated",
        total_skus
    )


with k2:

    st.metric(
        "High-Confidence Decisions",
        high_confidence
    )


with k3:

    st.metric(
        "Avg Predicted ROAS",
        f"{avg_roas:.2f}"
    )


with k4:

    st.metric(
        "Predicted Profit",
        f"₹{total_profit:,.0f}"
    )


with k5:

    st.metric(
        "Business Risk Signals",
        risk_alerts
    )


# ============================================================
# EXECUTIVE INTELLIGENCE
# ============================================================

st.subheader("🏆 Executive Intelligence Summary")


policy_action_column = find_action_column(
    action_df
)

policy_success_column = find_success_rate_column(
    action_df
)


strongest_policy = "N/A"
strongest_rate = 0.0


if (
    policy_action_column is not None
    and policy_success_column is not None
):

    policy_temp = action_df.copy()


    policy_temp[
        policy_success_column
    ] = pd.to_numeric(
        policy_temp[
            policy_success_column
        ],
        errors="coerce"
    )


    policy_temp = policy_temp.dropna(
        subset=[
            policy_success_column
        ]
    )


    if not policy_temp.empty:

        best_index = policy_temp[
            policy_success_column
        ].idxmax()


        best_row = policy_temp.loc[
            best_index
        ]


        strongest_policy = str(
            best_row[
                policy_action_column
            ]
        )


        strongest_rate = float(
            best_row[
                policy_success_column
            ]
        )


        # If stored as decimal instead of percentage
        if strongest_rate <= 1:

            strongest_rate *= 100


s1, s2, s3 = st.columns(3)


with s1:

    st.info(
        f"🧠 **Strongest Learned Policy**\n\n"
        f"**{strongest_policy}** achieved "
        f"**{strongest_rate:.2f}%** historical "
        f"success in the prototype feedback dataset."
    )


with s2:

    st.info(
        f"🎯 **Decision Intelligence**\n\n"
        f"The engine evaluated **{total_skus} SKUs** "
        f"and generated **{high_confidence} "
        f"high-confidence decisions**."
    )


with s3:

    st.info(
        "🔄 **Closed-Loop Learning**\n\n"
        "Historical action outcomes adjust future "
        "decision confidence and identify policies "
        "requiring further review."
    )


# ============================================================
# AUTONOMOUS DECISION ENGINE
# ============================================================

st.subheader("🤖 Autonomous Decision Engine")


sku_column = find_column(
    filtered_df,
    ["sku"]
)

product_column = find_column(
    filtered_df,
    ["product"]
)


decision_columns = []


for column in [
    sku_column,
    product_column,
    roas_column,
    profit_column,
    root_filter_column,
    action_filter_column,
    confidence_column
]:

    if (
        column is not None
        and column not in decision_columns
    ):

        decision_columns.append(
            column
        )


decision_table = filtered_df[
    decision_columns
].copy()


rename_dictionary = {}


if sku_column is not None:
    rename_dictionary[
        sku_column
    ] = "SKU"


if product_column is not None:
    rename_dictionary[
        product_column
    ] = "Product"


if roas_column is not None:
    rename_dictionary[
        roas_column
    ] = "Predicted ROAS"


if profit_column is not None:
    rename_dictionary[
        profit_column
    ] = "Predicted Profit"


if root_filter_column is not None:
    rename_dictionary[
        root_filter_column
    ] = "Root Cause"


if action_filter_column is not None:
    rename_dictionary[
        action_filter_column
    ] = "Autonomous Action"


if confidence_column is not None:
    rename_dictionary[
        confidence_column
    ] = "Confidence %"


decision_table = decision_table.rename(
    columns=rename_dictionary
)


if "Predicted ROAS" in decision_table.columns:

    decision_table[
        "Predicted ROAS"
    ] = decision_table[
        "Predicted ROAS"
    ].round(2)


if "Predicted Profit" in decision_table.columns:

    decision_table[
        "Predicted Profit"
    ] = decision_table[
        "Predicted Profit"
    ].round(0)


if "Confidence %" in decision_table.columns:

    decision_table[
        "Confidence %"
    ] = decision_table[
        "Confidence %"
    ].round(2)


st.dataframe(
    decision_table,
    width="stretch",
    hide_index=True
)


# ============================================================
# AI DECISION EXPLAINER
# ============================================================

st.subheader(
    "🔎 Why Did the AI Make This Decision?"
)


if (
    not filtered_df.empty
    and product_column is not None
):

    products = (
        filtered_df[
            product_column
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


    selected_product = st.selectbox(
        "Select a product",
        products
    )


    selected_rows = filtered_df[
        filtered_df[
            product_column
        ].astype(str)
        == selected_product
    ]


    if not selected_rows.empty:

        row = selected_rows.iloc[0]


        e1, e2, e3, e4 = st.columns(4)


        with e1:

            if roas_column is not None:

                st.metric(
                    "Predicted ROAS",
                    f"{float(row[roas_column]):.2f}"
                )


        with e2:

            if profit_column is not None:

                st.metric(
                    "Predicted Profit",
                    f"₹{float(row[profit_column]):,.0f}"
                )


        with e3:

            if root_filter_column is not None:

                st.metric(
                    "Root Cause",
                    str(
                        row[
                            root_filter_column
                        ]
                    )
                )


        with e4:

            if confidence_column is not None:

                confidence = float(
                    row[
                        confidence_column
                    ]
                )


                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


        if action_filter_column is not None:

            st.success(
                "🤖 **Autonomous Action:** "
                + str(
                    row[
                        action_filter_column
                    ]
                )
            )


        if "learning_adjustment" in row:

            st.write(
                "**Learning Adjustment:**",
                row["learning_adjustment"]
            )


        if "policy_recommendation" in row:

            st.write(
                "**Policy Recommendation:**",
                row["policy_recommendation"]
            )


else:

    st.warning(
        "No products match the selected filters."
    )


# ============================================================
# TOP 5 OPPORTUNITIES
# ============================================================

st.subheader(
    "🚀 Top 5 Growth Opportunities"
)


if roas_column is not None:

    opportunity_columns = []


    for column in [
        sku_column,
        product_column,
        roas_column,
        profit_column,
        find_column(
            final_df,
            ["opportunity_score"]
        ),
        action_filter_column
    ]:

        if (
            column is not None
            and column not in opportunity_columns
        ):

            opportunity_columns.append(
                column
            )


    opportunities = (
        final_df[
            opportunity_columns
        ]
        .sort_values(
            roas_column,
            ascending=False
        )
        .head(5)
        .copy()
    )


    opportunity_rename = {}


    if sku_column is not None:
        opportunity_rename[
            sku_column
        ] = "SKU"


    if product_column is not None:
        opportunity_rename[
            product_column
        ] = "Product"


    opportunity_rename[
        roas_column
    ] = "Predicted ROAS"


    if profit_column is not None:
        opportunity_rename[
            profit_column
        ] = "Predicted Profit"


    opportunity_score_column = find_column(
        final_df,
        ["opportunity_score"]
    )


    if opportunity_score_column is not None:

        opportunity_rename[
            opportunity_score_column
        ] = "Opportunity Score"


    if action_filter_column is not None:

        opportunity_rename[
            action_filter_column
        ] = "Recommended Action"


    opportunities = opportunities.rename(
        columns=opportunity_rename
    )


    st.dataframe(
        opportunities,
        width="stretch",
        hide_index=True
    )


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

st.subheader(
    "🎯 Autonomous Action Distribution"
)


if action_filter_column is not None:

    action_counts = (
        final_df[
            action_filter_column
        ]
        .value_counts()
    )


    st.bar_chart(
        action_counts,
        width="stretch"
    )


# ============================================================
# ROOT CAUSE DISTRIBUTION
# ============================================================

st.subheader(
    "🧠 Root Cause Intelligence"
)


if root_filter_column is not None:

    root_counts = (
        final_df[
            root_filter_column
        ]
        .value_counts()
    )


    st.bar_chart(
        root_counts,
        width="stretch"
    )


# ============================================================
# BUSINESS RISK
# ============================================================

st.subheader(
    "🚨 Business Risk Monitor"
)


inventory_risk = 0
payment_risk = 0
funnel_risk = 0


if root_filter_column is not None:

    causes = final_df[
        root_filter_column
    ].astype(str)


    inventory_risk = int(
        causes.str.contains(
            "INVENTORY",
            case=False
        ).sum()
    )


    payment_risk = int(
        causes.str.contains(
            "PAYMENT",
            case=False
        ).sum()
    )


    funnel_risk = int(
        causes.str.contains(
            "FUNNEL",
            case=False
        ).sum()
    )


r1, r2, r3 = st.columns(3)


with r1:

    st.metric(
        "📦 Inventory Risk",
        inventory_risk
    )

    st.caption(
        "Products requiring inventory attention"
    )


with r2:

    st.metric(
        "💳 Payment Friction",
        payment_risk
    )

    st.caption(
        "Products requiring payment investigation"
    )


with r3:

    st.metric(
        "🛒 Funnel Drop-Off",
        funnel_risk
    )

    st.caption(
        "Products requiring conversion optimization"
    )


# ============================================================
# CLOSED-LOOP POLICY LEARNING
# ============================================================

st.subheader(
    "🔄 Closed-Loop Policy Learning"
)


if not action_df.empty:

    policy_action_column = find_action_column(
        action_df
    )

    policy_success_column = find_success_rate_column(
        action_df
    )


    policy_count_column = find_column(
        action_df,
        [
            "decision_count",
            "decisions",
            "count"
        ]
    )


    policy_success_count_column = find_column(
        action_df,
        [
            "success_count",
            "successes"
        ]
    )


    policy_learning_column = find_column(
        action_df,
        [
            "learning_score",
            "learning"
        ]
    )


    policy_strength_column = find_column(
        action_df,
        [
            "policy_strength",
            "strength"
        ]
    )


    policy_columns = []


    for column in [
        policy_action_column,
        policy_count_column,
        policy_success_count_column,
        policy_success_column,
        policy_learning_column,
        policy_strength_column
    ]:

        if (
            column is not None
            and column not in policy_columns
        ):

            policy_columns.append(
                column
            )


    policy_table = action_df[
        policy_columns
    ].copy()


    policy_rename = {}


    if policy_action_column is not None:
        policy_rename[
            policy_action_column
        ] = "Action"


    if policy_count_column is not None:
        policy_rename[
            policy_count_column
        ] = "Decisions"


    if policy_success_count_column is not None:
        policy_rename[
            policy_success_count_column
        ] = "Successes"


    if policy_success_column is not None:
        policy_rename[
            policy_success_column
        ] = "Success Rate %"


    if policy_learning_column is not None:
        policy_rename[
            policy_learning_column
        ] = "Learning Score"


    if policy_strength_column is not None:
        policy_rename[
            policy_strength_column
        ] = "Policy Strength"


    policy_table = policy_table.rename(
        columns=policy_rename
    )


    st.dataframe(
        policy_table,
        width="stretch",
        hide_index=True
    )


    # --------------------------------------------------------
    # LEARNING CHART
    # --------------------------------------------------------

    if (
        policy_action_column is not None
        and policy_success_column is not None
    ):

        learning_chart = action_df[
            [
                policy_action_column,
                policy_success_column
            ]
        ].copy()


        learning_chart[
            policy_success_column
        ] = pd.to_numeric(
            learning_chart[
                policy_success_column
            ],
            errors="coerce"
        )


        learning_chart = (
            learning_chart
            .dropna()
            .set_index(
                policy_action_column
            )
        )


        st.bar_chart(
            learning_chart,
            width="stretch"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader(
    "📊 ROAS Model Performance"
)


if not evaluation_df.empty:

    st.dataframe(
        evaluation_df,
        width="stretch",
        hide_index=True
    )

else:

    st.info(
        "Model evaluation output is not available."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🧠 Top ROAS Prediction Drivers"
)


if not feature_df.empty:

    st.dataframe(
        feature_df.head(10),
        width="stretch",
        hide_index=True
    )

else:

    st.info(
        "Feature importance output is not available."
    )


# ============================================================
# ROAS TREND
# ============================================================

st.subheader(
    "📈 ROAS Prediction Output"
)


chart_source = pd.DataFrame()


if not prediction_df.empty:

    numeric_prediction_columns = (
        prediction_df
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )


    if numeric_prediction_columns:

        chart_source = prediction_df[
            numeric_prediction_columns
        ].copy()


if chart_source.empty and not decision_df.empty:

    numeric_decision_columns = (
        decision_df
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )


    if numeric_decision_columns:

        chart_source = decision_df[
            numeric_decision_columns
        ].copy()


if not chart_source.empty:

    preferred_columns = [
        column
        for column in [
            "next_day_roas",
            "predicted_next_day_roas",
            "roas",
            "predicted_roas",
            "prediction"
        ]
        if column in chart_source.columns
    ]


    if preferred_columns:

        st.line_chart(
            chart_source[
                preferred_columns
            ],
            width="stretch"
        )

    else:

        st.line_chart(
            chart_source,
            width="stretch"
        )

else:

    st.info(
        "ROAS prediction chart data is not available."
    )


# ============================================================
# AI PIPELINE
# ============================================================

st.subheader(
    "⚙️ Autonomous AI Pipeline"
)


pipeline = pd.DataFrame(
    {
        "Stage": [
            "1. Data Ingestion",
            "2. Data Reconciliation",
            "3. ROAS Prediction",
            "4. Root Cause Analysis",
            "5. Opportunity Scoring",
            "6. Autonomous Decision",
            "7. Outcome Tracking",
            "8. Feedback Learning",
            "9. Final Intelligence"
        ],

        "Status": [
            "✅ COMPLETE",
            "✅ COMPLETE",
            "✅ COMPLETE",
            "✅ COMPLETE",
            "✅ COMPLETE",
            "✅ COMPLETE",
            "✅ COMPLETE",
            "✅ COMPLETE",
            "✅ COMPLETE"
        ]
    }
)


st.dataframe(
    pipeline,
    width="stretch",
    hide_index=True
)


# ============================================================
# WHY ADPULSE AI
# ============================================================

st.subheader(
    "💡 Why ADPULSE AI?"
)


v1, v2, v3 = st.columns(3)


with v1:

    st.info(
        "🔮 **PREDICTIVE**\n\n"
        "Forecasts next-day ROAS instead of only "
        "reporting historical campaign performance."
    )


with v2:

    st.info(
        "🧠 **AUTONOMOUS**\n\n"
        "Converts predictions and root-cause signals "
        "into recommended business actions."
    )


with v3:

    st.info(
        "🔄 **SELF-LEARNING**\n\n"
        "Tracks historical action outcomes and adjusts "
        "confidence in future decision policies."
    )


# ============================================================
# FINAL STATUS
# ============================================================

st.divider()

st.success(
    "⚡ ADPULSE AI IS READY FOR DEMONSTRATION"
)

st.caption(
    "DataQuest 3.0 Hackathon Prototype | "
    "Predict → Diagnose → Decide → Learn"
)