import pandas as pd


IMPACT_FEATURES = [
    "total_amt",
    "incoming_total",
    "outgoing_total",
    "graph_total_amount",
    "graph_risk_score",
    "community_risk_score",
    "community_id",
    "fraud_ring_flag",
]

ACTION_PREVENTION_MULTIPLIER = {
    "FREEZE_ACCOUNT": 0.90,
    "ESCALATE_TO_COMPLIANCE": 0.65,
    "REDUCE_TRANSACTION_LIMIT": 0.45,
    "ENHANCED_MONITORING": 0.20,
    "ALLOW_WITH_MONITORING": 0.05,
}

ACTION_SLA_HOURS = {
    "FREEZE_ACCOUNT": 0.25,
    "ESCALATE_TO_COMPLIANCE": 1.0,
    "REDUCE_TRANSACTION_LIMIT": 2.0,
    "ENHANCED_MONITORING": 8.0,
    "ALLOW_WITH_MONITORING": 24.0,
}


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _build_exposure(series_df):
    total_amt = _safe_numeric(series_df.get("total_amt", 0))
    incoming_total = _safe_numeric(series_df.get("incoming_total", 0))
    outgoing_total = _safe_numeric(series_df.get("outgoing_total", 0))
    graph_total_amount = _safe_numeric(series_df.get("graph_total_amount", 0))

    return (
        total_amt * 0.30
        + incoming_total * 0.25
        + outgoing_total * 0.35
        + graph_total_amount * 0.10
    )


def apply_impact_metrics(submission_df, feature_df):
    """
    Attach dashboard-style impact metrics directly to the submission rows.
    """
    available_features = [column for column in IMPACT_FEATURES if column in feature_df.columns]
    feature_slice = feature_df[["account_id", *available_features]].copy()
    impact_df = submission_df.merge(feature_slice, on="account_id", how="left")

    for column in IMPACT_FEATURES:
        if column not in impact_df.columns:
            impact_df[column] = 0 if column != "fraud_ring_flag" else "NO"

    numeric_fill_cols = [column for column in IMPACT_FEATURES if column != "fraud_ring_flag"]
    impact_df[numeric_fill_cols] = impact_df[numeric_fill_cols].apply(_safe_numeric)
    impact_df["fraud_ring_flag"] = impact_df["fraud_ring_flag"].fillna("NO")

    impact_df["estimated_exposure_inr"] = _build_exposure(impact_df).round(2)
    impact_df["action_prevention_multiplier"] = impact_df["primary_action"].map(ACTION_PREVENTION_MULTIPLIER).fillna(0.05)
    impact_df["estimated_fraud_prevented_inr"] = (
        impact_df["estimated_exposure_inr"]
        * impact_df["action_prevention_multiplier"]
        * (impact_df["risk_score"] / 100.0)
    ).round(2)

    impact_df["case_generated"] = (
        (impact_df["risk_score"] >= 50)
        | (impact_df["auto_action_eligible"] == "YES")
        | (impact_df["suspicious_start"] != "N/A")
    ).map({True: "YES", False: "NO"})

    impact_df["impact_priority_score"] = (
        impact_df["risk_score"] * 0.45
        + impact_df["decision_confidence"] * 100 * 0.20
        + impact_df["graph_risk_score"] * 0.20
        + impact_df["community_risk_score"] * 0.15
    ).round(2)

    impact_df["review_sla_hours"] = impact_df["primary_action"].map(ACTION_SLA_HOURS).fillna(24.0)
    impact_df["investigation_queue"] = impact_df["primary_action"].map(
        {
            "FREEZE_ACCOUNT": "Fraud Operations Queue",
            "ESCALATE_TO_COMPLIANCE": "Compliance Review Queue",
            "REDUCE_TRANSACTION_LIMIT": "Risk Controls Queue",
            "ENHANCED_MONITORING": "Enhanced Monitoring Queue",
            "ALLOW_WITH_MONITORING": "Baseline Monitoring Queue",
        }
    ).fillna("Baseline Monitoring Queue")

    impact_df["impact_band"] = pd.cut(
        impact_df["impact_priority_score"],
        bins=[-1, 35, 60, 80, 200],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    ).astype(str)
    impact_df["ring_case_flag"] = (
        (impact_df["fraud_ring_flag"] == "YES")
        | (impact_df["community_risk_score"] >= 70)
    ).map({True: "YES", False: "NO"})

    return impact_df


def summarize_impact_metrics(submission_df):
    """
    Create a compact operational summary for console/dashboard usage.
    """
    total_accounts = len(submission_df)
    cases_generated = int((submission_df["case_generated"] == "YES").sum())
    auto_actions = int((submission_df["auto_action_eligible"] == "YES").sum())

    if total_accounts == 0:
        return {
            "total_accounts_screened": 0,
            "cases_generated": 0,
            "auto_actions_triggered": 0,
            "freeze_actions": 0,
            "compliance_alerts": 0,
            "limit_reductions": 0,
            "ring_cases": 0,
            "estimated_fraud_prevented_inr": 0.0,
            "average_risk_score": 0.0,
            "case_generation_rate": 0.0,
            "auto_action_rate": 0.0,
            "top_priority_account": "N/A",
        }

    sorted_df = submission_df.sort_values("impact_priority_score", ascending=False)

    return {
        "total_accounts_screened": total_accounts,
        "cases_generated": cases_generated,
        "auto_actions_triggered": auto_actions,
        "freeze_actions": int((submission_df["freeze_account"] == "YES").sum()),
        "compliance_alerts": int((submission_df["alert_compliance_team"] == "YES").sum()),
        "limit_reductions": int((submission_df["reduce_transaction_limit"] == "YES").sum()),
        "ring_cases": int((submission_df["ring_case_flag"] == "YES").sum()),
        "estimated_fraud_prevented_inr": round(float(submission_df["estimated_fraud_prevented_inr"].sum()), 2),
        "average_risk_score": round(float(submission_df["risk_score"].mean()), 2),
        "case_generation_rate": round(cases_generated / total_accounts * 100, 2),
        "auto_action_rate": round(auto_actions / total_accounts * 100, 2),
        "top_priority_account": str(sorted_df.iloc[0]["account_id"]),
    }
