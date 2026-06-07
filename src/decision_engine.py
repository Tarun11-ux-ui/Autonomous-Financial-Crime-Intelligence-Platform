import pandas as pd


DECISION_FEATURES = [
    "burst_count",
    "rapid_drain",
    "rapid_drain_ratio",
    "night_txn_count",
    "near_threshold_count",
    "high_risk_exposure",
    "high_risk_exposure_ratio",
    "unique_counterparty",
    "txn_velocity",
    "top_counterparty_ratio",
    "flow_through_score",
    "graph_risk_score",
    "community_risk_score",
    "community_size",
    "fraud_ring_flag",
    "influential_node_flag",
]

DECISION_DEFAULTS = {
    "burst_count": 0.0,
    "rapid_drain": 0.0,
    "rapid_drain_ratio": 0.0,
    "night_txn_count": 0.0,
    "near_threshold_count": 0.0,
    "high_risk_exposure": 0.0,
    "high_risk_exposure_ratio": 0.0,
    "unique_counterparty": 0.0,
    "txn_velocity": 0.0,
    "top_counterparty_ratio": 0.0,
    "flow_through_score": 0.0,
    "graph_risk_score": 0.0,
    "community_risk_score": 0.0,
    "community_size": 0.0,
    "fraud_ring_flag": "NO",
    "influential_node_flag": "NO",
}


def _safe_float(row, column, default=0.0):
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def _has_suspicious_window(row):
    return row.get("suspicious_start", "N/A") != "N/A"


def _build_reasons(row):
    reasons = []

    risk_score = _safe_float(row, "risk_score")
    burst_count = _safe_float(row, "burst_count")
    rapid_drain = _safe_float(row, "rapid_drain")
    rapid_drain_ratio = _safe_float(row, "rapid_drain_ratio")
    night_txn_count = _safe_float(row, "night_txn_count")
    near_threshold_count = _safe_float(row, "near_threshold_count")
    high_risk_exposure = _safe_float(row, "high_risk_exposure")
    high_risk_exposure_ratio = _safe_float(row, "high_risk_exposure_ratio")
    unique_counterparty = _safe_float(row, "unique_counterparty")
    txn_velocity = _safe_float(row, "txn_velocity")
    top_counterparty_ratio = _safe_float(row, "top_counterparty_ratio")
    flow_through_score = _safe_float(row, "flow_through_score")
    graph_risk_score = _safe_float(row, "graph_risk_score")
    community_risk_score = _safe_float(row, "community_risk_score")
    community_size = _safe_float(row, "community_size")

    if risk_score >= 90:
        reasons.append(f"Model risk score is critical at {risk_score:.1f}")
    elif risk_score >= 75:
        reasons.append(f"Model risk score is elevated at {risk_score:.1f}")

    if _has_suspicious_window(row):
        reasons.append("Suspicious transaction window detected")

    if burst_count >= 5:
        reasons.append(f"{int(burst_count)} rapid transaction bursts observed")

    if rapid_drain >= 3 or rapid_drain_ratio >= 0.25:
        reasons.append("Rapid outward fund movement suggests possible draining behaviour")

    if night_txn_count >= 3:
        reasons.append(f"{int(night_txn_count)} transactions occurred during night hours")

    if near_threshold_count >= 2:
        reasons.append("Repeated near-threshold transactions may indicate structuring")

    if high_risk_exposure >= 2 or high_risk_exposure_ratio >= 0.15:
        reasons.append("Exposure to high-risk counterparties is above baseline")

    if unique_counterparty >= 8:
        reasons.append(f"Connected to {int(unique_counterparty)} distinct counterparties")

    if txn_velocity >= 10:
        reasons.append("Transaction velocity is unusually high")

    if top_counterparty_ratio >= 0.7:
        reasons.append("Funds are concentrated around a dominant counterparty")

    if flow_through_score <= 0.15 and risk_score >= 70:
        reasons.append("Pass-through flow pattern is consistent with mule behaviour")

    if graph_risk_score >= 75:
        reasons.append(f"Graph intelligence score is elevated at {graph_risk_score:.1f}")

    if community_risk_score >= 70 and community_size >= 3:
        reasons.append(f"Account is embedded in a suspicious network community of size {int(community_size)}")

    if row.get("fraud_ring_flag", "NO") == "YES":
        reasons.append("Fraud-ring detection flagged the connected account cluster")

    if row.get("influential_node_flag", "NO") == "YES":
        reasons.append("Node influence in the transaction network is unusually high")

    if not reasons:
        reasons.append("No auto-action trigger fired; continue standard monitoring")

    return reasons


def _build_decision(row):
    risk_score = _safe_float(row, "risk_score")
    burst_count = _safe_float(row, "burst_count")
    rapid_drain = _safe_float(row, "rapid_drain")
    rapid_drain_ratio = _safe_float(row, "rapid_drain_ratio")
    near_threshold_count = _safe_float(row, "near_threshold_count")
    high_risk_exposure_ratio = _safe_float(row, "high_risk_exposure_ratio")
    unique_counterparty = _safe_float(row, "unique_counterparty")
    txn_velocity = _safe_float(row, "txn_velocity")
    top_counterparty_ratio = _safe_float(row, "top_counterparty_ratio")
    graph_risk_score = _safe_float(row, "graph_risk_score")
    community_risk_score = _safe_float(row, "community_risk_score")

    suspicious_window = _has_suspicious_window(row)
    repeated_anomalies = burst_count >= 5 or rapid_drain >= 3 or near_threshold_count >= 2
    suspicious_chain_detected = (
        high_risk_exposure_ratio >= 0.15
        and unique_counterparty >= 8
    ) or top_counterparty_ratio >= 0.7 or graph_risk_score >= 75 or community_risk_score >= 70
    rapid_movement = rapid_drain_ratio >= 0.25 or txn_velocity >= 10

    if risk_score >= 90 and (suspicious_window or suspicious_chain_detected or repeated_anomalies):
        primary_action = "FREEZE_ACCOUNT"
        action_priority = "P1"
        action_owner = "Fraud Operations"
        auto_action = "YES"
    elif risk_score >= 80 and suspicious_chain_detected:
        primary_action = "ESCALATE_TO_COMPLIANCE"
        action_priority = "P1"
        action_owner = "Compliance Team"
        auto_action = "YES"
    elif risk_score >= 70 and (repeated_anomalies or rapid_movement):
        primary_action = "REDUCE_TRANSACTION_LIMIT"
        action_priority = "P2"
        action_owner = "Risk Controls"
        auto_action = "YES"
    elif risk_score >= 50:
        primary_action = "ENHANCED_MONITORING"
        action_priority = "P3"
        action_owner = "Monitoring Queue"
        auto_action = "NO"
    else:
        primary_action = "ALLOW_WITH_MONITORING"
        action_priority = "P4"
        action_owner = "Monitoring Queue"
        auto_action = "NO"

    return {
        "primary_action": primary_action,
        "action_priority": action_priority,
        "action_owner": action_owner,
        "auto_action_eligible": auto_action,
        "freeze_account": "YES" if primary_action == "FREEZE_ACCOUNT" else "NO",
        "alert_compliance_team": "YES" if primary_action in {"FREEZE_ACCOUNT", "ESCALATE_TO_COMPLIANCE"} else "NO",
        "reduce_transaction_limit": "YES" if primary_action == "REDUCE_TRANSACTION_LIMIT" else "NO",
        "enhanced_monitoring": "YES" if primary_action in {"ENHANCED_MONITORING", "ALLOW_WITH_MONITORING"} else "NO",
        "decision_confidence": round(min(0.99, max(0.30, risk_score / 100.0)), 2),
        "decision_summary": _decision_summary(primary_action, suspicious_window, suspicious_chain_detected, repeated_anomalies, rapid_movement),
    }


def _decision_summary(primary_action, suspicious_window, suspicious_chain_detected, repeated_anomalies, rapid_movement):
    trigger_flags = []

    if suspicious_window:
        trigger_flags.append("window anomaly")
    if suspicious_chain_detected:
        trigger_flags.append("network risk")
    if repeated_anomalies:
        trigger_flags.append("repeat anomalies")
    if rapid_movement:
        trigger_flags.append("rapid movement")

    if not trigger_flags:
        trigger_flags.append("baseline monitoring")

    return f"{primary_action} triggered by " + ", ".join(trigger_flags)


def apply_decision_engine(scored_df, feature_df):
    """
    Convert model scores into operational actions and reason codes.

    Args:
        scored_df: Submission-like dataframe with account_id, risk_score, risk_level,
            suspicious_start, and suspicious_end columns.
        feature_df: Feature-enriched account dataframe used to score the account.

    Returns:
        DataFrame enriched with decision-engine outputs.
    """
    available_features = [feature for feature in DECISION_FEATURES if feature in feature_df.columns]
    feature_slice = feature_df[["account_id", *available_features]].copy()

    decision_df = scored_df.merge(feature_slice, on="account_id", how="left")
    for feature_name, default_value in DECISION_DEFAULTS.items():
        if feature_name not in decision_df.columns:
            decision_df[feature_name] = default_value
        else:
            decision_df[feature_name] = decision_df[feature_name].fillna(default_value)

    decisions = decision_df.apply(lambda row: pd.Series(_build_decision(row)), axis=1)
    decision_df = pd.concat([decision_df, decisions], axis=1)
    decision_df["decision_reasons"] = decision_df.apply(
        lambda row: " | ".join(_build_reasons(row)),
        axis=1,
    )

    return decision_df
