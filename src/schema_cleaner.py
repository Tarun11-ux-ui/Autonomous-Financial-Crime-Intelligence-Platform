from pathlib import Path

import pandas as pd


CANONICAL_COLUMN_ORDER = [
    "account_id",
    "is_mule",
    "risk_score",
    "risk_level",
    "suspicious_start",
    "suspicious_end",
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
    "community_id",
    "fraud_ring_flag",
    "influential_node_flag",
    "primary_action",
    "action_priority",
    "action_owner",
    "auto_action_eligible",
    "freeze_account",
    "alert_compliance_team",
    "reduce_transaction_limit",
    "enhanced_monitoring",
    "decision_confidence",
    "decision_summary",
    "decision_reasons",
    "explanation_generated",
    "models_used_for_explanation",
    "explanation_summary",
    "model_top_drivers",
    "model_driver_1",
    "model_driver_2",
    "model_driver_3",
    "human_readable_reasoning",
    "total_amt",
    "incoming_total",
    "outgoing_total",
    "graph_total_amount",
    "estimated_exposure_inr",
    "action_prevention_multiplier",
    "estimated_fraud_prevented_inr",
    "case_generated",
    "impact_priority_score",
    "review_sla_hours",
    "investigation_queue",
    "impact_band",
    "ring_case_flag",
    "connected_risky_accounts",
    "investigator_brief",
    "investigator_keywords",
    "retrieval_text",
    "retriever_document_id",
    "retriever_source_type",
    "retriever_document",
    "retriever_ready",
    "knowledge_store_ready",
]


PREFERRED_DUPLICATES = {
    "graph_risk_score": ["graph_risk_score", "graph_risk_score_x", "graph_risk_score_y"],
    "community_risk_score": ["community_risk_score", "community_risk_score_x", "community_risk_score_y"],
    "fraud_ring_flag": ["fraud_ring_flag", "fraud_ring_flag_x", "fraud_ring_flag_y"],
}


def _pick_first_non_null(frame, candidates):
    available = [column for column in candidates if column in frame.columns]
    if not available:
        return None

    result = frame[available[0]].copy()
    for column in available[1:]:
        result = result.combine_first(frame[column])
    return result


def clean_submission_schema(df):
    cleaned = df.copy()

    for canonical, candidates in PREFERRED_DUPLICATES.items():
        chosen = _pick_first_non_null(cleaned, candidates)
        if chosen is not None:
            cleaned[canonical] = chosen
        for column in candidates[1:]:
            if column in cleaned.columns:
                cleaned = cleaned.drop(columns=[column])

    dup_suffix_columns = [
        column for column in cleaned.columns
        if column.endswith("_x") or column.endswith("_y")
    ]
    if dup_suffix_columns:
        cleaned = cleaned.drop(columns=dup_suffix_columns)

    ordered = [column for column in CANONICAL_COLUMN_ORDER if column in cleaned.columns]
    remaining = [column for column in cleaned.columns if column not in ordered]
    return cleaned[ordered + remaining]


def clean_submission_csv(path="data/submission.csv"):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Submission file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cleaned = clean_submission_schema(df)
    cleaned.to_csv(csv_path, index=False)
    return cleaned


if __name__ == "__main__":
    cleaned_df = clean_submission_csv()
    print(f"Cleaned submission.csv with shape {cleaned_df.shape}")
