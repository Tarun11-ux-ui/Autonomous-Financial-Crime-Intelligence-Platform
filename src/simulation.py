import numpy as np
import pandas as pd
from src.decision_engine import apply_decision_engine
from src.graph_intelligence import build_graph_intelligence_from_transactions
from src.risk_scoring import compute_risk_score


SIMULATION_EVENTS = [
    {
        "timestamp": "2026-04-05 09:00:00",
        "stage": "Stage 1 - Account Activation",
        "scenario_event": "New mule account receives first inbound funds",
        "account_id": "SIM_MULE_01",
        "counterparty_id": "SIM_FUNDER_01",
        "amount": 42000,
        "txn_type": "C",
    },
    {
        "timestamp": "2026-04-05 09:04:00",
        "stage": "Stage 2 - Rapid Layering",
        "scenario_event": "Funds start moving through linked accounts",
        "account_id": "SIM_MULE_01",
        "counterparty_id": "SIM_RING_01",
        "amount": 20000,
        "txn_type": "D",
    },
    {
        "timestamp": "2026-04-05 09:06:00",
        "stage": "Stage 2 - Rapid Layering",
        "scenario_event": "Funds split toward another ring node",
        "account_id": "SIM_MULE_01",
        "counterparty_id": "SIM_RING_02",
        "amount": 19500,
        "txn_type": "D",
    },
    {
        "timestamp": "2026-04-05 09:08:00",
        "stage": "Stage 3 - Fraud Ring Expansion",
        "scenario_event": "Ring node forwards funds to third connected account",
        "account_id": "SIM_RING_01",
        "counterparty_id": "SIM_RING_03",
        "amount": 18000,
        "txn_type": "D",
    },
    {
        "timestamp": "2026-04-05 09:09:00",
        "stage": "Stage 3 - Fraud Ring Expansion",
        "scenario_event": "Another node returns funds to the mule pathway",
        "account_id": "SIM_RING_02",
        "counterparty_id": "SIM_MULE_01",
        "amount": 17500,
        "txn_type": "D",
    },
    {
        "timestamp": "2026-04-05 09:11:00",
        "stage": "Stage 4 - Burst Drain Attempt",
        "scenario_event": "Mule attempts rapid outward drain to cash-out account",
        "account_id": "SIM_MULE_01",
        "counterparty_id": "SIM_CASHOUT_01",
        "amount": 17000,
        "txn_type": "D",
    },
    {
        "timestamp": "2026-04-05 09:12:00",
        "stage": "Stage 4 - Burst Drain Attempt",
        "scenario_event": "Second burst drain transaction hits the network",
        "account_id": "SIM_MULE_01",
        "counterparty_id": "SIM_CASHOUT_02",
        "amount": 16800,
        "txn_type": "D",
    },
    {
        "timestamp": "2026-04-05 09:13:00",
        "stage": "Stage 5 - Auto Response",
        "scenario_event": "System should freeze mule account and alert compliance",
        "account_id": "SIM_MULE_01",
        "counterparty_id": "SIM_CASHOUT_03",
        "amount": 16500,
        "txn_type": "D",
    },
]


def _build_feature_snapshot(transactions, target_account):
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    df["amount"] = df["amount"].astype(float)
    df["is_in"] = (df["txn_type"] == "C").astype(int)
    df["is_out"] = (df["txn_type"] == "D").astype(int)
    df["hour"] = df["timestamp"].dt.hour
    df["night_txn"] = (df["hour"] <= 5).astype(int)
    df["time_diff"] = df.groupby("account_id")["timestamp"].diff().dt.total_seconds()
    df["burst_txn"] = (df["time_diff"] < 300).astype(int)
    df["rapid_out"] = ((df["is_out"] == 1) & (df["time_diff"] < 600)).astype(int)
    df["near_50k"] = ((df["amount"] > 49000) & (df["amount"] < 50000)).astype(int)

    cp_global_count = df["counterparty_id"].value_counts()
    high_risk_counterparties = cp_global_count[cp_global_count >= max(2, cp_global_count.max())].index
    df["high_risk_cp_flag"] = df["counterparty_id"].isin(high_risk_counterparties).astype(int)

    account_df = df[df["account_id"] == target_account].copy()
    if account_df.empty:
        return pd.DataFrame([{"account_id": target_account}])

    incoming_total = account_df.loc[account_df["is_in"] == 1, "amount"].sum()
    outgoing_total = account_df.loc[account_df["is_out"] == 1, "amount"].sum()
    cp_counts = account_df["counterparty_id"].value_counts()

    feature_row = {
        "account_id": target_account,
        "txn_count": float(len(account_df)),
        "total_amt": float(account_df["amount"].sum()),
        "avg_amt": float(account_df["amount"].mean()),
        "std_amt": float(account_df["amount"].std(ddof=0) if len(account_df) > 1 else 0.0),
        "max_amt": float(account_df["amount"].max()),
        "night_txn_count": float(account_df["night_txn"].sum()),
        "burst_count": float(account_df["burst_txn"].sum()),
        "near_threshold_count": float(account_df["near_50k"].sum()),
        "rapid_drain": float(account_df["rapid_out"].sum()),
        "high_risk_exposure": float(account_df["high_risk_cp_flag"].sum()),
        "unique_counterparty": float(account_df["counterparty_id"].nunique()),
        "incoming_total": float(incoming_total),
        "outgoing_total": float(outgoing_total),
        "out_degree": float(account_df.loc[account_df["is_out"] == 1, "counterparty_id"].nunique()),
        "in_degree": float(account_df.loc[account_df["is_in"] == 1, "counterparty_id"].nunique()),
    }

    active_days = max((account_df["timestamp"].max() - account_df["timestamp"].min()).days, 0)
    feature_row["active_days"] = float(active_days)
    feature_row["degree_ratio"] = feature_row["out_degree"] / (feature_row["in_degree"] + 1)
    feature_row["flow_through_score"] = abs(incoming_total - outgoing_total) / (feature_row["total_amt"] + 1)
    feature_row["rapid_drain_ratio"] = feature_row["rapid_drain"] / (feature_row["txn_count"] + 1)
    feature_row["high_risk_exposure_ratio"] = feature_row["high_risk_exposure"] / (feature_row["txn_count"] + 1)
    feature_row["txn_velocity"] = feature_row["txn_count"] / (feature_row["active_days"] + 1)
    feature_row["top_counterparty_ratio"] = float(cp_counts.max() / cp_counts.sum()) if not cp_counts.empty else 0.0

    entropy = 0.0
    if not cp_counts.empty:
        probs = cp_counts / cp_counts.sum()
        entropy = float(-(probs * probs.map(lambda x: 0 if x == 0 else np.log(x))).sum())
    feature_row["counterparty_entropy"] = entropy

    return pd.DataFrame([feature_row])


def _simulate_risk_score(feature_row, graph_row):
    burst_component = min(25, feature_row.get("burst_count", 0) * 4)
    rapid_component = min(20, feature_row.get("rapid_drain", 0) * 5)
    exposure_component = min(10, feature_row.get("high_risk_exposure", 0) * 3)
    velocity_component = min(10, feature_row.get("txn_velocity", 0) * 0.8)
    community_size = graph_row.get("community_size", 0)
    graph_component = graph_row.get("graph_risk_score", 0) * 0.25 if community_size >= 3 else 0.0
    community_component = graph_row.get("community_risk_score", 0) * 0.15 if community_size >= 3 else 0.0
    structuring_component = min(8, feature_row.get("near_threshold_count", 0) * 4)
    counterparty_component = min(10, feature_row.get("unique_counterparty", 0) * 2.5)
    flow_component = 10 if feature_row.get("txn_count", 0) >= 3 and feature_row.get("flow_through_score", 1) <= 0.2 else 0.0

    score = (
        5
        + burst_component
        + rapid_component
        + exposure_component
        + velocity_component
        + graph_component
        + community_component
        + structuring_component
        + counterparty_component
        + flow_component
    )
    return round(min(99.0, score), 2)


def generate_simulation_mode():
    """
    Simulate a staged fraud attack and return replay artifacts in memory.
    """
    transactions = pd.DataFrame(SIMULATION_EVENTS)
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)

    timeline_rows = []
    target_account = "SIM_MULE_01"

    for event_index in range(len(transactions)):
        window_df = transactions.iloc[: event_index + 1].copy()
        feature_snapshot = _build_feature_snapshot(window_df, target_account)
        graph_df = build_graph_intelligence_from_transactions(window_df.rename(columns={"timestamp": "transaction_timestamp"}))
        graph_snapshot = graph_df[graph_df["account_id"] == target_account].copy()

        if graph_snapshot.empty:
            graph_snapshot = pd.DataFrame([{"account_id": target_account}])

        feature_snapshot = feature_snapshot.merge(graph_snapshot, on="account_id", how="left")
        feature_snapshot = feature_snapshot.fillna(0)

        risk_score = _simulate_risk_score(feature_snapshot.iloc[0].to_dict(), graph_snapshot.iloc[0].to_dict())
        scored_df = pd.DataFrame(
            [
                {
                    "account_id": target_account,
                    "is_mule": risk_score / 100.0,
                    "risk_score": risk_score,
                    "suspicious_start": window_df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S") if risk_score >= 80 else "N/A",
                    "suspicious_end": window_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S") if risk_score >= 80 else "N/A",
                }
            ]
        )
        scored_df = compute_risk_score(scored_df)
        decision_df = apply_decision_engine(scored_df, feature_snapshot)
        decision_row = decision_df.iloc[0]

        timeline_rows.append(
            {
                "simulation_step": event_index + 1,
                "timestamp": window_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": window_df.iloc[-1]["stage"],
                "scenario_event": window_df.iloc[-1]["scenario_event"],
                "account_id": target_account,
                "cumulative_transactions": int(len(window_df[window_df["account_id"] == target_account])),
                "risk_score": float(decision_row["risk_score"]),
                "risk_level": decision_row["risk_level"],
                "graph_risk_score": float(decision_row.get("graph_risk_score", 0)),
                "community_risk_score": float(decision_row.get("community_risk_score", 0)),
                "primary_action": decision_row["primary_action"],
                "action_priority": decision_row["action_priority"],
                "auto_action_eligible": decision_row["auto_action_eligible"],
                "freeze_account": decision_row["freeze_account"],
                "alert_compliance_team": decision_row["alert_compliance_team"],
                "reduce_transaction_limit": decision_row["reduce_transaction_limit"],
                "decision_summary": decision_row["decision_summary"],
                "decision_reasons": decision_row["decision_reasons"],
            }
        )

    timeline_df = pd.DataFrame(timeline_rows)
    timeline_df["action_changed"] = timeline_df["primary_action"].ne(timeline_df["primary_action"].shift(1)).map({True: "YES", False: "NO"})
    timeline_df.loc[timeline_df.index[0], "action_changed"] = "YES"
    timeline_df["risk_delta"] = timeline_df["risk_score"].diff().fillna(timeline_df["risk_score"]).round(2)

    summary_df = pd.DataFrame(
        [
            {
                "scenario_name": "Synthetic Mule Attack Replay",
                "steps_simulated": int(len(timeline_df)),
                "final_risk_score": float(timeline_df.iloc[-1]["risk_score"]),
                "final_action": timeline_df.iloc[-1]["primary_action"],
                "peak_risk_score": float(timeline_df["risk_score"].max()),
                "first_auto_action_step": int(timeline_df.loc[timeline_df["auto_action_eligible"] == "YES", "simulation_step"].min()) if (timeline_df["auto_action_eligible"] == "YES").any() else 0,
                "freeze_triggered": "YES" if (timeline_df["freeze_account"] == "YES").any() else "NO",
                "compliance_alert_triggered": "YES" if (timeline_df["alert_compliance_team"] == "YES").any() else "NO",
                "limit_reduction_triggered": "YES" if (timeline_df["reduce_transaction_limit"] == "YES").any() else "NO",
                "stages_covered": timeline_df["stage"].nunique(),
            }
        ]
    )

    return {
        "transactions": transactions,
        "timeline": timeline_df,
        "summary": summary_df,
    }
