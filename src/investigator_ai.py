import re

import pandas as pd
from src.retriever_layer import FraudRetriever


def _as_text(value):
    if pd.isna(value):
        return "N/A"
    return str(value)


def _normalize_account_id(account_id):
    return _as_text(account_id).strip().lower()


def _split_reason_text(reason_text, limit=4):
    reasons = [item.strip() for item in _as_text(reason_text).split("|") if item.strip()]
    return reasons[:limit]


def _build_keywords(row):
    keywords = [
        _as_text(row.get("risk_level", "LOW")).lower(),
        _as_text(row.get("primary_action", "ALLOW_WITH_MONITORING")).lower(),
        _as_text(row.get("impact_band", "LOW")).lower(),
    ]

    if row.get("fraud_ring_flag", "NO") == "YES":
        keywords.append("fraud_ring")
    if row.get("auto_action_eligible", "NO") == "YES":
        keywords.append("auto_action")
    if row.get("freeze_account", "NO") == "YES":
        keywords.append("freeze")
    if row.get("alert_compliance_team", "NO") == "YES":
        keywords.append("compliance")
    if row.get("reduce_transaction_limit", "NO") == "YES":
        keywords.append("limit_reduction")
    if _as_text(row.get("suspicious_start", "N/A")) != "N/A":
        keywords.append("suspicious_window")
    if float(row.get("graph_risk_score", 0) or 0) >= 70:
        keywords.append("graph_risk")

    top_reasons = _split_reason_text(row.get("decision_reasons", ""), limit=3)
    keywords.extend(reason.lower().replace(" ", "_") for reason in top_reasons)
    return " | ".join(dict.fromkeys(keywords))


def _build_connected_accounts_map(df, max_accounts=5):
    connected_map = {}

    if "community_id" not in df.columns:
        for account_id in df["account_id"]:
            connected_map[account_id] = ""
        return connected_map

    for _, row in df.iterrows():
        community_id = row.get("community_id", 0)
        if pd.isna(community_id) or float(community_id) <= 0:
            connected_map[row["account_id"]] = ""
            continue

        connected = df[
            (df["community_id"] == community_id)
            & (df["account_id"] != row["account_id"])
            & (
                (df["risk_score"] >= 60)
                | (df["fraud_ring_flag"] == "YES")
                | (df["graph_risk_score"] >= 70)
            )
        ].sort_values(
            ["risk_score", "graph_risk_score", "impact_priority_score"],
            ascending=[False, False, False],
        )

        connected_accounts = connected["account_id"].astype(str).head(max_accounts).tolist()
        connected_map[row["account_id"]] = " | ".join(connected_accounts)

    return connected_map


def apply_investigator_layer(submission_df):
    """
    Enrich submission rows with retrieval-ready narratives for investigator workflows.
    """
    investigator_df = submission_df.copy()
    connected_map = _build_connected_accounts_map(investigator_df)
    investigator_df["connected_risky_accounts"] = investigator_df["account_id"].map(connected_map).fillna("")

    investigator_df["investigator_brief"] = investigator_df.apply(
        lambda row: (
            f"Account {row['account_id']} is {row['risk_level']} risk with action {row['primary_action']}. "
            f"Estimated fraud prevented is INR {float(row.get('estimated_fraud_prevented_inr', 0)):.2f}. "
            f"Top reasons: {'; '.join(_split_reason_text(row.get('decision_reasons', ''), limit=3)) or 'baseline monitoring'}. "
            f"Connected risky accounts: {row['connected_risky_accounts'] if row['connected_risky_accounts'] else 'none identified'}."
        ),
        axis=1,
    )
    investigator_df["investigator_keywords"] = investigator_df.apply(_build_keywords, axis=1)
    investigator_df["retrieval_text"] = investigator_df.apply(
        lambda row: " ".join(
            [
                _as_text(row.get("account_id")),
                _as_text(row.get("risk_level")),
                _as_text(row.get("primary_action")),
                _as_text(row.get("decision_summary")),
                _as_text(row.get("decision_reasons")),
                _as_text(row.get("model_top_drivers")),
                _as_text(row.get("explanation_summary")),
                _as_text(row.get("investigator_keywords")),
                _as_text(row.get("investigator_brief")),
            ]
        ),
        axis=1,
    )

    return investigator_df


def summarize_investigator_layer(submission_df):
    """
    Return a compact readiness summary for the investigator layer.
    """
    query_examples = [
        "Why is account <ACCOUNT_ID> suspicious?",
        "Show fraud patterns today",
        "Find connected risky users for <ACCOUNT_ID>",
        "Show top risky accounts",
    ]

    return {
        "investigator_ready_accounts": int((submission_df["case_generated"] == "YES").sum()) if "case_generated" in submission_df.columns else len(submission_df),
        "query_examples": query_examples,
        "accounts_with_connections": int((submission_df.get("connected_risky_accounts", "") != "").sum()) if "connected_risky_accounts" in submission_df.columns else 0,
    }


class InvestigatorAssistant:
    """
    Lightweight query layer over the enriched submission dataframe.
    """

    def __init__(self, submission_df):
        self.df = submission_df.copy()
        self.retriever = FraudRetriever(self.df)

    def why_is_account_suspicious(self, account_id):
        account_text = str(account_id).strip()
        normalized = _normalize_account_id(account_text)
        match = self.df[self.df["account_id"].astype(str).str.lower() == normalized]
        if match.empty:
            return {"answer": f"Account {account_text} was not found in the current screening run."}

        row = match.iloc[0]
        return {
            "account_id": row["account_id"],
            "answer": row.get("investigator_brief", ""),
            "risk_score": float(row.get("risk_score", 0)),
            "primary_action": row.get("primary_action", "ALLOW_WITH_MONITORING"),
            "top_reasons": _split_reason_text(row.get("decision_reasons", ""), limit=5),
            "connected_risky_accounts": row.get("connected_risky_accounts", ""),
        }

    def show_fraud_patterns(self, limit=5):
        top_patterns = []
        for reason in self.df.get("decision_reasons", pd.Series(dtype=str)).fillna(""):
            top_patterns.extend(_split_reason_text(reason, limit=10))

        pattern_series = pd.Series(top_patterns, dtype=str)
        top_reason_counts = (
            pattern_series.value_counts().head(limit).rename_axis("pattern").reset_index(name="count")
            if not pattern_series.empty
            else pd.DataFrame(columns=["pattern", "count"])
        )

        return {
            "high_risk_accounts": int((self.df["risk_score"] >= 80).sum()),
            "ring_cases": int((self.df.get("ring_case_flag", "NO") == "YES").sum()) if "ring_case_flag" in self.df.columns else 0,
            "auto_actions": int((self.df.get("auto_action_eligible", "NO") == "YES").sum()) if "auto_action_eligible" in self.df.columns else 0,
            "top_patterns": top_reason_counts.to_dict("records"),
        }

    def find_connected_risky_users(self, account_id, top_k=5):
        account_text = str(account_id).strip()
        normalized = _normalize_account_id(account_text)
        match = self.df[self.df["account_id"].astype(str).str.lower() == normalized]
        if match.empty:
            return {"account_id": account_text, "connected_accounts": []}

        row = match.iloc[0]
        community_id = row.get("community_id", 0)
        if pd.isna(community_id) or float(community_id) <= 0:
            return {"account_id": row["account_id"], "connected_accounts": []}

        connected = self.df[
            (self.df["community_id"] == community_id)
            & (self.df["account_id"] != row["account_id"])
        ].sort_values(
            ["risk_score", "graph_risk_score", "impact_priority_score"],
            ascending=[False, False, False],
        )

        return {
            "account_id": row["account_id"],
            "community_id": int(community_id),
            "connected_accounts": connected[
                ["account_id", "risk_score", "primary_action", "graph_risk_score", "impact_priority_score"]
            ].head(top_k).to_dict("records"),
        }

    def top_risky_accounts(self, limit=5):
        top_df = self.df.sort_values(
            ["impact_priority_score", "risk_score", "graph_risk_score"],
            ascending=[False, False, False],
        )
        return top_df[
            [
                "account_id",
                "risk_score",
                "primary_action",
                "impact_priority_score",
                "estimated_fraud_prevented_inr",
            ]
        ].head(limit).to_dict("records")

    def answer_query(self, query):
        query_text = str(query).strip()
        lowered = query_text.lower()

        why_match = re.search(r"why\s+is\s+account\s+([a-zA-Z0-9_\-]+)", lowered)
        if why_match:
            return self.why_is_account_suspicious(why_match.group(1))

        connected_match = re.search(r"(?:find|show).*(?:connected risky users|connected users|connected accounts).*(?:for|of)\s+([a-zA-Z0-9_\-]+)", lowered)
        if connected_match:
            return self.find_connected_risky_users(connected_match.group(1))

        if "fraud patterns" in lowered or "patterns today" in lowered:
            return self.show_fraud_patterns()

        if "top risky accounts" in lowered or "top accounts" in lowered:
            return {"top_risky_accounts": self.top_risky_accounts()}

        return {
            "query": query_text,
            "matches": self.retriever.search(query_text, top_k=5),
        }

    def answer_query_with_ollama(self, query, model="phi3", base_url="http://localhost:11434/api", top_k=5):
        from src.ollama_rag import OllamaRAGAssistant

        rag_assistant = OllamaRAGAssistant(self.df, model=model, base_url=base_url)
        return rag_assistant.answer_query(query, top_k=top_k)
