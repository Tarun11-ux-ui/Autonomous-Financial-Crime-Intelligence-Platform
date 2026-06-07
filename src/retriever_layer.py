import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.knowledge_store import CaseKnowledgeStore


RETRIEVER_TEXT_COLUMNS = [
    "account_id",
    "decision_summary",
    "decision_reasons",
    "explanation_summary",
    "human_readable_reasoning",
    "investigator_brief",
    "investigator_keywords",
    "retrieval_text",
    "connected_risky_accounts",
    "primary_action",
    "risk_level",
    "impact_band",
]

QUERY_EXAMPLES = [
    "Why is account <ACCOUNT_ID> suspicious?",
    "Find connected risky users for <ACCOUNT_ID>",
    "Show high risk fraud ring accounts",
    "Summarize this case",
    "Show top risky accounts",
]

ACCOUNT_QUERY_PATTERN = re.compile(r"\baccount\s+([a-zA-Z0-9_\-]+)\b", re.IGNORECASE)
WHY_ACCOUNT_PATTERN = re.compile(r"\bwhy\s+is\s+account\s+([a-zA-Z0-9_\-]+)\b", re.IGNORECASE)
CONNECTED_ACCOUNT_PATTERN = re.compile(
    r"(?:find|show|list).*(?:connected risky users|connected users|connected accounts|linked accounts).*(?:for|of)\s+([a-zA-Z0-9_\-]+)",
    re.IGNORECASE,
)
SUMMARY_ACCOUNT_PATTERN = re.compile(
    r"(?:summarize|summary).*(?:account|case)\s+([a-zA-Z0-9_\-]+)",
    re.IGNORECASE,
)


def _as_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_query(query):
    return re.sub(r"\s+", " ", _as_text(query).lower()).strip()


def _safe_series(df, column, default=0.0):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _split_connected_accounts(value):
    text = _as_text(value)
    if not text or text.lower() == "n/a":
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def extract_account_id_from_query(query):
    match = ACCOUNT_QUERY_PATTERN.search(_as_text(query))
    if match:
        return match.group(1).strip()
    return None


def classify_query_type(query):
    query_text = _as_text(query)
    normalized = _normalize_query(query_text)

    if WHY_ACCOUNT_PATTERN.search(query_text):
        return "account_why"
    if CONNECTED_ACCOUNT_PATTERN.search(query_text):
        return "account_connected"
    if SUMMARY_ACCOUNT_PATTERN.search(query_text):
        return "account_summary"
    if extract_account_id_from_query(query_text):
        return "account_lookup"
    if "fraud patterns" in normalized or "patterns today" in normalized:
        return "fraud_patterns"
    if "top risky accounts" in normalized or "top accounts" in normalized:
        return "top_accounts"
    if "ring" in normalized:
        return "fraud_ring"
    return "generic"


def _build_retriever_document(row):
    return " ".join(_as_text(row.get(column, "")) for column in RETRIEVER_TEXT_COLUMNS if column in row.index).strip()


def _query_expansion(query):
    normalized = _normalize_query(query)
    expansions = [normalized]

    synonym_map = {
        "suspicious": "fraud risk suspicious anomaly",
        "connected": "linked network community fraud ring counterparties",
        "ring": "fraud ring community linked network",
        "top": "highest priority critical high risk",
        "case": "account decision explanation investigation",
        "patterns": "patterns indicators reasons signals",
    }

    expanded_terms = [normalized]
    for token, mapped_text in synonym_map.items():
        if token in normalized:
            expanded_terms.append(mapped_text)

    expansions.append(" ".join(expanded_terms))
    return " ".join(expansions).strip()


def apply_retriever_layer(submission_df):
    """
    Add retriever-ready document identifiers and corpus text to the submission rows.
    """
    retriever_df = submission_df.copy()
    retriever_df["retriever_document_id"] = [f"DOC-{idx:06d}" for idx in range(1, len(retriever_df) + 1)]
    retriever_df["retriever_source_type"] = "account_case"
    retriever_df["retriever_document"] = retriever_df.apply(_build_retriever_document, axis=1)
    retriever_df["retriever_ready"] = retriever_df["retriever_document"].str.len().gt(0).map({True: "YES", False: "NO"})
    retriever_df["knowledge_store_ready"] = retriever_df["retriever_ready"]
    return retriever_df


def summarize_retriever_layer(submission_df):
    """
    Return a compact readiness summary for the retriever layer.
    """
    if "retriever_ready" in submission_df.columns:
        ready_mask = submission_df["retriever_ready"] == "YES"
    else:
        ready_mask = pd.Series([False] * len(submission_df), index=submission_df.index)

    if "connected_risky_accounts" in submission_df.columns:
        connection_mask = submission_df["connected_risky_accounts"].fillna("").astype(str).str.len() > 0
    else:
        connection_mask = pd.Series([False] * len(submission_df), index=submission_df.index)

    return {
        "retriever_ready_documents": int(ready_mask.sum()) if hasattr(ready_mask, "sum") else 0,
        "documents_with_network_context": int(connection_mask.sum()) if hasattr(connection_mask, "sum") else 0,
        "query_examples": QUERY_EXAMPLES,
    }


class FraudRetriever:
    """
    Lightweight TF-IDF retriever over the enriched screening output.
    """

    def __init__(self, submission_df):
        self.df = submission_df.copy().reset_index(drop=True)
        if "retriever_document" not in self.df.columns:
            self.df = apply_retriever_layer(self.df)
        self.df["account_id_normalized"] = self.df["account_id"].astype(str).str.strip().str.lower()

        self.documents = self.df["retriever_document"].fillna("").astype(str)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
        if len(self.documents) == 0 or self.documents.str.len().eq(0).all():
            self.matrix = None
        else:
            try:
                self.matrix = self.vectorizer.fit_transform(self.documents)
            except ValueError:
                self.matrix = None
        self.knowledge_store = CaseKnowledgeStore(
            self.df["retriever_document_id"].tolist(),
            self.matrix,
        ) if self.matrix is not None else CaseKnowledgeStore([], None)

    def _score_query(self, query, vector_top_k=10):
        if self.matrix is None:
            return self.df.iloc[0:0].copy()

        expanded_query = _query_expansion(query)
        query_vector = self.vectorizer.transform([expanded_query])
        cosine_scores = linear_kernel(query_vector, self.matrix).flatten()
        vector_hits = self.knowledge_store.search(query_vector, top_k=max(vector_top_k, 5))

        scored = self.df.copy()
        scored["retriever_score"] = cosine_scores
        scored["vector_score"] = 0.0
        if not vector_hits.empty:
            scored = scored.merge(vector_hits, on="retriever_document_id", how="left", suffixes=("", "_hit"))
            scored["vector_score"] = scored["vector_score_hit"].fillna(scored["vector_score"])
            scored = scored.drop(columns=["vector_score_hit"])
            scored["retriever_score"] = scored["retriever_score"] * 0.65 + scored["vector_score"] * 0.35

        account_id = extract_account_id_from_query(query)
        if account_id:
            scored.loc[
                scored["account_id_normalized"] == account_id.lower(),
                "retriever_score",
            ] += 2.0

        if "top" in _normalize_query(query):
            scored["retriever_score"] += _safe_series(scored, "impact_priority_score") / 1000.0

        if "ring" in _normalize_query(query) or "connected" in _normalize_query(query):
            fraud_ring_series = (
                scored["fraud_ring_flag"].fillna("NO")
                if "fraud_ring_flag" in scored.columns
                else pd.Series(["NO"] * len(scored), index=scored.index)
            )
            scored["retriever_score"] += (
                (fraud_ring_series == "YES").astype(float) * 0.4
                + _safe_series(scored, "community_risk_score") / 1000.0
            )

        if "suspicious" in _normalize_query(query):
            scored["retriever_score"] += _safe_series(scored, "risk_score") / 1000.0

        scored["impact_priority_score"] = _safe_series(scored, "impact_priority_score")
        scored["risk_score"] = _safe_series(scored, "risk_score")
        scored["context_relation"] = "general_match"
        scored["target_account_id"] = ""

        return scored.sort_values(
            ["retriever_score", "impact_priority_score", "risk_score"],
            ascending=[False, False, False],
        )

    def _score_account_scoped_query(self, query, query_type, top_k):
        target_account_id = extract_account_id_from_query(query)
        if not target_account_id:
            return None

        normalized_target = target_account_id.lower()
        account_matches = self.df[self.df["account_id_normalized"] == normalized_target]
        if account_matches.empty:
            return None

        scored = self._score_query(query, vector_top_k=max(top_k * 2, 10))
        if scored.empty:
            return scored

        target_row = account_matches.iloc[0]
        explicit_connections = set(
            account_id.lower() for account_id in _split_connected_accounts(target_row.get("connected_risky_accounts", ""))
        )
        target_community = target_row.get("community_id", 0)
        has_target_community = not pd.isna(target_community) and float(target_community) > 0

        exact_mask = scored["account_id_normalized"] == normalized_target
        connected_mask = scored["account_id_normalized"].isin(explicit_connections)
        community_mask = (
            (scored.get("community_id", pd.Series([0] * len(scored), index=scored.index)) == target_community)
            & ~exact_mask
            & ~connected_mask
        ) if has_target_community and "community_id" in scored.columns else pd.Series([False] * len(scored), index=scored.index)

        candidate_mask = exact_mask.copy()
        if explicit_connections:
            candidate_mask = candidate_mask | connected_mask
        elif has_target_community:
            candidate_mask = candidate_mask | community_mask

        scoped = scored[candidate_mask].copy()
        if scoped.empty:
            return None

        scoped.loc[exact_mask.loc[scoped.index], "context_relation"] = "exact_match"
        if explicit_connections:
            scoped.loc[connected_mask.loc[scoped.index], "context_relation"] = "connected_match"
            scoped.loc[connected_mask.loc[scoped.index], "retriever_score"] += 25.0
        else:
            scoped.loc[community_mask.loc[scoped.index], "context_relation"] = "community_match"
            scoped.loc[community_mask.loc[scoped.index], "retriever_score"] += 10.0

        scoped.loc[exact_mask.loc[scoped.index], "retriever_score"] += 100.0
        scoped["target_account_id"] = target_row["account_id"]

        if query_type == "account_connected":
            scoped = scoped[scoped["context_relation"] != "exact_match"]
        elif query_type in {"account_why", "account_summary", "account_lookup"}:
            scoped = scoped.sort_values(
                ["retriever_score", "impact_priority_score", "risk_score"],
                ascending=[False, False, False],
            ).head(top_k)
            return scoped

        return scoped.sort_values(
            ["retriever_score", "impact_priority_score", "risk_score"],
            ascending=[False, False, False],
        ).head(top_k)

    def search(self, query, top_k=5):
        query_type = classify_query_type(query)
        results = None

        if query_type.startswith("account_"):
            results = self._score_account_scoped_query(query, query_type=query_type, top_k=top_k)

        if results is None:
            results = self._score_query(query, vector_top_k=max(top_k * 2, 10)).head(top_k)

        columns = [
            "retriever_document_id",
            "account_id",
            "risk_score",
            "primary_action",
            "impact_priority_score",
            "retriever_score",
            "vector_score",
            "context_relation",
            "target_account_id",
            "investigator_brief",
            "connected_risky_accounts",
        ]
        available_columns = [column for column in columns if column in results.columns]
        return results[available_columns].to_dict("records")

    def knowledge_store_summary(self):
        return self.knowledge_store.summary()
