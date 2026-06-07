from glob import glob
from time import perf_counter
import sys

import networkx as nx
import numpy as np
import pandas as pd

from src.config import TRANSACTION_PATH
from src.progress_utils import build_progress_message, build_weighted_progress_message, format_duration

GRAPH_NODE_COLUMNS = [
    "account_id",
    "pagerank_score",
    "betweenness_centrality",
    "clustering_coefficient",
    "graph_in_degree",
    "graph_out_degree",
    "graph_in_amount",
    "graph_out_amount",
    "graph_total_amount",
    "community_id",
    "community_size",
    "community_density",
    "community_amount_total",
    "community_risk_score",
    "fraud_ring_flag",
    "pagerank_percentile",
    "betweenness_percentile",
    "amount_percentile",
    "clustering_percentile",
    "graph_risk_score",
    "influential_node_flag",
]


def _percentile_score(series):
    if series.empty:
        return pd.Series(dtype=float)
    return series.rank(method="average", pct=True).fillna(0).mul(100).round(2)


def _log_graph_phase(phase_name, completed, total, start_time):
    print("   [ETA] " + build_progress_message(phase_name, completed, total, start_time))
    sys.stdout.flush()


def _log_graph_weighted_progress(completed_weight, total_weight, start_time):
    print("   [ETA] " + build_weighted_progress_message("Graph intelligence overall", completed_weight, total_weight, start_time))
    sys.stdout.flush()


def _load_edge_table(files):
    edge_chunks = []
    load_start = perf_counter()

    for idx, file in enumerate(files, 1):
        print(
            "   [ETA] " + build_progress_message("Graph edge loading", idx, len(files), load_start),
            end="\r",
        )
        sys.stdout.flush()
        df = pd.read_parquet(
            file,
            columns=["account_id", "counterparty_id", "amount", "txn_type"],
        )
        df = df.dropna(subset=["account_id", "counterparty_id", "amount", "txn_type"]).copy()

        if df.empty:
            continue

        df["amount"] = df["amount"].astype(float)
        df["source_account"] = np.where(
            df["txn_type"] == "D",
            df["account_id"],
            df["counterparty_id"],
        )
        df["target_account"] = np.where(
            df["txn_type"] == "D",
            df["counterparty_id"],
            df["account_id"],
        )
        df = df[df["source_account"] != df["target_account"]]

        edges = (
            df.groupby(["source_account", "target_account"])
            .agg(
                graph_txn_count=("amount", "size"),
                graph_total_amount=("amount", "sum"),
            )
            .reset_index()
        )
        edge_chunks.append(edges)

    if files:
        print()
        print(f"   [ETA] Graph edge loading completed in {format_duration(perf_counter() - load_start)}")
        sys.stdout.flush()

    if not edge_chunks:
        return pd.DataFrame(
            columns=[
                "source_account",
                "target_account",
                "graph_txn_count",
                "graph_total_amount",
            ]
        )

    edge_table = pd.concat(edge_chunks, ignore_index=True)
    edge_table = (
        edge_table.groupby(["source_account", "target_account"], as_index=False)
        .agg(
            graph_txn_count=("graph_txn_count", "sum"),
            graph_total_amount=("graph_total_amount", "sum"),
        )
    )
    return edge_table


def _prepare_edge_table_from_transactions(df):
    df = df.dropna(subset=["account_id", "counterparty_id", "amount", "txn_type"]).copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "source_account",
                "target_account",
                "graph_txn_count",
                "graph_total_amount",
            ]
        )

    df["amount"] = df["amount"].astype(float)
    df["source_account"] = np.where(
        df["txn_type"] == "D",
        df["account_id"],
        df["counterparty_id"],
    )
    df["target_account"] = np.where(
        df["txn_type"] == "D",
        df["counterparty_id"],
        df["account_id"],
    )
    df = df[df["source_account"] != df["target_account"]]

    return (
        df.groupby(["source_account", "target_account"], as_index=False)
        .agg(
            graph_txn_count=("amount", "size"),
            graph_total_amount=("amount", "sum"),
        )
    )


def _prepare_edge_table_from_batch(df):
    edge_df = df[["account_id", "counterparty_id", "amount", "txn_type"]].dropna(
        subset=["account_id", "counterparty_id", "amount", "txn_type"]
    ).copy()

    if edge_df.empty:
        return pd.DataFrame(
            columns=[
                "source_account",
                "target_account",
                "graph_txn_count",
                "graph_total_amount",
            ]
        )

    edge_df["amount"] = edge_df["amount"].astype(float)
    edge_df["source_account"] = np.where(
        edge_df["txn_type"] == "D",
        edge_df["account_id"],
        edge_df["counterparty_id"],
    )
    edge_df["target_account"] = np.where(
        edge_df["txn_type"] == "D",
        edge_df["counterparty_id"],
        edge_df["account_id"],
    )
    edge_df = edge_df[edge_df["source_account"] != edge_df["target_account"]]

    return (
        edge_df.groupby(["source_account", "target_account"], as_index=False)
        .agg(
            graph_txn_count=("amount", "size"),
            graph_total_amount=("amount", "sum"),
        )
    )


def _build_graphs(edge_table):
    directed_graph = nx.DiGraph()

    for row in edge_table.itertuples(index=False):
        directed_graph.add_edge(
            row.source_account,
            row.target_account,
            weight=float(row.graph_total_amount),
            txn_count=float(row.graph_txn_count),
        )

    undirected_graph = nx.Graph()
    for source, target, data in directed_graph.edges(data=True):
        amount = float(data.get("weight", 0.0))
        txn_count = float(data.get("txn_count", 0.0))
        if undirected_graph.has_edge(source, target):
            undirected_graph[source][target]["weight"] += amount
            undirected_graph[source][target]["txn_count"] += txn_count
        else:
            undirected_graph.add_edge(source, target, weight=amount, txn_count=txn_count)

    return directed_graph, undirected_graph


def _compute_betweenness(graph):
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return {}
    if node_count > 20000:
        sample_size = min(64, node_count)
        return nx.betweenness_centrality(graph, k=sample_size, normalized=True, seed=42)
    if node_count > 5000:
        sample_size = min(80, node_count)
        return nx.betweenness_centrality(graph, k=sample_size, normalized=True, seed=42)
    if node_count > 250:
        sample_size = min(100, node_count)
        return nx.betweenness_centrality(graph, k=sample_size, normalized=True, seed=42)
    return nx.betweenness_centrality(graph, normalized=True)


def _detect_communities(undirected_graph, pagerank_scores):
    if undirected_graph.number_of_nodes() == 0:
        return {}, pd.DataFrame()

    try:
        communities = nx.community.louvain_communities(undirected_graph, weight="weight", seed=42)
    except Exception:
        communities = [set(component) for component in nx.connected_components(undirected_graph)]

    communities = sorted(communities, key=len, reverse=True)
    community_map = {}
    community_rows = []

    for community_id, members in enumerate(communities, start=1):
        subgraph = undirected_graph.subgraph(members)
        community_size = len(members)
        possible_edges = community_size * (community_size - 1) / 2
        community_density = 0.0 if community_size < 2 else subgraph.number_of_edges() / max(possible_edges, 1)
        internal_amount = sum(data.get("weight", 0.0) for _, _, data in subgraph.edges(data=True))
        avg_pagerank = float(np.mean([pagerank_scores.get(node, 0.0) for node in members])) if members else 0.0

        sample_members = list(members)[:25]
        community_rows.append(
            {
                "community_id": community_id,
                "community_size": community_size,
                "community_density": round(community_density, 4),
                "community_amount_total": round(internal_amount, 2),
                "community_avg_pagerank": avg_pagerank,
                "community_members_preview": " | ".join(map(str, sample_members)),
            }
        )

        for node in members:
            community_map[node] = community_id

    community_df = pd.DataFrame(community_rows)
    if not community_df.empty:
        community_df["community_risk_score"] = (
            _percentile_score(community_df["community_size"])
            + _percentile_score(community_df["community_density"])
            + _percentile_score(community_df["community_amount_total"])
            + _percentile_score(community_df["community_avg_pagerank"])
        ) / 4
        community_df["community_risk_score"] = community_df["community_risk_score"].round(2)
        community_df["fraud_ring_flag"] = np.where(
            (community_df["community_size"] >= 3) & (community_df["community_risk_score"] >= 70),
            "YES",
            "NO",
        )
        community_df = community_df.sort_values(
            ["community_risk_score", "community_amount_total"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return community_map, community_df


def _build_node_features_from_edge_table(edge_table, graph_start=None, log_progress=False):
    total_weight = 40.0 if log_progress else 0.0
    completed_weight = 0.0

    directed_graph, undirected_graph = _build_graphs(edge_table)
    if log_progress:
        print(
            f"   [ETA] Graph build stats: {directed_graph.number_of_nodes()} nodes, "
            f"{directed_graph.number_of_edges()} directed edges"
        )
        completed_weight += 5.0
        _log_graph_weighted_progress(60.0 + completed_weight, 100.0, graph_start)

    pagerank_scores = nx.pagerank(directed_graph, weight="weight")
    if log_progress:
        completed_weight += 10.0
        _log_graph_weighted_progress(60.0 + completed_weight, 100.0, graph_start)

    betweenness_scores = _compute_betweenness(directed_graph)
    if log_progress:
        completed_weight += 15.0
        _log_graph_weighted_progress(60.0 + completed_weight, 100.0, graph_start)

    clustering_scores = nx.clustering(undirected_graph, weight="weight")
    community_map, community_df = _detect_communities(undirected_graph, pagerank_scores)
    if log_progress:
        completed_weight += 5.0
        _log_graph_weighted_progress(60.0 + completed_weight, 100.0, graph_start)

    nodes = sorted(directed_graph.nodes())
    node_df = pd.DataFrame({"account_id": nodes})
    node_df["pagerank_score"] = node_df["account_id"].map(pagerank_scores).fillna(0.0)
    node_df["betweenness_centrality"] = node_df["account_id"].map(betweenness_scores).fillna(0.0)
    node_df["clustering_coefficient"] = node_df["account_id"].map(clustering_scores).fillna(0.0)
    node_df["graph_in_degree"] = node_df["account_id"].map(dict(directed_graph.in_degree())).fillna(0).astype(float)
    node_df["graph_out_degree"] = node_df["account_id"].map(dict(directed_graph.out_degree())).fillna(0).astype(float)
    node_df["graph_in_amount"] = node_df["account_id"].map(dict(directed_graph.in_degree(weight="weight"))).fillna(0.0)
    node_df["graph_out_amount"] = node_df["account_id"].map(dict(directed_graph.out_degree(weight="weight"))).fillna(0.0)
    node_df["graph_total_amount"] = node_df["graph_in_amount"] + node_df["graph_out_amount"]
    node_df["community_id"] = node_df["account_id"].map(community_map).fillna(0).astype(int)

    if community_df.empty:
        node_df["community_size"] = 0.0
        node_df["community_density"] = 0.0
        node_df["community_amount_total"] = 0.0
        node_df["community_risk_score"] = 0.0
        node_df["fraud_ring_flag"] = "NO"
    else:
        community_lookup = community_df.set_index("community_id")
        node_df["community_size"] = node_df["community_id"].map(community_lookup["community_size"]).fillna(0).astype(float)
        node_df["community_density"] = node_df["community_id"].map(community_lookup["community_density"]).fillna(0.0)
        node_df["community_amount_total"] = node_df["community_id"].map(community_lookup["community_amount_total"]).fillna(0.0)
        node_df["community_risk_score"] = node_df["community_id"].map(community_lookup["community_risk_score"]).fillna(0.0)
        node_df["fraud_ring_flag"] = node_df["community_id"].map(community_lookup["fraud_ring_flag"]).fillna("NO")

    node_df["pagerank_percentile"] = _percentile_score(node_df["pagerank_score"])
    node_df["betweenness_percentile"] = _percentile_score(node_df["betweenness_centrality"])
    node_df["amount_percentile"] = _percentile_score(node_df["graph_total_amount"])
    node_df["clustering_percentile"] = _percentile_score(node_df["clustering_coefficient"])
    node_df["graph_risk_score"] = (
        node_df["pagerank_percentile"]
        + node_df["betweenness_percentile"]
        + node_df["amount_percentile"]
        + node_df["community_risk_score"]
    ) / 4
    node_df["graph_risk_score"] = node_df["graph_risk_score"].round(2)
    node_df["influential_node_flag"] = np.where(
        (node_df["pagerank_percentile"] >= 85) | (node_df["betweenness_percentile"] >= 85),
        "YES",
        "NO",
    )
    node_df = node_df[GRAPH_NODE_COLUMNS]

    if log_progress:
        completed_weight += 5.0
        _log_graph_weighted_progress(60.0 + completed_weight, 100.0, graph_start)
        print(f"   [ETA] Graph intelligence layer completed in {format_duration(perf_counter() - graph_start)}")
        sys.stdout.flush()

    return node_df


def build_graph_intelligence(files=None):
    """
    Build graph-based fraud intelligence features.
    """
    graph_start = perf_counter()
    files = files or glob(TRANSACTION_PATH)
    edge_table = _load_edge_table(files)
    _log_graph_weighted_progress(60.0, 100.0, graph_start)

    if edge_table.empty:
        empty_node_df = pd.DataFrame(columns=GRAPH_NODE_COLUMNS)
        return empty_node_df

    return _build_node_features_from_edge_table(edge_table, graph_start=graph_start, log_progress=True)


def build_graph_intelligence_from_edge_table(edge_table, log_progress=False):
    """
    Build graph intelligence directly from a prepared edge table.
    """
    if edge_table.empty:
        return pd.DataFrame(columns=GRAPH_NODE_COLUMNS)

    graph_start = perf_counter() if log_progress else None
    return _build_node_features_from_edge_table(edge_table, graph_start=graph_start, log_progress=log_progress)


def build_graph_intelligence_from_transactions(transaction_df):
    """
    Build graph intelligence directly from an in-memory transaction dataframe.
    """
    edge_table = _prepare_edge_table_from_transactions(transaction_df)

    if edge_table.empty:
        return pd.DataFrame(columns=GRAPH_NODE_COLUMNS)

    return _build_node_features_from_edge_table(edge_table, log_progress=False)
