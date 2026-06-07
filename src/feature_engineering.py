import pandas as pd
import numpy as np
import sys
from time import perf_counter
from glob import glob
from src.graph_intelligence import build_graph_intelligence, build_graph_intelligence_from_edge_table, _prepare_edge_table_from_batch
from src.config import TRANSACTION_PATH
from src.progress_utils import build_progress_message, format_duration


TRANSACTION_COLUMNS = [
    "account_id",
    "counterparty_id",
    "transaction_timestamp",
    "amount",
    "txn_type",
]


def build_transaction_features(return_metadata=False):

    files = glob(TRANSACTION_PATH)
    feature_chunks = []
    edge_chunks = []
    date_ranges = {}
    processing_start = perf_counter()

    print(f"Found {len(files)} transaction files")
    sys.stdout.flush()

    for idx, file in enumerate(files, 1):
        print(
            "  " + build_progress_message("Processing transaction files", idx, len(files), processing_start),
            end="\r",
        )
        sys.stdout.flush()

        df = pd.read_parquet(file, columns=TRANSACTION_COLUMNS)

        df["timestamp"] = pd.to_datetime(df["transaction_timestamp"], format="mixed")
        df = df.sort_values(["account_id", "timestamp"])

        df["amount"] = df["amount"].astype(float)

        df["is_in"] = (df["txn_type"] == "C").astype(int)
        df["is_out"] = (df["txn_type"] == "D").astype(int)

        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.weekday
        df["night_txn"] = (df["hour"] <= 5).astype(int)

        df["time_diff"] = (
            df.groupby("account_id")["timestamp"]
            .diff()
            .dt.total_seconds()
        )

        df["burst_txn"] = (df["time_diff"] < 300).astype(int)

        df["rapid_out"] = (
            (df["is_out"] == 1) & (df["time_diff"] < 600)
        ).astype(int)

        df["near_50k"] = (
            (df["amount"] > 49000) & (df["amount"] < 50000)
        ).astype(int)

        # High risk counterparties
        cp_global_count = df["counterparty_id"].value_counts()

        high_risk_counterparties = cp_global_count[
            cp_global_count > cp_global_count.quantile(0.99)
        ].index

        df["high_risk_cp_flag"] = (
            df["counterparty_id"].isin(high_risk_counterparties)
        ).astype(int)

        # Counterparty entropy
        entropy = df.groupby("account_id")["counterparty_id"].apply(
            lambda x: -(x.value_counts(normalize=True) *
                        np.log(x.value_counts(normalize=True))).sum()
        )

        cp_counts = df.groupby(["account_id", "counterparty_id"]).size()

        agg = df.groupby("account_id").agg(
            txn_count=("amount", "count"),
            total_amt=("amount", "sum"),
            avg_amt=("amount", "mean"),
            std_amt=("amount", "std"),
            max_amt=("amount", "max"),

            night_txn_count=("night_txn", "sum"),
            burst_count=("burst_txn", "sum"),
            near_threshold_count=("near_50k", "sum"),
            rapid_drain=("rapid_out", "sum"),
            high_risk_exposure=("high_risk_cp_flag", "sum"),

            unique_counterparty=("counterparty_id", "nunique"),

            first_txn=("timestamp", "min"),
            last_txn=("timestamp", "max"),
        ).reset_index()

        # Separate aggregations for incoming/outgoing
        incoming_total = df[df["is_in"] == 1].groupby("account_id")["amount"].sum()
        outgoing_total = df[df["is_out"] == 1].groupby("account_id")["amount"].sum()
        agg = agg.merge(incoming_total.to_frame("incoming_total").reset_index(), on="account_id", how="left")
        agg = agg.merge(outgoing_total.to_frame("outgoing_total").reset_index(), on="account_id", how="left")

        out_degree = (
            df[df["is_out"] == 1]
            .groupby("account_id")["counterparty_id"]
            .nunique()
        )

        in_degree = (
            df[df["is_in"] == 1]
            .groupby("account_id")["counterparty_id"]
            .nunique()
        )

        agg = agg.merge(out_degree.to_frame("out_degree").reset_index(), on="account_id", how="left")
        agg = agg.merge(in_degree.to_frame("in_degree").reset_index(), on="account_id", how="left")

        agg["active_days"] = (
            (agg["last_txn"] - agg["first_txn"]).dt.days
        )

        agg["degree_ratio"] = agg["out_degree"] / (agg["in_degree"] + 1)

        agg["flow_through_score"] = (
            abs(agg["incoming_total"] - agg["outgoing_total"])
            / (agg["total_amt"] + 1)
        )

        agg["rapid_drain_ratio"] = (
            agg["rapid_drain"] / (agg["txn_count"] + 1)
        )

        agg["high_risk_exposure_ratio"] = (
            agg["high_risk_exposure"] / (agg["txn_count"] + 1)
        )

        agg["txn_velocity"] = agg["txn_count"] / (agg["active_days"] + 1)

        top_ratio = (
            cp_counts.groupby(level=0)
            .apply(lambda x: x.max() / x.sum())
        )

        agg = agg.merge(
            top_ratio.rename("top_counterparty_ratio"),
            left_on="account_id",
            right_index=True,
            how="left",
        )

        agg = agg.merge(
            entropy.rename("counterparty_entropy"),
            left_on="account_id",
            right_index=True,
            how="left",
        )

        agg = agg.fillna(0)
        # Ensure all numeric columns are float type
        numeric_cols = agg.select_dtypes(include=["number"]).columns
        agg[numeric_cols] = agg[numeric_cols].astype(float)
        account_dates = agg[["account_id", "first_txn", "last_txn"]].copy()

        for row in account_dates.itertuples(index=False):
            existing = date_ranges.get(row.account_id)
            if existing is None:
                date_ranges[row.account_id] = (row.first_txn, row.last_txn)
            else:
                date_ranges[row.account_id] = (
                    min(existing[0], row.first_txn),
                    max(existing[1], row.last_txn),
                )

        agg = agg.drop(columns=["first_txn", "last_txn"])

        feature_chunks.append(agg)
        edge_chunks.append(_prepare_edge_table_from_batch(df))

    print()
    print(f"   [ETA] Transaction file processing completed in {format_duration(perf_counter() - processing_start)}")

    final = pd.concat(feature_chunks)
    final = final.groupby("account_id").mean().reset_index()
    final = final.fillna(0)

    print("\n[>] Building graph intelligence layer...")
    sys.stdout.flush()
    if edge_chunks:
        edge_table = pd.concat(edge_chunks, ignore_index=True)
        edge_table = edge_table.groupby(["source_account", "target_account"], as_index=False).agg(
            graph_txn_count=("graph_txn_count", "sum"),
            graph_total_amount=("graph_total_amount", "sum"),
        )
        graph_features = build_graph_intelligence_from_edge_table(edge_table, log_progress=True)
    else:
        graph_features = build_graph_intelligence(files)
    if not graph_features.empty:
        final = final.merge(graph_features, on="account_id", how="left")
        numeric_cols = final.select_dtypes(include=["number"]).columns
        object_cols = final.select_dtypes(include=["object"]).columns
        final[numeric_cols] = final[numeric_cols].fillna(0)
        for column in object_cols:
            final[column] = final[column].fillna("NO")

    if return_metadata:
        return final, {"date_ranges": date_ranges}

    return final
