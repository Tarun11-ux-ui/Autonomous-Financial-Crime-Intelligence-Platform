import numpy as np
import shap
import pandas as pd
from time import perf_counter
from src.progress_utils import build_progress_message, format_duration


FEATURE_LABELS = {
    "burst_count": "rapid transaction bursts",
    "rapid_drain": "rapid outward fund movements",
    "rapid_drain_ratio": "rapid outward drain ratio",
    "night_txn_count": "night-hour transaction activity",
    "near_threshold_count": "near-threshold transaction count",
    "high_risk_exposure": "high-risk counterparty exposure",
    "high_risk_exposure_ratio": "high-risk counterparty exposure ratio",
    "unique_counterparty": "breadth of connected counterparties",
    "txn_velocity": "transaction velocity",
    "top_counterparty_ratio": "counterparty concentration",
    "flow_through_score": "pass-through fund flow pattern",
    "degree_ratio": "outgoing to incoming degree ratio",
    "counterparty_entropy": "counterparty entropy",
    "incoming_total": "incoming transaction value",
    "outgoing_total": "outgoing transaction value",
    "txn_count": "transaction count",
    "avg_amt": "average transaction amount",
    "pagerank_score": "graph influence score",
    "betweenness_centrality": "network bridge centrality",
    "clustering_coefficient": "local fraud-ring clustering",
    "community_size": "community size",
    "community_density": "community density",
    "community_risk_score": "community risk score",
    "graph_risk_score": "graph intelligence risk score",
}


def _feature_label(feature_name):
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))


def _normalize_shap_values(shap_values):
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        class_index = 1 if shap_array.shape[-1] > 1 else 0
        shap_array = shap_array[:, :, class_index]

    return shap_array


def _compute_model_shap(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return _normalize_shap_values(shap_values)


def _compute_ensemble_shap(models, X_sample):
    if not models:
        return pd.DataFrame(), []

    lgb_model, xgb_model, cat_model = models[0]

    weighted_values = None
    total_weight = 0.0
    models_used = []

    model_specs = [
        ("LightGBM", lgb_model, 0.4),
        ("XGBoost", xgb_model, 0.3),
        ("CatBoost", cat_model, 0.3),
    ]
    shap_start = perf_counter()

    for idx, (model_name, model, weight) in enumerate(model_specs, 1):
        try:
            shap_values = _compute_model_shap(model, X_sample)
            if weighted_values is None:
                weighted_values = weight * shap_values
            else:
                weighted_values += weight * shap_values
            total_weight += weight
            models_used.append(model_name)
            print("   [ETA] " + build_progress_message("Explainability model steps", idx, len(model_specs), shap_start))
        except Exception as exc:
            print(f"{model_name} SHAP explanation failed: {exc}")

    if weighted_values is None or total_weight == 0:
        return pd.DataFrame(), models_used

    ensemble_values = weighted_values / total_weight
    print(f"   [ETA] Explainability SHAP computation completed in {format_duration(perf_counter() - shap_start)}")
    return pd.DataFrame(ensemble_values, columns=X_sample.columns, index=X_sample.index), models_used


def _format_feature_value(feature_name, feature_value):
    if pd.isna(feature_value):
        return "not available"

    if feature_name in {
        "burst_count",
        "rapid_drain",
        "night_txn_count",
        "near_threshold_count",
        "high_risk_exposure",
        "unique_counterparty",
        "txn_count",
    }:
        return str(int(round(feature_value)))

    if abs(feature_value) >= 1000:
        return f"{feature_value:,.0f}"

    return f"{feature_value:.2f}"


def _driver_sentence(feature_name, feature_value, shap_impact):
    direction = "increased" if shap_impact >= 0 else "reduced"
    return (
        f"{_feature_label(feature_name)} ({_format_feature_value(feature_name, feature_value)}) "
        f"{direction} the model risk score"
    )


def _default_explanation_row(row):
    return {
        "explanation_generated": "NO",
        "models_used_for_explanation": "N/A",
        "explanation_summary": f"Account remains {row['risk_level']} risk with no enhanced explanation required.",
        "model_top_drivers": "Baseline monitoring only",
        "model_driver_1": "N/A",
        "model_driver_2": "N/A",
        "model_driver_3": "N/A",
    }


def _build_local_explanation(scored_row, feature_row, shap_row, feature_cols, models_used, top_k=3):
    positive_impacts = shap_row[shap_row > 0].sort_values(ascending=False)
    explanation_prefix = f"Account is {scored_row['risk_level']} risk because "
    if positive_impacts.empty:
        positive_impacts = shap_row.abs().sort_values(ascending=False)
        explanation_prefix = (
            f"Account is {scored_row['risk_level']} risk. The strongest model signals were "
        )

    top_features = positive_impacts.head(top_k).index.tolist()

    drivers = []
    summary_parts = []
    for feature_name in top_features:
        feature_value = feature_row.get(feature_name, 0)
        shap_impact = shap_row[feature_name]
        drivers.append(f"{_feature_label(feature_name)} ({shap_impact:+.3f})")
        summary_parts.append(_driver_sentence(feature_name, feature_value, shap_impact))

    if not summary_parts:
        summary_parts.append("No dominant feature drivers were available for this account.")

    explanation_summary = explanation_prefix + "; ".join(summary_parts) + "."

    driver_values = drivers + ["N/A"] * max(0, top_k - len(drivers))

    return {
        "explanation_generated": "YES",
        "models_used_for_explanation": ", ".join(models_used) if models_used else "N/A",
        "explanation_summary": explanation_summary,
        "model_top_drivers": " | ".join(drivers) if drivers else "N/A",
        "model_driver_1": driver_values[0],
        "model_driver_2": driver_values[1],
        "model_driver_3": driver_values[2],
    }


def generate_explanations(models, X_sample, save_plots=False):
    """
    Generate SHAP explanations using the first fold's models.
    
    Args:
        models: List of tuples (lgb_model, xgb_model, cat_model) from train_final_model
        X_sample: Sample of features to explain
        save_plots: Whether to save SHAP plots
    
    Returns:
        Dictionary with feature importance from each model
    """
    
    print("\n" + "="*60)
    print("[**] GENERATING EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    if not models:
        print("Warning: No models provided")
        return {}

    shap_df, models_used = _compute_ensemble_shap(models, X_sample)
    if shap_df.empty:
        print("Warning: SHAP generation failed for all models")
        return {}

    global_importance = shap_df.abs().mean().sort_values(ascending=False)
    importance_df = (
        global_importance.rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    importance_df["feature_label"] = importance_df["feature"].apply(_feature_label)
    importance_df["models_used"] = ", ".join(models_used)

    print("\nTop 15 Features (Ensemble SHAP Importance):")
    print(importance_df.head(15)[["feature", "importance"]])
    print("\n[OK] Feature importance computed in memory")

    print("\n" + "="*60)
    return {
        "Ensemble": global_importance.head(15),
        "importance_table": importance_df,
        "models_used": models_used,
    }


def generate_account_explanations(models, feature_df, scored_df, risk_threshold=50, top_k=3):
    """
    Create per-account explanations for accounts that require human review.

    Args:
        models: Trained ensemble models.
        feature_df: Feature-enriched dataframe used for scoring.
        scored_df: Output dataframe containing risk and decision columns.
        risk_threshold: Minimum risk score that triggers model-level explanation.
        top_k: Number of top feature drivers to expose.

    Returns:
        DataFrame enriched with explanation columns.
    """
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns
    feature_cols = [col for col in numeric_cols if col not in ["account_id", "is_mule"]]

    explanation_df = scored_df.copy()
    explanation_df["explanation_generated"] = "NO"
    explanation_df["models_used_for_explanation"] = "N/A"
    explanation_df["explanation_summary"] = ""
    explanation_df["model_top_drivers"] = "N/A"
    explanation_df["model_driver_1"] = "N/A"
    explanation_df["model_driver_2"] = "N/A"
    explanation_df["model_driver_3"] = "N/A"

    explanation_mask = (
        (explanation_df["risk_score"] >= risk_threshold)
        | (explanation_df["auto_action_eligible"] == "YES")
        | (explanation_df["suspicious_start"] != "N/A")
    )

    explanation_candidates = explanation_df.loc[explanation_mask, ["account_id"]].merge(
        feature_df[["account_id", *feature_cols]],
        on="account_id",
        how="left",
    )

    if explanation_candidates.empty:
        defaults = explanation_df.apply(_default_explanation_row, axis=1, result_type="expand")
        explanation_df[defaults.columns] = defaults
        return explanation_df

    feature_matrix = explanation_candidates[feature_cols].fillna(0)
    shap_df, models_used = _compute_ensemble_shap(models, feature_matrix)

    if shap_df.empty:
        defaults = explanation_df.apply(_default_explanation_row, axis=1, result_type="expand")
        explanation_df[defaults.columns] = defaults
    else:
        candidate_output = explanation_candidates[["account_id", *feature_cols]].copy()
        candidate_output.index = explanation_candidates["account_id"]
        shap_df.index = explanation_candidates["account_id"]

        local_rows = []
        for account_id in shap_df.index:
            scored_row = explanation_df.loc[explanation_df["account_id"] == account_id].iloc[0]
            feature_row = candidate_output.loc[account_id]
            local_rows.append(
                pd.Series(
                    _build_local_explanation(
                        scored_row,
                        feature_row,
                        shap_df.loc[account_id],
                        feature_cols,
                        models_used,
                        top_k=top_k,
                    ),
                    name=account_id,
                )
            )

        local_explanations = pd.DataFrame(local_rows).reset_index().rename(columns={"index": "account_id"})
        explanation_df = explanation_df.merge(local_explanations, on="account_id", how="left", suffixes=("", "_local"))

        for column in [
            "explanation_generated",
            "models_used_for_explanation",
            "explanation_summary",
            "model_top_drivers",
            "model_driver_1",
            "model_driver_2",
            "model_driver_3",
        ]:
            explanation_df[column] = explanation_df[f"{column}_local"].fillna(explanation_df[column])
            explanation_df = explanation_df.drop(columns=[f"{column}_local"])

        default_mask = explanation_df["explanation_summary"] == ""
        defaults = explanation_df.loc[default_mask].apply(_default_explanation_row, axis=1, result_type="expand")
        explanation_df.loc[default_mask, defaults.columns] = defaults

    explanation_df["human_readable_reasoning"] = explanation_df["decision_reasons"] + " | " + explanation_df["explanation_summary"]

    return explanation_df
