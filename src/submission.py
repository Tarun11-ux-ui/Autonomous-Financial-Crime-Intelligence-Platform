from time import perf_counter

import pandas as pd
from src.config import SUBMISSION_PATH
from src.decision_engine import apply_decision_engine
from src.explainability import generate_account_explanations
from src.impact_metrics import apply_impact_metrics, summarize_impact_metrics
from src.investigator_ai import apply_investigator_layer, summarize_investigator_layer
from src.retriever_layer import FraudRetriever, apply_retriever_layer, summarize_retriever_layer
from src.progress_utils import build_progress_message
from src.risk_scoring import compute_risk_score


def _format_duration(seconds):
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"


def create_submission(models, test, suspicious_dict, predictions=None):
    submission_start = perf_counter()
    total_submission_phases = 8
    completed_submission_phases = 0

    def log_submission_progress():
        print("   [ETA] " + build_progress_message("Submission progress", completed_submission_phases, total_submission_phases, submission_start))

    # Only use numeric features (same as training)
    numeric_cols = test.select_dtypes(include=["number"]).columns
    features = [col for col in numeric_cols if col not in ["account_id"]]

    # Use pre-computed predictions if provided, otherwise compute them
    if predictions is None:
        X_test = test[features]
        # Ensemble predictions from stacked models
        lgb_model, xgb_model, cat_model = models[0]  # Using first fold models
        
        preds = (
            0.4 * lgb_model.predict_proba(X_test)[:, 1]
            + 0.3 * xgb_model.predict_proba(X_test)[:, 1]
            + 0.3 * cat_model.predict_proba(X_test)[:, 1]
        )
    else:
        preds = predictions

    submission = pd.DataFrame()
    submission["account_id"] = test["account_id"]
    submission["is_mule"] = preds

    # Risk scoring
    submission = compute_risk_score(submission)

    # Populate suspicious windows from detected data
    submission["suspicious_start"] = submission["account_id"].map(
        lambda x: suspicious_dict.get(x, ("N/A", "N/A"))[0]
    )
    submission["suspicious_end"] = submission["account_id"].map(
        lambda x: suspicious_dict.get(x, ("N/A", "N/A"))[1]
    )
    completed_submission_phases += 1
    log_submission_progress()

    layer_start = perf_counter()
    submission = apply_decision_engine(submission, test)
    print(f"   [TIME] Decision engine: {_format_duration(perf_counter() - layer_start)}")
    completed_submission_phases += 1
    log_submission_progress()

    layer_start = perf_counter()
    submission = generate_account_explanations(models, test, submission)
    print(f"   [TIME] Explainability enrichment: {_format_duration(perf_counter() - layer_start)}")
    completed_submission_phases += 1
    log_submission_progress()

    layer_start = perf_counter()
    submission = apply_impact_metrics(submission, test)
    print(f"   [TIME] Impact metrics enrichment: {_format_duration(perf_counter() - layer_start)}")
    completed_submission_phases += 1
    log_submission_progress()

    layer_start = perf_counter()
    submission = apply_investigator_layer(submission)
    print(f"   [TIME] Investigator layer: {_format_duration(perf_counter() - layer_start)}")
    completed_submission_phases += 1
    log_submission_progress()

    layer_start = perf_counter()
    submission = apply_retriever_layer(submission)
    print(f"   [TIME] Retriever layer: {_format_duration(perf_counter() - layer_start)}")
    completed_submission_phases += 1
    log_submission_progress()

    layer_start = perf_counter()
    knowledge_store_summary = FraudRetriever(submission).knowledge_store_summary()
    print(f"   [TIME] Knowledge store build: {_format_duration(perf_counter() - layer_start)}")
    completed_submission_phases += 1
    log_submission_progress()

    write_start = perf_counter()
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"   [TIME] Write submission.csv: {_format_duration(perf_counter() - write_start)}")
    completed_submission_phases += 1
    log_submission_progress()
    print("[OK] Submission saved with risk scoring, suspicious windows, actions, and explanations.")
    print(f"   [TIME] Total submission stage: {_format_duration(perf_counter() - submission_start)}")
    return (
        submission,
        summarize_impact_metrics(submission),
        summarize_investigator_layer(submission),
        summarize_retriever_layer(submission),
        knowledge_store_summary,
    )
