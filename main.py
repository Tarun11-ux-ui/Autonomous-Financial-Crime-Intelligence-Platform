from time import perf_counter

from src.data_loader import load_train_data, load_test_data
from src.feature_engineering import build_transaction_features
from src.model import train_final_model
from src.submission import create_submission
from src.suspicious_window import detect_suspicious_windows
from src.explainability import generate_explanations
from src.ollama_rag import summarize_ollama_integration
from src.progress_utils import build_progress_message


def format_duration(seconds):
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {remaining_seconds:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"


def log_stage_duration(stage_name, start_time):
    elapsed = perf_counter() - start_time
    print(f"   [TIME] {stage_name}: {format_duration(elapsed)}")
    return elapsed


pipeline_start = perf_counter()
stage_timings = []
total_pipeline_phases = 10


def log_pipeline_progress(completed_phases):
    print("   [ETA] " + build_progress_message("Pipeline progress", completed_phases, total_pipeline_phases, pipeline_start))

print("=" * 70)
print("[*] MULE DETECTION PIPELINE - STARTING")
print("=" * 70)

print("\n[>] Loading training data...")
stage_start = perf_counter()
train = load_train_data()
print(f"   [OK] Training data shape: {train.shape}")
stage_timings.append(("Load training data", log_stage_duration("Load training data", stage_start)))
log_pipeline_progress(1)

print("\n[>] Building transaction features...")
stage_start = perf_counter()
txn_features, pipeline_metadata = build_transaction_features(return_metadata=True)
print(f"   [OK] Transaction features shape: {txn_features.shape}")
stage_timings.append(("Build transaction features", log_stage_duration("Build transaction features", stage_start)))
log_pipeline_progress(2)

print("\n[>] Merging features with training data...")
stage_start = perf_counter()
train = train.merge(txn_features, on="account_id", how="left")
train = train.fillna(0)
print(f"   [OK] Training data after merge: {train.shape}")
stage_timings.append(("Merge training features", log_stage_duration("Merge training features", stage_start)))
log_pipeline_progress(3)

print("\n[>] Training ensemble models...")
stage_start = perf_counter()
models = train_final_model(train)
print(f"   [OK] Trained {len(models)} fold models")
stage_timings.append(("Train ensemble models", log_stage_duration("Train ensemble models", stage_start)))
log_pipeline_progress(4)

print("\n[>] Generating model explanations...")
stage_start = perf_counter()
numeric_cols = train.select_dtypes(include=["number"]).columns
X_train_sample = train[numeric_cols].drop(columns=["is_mule"]).sample(
    n=min(1000, len(train)),
    random_state=42,
)
explanations = generate_explanations(models, X_train_sample)
stage_timings.append(("Generate model explanations", log_stage_duration("Generate model explanations", stage_start)))
log_pipeline_progress(5)

print("\n[>] Loading test data...")
stage_start = perf_counter()
test = load_test_data()
print(f"   [OK] Test data shape: {test.shape}")
stage_timings.append(("Load test data", log_stage_duration("Load test data", stage_start)))
log_pipeline_progress(6)

print("\n[>] Merging test features...")
stage_start = perf_counter()
test = test.merge(txn_features, on="account_id", how="left")
test = test.fillna(0)
print(f"   [OK] Test data after merge: {test.shape}")
stage_timings.append(("Merge test features", log_stage_duration("Merge test features", stage_start)))
log_pipeline_progress(7)

print("\n[>] Computing ensemble predictions...")
stage_start = perf_counter()
numeric_cols = test.select_dtypes(include=["number"]).columns
features = [col for col in numeric_cols if col not in ["account_id"]]
X_test = test[features]

lgb_model, xgb_model, cat_model = models[0]
predictions = (
    0.4 * lgb_model.predict_proba(X_test)[:, 1]
    + 0.3 * xgb_model.predict_proba(X_test)[:, 1]
    + 0.3 * cat_model.predict_proba(X_test)[:, 1]
)
print("   [OK] Predictions computed")
stage_timings.append(("Compute ensemble predictions", log_stage_duration("Compute ensemble predictions", stage_start)))
log_pipeline_progress(8)

print("\n[!] Detecting suspicious transaction windows...")
stage_start = perf_counter()
suspicious_dict = detect_suspicious_windows(
    models=models,
    test=test,
    predictions=predictions,
    date_ranges=pipeline_metadata.get("date_ranges"),
)
print(f"   [OK] Found suspicious activity in {len(suspicious_dict)} accounts")
stage_timings.append(("Detect suspicious windows", log_stage_duration("Detect suspicious windows", stage_start)))
log_pipeline_progress(9)

print("\n[>] Creating submission...")
stage_start = perf_counter()
submission, impact_summary, investigator_summary, retriever_summary, knowledge_store_summary = create_submission(
    models,
    test,
    suspicious_dict,
    predictions=predictions,
)
stage_timings.append(("Create submission", log_stage_duration("Create submission", stage_start)))
log_pipeline_progress(10)

print("\n[>] Impact dashboard metrics...")
print(f"   [OK] Accounts screened: {impact_summary['total_accounts_screened']}")
print(f"   [OK] Cases generated: {impact_summary['cases_generated']} ({impact_summary['case_generation_rate']}%)")
print(f"   [OK] Auto actions: {impact_summary['auto_actions_triggered']} ({impact_summary['auto_action_rate']}%)")
print(f"   [OK] Freeze actions: {impact_summary['freeze_actions']}")
print(f"   [OK] Compliance alerts: {impact_summary['compliance_alerts']}")
print(f"   [OK] Estimated fraud prevented: INR {impact_summary['estimated_fraud_prevented_inr']:.2f}")

print("\n[>] Investigator AI layer...")
print(f"   [OK] Investigator-ready accounts: {investigator_summary['investigator_ready_accounts']}")
print(f"   [OK] Accounts with connected-risk context: {investigator_summary['accounts_with_connections']}")
print(f"   [OK] Supported queries include: {investigator_summary['query_examples'][0]}")

print("\n[>] Retriever layer...")
print(f"   [OK] Retriever-ready documents: {retriever_summary['retriever_ready_documents']}")
print(f"   [OK] Documents with network context: {retriever_summary['documents_with_network_context']}")
print(f"   [OK] Supported retrieval query: {retriever_summary['query_examples'][0]}")

print("\n[>] Knowledge store...")
print(f"   [OK] Knowledge store backend: {knowledge_store_summary['knowledge_store_backend']}")
print(f"   [OK] Indexed documents: {knowledge_store_summary['knowledge_store_documents']}")
print(f"   [OK] Vector dimension: {knowledge_store_summary['knowledge_store_vector_dim']}")

ollama_summary = summarize_ollama_integration()
print("\n[>] Ollama RAG integration...")
print(f"   [OK] Base URL: {ollama_summary['ollama_base_url']}")
print(f"   [OK] Model target: {ollama_summary['ollama_model']}")
print(f"   [OK] Ollama available: {ollama_summary['ollama_available']}")
print(f"   [OK] Model installed: {ollama_summary['ollama_model_installed']}")

total_runtime = perf_counter() - pipeline_start

print("\n[>] Execution timing summary...")
for stage_name, elapsed in stage_timings:
    print(f"   [TIME] {stage_name:<32} {format_duration(elapsed)}")
print(f"   [TIME] Total pipeline runtime: {format_duration(total_runtime)}")

print("\n" + "=" * 70)
print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nOutput files:")
print("  [*] data/submission.csv - Predictions with risk scores, suspicious windows, actions, and explanations")
print("=" * 70)
