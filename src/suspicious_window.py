import pandas as pd
from time import perf_counter
from glob import glob
from src.data_loader import load_test_data
from src.feature_engineering import build_transaction_features
from src.model import train_final_model
from src.data_loader import load_train_data
from src.config import TRANSACTION_PATH
from src.progress_utils import build_progress_message, format_duration


def _load_transaction_dates():
    """Load transaction data and extract date ranges per account."""
    files = glob(TRANSACTION_PATH)
    date_ranges = {}
    load_start = perf_counter()
    
    for idx, file in enumerate(files, 1):
        print(
            "   [ETA] " + build_progress_message("Loading suspicious window dates", idx, len(files), load_start),
            end="\r",
        )
        df = pd.read_parquet(file)
        df["timestamp"] = pd.to_datetime(df["transaction_timestamp"], format="mixed")
        
        # Get min and max timestamps per account
        account_dates = df.groupby("account_id")["timestamp"].agg(["min", "max"])
        
        for account_id, row in account_dates.iterrows():
            if account_id not in date_ranges:
                date_ranges[account_id] = (row["min"], row["max"])
            else:
                # Update with earliest start and latest end
                date_ranges[account_id] = (
                    min(date_ranges[account_id][0], row["min"]),
                    max(date_ranges[account_id][1], row["max"])
                )
    
    if files:
        print()
        print(f"   [ETA] Suspicious window date scan completed in {format_duration(perf_counter() - load_start)}")

    return date_ranges


def detect_suspicious_windows(models=None, test=None, predictions=None, date_ranges=None):
    """
    Detect suspicious transaction windows for accounts with high risk scores.
    
    Args:
        models: Pre-trained ensemble models (optional, for efficiency)
        test: Test dataframe (optional)
        predictions: Pre-computed predictions (optional)
    
    Returns:
        Dictionary mapping account_id to (suspicious_start, suspicious_end) tuples.
    """
    
    # If predictions not provided, compute them
    if predictions is None:
        if test is None:
            test = load_test_data()
        if models is None:
            train = load_train_data()
            txn_features = build_transaction_features()
            train = train.merge(txn_features, on="account_id", how="left").fillna(0)
            test = test.merge(txn_features, on="account_id", how="left").fillna(0)
            models = train_final_model(train)
        else:
            # test already has features merged
            pass
        
        # Get predictions on test set
        numeric_cols = test.select_dtypes(include=["number"]).columns
        features = [col for col in numeric_cols if col not in ["account_id"]]
        X_test = test[features]
        
        # Ensemble predictions
        lgb_model, xgb_model, cat_model = models[0]  # Using first fold models
        
        final_pred = (
            0.4 * lgb_model.predict_proba(X_test)[:, 1]
            + 0.3 * xgb_model.predict_proba(X_test)[:, 1]
            + 0.3 * cat_model.predict_proba(X_test)[:, 1]
        )
    else:
        final_pred = predictions
    
    # Create submission dataframe with predictions
    submission = pd.DataFrame({
        "account_id": test["account_id"].values,
        "risk_score": final_pred
    })
    
    threshold = 0.80
    
    # Mark suspicious accounts
    submission["suspicious_flag"] = (submission["risk_score"] > threshold).astype(int)
    
    # Count unique suspicious accounts
    suspicious_accounts = submission.loc[
        submission["suspicious_flag"] == 1, "account_id"
    ].nunique()
    
    print(f"   [OK] Found suspicious activity in {suspicious_accounts} accounts")
    
    # Load actual transaction dates
    if date_ranges is None:
        date_ranges = _load_transaction_dates()
    
    # Create mapping of account_id to suspicious windows
    suspicious_dict = {}
    for _, row in submission[submission["suspicious_flag"] == 1].iterrows():
        account_id = row["account_id"]
        if account_id in date_ranges:
            start_date, end_date = date_ranges[account_id]
            # Format dates as strings
            suspicious_dict[account_id] = (
                start_date.strftime("%Y-%m-%d %H:%M:%S"),
                end_date.strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            suspicious_dict[account_id] = ("N/A", "N/A")
    
    return suspicious_dict
