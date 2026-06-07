import os
import joblib
import numpy as np
import pandas as pd
from time import perf_counter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from src.progress_utils import build_progress_message, format_duration


def train_final_model(train_df):

    print("Training ensemble models...")
    training_start = perf_counter()

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Extract features and target
    numeric_cols = train_df.select_dtypes(include=["number"]).columns
    X = train_df[numeric_cols].drop(columns=["is_mule", "account_id"], errors="ignore")
    y = train_df["is_mule"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ensemble_models = []

    auc_scores = []
    total_training_steps = skf.get_n_splits() * 3
    completed_training_steps = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start = perf_counter()

        print(f"\n--- Ensemble Fold {fold+1} ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # ---------------------------
        # LightGBM
        # ---------------------------

        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        lgb_model.fit(X_train, y_train)

        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]

        # Save model
        lgb_path = f"models/lgbm_fold{fold+1}.pkl"
        joblib.dump(lgb_model, lgb_path)

        print(f"Saved {lgb_path}")
        completed_training_steps += 1
        print(f"   [ETA] {build_progress_message('Model training progress', completed_training_steps, total_training_steps, training_start)}")

        # ---------------------------
        # XGBoost
        # ---------------------------

        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        xgb_model.fit(X_train, y_train)

        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]

        xgb_path = f"models/xgb_fold{fold+1}.pkl"
        joblib.dump(xgb_model, xgb_path)

        print(f"Saved {xgb_path}")
        completed_training_steps += 1
        print(f"   [ETA] {build_progress_message('Model training progress', completed_training_steps, total_training_steps, training_start)}")

        # ---------------------------
        # CatBoost
        # ---------------------------

        cat_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            verbose=False,
            random_seed=42
        )

        cat_model.fit(X_train, y_train)

        cat_pred = cat_model.predict_proba(X_val)[:, 1]

        cat_path = f"models/cat_fold{fold+1}.pkl"
        joblib.dump(cat_model, cat_path)

        print(f"Saved {cat_path}")
        completed_training_steps += 1
        print(f"   [ETA] {build_progress_message('Model training progress', completed_training_steps, total_training_steps, training_start)}")

        # ---------------------------
        # Ensemble prediction
        # ---------------------------

        ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3

        auc = roc_auc_score(y_val, ensemble_pred)

        print(f"Fold {fold+1} Ensemble AUC: {auc:.4f}")

        auc_scores.append(auc)

        # Store as tuple for this fold
        ensemble_models.append((lgb_model, xgb_model, cat_model))
        print(f"Fold {fold+1} completed in {format_duration(perf_counter() - fold_start)}")

    print("\n[OK] Mean Ensemble AUC:", np.mean(auc_scores))
    print("[OK] Trained 5 fold models")
    print(f"[ETA] Total model training time: {format_duration(perf_counter() - training_start)}")

    return ensemble_models
