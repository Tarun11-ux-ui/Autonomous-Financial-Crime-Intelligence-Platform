import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_stacked_model(train_df):

    numeric_cols = train_df.select_dtypes(include=["number"]).columns
    X = train_df[numeric_cols].drop(columns=["is_mule"])
    y = train_df["is_mule"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    lgb_model = lgb.LGBMClassifier(n_estimators=500, class_weight="balanced")
    xgb_model = xgb.XGBClassifier(n_estimators=500, scale_pos_weight=5)

    lgb_model.fit(X_train,y_train)
    xgb_model.fit(X_train,y_train)

    lgb_pred = lgb_model.predict_proba(X_val)[:,1]
    xgb_pred = xgb_model.predict_proba(X_val)[:,1]

    final_pred = 0.6*lgb_pred + 0.4*xgb_pred

    auc = roc_auc_score(y_val, final_pred)
    print("Stacked AUC:", auc)

    return lgb_model, xgb_model