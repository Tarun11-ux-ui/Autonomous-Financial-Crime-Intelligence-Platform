from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

def tune_model(X,y):

    params = {
        "num_leaves":[31,63,127],
        "max_depth":[6,8,10],
        "learning_rate":[0.01,0.05,0.1],
        "n_estimators":[300,500,800]
    }

    model = lgb.LGBMClassifier(class_weight="balanced")

    search = RandomizedSearchCV(
        model,
        params,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    search.fit(X,y)

    print("Best params:", search.best_params_)
    return search.best_estimator_