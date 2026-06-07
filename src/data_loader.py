import pandas as pd
from src.config import DATA_PATH

def load_train_data():
    accounts = pd.read_parquet(DATA_PATH + "accounts.parquet")
    labels = pd.read_parquet(DATA_PATH + "train_labels.parquet")

    train = accounts.merge(
        labels[["account_id", "is_mule"]],
        on="account_id",
        how="inner"
    )
    
    if len(train) == 0:
        raise ValueError("No matching accounts found in training data merge")

    return train


def load_test_data():
    accounts = pd.read_parquet(DATA_PATH + "accounts.parquet")
    test_ids = pd.read_parquet(DATA_PATH + "test_accounts.parquet")

    test = accounts.merge(
        test_ids,
        on="account_id",
        how="inner"
    )
    
    if len(test) == 0:
        raise ValueError("No matching accounts found in test data merge")

    return test