import pandas as pd

def compute_risk_score(df):

    # Risk score from model probability
    df["risk_score"] = (df["is_mule"] * 100)

    # Risk level classification
    def classify_risk(score):

        if score >= 80:
            return "CRITICAL"

        elif score >= 60:
            return "HIGH"

        elif score >= 40:
            return "MEDIUM"

        else:
            return "LOW"

    df["risk_level"] = df["risk_score"].apply(classify_risk)

    return df