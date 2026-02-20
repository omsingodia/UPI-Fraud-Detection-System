import pandas as pd


def apply_features(df, training=False):
    """
    Applies consistent feature engineering.
    training=True -> keeps isFraud
    """

    # Drop identifiers
    df = df.copy()
    df.drop(["nameOrig", "nameDest"], axis=1, inplace=True, errors="ignore")

    # -----------------------------
    # Core Features
    # -----------------------------

    df["relative_amount"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    df["balance_change_ratio"] = (
        df["newbalanceOrig"] - df["oldbalanceOrg"]
    ) / (df["oldbalanceOrg"] + 1)

    # Transaction velocity
    df["velocity"] = df.groupby("step")["amount"].transform("count")

    # Receiver risk
    receiver_counts = df["oldbalanceDest"].value_counts()
    df["receiver_risk"] = df["oldbalanceDest"].map(receiver_counts)

    df["rare_receiver"] = df["receiver_risk"].apply(lambda x: 1 if x < 3 else 0)

    # Mule indicator
    df["mule_indicator"] = df.groupby(
        ["step", "oldbalanceDest"]
    )["amount"].transform("count")

    # Encode type
    df["type"] = df["type"].astype("category").cat.codes

    if not training:
        df.drop("isFraud", axis=1, inplace=True, errors="ignore")

    return df