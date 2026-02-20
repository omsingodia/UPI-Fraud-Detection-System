import pandas as pd


def apply_features(df, training=False):
    """
    Applies consistent feature engineering.
    training=True -> keeps isFraud
    Must match the exact logic used during training.
    """

    df = df.copy()
    df.drop(["nameOrig", "nameDest"], axis=1, inplace=True, errors="ignore")

    # ------------------------------------------------------------------
    # Core Features (must match train.py exactly)
    # ------------------------------------------------------------------

    df["relative_amount"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    df["balance_change_ratio"] = (
        df["newbalanceOrig"] - df["oldbalanceOrg"]
    ) / (df["oldbalanceOrg"] + 1)

    # Transaction velocity — for single rows, default to 1
    if "step" in df.columns and len(df) > 1:
        df["velocity"] = df.groupby("step")["amount"].transform("count")
    else:
        df["velocity"] = 1

    # Receiver risk — for single rows, use oldbalanceDest directly
    if len(df) > 1:
        receiver_counts = df["oldbalanceDest"].value_counts()
        df["receiver_risk"] = df["oldbalanceDest"].map(receiver_counts)
    else:
        df["receiver_risk"] = 1

    df["rare_receiver"] = df["receiver_risk"].apply(lambda x: 1 if x < 3 else 0)

    # Mule indicator — for single rows, default to 1
    if len(df) > 1:
        df["mule_indicator"] = df.groupby(
            ["step", "oldbalanceDest"]
        )["amount"].transform("count")
    else:
        df["mule_indicator"] = 1

    # Encode type
    type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "CASH_IN": 3, "DEBIT": 4}
    if df["type"].dtype == object:
        df["type"] = df["type"].map(type_map).fillna(1)
    else:
        df["type"] = df["type"].astype("category").cat.codes

    if not training:
        df.drop("isFraud", axis=1, inplace=True, errors="ignore")

    return df
