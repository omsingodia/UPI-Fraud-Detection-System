import pandas as pd
from fraud_engine import FraudEngine

print("🚀 Testing Fraud Engine...")

# Load small sample
df = pd.read_csv("../dataset/PS_20174392719_1491204439457_log.csv", nrows=5)

# SAME feature engineering as training
df.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

df["relative_amount"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

df["balance_change_ratio"] = (
    df["newbalanceOrig"] - df["oldbalanceOrg"]
) / (df["oldbalanceOrg"] + 1)

df["velocity"] = df.groupby("step")["amount"].transform("count")

receiver_counts = df["oldbalanceDest"].value_counts()
df["receiver_risk"] = df["oldbalanceDest"].map(receiver_counts)
df["rare_receiver"] = df["receiver_risk"].apply(lambda x: 1 if x < 3 else 0)

df["mule_indicator"] = df.groupby(
    ["step", "oldbalanceDest"]
)["amount"].transform("count")

df["type"] = df["type"].astype("category").cat.codes

df = df.drop("isFraud", axis=1)

engine = FraudEngine()

result = engine.predict(df.iloc[[0]])

print("Prediction Output:")
print(result)