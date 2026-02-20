import pandas as pd
from fraud_engine import FraudEngine

print("🚀 Loading dataset for fraud stress test...")

df = pd.read_csv("../dataset/PS_20174392719_1491204439457_log.csv")

# -----------------------------
# SAME FEATURE ENGINEERING
# -----------------------------

df.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

df["relative_amount"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

df["balance_change_ratio"] = (
    df["newbalanceOrig"] - df["oldbalanceOrg"]
) / (df["oldbalanceOrg"] + 1)

df["velocity"] = df.groupby("step")["amount"].transform("count")

receiver_counts = df["oldbalanceDest"].value_counts()
df["receiver_risk"] = df["oldbalanceDest"].map(receiver_counts)

df["rare_receiver"] = df["receiver_risk"].apply(
    lambda x: 1 if x < 3 else 0
)

df["type"] = df["type"].astype("category").cat.codes

fraud_df = df[df["isFraud"] == 1].head(50)

fraud_df = fraud_df.drop("isFraud", axis=1)

engine = FraudEngine()

high = 0
medium = 0

print("⚡ Running fraud-only predictions...")

for i in range(len(fraud_df)):
    row = fraud_df.iloc[[i]]
    result = engine.predict(row)

    if result["risk_bucket"] == "HIGH RISK":
        high += 1
    elif result["risk_bucket"] == "MEDIUM RISK":
        medium += 1

print("\n📊 Fraud Detection Power:")
print("Fraud Cases Tested:", len(fraud_df))
print("High Risk Flagged:", high)
print("Medium Risk Flagged:", medium)
print("Total Flagged:", high + medium)

print("\n✅ Fraud stress test complete.")