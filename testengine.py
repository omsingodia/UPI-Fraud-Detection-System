import pandas as pd
from fraud_engine import FraudEngine

# Load one sample row from dataset
df = pd.read_csv("../dataset/PS_20174392719_1491204439457_log.csv", nrows=5)

# Apply SAME feature engineering as training
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

df["type"] = df["type"].astype("category").cat.codes

# Drop target column
X = df.drop("isFraud", axis=1)

# Initialize engine
engine = FraudEngine()

# Predict first row
fraud_row = df[df["isFraud"] == 1].iloc[[0]]
X_fraud = fraud_row.drop("isFraud", axis=1)

result = engine.predict(X_fraud)

print("\nFraud Engine Output:")
print(result)