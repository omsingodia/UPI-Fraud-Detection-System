import pandas as pd
import time
from fraud_engine import FraudEngine

print("🚀 Starting Production Stress Test (Optimized)...")

# Load 5000 rows
df = pd.read_csv(
    "../dataset/PS_20174392719_1491204439457_log.csv",
    nrows=5000
)

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
df["rare_receiver"] = df["receiver_risk"].apply(lambda x: 1 if x < 3 else 0)

df["mule_indicator"] = df.groupby(
    ["step", "oldbalanceDest"]
)["amount"].transform("count")

df["type"] = df["type"].astype("category").cat.codes

df = df.drop("isFraud", axis=1)

# -----------------------------
# RUN OPTIMIZED STRESS TEST
# -----------------------------

engine = FraudEngine()

print("⚡ Running batch prediction on 1000 transactions...")

batch_data = df.iloc[:1000]

start = time.time()

# 🔥 FAST BATCH INFERENCE
probs = engine.batch_predict(batch_data)

end = time.time()

total_time = end - start

# Risk classification for stats
high = (probs * 100 >= 85).sum()
medium = ((probs * 100 >= 55) & (probs * 100 < 85)).sum()
low = (probs * 100 < 55).sum()

print("\n📊 Risk Distribution:")
print("HIGH RISK:", high)
print("MEDIUM RISK:", medium)
print("LOW RISK:", low)

print("\n⚡ Performance Metrics:")
print("Total Time:", round(total_time, 4), "seconds")
print("Per Transaction Latency:", round(total_time / 1000, 6), "seconds")
print("Transactions Per Second (TPS):", round(1000 / total_time, 2))

print("\n✅ Optimized Stress Test Complete.")