import pandas as pd
import time
from fraud_engine import FraudEngine

print("🚀 Starting Pro-Secure UPI Fraud Engine...")

# Load small chunk for testing
df = pd.read_csv("../dataset/PS_20174392719_1491204439457_log.csv", nrows=5000)

# -----------------------------
# ADVANCED FEATURE ENGINEERING
# -----------------------------

# A. Transaction Flow Features
df["relative_amount"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balance_change_ratio"] = (df["newbalanceOrig"] - df["oldbalanceOrg"]) / (df["oldbalanceOrg"] + 1)

# B. Graph-Based Mule Detection Logic
# Count transactions to same destination in the same time step
df["mule_indicator"] = df.groupby(["step", "nameDest"])["amount"].transform("count")

# C. Receiver Profile
receiver_counts = df["nameDest"].value_counts()
df["receiver_risk"] = df["nameDest"].map(receiver_counts)
df["rare_receiver"] = df["receiver_risk"].apply(lambda x: 1 if x < 2 else 0)

# D. Encoding & Cleaning
df["type_code"] = df["type"].astype("category").cat.codes
df_final = df.drop(["nameOrig", "nameDest", "type", "isFraud", "isFlaggedFraud"], axis=1, errors='ignore')

# -----------------------------
# EXECUTION
# -----------------------------

engine = FraudEngine()
results_summary = {"HIGH RISK": 0, "MEDIUM RISK": 0, "LOW RISK": 0}

print(f"⚡ Testing on 200 samples with Adaptive Friction...")

for i in range(200):
    row = df_final.iloc[[i]]
    res = engine.predict(row)
    results_summary[res["risk_bucket"]] += 1
    
    # Print only suspicious ones for demo
    if res["risk_score"] > 50:
        print(f"Txn {i}: Score: {res['risk_score']} | Decision: {res['decision']}")

print("\n📊 Final Risk Distribution:", results_summary)
print("✅ Test Complete.")