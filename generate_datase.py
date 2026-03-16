import pandas as pd
import numpy as np

def generate_synthetic_paysim(num_rows=50000, fraud_rate=0.03):
    print(f"[*] Generating {num_rows} synthetic transactions (Fraud rate: {fraud_rate * 100}%)...")
    np.random.seed(42)
    
    num_frauds = int(num_rows * fraud_rate)
    num_legit = num_rows - num_frauds
    
    # ── LEGIT TRANSACTIONS ──
    # Mostly PAYMENT (0), CASH_IN (3), CASH_OUT (2), TRANSFER (4), DEBIT (1)
    # Legit types mapped to PaySim standard
    legit_types = np.random.choice([0, 1, 2, 3, 4], size=num_legit, p=[0.4, 0.05, 0.25, 0.2, 0.1])
    legit_amounts = np.abs(np.random.normal(5000, 15000, size=num_legit))
    
    # Legit balances: plenty left over
    legit_old_bal_orig = legit_amounts + np.abs(np.random.normal(50000, 200000, size=num_legit))
    legit_new_bal_orig = legit_old_bal_orig - legit_amounts
    
    legit_old_bal_dest = np.abs(np.random.normal(10000, 100000, size=num_legit))
    legit_new_bal_dest = legit_old_bal_dest + legit_amounts
    
    legit_df = pd.DataFrame({
        "step": np.random.randint(1, 743, size=num_legit),
        "type": legit_types,
        "amount": legit_amounts,
        "oldbalanceOrg": legit_old_bal_orig,
        "newbalanceOrig": legit_new_bal_orig,
        "oldbalanceDest": legit_old_bal_dest,
        "newbalanceDest": legit_new_bal_dest,
        "isFraud": 0
    })
    
    # ── FRAUD TRANSACTIONS ──
    # Fraud mostly TRANSFER (4) and CASH_OUT (2)
    fraud_types = np.random.choice([2, 4], size=num_frauds, p=[0.5, 0.5])
    fraud_amounts = np.abs(np.random.normal(150000, 800000, size=num_frauds))
    
    # Fraud signatures: account completely drained
    fraud_old_bal_orig = fraud_amounts 
    fraud_new_bal_orig = np.zeros(num_frauds)
    
    # Fraud destinations: often empty
    fraud_old_bal_dest = np.zeros(num_frauds) 
    fraud_new_bal_dest = fraud_amounts
    
    fraud_df = pd.DataFrame({
        "step": np.random.randint(1, 743, size=num_frauds),
        "type": fraud_types,
        "amount": fraud_amounts,
        "oldbalanceOrg": fraud_old_bal_orig,
        "newbalanceOrig": fraud_new_bal_orig,
        "oldbalanceDest": fraud_old_bal_dest,
        "newbalanceDest": fraud_new_bal_dest,
        "isFraud": 1
    })
    
    # Combine and shuffle
    df = pd.concat([legit_df, fraud_df]).sample(frac=1).reset_index(drop=True)
    
    # Apply standard PaySim Feature Engineering
    print("[*] Applying feature engineering...")
    df["relative_amount"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_change_ratio"] = (df["newbalanceOrig"] - df["oldbalanceOrg"]) / (df["oldbalanceOrg"] + 1)
    
    # Mules and Velocity (synthetic simplified)
    df["velocity"] = np.where(df["isFraud"] == 1, 1, np.random.randint(1, 5, size=num_rows))
    df["receiver_risk"] = np.where(df["isFraud"] == 1, 1, np.random.randint(5, 50, size=num_rows))
    df["rare_receiver"] = np.where(df["receiver_risk"] < 3, 1, 0)
    df["mule_indicator"] = np.where(df["isFraud"] == 1, 1, np.random.randint(2, 10, size=num_rows))
    
    print("[*] Saving dataset to synthetic_paysim.csv...")
    df.to_csv("synthetic_paysim.csv", index=False)
    print(df["isFraud"].value_counts())
    print("[OK] Dataset generated successfully.")

if __name__ == "__main__":
    generate_synthetic_paysim()
