import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_models():
    print("[*] Loading synthetic dataset...")
    df = pd.read_csv("synthetic_paysim.csv")
    
    # Define features and target
    target = "isFraud"
    features = [
        "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", 
        "oldbalanceDest", "newbalanceDest", "relative_amount", 
        "balance_diff", "balance_change_ratio", "velocity", 
        "receiver_risk", "rare_receiver", "mule_indicator"
    ]
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[*] Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # XGBoost
    print("[*] Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='auc'
    )
    xgb_model.fit(X_train, y_train)
    
    # LightGBM
    print("[*] Training LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    
    # Evaluation
    print("\n[*] --- Model Evaluation ---")
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    print("XGBoost AUC:", roc_auc_score(y_test, xgb_probs))
    
    lgb_preds = lgb_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    print("LightGBM AUC:", roc_auc_score(y_test, lgb_probs))
    
    # Ensemble eval
    ensemble_probs = (xgb_probs * 0.6) + (lgb_probs * 0.4)
    print("Ensemble AUC:", roc_auc_score(y_test, ensemble_probs))
    
    print("\n[*] Saving models to d:\\project\\upi-fraud-backend\\...")
    joblib.dump(xgb_model, "..\\upi-fraud-backend\\xgb_model.pkl")
    joblib.dump(lgb_model, "..\\upi-fraud-backend\\lgb_model.pkl")
    joblib.dump(features, "..\\upi-fraud-backend\\feature_order.pkl")
    
    print("[OK] Models trained and saved successfully.")

if __name__ == "__main__":
    train_models()
