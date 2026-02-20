import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from feature_pipeline import apply_features

print("🚀 Loading 10 lakh rows...")

df = pd.read_csv(
    "../dataset/PS_20174392719_1491204439457_log.csv",
    nrows=1000000
)

# Apply features
df = apply_features(df, training=True)

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# -----------------------------
# XGBoost
# -----------------------------

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

# -----------------------------
# LightGBM
# -----------------------------

lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

lgb.fit(X_train, y_train)

# -----------------------------
# Ensemble
# -----------------------------

xgb_prob = xgb.predict_proba(X_test)[:, 1]
lgb_prob = lgb.predict_proba(X_test)[:, 1]

final_prob = 0.6 * xgb_prob + 0.4 * lgb_prob

THRESHOLD = 0.55
y_pred = (final_prob >= THRESHOLD).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, final_prob))

# -----------------------------
# Save models + feature order
# -----------------------------

joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(lgb, "lgb_model.pkl")
joblib.dump(X.columns.tolist(), "feature_order.pkl")

print("\n✅ PRODUCTION-GRADE MODEL COMPLETE")