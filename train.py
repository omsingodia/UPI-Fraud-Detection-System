import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

print("🚀 Loading 10 lakh rows...")

df = pd.read_csv(
    "../dataset/PS_20174392719_1491204439457_log.csv",
    nrows=1000000
)

df.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

# -----------------------------
# 🔧 Advanced Feature Engineering
# -----------------------------

df["relative_amount"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

df["balance_change_ratio"] = (
    df["newbalanceOrig"] - df["oldbalanceOrg"]
) / (df["oldbalanceOrg"] + 1)

df["velocity"] = df.groupby("step")["amount"].transform("count")

receiver_counts = df["oldbalanceDest"].value_counts()
df["receiver_risk"] = df["oldbalanceDest"].map(receiver_counts)
df["rare_receiver"] = df["receiver_risk"].apply(lambda x: 1 if x < 3 else 0)

# Encode
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

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
# 🔥 XGBoost
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
# 🔥 LightGBM
# -----------------------------

lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

lgb.fit(X_train, y_train)

# -----------------------------
# 🔥 Ensemble
# -----------------------------

xgb_prob = xgb.predict_proba(X_test)[:, 1]
lgb_prob = lgb.predict_proba(X_test)[:, 1]

final_prob = 0.6 * xgb_prob + 0.4 * lgb_prob

print("\n=== Classification Report (Ensemble) ===")
y_pred = (final_prob >= 0.7).astype(int)
print(classification_report(y_test, y_pred))

print("Final ROC-AUC:", roc_auc_score(y_test, final_prob))

# -----------------------------
# 🔥 Precision-Recall Curve
# -----------------------------

precision, recall, _ = precision_recall_curve(y_test, final_prob)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("precision_recall_curve.png")
print("PR-AUC:", pr_auc)

# -----------------------------
# 🔥 Calibration Curve
# -----------------------------

prob_true, prob_pred = calibration_curve(y_test, final_prob, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.savefig("calibration_curve.png")
print("Calibration curve saved.")

# -----------------------------
# 🔥 SHAP Explainability
# -----------------------------

print("Generating SHAP summary...")

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test[:1000])

shap.summary_plot(shap_values, X_test[:1000], show=False)
plt.savefig("shap_summary.png")

print("SHAP summary saved.")

# -----------------------------
# Save models
# -----------------------------

joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(lgb, "lgb_model.pkl")

print("\n✅ ULTRA HARD MODEL COMPLETE")
