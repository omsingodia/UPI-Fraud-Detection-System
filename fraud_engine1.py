import joblib
import numpy as np
import os


class FraudEngine:
    """
    XGBoost + LightGBM ensemble fraud engine.
    Models are loaded from the same directory as this file.
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        print("[*] Loading production models...")
        self.xgb = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
        self.lgb = joblib.load(os.path.join(model_dir, "lgb_model.pkl"))
        self.feature_order = joblib.load(os.path.join(model_dir, "feature_order.pkl"))
        self.THRESHOLD = 0.55
        print(f"[OK] Models loaded. Feature order: {self.feature_order}")

    def risk_bucket(self, score):
        if score >= 85:
            return "HIGH RISK"
        elif score >= 55:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"

    def decision_engine(self, score, rare_receiver, mule_count, drain_ratio):
        if drain_ratio > 0.8 and rare_receiver == 1:
            return "HIGH_DRAIN_RISK_VERIFICATION"
        if score >= 85 or mule_count > 10:
            return "BLOCK_AND_COOLDOWN"
        if score >= 70:
            return "INTERACTIVE_SECURITY_QUIZ"
        if score >= 55 or rare_receiver == 1:
            return "MANDATORY_BIOMETRIC_VERIFICATION"
        return "ALLOW"

    def predict(self, data):
        """
        data: pandas DataFrame with a single row, feature-engineered.
        """
        mule_val = float(data.get("mule_indicator", [1])[0]) if hasattr(data, "get") else float(data["mule_indicator"].iloc[0])
        rare_val = float(data.get("rare_receiver", [0])[0]) if hasattr(data, "get") else float(data["rare_receiver"].iloc[0])
        amount   = float(data.get("amount", [0])[0])        if hasattr(data, "get") else float(data["amount"].iloc[0])
        balance  = float(data.get("oldbalanceOrg", [1])[0]) if hasattr(data, "get") else float(data["oldbalanceOrg"].iloc[0])

        drain_ratio = amount / (balance + 1)

        # Select and order features
        model_data = data[self.feature_order]

        xgb_prob = self.xgb.predict_proba(model_data)[:, 1]
        lgb_prob = self.lgb.predict_proba(model_data)[:, 1]

        final_prob = (0.6 * xgb_prob) + (0.4 * lgb_prob)
        probability = float(final_prob[0])
        risk_score = round(probability * 100, 2)

        bucket = self.risk_bucket(risk_score)
        decision = self.decision_engine(risk_score, rare_val, mule_val, drain_ratio)

        return {
            "probability": round(probability, 4),
            "risk_score": risk_score,
            "risk_bucket": bucket,
            "decision": decision,
            "mule_detected": True if mule_val > 5 else False,
            "drain_ratio": round(drain_ratio, 4),
            "rare_receiver": int(rare_val),
        }
