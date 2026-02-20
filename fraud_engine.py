import joblib
import numpy as np


class FraudEngine:

    def __init__(self):
        print("🔐 Loading production models...")

        self.xgb = joblib.load("xgb_model.pkl")
        self.lgb = joblib.load("lgb_model.pkl")
        self.feature_order = joblib.load("feature_order.pkl")

        self.THRESHOLD = 0.55

        print("✅ Models loaded successfully.")

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

        # Extract contextual features BEFORE filtering
        mule_val = data.get("mule_indicator", [0])[0]
        rare_val = data.get("rare_receiver", [0])[0]
        amount = data.get("amount", [0])[0]
        balance = data.get("oldbalanceOrg", [1])[0]
        overlay_flag = data.get("overlay_flag", [0])[0]

        drain_ratio = amount / (balance + 1)

        model_data = data[self.feature_order]

        xgb_prob = self.xgb.predict_proba(model_data)[:, 1]
        lgb_prob = self.lgb.predict_proba(model_data)[:, 1]

        final_prob = (0.6 * xgb_prob) + (0.4 * lgb_prob)

        probability = float(final_prob[0])
        risk_score = probability * 100

        if overlay_flag == 1:
            return {
                "probability": round(probability, 4),
                "risk_score": round(risk_score, 2),
                "risk_bucket": "CRITICAL DEVICE RISK",
                "decision": "BLOCK_DUE_TO_OVERLAY",
                "mule_detected": False
            }

        bucket = self.risk_bucket(risk_score)
        decision = self.decision_engine(
            risk_score, rare_val, mule_val, drain_ratio
        )

        return {
            "probability": round(probability, 4),
            "risk_score": round(risk_score, 2),
            "risk_bucket": bucket,
            "decision": decision,
            "mule_detected": True if mule_val > 5 else False
        }

    def batch_predict(self, data):

        model_data = data[self.feature_order]

        xgb_prob = self.xgb.predict_proba(model_data)[:, 1]
        lgb_prob = self.lgb.predict_proba(model_data)[:, 1]

        final_prob = (0.6 * xgb_prob) + (0.4 * lgb_prob)

        return final_prob