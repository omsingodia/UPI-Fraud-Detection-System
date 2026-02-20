import numpy as np
import joblib

# Load trained models
# Note: Agar mule_indicator train nahi kiya hai, toh ye engine logic level pe handle karega
try:
    xgb = joblib.load("xgb_model.pkl")
    lgb = joblib.load("lgb_model.pkl")
except:
    print("⚠️ Warning: Model files not found. Ensure pkl files are in the same directory.")

def risk_bucket(score):
    if score >= 75:
        return "HIGH RISK"
    elif score >= 50:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

def decision_engine(score, rare_receiver, mule_count):
    """
    HACKATHON SPECIAL: Adaptive Friction Logic
    """
    # 1. MULE DETECTION / CRITICAL RISK (Red Zone)
    # Agar model score high hai ya receiver account ek hotspot (mule) hai
    if score >= 85 or mule_count > 10: 
        return "BLOCK_AND_COOLDOWN (30 Min Hold)"
    
    # 2. SOCIAL ENGINEERING DEFENSE (Orange Zone)
    elif score >= 70:
        return "INTERACTIVE_SECURITY_QUIZ"
        
    # 3. ADAPTIVE AUTHENTICATION (Yellow Zone)
    elif score >= 50 or rare_receiver == 1:
        return "MANDATORY_BIOMETRIC_VERIFICATION"
        
    # 4. SAFE ZONE (Green)
    else:
        return "ALLOW"

class FraudEngine:
    def __init__(self):
        self.xgb = xgb
        self.lgb = lgb

    def predict(self, data):
        # Ensembling: 60% XGBoost + 40% LightGBM
        xgb_prob = self.xgb.predict_proba(data)[:, 1]
        lgb_prob = self.lgb.predict_proba(data)[:, 1]

        final_prob = (0.6 * xgb_prob) + (0.4 * lgb_prob)
        risk_score = final_prob * 100

        # Extracting feature values for decision engine
        is_rare = data["rare_receiver"].values[0]
        mule_val = data["mule_indicator"].values[0]

        bucket = risk_bucket(risk_score[0])
        decision = decision_engine(risk_score[0], is_rare, mule_val)

        return {
            "probability": round(float(final_prob[0]), 4),
            "risk_score": round(float(risk_score[0]), 2),
            "risk_bucket": bucket,
            "decision": decision,
            "mule_detected": True if mule_val > 5 else False
        }