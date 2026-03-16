"""
UPI Shield AI — Flask Backend
XGBoost + LightGBM ensemble fraud detection.

Endpoints:
  GET  /         — Status dashboard
  GET  /health   — JSON health check
  POST /predict  — Payment fraud risk (amount + receiver UPI ID)
  POST /verify  — Phone / Bank Account risk check
"""

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import hashlib
import logging
import os

from fraud_engine import FraudEngine

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["*"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Load models once at startup
engine = FraudEngine()

# ── Feature engineering helpers ────────────────────────────────────────────────

def risk_from_hash(identifier: str, salt: str = "") -> float:
    """Deterministic risk score 0.0–1.0 from identifier string."""
    h = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def build_row(amount, tx_type_code, step, old_bal, new_bal,
              old_bal_dest, new_bal_dest, mule, rare, velocity):
    """
    Build a feature-engineered DataFrame row that the model can score.
    Matches the feature_pipeline.py logic used during training.
    """
    df = pd.DataFrame([{
        "step":            step,
        "type":            tx_type_code,
        "amount":          amount,
        "oldbalanceOrg":   old_bal,
        "newbalanceOrig":  new_bal,
        "oldbalanceDest":  old_bal_dest,
        "newbalanceDest":  new_bal_dest,
        "isFlaggedFraud":  0,
    }])
    df["relative_amount"]     = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["balance_diff"]        = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_change_ratio"] = (df["newbalanceOrig"] - df["oldbalanceOrg"]) / (df["oldbalanceOrg"] + 1)
    df["velocity"]            = velocity
    df["receiver_risk"]       = 1
    df["rare_receiver"]       = rare
    df["mule_indicator"]      = mule
    return df


def paySim_features(risk: float, amount: float, rng):

    # -------------------------------
    # 1️⃣ Transaction Type
    # -------------------------------
    if risk > 0.7:
        tx_type_code = 1   # CASH_OUT (fraud signature)
    elif risk > 0.4:
        tx_type_code = 4   # TRANSFER
    else:
        tx_type_code = 3   # PAYMENT

    # -------------------------------
    # 2️⃣ Fraud Balance Simulation
    # -------------------------------
    if risk > 0.75:
        # TRUE PaySim fraud pattern
        old_bal = amount
        new_bal = 0.0
    else:
        old_bal = amount * rng.uniform(2, 6)
        new_bal = old_bal - amount

    # -------------------------------
    # 3️⃣ Destination Pattern
    # -------------------------------
    if risk > 0.75:
        old_bal_dest = 0.0
    else:
        old_bal_dest = float(rng.uniform(500, 10000))

    new_bal_dest = old_bal_dest + amount

    # -------------------------------
    # 4️⃣ Fraud Signals
    # -------------------------------
    mule = 1 if risk > 0.75 else int(rng.integers(1, 6))
    rare = 1 if risk > 0.6 else 0
    velocity = 1 if risk > 0.75 else int(rng.integers(1, 6))

    step = int(rng.integers(1, 743))

    return build_row(
        amount,
        tx_type_code,
        step,
        round(old_bal, 2),
        round(new_bal, 2),
        round(old_bal_dest, 2),
        round(new_bal_dest, 2),
        mule,
        rare,
        velocity
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return """<!DOCTYPE html>
<html>
<head>
  <title>UPI Shield AI — Backend</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:#04080f; color:#e0e0e0; font-family:'Segoe UI',sans-serif;
           display:flex; align-items:center; justify-content:center; min-height:100vh; }
    .card { background:rgba(255,255,255,0.04); border:1px solid rgba(0,212,255,0.2);
            border-radius:20px; padding:40px 48px; max-width:520px; width:100%; text-align:center; }
    .logo { font-size:36px; margin-bottom:16px; }
    h1 { font-size:22px; font-weight:800; color:#fff; }
    .badge { display:inline-block; background:rgba(0,255,136,0.12); color:#00ff88;
             border:1px solid rgba(0,255,136,0.3); border-radius:20px;
             padding:4px 14px; font-size:12px; font-weight:700; margin:12px 0 28px; }
    table { width:100%; border-collapse:collapse; text-align:left; }
    th { font-size:11px; color:rgba(255,255,255,0.3); text-transform:uppercase;
         letter-spacing:1px; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.08); }
    td { padding:10px 0; font-size:13px; border-bottom:1px solid rgba(255,255,255,0.04); }
    .method { background:rgba(0,212,255,0.1); color:#00d4ff; border-radius:5px;
              padding:2px 8px; font-size:11px; font-weight:700; font-family:monospace; }
    .post { background:rgba(120,80,255,0.12); color:#9b59ff; }
    .path { font-family:monospace; color:#fff; }
    .desc { color:rgba(255,255,255,0.4); }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">&#9889;</div>
    <h1>UPI Shield AI Backend</h1>
    <div class="badge">&#9679; ONLINE &mdash; XGB + LGB Ensemble v1.0</div>
    <table>
      <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
      <tr><td><span class="method">GET</span></td><td class="path">/health</td><td class="desc">Server status</td></tr>
      <tr><td><span class="method post">POST</span></td><td class="path">/predict</td><td class="desc">Payment fraud risk</td></tr>
      <tr><td><span class="method post">POST</span></td><td class="path">/verify</td><td class="desc">Phone / bank check</td></tr>
    </table>
  </div>
</body>
</html>"""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "XGB+LGB ensemble v1.0"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)

        amount = float(body.get("amount", 0))
        receiver = str(body.get("receiver", ""))

        if amount <= 0:
            return jsonify({"error": "Invalid amount"}), 400

        # Get explicit balance and time variables from Frontend
        sender_balance = float(body.get("sender_balance", 0))
        receiver_balance = float(body.get("receiver_balance", 0))
        time_hour = int(body.get("time", 12))

        # Calculate Post-Transaction Balances
        newbalanceOrig = max(0, sender_balance - amount)
        newbalanceDest = receiver_balance + amount

        df = pd.DataFrame([{
            "step": time_hour,  # Time of transaction
            "type": 4,          # TRANSFER
            "amount": amount,
            "oldbalanceOrg": sender_balance,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": receiver_balance,
            "newbalanceDest": newbalanceDest,
            "isFlaggedFraud": 0
        }])

        # Feature engineering (same as training)
        df["relative_amount"] = amount / (sender_balance + 1)
        df["balance_diff"] = sender_balance - newbalanceOrig
        df["balance_change_ratio"] = (
            newbalanceOrig - sender_balance
        ) / (sender_balance + 1)

        df["velocity"] = 1
        df["receiver_risk"] = 1
        df["rare_receiver"] = 0
        df["mule_indicator"] = 1

        result = engine.predict(df)
        
        # ─────────────────────────────
        # 🔥 Strong Final Risk Logic
        # ─────────────────────────────
        
        ml_score = result["risk_score"]
        
        # NaN Guard — if ML model returned NaN default to behavioural-only
        import math
        if math.isnan(float(ml_score)):
            ml_score = 50  # neutral baseline
        
        # Define missing variables for the logic
        drain_ratio = float(df["relative_amount"].iloc[0])
        receiver_flag = True if amount > 50_000 else False
        risk_boost = (drain_ratio * 15) if amount > 20000 else 0

        # Late Night Rule (10 PM to 4 AM)
        if time_hour >= 22 or time_hour <= 4:
            risk_boost += 20

        # Receiver has 0 balance = mule account flag
        if receiver_balance == 0:
            risk_boost += 15

        # Combine ML + Behavioral Risk
        final_score = min(100, max(0, int(
            (ml_score * 0.7) + (risk_boost * 1.2)
        )))

        # Safety floor — high drain must never be low risk
        if drain_ratio > 0.9:
            final_score = max(final_score, 85)

        if receiver_flag and amount > 100000:
            final_score = max(final_score, 75)

        # Decision Buckets
        if final_score >= 80:
            bucket = "HIGH RISK"
            decision = "BLOCK_TRANSACTION"
        elif final_score >= 50:
            bucket = "MEDIUM RISK"
            decision = "OTP_VERIFICATION"
        else:
            bucket = "LOW RISK"
            decision = "ALLOW"
            
        result["risk_score"] = final_score
        result["risk_bucket"] = bucket
        result["decision"] = decision

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/verify", methods=["POST"])
def verify():
    """
    Phone / Bank Account risk check.

    Request:  { "identifier": "9876543210", "type": "phone" }
              { "identifier": "123456789012", "type": "bank", "ifsc": "HDFC0001234" }
    Response: { probability, risk_score, risk_bucket, decision, signals }
    """
    try:
        body       = request.get_json(force=True)
        identifier = str(body.get("identifier", ""))
        id_type    = str(body.get("type", "phone"))
        ifsc       = str(body.get("ifsc", ""))

        if id_type == "bank" and ifsc:
            identifier = f"{identifier}:{ifsc}"

        log.info(f"/verify  type={id_type}  id={identifier}")

        risk = risk_from_hash(identifier, salt=id_type)
        seed = int(hashlib.sha256(f"{id_type}:{identifier}".encode()).hexdigest(), 16) % (2**32)
        rng  = np.random.default_rng(seed)

        # No hardcoded overrides. Risk is purely deterministic based on the identifier hash.

        # Use a representative amount (average UPI fraud transaction ~₹25,000)
        amount = float(rng.uniform(5_000, 150_000))

        df = paySim_features(risk, amount, rng)
        raw_result = engine.predict(df)

        # Safe extraction (avoid KeyError)
        probability  = float(raw_result.get("probability", 0.0))
        risk_score   = int(raw_result.get("risk_score", 0))
        risk_bucket  = raw_result.get("risk_bucket", "UNKNOWN")
        decision     = raw_result.get("decision", "REVIEW")
        mule_detected = bool(raw_result.get("mule_detected", False))

        # Drain ratio calculation
        drain = 1.0 - float(df["newbalanceOrig"].iloc[0]) / (
            float(df["oldbalanceOrg"].iloc[0]) + 1
        )

        response = {
            "probability": probability,
            "risk_score": risk_score,
            "risk_bucket": risk_bucket,
            "decision": decision,
            "mule_detected": mule_detected,
            "signals": {
                "amount": round(amount, 2),
                "txn_velocity": int(df["velocity"].iloc[0]),
                "mule_indicator": int(df["mule_indicator"].iloc[0]),
                "rare_receiver": int(df["rare_receiver"].iloc[0]),
                "drain_ratio": round(max(0, drain), 3),
            }
        }

        log.info(f" -> risk_score={risk_score}  bucket={risk_bucket}")

        return jsonify(response)

    except Exception as e:
        log.error(f"/verify error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[*] UPI Shield AI Backend — http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
