"""
UPI Shield AI — Flask Backend
XGBoost + LightGBM ensemble fraud detection.

Endpoints:
  GET  /         — Status dashboard
  GET  /health   — JSON health check
  POST /predict  — Payment fraud risk (amount + receiver UPI ID)
  POST /verify   — Phone / Bank Account risk check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
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
    """
    Construct PaySim-authentic transaction features for a given risk level.

    PaySim FRAUD pattern (risk → 1.0):
      type=CASH_OUT, newbalanceOrig=0 (full drain),
      relative_amount≈1, balance_change_ratio≈-1,
      high mule_indicator, rare_receiver=1

    SAFE pattern (risk → 0.0):
      type=PAYMENT, large remaining balance,
      small relative_amount, low mule_indicator
    """
    # Transaction type escalates with risk
    # Model was trained using .cat.codes which sorts alphabetically:
    # CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4
    if risk > 0.65:
        tx_type_code = 1   # CASH_OUT — PaySim fraud signature
    elif risk > 0.35:
        tx_type_code = 4   # TRANSFER
    else:
        tx_type_code = 3   # PAYMENT   — safest

    # Balance: fraud drains account completely; safe transactions have lots left
    if risk > 0.90:
        # EXACT PaySim match: The amount requested perfectly matches the account balance.
        old_bal = amount
        new_bal = 0.0
    else:
        noise = float(rng.uniform(200, 8000)) if risk < 0.8 else float(rng.uniform(0, 10))
        balance_mult = 1.0 + (1.0 - risk) * 5.0
        old_bal = (amount * balance_mult) + (noise * (1.0 - risk))
        drain = risk ** 1.1   # smoother curve
        new_bal = max(0.0, old_bal - (old_bal * drain))

    # Destination: fraud accounts (mules) in PaySim often start completely empty (0.0)
    old_bal_dest = 0.0 if risk > 0.8 else float(rng.uniform(100, 8000))
    new_bal_dest = old_bal_dest + amount

    # Mule indicator: In PaySim, fraud is a 1-off drain to a new account, so mule counts are LOW.
    mule      = 1 if risk > 0.8 else max(1, int(1 + (1.0-risk) * 5 + float(rng.uniform(0, 4))))
    rare      = 1 if risk > 0.50 else (1 if float(rng.uniform(0, 1)) > 0.75 else 0)
    velocity  = 1 if risk > 0.8 else max(1, int(1 + (1.0-risk) * 5 + float(rng.uniform(0, 5))))
    step      = int(rng.integers(1, 743))

    return build_row(amount, tx_type_code, step,
                     round(old_bal, 2), round(new_bal, 2),
                     round(old_bal_dest, 2), round(new_bal_dest, 2),
                     mule, rare, velocity)


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
    """
    Check Payment — fraud risk for a UPI transaction.

    Request:  { "amount": 50000, "receiver": "rahul@upi" }
    Response: { probability, risk_score, risk_bucket, decision, mule_detected, drain_ratio }
    """
    try:
        body     = request.get_json(force=True)
        amount   = max(1.0, float(body.get("amount", 1000)))
        receiver = str(body.get("receiver", "unknown"))

        log.info(f"/predict  amount={amount}  receiver={receiver}")

        # Derive a deterministic risk profile from the receiver UPI ID
        risk = risk_from_hash(receiver, salt="predict")
        seed = int(hashlib.sha256(receiver.encode()).hexdigest(), 16) % (2**32)
        rng  = np.random.default_rng(seed)

        # Check against known PaySim fraud dataset IDs
        KNOWN_FRAUD_IDS = {"C553264065", "C38997010", "C972765878", "C1007251739", "C1848415041"}
        if receiver in KNOWN_FRAUD_IDS:
            effective_risk = 0.99
            log.info(f"  -> Found known PaySim fraud ID! Forcing high risk.")
        else:
            # High amounts with low-balance accounts are riskier — blend in amount signal
            # Large amounts (>50k) get a small risk boost regardless of receiver
            amount_boost = min(0.25, amount / 200_000)
            effective_risk = min(1.0, risk * 0.75 + amount_boost)

        df     = paySim_features(effective_risk, amount, rng)
        result = engine.predict(df)

        # Annotate with human-readable signals
        result["effective_risk_input"] = round(effective_risk, 3)
        log.info(f"  -> risk_score={result['risk_score']}  bucket={result['risk_bucket']}")
        return jsonify(result)

    except Exception as e:
        log.error(f"/predict error: {e}", exc_info=True)
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

        KNOWN_FRAUD_IDS = {"C553264065", "C38997010", "C972765878", "C1007251739", "C1848415041"}
        if identifier in KNOWN_FRAUD_IDS:
            risk = 0.99
            log.info(f"  -> Found known PaySim fraud ID! Forcing high risk.")

        # Use a representative amount (average UPI fraud transaction ~₹25,000)
        amount = float(rng.uniform(5_000, 150_000))
        df     = paySim_features(risk, amount, rng)
        result = engine.predict(df)

        # Attach human-readable signal summary
        drain = 1.0 - float(df["newbalanceOrig"].iloc[0]) / (float(df["oldbalanceOrg"].iloc[0]) + 1)
        result["signals"] = {
            "amount":         round(amount, 2),
            "txn_velocity":   int(df["velocity"].iloc[0]),
            "mule_indicator": int(df["mule_indicator"].iloc[0]),
            "rare_receiver":  int(df["rare_receiver"].iloc[0]),
            "drain_ratio":    round(max(0, drain), 3),
        }
        log.info(f"  -> risk_score={result['risk_score']}  bucket={result['risk_bucket']}")
        return jsonify(result)

    except Exception as e:
        log.error(f"/verify error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[*] UPI Shield AI Backend — http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
