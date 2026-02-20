import pandas as pd
import time
from fraud_engine import FraudEngine
from feature_pipeline import apply_features

print("🚀 Starting Production Stress Test (Optimized)...")

df = pd.read_csv(
    "../dataset/PS_20174392719_1491204439457_log.csv",
    nrows=1000
)

df = apply_features(df)

engine = FraudEngine()

start = time.time()

probs = engine.batch_predict(df)

end = time.time()

print("\n⚡ Performance Metrics:")
print("Total Time:", round(end - start, 4))
print("TPS:", round(1000 / (end - start), 2))

print("\n✅ Batch Test Complete")