"""
Generates a synthetic credit card fraud dataset (Kaggle-compatible format).
Run this once to create data/transactions.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N_LEGIT = 9000
N_FRAUD = 500

def make_legit():
    rows = []
    for i in range(N_LEGIT):
        v = np.random.randn(28)
        amount = np.abs(np.random.exponential(80))
        time = np.random.uniform(0, 172800)
        rows.append([time] + list(v) + [round(amount, 2), 0])
    return rows

def make_fraud():
    rows = []
    for i in range(N_FRAUD):
        v = np.random.randn(28)
        v[0] -= 3.5
        v[3] += 2.0
        v[9] -= 2.5
        amount = np.abs(np.random.exponential(300)) + 100
        time = np.random.uniform(0, 172800)
        rows.append([time] + list(v) + [round(amount, 2), 1])
    return rows

cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
data = make_legit() + make_fraud()
df = pd.DataFrame(data, columns=cols).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/transactions.csv", index=False)
print(f"Dataset created: {len(df)} rows ({N_FRAUD} fraud, {N_LEGIT} legit)")
