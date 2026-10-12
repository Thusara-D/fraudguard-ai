
from pathlib import Path
import pandas as pd

# Find our project folder
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"

datasets = ["train", "validation", "test"]

results = {}

# Check every dataset
for name in datasets:

    file_path = DATA_DIR / f"{name}.csv"

    rows = 0
    frauds = 0
    min_step = float("inf")
    max_step = float("-inf")

    # Read in smaller chunks to save memory
    for chunk in pd.read_csv(
        file_path,
        usecols=["step", "isFraud"],
        chunksize=100000
    ):
        rows += len(chunk)
        frauds += int(chunk["isFraud"].sum())

        min_step = min(min_step, chunk["step"].min())
        max_step = max(max_step, chunk["step"].max())

    if rows == 0:
        raise ValueError(f"{name} dataset is empty")

    results[name] = {
        "rows": rows,
        "frauds": frauds,
        "min_step": int(min_step),
        "max_step": int(max_step)
    }

# Calculate totals
total_rows = sum(r["rows"] for r in results.values())
total_frauds = sum(r["frauds"] for r in results.values())

print("\nFraudGuard AI - Split Verification")
print("---------------------------------")

for name, r in results.items():

    percentage = r["rows"] / total_rows * 100
    fraud_rate = r["frauds"] / r["rows"] * 100

    print(f"\n{name.upper()}")
    print(f"Transactions: {r['rows']:,}")
    print(f"Dataset percentage: {percentage:.2f}%")
    print(f"Fraud transactions: {r['frauds']:,}")
    print(f"Fraud rate: {fraud_rate:.4f}%")
    print(f"Time steps: {r['min_step']} to {r['max_step']}")

print("\nTOTAL TRANSACTIONS:", f"{total_rows:,}")
print("TOTAL FRAUD:", f"{total_frauds:,}")

# Verify expected PaySim totals
assert total_rows == 6362620, "Total rows do not match"
assert total_frauds == 8213, "Fraud count does not match"

# Verify chronological order between datasets
assert results["train"]["max_step"] < results["validation"]["min_step"]
assert results["validation"]["max_step"] < results["test"]["min_step"]

# Ensure every dataset contains fraud examples
assert all(results[name]["frauds"] > 0 for name in datasets)

print("\nAll split verification checks PASSED!")
