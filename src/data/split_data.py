
from pathlib import Path
import pandas as pd

# Step 1: Define file locations
PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_DIR / "data" / "raw" / "paysim.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100000

# Step 2: Count transactions in each time step
step_counts = pd.Series(dtype="float64")

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["step"],
    chunksize=CHUNK_SIZE
):
    counts = chunk["step"].value_counts()
    step_counts = step_counts.add(counts, fill_value=0)

step_counts = step_counts.sort_index()

total_rows = int(step_counts.sum())
cumulative = step_counts.cumsum()

# Step 3: Find chronological split boundaries
train_target = total_rows * 0.70
validation_target = total_rows * 0.85

train_end = int(cumulative[
    cumulative >= train_target
].index[0])

validation_end = int(cumulative[
    cumulative >= validation_target
].index[0])

if (
    train_end >= validation_end
    or validation_end >= step_counts.index[-1]
):
    raise ValueError("Invalid time split boundaries")

print("Training ends at step:", train_end)
print("Validation ends at step:", validation_end)

# Step 4: Define output files
files = {
    "train": OUTPUT_DIR / "train.csv",
    "validation": OUTPUT_DIR / "validation.csv",
    "test": OUTPUT_DIR / "test.csv"
}

# Remove files from previous runs
for path in files.values():
    if path.exists():
        path.unlink()

row_counts = {name: 0 for name in files}
fraud_counts = {name: 0 for name in files}

# Step 5: Read data in chunks and save each split
for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):
    groups = {
        "train": chunk["step"] <= train_end,
        "validation": (
            (chunk["step"] > train_end)
            & (chunk["step"] <= validation_end)
        ),
        "test": chunk["step"] > validation_end
    }

    for name, mask in groups.items():
        subset = chunk.loc[mask]

        if subset.empty:
            continue

        output_file = files[name]

        subset.to_csv(
            output_file,
            mode="a",
            header=not output_file.exists(),
            index=False
        )

        row_counts[name] += len(subset)
        fraud_counts[name] += int(
            (subset["isFraud"] == 1).sum()
        )

# Step 6: Verify and display results
assert sum(row_counts.values()) == total_rows

print("\nFraudGuard AI - Data Split Report")
print("--------------------------------")

for name in files:
    rows = row_counts[name]
    frauds = fraud_counts[name]

    print(f"\n{name.upper()}")
    print(f"Transactions: {rows:,}")
    print(f"Fraud transactions: {frauds:,}")
    print(f"Fraud percentage: {frauds / rows * 100:.4f}%")

print("\nChronological data splitting completed!")
