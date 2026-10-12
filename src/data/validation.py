
from pathlib import Path
import pandas as pd

# Location of our PaySim dataset
project_dir = Path(__file__).resolve().parents[2]
data_path = project_dir / "data" / "raw" / "paysim.csv"

# Expected PaySim columns
required_columns = {
    "step", "type", "amount", "nameOrig",
    "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest",
    "newbalanceDest", "isFraud",
    "isFlaggedFraud"
}

total_rows = 0
fraud_count = 0
missing_values = 0
negative_amounts = 0
invalid_labels = 0
time_order_errors = 0
previous_step = None

# Read the dataset in smaller chunks
for chunk in pd.read_csv(data_path, chunksize=100000):

    # Check required columns
    if not required_columns.issubset(chunk.columns):
        raise ValueError("Required columns are missing!")

    # Count transactions
    total_rows += len(chunk)

    # Count fraud
    fraud_count += (chunk["isFraud"] == 1).sum()

    # Check missing values
    missing_values += chunk.isna().sum().sum()

    # Check negative transaction amounts
    negative_amounts += (chunk["amount"] < 0).sum()

    # Check fraud labels (only 0 or 1 allowed)
    invalid_labels += (~chunk["isFraud"].isin([0, 1])).sum()

    # Check chronological order
    steps = chunk["step"]

    time_order_errors += (steps.diff().dropna() < 0).sum()

    if previous_step is not None:
        if steps.iloc[0] < previous_step:
            time_order_errors += 1

    previous_step = steps.iloc[-1]

print("\nFraudGuard AI - Data Validation Report")
print("--------------------------------------")
print("Total transactions:", total_rows)
print("Fraud transactions:", fraud_count)
print("Missing values:", missing_values)
print("Negative amounts:", negative_amounts)
print("Invalid labels:", invalid_labels)
print("Time order errors:", time_order_errors)

# Verify the standard PaySim dataset
assert total_rows == 6362620, "Unexpected row count"
assert fraud_count == 8213, "Unexpected fraud count"
assert missing_values == 0, "Missing values found"
assert negative_amounts == 0, "Negative amounts found"
assert invalid_labels == 0, "Invalid fraud labels found"
assert time_order_errors == 0, "Transactions are not time-ordered"

print("\nData validation PASSED!")
