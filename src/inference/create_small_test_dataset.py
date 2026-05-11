"""
Script to create a small test dataset (100 rows) for model testing.
This samples from the 2021-2022 test data with stratification to ensure
both positive and negative cases are included.
"""

import pandas as pd
import numpy as np

from config import NRD_2021_TEST, NRD_2022_TEST, SMALL_DATASET_DIR

# Configuration
SAMPLE_SIZE = 20
RANDOM_SEED = 42

# Define the outcome variable you want to test
# Options: 'died', 'mor30', 'rea30'
OUTCOME_VAR = 'rea30'  # Change this as needed
OUTPUT_FILE = SMALL_DATASET_DIR / f"small_test_dataset_{OUTCOME_VAR}.csv"

print(f"Creating small test dataset for outcome: {OUTCOME_VAR}")

# Load 2021 and 2022 test data
print("Loading test data...")
file_2021 = NRD_2021_TEST
file_2022 = NRD_2022_TEST

df1 = pd.read_csv(file_2021)
df2 = pd.read_csv(file_2022)

# Combine
full_data = pd.concat([df1, df2], ignore_index=True)
print(f"Combined data shape: {full_data.shape}")

# Convert column names to uppercase for consistency
full_data.columns = full_data.columns.str.upper()

# Filter based on outcome variable
if 'DIED' in full_data.columns:
    print("Filtering out patients who died...")
    full_data = full_data[full_data['DIED'] != 1]

# Remove rows with missing outcome
full_data = full_data.dropna(subset=[OUTCOME_VAR.upper()])

# Remove rows with missing critical features
required_cols = ['AGE', 'FEMALE', 'PAY1', 'ZIPINC_QRTL']
icd_cols = [f'I10_DX{i}' for i in range(1, 41)]
full_data = full_data.dropna(subset=required_cols)

print(f"After filtering: {full_data.shape}")
print(f"Outcome distribution:\n{full_data[OUTCOME_VAR.upper()].value_counts()}")

# Stratified sampling to ensure we have both positive and negative cases
np.random.seed(RANDOM_SEED)

positive_cases = full_data[full_data[OUTCOME_VAR.upper()] == 1]
negative_cases = full_data[full_data[OUTCOME_VAR.upper()] == 0]

# Sample proportionally (but ensure at least 3 positive cases for meaningful testing)
positive_ratio = len(positive_cases) / len(full_data)
n_positive = max(10, int(SAMPLE_SIZE * positive_ratio))
n_negative = SAMPLE_SIZE - n_positive

# Ensure we don't sample more than available
n_positive = min(n_positive, len(positive_cases))
n_negative = min(n_negative, len(negative_cases))

print(f"\nSampling {n_positive} positive and {n_negative} negative cases...")

sampled_positive = positive_cases.sample(n=n_positive, random_state=RANDOM_SEED)
sampled_negative = negative_cases.sample(n=n_negative, random_state=RANDOM_SEED)

# Combine and shuffle
small_dataset = pd.concat([sampled_positive, sampled_negative], ignore_index=True)
small_dataset = small_dataset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nFinal dataset shape: {small_dataset.shape}")
print(f"Final outcome distribution:\n{small_dataset[OUTCOME_VAR.upper()].value_counts()}")

# Save to CSV
small_dataset.to_csv(OUTPUT_FILE, index=False)
print(f"\nSmall test dataset saved to: {OUTPUT_FILE}")

# Print column summary
print("\nColumn summary:")
print(f"- ICD columns: {icd_cols}")
print(f"- Demographic: AGE, FEMALE")
print(f"- Payer: PAY1 (unique values: {small_dataset['PAY1'].nunique()})")
print(f"- Income: ZIPINC_QRTL (unique values: {small_dataset['ZIPINC_QRTL'].nunique()})")
print(f"- Outcome: {OUTCOME_VAR.upper()}")
print(f"- Traditional indices: CHARLINDEX, CHARLINDEX_AGE_ADJUST, INDEX_READMISSION, INDEX_MORTALITY")
