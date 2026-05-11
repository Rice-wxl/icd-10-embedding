"""
Debug script to compare preprocessing between two methods.
This will help identify why predict_single_patient.py and predict_on_small_dataset.py
give different predictions for the same patient data.
"""

import pandas as pd
import numpy as np
import pickle
import sys

# Load the CSV file
csv_file = 'small_dataset_rea30_custom.csv'
print("="*60)
print("DEBUGGING PREPROCESSING DISCREPANCY")
print("="*60)
print(f"\nLoading CSV: {csv_file}")
df_original = pd.read_csv(csv_file)
print(f"Original data shape: {df_original.shape}")
print("\nOriginal data:")
print(df_original)
print("\nColumn names:", df_original.columns.tolist())

# Uppercase columns (as done in predict_on_small_dataset.py)
df_original.columns = df_original.columns.str.upper()

# Extract patient data from CSV
age = df_original['AGE'].values[0]
female = df_original['FEMALE'].values[0]
pay1 = df_original['PAY1'].values[0]
zipinc_qrtl = df_original['ZIPINC_QRTL'].values[0]

# Load encoders
print("\nLoading encoders...")
with open('Model/full_label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('Model/full_age_scaler.pkl', 'rb') as f:
    age_scaler = pickle.load(f)

# Define ICD columns
icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

# Extract ICD codes from CSV
print("\n" + "="*60)
print("METHOD 1: predict_on_small_dataset.py (CSV direct)")
print("="*60)
df_csv = df_original.copy()

print("\nRaw ICD codes from CSV:")
for i, col in enumerate(icd_columns[:5], 1):  # Show first 5
    raw_val = df_csv[col].values[0]
    print(f"  {col}: {repr(raw_val)} (type: {type(raw_val).__name__})")

# Process like predict_on_small_dataset.py
label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
unknown_label_int = encoder.transform(["NAN"])[0]
print(f"\nunknown_label_int (for 'NAN'): {unknown_label_int}")

for col in icd_columns:
    df_csv[col] = df_csv[col].astype(str).str.upper()
    df_csv[col] = df_csv[col].map(label_to_int).fillna(unknown_label_int).astype(int)

print("\nEncoded ICD codes (first 5):")
for i, col in enumerate(icd_columns[:5], 1):
    print(f"  {col}: {df_csv[col].values[0]}")

# Normalize age
df_csv['AGE'] = age_scaler.transform(df_csv[['AGE']])

# Handle missing value codes
df_csv['PAY1'] = df_csv['PAY1'].replace([-8, -9], np.nan)
df_csv['ZIPINC_QRTL'] = df_csv['ZIPINC_QRTL'].replace([-8, -9], np.nan)

# One-hot encode
df_csv = pd.get_dummies(df_csv, columns=['PAY1', 'ZIPINC_QRTL'],
                        prefix=['PAY1', 'ZIPINC_QRTL'])

# Add missing columns
expected_pay1_columns = ['PAY1_1.0', 'PAY1_2.0', 'PAY1_3.0', 'PAY1_4.0', 'PAY1_5.0', 'PAY1_6.0']
expected_zipinc_columns = ['ZIPINC_QRTL_1.0', 'ZIPINC_QRTL_2.0', 'ZIPINC_QRTL_3.0', 'ZIPINC_QRTL_4.0']

for col in expected_pay1_columns:
    if col not in df_csv.columns:
        df_csv[col] = 0

for col in expected_zipinc_columns:
    if col not in df_csv.columns:
        df_csv[col] = 0

X_csv = df_csv[['AGE', 'FEMALE'] + expected_pay1_columns + expected_zipinc_columns + icd_columns]

print("\n" + "="*60)
print("METHOD 2: predict_single_patient.py (manual input)")
print("="*60)

# Now simulate predict_single_patient.py method
# Extract ICD codes as list
icd_codes_raw = []
for col in icd_columns:
    val = df_original[col].values[0]
    if pd.notna(val) and str(val).strip() != '':
        icd_codes_raw.append(str(val).strip())

print(f"\nExtracted ICD codes list: {icd_codes_raw}")
print(f"Number of codes: {len(icd_codes_raw)}")

# Create DataFrame like predict_single_patient.py does
data = {}
data['AGE'] = [age]
data['FEMALE'] = [female]
data['PAY1'] = [pay1]
data['ZIPINC_QRTL'] = [zipinc_qrtl]

# Add ICD codes with 'NAN' padding
for i, col in enumerate(icd_columns):
    if i < len(icd_codes_raw):
        data[col] = [icd_codes_raw[i]]
    else:
        data[col] = ['NAN']

df_manual = pd.DataFrame(data)

print("\nRaw ICD codes in manual DataFrame (first 5):")
for i, col in enumerate(icd_columns[:5], 1):
    print(f"  {col}: {repr(df_manual[col].values[0])}")

# Encode ICD codes
for col in icd_columns:
    df_manual[col] = df_manual[col].astype(str).str.upper()
    df_manual[col] = df_manual[col].map(label_to_int).fillna(unknown_label_int).astype(int)

print("\nEncoded ICD codes (first 5):")
for i, col in enumerate(icd_columns[:5], 1):
    print(f"  {col}: {df_manual[col].values[0]}")

# Normalize age
df_manual['AGE'] = age_scaler.transform(df_manual[['AGE']])

# Handle missing value codes
df_manual['PAY1'] = df_manual['PAY1'].replace([-8, -9], np.nan)
df_manual['ZIPINC_QRTL'] = df_manual['ZIPINC_QRTL'].replace([-8, -9], np.nan)

# One-hot encode
df_manual = pd.get_dummies(df_manual, columns=['PAY1', 'ZIPINC_QRTL'],
                          prefix=['PAY1', 'ZIPINC_QRTL'])

# Add missing columns
for col in expected_pay1_columns:
    if col not in df_manual.columns:
        df_manual[col] = 0

for col in expected_zipinc_columns:
    if col not in df_manual.columns:
        df_manual[col] = 0

X_manual = df_manual[['AGE', 'FEMALE'] + expected_pay1_columns + expected_zipinc_columns + icd_columns]

# Compare the two
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

print("\nComparing all columns...")
all_match = True
for col in X_csv.columns:
    csv_val = X_csv[col].values[0]
    manual_val = X_manual[col].values[0]
    if csv_val != manual_val:
        print(f"  MISMATCH in {col}:")
        print(f"    CSV method:    {csv_val}")
        print(f"    Manual method: {manual_val}")
        all_match = False

if all_match:
    print("  ✓ All columns match!")
else:
    print("\n⚠️  Found mismatches!")

# Show summary of inputs
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nCSV method shape:", X_csv.shape)
print("Manual method shape:", X_manual.shape)

print("\nCSV method AGE:", X_csv['AGE'].values[0])
print("Manual method AGE:", X_manual['AGE'].values[0])

print("\nCSV method FEMALE:", X_csv['FEMALE'].values[0])
print("Manual method FEMALE:", X_manual['FEMALE'].values[0])

print("\nCSV method PAY1 columns:")
for col in expected_pay1_columns:
    print(f"  {col}: {X_csv[col].values[0]}")

print("\nManual method PAY1 columns:")
for col in expected_pay1_columns:
    print(f"  {col}: {X_manual[col].values[0]}")

print("\nCSV method ZIPINC_QRTL columns:")
for col in expected_zipinc_columns:
    print(f"  {col}: {X_csv[col].values[0]}")

print("\nManual method ZIPINC_QRTL columns:")
for col in expected_zipinc_columns:
    print(f"  {col}: {X_manual[col].values[0]}")

# Check for differences in ICD codes
print("\nICD code differences:")
icd_diffs = []
for col in icd_columns:
    if X_csv[col].values[0] != X_manual[col].values[0]:
        icd_diffs.append((col, X_csv[col].values[0], X_manual[col].values[0]))

if icd_diffs:
    print(f"  Found {len(icd_diffs)} ICD code mismatches:")
    for col, csv_val, manual_val in icd_diffs[:10]:  # Show first 10
        print(f"    {col}: CSV={csv_val}, Manual={manual_val}")
else:
    print("  ✓ All ICD codes match!")
