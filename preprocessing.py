import zipfile
import os
from tqdm import tqdm
import pandas as pd
import pickle

from config import NRD_RAW_CSV, OUTCOME

output_csv = str(NRD_RAW_CSV)
outcome_var = OUTCOME  # Define the outcome variable here, e.g., 'DIED', 'MOR30', 'REA30'

# ## First pass

# icd_columns = [f"i10_dx{i}" for i in range(1, 41)]
# # We also need AGE column in this pass
# columns_to_read = icd_columns + ["age"]

# unique_icd_codes = set()
# age_min, age_max = float("inf"), -float("inf")

# # Stream in chunks
# chunk_iter = pd.read_csv(
#     output_csv,
#     usecols=columns_to_read,
#     chunksize=500_000,
#     dtype=str  # read everything as string to avoid dtype issues
# )

# for i, chunk in enumerate(chunk_iter):
#     # === 1) Update ICD vocabulary ===
#     icd_series = pd.Series(chunk[icd_columns].values.ravel())
#     icd_series = icd_series.str.upper()
#     unique_icd_codes.update(icd_series.unique())

#     # === 2) Update AGE min/max ===
#     # Convert AGE to numeric (coerce errors → NaN), drop missing
#     age_vals = pd.to_numeric(chunk["age"], errors="coerce").dropna()
#     if not age_vals.empty:
#         age_min = min(age_min, age_vals.min())
#         age_max = max(age_max, age_vals.max())

#     if i % 10 == 0:
#         print(f"Processed {i} chunks... ICD vocab size: {len(unique_icd_codes)}")

# print(f"\n✅ Finished scanning all chunks!")
# print(f"Total unique ICD codes: {len(unique_icd_codes)}")
# print(f"AGE range: {age_min} - {age_max}")


# ## Get the ICD encoder
# from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()
# unique_icd_codes = [("NAN" if str(x) == "nan" else str(x)) for x in unique_icd_codes]
# encoder.fit(list(unique_icd_codes))
# print("Num unique ICD codes:", len(encoder.classes_))


# ## Get the age encoder
# from sklearn.preprocessing import MinMaxScaler

# age_scaler = MinMaxScaler(feature_range=(0,1))
# age_scaler.min_ = 0.0
# age_scaler.scale_ = 1.0 / (age_max - age_min)  # precomputed scaling

# # Save LabelEncoder
# with open('Model/full_label_encoder.pkl', 'wb') as file:
#     pickle.dump(encoder, file)

# # Save MinMaxScaler for 'AGE'
# with open('Model/full_age_scaler.pkl', 'wb') as file:
#     pickle.dump(age_scaler, file)


## Second pass

# Load the LabelEncoder for ICD codes
with open('Model/full_label_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Load the MinMaxScaler for 'AGE'
with open('Model/full_age_scaler.pkl', 'rb') as file:
    age_scaler = pickle.load(file)

# Columns we’ll process
icd_columns = [f"I10_DX{i}" for i in range(1, 41)]

X_list = []
y_list = []

# Initialize counters for tracking patient numbers
total_patients_original = 0
total_patients_after_died_filter = 0
total_patients_after_outcome_filter = 0

chunk_iter = pd.read_csv(
    output_csv,
    chunksize=500_000,
)

for i, data in enumerate(chunk_iter):
    print(f"Processing chunk {i}...")

    # Track original number of patients in this chunk
    chunk_original_count = len(data)
    total_patients_original += chunk_original_count

    # Step 1: onvert all column names to uppercase
    data.columns = data.columns.str.upper()

    # Step 2: Data Preprocessing
    # Filter out observations where DIED == 1 if outcome_var is REA30
    # if outcome_var == 'REA30' and 'DIED' in data.columns:
    if 'DIED' in data.columns:
        data = data[data['DIED'] != 1]

    # Track patients after DIED filter
    chunk_after_died = len(data)
    total_patients_after_died_filter += chunk_after_died

    # Handle missing values in the target variable
    data = data.dropna(subset=[outcome_var])

    # Track patients after outcome filter
    chunk_after_outcome = len(data)
    total_patients_after_outcome_filter += chunk_after_outcome

    # --- 1) Normalize AGE ---
    # data['AGE'] = age_scaler.transform(data['AGE'])
    data.loc[:, 'AGE'] = age_scaler.transform(data[['AGE']])

    # --- 2) Encode ICD codes ---
    for col in icd_columns:
        # data[col] = encoder.transform(data[col].astype(str).str.upper())
        data.loc[:, col] = encoder.transform(data[col].astype(str).str.upper())

    # --- 3) One-hot encode PAY1 + ZIPINC_QRTL ---
    data = pd.get_dummies(data, columns=['PAY1', 'ZIPINC_QRTL'], prefix=['PAY1', 'ZIPINC_QRTL'])

    indices_name = ['CHARLINDEX', 'CHARLINDEX_AGE_ADJUST', 'INDEX_READMISSION', 'INDEX_MORTALITY']
    # Separate features and target variable
    X_chunk = data[['AGE', 'FEMALE'] + list(data.filter(regex='PAY1_').columns) + list(data.filter(regex='ZIPINC_QRTL_').columns) + icd_columns + indices_name]
    y_chunk = data[outcome_var]
    
    # Handle missing values in features (if any)
    X_chunk = X_chunk.dropna()
    y_chunk = y_chunk.loc[X_chunk.index]
    
    # ✅ Append to global list
    X_list.append(X_chunk)
    y_list.append(y_chunk) 
    
    print(f"Chunk {i} processed: X={X_chunk.shape}, y={y_chunk.shape}")


# ✅ After all chunks processed → Concatenate everything
X_all = pd.concat(X_list, ignore_index=True)
y_all = pd.concat(y_list, ignore_index=True)

print("\n✅ All chunks processed & combined!")
print("Final combined shape:", X_all.shape, y_all.shape)

# Print patient filtering statistics
print("\n" + "="*60)
print("PATIENT FILTERING STATISTICS")
print("="*60)
print(f"Total patients in original dataset:        {total_patients_original:,}")
print(f"Patients after DIED=1 filter:              {total_patients_after_died_filter:,}")
print(f"  (Removed: {total_patients_original - total_patients_after_died_filter:,} patients)")
print(f"Patients after removing missing {outcome_var}:  {total_patients_after_outcome_filter:,}")
print(f"  (Removed: {total_patients_after_died_filter - total_patients_after_outcome_filter:,} patients)")
print(f"Final patients after all preprocessing:    {len(X_all):,}")
print(f"  (Additional removed during feature processing: {total_patients_after_outcome_filter - len(X_all):,} patients)")
print(f"\nTotal retention rate: {len(X_all)/total_patients_original*100:.2f}%")
print("="*60 + "\n")

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (stratify to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.1, random_state=42, stratify=y_all
)

# Print train/test split statistics
print("\n" + "="*60)
print("TRAIN/VALIDATION SPLIT STATISTICS")
print("="*60)
print(f"Training set size:   {len(X_train):,} patients ({len(X_train)/len(X_all)*100:.1f}%)")
print(f"Validation set size: {len(X_test):,} patients ({len(X_test)/len(X_all)*100:.1f}%)")
print(f"Total:               {len(X_all):,} patients")
print("="*60 + "\n")

print("Original class distribution:", pd.Series(y_train).value_counts())

# from sklearn.utils import resample
# from sklearn.utils import shuffle
# import numpy as np

# # Separate the majority and minority classes
# X_majority = X_train[y_train == 0]
# X_minority = X_train[y_train == 1]
# y_majority = y_train[y_train == 0]
# y_minority = y_train[y_train == 1]

# # Downsample the majority class
# X_majority_downsampled, y_majority_downsampled = resample(
#     X_majority, y_majority, 
#     replace=False,                  # No resampling; we want unique samples
#     n_samples=len(y_minority),      # Match the minority class count
#     random_state=42
# )

# # Combine the downsampled data
# # Combine minority class with downsampled majority class
# X_train_downsampled = pd.concat([X_minority, X_majority_downsampled])
# y_train_downsampled = pd.concat([y_minority, y_majority_downsampled])

# X_train_downsampled, y_train_downsampled = shuffle(X_train_downsampled, y_train_downsampled, random_state=42)

# # Print downsampling statistics
# print("\n" + "="*60)
# print("DOWNSAMPLING STATISTICS")
# print("="*60)
# print(f"Original training set:    {len(X_train):,} patients")
# print(f"  - Class 0 (negative):   {len(y_majority):,} patients")
# print(f"  - Class 1 (positive):   {len(y_minority):,} patients")
# print(f"  - Imbalance ratio:      {len(y_majority)/len(y_minority):.2f}:1")
# print(f"\nDownsampled training set: {len(X_train_downsampled):,} patients")
# print(f"  - Class 0 (negative):   {len(y_majority_downsampled):,} patients")
# print(f"  - Class 1 (positive):   {len(y_minority):,} patients")
# print(f"  - Balanced ratio:       1.00:1")
# print(f"\nValidation set (unchanged): {len(X_test):,} patients")
# print("="*60 + "\n")

# # Save to a new file for further use
# X_train_downsampled.to_csv("/users/xwang259/scratch/NRD_index/mort_nodie/X_train_downsampled.csv", index=False)
# y_train_downsampled.to_csv("/users/xwang259/scratch/NRD_index/mort_nodie/y_train_downsampled.csv", index=False)
# X_test.to_csv("/users/xwang259/scratch/NRD_index/mort_nodie/X_test.csv", index=False)
# y_test.to_csv("/users/xwang259/scratch/NRD_index/mort_nodie/y_test.csv", index=False)
