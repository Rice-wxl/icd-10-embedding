#!/usr/bin/env python3
"""
Script to check for presence of a specific ICD-10 code in NRD dataset.
Usage: python check_code.py <path_to_csv> [--code CODE]
"""

import argparse
import pandas as pd
import sys

def check_code_in_dataset(file_path, code="I61"):
    """
    Check for presence of a specific ICD-10 code in diagnosis columns.

    Args:
        file_path: Path to the CSV file
        code: ICD-10 code to search for (default: "I61")

    Returns:
        Dictionary with statistics about code presence
    """
    print(f"Loading dataset: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"Dataset shape: {df.shape}")
    df.columns = df.columns.str.upper()
    
    # Define diagnosis columns to check
    dx_columns = [f'I10_DX{i}' for i in range(1, 41)]

    # Check which columns exist in the dataset
    available_dx_columns = [col for col in dx_columns if col in df.columns]
    missing_dx_columns = [col for col in dx_columns if col not in df.columns]

    if missing_dx_columns:
        print(f"Warning: {len(missing_dx_columns)} diagnosis columns not found in dataset")

    print(f"\nSearching for code '{code}' in {len(available_dx_columns)} diagnosis columns...")

    # Check for exact matches
    matches_per_column = {}
    total_matches = 0
    patients_with_code = set()

    for col in available_dx_columns:
        # Count exact matches in this column
        matches = (df[col] == code).sum()
        if matches > 0:
            matches_per_column[col] = matches
            total_matches += matches
            # Track patient indices with this code
            patients_with_code.update(df[df[col] == code].index.tolist())

    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS FOR CODE: {code}")
    print(f"{'='*60}")
    print(f"Total occurrences: {total_matches}")
    print(f"Unique patients with code: {len(patients_with_code)}")
    print(f"Percentage of patients: {len(patients_with_code) / len(df) * 100:.2f}%")

    if matches_per_column:
        print(f"\nBreakdown by column:")
        for col, count in sorted(matches_per_column.items(), key=lambda x: x[1], reverse=True):
            print(f"  {col}: {count} occurrences")
    else:
        print(f"\nNo occurrences of code '{code}' found in the dataset.")

    return {
        'code': code,
        'file_path': file_path,
        'total_occurrences': total_matches,
        'unique_patients': len(patients_with_code),
        'total_patients': len(df),
        'percentage': len(patients_with_code) / len(df) * 100 if len(df) > 0 else 0,
        'columns_checked': len(available_dx_columns),
        'matches_per_column': matches_per_column
    }

def main():
    parser = argparse.ArgumentParser(
        description='Check for presence of ICD-10 code in NRD dataset'
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the CSV file to check'
    )
    parser.add_argument(
        '--code',
        type=str,
        default='I61',
        help='ICD-10 code to search for (default: I61)'
    )

    args = parser.parse_args()

    check_code_in_dataset(args.file_path, args.code)

if __name__ == "__main__":
    main()
