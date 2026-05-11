"""Quick script to inspect the age scaler range"""
import pickle

# Load the age scaler
with open('Model/full_age_scaler.pkl', 'rb') as f:
    age_scaler = pickle.load(f)

print("Age Scaler Information:")
print("=" * 50)
print(f"Scaler type: {type(age_scaler).__name__}")
print(f"\nAvailable attributes:")
print([attr for attr in dir(age_scaler) if not attr.startswith('_')])

# Check actual scaler parameters
print(f"\nFeature range (target): {age_scaler.feature_range}")

# Try different attribute names depending on sklearn version
if hasattr(age_scaler, 'scale_') and age_scaler.scale_ is not None:
    print(f"\nScale factor: {age_scaler.scale_}")
    print(f"Min offset: {age_scaler.min_}")

    # Compute original data range from scale and min
    # MinMaxScaler formula: X_scaled = X * scale_ + min_
    # where scale_ = 1/(data_max - data_min) and min_ = -data_min * scale_

    # Handle both scalar and array types
    scale = age_scaler.scale_ if isinstance(age_scaler.scale_, float) else age_scaler.scale_[0]
    min_offset = age_scaler.min_ if isinstance(age_scaler.min_, float) else age_scaler.min_[0]

    # Reverse engineer the original range
    data_min = -min_offset / scale
    data_range = 1.0 / scale
    data_max = data_min + data_range

    print(f"\nDerived from scaler parameters:")
    print(f"  Original data min (youngest age): {data_min:.2f}")
    print(f"  Original data max (oldest age): {data_max:.2f}")
    print(f"  Data range: {data_range:.2f}")
    print(f"\nTransformation formula:")
    print(f"  scaled_age = (age - {data_min:.2f}) / {data_range:.2f}")
    print(f"  This maps age [{data_min:.2f}, {data_max:.2f}] to [0, 1]")
else:
    print("\nScaler has not been fitted yet!")
