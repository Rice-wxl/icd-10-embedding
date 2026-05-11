"""
Check the encoder contents
"""

import pickle

# Load encoder
with open('Model/full_label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

print(f"\nTotal classes: {len(encoder.classes_)}")
print(f"\nFirst 20 classes:")
for i, label in enumerate(encoder.classes_[:20]):
    print(f"  {i}: {label}")

print(f"\nLast 20 classes:")
for i, label in enumerate(encoder.classes_[-20:], start=len(encoder.classes_)-20):
    print(f"  {i}: {label}")

# Check for specific codes
codes_to_check = ['I61', 'NAN', 'nan', 'i61', 'I610']

for code in codes_to_check:
    if code in encoder.classes_:
        idx = encoder.transform([code])[0]
        print(f"  '{code}': found at index {idx}")
    else:
        print(f"  '{code}': NOT FOUND")

# Search for codes starting with 'I6'
print(f"\nCodes starting with 'I6':")
i6_codes = [c for c in encoder.classes_ if str(c).startswith('I61')]
print(f"  Found {len(i6_codes)} codes")
if i6_codes:
    for code in i6_codes[:30]:  # Show first 30
        idx = encoder.transform([code])[0]
        print(f"    {idx}: {code}")