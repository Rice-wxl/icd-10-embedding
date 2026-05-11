import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap

from config import FEATURE_IMPORTANCE_DIR, FEATURE_IMPORTANCE_FIG_DIR

output_var = "mor30"

# Read the CSV files
positive_df = pd.read_csv(FEATURE_IMPORTANCE_DIR / 'mort_top10_positive.csv')
negative_df = pd.read_csv(FEATURE_IMPORTANCE_DIR / 'mort_top10_negative.csv')

# Select relevant columns and combine
positive_data = positive_df[['icd_code', 'ig_signed_mean', 'icd_description']].copy()
negative_data = negative_df[['icd_code', 'ig_signed_mean', 'icd_description']].copy()

# Combine the dataframes
combined_df = pd.concat([positive_data, negative_data], ignore_index=True)

# Sort by ig_signed_mean to have negative values at bottom, positive at top
combined_df = combined_df.sort_values('ig_signed_mean')

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 10))

# Get values
y_pos = np.arange(len(combined_df))
values = combined_df['ig_signed_mean'].values
codes = combined_df['icd_code'].values
descriptions = combined_df['icd_description'].values

# Create wrapped labels
def create_wrapped_label(code, desc, max_width=50):
    """Create a label with code and wrapped description."""
    # Wrap the description to max_width characters
    wrapped_lines = textwrap.wrap(desc, width=max_width)
    if len(wrapped_lines) == 1:
        return f"{code}: {wrapped_lines[0]}"
    else:
        # Join first two lines
        return f"{code}: {wrapped_lines[0]}\n{wrapped_lines[1]}" if len(wrapped_lines) > 1 else f"{code}: {wrapped_lines[0]}"

labels = [create_wrapped_label(code, desc) for code, desc in zip(codes, descriptions)]

# Plot lollipops
for i, (val, label) in enumerate(zip(values, labels)):
    # Determine color based on positive/negative
    color = 'green' if val > 0 else 'red'

    # Draw the line (stem) without markers
    ax.plot([0, val], [i, i], '-', color=color, linewidth=2)

    # Draw only the marker at the end point
    ax.plot(val, i, 'o', color=color, markersize=8)

    # Add text labels
    if val > 0:  # Positive values - label on the left
        ax.text(-0.00005, i, label, va='center', ha='right', fontsize=9)
    else:  # Negative values - label on the right
        ax.text(0.00005, i, label, va='center', ha='left', fontsize=9)

# Add a vertical line at x=0
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)

# Set labels
ax.set_xlabel('Integrated Gradients (IG) per Occurrence', fontsize=12)
ax.set_yticks([])  # Remove y-axis ticks since labels are embedded

# Format x-axis to show values in scientific notation
ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# Add grid for better readability
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_path = FEATURE_IMPORTANCE_FIG_DIR / f'icd_importance_{output_var}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Lollipop chart saved to: {output_path}")

# # Also save as PDF for better quality
# output_path_pdf = FEATURE_IMPORTANCE_FIG_DIR / 'icd_importance_lollipop.pdf'
# plt.savefig(output_path_pdf, bbox_inches='tight')
# print(f"Lollipop chart (PDF) saved to: {output_path_pdf}")

# plt.show()
