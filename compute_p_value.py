#!/usr/bin/env python3
"""
Compare two AUC values and their confidence intervals to compute statistical significance.

Usage:
    python compute_p_value.py --auc1 0.85 --ci1 0.82 0.88 --auc2 0.78 --ci2 0.75 0.81
"""

import numpy as np
import scipy.stats as stats
import argparse


def compare_aucs_from_ci(auc1, ci1_low, ci1_high, auc2, ci2_low, ci2_high):
    """Approximate comparison when you only have summary statistics."""
    # Estimate SE from CI (assuming 95% CI)
    se1 = (ci1_high - ci1_low) / (2 * 1.96)
    se2 = (ci2_high - ci2_low) / (2 * 1.96)

    # If independent samples (different datasets):
    z = (auc1 - auc2) / np.sqrt(se1**2 + se2**2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


def main():
    parser = argparse.ArgumentParser(
        description='Compare two AUC values with confidence intervals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python compute_p_value.py --auc1 0.85 --ci1 0.82 0.88 --auc2 0.78 --ci2 0.75 0.81
        """
    )

    parser.add_argument('--auc1', type=float, required=True,
                        help='First AUC value')
    parser.add_argument('--ci1', type=float, nargs=2, required=True,
                        metavar=('LOW', 'HIGH'),
                        help='95%% CI for first AUC (lower upper)')
    parser.add_argument('--auc2', type=float, required=True,
                        help='Second AUC value')
    parser.add_argument('--ci2', type=float, nargs=2, required=True,
                        metavar=('LOW', 'HIGH'),
                        help='95%% CI for second AUC (lower upper)')

    args = parser.parse_args()

    # Extract CI bounds
    ci1_low, ci1_high = args.ci1
    ci2_low, ci2_high = args.ci2

    # Compute p-value
    z, p_value = compare_aucs_from_ci(
        args.auc1, ci1_low, ci1_high,
        args.auc2, ci2_low, ci2_high
    )

    # Print results
    print("\n" + "="*60)
    print("AUC Comparison Results")
    print("="*60)
    print(f"\nModel 1:")
    print(f"  AUC = {args.auc1:.4f} (95% CI: {ci1_low:.4f}-{ci1_high:.4f})")
    print(f"  SE  = {(ci1_high - ci1_low) / (2 * 1.96):.4f}")

    print(f"\nModel 2:")
    print(f"  AUC = {args.auc2:.4f} (95% CI: {ci2_low:.4f}-{ci2_high:.4f})")
    print(f"  SE  = {(ci2_high - ci2_low) / (2 * 1.96):.4f}")

    print(f"\nStatistical Test:")
    print(f"  Difference = {args.auc1 - args.auc2:.4f}")
    print(f"  Z-statistic = {z:.4f}")
    print(f"  P-value = {p_value:.6f}")

    # Interpret result
    print(f"\nInterpretation:")
    if p_value < 0.001:
        print(f"  *** Highly significant (p < 0.001)")
    elif p_value < 0.01:
        print(f"  ** Very significant (p < 0.01)")
    elif p_value < 0.05:
        print(f"  * Significant (p < 0.05)")
    else:
        print(f"  Not significant (p >= 0.05)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
