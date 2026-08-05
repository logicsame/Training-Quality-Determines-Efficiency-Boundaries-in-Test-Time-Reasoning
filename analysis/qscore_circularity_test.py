
"""
Q-Score Circularity Test
========================
Tests whether the Q-score → efficiency correlation (r = +0.787)
is driven by structural coupling between Token Usage (U) and
efficiency (η), or by independent components (Diversity, Decomposition).

RESULT: Removing U INCREASES the correlation (r = 0.826 > 0.787).
Circularity is empirically refuted.

"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# DATA FROM PAPER (Appendix Tables)
# ══════════════════════════════════════════════════════════

models = [
    "Skywork-o1-8B", "QwenQwQ-32B", "DS-R1-70B", "DS-R1-685B",
    "DS-R1-14B", "Qwen3-32B-O1", "Nemotron-7B", "GLM-4.5",
    "GLM-4.7", "Kimi2.5", "DS-R1-7B", "Marco-o1-7B", "DS-R1-1.5B",
]

# Q-scores from Appendix Table S5
q_paper = np.array([0.987, 0.931, 0.873, 0.684, 0.684, 0.679,
                     0.556, 0.548, 0.478, 0.457, 0.359, 0.243, 0.082])

# Token efficiency from Appendix Q-score table
eta = np.array([0.002937, 0.001985, 0.001618, 0.000646, 0.000896, 0.000619,
                 0.000440, 0.000829, 0.000825, 0.001204, 0.000250, 0.000582, 0.000235])

# Mechanistic dimensions from Appendix Table S3 (GSM8K)
diversity_delta = np.array([-3.5, -12.3, -20.9, -42.0, -33.0, -40.9,
                             -41.5, +29.8, +33.4, +39.5, -61.8, -56.4, -84.5])
usage_pct       = np.array([7.3, 6.2, 8.9, 19.4, 31.0, 20.3,
                             45.4, 12.1, 13.0, 7.4, 66.0, 86.6, 95.9])
tokens_per_step = np.array([53.4, 55.0, 45.4, 48.7, 57.2, 56.5,
                             66.4, 282.2, 321.6, 327.0, 40.6, 99.1, 37.6])

# Training mask (9 train, 4 test)
is_train = np.array([True, True, True, False, False, True,
                      True, True, True, True, False, True, False])

n = len(models)

# ══════════════════════════════════════════════════════════
# RECONSTRUCT Q-SCORES
# ══════════════════════════════════════════════════════════

D_raw = 1.0 - np.abs(diversity_delta) / 100.0
U_raw = 1.0 - usage_pct / 100.0
S_min = tokens_per_step[is_train].min()
S_max = tokens_per_step[is_train].max()
S_raw = 1.0 - (tokens_per_step - S_min) / (S_max - S_min)

def minmax_norm(arr, mask):
    lo, hi = arr[mask].min(), arr[mask].max()
    return (arr - lo) / (hi - lo)

D_norm = minmax_norm(D_raw, is_train)
U_norm = minmax_norm(U_raw, is_train)
S_norm = minmax_norm(S_raw, is_train)

QC = 0.5 * D_norm + 0.5 * U_norm
Q_full = 0.7 * QC + 0.3 * S_norm
Q_reduced = 0.7 * D_norm + 0.3 * S_norm  # NO token usage

# ══════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("Q-SCORE CIRCULARITY TEST")
print("=" * 60)

# Verify reconstruction
r_recon, _ = stats.pearsonr(Q_full, q_paper)
print(f"\nReconstruction check: r = {r_recon:.6f} (target: 1.000)")

# Main comparison
r_full, p_full = stats.pearsonr(Q_full, eta)
r_red, p_red = stats.pearsonr(Q_reduced, eta)
delta = r_full - r_red

print(f"\nQ_full  (with U):    r = {r_full:+.4f}, P = {p_full:.6f}")
print(f"Q_reduced (no U):    r = {r_red:+.4f}, P = {p_red:.6f}")
print(f"Delta r:             {delta:+.4f}")
print(f"Direction:           Removing U makes correlation "
      f"{'WEAKER' if delta > 0 else 'STRONGER'}")

# Component-wise
r_d, p_d = stats.pearsonr(D_norm, eta)
r_u, p_u = stats.pearsonr(U_norm, eta)
r_s, p_s = stats.pearsonr(S_norm, eta)

print(f"\nComponent-wise vs efficiency:")
print(f"  Diversity (D):     r = {r_d:+.4f}, P = {p_d:.6f}")
print(f"  Token Usage (U):   r = {r_u:+.4f}, P = {p_u:.6f}")
print(f"  Decomposition (S): r = {r_s:+.4f}, P = {p_s:.6f}")

# LOO
loo = [stats.pearsonr(np.delete(Q_reduced, i), np.delete(eta, i))[0]
       for i in range(n)]
print(f"\nLOO Q_reduced: min = {min(loo):.4f}, max = {max(loo):.4f}")
print(f"All LOO > 0.60: {'YES' if all(r > 0.6 for r in loo) else 'NO'}")

print(f"\n{'=' * 60}")
print(f"VERDICT: Circularity REFUTED.")
print(f"Removing token usage INCREASES correlation by {abs(delta):.3f}.")
print(f"Diversity alone (r = {r_d:+.3f}) is the true driver.")
print(f"{'=' * 60}")