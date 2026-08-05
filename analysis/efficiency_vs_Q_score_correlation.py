import scipy.stats as stats
import numpy as np

# ============================================================
# Q-score vs Token Efficiency Correlation
# All 13 reasoning models from Table 14 (paper)
# Efficiency = Peak GSM8K Accuracy / Avg Tokens at Peak Limit
# ============================================================

models = [
    # (name, Q-score, peak_acc, avg_tokens_at_peak_limit)
    # Q-scores from Table 14
    # Peak accuracy and avg tokens from Table 12 (raw data)
    ("Skywork-o1-8B",   0.9867, 0.86, 292.8),   # peak at 1000 tokens
    ("QwenQwQ-32B",     0.9315, 0.98, 493.6),   # peak at 2000 tokens
    ("DS-R1-70B",       0.8731, 0.96, 593.2),   # peak at 2000 tokens
    ("DS-R1-685B",      0.6843, 1.00, 1547.0),  # peak at 4000 tokens
    ("DS-R1-14B",       0.6843, 0.82, 914.9),   # peak at 2000 tokens
    ("Qwen3-32B",       0.6793, 0.98, 1583.7),  # peak at 4000 tokens
    ("Nemotron-7B",     0.5556, 0.80, 1816.9),  # peak at 4000 tokens
    ("GLM-4.5",         0.5480, 0.80, 965.3),   # peak at 8000 tokens
    ("GLM-4.7",         0.4783, 0.86, 1042.0),  # peak at 8000 tokens
    ("Kimi2.5",         0.4566, 0.72, 597.8),   # peak at 4000 tokens
    ("DS-R1-7B",        0.3591, 0.66, 2641.6),  # peak at 4000 tokens
    ("Marco-o1-7B",     0.2428, 0.56, 961.9),   # peak at 1000 tokens
    ("DS-R1-1.5B",      0.0819, 0.46, 1957.6),  # peak at 2000 tokens
]

# ============================================================
# Step 1: Calculate efficiency for each model
# ============================================================
print("="*70)
print("STEP 1: Token Efficiency Calculation")
print("Formula: efficiency = peak_accuracy / avg_tokens_at_peak")
print("="*70)
print(f"{'Model':<20} {'Q-score':>8} {'Peak%':>7} {'AvgTok':>8} {'Efficiency':>12}")
print("-"*60)

q_scores     = []
efficiencies = []

for name, q, acc, tokens in models:
    eff = acc / tokens
    q_scores.append(q)
    efficiencies.append(eff)
    print(f"{name:<20} {q:>8.4f} {acc*100:>6.0f}% {tokens:>8.1f} {eff:>12.6f}")

# ============================================================
# Step 2: Pearson correlation
# ============================================================
pearson_r, pearson_p = stats.pearsonr(q_scores, efficiencies)

# ============================================================
# Step 3: Spearman correlation
# ============================================================
spearman_r, spearman_p = stats.spearmanr(q_scores, efficiencies)

print("\n" + "="*70)
print("STEP 2 & 3: Correlation Results")
print("="*70)
print(f"n = {len(models)} models")
print(f"\nPearson  r = {pearson_r:+.3f},  P = {pearson_p:.4f}")
print(f"Spearman ρ = {spearman_r:+.3f},  P = {spearman_p:.4f}")
print(f"\nPearson r² = {pearson_r**2:.3f} ({pearson_r**2*100:.1f}% variance explained)")
n = len(models)
z = np.arctanh(pearson_r)
se = 1 / np.sqrt(n - 3)
ci_low = np.tanh(z - 1.96 * se)
ci_high = np.tanh(z + 1.96 * se)
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
# ============================================================
# Step 4: Interpretation
# ============================================================
print("\n" + "="*70)
print("STEP 4: Interpretation")
print("="*70)
if pearson_p < 0.001:
    sig = "P < 0.001 (highly significant)"
elif pearson_p < 0.01:
    sig = f"P = {pearson_p:.4f} (significant)"
elif pearson_p < 0.05:
    sig = f"P = {pearson_p:.4f} (significant)"
else:
    sig = f"P = {pearson_p:.4f} (not significant)"

print(f"Significance: {sig}")
print(f"Effect size:  r = {pearson_r:.3f} ({'strong' if abs(pearson_r) > 0.7 else 'moderate'})")
print(f"\nLaTeX for paper:")
print(f"(Pearson $r = {pearson_r:+.3f}$, $P = {pearson_p:.4f}$, $n = {len(models)}$ models)")

# ============================================================
# Step 5: Check monotonic relationship tier by tier
# ============================================================
print("\n" + "="*70)
print("STEP 5: Tier-level monotonicity check")
print("="*70)

tier_models = {
    1: [m for m in models if m[1] >= 0.88],
    2: [m for m in models if 0.79 <= m[1] < 0.88],
    3: [m for m in models if 0.58 <= m[1] < 0.79],
    4: [m for m in models if 0.38 <= m[1] < 0.58],
    5: [m for m in models if m[1] < 0.38],
}

tier_labels = {1:'Excellent', 2:'Good', 3:'Moderate', 4:'Poor', 5:'Insufficient'}

print(f"{'Tier':<6} {'Label':<12} {'Mean Q':>8} {'Mean Eff':>12} {'n':>4}")
print("-"*45)

prev_mean_eff = None
tier_monotonic = True
for tier in [1,2,3,4,5]:
    ms = tier_models[tier]
    if not ms:
        continue
    mean_q   = np.mean([m[1] for m in ms])
    mean_eff = np.mean([m[2]/m[3] for m in ms])
    flag = ""
    if prev_mean_eff is not None and mean_eff > prev_mean_eff:
        flag = " ← REVERSAL"
        tier_monotonic = False
    print(f"Tier {tier}  {tier_labels[tier]:<12} {mean_q:>8.3f} {mean_eff:>12.6f}{flag}")
    prev_mean_eff = mean_eff

print(f"\nMonotonic at tier level: {tier_monotonic}")
print("\n" + "="*70)
print("LEAVE-ONE-OUT ROBUSTNESS")
print("="*70)
loo_r_values = []
for i, (name, q, acc, tokens) in enumerate(models):
  q_loo = [q_scores[j] for j in range(len(models)) if j != i]
  eta_loo = [efficiencies[j] for j in range(len(models)) if j != i]
  r_loo, _ = stats.pearsonr(q_loo, eta_loo)
  loo_r_values.append(r_loo)
  print(f"{name:<20} r = {r_loo:.3f}")
print(f"\nMin LOO r : {min(loo_r_values):.3f}")
print(f"Max LOO r : {max(loo_r_values):.3f}")