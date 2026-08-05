
from scipy import stats
import numpy as np

models = ["DS-R1-1.5B","DS-R1-14B","DS-R1-70B","DS-R1-7B","DeepSeek-R1-685B",
          "GLM-4.5","GLM-4.7","Kimi2.5","Marco-o1-7B","Nemotron-7B",
          "Qwen3-32B-O1","QwenQwQ-32B","Skywork-o1-8B"]

# data from paper appendix table 20
gpqa  = np.array([50,74,68,70,44,66,46,60,54,58,52,64,42])
gsm8k = np.array([46,82,96,66,100,80,86,72,56,80,98,98,86])

r_p, p_p = stats.pearsonr(gsm8k, gpqa)
r_s, p_s = stats.spearmanr(gsm8k, gpqa)

print("=" * 60)
print("GSM8K vs GPQA — ACTUAL DATA")
print("=" * 60)
print(f"Pearson  r = {r_p:+.4f}, P = {p_p:.4f}")
print(f"Spearman ρ = {r_s:+.4f}, P = {p_s:.4f}")
print(f"\nPaper claims: r = -0.03, P = 0.91")
print(f"Paper claims: ρ = -0.21, P = 0.50")

# Models with >=80% GSM8K
print(f"\n{'='*60}")
print("MODELS WITH >=80% GSM8K")
high = [(m,g,gp) for m,g,gp in zip(models,gsm8k,gpqa) if g >= 80]
gp_vals = [gp for _,_,gp in high]
print(f"Count: {len(high)}")
print(f"GPQA range: {min(gp_vals)}–{max(gp_vals)}%")
print(f"GPQA mean: {np.mean(gp_vals):.1f}%")
print(f"Degradation: {np.mean([g-gp for _,g,gp in high]):.1f}pp")
print(f"Paper claims: 9 models, 42-74%, mean ≈57%, ~32pp")
for m,g,gp in high:
    print(f"  {m:<20s}: GSM8K={g}%, GPQA={gp}%, gap={g-gp}pp")

# Models >=96% GSM8K
print(f"\n{'='*60}")
print("MODELS WITH >=96% GSM8K (caption check)")
vh = [(m,g,gp) for m,g,gp in zip(models,gsm8k,gpqa) if g >= 96]
vgp = [gp for _,_,gp in vh]
print(f"Range: {min(vgp)}–{max(vgp)}%, spread={max(vgp)-min(vgp)}pp")
print(f"Caption claims: 44-68%, 24pp")
for m,g,gp in vh:
    print(f"  {m:<20s}: GSM8K={g}%, GPQA={gp}%")

# Collapse (<50% GPQA despite >=80% GSM8K)
print(f"\n{'='*60}")
print("COLLAPSE: >=80% GSM8K but <50% GPQA")
collapse = [(m,g,gp) for m,g,gp in zip(models,gsm8k,gpqa) if g >= 80 and gp < 50]
print(f"Count: {len(collapse)}")
print(f"Paper claims: 'three models collapse below 50%'")
for m,g,gp in collapse:
    print(f"  {m:<20s}: GSM8K={g}%, GPQA={gp}%")

# Best GPQA
best = max(range(len(models)), key=lambda i: gpqa[i])
print(f"\nBest GPQA: {models[best]} = {gpqa[best]}%")
print(f"Paper claims: DS-R1-14B at 74%")
