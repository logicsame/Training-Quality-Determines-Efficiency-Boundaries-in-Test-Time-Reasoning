"""
COMPLETE EFFICIENCY ANALYSIS - ALL 13 MODELS
Fixed model name matching and proper efficiency calculation
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load token ablation
small_models = pd.read_csv('data/processed/reasoning_models_ablation_data/small_reasoning_model_token_ablation_summary.csv')
large_models = pd.read_csv('data/processed/reasoning_models_ablation_data/token_ablation_summary.csv')
token_ablation = pd.concat([small_models, large_models], ignore_index=True)

# Load functional analysis (for Q-scores)
functional_df = pd.read_csv('aggregate_results/functional_analysis_summary.csv')

# FIX MODEL NAMES: Remove "_model" suffix from functional analysis
functional_df['model'] = functional_df['model'].str.replace('_model', '', regex=False)

print(f" Token ablation: {len(token_ablation)} rows")
print(f" Functional analysis: {len(functional_df)} rows")

# ============================================================================
# CALCULATE Q-SCORES FOR ALL MODELS
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING Q-SCORES")
print("=" * 80)

# Filter for GSM8K
gsm8k_functional = functional_df[functional_df['dataset'] == 'gsm8k'].copy()

# Calculate base scores
def calculate_base_scores(row):
    # Diversity (lower loss = better)
    diversity_loss = abs(row['diversity_change_pct']) / 100
    diversity_score = 1 - diversity_loss
    
    # Regulation (lower usage = better)
    regulation_score = 1 - (row['token_usage_pct'] / 100)
    
    return diversity_score, regulation_score

gsm8k_functional[['diversity_raw', 'regulation_raw']] = gsm8k_functional.apply(
    calculate_base_scores, axis=1, result_type='expand'
)

# Decomposition
tokens_per_step = gsm8k_functional['tokens_per_step']
gsm8k_functional['decomposition_raw'] = 1 - ((tokens_per_step - tokens_per_step.min()) / 
                                              (tokens_per_step.max() - tokens_per_step.min()))

# Min-max normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

gsm8k_functional[['diversity_score', 'regulation_score', 'decomposition_score']] = scaler.fit_transform(
    gsm8k_functional[['diversity_raw', 'regulation_raw', 'decomposition_raw']]
)

# Quality Control (consolidate diversity + regulation)
gsm8k_functional['quality_control'] = (
    0.5 * gsm8k_functional['diversity_score'] + 
    0.5 * gsm8k_functional['regulation_score']
)

# Final Q-score
gsm8k_functional['Q_score'] = (
    0.7 * gsm8k_functional['quality_control'] +
    0.3 * gsm8k_functional['decomposition_score']
)

print("\n Q-scores calculated for all models:")
print(gsm8k_functional[['model', 'Q_score']].sort_values('Q_score', ascending=False).to_string(index=False))

# ============================================================================
# EFFICIENCY CALCULATION WITH CORRECTED METHODOLOGY
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING EFFICIENCY (CORRECTED METHODOLOGY)")
print("=" * 80)

def calculate_efficiency_corrected(df, model_name, q_score, dataset='gsm8k'):
    """
    Calculate efficiency with methodology from Supplementary Note S5:
    - Well-trained models (Q ≥ 0.58): Use peak performance
    - Poorly-trained models (Q < 0.58): Use 4000-token performance to show inefficiency
    
    NOTE: The threshold of 0.58 corresponds to Tier 3/4 boundary.
          Models below this are inefficient and use compensatory verbosity.
    """
    model_data = df[(df['model'] == model_name) & (df['dataset'] == dataset)].copy()
    
    if len(model_data) == 0:
        return None
    
    model_data = model_data.sort_values('token_limit')
    
    # Find peak
    peak_acc = model_data['accuracy'].max()
    peak_row = model_data[model_data['accuracy'] == peak_acc].iloc[0]
    
    # CORRECTED THRESHOLD: Use 0.58 instead of 0.38
    # This captures Tier 4 (Poor) and Tier 5 (Insufficient)
    if q_score < 0.58:  # Poor or Insufficient training
        limit_4000 = model_data[model_data['token_limit'] == 4000]
        if len(limit_4000) > 0:
            use_row = limit_4000.iloc[0]
            method = "4000-token (poor training)"
        else:
            # If no 4000 limit, use highest available
            use_row = model_data.iloc[-1]
            method = f"{int(use_row['token_limit'])}-token (poor training)"
    else:
        # Well-trained or moderate: Use peak
        use_row = peak_row
        method = f"peak at {int(peak_row['token_limit'])}"
    
    efficiency = use_row['accuracy'] / use_row['avg_tokens'] if use_row['avg_tokens'] > 0 else 0
    
    return {
        'model': model_name,
        'Q_score': q_score,
        'tier': 'Excellent/Good/Moderate' if q_score >= 0.58 else 'Poor/Insufficient',
        'method': method,
        'token_limit_used': int(use_row['token_limit']),
        'accuracy_used': use_row['accuracy'],
        'avg_tokens_used': use_row['avg_tokens'],
        'efficiency_ratio': efficiency,
        'peak_accuracy': peak_acc,
        'peak_token_limit': int(peak_row['token_limit']),
        'peak_avg_tokens': peak_row['avg_tokens']
    }

# Calculate for ALL models with Q-scores
all_results = []

for idx, row in gsm8k_functional.iterrows():
    model = row['model']
    q_score = row['Q_score']
    
    result = calculate_efficiency_corrected(token_ablation, model, q_score, 'gsm8k')
    if result:
        all_results.append(result)

results_df = pd.DataFrame(all_results)

print("\n Complete Efficiency Analysis (ALL 13 MODELS):")
print(results_df[['model', 'Q_score', 'method', 'accuracy_used', 'avg_tokens_used', 'efficiency_ratio']]\
      .sort_values('efficiency_ratio', ascending=False)\
      .to_string(index=False))

# ============================================================================
# ASSIGN PARAMETER SCALES
# ============================================================================

model_params = {
    'SkyworkSkywork-o1-Open-Llama-3.1-8B': 8.0,
    'AIDC-AIMarco-o1_7b': 7.0,
    'nvidiaOpenReasoning-Nemotron-7B': 7.0,
    'deepseek-aiDeepSeek-R1-Distill-Qwen-7B': 7.0,
    'deepseek-aiDeepSeek-R1-Distill-Qwen-14B': 14.0,
    'deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B': 1.5,
    'QwenQwQ-32B-Preview': 32.0,
    'qwenqwen3-32b_o1': 32.0,
    'deepseek-aiDeepSeek-R1-Distill-Llama-70B': 70.0,
    'deepseek-r1-685b': 685.0,
    'glm_4.5': 10.0,
    'glm_4.7_355 billion_32b': 32.0,
    'kimi2.5_1t_32b_sctive': 32.0
}

results_df['params_b'] = results_df['model'].map(model_params)

# ============================================================================
# MATCHED-SCALE COMPARISON (7-8B) → 18.8× CALCULATION
# ============================================================================

print("\n" + "=" * 80)
print("MATCHED-SCALE COMPARISON (7-8B MODELS)")
print("=" * 80)

matched_scale = results_df[
    (results_df['params_b'] >= 6.5) & 
    (results_df['params_b'] <= 8.5)
].copy()

print("\n Models at 7-8B scale:")
print(matched_scale[['model', 'params_b', 'Q_score', 'method', 'accuracy_used', 
                     'avg_tokens_used', 'efficiency_ratio']]\
      .sort_values('efficiency_ratio', ascending=False)\
      .to_string(index=False))

# Find best and worst at matched scale
best = matched_scale.loc[matched_scale['efficiency_ratio'].idxmax()]
worst = matched_scale.loc[matched_scale['efficiency_ratio'].idxmin()]

improvement = best['efficiency_ratio'] / worst['efficiency_ratio']

print("\n" + "=" * 80)
print(" THE 18.8× CALCULATION")
print("=" * 80)

print(f"\n BEST MODEL (Excellent Training):")
print(f"   Model: {best['model']}")
print(f"   Parameters: {best['params_b']:.1f}B")
print(f"   Q-score: {best['Q_score']:.3f}")
print(f"   Method: {best['method']}")
print(f"   Accuracy: {best['accuracy_used']*100:.1f}%")
print(f"   Avg Tokens: {best['avg_tokens_used']:.2f}")
print(f"   Efficiency (η): {best['efficiency_ratio']:.6f}")

print(f"\n WORST MODEL (Poor Training):")
print(f"   Model: {worst['model']}")
print(f"   Parameters: {worst['params_b']:.1f}B")
print(f"   Q-score: {worst['Q_score']:.3f}")
print(f"   Method: {worst['method']}")
print(f"   Accuracy: {worst['accuracy_used']*100:.1f}%")
print(f"   Avg Tokens: {worst['avg_tokens_used']:.2f}")
print(f"   Efficiency (η): {worst['efficiency_ratio']:.6f}")

print(f"\n EFFICIENCY IMPROVEMENT:")
print(f"   {best['efficiency_ratio']:.6f} / {worst['efficiency_ratio']:.6f}")
print(f"   = {improvement:.1f}×")

if abs(improvement - 18.8) < 1.0:
    print(f"\n    MATCHES REPORTED 18.8× IMPROVEMENT!")
else:
    print(f"\n    Calculated: {improvement:.1f}×")
    print(f"    Reported: 18.8×")
    print(f"   Difference: {abs(improvement - 18.8):.1f}×")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("VERIFICATION AGAINST SUPPLEMENTARY NOTE S5")
print("=" * 80)

# Find Skywork and Marco in results
skywork = results_df[results_df['model'] == 'SkyworkSkywork-o1-Open-Llama-3.1-8B'].iloc[0]
marco = results_df[results_df['model'] == 'AIDC-AIMarco-o1_7b'].iloc[0]

print("\nSkywork-o1-8B:")
print(f"  Expected: 0.86 / 292.84 = 0.002937")
print(f"  Calculated: {skywork['accuracy_used']:.2f} / {skywork['avg_tokens_used']:.2f} = {skywork['efficiency_ratio']:.6f}")
print(f"  {' Match!' if abs(skywork['efficiency_ratio'] - 0.002937) < 0.000001 else ' Close'}")

print("\nMarco-o1-7B:")
print(f"  Expected: 0.54 / 3464.80 = 0.000156")
print(f"  Calculated: {marco['accuracy_used']:.2f} / {marco['avg_tokens_used']:.2f} = {marco['efficiency_ratio']:.6f}")
print(f"  {' Match!' if abs(marco['efficiency_ratio'] - 0.000156) < 0.000001 else ' Close'}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION: Q-SCORE vs EFFICIENCY")
print("=" * 80)

valid_data = results_df.dropna(subset=['Q_score', 'efficiency_ratio'])

if len(valid_data) >= 3:
    r_pearson, p_pearson = pearsonr(valid_data['Q_score'], valid_data['efficiency_ratio'])
    r_spearman, p_spearman = spearmanr(valid_data['Q_score'], valid_data['efficiency_ratio'])
    
    print(f"\n Pearson: r = {r_pearson:+.3f}, p = {p_pearson:.4f}")
    print(f" Spearman: ρ = {r_spearman:+.3f}, p = {p_spearman:.4f}")
    
    if r_pearson > 0.7 and p_pearson < 0.05:
        print(f"\n    STRONG positive correlation!")
    elif r_pearson > 0.4 and p_pearson < 0.05:
        print(f"\n    Moderate positive correlation")
    else:
        print(f"\n    Weak correlation")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_df.to_csv('complete_efficiency_all_13_models.csv', index=False)
matched_scale.to_csv('matched_scale_7_8B_final.csv', index=False)

print("\n Saved:")
print("  • complete_efficiency_all_13_models.csv")
print("  • matched_scale_7_8B_final.csv")

print("\n" + "=" * 80)