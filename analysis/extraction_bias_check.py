
"""
Extraction Bias Checker
=======================
Checks whether answer extraction accuracy differs between 
small and large models — the reviewer attack surface from 
ICML Reviewer nex6.

If extraction fails more on large models (they answer correctly 
but in unexpected formats), the entire inverse scaling finding 
could be an artifact.

USAGE:
    python check_extraction_bias.py --base_dir "E:\publication\inverse_scaling\Less _is _More\cross_model_archaeology"

Expected directory structure:
    base_dir/
    ├── Model1_model/
    │   ├── gsm8k/
    │   │   ├── gsm8k_300_results.csv
    │   │   ├── gsm8k_500_results.csv
    │   │   └── ...
    │   ├── boolq/
    │   └── arc-easy/
    └── Model2_model/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ================================================================
# MODEL SIZE LOOKUP (update with your actual model names/sizes)
# ================================================================
MODEL_SIZES = {
    # Phase 1: Standard models (32 models)
    "Qwen2.5-0.5B":        0.5,
    "Gemma-3-1B":           1.0,
    "Llama-3.2-1B":         1.0,
    "StableLM-2-1.6B":      1.6,
    "Gemma-2-2B":           2.0,
    "Llama-3.2-3B":         3.0,
    "StableLM-Zephyr-3B":   3.0,
    "Qwen2.5-3B":           3.0,
    "Phi-3-Mini":           3.8,
    "Phi-3.5-Mini":         3.8,
    "Minitron-4B-Width":    4.0,
    "Minitron-4B-Depth":    4.0,
    "Yi-1.5-6B":            6.0,
    "Mistral-7B":           7.0,
    "DeepSeek-LLM-7B":      7.0,
    "Qwen2.5-7B":           7.0,
    "Llama-3.1-8B":         8.0,
    "Nemotron-Nano-8B":     8.0,
    "Gemma-2-9B":           9.0,
    "Llama-2-13B":          13.0,
    "Qwen2.5-14B":          14.0,
    "GPT-OSS-20B":          20.0,
    "Mistral-Small-24B":    24.0,
    "Kimi-K2-32B":          32.0,
    "Qwen2.5-32B":          32.0,
    "Gemini-2.0-Flash":     50.0,
    "DeepSeek-LLM-67B":     67.0,
    "Llama-3-70B":          70.0,
    "Llama-3-70B-Inst":     70.0,
    "Llama-3.3-70B":        70.0,
    "Llama-3.1-405B":       405.0,
    
    # Phase 3: Reasoning models (13 models)
    "DS-R1-1.5B":           1.5,
    "DR1-Qwen-1.5B":        1.5,
    "Marco-o1-7B":          7.0,
    "DR1-Qwen-7B":          7.0,
    "DS-R1-7B":             7.0,
    "Nemotron-7B":          7.0,
    "Skywork-o1-8B":        8.0,
    "GLM-4.5":              9.0,
    "DR1-Qwen-14B":         14.0,
    "DS-R1-14B":            14.0,
    "QwQ-32B":              32.0,
    "QwenQwQ-32B":          32.0,
    "Qwen3-32B-o1":         32.0,
    "GLM-4.7":              32.0,
    "Kimi-2.5":             32.0,
    "Kimi2.5-1T":           32.0,
    "DR1-Llama-70B":        70.0,
    "DS-R1-70B":            70.0,
    "DR1-685B":             685.0,
    "DS-R1-685B":           685.0,
    "DeepSeek-R1-685B":     685.0,
}


def guess_model_size(model_name):
    """Try to extract model size from name or lookup table."""
    # Direct lookup
    for key, size in MODEL_SIZES.items():
        if key.lower() in model_name.lower():
            return size
    
    # Try to extract number + B pattern
    import re
    patterns = [
        r'(\d+\.?\d*)B',    # e.g., "7B", "1.5B"
        r'(\d+)b',          # lowercase
        r'-(\d+)-',          # e.g., "-70-"
    ]
    for pat in patterns:
        match = re.search(pat, model_name, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return None


def classify_extraction(row):
    """
    Classify each response into one of:
    - 'extracted_correct': answer extracted AND correct
    - 'extracted_wrong':   answer extracted AND incorrect  
    - 'extraction_failed': no answer could be extracted
    - 'no_generation':     model produced no output
    """
    generation = str(row.get('full_generation', ''))
    extracted = row.get('extracted_answer', None)
    is_correct = row.get('is_correct', None)
    
    # No generation at all
    if generation in ('', 'None', 'nan', 'NaN') or pd.isna(generation):
        return 'no_generation'
    
    # Extraction failed (model generated but we couldn't parse answer)
    if extracted is None or str(extracted) in ('', 'None', 'nan', 'NaN', 
                                                'EXTRACTION_FAILED', 
                                                'NO_ANSWER_FOUND'):
        return 'extraction_failed'
    if pd.isna(extracted):
        return 'extraction_failed'
    
    # Extracted successfully
    if is_correct is True or str(is_correct).lower() == 'true':
        return 'extracted_correct'
    else:
        return 'extracted_wrong'


def load_results(base_dir):
    """Walk directory structure and load all CSV results."""
    all_rows = []
    base_path = Path(base_dir)
    
    csv_count = 0
    for csv_file in base_path.rglob("responses_*.csv"):
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Try to get model name from CSV or directory
            if 'model_name' not in df.columns:
                # Extract from directory path
                parts = csv_file.relative_to(base_path).parts
                df['model_name'] = parts[0] if len(parts) > 0 else 'unknown'
            
            # Try to get dataset from CSV or directory
            if 'dataset_name' not in df.columns:
                parts = csv_file.relative_to(base_path).parts
                df['dataset_name'] = parts[1] if len(parts) > 1 else 'unknown'
            
            # Get token limit from filename
            fname = csv_file.stem  # e.g., "gsm8k_300_results"
            import re
            token_match = re.search(r'_(\d+)_results', fname)
            if token_match:
                df['token_limit'] = int(token_match.group(1))
            
            df['source_file'] = str(csv_file)
            all_rows.append(df)
            csv_count += 1
            
        except Exception as e:
            print(f"  Warning: Could not read {csv_file}: {e}")
    
    if csv_count == 0:
        print(f"ERROR: No *_results.csv files found in {base_dir}")
        print(f"Expected structure: base_dir/ModelName/dataset/dataset_NNN_results.csv")
        sys.exit(1)
    
    print(f"Loaded {csv_count} CSV files")
    combined = pd.concat(all_rows, ignore_index=True)
    print(f"Total rows: {len(combined)}")
    return combined


def analyze_extraction_bias(df):
    """Main analysis: check extraction rates by model size."""
    
    # ── 1. Classify each response ────────────────────────────
    df['extraction_status'] = df.apply(classify_extraction, axis=1)
    
    # ── 2. Get model sizes ───────────────────────────────────
    df['model_size_B'] = df['model_name'].apply(guess_model_size)
    
    unknown = df[df['model_size_B'].isna()]['model_name'].unique()
    if len(unknown) > 0:
        print(f"\nWARNING: Could not determine size for {len(unknown)} models:")
        for m in unknown[:10]:
            print(f"  - {m}")
        print("  → Add these to MODEL_SIZES dict in the script")
    
    # Drop rows with unknown size
    df_sized = df[df['model_size_B'].notna()].copy()
    
    # ── 3. Per-model extraction stats ────────────────────────
    print("\n" + "=" * 80)
    print("PER-MODEL EXTRACTION STATISTICS")
    print("=" * 80)
    
    model_stats = []
    for model in sorted(df_sized['model_name'].unique()):
        mdf = df_sized[df_sized['model_name'] == model]
        size = mdf['model_size_B'].iloc[0]
        total = len(mdf)
        
        n_correct = (mdf['extraction_status'] == 'extracted_correct').sum()
        n_wrong = (mdf['extraction_status'] == 'extracted_wrong').sum()
        n_failed = (mdf['extraction_status'] == 'extraction_failed').sum()
        n_no_gen = (mdf['extraction_status'] == 'no_generation').sum()
        
        n_extracted = n_correct + n_wrong
        extraction_rate = n_extracted / total * 100 if total > 0 else 0
        failure_rate = n_failed / total * 100 if total > 0 else 0
        
        model_stats.append({
            'model': model,
            'size_B': size,
            'total': total,
            'extracted': n_extracted,
            'extraction_rate': extraction_rate,
            'failed': n_failed,
            'failure_rate': failure_rate,
            'no_generation': n_no_gen,
            'accuracy': n_correct / total * 100 if total > 0 else 0,
        })
    
    stats_df = pd.DataFrame(model_stats).sort_values('size_B')
    
    print(f"\n{'Model':<30s} {'Size':>6s} {'Total':>6s} {'Extract%':>9s} "
          f"{'Fail%':>7s} {'NoGen':>6s} {'Acc%':>7s}")
    print("-" * 80)
    for _, r in stats_df.iterrows():
        print(f"{r['model']:<30s} {r['size_B']:>5.1f}B {r['total']:>6d} "
              f"{r['extraction_rate']:>8.1f}% {r['failure_rate']:>6.1f}% "
              f"{r['no_generation']:>5d} {r['accuracy']:>6.1f}%")
    
    # ── 4. Grouped by size category ──────────────────────────
    print("\n" + "=" * 80)
    print("EXTRACTION RATE BY SIZE CATEGORY")
    print("=" * 80)
    
    def size_category(size):
        if size <= 3:
            return 'Tiny (≤3B)'
        elif size <= 10:
            return 'Small (3-10B)'
        elif size <= 35:
            return 'Medium (10-35B)'
        elif size <= 100:
            return 'Large (35-100B)'
        else:
            return 'Giant (>100B)'
    
    df_sized['size_cat'] = df_sized['model_size_B'].apply(size_category)
    
    cat_order = ['Tiny (≤3B)', 'Small (3-10B)', 'Medium (10-35B)', 
                 'Large (35-100B)', 'Giant (>100B)']
    
    print(f"\n{'Category':<20s} {'N models':>9s} {'N rows':>8s} "
          f"{'Extract%':>10s} {'Fail%':>8s} {'Acc%':>8s}")
    print("-" * 65)
    
    for cat in cat_order:
        cdf = df_sized[df_sized['size_cat'] == cat]
        if len(cdf) == 0:
            continue
        n_models = cdf['model_name'].nunique()
        total = len(cdf)
        n_extracted = ((cdf['extraction_status'] == 'extracted_correct') | 
                       (cdf['extraction_status'] == 'extracted_wrong')).sum()
        n_failed = (cdf['extraction_status'] == 'extraction_failed').sum()
        n_correct = (cdf['extraction_status'] == 'extracted_correct').sum()
        
        ext_rate = n_extracted / total * 100
        fail_rate = n_failed / total * 100
        acc = n_correct / total * 100
        
        print(f"{cat:<20s} {n_models:>8d} {total:>8d} "
              f"{ext_rate:>9.1f}% {fail_rate:>7.1f}% {acc:>7.1f}%")
    
    # ── 5. Statistical test ──────────────────────────────────
    print("\n" + "=" * 80)
    print("STATISTICAL TEST: EXTRACTION BIAS")
    print("=" * 80)
    
    # Small (≤10B) vs Large (≥70B) — matching the paper's categories
    small_mask = df_sized['model_size_B'] <= 10
    large_mask = df_sized['model_size_B'] >= 70
    
    small_df = df_sized[small_mask]
    large_df = df_sized[large_mask]
    
    if len(small_df) > 0 and len(large_df) > 0:
        small_extracted = ((small_df['extraction_status'] == 'extracted_correct') | 
                          (small_df['extraction_status'] == 'extracted_wrong'))
        large_extracted = ((large_df['extraction_status'] == 'extracted_correct') | 
                          (large_df['extraction_status'] == 'extracted_wrong'))
        
        small_rate = small_extracted.mean() * 100
        large_rate = large_extracted.mean() * 100
        
        small_fail = (small_df['extraction_status'] == 'extraction_failed').mean() * 100
        large_fail = (large_df['extraction_status'] == 'extraction_failed').mean() * 100
        
        print(f"\nSmall models (≤10B):  extraction rate = {small_rate:.1f}%, "
              f"failure rate = {small_fail:.1f}%  (n={len(small_df)})")
        print(f"Large models (≥70B):  extraction rate = {large_rate:.1f}%, "
              f"failure rate = {large_fail:.1f}%  (n={len(large_df)})")
        print(f"Difference: {large_rate - small_rate:+.1f}pp")
        
        # Chi-squared test
        from scipy.stats import chi2_contingency
        
        small_success = small_extracted.sum()
        small_total = len(small_df)
        large_success = large_extracted.sum()
        large_total = len(large_df)
        
        contingency = [
            [small_success, small_total - small_success],
            [large_success, large_total - large_success]
        ]
        
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
            print(f"\nChi-squared test: χ² = {chi2:.2f}, P = {p:.4f}")
            
            if p < 0.05:
                print("⚠️  SIGNIFICANT difference in extraction rates!")
                print("    This means extraction bias may confound results.")
                print("    ACTION: Investigate which answer formats are being missed.")
            else:
                print("✓  No significant difference in extraction rates.")
                print("    Extraction bias is NOT confounding the results.")
        except Exception as e:
            print(f"  Could not run chi-squared: {e}")
    
    # ── 6. Per-dataset breakdown ─────────────────────────────
    print("\n" + "=" * 80)
    print("PER-DATASET EXTRACTION RATES (Small ≤10B vs Large ≥70B)")
    print("=" * 80)
    
    for dataset in sorted(df_sized['dataset_name'].dropna().unique()):
        ddf = df_sized[df_sized['dataset_name'] == dataset]
        
        s = ddf[ddf['model_size_B'] <= 10]
        l = ddf[ddf['model_size_B'] >= 70]
        
        if len(s) == 0 or len(l) == 0:
            continue
        
        s_rate = ((s['extraction_status'] == 'extracted_correct') | 
                  (s['extraction_status'] == 'extracted_wrong')).mean() * 100
        l_rate = ((l['extraction_status'] == 'extracted_correct') | 
                  (l['extraction_status'] == 'extracted_wrong')).mean() * 100
        
        s_fail = (s['extraction_status'] == 'extraction_failed').mean() * 100
        l_fail = (l['extraction_status'] == 'extraction_failed').mean() * 100
        
        diff = l_rate - s_rate
        flag = " ⚠️" if abs(diff) > 5 else " ✓"
        
        print(f"\n  {dataset}:")
        print(f"    Small (≤10B): {s_rate:5.1f}% extracted, "
              f"{s_fail:5.1f}% failed  (n={len(s)})")
        print(f"    Large (≥70B): {l_rate:5.1f}% extracted, "
              f"{l_fail:5.1f}% failed  (n={len(l)})")
        print(f"    Diff: {diff:+.1f}pp{flag}")
    
    # ── 7. Worst-case models ─────────────────────────────────
    print("\n" + "=" * 80)
    print("MODELS WITH HIGHEST EXTRACTION FAILURE RATES")
    print("=" * 80)
    
    worst = stats_df.nlargest(10, 'failure_rate')
    for _, r in worst.iterrows():
        if r['failure_rate'] > 0:
            print(f"  {r['model']:<30s} ({r['size_B']:.0f}B): "
                  f"{r['failure_rate']:.1f}% failures "
                  f"({r['failed']:.0f}/{r['total']:.0f})")
    
    # ── 8. Pearson correlation: size vs extraction rate ───────
    print("\n" + "=" * 80)
    print("CORRELATION: Model Size vs Extraction Rate")
    print("=" * 80)
    
    if len(stats_df) > 3:
        from scipy.stats import pearsonr, spearmanr
        sizes = stats_df['size_B'].values
        rates = stats_df['extraction_rate'].values
        
        r_p, p_p = pearsonr(np.log10(sizes + 1), rates)
        r_s, p_s = spearmanr(sizes, rates)
        
        print(f"  Pearson (log-size vs rate):  r = {r_p:+.3f}, P = {p_p:.4f}")
        print(f"  Spearman (size vs rate):     ρ = {r_s:+.3f}, P = {p_s:.4f}")
        
        if p_p < 0.05 or p_s < 0.05:
            print("  ⚠️  Significant correlation — larger models have "
                  "different extraction rates!")
        else:
            print("  ✓  No significant correlation — extraction is "
                  "unbiased across scales.")
    
    # ── 9. Summary verdict ───────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    max_fail = stats_df['failure_rate'].max()
    mean_fail = stats_df['failure_rate'].mean()
    
    if mean_fail < 2 and max_fail < 5:
        print("  ✓ CLEAN: Extraction failure rate is uniformly low.")
        print(f"    Mean: {mean_fail:.1f}%, Max: {max_fail:.1f}%")
        print("    Safe to state in rebuttal: 'Extraction accuracy")
        print("    is comparable across model scales.'")
    elif mean_fail < 5:
        print("  ~ ACCEPTABLE: Low extraction failures overall,")
        print(f"    but check outlier models (max: {max_fail:.1f}%).")
    else:
        print(f"  ⚠️ CONCERN: Mean failure rate = {mean_fail:.1f}%.")
        print("    Investigate answer format differences by model size.")
    
    return stats_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check extraction bias across model sizes"
    )
    parser.add_argument(
        '--base_dir', type=str, required=True,
        help='Path to results directory with Model/dataset/csv structure'
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXTRACTION BIAS CHECKER")
    print("Reviewer defense: Does answer extraction differ by model size?")
    print("=" * 80)
    
    df = load_results(args.base_dir)
    stats = analyze_extraction_bias(df)
    
    print("\nDone. Save this output for rebuttal evidence.")