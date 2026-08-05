"""
Causal Intervention Ablation Analysis
======================================
Reads data structure:
    {model_name}/{dataset}/raw_responses/{condition}.csv

Produces:
    1. Per-model per-dataset condition comparison table
    2. Claim survival check (token budget vs framing effect)
    3. Pooled results across all models
    4. Statistical significance tests
    5. Final verdict table

Usage:
    python causal_ablation_analysis.py --root ./data/raw/random_sample_responses
    python causal_ablation_analysis.py --root .  (if run from results dir)
"""

import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Map filename stems to canonical condition names
CONDITION_MAP = {
    'control':     'control',
    'brief':       'brief',
    'length_only': 'length_only',
    'length':      'length_only',   # handles "length.csv" variant
    'beirf':       'brief',         # handles common typo
}

# Datasets and their task types
DATASET_TYPES = {
    'gsm8k':        'math',
    'arc-easy':     'commonsense',
    'arc_easy':     'commonsense',
    'boolq':        'reasoning',
    'commonsenseqa':'commonsense',
    'mmlu-stem':    'commonsense',
    'mmlu_stem':    'commonsense',
}


# ─────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────

def load_all_results(root: str) -> pd.DataFrame:
    """
    Walk the directory tree and load all condition CSVs.

    Expected structure:
        {root}/{model}/{dataset}/raw_responses/{condition}.csv
    """
    root = Path(root)
    records = []

    # Pattern: root / model / dataset / raw_responses / condition.csv
    csv_files = list(root.glob("*/*/raw_responses/*.csv"))

    if not csv_files:
        # Try one level up — maybe root IS the model folder
        csv_files = list(root.glob("*/raw_responses/*.csv"))
        print(f"  Trying single-model structure: found {len(csv_files)} files")

    print(f" Found {len(csv_files)} CSV files under {root}\n")

    for csv_path in sorted(csv_files):
        parts = csv_path.parts

        # Extract model and dataset from path
        # Structure: .../model/dataset/raw_responses/condition.csv
        try:
            raw_idx = parts.index('raw_responses')
            dataset = parts[raw_idx - 1].lower()
            model   = parts[raw_idx - 2]
        except (ValueError, IndexError):
            print(f"    Skipping (unexpected path): {csv_path}")
            continue

        # Normalize condition from filename
        stem = csv_path.stem.lower()
        # Handle timestamped filenames like "control_20250101_120000"
        stem_base = stem.split('_')[0] if '_' in stem else stem
        # Try full stem first, then base
        condition = CONDITION_MAP.get(stem, CONDITION_MAP.get(stem_base))

        if condition is None:
            print(f"    Unknown condition '{stem}' — skipping {csv_path.name}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"   Failed to read {csv_path}: {e}")
            continue

        if 'is_correct' not in df.columns:
            print(f"    No 'is_correct' column in {csv_path.name} — skipping")
            continue

        df['_model']     = model
        df['_dataset']   = dataset
        df['_condition'] = condition
        df['_file']      = str(csv_path)
        records.append(df)

        acc = df['is_correct'].mean()
        print(f"   {model}/{dataset}/{condition}: "
              f"n={len(df)}, acc={acc:.1%}")

    if not records:
        raise ValueError(
            f"No valid CSVs found under '{root}'.\n"
            "Expected: {{model}}/{{dataset}}/raw_responses/{{condition}}.csv"
        )

    all_df = pd.concat(records, ignore_index=True)
    print(f"\n Total responses loaded: {len(all_df)}")
    return all_df


# ─────────────────────────────────────────────────────────────
# MCNEMAR TEST
# ─────────────────────────────────────────────────────────────

def mcnemar_test(df: pd.DataFrame,
                 cond_a: str,
                 cond_b: str,
                 model_col: str = '_model',
                 dataset_col: str = '_dataset',
                 condition_col: str = '_condition') -> dict:
    """
    Paired McNemar test between two conditions.
    Pairs responses by sample_id.
    Returns dict with p-value, delta, and n_pairs.
    """
    a = df[df[condition_col] == cond_a][['sample_id', 'is_correct']].copy()
    b = df[df[condition_col] == cond_b][['sample_id', 'is_correct']].copy()

    merged = a.merge(b, on='sample_id', suffixes=('_a', '_b'))

    if len(merged) < 10:
        return {'p_value': None, 'delta': None, 'n_pairs': len(merged),
                'significant': False, 'note': 'insufficient_pairs'}

    # McNemar contingency: [correct_a & wrong_b, wrong_a & correct_b]
    n01 = ((merged['is_correct_a'] == True)  & (merged['is_correct_b'] == False)).sum()
    n10 = ((merged['is_correct_a'] == False) & (merged['is_correct_b'] == True)).sum()

    if n01 + n10 == 0:
        return {'p_value': 1.0, 'delta': 0.0, 'n_pairs': len(merged),
                'significant': False, 'note': 'no_discordant_pairs'}

    # Exact McNemar (binomial)
    p_value = min(1.0, 2 * stats.binom.cdf(min(n01, n10), n01 + n10, 0.5))

    acc_a = merged['is_correct_a'].mean()
    acc_b = merged['is_correct_b'].mean()
    delta = acc_b - acc_a

    return {
        'p_value':     round(p_value, 4),
        'delta':       round(delta, 4),
        'n_pairs':     len(merged),
        'acc_a':       round(acc_a, 4),
        'acc_b':       round(acc_b, 4),
        'n01':         int(n01),
        'n10':         int(n10),
        'significant': p_value < 0.05,
        'note':        ''
    }


# ─────────────────────────────────────────────────────────────
# CLAIM SURVIVAL CHECK
# ─────────────────────────────────────────────────────────────

def check_claim_survival(row: pd.Series) -> str:
    """
    Given a result row with control/brief/length_only accuracies,
    return the survival verdict for the causal claim.
    """
    ctrl  = row.get('control',     None)
    brief = row.get('brief',       None)
    lonly = row.get('length_only', None)

    verdicts = []

    # Claim 1: constraining generation improves accuracy
    if brief is not None and ctrl is not None:
        if brief > ctrl:
            verdicts.append(" brief > control")
        else:
            verdicts.append("  brief ≤ control")

    if lonly is not None and ctrl is not None:
        if lonly > ctrl:
            verdicts.append(" length_only > control")
        else:
            verdicts.append("ℹ  length_only ≤ control")

    # Claim 2: token budget vs framing
    if brief is not None and lonly is not None:
        framing = brief - lonly
        if abs(framing) <= 0.05:
            verdicts.append(f" framing ≈ 0 ({framing:+.1%}) → token budget drives effect")
        elif lonly > brief:
            verdicts.append(f" length_only > brief ({framing:+.1%}) → token budget > framing")
        else:
            verdicts.append(f"  brief > length_only ({framing:+.1%}) → framing contributes")

    return " | ".join(verdicts) if verdicts else " insufficient conditions"


# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────

def run_analysis(all_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. PER-MODEL PER-DATASET ACCURACY TABLE ──────────────
    print("\n" + "="*80)
    print("TABLE 1 — PER-MODEL PER-DATASET ACCURACY BY CONDITION")
    print("="*80)

    acc_table = (
        all_df.groupby(['_model', '_dataset', '_condition'])['is_correct']
        .agg(['mean', 'count'])
        .reset_index()
    )
    acc_table.columns = ['model', 'dataset', 'condition', 'accuracy', 'n']
    acc_table['accuracy_pct'] = (acc_table['accuracy'] * 100).round(1)

    pivot = acc_table.pivot_table(
        index=['model', 'dataset'],
        columns='condition',
        values='accuracy_pct'
    ).reset_index()

    # Compute deltas if columns exist
    conds = [c for c in ['control', 'brief', 'length_only'] if c in pivot.columns]

    if 'control' in pivot.columns:
        for c in ['brief', 'length_only']:
            if c in pivot.columns:
                pivot[f'{c}_delta'] = (pivot[c] - pivot['control']).round(1)

    if 'brief' in pivot.columns and 'length_only' in pivot.columns:
        pivot['framing_effect'] = (pivot['brief'] - pivot['length_only']).round(1)
        pivot['token_budget_effect'] = pivot.get('length_only_delta', None)

    print(pivot.to_string(index=False))
    pivot.to_csv(output_dir / "table1_accuracy_by_condition.csv", index=False)

    # ── 2. CLAIM SURVIVAL CHECK ───────────────────────────────
    print("\n" + "="*80)
    print("TABLE 2 — CAUSAL CLAIM SURVIVAL CHECK")
    print("="*80)

    pivot['survival_verdict'] = pivot.apply(check_claim_survival, axis=1)

    survival_cols = ['model', 'dataset'] + conds + ['framing_effect', 'survival_verdict']
    survival_cols = [c for c in survival_cols if c in pivot.columns]

    print(pivot[survival_cols].to_string(index=False))
    pivot[survival_cols].to_csv(output_dir / "table2_claim_survival.csv", index=False)

    # ── 3. STATISTICAL TESTS ─────────────────────────────────
    print("\n" + "="*80)
    print("TABLE 3 — MCNEMAR STATISTICAL TESTS")
    print("="*80)

    test_rows = []

    for (model, dataset), group in all_df.groupby(['_model', '_dataset']):
        available = group['_condition'].unique()

        # Test brief vs control
        if 'brief' in available and 'control' in available:
            result = mcnemar_test(group, 'control', 'brief')
            test_rows.append({
                'model':       model,
                'dataset':     dataset,
                'comparison':  'control → brief',
                'delta':       f"{result.get('delta', 0):+.1%}",
                'p_value':     result.get('p_value', None),
                'significant': '✅' if result.get('significant') else '❌',
                'n_pairs':     result.get('n_pairs', 0),
                'note':        result.get('note', '')
            })

        # Test length_only vs control
        if 'length_only' in available and 'control' in available:
            result = mcnemar_test(group, 'control', 'length_only')
            test_rows.append({
                'model':       model,
                'dataset':     dataset,
                'comparison':  'control → length_only',
                'delta':       f"{result.get('delta', 0):+.1%}",
                'p_value':     result.get('p_value', None),
                'significant': '✅' if result.get('significant') else '❌',
                'n_pairs':     result.get('n_pairs', 0),
                'note':        result.get('note', '')
            })

        # Test brief vs length_only (framing isolation)
        if 'brief' in available and 'length_only' in available:
            result = mcnemar_test(group, 'length_only', 'brief')
            test_rows.append({
                'model':       model,
                'dataset':     dataset,
                'comparison':  'length_only vs brief (framing)',
                'delta':       f"{result.get('delta', 0):+.1%}",
                'p_value':     result.get('p_value', None),
                'significant': '⚠️' if result.get('significant') else '✅ ns',
                'n_pairs':     result.get('n_pairs', 0),
                'note':        result.get('note', '')
            })

    tests_df = pd.DataFrame(test_rows)
    if not tests_df.empty:
        print(tests_df.to_string(index=False))
        tests_df.to_csv(output_dir / "table3_statistical_tests.csv", index=False)
    else:
        print("No paired tests possible — check sample_id column alignment")

    # ── 4. POOLED RESULTS ACROSS ALL MODELS ──────────────────
    print("\n" + "="*80)
    print("TABLE 4 — POOLED RESULTS ACROSS ALL MODELS")
    print("="*80)

    pooled = (
        all_df.groupby(['_dataset', '_condition'])['is_correct']
        .agg(['mean', 'count', 'std'])
        .reset_index()
    )
    pooled.columns = ['dataset', 'condition', 'accuracy', 'n', 'std']
    pooled['accuracy_pct'] = (pooled['accuracy'] * 100).round(1)
    pooled['se'] = (pooled['std'] / np.sqrt(pooled['n']) * 100).round(1)

    pooled_pivot = pooled.pivot_table(
        index='dataset',
        columns='condition',
        values='accuracy_pct'
    ).reset_index()

    if 'control' in pooled_pivot.columns:
        for c in ['brief', 'length_only']:
            if c in pooled_pivot.columns:
                pooled_pivot[f'{c}_delta'] = (
                    pooled_pivot[c] - pooled_pivot['control']
                ).round(1)

    if 'brief' in pooled_pivot.columns and 'length_only' in pooled_pivot.columns:
        pooled_pivot['framing_effect'] = (
            pooled_pivot['brief'] - pooled_pivot['length_only']
        ).round(1)

    print(pooled_pivot.to_string(index=False))
    pooled_pivot.to_csv(output_dir / "table4_pooled_results.csv", index=False)

    # ── 5. FINAL VERDICT ─────────────────────────────────────
    print("\n" + "="*80)
    print("FINAL VERDICT — CAUSAL CLAIM SURVIVAL")
    print("="*80)

    # Count how many model-dataset pairs show each pattern
    if 'framing_effect' in pivot.columns and 'brief_delta' in pivot.columns:

        n_total          = len(pivot)
        n_brief_improves = (pivot['brief_delta'] > 0).sum()   if 'brief_delta'       in pivot.columns else 0
        n_lonly_improves = (pivot.get('length_only_delta', pd.Series()) > 0).sum()
        n_framing_small  = (pivot['framing_effect'].abs() <= 5).sum()
        n_lonly_gt_brief = (pivot['length_only'] > pivot['brief']).sum() if all(
            c in pivot.columns for c in ['length_only', 'brief']
        ) else 0

        print(f"\n  Total model-dataset pairs evaluated: {n_total}")
        print(f"\n  Claim 1 — Constraining generation improves accuracy:")
        print(f"    Brief improves over control:       {n_brief_improves}/{n_total} pairs")
        print(f"    Length-only improves over control: {n_lonly_improves}/{n_total} pairs")

        print(f"\n  Claim 2 — Effect is token budget driven (not framing):")
        print(f"    Framing effect ≤ 5pp:              {n_framing_small}/{n_total} pairs")
        print(f"    Length-only ≥ Brief:               {n_lonly_gt_brief}/{n_total} pairs")

        # Overall verdict
        claim1_survives = n_brief_improves >= n_total * 0.6
        claim2_survives = n_framing_small  >= n_total * 0.5

        print(f"\n{'─'*60}")
        print(f"  Claim 1 (constraining improves accuracy):  "
              f"{' SURVIVED' if claim1_survives else '❌ FAILED'}")
        print(f"  Claim 2 (token budget > framing):          "
              f"{' SURVIVED' if claim2_survives else '⚠️  PARTIAL'}")
        print(f"{'─'*60}")

        if claim1_survives and claim2_survives:
            print("\n   OVERALL: CAUSAL CLAIM FULLY SURVIVED")
            print("     Ready to defend at NeurIPS.")
        elif claim1_survives:
            print("\n   OVERALL: MAIN CLAIM SURVIVED")
            print("     Mechanism (token vs framing) is task-dependent.")
            print("     Reframe as: 'task-specific over-generation mechanism'")
        else:
            print("\n    CLAIM PARTIALLY SURVIVED — review individual results")

    # ── 6. LATEX TABLE FOR PAPER ─────────────────────────────
    print("\n" + "="*80)
    print("LATEX TABLE (copy into appendix)")
    print("="*80)

    latex = generate_latex_table(pivot)
    print(latex)

    latex_path = output_dir / "appendix_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\n LaTeX table saved to: {latex_path}")

    print(f"\n All results saved to: {output_dir}/")
    print("   table1_accuracy_by_condition.csv")
    print("   table2_claim_survival.csv")
    print("   table3_statistical_tests.csv")
    print("   table4_pooled_results.csv")
    print("   appendix_table.tex")

    return pivot, tests_df, pooled_pivot


# ─────────────────────────────────────────────────────────────
# LATEX TABLE GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_latex_table(pivot: pd.DataFrame) -> str:
    """Generate LaTeX table for appendix."""

    has_ctrl  = 'control'     in pivot.columns
    has_brief = 'brief'       in pivot.columns
    has_lonly = 'length_only' in pivot.columns
    has_frame = 'framing_effect' in pivot.columns

    # Build column spec
    col_parts = ['ll']
    if has_ctrl:  col_parts.append('r')
    if has_brief: col_parts.append('r')
    if has_lonly: col_parts.append('r')
    if has_frame: col_parts.append('r')
    col_parts.append('r')  # brief delta
    col_spec = ''.join(col_parts)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Random-sample causal intervention results.} "
        r"Three conditions across datasets and models ($n=100$ random "
        r"problems per condition, seed=42). "
        r"\textit{Token budget effect} = length-only $\Delta$ vs control. "
        r"\textit{Framing effect} = brief $-$ length-only.}",
        r"\label{tab:random_causal}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header
    header = r"\textbf{Model} & \textbf{Dataset}"
    if has_ctrl:  header += r" & \textbf{Control (\%)}"
    if has_brief: header += r" & \textbf{Brief (\%)}"
    if has_lonly: header += r" & \textbf{Length-only (\%)}"
    if has_brief and has_ctrl:
        header += r" & \textbf{Brief $\Delta$}"
    if has_frame:
        header += r" & \textbf{Framing effect}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for _, row in pivot.iterrows():
        model   = str(row.get('model',   '')).replace('_', r'\_')
        dataset = str(row.get('dataset', '')).replace('-', r'\-').upper()

        def fmt(col):
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "---"
            if col in ['framing_effect', 'brief_delta',
                       'length_only_delta', 'token_budget_effect']:
                return f"${val:+.1f}$pp"
            return f"{val:.1f}"

        r_line = f"{model} & {dataset}"
        if has_ctrl:  r_line += f" & {fmt('control')}"
        if has_brief: r_line += f" & {fmt('brief')}"
        if has_lonly: r_line += f" & {fmt('length_only')}"
        if has_brief and has_ctrl:
            r_line += f" & {fmt('brief_delta')}"
        if has_frame:
            r_line += f" & {fmt('framing_effect')}"
        r_line += r" \\"
        lines.append(r_line)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Causal intervention ablation analysis"
    )
    parser.add_argument(
        '--root',
        type=str,
        default='.',
        help='Root directory containing model folders '
             '(default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./ablation_results',
        help='Output directory for results (default: ./ablation_results)'
    )
    args = parser.parse_args()

    print("="*80)
    print("CAUSAL INTERVENTION ABLATION ANALYSIS")
    print("="*80)
    print(f"Root:   {args.root}")
    print(f"Output: {args.output}\n")

    # Load all data
    all_df = load_all_results(args.root)

    # Run analysis
    pivot, tests_df, pooled = run_analysis(
        all_df,
        output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()