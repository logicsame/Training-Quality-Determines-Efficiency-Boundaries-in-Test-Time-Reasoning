"""
MULTI-MODEL TOKEN LIMIT ABLATION ANALYSIS
==========================================
Analyzes token limit experiments across multiple models and datasets

DIRECTORY STRUCTURE EXPECTED:
base_dir/
├── Model1_model/
│   ├── gsm8k/
│   │   ├── gsm8k_300_results.csv
│   │   ├── gsm8k_500_results.csv
│   │   ├── gsm8k_1000_results.csv
│   │   └── gsm8k_2000_results.csv
│   ├── boolq/
│   │   └── (same pattern)
│   └── arc-easy/
│       └── (same pattern)
├── Model2_model/
│   └── (same structure)
└── ...

USAGE:
    python reasoning_ablation_analysis_small_models.py --base_dir data/raw/phase3_responses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import norm
import json
import argparse
from typing import Dict, List, Tuple
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = ['gsm8k', 'boolq', 'arc-easy', 'gpqa']
TOKEN_LIMITS = [300, 500, 1000, 2000, 4000]

# Color scheme for visualizations
COLORS = {
    'gsm8k': '#2E86AB',
    'boolq': '#A23B72', 
    'arc-easy': '#F18F01',
    'gpqa': '#06A77D', 
    
}

# ============================================================================
# DATA LOADING
# ============================================================================

def find_model_directories(base_dir: Path) -> List[Path]:
    """Find all model directories (ending with _model)"""
    model_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.endswith('_model'):
            model_dirs.append(item)
    return sorted(model_dirs)

def load_model_data(model_dir: Path) -> Dict:
    """Load all data for a single model"""
    
    model_name = model_dir.name.replace('_model', '')
    
    print(f"\n Loading data for: {model_name}")
    print(f"   Path: {model_dir}")
    
    data = {
        'model_name': model_name,
        'datasets': {}
    }
    
    for dataset in DATASETS:
        data['datasets'][dataset] = {}
        dataset_dir = model_dir / dataset
        
        if not dataset_dir.exists():
            print(f"     Missing dataset directory: {dataset}")
            continue
        
        for limit in TOKEN_LIMITS:
            filepath = dataset_dir / f"{dataset}_{limit}_results.csv"
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                data['datasets'][dataset][limit] = df
                print(f"    {dataset:12s} @ {limit:4d} tokens: {len(df):3d} samples")
            else:
                print(f"    Missing: {dataset}_{limit}_results.csv")
    
    return data

def load_all_models(base_dir: Path) -> List[Dict]:
    """Load data for all models"""
    
    print("="*80)
    print("LOADING ALL MODEL DATA")
    print("="*80)
    
    model_dirs = find_model_directories(base_dir)
    
    if not model_dirs:
        print(f" No model directories found in {base_dir}")
        print(f"   Looking for directories ending with '_model'")
        return []
    
    print(f"\nFound {len(model_dirs)} model(s):")
    for model_dir in model_dirs:
        print(f"  • {model_dir.name}")
    
    all_model_data = []
    for model_dir in model_dirs:
        model_data = load_model_data(model_dir)
        all_model_data.append(model_data)
    
    return all_model_data

# ============================================================================
# METRIC CALCULATION
# ============================================================================

def calculate_metrics(df: pd.DataFrame, token_limit: int) -> Dict:
    """Calculate metrics for a single dataset at one token limit"""
    
    metrics = {
        'accuracy': df['is_correct'].mean(),
        'n_samples': len(df),
        'n_correct': df['is_correct'].sum(),
        'avg_tokens': df['output_tokens'].mean(),
        'median_tokens': df['output_tokens'].median(),
        'max_tokens': df['output_tokens'].max(),
        'min_tokens': df['output_tokens'].min(),
        'std_tokens': df['output_tokens'].std(),
        'hit_limit_count': (df['output_tokens'] >= (token_limit - 10)).sum(),
        'hit_limit_pct': ((df['output_tokens'] >= (token_limit - 10)).sum() / len(df)) * 100
    }
    
    return metrics

def calculate_all_metrics(all_model_data: List[Dict]) -> pd.DataFrame:
    """Calculate metrics for all models and datasets"""
    
    print("\n" + "="*80)
    print("CALCULATING METRICS FOR ALL MODELS")
    print("="*80)
    
    results = []
    
    for model_data in all_model_data:
        model_name = model_data['model_name']
        print(f"\n{model_name}:")
        
        for dataset in DATASETS:
            print(f"\n  {dataset.upper()}:")
            
            if dataset not in model_data['datasets']:
                continue
            
            for limit in TOKEN_LIMITS:
                if limit not in model_data['datasets'][dataset]:
                    continue
                
                df = model_data['datasets'][dataset][limit]
                metrics = calculate_metrics(df, limit)
                
                result = {
                    'model': model_name,
                    'dataset': dataset,
                    'token_limit': limit,
                    **metrics
                }
                
                results.append(result)
                
                print(f"    {limit:4d} tokens: {metrics['accuracy']*100:5.1f}% "
                      f"(n={metrics['n_samples']}, "
                      f"avg={metrics['avg_tokens']:.0f} tokens, "
                      f"hit limit: {metrics['hit_limit_pct']:.0f}%)")
    
    return pd.DataFrame(results)

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def run_statistical_tests(all_model_data: List[Dict]) -> pd.DataFrame:
    """Run statistical tests for all models"""
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    stat_tests = []
    
    for model_data in all_model_data:
        model_name = model_data['model_name']
        print(f"\n{model_name}:")
        
        for dataset in DATASETS:
            print(f"\n  {dataset.upper()}:")
            
            if dataset not in model_data['datasets']:
                continue
            
            # Test each consecutive pair
            for i in range(len(TOKEN_LIMITS) - 1):
                limit_a = TOKEN_LIMITS[i]
                limit_b = TOKEN_LIMITS[i + 1]
                
                if limit_a not in model_data['datasets'][dataset] or \
                   limit_b not in model_data['datasets'][dataset]:
                    continue
                
                df_a = model_data['datasets'][dataset][limit_a]
                df_b = model_data['datasets'][dataset][limit_b]
                
                # Calculate statistics
                n_a = len(df_a)
                n_b = len(df_b)
                correct_a = df_a['is_correct'].sum()
                correct_b = df_b['is_correct'].sum()
                
                p1 = correct_a / n_a
                p2 = correct_b / n_b
                
                # Two-proportion z-test
                p_pool = (correct_a + correct_b) / (n_a + n_b)
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
                z = (p2 - p1) / se if se > 0 else 0
                p_value = 2 * (1 - norm.cdf(abs(z)))
                
                # Effect size (Cohen's h)
                h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
                
                delta = (p2 - p1) * 100
                
                stat_tests.append({
                    'model': model_name,
                    'dataset': dataset,
                    'comparison': f'{limit_a}→{limit_b}',
                    'limit_a': limit_a,
                    'limit_b': limit_b,
                    'acc_before': p1,
                    'acc_after': p2,
                    'delta_pp': delta,
                    'p_value': p_value,
                    'cohens_h': h,
                    'significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                })
                
                sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                
                print(f"    {limit_a:4d}→{limit_b:4d}: {delta:+6.1f}pp "
                      f"(p={p_value:.4f} {sig_marker}, h={h:.3f})")
    
    return pd.DataFrame(stat_tests)

# ============================================================================
# KEY FINDINGS
# ============================================================================

def extract_key_findings(results_df: pd.DataFrame, stat_tests_df: pd.DataFrame) -> Dict:
    """Extract key findings for all models"""
    
    print("\n" + "="*80)
    print(" KEY FINDINGS")
    print("="*80)
    
    findings = {}
    
    # Finding 1: Plateau Effect
    print("\nFINDING 1: UNIVERSAL PLATEAU EFFECT (1000→2000)")
    print("-" * 60)
    
    plateau_findings = {}
    for model in results_df['model'].unique():
        plateau_findings[model] = {}
        print(f"\n{model}:")
        
        for dataset in DATASETS:
            subset = results_df[(results_df['model']==model) & (results_df['dataset']==dataset)]
            
            if len(subset[subset['token_limit']==1000]) == 0 or len(subset[subset['token_limit']==2000]) == 0:
                continue
            
            acc_1000 = subset[subset['token_limit']==1000]['accuracy'].values[0]
            acc_2000 = subset[subset['token_limit']==2000]['accuracy'].values[0]
            delta = (acc_2000 - acc_1000) * 100
            
            # Get p-value
            test_row = stat_tests_df[
                (stat_tests_df['model']==model) & 
                (stat_tests_df['dataset']==dataset) & 
                (stat_tests_df['comparison']=='1000→2000')
            ]
            p_val = test_row['p_value'].values[0] if len(test_row) > 0 else 1.0
            
            plateau = "✓ PLATEAU" if abs(delta) < 1.0 else "✗ Continues"
            
            plateau_findings[model][dataset] = {
                'acc_1000': acc_1000,
                'acc_2000': acc_2000,
                'delta': delta,
                'p_value': p_val,
                'is_plateau': abs(delta) < 1.0
            }
            
            print(f"  {dataset:12s}: 1000→2000 = {delta:+5.1f}pp (p={p_val:.3f}) {plateau}")
    
    findings['plateau'] = plateau_findings
    
    # Finding 2: Constraint Severity
    print("\n\nFINDING 2: CONSTRAINT SEVERITY @ 300 TOKENS")
    print("-" * 60)
    
    constraint_findings = {}
    for model in results_df['model'].unique():
        constraint_findings[model] = {}
        print(f"\n{model}:")
        
        for dataset in DATASETS:
            subset = results_df[(results_df['model']==model) & (results_df['dataset']==dataset) & (results_df['token_limit']==300)]
            
            if len(subset) == 0:
                continue
            
            row = subset.iloc[0]
            
            hit_pct = row['hit_limit_pct']
            avg_tok = row['avg_tokens']
            acc = row['accuracy']
            
            severity = "SEVERE" if hit_pct > 50 else "MODERATE" if hit_pct > 20 else "MILD"
            
            constraint_findings[model][dataset] = {
                'hit_limit_pct': hit_pct,
                'avg_tokens': avg_tok,
                'accuracy': acc,
                'severity': severity
            }
            
            print(f"  {dataset:12s}: {hit_pct:5.1f}% hit limit "
                  f"(avg={avg_tok:.0f} tokens, acc={acc*100:.1f}%) - {severity}")
    
    findings['constraint'] = constraint_findings
    
    # Finding 3: Improvement Magnitude
    print("\n\nFINDING 3: IMPROVEMENT 300→500")
    print("-" * 60)
    
    improvement_findings = {}
    for model in results_df['model'].unique():
        improvement_findings[model] = {}
        print(f"\n{model}:")
        
        for dataset in DATASETS:
            test_row = stat_tests_df[
                (stat_tests_df['model']==model) & 
                (stat_tests_df['dataset']==dataset) & 
                (stat_tests_df['comparison']=='300→500')
            ]
            
            if len(test_row) > 0:
                delta = test_row['delta_pp'].values[0]
                p_val = test_row['p_value'].values[0]
                h = test_row['cohens_h'].values[0]
                
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                
                improvement_findings[model][dataset] = {
                    'delta': delta,
                    'p_value': p_val,
                    'cohens_h': h,
                    'significant': sig
                }
                
                print(f"  {dataset:12s}: {delta:+6.1f}pp (p={p_val:.4f} {sig}, h={h:.3f})")
    
    findings['improvement'] = improvement_findings
    
    return findings

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_multi_model_visualizations(results_df: pd.DataFrame, 
                                     stat_tests_df: pd.DataFrame,
                                     output_dir: Path):
    """Create comprehensive visualizations for all models"""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    models = results_df['model'].unique()
    n_models = len(models)
    
    # ========================================================================
    # FIGURE 1: Comprehensive Multi-Model Overview
    # ========================================================================
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Multi-Model Token Limit Ablation Study', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # For each dataset, create a subplot
    for dataset_idx, dataset in enumerate(DATASETS):
        ax = axes_flat[dataset_idx * 3]  # Left column: accuracy
        
        for model in models:
            model_data = results_df[(results_df['model']==model) & (results_df['dataset']==dataset)].sort_values('token_limit')
            
            if len(model_data) > 0:
                ax.plot(model_data['token_limit'], 
                       model_data['accuracy'] * 100,
                       marker='o', linewidth=2, markersize=8,
                       label=model, alpha=0.8)
        
        ax.set_xlabel('Token Limit', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title(f'{dataset.upper()}: Accuracy vs Token Limit', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(TOKEN_LIMITS)
        
        # Middle column: Token utilization
        ax = axes_flat[dataset_idx * 3 + 1]
        
        for model in models:
            model_data = results_df[(results_df['model']==model) & (results_df['dataset']==dataset)].sort_values('token_limit')
            
            if len(model_data) > 0:
                ax.plot(model_data['token_limit'], 
                       model_data['avg_tokens'],
                       marker='s', linewidth=2, markersize=8,
                       label=model, alpha=0.8)
        
        ax.plot([300, 2000], [300, 2000], 'k--', linewidth=1, alpha=0.3, label='Limit')
        ax.set_xlabel('Token Limit', fontsize=10)
        ax.set_ylabel('Avg Tokens Used', fontsize=10)
        ax.set_title(f'{dataset.upper()}: Token Utilization', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(TOKEN_LIMITS)
        
        # Right column: % Hitting limit
        ax = axes_flat[dataset_idx * 3 + 2]
        
        x = np.arange(len(TOKEN_LIMITS))
        width = 0.8 / n_models
        
        for model_idx, model in enumerate(models):
            model_data = results_df[(results_df['model']==model) & (results_df['dataset']==dataset)].sort_values('token_limit')
            
            if len(model_data) > 0:
                ax.bar(x + model_idx * width, 
                      model_data['hit_limit_pct'], 
                      width,
                      label=model, 
                      alpha=0.8)
        
        ax.set_xlabel('Token Limit', fontsize=10)
        ax.set_ylabel('% Hitting Limit', fontsize=10)
        ax.set_title(f'{dataset.upper()}: Constraint Severity', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x + width * (n_models-1) / 2)
        ax.set_xticklabels(TOKEN_LIMITS)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'multi_model_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_path}")
    plt.close()
    
    # ========================================================================
    # FIGURE 2: Per-Model Detailed Analysis
    # ========================================================================
    
    for model in models:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Token Limit Ablation: {model}', 
                     fontsize=14, fontweight='bold')
        
        model_data = results_df[results_df['model']==model]
        model_stats = stat_tests_df[stat_tests_df['model']==model]
        
        # Panel A: Accuracy vs Token Limit
        ax = axes[0, 0]
        for dataset in DATASETS:
            dataset_data = model_data[model_data['dataset']==dataset].sort_values('token_limit')
            if len(dataset_data) > 0:
                ax.plot(dataset_data['token_limit'], 
                       dataset_data['accuracy'] * 100,
                       marker='o', linewidth=2.5, markersize=9,
                       label=dataset.upper(), color=COLORS[dataset])
        
        ax.set_xlabel('Token Limit', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('A. Accuracy vs Token Limit', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(TOKEN_LIMITS)
        
        # Panel B: Incremental Gains
        ax = axes[0, 1]
        transitions = ['300→500', '500→1000', '1000→2000']
        x = np.arange(len(transitions))
        width = 0.25
        
        for i, dataset in enumerate(DATASETS):
            dataset_stats = model_stats[model_stats['dataset']==dataset]
            deltas = []
            for t in transitions:
                row = dataset_stats[dataset_stats['comparison']==t]
                if len(row) > 0:
                    deltas.append(row['delta_pp'].values[0])
                else:
                    deltas.append(0)
            
            ax.bar(x + i*width, deltas, width, 
                  label=dataset.upper(), color=COLORS[dataset])
        
        ax.set_xlabel('Token Limit Transition', fontsize=11)
        ax.set_ylabel('Accuracy Gain (pp)', fontsize=11)
        ax.set_title('B. Incremental Gains', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(transitions)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Panel C: % Hitting Limit
        ax = axes[1, 0]
        x = np.arange(len(TOKEN_LIMITS))
        width = 0.25
        
        for i, dataset in enumerate(DATASETS):
            dataset_data = model_data[model_data['dataset']==dataset].sort_values('token_limit')
            if len(dataset_data) > 0:
                ax.bar(x + i*width, 
                      dataset_data['hit_limit_pct'], 
                      width,
                      label=dataset.upper(), 
                      color=COLORS[dataset])
        
        ax.set_xlabel('Token Limit', fontsize=11)
        ax.set_ylabel('% Hitting Limit', fontsize=11)
        ax.set_title('C. Constraint Severity', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(TOKEN_LIMITS)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel D: Summary Table
        ax = axes[1, 1]
        ax.axis('off')
        
        table_data = []
        for dataset in DATASETS:
            dataset_data = model_data[model_data['dataset']==dataset]
            
            if len(dataset_data[dataset_data['token_limit']==300]) == 0:
                continue
            
            acc_300 = dataset_data[dataset_data['token_limit']==300]['accuracy'].values[0]
            acc_1000 = dataset_data[dataset_data['token_limit']==1000]['accuracy'].values[0] if len(dataset_data[dataset_data['token_limit']==1000]) > 0 else 0
            acc_2000 = dataset_data[dataset_data['token_limit']==2000]['accuracy'].values[0] if len(dataset_data[dataset_data['token_limit']==2000]) > 0 else 0
            
            delta_1k = (acc_1000 - acc_300) * 100
            delta_2k = (acc_2000 - acc_1000) * 100
            
            hit_300 = dataset_data[dataset_data['token_limit']==300]['hit_limit_pct'].values[0]
            
            table_data.append([
                dataset.upper(),
                f'{acc_300*100:.0f}%',
                f'{acc_1000*100:.0f}%' if acc_1000 > 0 else 'N/A',
                f'{acc_2000*100:.0f}%' if acc_2000 > 0 else 'N/A',
                f'{delta_1k:+.0f}pp',
                f'{delta_2k:+.0f}pp',
                f'{hit_300:.0f}%'
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Dataset', '300', '1000', '2000', 'Δ300→1K', 'Δ1K→2K', 'Hit@300'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0.15, 1, 0.7])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(7):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.text(0.5, 0.95, 'D. Summary Statistics', 
               ha='center', va='top', fontsize=12, fontweight='bold',
               transform=ax.transAxes)
        
        plt.tight_layout()
        output_path = output_dir / f'{model.replace("/", "_")}_token_ablation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" Saved: {output_path}")
        plt.close()
    
    print("\n All visualizations complete!")

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results_df: pd.DataFrame, 
                stat_tests_df: pd.DataFrame,
                findings: Dict,
                output_dir: Path):
    """Save all results to files"""
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save summary
    output_path = output_dir / 'token_ablation_summary.csv'
    results_df.to_csv(output_path, index=False)
    print(f" Saved: {output_path}")
    
    # Save statistics
    output_path = output_dir / 'token_ablation_statistics.csv'
    stat_tests_df.to_csv(output_path, index=False)
    print(f" Saved: {output_path}")
    
    # Save findings as JSON
    output_path = output_dir / 'token_ablation_findings.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(findings, f, indent=2, default=str)
    print(f" Saved: {output_path}")
    
    # Create summary report
    create_summary_report(results_df, stat_tests_df, findings, output_dir)

def create_summary_report(results_df: pd.DataFrame,
                         stat_tests_df: pd.DataFrame,
                         findings: Dict,
                         output_dir: Path):
    """Create human-readable summary report"""
    
    report_path = output_dir / 'SUMMARY_REPORT.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TOKEN LIMIT ABLATION STUDY: SUMMARY REPORT\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Models tested
        models = results_df['model'].unique()
        f.write(f"## Models Tested ({len(models)})\n\n")
        for model in models:
            f.write(f"- {model}\n")
        f.write("\n---\n\n")
        
        # Datasets
        f.write("## Datasets\n\n")
        f.write("- GSM8K (grade-school math)\n")
        f.write("- BoolQ (yes/no questions)\n")
        f.write("- ARC-Easy (multiple choice science)\n")
        f.write("- GPQA (Graduate-Level Google-Proof Q&A)\n\n")
        f.write("---\n\n")
        
        # Key findings
        f.write("## 🎯 KEY FINDINGS\n\n")
        
        # Finding 1: Plateau
        f.write("### Finding 1: Plateau Effect (1000→2000 tokens)\n\n")
        f.write("| Model | GSM8K | BoolQ | ARC-Easy |\n")
        f.write("|-------|-------|-------|----------|\n")
        
        for model in models:
            if model in findings['plateau']:
                row = f"| {model} |"
                for dataset in DATASETS:
                    if dataset in findings['plateau'][model]:
                        delta = findings['plateau'][model][dataset]['delta']
                        p = findings['plateau'][model][dataset]['p_value']
                        status = "✓" if findings['plateau'][model][dataset]['is_plateau'] else "✗"
                        row += f" {delta:+.1f}pp (p={p:.2f}) {status} |"
                    else:
                        row += " N/A |"
                f.write(row + "\n")
        
        f.write("\n---\n\n")
        
        # Finding 2: Constraint Severity
        f.write("### Finding 2: Constraint Severity @ 300 Tokens\n\n")
        f.write("| Model | Dataset | Hit Limit % | Avg Tokens | Accuracy | Severity |\n")
        f.write("|-------|---------|-------------|------------|----------|----------|\n")
        
        for model in models:
            if model in findings['constraint']:
                for dataset in DATASETS:
                    if dataset in findings['constraint'][model]:
                        c = findings['constraint'][model][dataset]
                        f.write(f"| {model} | {dataset.upper()} | "
                               f"{c['hit_limit_pct']:.1f}% | "
                               f"{c['avg_tokens']:.0f} | "
                               f"{c['accuracy']*100:.1f}% | "
                               f"{c['severity']} |\n")
        
        f.write("\n---\n\n")
        
        # Recommendation
        f.write("##  RECOMMENDATIONS\n\n")
        f.write("Based on the analysis:\n\n")
        f.write("1. **Optimal token budget:** 1000 tokens for most tasks\n")
        f.write("2. **Complex tasks (GSM8K):** Need 500+ tokens to avoid severe truncation\n")
        f.write("3. **Simple tasks (BoolQ):** Can use 300 tokens safely\n")
        f.write("4. **Cost optimization:** Going beyond 1000 tokens provides minimal benefit\n")
    
    print(f" Saved: {report_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze multi-model token ablation results')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing model subdirectories')
    parser.add_argument('--output_dir', type=str, default='./token_ablation_r1_model_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("MULTI-MODEL TOKEN LIMIT ABLATION ANALYSIS")
    print("="*80)
    print(f"\nBase directory: {base_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Load all data
    all_model_data = load_all_models(base_dir)
    
    if not all_model_data:
        print("\n❌ No data loaded. Exiting.")
        return
    # Calculate metrics
    results_df = calculate_all_metrics(all_model_data)
    
    # Run statistical tests
    stat_tests_df = run_statistical_tests(all_model_data)
    
    # Extract key findings
    findings = extract_key_findings(results_df, stat_tests_df)
    
    # Create visualizations
    create_multi_model_visualizations(results_df, stat_tests_df, output_dir)
    
    # Save results
    save_results(results_df, stat_tests_df, findings, output_dir)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey files:")
    print(f"  • token_ablation_summary.csv - All metrics")
    print(f"  • token_ablation_statistics.csv - Statistical tests")
    print(f"  • token_ablation_findings.json - Key findings")
    print(f"  • SUMMARY_REPORT.md - Human-readable report")
    print(f"  • multi_model_comprehensive.png - Overview figure")
    print(f"  • [model]_token_ablation.png - Per-model figures")

if __name__ == "__main__":
    main()