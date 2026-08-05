"""
FUNCTIONAL ANALYSIS OF TOKEN LIMIT EFFECTS - BATCH VERSION
===========================================================
Analyzes quality degradation patterns, step efficiency, and self-regulation
across different token limits to understand WHY models plateau or decline.

AUTO-DISCOVERS AND PROCESSES ALL MODELS AND DATASETS IN BASE DIRECTORY

USAGE:
    python functional_analysis_batch.py --base_dir "data/raw/phase3_responses"
    
DIRECTORY STRUCTURE EXPECTED:
    base_dir/
        deepseek-aiDeepSeek-R1-Distill-Qwen-7B/
            gsm8k/
                gsm8k_300_results.csv
                gsm8k_1000_results.csv
                ...
            boolq/
                boolq_300_results.csv
                ...
        AIDC-AI-Marco-o1/
            gsm8k/
                ...
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re
import argparse
from typing import Dict, List, Tuple
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = ['gsm8k', 'boolq', 'arc-easy', 'gpqa']
TOKEN_LIMITS = [300, 500, 1000, 2000, 4000, 8000]

# Color scheme
COLORS = {
    'quality': '#2E86AB',
    'efficiency': '#A23B72',
    'regulation': '#F18F01'
}

# ============================================================================
# MODEL & DATASET DISCOVERY
# ============================================================================

def discover_models(base_dir: Path) -> List[str]:
    """
    Auto-discover all model directories in base_dir
    Returns list of model names (without _model suffix)
    """
    
    print(f"\n{'='*80}")
    print("🔍 DISCOVERING MODELS")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}\n")
    
    models = []
    
    # Look for directories (not _model suffix, just the actual directory names)
    for item in base_dir.iterdir():
        if item.is_dir():
            # Check if it contains any dataset subdirectories
            has_datasets = any((item / ds).exists() for ds in DATASETS)
            if has_datasets:
                models.append(item.name)
                print(f"   Found: {item.name}")
    
    if not models:
        print("   No model directories found!")
        print(f"\n  Expected structure:")
        print(f"    {base_dir}/")
        print(f"      model_name/")
        print(f"        gsm8k/")
        print(f"          gsm8k_300_results.csv")
        print(f"          gsm8k_1000_results.csv")
        print(f"          ...")
    
    print(f"\n Total models found: {len(models)}")
    return sorted(models)

def discover_datasets(base_dir: Path, model_name: str) -> List[str]:
    """
    Auto-discover which datasets exist for a given model
    """
    
    model_dir = base_dir / model_name
    available_datasets = []
    
    for dataset in DATASETS:
        dataset_dir = model_dir / dataset
        if dataset_dir.exists():
            # Check if it has any CSV files
            csv_files = list(dataset_dir.glob('*.csv'))
            if csv_files:
                available_datasets.append(dataset)
    
    return available_datasets

# ============================================================================
# DATA LOADING
# ============================================================================

def load_model_data(base_dir: Path, model_name: str, dataset: str) -> Dict:
    """
    Load data for a specific model and dataset
    """
    
    model_dir = base_dir / model_name
    dataset_dir = model_dir / dataset
    
    print(f"\n   Loading {dataset} data...")
    
    if not dataset_dir.exists():
        print(f"     Directory not found: {dataset_dir}")
        return None
    
    data = {
        'model_name': model_name,
        'dataset': dataset,
        'token_limits': {}
    }
    
    # Load each token limit
    for limit in TOKEN_LIMITS:
        filepath = dataset_dir / f"{dataset}_{limit}_results.csv"
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            data['token_limits'][limit] = df
            print(f"     {limit:5d} tokens: {len(df):3d} samples")
        else:
            # Silently skip missing token limits
            pass
    
    if not data['token_limits']:
        print(f"     No data files found")
        return None
    
    print(f"     Loaded {len(data['token_limits'])} token limit conditions")
    return data

# ============================================================================
# QUALITY METRICS ANALYSIS
# ============================================================================

def calculate_quality_metrics(text: str) -> Dict:
    """
    Calculate quality degradation metrics from generated text
    
    Metrics:
    1. Vocabulary diversity (unique words / total words)
    2. Bigram repetition rate (repeated bigrams / total bigrams)
    3. Self-similarity (sentence overlap rate)
    4. Average sentence length
    """
    
    if not text or not isinstance(text, str) or len(text) < 20:
        return None
    
    # Tokenize
    words = text.lower().split()
    if len(words) < 10:
        return None
    
    # 1. Vocabulary diversity
    unique_words = len(set(words))
    diversity = unique_words / len(words)
    
    # 2. Bigram repetition
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    if bigrams:
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        bigram_repetition = repeated_bigrams / len(set(bigrams))
    else:
        bigram_repetition = 0
    
    # 3. Self-similarity (sentence-level overlap)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    similar_pairs = 0
    total_pairs = 0
    
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            total_pairs += 1
            words_i = set(sentences[i].lower().split())
            words_j = set(sentences[j].lower().split())
            
            if words_i and words_j:
                overlap = len(words_i & words_j) / max(len(words_i), len(words_j))
                if overlap > 0.5:  # >50% word overlap
                    similar_pairs += 1
    
    self_similarity = similar_pairs / total_pairs if total_pairs > 0 else 0
    
    # 4. Average sentence length
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    return {
        'diversity': diversity,
        'bigram_repetition': bigram_repetition,
        'self_similarity': self_similarity,
        'avg_sentence_length': avg_sentence_length,
        'total_words': len(words),
        'unique_words': unique_words,
        'n_sentences': len(sentences)
    }

def analyze_quality_degradation(data: Dict) -> pd.DataFrame:
    """
    Analyze quality metrics across token limits
    """
    
    results = []
    
    for limit, df in sorted(data['token_limits'].items()):
        
        quality_metrics = []
        
        for idx, row in df.iterrows():
            text = str(row.get('full_generation', ''))
            metrics = calculate_quality_metrics(text)
            
            if metrics:
                metrics['is_correct'] = row.get('is_correct', False)
                metrics['output_tokens'] = row.get('output_tokens', 0)
                metrics['sample_id'] = row.get('sample_id', idx)
                quality_metrics.append(metrics)
        
        if quality_metrics:
            df_metrics = pd.DataFrame(quality_metrics)
            
            # Aggregate statistics
            result = {
                'token_limit': limit,
                'n_samples': len(df_metrics),
                'avg_diversity': df_metrics['diversity'].mean(),
                'std_diversity': df_metrics['diversity'].std(),
                'avg_repetition': df_metrics['bigram_repetition'].mean(),
                'std_repetition': df_metrics['bigram_repetition'].std(),
                'avg_self_similarity': df_metrics['self_similarity'].mean(),
                'std_self_similarity': df_metrics['self_similarity'].std(),
                'avg_sentence_length': df_metrics['avg_sentence_length'].mean(),
                'diversity_correct': df_metrics[df_metrics['is_correct']]['diversity'].mean(),
                'diversity_incorrect': df_metrics[~df_metrics['is_correct']]['diversity'].mean()
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

# ============================================================================
# STEP EFFICIENCY ANALYSIS
# ============================================================================

def count_reasoning_steps(text: str) -> int:
    """Count reasoning steps from generated text"""
    
    if not text or not isinstance(text, str):
        return 0
    
    # Look for step markers
    step_patterns = [
        r'Step \d+',
        r'\d+\.',
        r'\d+\)',
        r'First,|Second,|Third,|Fourth,|Finally,',
    ]
    
    steps = 0
    for pattern in step_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        steps = max(steps, len(matches))
    
    return max(steps, 1)  # At least 1 step

def analyze_step_efficiency(data: Dict) -> pd.DataFrame:
    """
    Analyze reasoning step efficiency across token limits
    """
    
    results = []
    
    for limit, df in sorted(data['token_limits'].items()):
        
        steps = []
        tokens = []
        
        for idx, row in df.iterrows():
            text = str(row.get('full_generation', ''))
            n_steps = count_reasoning_steps(text)
            n_tokens = row.get('output_tokens', 0)
            
            if n_steps > 0 and n_tokens > 0:
                steps.append(n_steps)
                tokens.append(n_tokens)
        
        if steps:
            result = {
                'token_limit': limit,
                'n_samples': len(steps),
                'avg_steps': np.mean(steps),
                'std_steps': np.std(steps),
                'avg_tokens': np.mean(tokens),
                'tokens_per_step': np.mean(tokens) / np.mean(steps),
                'step_efficiency': np.mean(steps) / (np.mean(tokens) / 100)  # steps per 100 tokens
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

# ============================================================================
# SELF-REGULATION ANALYSIS
# ============================================================================

def analyze_self_regulation(data: Dict) -> pd.DataFrame:
    """
    Analyze whether model self-regulates (stops before hitting token limit)
    """
    
    results = []
    
    for limit, df in sorted(data['token_limits'].items()):
        
        output_tokens = []
        hit_limit = 0
        
        for idx, row in df.iterrows():
            tokens = row.get('output_tokens', 0)
            if tokens > 0:
                output_tokens.append(tokens)
                
                # Consider "hit limit" if within 5% of limit
                if tokens >= limit * 0.95:
                    hit_limit += 1
        
        if output_tokens:
            result = {
                'token_limit': limit,
                'n_samples': len(output_tokens),
                'avg_tokens_used': np.mean(output_tokens),
                'std_tokens_used': np.std(output_tokens),
                'usage_pct': (np.mean(output_tokens) / limit) * 100,
                'hit_limit_count': hit_limit,
                'hit_limit_pct': (hit_limit / len(output_tokens)) * 100
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

# ============================================================================
# ANSWER STABILITY ANALYSIS
# ============================================================================

def analyze_answer_stability(data: Dict) -> pd.DataFrame:
    """
    Analyze how answers change between token limits (correct→wrong or wrong→correct)
    """
    
    token_limits_sorted = sorted(data['token_limits'].keys())
    
    if len(token_limits_sorted) < 2:
        return pd.DataFrame()  # Need at least 2 limits to compare
    
    results = []
    
    for i in range(len(token_limits_sorted) - 1):
        limit1 = token_limits_sorted[i]
        limit2 = token_limits_sorted[i + 1]
        
        df1 = data['token_limits'][limit1]
        df2 = data['token_limits'][limit2]
        
        # Match samples by ID
        df1_indexed = df1.set_index('sample_id') if 'sample_id' in df1.columns else df1
        df2_indexed = df2.set_index('sample_id') if 'sample_id' in df2.columns else df2
        
        common_ids = df1_indexed.index.intersection(df2_indexed.index)
        
        if len(common_ids) == 0:
            continue
        
        improved = 0
        degraded = 0
        stable_correct = 0
        stable_incorrect = 0
        
        for sample_id in common_ids:
            correct1 = df1_indexed.loc[sample_id, 'is_correct']
            correct2 = df2_indexed.loc[sample_id, 'is_correct']
            
            if not correct1 and correct2:
                improved += 1
            elif correct1 and not correct2:
                degraded += 1
            elif correct1 and correct2:
                stable_correct += 1
            else:
                stable_incorrect += 1
        
        total = len(common_ids)
        
        result = {
            'transition': f'{limit1}→{limit2}',
            'limit1': limit1,
            'limit2': limit2,
            'n_samples': total,
            'improved': improved,
            'degraded': degraded,
            'stable_correct': stable_correct,
            'stable_incorrect': stable_incorrect,
            'improved_pct': (improved / total) * 100,
            'degraded_pct': (degraded / total) * 100,
            'net_change': improved - degraded
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def compute_mechanistic_dimensions(quality_df, regulation_df, efficiency_df):
    """Convert per-limit data to paper's Table 18 format."""
    first = quality_df.iloc[0]   # shortest token limit
    last = quality_df.iloc[-1]   # longest token limit
    
    diversity_delta = ((last['avg_diversity'] - first['avg_diversity']) 
                       / first['avg_diversity']) * 100
    repetition_delta = ((last['avg_repetition'] - first['avg_repetition'])
                        / max(first['avg_repetition'], 0.001)) * 100
    token_usage = regulation_df.iloc[-1]['usage_pct']
    tokens_per_step = efficiency_df.iloc[-1]['tokens_per_step']
    
    return {
        'diversity_delta_pct': diversity_delta,
        'repetition_delta_pct': repetition_delta,
        'token_usage_pct': token_usage,
        'tokens_per_step': tokens_per_step,
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_visualization(quality_df: pd.DataFrame,
                                      efficiency_df: pd.DataFrame,
                                      regulation_df: pd.DataFrame,
                                      stability_df: pd.DataFrame,
                                      model_name: str,
                                      dataset: str,
                                      output_dir: Path):
    """Create comprehensive functional analysis visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Functional Analysis: {model_name} ({dataset})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Quality Degradation
    ax = axes[0, 0]
    if not quality_df.empty:
        ax.plot(quality_df['token_limit'], quality_df['avg_diversity'], 
               'o-', color=COLORS['quality'], linewidth=2, markersize=8, label='Diversity')
        ax.fill_between(quality_df['token_limit'],
                        quality_df['avg_diversity'] - quality_df['std_diversity'],
                        quality_df['avg_diversity'] + quality_df['std_diversity'],
                        alpha=0.2, color=COLORS['quality'])
        
        ax.set_xlabel('Token Limit', fontsize=11)
        ax.set_ylabel('Vocabulary Diversity', fontsize=11, color=COLORS['quality'])
        ax.tick_params(axis='y', labelcolor=COLORS['quality'])
        ax.set_title('Quality Maintenance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Secondary y-axis for repetition
        ax2 = ax.twinx()
        ax2.plot(quality_df['token_limit'], quality_df['avg_repetition'],
                's--', color='red', linewidth=2, markersize=6, label='Repetition', alpha=0.7)
        ax2.set_ylabel('Bigram Repetition', fontsize=11, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # 2. Step Efficiency
    ax = axes[0, 1]
    if not efficiency_df.empty:
        ax.plot(efficiency_df['token_limit'], efficiency_df['avg_steps'],
               'o-', color=COLORS['efficiency'], linewidth=2, markersize=8)
        ax.set_xlabel('Token Limit', fontsize=11)
        ax.set_ylabel('Average Steps', fontsize=11)
        ax.set_title('Reasoning Step Efficiency', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add tokens per step as annotation
        for _, row in efficiency_df.iterrows():
            ax.annotate(f"{row['tokens_per_step']:.0f} t/s",
                       xy=(row['token_limit'], row['avg_steps']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    # 3. Self-Regulation
    ax = axes[1, 0]
    if not regulation_df.empty:
        bars = ax.bar(regulation_df['token_limit'], regulation_df['usage_pct'],
                     color=COLORS['regulation'], alpha=0.7)
        
        # Color bars based on usage
        for bar, usage in zip(bars, regulation_df['usage_pct']):
            if usage < 50:
                bar.set_color('green')
                bar.set_alpha(0.6)
            elif usage < 70:
                bar.set_color(COLORS['regulation'])
                bar.set_alpha(0.7)
            else:
                bar.set_color('red')
                bar.set_alpha(0.7)
        
        ax.axhline(y=60, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(regulation_df['token_limit'].mean(), 62, 'Weak regulation threshold',
               ha='center', fontsize=9, alpha=0.7)
        
        ax.set_xlabel('Token Limit', fontsize=11)
        ax.set_ylabel('Token Usage (%)', fontsize=11)
        ax.set_title('Self-Regulation', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Answer Stability
    ax = axes[1, 1]
    if not stability_df.empty:
        x = np.arange(len(stability_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, stability_df['improved_pct'], width,
                      label='Improved', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, stability_df['degraded_pct'], width,
                      label='Degraded', color='red', alpha=0.7)
        
        ax.set_xlabel('Token Limit Transition', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title('Answer Stability', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stability_df['transition'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'{model_name}_{dataset}_functional_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(quality_df: pd.DataFrame,
                efficiency_df: pd.DataFrame,
                regulation_df: pd.DataFrame,
                stability_df: pd.DataFrame,
                model_name: str,
                dataset: str,
                output_dir: Path):
    """Save analysis results to CSV files"""
    
    # Create model-specific subdirectory
    model_dir = output_dir / model_name
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Save CSVs
    quality_df.to_csv(model_dir / f'{dataset}_quality_metrics.csv', index=False)
    efficiency_df.to_csv(model_dir / f'{dataset}_step_efficiency.csv', index=False)
    regulation_df.to_csv(model_dir / f'{dataset}_self_regulation.csv', index=False)
    
    if not stability_df.empty:
        stability_df.to_csv(model_dir / f'{dataset}_answer_stability.csv', index=False)
    
    # Create summary report
    create_summary_report(quality_df, efficiency_df, regulation_df, stability_df,
                         model_name, dataset, model_dir)

def create_summary_report(quality_df: pd.DataFrame,
                         efficiency_df: pd.DataFrame,
                         regulation_df: pd.DataFrame,
                         stability_df: pd.DataFrame,
                         model_name: str,
                         dataset: str,
                         output_dir: Path):
    """Create human-readable summary report"""
    
    report_path = output_dir / f'{dataset}_FUNCTIONAL_ANALYSIS_REPORT.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# FUNCTIONAL ANALYSIS REPORT\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Dataset:** {dataset}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Summary
        f.write("##  EXECUTIVE SUMMARY\n\n")
        
        if len(quality_df) >= 2:
            first = quality_df.iloc[0]
            last = quality_df.iloc[-1]
            div_change = last['avg_diversity'] - first['avg_diversity']
            rep_change = last['avg_repetition'] - first['avg_repetition']
            
            if abs(div_change) < 0.02:
                f.write(f"**Quality Status:**  MAINTAINED\n\n")
            elif div_change < -0.05:
                f.write(f"**Quality Status:**  DEGRADING\n\n")
            else:
                f.write(f"**Quality Status:**  CHANGING\n\n")
            
            f.write(f"- Diversity: {first['avg_diversity']:.3f} → {last['avg_diversity']:.3f} ({div_change:+.3f})\n")
            f.write(f"- Repetition: {first['avg_repetition']:.3f} → {last['avg_repetition']:.3f} ({rep_change:+.3f})\n\n")
        
        if len(regulation_df) >= 1:
            last_reg = regulation_df.iloc[-1]
            
            if last_reg['usage_pct'] < 50:
                f.write(f"**Self-Regulation:**  STRONG ({last_reg['usage_pct']:.1f}% usage)\n\n")
            elif last_reg['usage_pct'] < 70:
                f.write(f"**Self-Regulation:**  MODERATE ({last_reg['usage_pct']:.1f}% usage)\n\n")
            else:
                f.write(f"**Self-Regulation:**  WEAK ({last_reg['usage_pct']:.1f}% usage)\n\n")
        
        f.write("---\n\n")
        
        # Detailed findings
        f.write("##  DETAILED FINDINGS\n\n")
        
        f.write("### Finding 1: Quality Maintenance\n\n")
        f.write("| Token Limit | Diversity | Repetition | Self-Similarity |\n")
        f.write("|-------------|-----------|------------|------------------|\n")
        for _, row in quality_df.iterrows():
            f.write(f"| {int(row['token_limit']):5d} | {row['avg_diversity']:.3f} | "
                   f"{row['avg_repetition']:.3f} | {row['avg_self_similarity']:.3f} |\n")
        
        f.write("\n### Finding 2: Step Efficiency\n\n")
        f.write("| Token Limit | Avg Steps | Tokens/Step | Efficiency |\n")
        f.write("|-------------|-----------|-------------|------------|\n")
        for _, row in efficiency_df.iterrows():
            if row['tokens_per_step'] < 100:
                eff_rating = " Good"
            elif row['tokens_per_step'] < 120:
                eff_rating = " Moderate"
            else:
                eff_rating = " Poor"
            f.write(f"| {int(row['token_limit']):5d} | {row['avg_steps']:.1f} | "
                   f"{row['tokens_per_step']:.1f} | {eff_rating} |\n")
        
        f.write("\n### Finding 3: Self-Regulation\n\n")
        f.write("| Token Limit | Usage % | Hit Limit % | Status |\n")
        f.write("|-------------|---------|-------------|--------|\n")
        for _, row in regulation_df.iterrows():
            if row['usage_pct'] < 50:
                status = " Strong"
            elif row['usage_pct'] < 70:
                status = " Moderate"
            else:
                status = " Weak"
            f.write(f"| {int(row['token_limit']):5d} | {row['usage_pct']:.1f}% | "
                   f"{row['hit_limit_pct']:.1f}% | {status} |\n")
        
        if not stability_df.empty:
            f.write("\n### Finding 4: Answer Stability\n\n")
            f.write("| Transition | Improved | Degraded | Net Change |\n")
            f.write("|------------|----------|----------|------------|\n")
            for _, row in stability_df.iterrows():
                f.write(f"| {row['transition']} | {row['improved']} ({row['improved_pct']:.1f}%) | "
                       f"{row['degraded']} ({row['degraded_pct']:.1f}%) | {row['net_change']:+d} |\n")
        
        f.write("\n---\n\n")
        
        # Interpretation
        f.write("## INTERPRETATION\n\n")
        
        if len(quality_df) >= 2:
            first = quality_df.iloc[0]
            last = quality_df.iloc[-1]
            div_change = last['avg_diversity'] - first['avg_diversity']
            
            if abs(div_change) < 0.02:
                f.write("1.  **Quality is maintained** - no evidence of degradation\n")
            elif div_change < -0.05:
                f.write("1.  **Quality degrades** - vocabulary becomes less diverse\n")
            else:
                f.write("1.  **Quality shows variation** - monitor for patterns\n")
        
        if len(regulation_df) >= 1:
            last_reg = regulation_df.iloc[-1]
            if last_reg['usage_pct'] < 50:
                f.write("2.  **Strong self-regulation** - stops when reasoning complete\n")
            elif last_reg['usage_pct'] < 70:
                f.write("2.  **Moderate self-regulation** - sometimes uses full budget\n")
            else:
                f.write("2.  **Weak self-regulation** - frequently exhausts token limit\n")
        
        if not stability_df.empty and len(stability_df) > 0:
            last_stab = stability_df.iloc[-1]
            if last_stab['net_change'] > 0:
                f.write("3.  **Net improvement** - benefits from extended reasoning\n")
            elif last_stab['net_change'] < 0:
                f.write("3.  **Over-reasoning detected** - quality degrades with more tokens\n")
            else:
                f.write("3.  **Stable performance** - no systematic change\n")

# ============================================================================
# MAIN - BATCH PROCESSING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch functional analysis for all models and datasets'
    )
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing all model subdirectories')
    parser.add_argument('--output_dir', type=str, default='./functional_analysis',
                       help='Output directory for results')
    parser.add_argument('--models', type=str, nargs='*',
                       help='Optional: specific models to process (default: all)')
    parser.add_argument('--datasets', type=str, nargs='*', choices=DATASETS,
                       help='Optional: specific datasets to process (default: all available)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("🔬 BATCH FUNCTIONAL ANALYSIS")
    print("="*80)
    print(f"\nBase directory: {base_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Discover models
    if args.models:
        models = args.models
        print(f" Using specified models: {models}")
    else:
        models = discover_models(base_dir)
        if not models:
            print("\n No models found! Exiting.")
            return
    
    # Process each model
    total_processed = 0
    total_skipped = 0
    
    for i, model_name in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f" PROCESSING MODEL {i}/{len(models)}: {model_name}")
        print(f"{'='*80}")
        
        # Discover datasets for this model
        if args.datasets:
            datasets_to_process = args.datasets
        else:
            datasets_to_process = discover_datasets(base_dir, model_name)
        
        if not datasets_to_process:
            print(f"    No datasets found for {model_name}, skipping...")
            total_skipped += 1
            continue
        
        print(f"   Datasets to process: {', '.join(datasets_to_process)}")
        
        # Process each dataset
        for dataset in datasets_to_process:
            print(f"\n  {'─'*76}")
            print(f"   Processing: {dataset}")
            print(f"  {'─'*76}")
            
            # Load data
            data = load_model_data(base_dir, model_name, dataset)
            
            if not data or not data['token_limits']:
                print(f"      No data available, skipping...")
                continue
            
            # Run analyses
            print(f"     Running analyses...")
            quality_df = analyze_quality_degradation(data)
            efficiency_df = analyze_step_efficiency(data)
            regulation_df = analyze_self_regulation(data)
            stability_df = analyze_answer_stability(data)
            
            # Create visualization
            print(f"     Creating visualization...")
            create_comprehensive_visualization(
                quality_df, efficiency_df, regulation_df, stability_df,
                model_name, dataset, output_dir
            )
            
            # Save results
            print(f"     Saving results...")
            save_results(
                quality_df, efficiency_df, regulation_df, stability_df,
                model_name, dataset, output_dir
            )
            
            total_processed += 1
            print(f"     Completed: {model_name}/{dataset}")
    
    # Final summary
    print("\n" + "="*80)
    print(" BATCH PROCESSING COMPLETE!")
    print("="*80)
    print(f"\n Summary:")
    print(f"  • Models processed: {len(models)}")
    print(f"  • Analyses completed: {total_processed}")
    print(f"  • Skipped: {total_skipped}")
    print(f"\n Results saved to: {output_dir}")
    print(f"\n Each model has its own subdirectory with:")
    print(f"  • Quality metrics CSV")
    print(f"  • Step efficiency CSV")
    print(f"  • Self-regulation CSV")
    print(f"  • Answer stability CSV")
    print(f"  • Visualization PNG")
    print(f"  • Summary report MD")

if __name__ == "__main__":
    main()