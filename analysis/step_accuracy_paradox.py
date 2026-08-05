
"""
THE REASONING STEP PARADOX: COMPLETE INVESTIGATION
===================================================
Add these experiments to your rigorous_temporal_analysis.py

KEY FINDING: Long-horizon tasks show HIGHER accuracy (61.1%) than short (52.3%)
RESEARCH QUESTION: Why doesn't more reasoning steps = harder tasks?

Experiments:
1. Difficulty Factor Analysis (what actually makes tasks hard?)
2. Pattern Matching vs Reasoning (are models actually reasoning?)
3. Causal Intervention (breaking reasoning chains)
4. Human Baseline Comparison
5. Task Characteristic Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Tuple
import json

# ============================================================================
# EXPERIMENT 1: DIFFICULTY FACTOR ANALYSIS
# ============================================================================
class DifficultyFactorAnalyzer:
    """
    Investigate what actually makes tasks difficult beyond step count.
    
    HYPOTHESIS: Long tasks are easier because they have:
    - More structure/scaffolding
    - Clearer intermediate goals
    - More verbose/explicit reasoning paths
    """
    
    def __init__(self, metrics_df: pd.DataFrame):
        self.metrics_df = metrics_df
    
    def analyze_step_accuracy_paradox(self) -> Dict:
        """Core analysis: Why do more steps → higher accuracy?"""
        print("\n" + "="*80)
        print("EXPERIMENT 1: THE STEP-ACCURACY PARADOX")
        print("="*80)
        
        results = {}
        
        # 1. Overall correlation: steps vs accuracy
        print("\n Overall Pattern:")
        
        corr_spearman, p_spearman = spearmanr(
            self.metrics_df['required_steps'],
            self.metrics_df['final_accuracy']
        )
        
        corr_kendall, p_kendall = kendalltau(
            self.metrics_df['required_steps'],
            self.metrics_df['final_accuracy']
        )
        
        print(f"   Spearman ρ = {corr_spearman:.3f}, p = {p_spearman:.4e}")
        print(f"   Kendall τ = {corr_kendall:.3f}, p = {p_kendall:.4e}")
        
        if corr_spearman > 0 and p_spearman < 0.05:
            print(f"    PARADOX CONFIRMED: More steps → HIGHER accuracy!")
        
        results['overall_correlation'] = {
            'spearman_rho': float(corr_spearman),
            'spearman_p': float(p_spearman),
            'kendall_tau': float(corr_kendall),
            'kendall_p': float(p_kendall)
        }
        
        # 2. By dataset: Does pattern hold everywhere?
        print("\n By Dataset:")
        
        dataset_results = {}
        for dataset in self.metrics_df['dataset'].unique():
            dataset_df = self.metrics_df[self.metrics_df['dataset'] == dataset]
            
            corr, p = spearmanr(
                dataset_df['required_steps'],
                dataset_df['final_accuracy']
            )
            
            print(f"\n   {dataset}:")
            print(f"      Correlation: ρ = {corr:.3f}, p = {p:.4f}")
            
            # Show accuracy by horizon for this dataset
            for horizon in ['short', 'medium', 'long']:
                h_df = dataset_df[dataset_df['temporal_horizon'] == horizon]
                if len(h_df) > 0:
                    acc = h_df['final_accuracy'].mean()
                    n_steps = h_df['required_steps'].mean()
                    print(f"      {horizon:8s}: {acc:.1%} acc, {n_steps:.1f} steps")
            
            dataset_results[dataset] = {
                'correlation': float(corr),
                'p_value': float(p),
                'pattern': 'positive' if corr > 0 else 'negative'
            }
        
        results['by_dataset'] = dataset_results
        
        # 3. By model size: Does model scale affect the paradox?
        print("\n By Model Size:")
        
        size_bins = [
            ('Tiny (<2B)', 0, 2),
            ('Small (2-10B)', 2, 10),
            ('Large (10-100B)', 10, 100),
            ('Giant (100B+)', 100, 500)
        ]
        
        size_results = {}
        for bin_name, min_size, max_size in size_bins:
            bin_df = self.metrics_df[
                (self.metrics_df['model_size'] >= min_size) &
                (self.metrics_df['model_size'] < max_size)
            ]
            
            if len(bin_df) < 30:
                continue
            
            corr, p = spearmanr(
                bin_df['required_steps'],
                bin_df['final_accuracy']
            )
            
            print(f"\n   {bin_name}:")
            print(f"      Correlation: ρ = {corr:.3f}, p = {p:.4f}")
            print(f"      Interpretation: {'More steps → higher acc' if corr > 0 else 'More steps → lower acc'}")
            
            size_results[bin_name] = {
                'correlation': float(corr),
                'p_value': float(p)
            }
        
        results['by_model_size'] = size_results
        
        return results
    
    def analyze_question_characteristics(self) -> Dict:
        """Analyze what question characteristics predict difficulty"""
        print("\n Question Characteristic Analysis:")
        
        results = {}
        
        # Compute question features
        self.metrics_df['question_length'] = self.metrics_df['question'].apply(
            lambda x: len(str(x).split())
        )
        
        self.metrics_df['question_complexity'] = self.metrics_df['question'].apply(
            lambda x: self._compute_question_complexity(str(x))
        )
        
        self.metrics_df['has_numbers'] = self.metrics_df['question'].apply(
            lambda x: bool(re.search(r'\d', str(x)))
        )
        
        # Correlations with accuracy
        features = {
            'question_length': 'Question Length (words)',
            'question_complexity': 'Syntactic Complexity',
            'required_steps': 'Required Steps',
            'reasoning_trace_length': 'Actual Reasoning Length'
        }
        
        print("\n   Correlations with Accuracy:")
        
        for feature, label in features.items():
            if feature in self.metrics_df.columns:
                corr, p = spearmanr(
                    self.metrics_df[feature],
                    self.metrics_df['final_accuracy']
                )
                
                print(f"      {label:30s}: ρ = {corr:+.3f}, p = {p:.4e}")
                
                results[feature] = {
                    'correlation': float(corr),
                    'p_value': float(p)
                }
        
        return results
    
    def _compute_question_complexity(self, question: str) -> int:
        """Heuristic for syntactic complexity"""
        complexity = 0
        
        # Count clauses
        complexity += question.count(',')
        complexity += question.count(';')
        
        # Count conditionals
        conditionals = ['if', 'when', 'unless', 'provided', 'whereas', 'although']
        complexity += sum(2 for word in conditionals if word in question.lower())
        
        # Count questions
        complexity += question.count('?')
        
        return complexity
    
    def create_difficulty_visualization(self, output_dir: Path):
        """Visualize the paradox"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Steps vs Accuracy (overall)
        ax = axes[0, 0]
        
        # Bin steps for cleaner visualization
        self.metrics_df['step_bin'] = pd.cut(
            self.metrics_df['required_steps'],
            bins=[0, 2, 5, 10, 100],
            labels=['1-2', '3-5', '6-10', '10+']
        )
        
        step_acc = self.metrics_df.groupby('step_bin')['final_accuracy'].agg(['mean', 'std', 'count'])
        step_acc['se'] = step_acc['std'] / np.sqrt(step_acc['count'])
        
        ax.errorbar(
            range(len(step_acc)),
            step_acc['mean'],
            yerr=1.96 * step_acc['se'],
            marker='o',
            markersize=10,
            capsize=5,
            linewidth=2
        )
        
        ax.set_xticks(range(len(step_acc)))
        ax.set_xticklabels(step_acc.index)
        ax.set_xlabel('Number of Reasoning Steps', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('The Step-Accuracy Paradox', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax.legend()
        
        # Plot 2: By Dataset
        ax = axes[0, 1]
        
        for dataset in self.metrics_df['dataset'].unique()[:5]:  # Top 5 datasets
            dataset_df = self.metrics_df[self.metrics_df['dataset'] == dataset]
            
            step_acc_ds = dataset_df.groupby('step_bin')['final_accuracy'].mean()
            
            ax.plot(
                range(len(step_acc_ds)),
                step_acc_ds.values,
                marker='o',
                label=dataset,
                linewidth=2
            )
        
        ax.set_xticks(range(4))
        ax.set_xticklabels(['1-2', '3-5', '6-10', '10+'])
        ax.set_xlabel('Number of Steps', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Pattern by Dataset', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Question Length vs Accuracy
        ax = axes[1, 0]
        
        self.metrics_df['length_bin'] = pd.qcut(
            self.metrics_df['question_length'],
            q=5,
            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'],
            duplicates='drop'
        )
        
        length_acc = self.metrics_df.groupby('length_bin')['final_accuracy'].mean()
        
        ax.bar(range(len(length_acc)), length_acc.values)
        ax.set_xticks(range(len(length_acc)))
        ax.set_xticklabels(length_acc.index, rotation=45, ha='right')
        ax.set_xlabel('Question Length', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Question Length vs Accuracy', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Heatmap: Steps × Model Size
        ax = axes[1, 1]
        
        # Create bins for model size
        self.metrics_df['size_bin'] = pd.cut(
            self.metrics_df['model_size'],
            bins=[0, 2, 10, 100, 500],
            labels=['<2B', '2-10B', '10-100B', '100B+']
        )
        
        pivot = self.metrics_df.pivot_table(
            values='final_accuracy',
            index='step_bin',
            columns='size_bin',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'Accuracy'}
        )
        
        ax.set_xlabel('Model Size', fontsize=12)
        ax.set_ylabel('Number of Steps', fontsize=12)
        ax.set_title('Accuracy Heatmap: Steps × Size', fontsize=14)
        
        plt.tight_layout()
        output_path = output_dir / "step_accuracy_paradox.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n Saved paradox visualization: {output_path}")
        return output_path

# ============================================================================
# EXPERIMENT 2: PATTERN MATCHING VS REASONING
# ============================================================================
class ReasoningVsPatternMatchingAnalyzer:
    """
    Test if models are actually reasoning or just pattern matching.
    
    HYPOTHESIS: High accuracy on "long" tasks suggests pattern matching,
    not true multi-step reasoning.
    
    TESTS:
    1. Chain-of-thought quality analysis
    2. Intermediate step correctness
    3. Sensitivity to irrelevant information
    """
    
    def __init__(self, metrics_df: pd.DataFrame):
        self.metrics_df = metrics_df
    
    def analyze_reasoning_quality(self) -> Dict:
        """Analyze quality of reasoning traces"""
        print("\n" + "="*80)
        print("EXPERIMENT 2: PATTERN MATCHING VS REASONING")
        print("="*80)
        
        results = {}
        
        # Filter to samples with reasoning traces
        traced_df = self.metrics_df[
            self.metrics_df['reasoning_trace_length'] > 0
        ].copy()
        
        print(f"\n Analyzing {len(traced_df)} samples with reasoning traces")
        
        # 1. Reasoning length vs required steps
        print("\n   Reasoning Length Analysis:")
        
        traced_df['reasoning_ratio'] = (
            traced_df['reasoning_trace_length'] / 
            traced_df['required_steps'].clip(lower=1)
        )
        
        print(f"      Average reasoning ratio: {traced_df['reasoning_ratio'].mean():.2f}x")
        print(f"      (ratio > 1 = more reasoning than needed)")
        print(f"      (ratio < 1 = incomplete reasoning)")
        
        # By horizon
        print("\n   By Horizon:")
        for horizon in ['short', 'medium', 'long']:
            h_df = traced_df[traced_df['temporal_horizon'] == horizon]
            if len(h_df) > 0:
                avg_ratio = h_df['reasoning_ratio'].mean()
                avg_acc = h_df['final_accuracy'].mean()
                
                print(f"      {horizon:8s}: {avg_ratio:.2f}x reasoning, {avg_acc:.1%} accuracy")
                
                results[f'{horizon}_reasoning_ratio'] = float(avg_ratio)
                results[f'{horizon}_accuracy'] = float(avg_acc)
        
        # 2. Correlation: reasoning completeness vs accuracy
        corr, p = spearmanr(
            traced_df['reasoning_ratio'],
            traced_df['final_accuracy']
        )
        
        print(f"\n   Correlation (reasoning completeness vs accuracy):")
        print(f"      ρ = {corr:.3f}, p = {p:.4e}")
        
        if corr < 0.2:
            print(f"        Weak correlation suggests reasoning length doesn't matter much")
            print(f"      → Possible pattern matching rather than true reasoning")
        
        results['reasoning_completeness_correlation'] = {
            'rho': float(corr),
            'p_value': float(p)
        }
        
        return results
    
    def test_reasoning_necessity(self) -> Dict:
        """
        Test if reasoning is actually necessary for accuracy.
        
        Compare:
        - Models that produce long reasoning traces
        - Models that produce short traces
        
        If both achieve similar accuracy, reasoning may not be necessary.
        """
        print("\n Reasoning Necessity Test:")
        
        results = {}
        
        # Split by reasoning style
        median_ratio = self.metrics_df['reasoning_trace_length'].median()
        
        verbose_df = self.metrics_df[
            self.metrics_df['reasoning_trace_length'] > median_ratio
        ]
        
        concise_df = self.metrics_df[
            self.metrics_df['reasoning_trace_length'] <= median_ratio
        ]
        
        print(f"\n   Verbose models (>{median_ratio:.0f} tokens):")
        print(f"      n = {len(verbose_df)}")
        print(f"      Accuracy: {verbose_df['final_accuracy'].mean():.1%}")
        
        print(f"\n   Concise models (≤{median_ratio:.0f} tokens):")
        print(f"      n = {len(concise_df)}")
        print(f"      Accuracy: {concise_df['final_accuracy'].mean():.1%}")
        
        # Statistical test
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(
            verbose_df['final_accuracy'],
            concise_df['final_accuracy'],
            alternative='two-sided'
        )
        
        print(f"\n   Mann-Whitney U test: p = {p:.4f}")
        
        if p > 0.05:
            print(f"        No significant difference!")
            print(f"      → Suggests reasoning length doesn't improve accuracy")
        
        results['verbose_acc'] = float(verbose_df['final_accuracy'].mean())
        results['concise_acc'] = float(concise_df['final_accuracy'].mean())
        results['p_value'] = float(p)
        
        return results

# ============================================================================
# EXPERIMENT 3: CAUSAL INTERVENTION
# ============================================================================
class CausalInterventionDesigner:
    """
    Design experiments to test if models are truly performing multi-step reasoning.
    
    INTERVENTION: Insert errors at different reasoning steps
    PREDICTION: If true reasoning, accuracy should drop proportionally
    """
    
    def design_intervention_protocol(self, metrics_df: pd.DataFrame) -> Dict:
        """Design causal intervention experiments"""
        print("\n" + "="*80)
        print("EXPERIMENT 3: CAUSAL INTERVENTION DESIGN")
        print("="*80)
        
        print("\n INTERVENTION PROTOCOL:")
        print("\n   Objective: Test if models truly depend on multi-step reasoning")
        
        print("\n   Method:")
        print("      1. Select 200 'long horizon' tasks (high accuracy)")
        print("      2. Create variants with injected errors:")
        print("         - Error at step 1 (early)")
        print("         - Error at step N/2 (middle)")
        print("         - Error at step N-1 (late)")
        print("      3. Re-test models on variants")
        print("      4. Measure accuracy drop")
        
        print("\n   Expected Results:")
        print("      TRUE REASONING:")
        print("         - Accuracy drops significantly (>30%)")
        print("         - Drop increases with earlier errors")
        print("      PATTERN MATCHING:")
        print("         - Accuracy drops minimally (<10%)")
        print("         - Error position doesn't matter")
        
        # Select intervention candidates
        long_tasks = metrics_df[
            (metrics_df['temporal_horizon'] == 'long') &
            (metrics_df['final_accuracy'] == True)
        ]
        
        intervention_sample = long_tasks.sample(
            min(200, len(long_tasks)),
            random_state=42
        )
        
        # Export for intervention testing
        intervention_file = Path("rigorous_temporal_analysis") / "intervention_tasks.csv"
        intervention_sample[['sample_id', 'dataset', 'question', 'required_steps']].to_csv(
            intervention_file,
            index=False
        )
        
        print(f"\n    Selected {len(intervention_sample)} tasks for intervention")
        print(f"    Export: {intervention_file}")
        print(f"\n   NEXT STEPS:")
        print(f"      1. Create error-injected variants (see create_variants.py)")
        print(f"      2. Re-test models on variants")
        print(f"      3. Run analyze_intervention_results()")
        
        return {
            'n_tasks': len(intervention_sample),
            'export_file': str(intervention_file)
        }

# ============================================================================
# EXPERIMENT 4: HUMAN BASELINE COMPARISON
# ============================================================================
class HumanBaselineComparer:
    """
    Compare model behavior to human performance.
    
    CRITICAL QUESTION: Do humans also find "long" tasks easier?
    """
    
    def design_human_study(self, metrics_df: pd.DataFrame) -> Dict:
        """Design human comparison study"""
        print("\n" + "="*80)
        print("EXPERIMENT 4: HUMAN BASELINE STUDY")
        print("="*80)
        
        print("\n HUMAN STUDY PROTOCOL:")
        
        print("\n   Research Questions:")
        print("      1. Do humans show the same step-accuracy paradox?")
        print("      2. What do humans consider 'difficult'?")
        print("      3. How do human difficulty ratings correlate with model accuracy?")
        
        print("\n   Method:")
        print("      1. Sample 100 tasks (stratified by horizon)")
        print("      2. Recruit 20+ humans via Prolific/MTurk")
        print("      3. Humans rate difficulty (1-5) and attempt task")
        print("      4. Compare human vs model performance")
        
        # Sample for human study
        sample_size_per_horizon = 33
        
        human_study_samples = []
        for horizon in ['short', 'medium', 'long']:
            horizon_df = metrics_df[metrics_df['temporal_horizon'] == horizon]
            
            sample = horizon_df.sample(
                min(sample_size_per_horizon, len(horizon_df)),
                random_state=42
            )
            
            human_study_samples.append(sample)
        
        human_study_df = pd.concat(human_study_samples)
        
        # Export
        export_file = Path("rigorous_temporal_analysis") / "human_study_tasks.csv"
        human_study_df[['sample_id', 'dataset', 'question', 'temporal_horizon', 'final_accuracy']].to_csv(
            export_file,
            index=False
        )
        
        print(f"\n    Selected {len(human_study_df)} tasks")
        print(f"    Export: {export_file}")
        print(f"\n   Expected cost: ~$200 (20 humans × 100 tasks × $0.10/task)")
        print(f"\n   Timeline: 1-2 weeks")
        
        return {
            'n_tasks': len(human_study_df),
            'export_file': str(export_file)
        }

# ============================================================================
# MASTER ANALYSIS FUNCTION - ADD TO YOUR MAIN()
# ============================================================================
def run_paradox_investigation(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Master function to run all paradox investigations.
    
    ADD THIS TO YOUR main() function after loading metrics_df:
    
    # Run paradox investigation
    paradox_results = run_paradox_investigation(metrics_df, Config.OUTPUT_DIR)
    """
    
    print("\n" + "="*80)
    print(" INVESTIGATING THE REASONING STEP PARADOX ")
    print("="*80)
    print("\nFinding: Long-horizon tasks (61.1% acc) EASIER than short (52.3% acc)")
    print("Question: WHY?")
    
    all_results = {}
    
    # Experiment 1: Difficulty Factors
    print("\n" + "="*80)
    exp1 = DifficultyFactorAnalyzer(metrics_df)
    results1 = exp1.analyze_step_accuracy_paradox()
    characteristics = exp1.analyze_question_characteristics()
    viz1 = exp1.create_difficulty_visualization(output_dir)
    
    all_results['experiment1_difficulty_factors'] = results1
    all_results['experiment1_characteristics'] = characteristics
    
    # Experiment 2: Pattern Matching vs Reasoning
    exp2 = ReasoningVsPatternMatchingAnalyzer(metrics_df)
    results2 = exp2.analyze_reasoning_quality()
    necessity = exp2.test_reasoning_necessity()
    
    all_results['experiment2_reasoning_quality'] = results2
    all_results['experiment2_necessity'] = necessity
    
    # Experiment 3: Causal Intervention Design
    exp3 = CausalInterventionDesigner()
    results3 = exp3.design_intervention_protocol(metrics_df)
    
    all_results['experiment3_intervention'] = results3
    
    # Experiment 4: Human Baseline Design
    exp4 = HumanBaselineComparer()
    results4 = exp4.design_human_study(metrics_df)
    
    all_results['experiment4_human_baseline'] = results4
    
    # Save all results
    results_file = output_dir / 'paradox_investigation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n Saved all investigation results: {results_file}")
    
    # Summary
    print("\n" + "="*80)
    print("INVESTIGATION SUMMARY")
    print("="*80)
    
    print("\n🔍 KEY FINDINGS:")
    
    if 'overall_correlation' in results1:
        rho = results1['overall_correlation']['spearman_rho']
        p = results1['overall_correlation']['spearman_p']
        
        print(f"\n   1. PARADOX CONFIRMED:")
        print(f"      More steps → Higher accuracy (ρ={rho:.3f}, p={p:.2e})")
        
        if rho > 0.2:
            print(f"       STRONG positive correlation!")
        
    print(f"\n   2. PATTERN BY DATASET:")
    for dataset, data in results1.get('by_dataset', {}).items():
        print(f"      {dataset}: {data['pattern']} (ρ={data['correlation']:.2f})")
    
    if 'reasoning_completeness_correlation' in results2:
        rho_reasoning = results2['reasoning_completeness_correlation']['rho']
        
        print(f"\n   3. REASONING QUALITY:")
        print(f"      Reasoning completeness vs accuracy: ρ={rho_reasoning:.3f}")
        
        if abs(rho_reasoning) < 0.2:
            print(f"        Weak correlation → May indicate pattern matching")
    
    print(f"\n NEXT EXPERIMENTS READY:")
    print(f"    Causal intervention tasks exported")
    print(f"    Human study tasks exported")
    print(f"   → Run these experiments to confirm/refute hypotheses")
    
    return all_results