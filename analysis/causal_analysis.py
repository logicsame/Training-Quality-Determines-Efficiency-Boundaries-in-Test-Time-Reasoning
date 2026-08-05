"""
CAUSAL INTERVENTION ABLATION STUDIES
=====================================
Comprehensive ablation studies for the brevity intervention experiment

Tests:
1. Problem-level heterogeneity (does it work for all problems?)
2. Model-specific effects (which models benefit most?)
3. Dataset-specific robustness (generalizable?)
4. Dose-response analysis (optimal verbosity level)
5. Bootstrap confidence intervals
6. Sensitivity to outliers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_rel, spearmanr
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION
# ============================================================================
class InterventionConfig:
    """Configuration for intervention ablations"""
    
    CAUSAL_DIR = Path("data/raw/causal_responses")
    OUTPUT_DIR = Path("./intervention_ablation_results")
    
    # Inverse problems tested [manually loaded from inverse_scaling_problem_ids.json]
    INVERSE_PROBLEMS = {
        "gsm8k": [40, 49, 61, 113, 121, 137, 211, 215, 222, 232, 235, 239, 248],
        "boolq": [2, 12, 21, 37, 42, 49, 66, 88, 102, 114, 119, 121, 122, 124, 136, 162, 174, 177, 182, 188, 200, 210, 224, 229, 241, 249, 254, 261, 262, 265, 278, 283, 292, 293],
        "arc-easy": [0, 9, 27, 38, 48, 54, 55, 63, 66, 71, 103, 104, 123, 127, 151, 173, 179, 184, 189, 210, 214, 215, 222, 232, 245, 253, 255, 280],
        "commonsenseqa": [10, 33, 57, 63, 75, 88, 91, 118, 135, 136, 140, 144, 172, 174, 185, 190, 199, 211, 216, 230, 231, 232, 241, 249, 265, 269, 270, 275, 299],
        "mmlu-stem": [11, 49, 61, 92, 102, 126, 155, 246, 250, 258, 282]
    }
    
    MODEL_SIZES = {
        'gemma-2b-it': 2.0,
        'meta-llama/Llama-3.2-3B-Instruct': 3.0,
        'Qwen/Qwen2.5-3B-Instruct': 3.0,
        "Qwen/Qwen3-next-80b-a3b-instruct": 80.0,
        'Qwen/Qwen2.5-32B-Instruct': 32.0,
        'meta/llama-3.3-70b-instruct': 70.0,
        'meta-llama-3-1-405b-instruct': 405.0
    }

class InterventionDataLoader:
    """Load intervention experiment data"""
    
    def __init__(self, causal_dir: Path):
        self.causal_dir = causal_dir
    
    def load_intervention_data(self) -> pd.DataFrame:
        """Load all intervention data with maximum robustness"""
        print("="*80)
        print("LOADING INTERVENTION DATA")
        print("="*80)
        
        all_data = []
        
        # Find all model directories
        model_dirs = [d for d in self.causal_dir.iterdir() if d.is_dir()]
        
        print(f"\nFound {len(model_dirs)} model directories")
        
        for model_dir in model_dirs:
            model_name = self._parse_model_name(model_dir.name)
            
            for dataset in ['gsm8k', 'boolq', 'arc-easy', 'commonsenseqa', 'mmlu-stem']:
                # Find ANY folder containing 'raw_respon'
                dataset_folders = list((model_dir / dataset).glob("raw_respon*"))
                
                if not dataset_folders:
                    continue
                
                dataset_path = dataset_folders[0]
                
                # For each condition, find ANY CSV file containing the condition name
                for condition in ['control', 'brief', 'direct']:
                    csv_files = list(dataset_path.glob(f"*{condition}*.csv*"))
                    
                    if not csv_files:
                        continue
                    
                    csv_file = csv_files[0]
                    
                    try:
                        # Try multiple read methods
                        df = None
                        
                        # Method 1: Standard read
                        try:
                            df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='utf-8')
                        except:
                            pass
                        
                        # Method 2: Different encoding
                        if df is None:
                            try:
                                df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='latin-1')
                            except:
                                pass
                        
                        # Method 3: Force string dtype for all columns initially
                        if df is None:
                            try:
                                df = pd.read_csv(csv_file, dtype=str, on_bad_lines='skip')
                            except:
                                pass
                        
                        if df is None:
                            print(f" All methods failed for {csv_file.name}")
                            continue
                        
                        # Verify required columns
                        required_cols = ['is_correct', 'sample_id']
                        if not all(col in df.columns for col in required_cols):
                            print(f" Missing required columns in {csv_file.name}")
                            print(f"   Available columns: {df.columns.tolist()}")
                            continue
                        
                        # Convert is_correct to boolean if needed
                        if df['is_correct'].dtype == 'object':
                            df['is_correct'] = df['is_correct'].map({'True': True, 'False': False, True: True, False: False})
                        
                        df['model_name'] = model_name
                        df['dataset'] = dataset
                        df['condition'] = condition
                        
                        # Add model size
                        df['model_size'] = InterventionConfig.MODEL_SIZES.get(model_name, 0)
                        df['size_category'] = np.where(df['model_size'] < 10, 'small', 'large') 
                        all_data.append(df)
                        print(f" Loaded {len(df)} rows: {model_name}/{dataset}/{condition}")
                        
                    except Exception as e:
                        print(f" Error loading {csv_file.name}: {e}")
                        import traceback
                        traceback.print_exc()
        
        if not all_data:
            raise ValueError("No intervention data found!")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n Loaded {len(combined_df)} responses")
        print(f"   Models: {combined_df['model_name'].nunique()}")
        print(f"   Datasets: {combined_df['dataset'].nunique()}")
        print(f"   Conditions: {combined_df['condition'].nunique()}")
        print(f"   Problems: {combined_df['sample_id'].nunique()}")
        
        return combined_df
    
    def _parse_model_name(self, dirname: str) -> str:
        """Parse model name from directory"""
        dirname = dirname.replace('_model', '')
        
        # Normalize names
        name_map = {
            'gemma-2b-it': 'gemma-2b-it',
            'meta-llamaLlama-3.2-3B-Instruct': 'meta-llama/Llama-3.2-3B-Instruct',
            'QwenQwen2.5-3B-Instruct': 'Qwen/Qwen2.5-3B-Instruct',
            'qwenqwen3-32b': 'Qwen/Qwen2.5-32B-Instruct',
            'llama-3.3-70b-versatile': 'meta/llama-3.3-70b-versatile',
            'databricks-meta-llama-3-1-405b-instruct': 'databricks-meta-llama-3-1-405b-instruct'
        }
        
        return name_map.get(dirname, dirname)

# ============================================================================
# ABLATION 1: PROBLEM-LEVEL HETEROGENEITY
# ============================================================================
class ProblemLevelAblation:
    """Test if intervention works uniformly across all problems"""
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
    
    def analyze_problem_heterogeneity(self) -> Dict:
        """Analyze effect at individual problem level"""
        print("\n" + "="*80)
        print("ABLATION 1: PROBLEM-LEVEL HETEROGENEITY")
        print("="*80)
        print("\n QUESTION: Does brevity help on ALL problems or just some?")
        
        results = {}
        
        # For each problem, compute gap reduction
        problem_effects = []
        
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            
            for problem_id in dataset_df['sample_id'].unique():
                problem_df = dataset_df[dataset_df['sample_id'] == problem_id]
                
                # Control gap
                control_df = problem_df[problem_df['condition'] == 'control']
                control_small = control_df[control_df['size_category'] == 'small']['is_correct'].mean()
                control_large = control_df[control_df['size_category'] == 'large']['is_correct'].mean()
                control_gap = control_small - control_large
                
                # Brief gap
                brief_df = problem_df[problem_df['condition'] == 'brief']
                brief_small = brief_df[brief_df['size_category'] == 'small']['is_correct'].mean()
                brief_large = brief_df[brief_df['size_category'] == 'large']['is_correct'].mean()
                brief_gap = brief_small - brief_large
                
                # Effect
                if not np.isnan(control_gap) and not np.isnan(brief_gap):
                    gap_reduction = control_gap - brief_gap
                    
                    problem_effects.append({
                        'dataset': dataset,
                        'problem_id': problem_id,
                        'control_gap': control_gap,
                        'brief_gap': brief_gap,
                        'gap_reduction': gap_reduction,
                        'helped': gap_reduction > 0.05  # >5% improvement
                    })
        
        effects_df = pd.DataFrame(problem_effects)
        
        if len(effects_df) == 0:
            print("\n Insufficient data for problem-level analysis")
            return {}
        
        # Summary statistics
        print(f"\n Analysis of {len(effects_df)} problems:")
        
        helped_count = effects_df['helped'].sum()
        helped_pct = helped_count / len(effects_df) * 100
        
        print(f"\n   Problems where brevity HELPED: {helped_count}/{len(effects_df)} ({helped_pct:.1f}%)")
        print(f"   Problems where brevity HURT: {len(effects_df) - helped_count}/{len(effects_df)} ({100-helped_pct:.1f}%)")
        
        print(f"\n   Gap reduction statistics:")
        print(f"      Mean: {effects_df['gap_reduction'].mean():+.3f}")
        print(f"      Median: {effects_df['gap_reduction'].median():+.3f}")
        print(f"      Std: {effects_df['gap_reduction'].std():.3f}")
        print(f"      Range: [{effects_df['gap_reduction'].min():+.3f}, {effects_df['gap_reduction'].max():+.3f}]")
        
        # Test consistency
        # If intervention is robust, gap_reduction should be consistently positive
        one_sample_t, p_val = stats.ttest_1samp(effects_df['gap_reduction'], 0)
        
        print(f"\n   One-sample t-test (H0: mean gap reduction = 0):")
        print(f"      t = {one_sample_t:.3f}, p = {p_val:.4f}")
        
        if p_val < 0.05 and effects_df['gap_reduction'].mean() > 0:
            print(f"       CONSISTENT EFFECT: Brevity reliably helps")
        elif p_val >= 0.05:
            print(f"       INCONSISTENT: Effect varies widely across problems")
        else:
            print(f"       HARMFUL: Brevity makes things worse on average")
        
        # By dataset
        print(f"\n   By dataset:")
        for dataset in effects_df['dataset'].unique():
            ds_effects = effects_df[effects_df['dataset'] == dataset]
            helped_ds = ds_effects['helped'].sum() / len(ds_effects) * 100
            mean_reduction = ds_effects['gap_reduction'].mean()
            
            marker = "" if helped_ds > 60 else "⚠️" if helped_ds > 40 else "❌"
            print(f"      {marker} {dataset:<20}: {helped_ds:>5.1f}% helped, avg reduction = {mean_reduction:+.3f}")
        
        # Identify best and worst problems
        print(f"\n   🔝 TOP 5 Problems (biggest gap reduction):")
        top5 = effects_df.nlargest(5, 'gap_reduction')
        for _, row in top5.iterrows():
            print(f"      {row['dataset']:>15} #{row['problem_id']:<4}: {row['gap_reduction']:+.3f}")
        
        print(f"\n    WORST 5 Problems (gap increased):")
        worst5 = effects_df.nsmallest(5, 'gap_reduction')
        for _, row in worst5.iterrows():
            print(f"      {row['dataset']:>15} #{row['problem_id']:<4}: {row['gap_reduction']:+.3f}")
        
        results = {
            'effects_df': effects_df,
            'helped_pct': helped_pct,
            'mean_reduction': effects_df['gap_reduction'].mean(),
            'median_reduction': effects_df['gap_reduction'].median(),
            't_statistic': one_sample_t,
            'p_value': p_val,
            'consistent': p_val < 0.05 and effects_df['gap_reduction'].mean() > 0
        }
        
        return results

# ============================================================================
# ABLATION 2: MODEL-SPECIFIC EFFECTS
# ============================================================================
class ModelSpecificAblation:
    """Test which models benefit most from brevity"""
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
    
    def analyze_model_effects(self) -> Dict:
        """Analyze effect for each model"""
        print("\n" + "="*80)
        print("ABLATION 2: MODEL-SPECIFIC EFFECTS")
        print("="*80)
        print("\n🎯 QUESTION: Which models benefit most from brevity?")
        
        results = {}
        
        # Only analyze large models (small models already perform well)
        large_df = self.df[self.df['size_category'] == 'large']
        
        print(f"\n Analyzing {large_df['model_name'].nunique()} large models:")
        
        model_effects = []
        
        for model in large_df['model_name'].unique():
            model_df = large_df[large_df['model_name'] == model]
            
            # Control accuracy
            control_acc = model_df[model_df['condition'] == 'control']['is_correct'].mean()
            control_n = len(model_df[model_df['condition'] == 'control'])
            
            # Brief accuracy
            brief_acc = model_df[model_df['condition'] == 'brief']['is_correct'].mean()
            brief_n = len(model_df[model_df['condition'] == 'brief'])
            
            # Improvement
            improvement = brief_acc - control_acc
            
            if control_n > 0 and brief_n > 0:
                # Statistical test
                control_vals = model_df[model_df['condition'] == 'control']['is_correct']
                brief_vals = model_df[model_df['condition'] == 'brief']['is_correct']
                
                if len(control_vals) > 5 and len(brief_vals) > 5:
                    _, p_val = mannwhitneyu(brief_vals, control_vals, alternative='greater')
                else:
                    p_val = 1.0
                
                model_effects.append({
                    'model': model,
                    'size': InterventionConfig.MODEL_SIZES.get(model, 0),
                    'control_acc': control_acc,
                    'brief_acc': brief_acc,
                    'improvement': improvement,
                    'p_value': p_val,
                    'n_control': control_n,
                    'n_brief': brief_n
                })
        
        effects_df = pd.DataFrame(model_effects).sort_values('improvement', ascending=False)
        
        print(f"\n   {'Model':<40} {'Size':>8} {'Control':>10} {'Brief':>10} {'Δ':>10} {'p-val':>8}")
        print(f"   {'-'*95}")
        
        for _, row in effects_df.iterrows():
            marker = "" if row['improvement'] > 0.05 else "⚠️" if row['improvement'] > 0 else "❌"
            sig = "*" if row['p_value'] < 0.05 else " "
            
            print(f"   {marker} {row['model']:<37} {row['size']:>7.0f}B {row['control_acc']:>9.1%} "
                  f"{row['brief_acc']:>9.1%} {row['improvement']:>+9.1%} {row['p_value']:>7.3f}{sig}")
        
        # Correlation with size
        if len(effects_df) >= 3:
            corr, p_corr = spearmanr(effects_df['size'], effects_df['improvement'])
            
            print(f"\n   Size vs Improvement correlation:")
            print(f"      ρ = {corr:+.3f}, p = {p_corr:.4f}")
            
            if abs(corr) > 0.5 and p_corr < 0.05:
                if corr > 0:
                    print(f"       FINDING: LARGER models benefit MORE from brevity")
                else:
                    print(f"       FINDING: SMALLER models benefit MORE from brevity")
            else:
                print(f"       FINDING: Benefit is SIZE-INDEPENDENT")
        
        results = {
            'effects_df': effects_df,
            'best_model': effects_df.iloc[0]['model'],
            'best_improvement': effects_df.iloc[0]['improvement'],
            'worst_model': effects_df.iloc[-1]['model'],
            'worst_improvement': effects_df.iloc[-1]['improvement']
        }
        
        return results

# ============================================================================
# ABLATION 3: DATASET-SPECIFIC ROBUSTNESS
# ============================================================================
class DatasetRobustnessAblation:
    """Test if effect generalizes across datasets"""
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
    
    def analyze_dataset_robustness(self) -> Dict:
        """Test generalization across datasets"""
        print("\n" + "="*80)
        print("ABLATION 3: DATASET-SPECIFIC ROBUSTNESS")
        print("="*80)
        print("\n QUESTION: Does effect generalize across all datasets?")
        
        results = {}
        
        dataset_effects = []
        
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            
            # Large models only
            large_df = dataset_df[dataset_df['size_category'] == 'large']
            
            # Control vs Brief
            control_acc = large_df[large_df['condition'] == 'control']['is_correct'].mean()
            brief_acc = large_df[large_df['condition'] == 'brief']['is_correct'].mean()
            
            improvement = brief_acc - control_acc
            
            # Sample sizes
            n_control = len(large_df[large_df['condition'] == 'control'])
            n_brief = len(large_df[large_df['condition'] == 'brief'])
            
            # Number of inverse problems in this dataset
            n_inverse = len(InterventionConfig.INVERSE_PROBLEMS.get(dataset, []))
            
            dataset_effects.append({
                'dataset': dataset,
                'control_acc': control_acc,
                'brief_acc': brief_acc,
                'improvement': improvement,
                'n_inverse': n_inverse,
                'n_control': n_control,
                'n_brief': n_brief
            })
        
        effects_df = pd.DataFrame(dataset_effects).sort_values('improvement', ascending=False)
        
        print(f"\n   {'Dataset':<20} {'N Inv':>8} {'Control':>10} {'Brief':>10} {'Improvement':>12}")
        print(f"   {'-'*70}")
        
        for _, row in effects_df.iterrows():
            marker = "" if row['improvement'] > 0.05 else "⚠️" if row['improvement'] > 0 else "❌"
            
            print(f"   {marker} {row['dataset']:<17} {row['n_inverse']:>8} {row['control_acc']:>9.1%} "
                  f"{row['brief_acc']:>9.1%} {row['improvement']:>+11.1%}")
        
        # Test consistency across datasets
        improvements = effects_df['improvement'].values
        
        one_sample_t, p_val = stats.ttest_1samp(improvements, 0)
        
        print(f"\n   Consistency test:")
        print(f"      Mean improvement: {np.mean(improvements):+.1%}")
        print(f"      Std improvement: {np.std(improvements):.1%}")
        print(f"      One-sample t-test: t = {one_sample_t:.3f}, p = {p_val:.4f}")
        
        positive_count = sum(1 for i in improvements if i > 0)
        
        print(f"\n      Datasets with positive effect: {positive_count}/{len(improvements)}")
        
        if positive_count == len(improvements):
            print(f"       ROBUST: Effect generalizes perfectly")
        elif positive_count >= len(improvements) * 0.8:
            print(f"       MOSTLY ROBUST: Effect generalizes well")
        elif positive_count >= len(improvements) * 0.5:
            print(f"       MIXED: Effect is dataset-dependent")
        else:
            print(f"       NOT ROBUST: Effect doesn't generalize")
        
        results = {
            'effects_df': effects_df,
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements),
            'positive_count': positive_count,
            'total_datasets': len(improvements),
            'robust': positive_count >= len(improvements) * 0.8
        }
        
        return results

# ============================================================================
# ABLATION 4: DOSE-RESPONSE ANALYSIS
# ============================================================================
class DoseResponseAblation:
    """Analyze if there's optimal response length"""
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
    
    def analyze_dose_response(self) -> Dict:
        """Analyze accuracy vs response length"""
        print("\n" + "="*80)
        print("ABLATION 4: DOSE-RESPONSE ANALYSIS")
        print("="*80)
        print("\n QUESTION: Is there an optimal response length?")
        
        # Compute response lengths
        self.df['response_length'] = self.df['full_generation'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Focus on large models
        large_df = self.df[self.df['size_category'] == 'large']
        
        print(f"\n Analyzing {len(large_df)} responses from large models")
        
        # Bin by response length
        large_df['length_bin'] = pd.qcut(
            large_df['response_length'],
            q=5,
            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'],
            duplicates='drop'
        )
        
        # Accuracy by length bin
        length_acc = large_df.groupby('length_bin')['is_correct'].agg(['mean', 'count', 'std'])
        
        print(f"\n   Accuracy by response length:")
        print(f"   {'Length Bin':<15} {'Accuracy':>12} {'N':>8} {'SD':>8}")
        print(f"   {'-'*50}")
        
        for idx, row in length_acc.iterrows():
            print(f"   {idx:<15} {row['mean']:>11.1%} {int(row['count']):>8} {row['std']:>7.3f}")
        
        # Find optimal
        optimal_bin = length_acc['mean'].idxmax()
        optimal_acc = length_acc.loc[optimal_bin, 'mean']
        
        print(f"\n    OPTIMAL LENGTH: {optimal_bin}")
        print(f"      Accuracy: {optimal_acc:.1%}")
        
        # Correlation
        corr, p_val = spearmanr(large_df['response_length'], large_df['is_correct'])
        
        print(f"\n   Correlation (length vs accuracy):")
        print(f"      ρ = {corr:+.3f}, p = {p_val:.4f}")
        
        if corr < -0.2 and p_val < 0.05:
            print(f"       FINDING: Shorter is BETTER (negative correlation)")
        elif corr > 0.2 and p_val < 0.05:
            print(f"       FINDING: Longer is BETTER (positive correlation)")
        else:
            print(f"       FINDING: Length doesn't matter much")
        
        # By condition
        print(f"\n   Average length by condition:")
        for condition in ['control', 'brief', 'direct']:
            cond_df = large_df[large_df['condition'] == condition]
            avg_len = cond_df['response_length'].mean()
            avg_acc = cond_df['is_correct'].mean()
            
            print(f"      {condition:<10}: {avg_len:>7.1f} words, {avg_acc:>5.1%} accuracy")
        
        results = {
            'length_accuracy': length_acc.to_dict(),
            'optimal_bin': optimal_bin,
            'optimal_accuracy': optimal_acc,
            'correlation': corr,
            'p_value': p_val
        }
        
        return results

# ============================================================================
# ABLATION 5: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
class BootstrapAblation:
    """Bootstrap confidence intervals for all effects"""
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
    
    def bootstrap_confidence(self, n_iterations: int = 1000) -> Dict:
        """Bootstrap 95% CIs"""
        print("\n" + "="*80)
        print("ABLATION 5: BOOTSTRAP CONFIDENCE INTERVALS")
        print("="*80)
        print(f"\n Computing 95% CIs with {n_iterations} bootstrap iterations")
        
        large_df = self.df[self.df['size_category'] == 'large']
        
        # Bootstrap the main effect
        bootstrap_improvements = []
        
        for i in range(n_iterations):
            # Resample with replacement
            sample_df = large_df.sample(n=len(large_df), replace=True)
            
            # Compute effect
            control_acc = sample_df[sample_df['condition'] == 'control']['is_correct'].mean()
            brief_acc = sample_df[sample_df['condition'] == 'brief']['is_correct'].mean()
            
            improvement = brief_acc - control_acc
            bootstrap_improvements.append(improvement)
        
        # Compute CI
        ci_lower = np.percentile(bootstrap_improvements, 2.5)
        ci_upper = np.percentile(bootstrap_improvements, 97.5)
        point_estimate = np.mean(bootstrap_improvements)
        
        print(f"\n   Main effect (brevity improvement):")
        print(f"      Point estimate: {point_estimate:+.1%}")
        print(f"      95% CI: [{ci_lower:+.1%}, {ci_upper:+.1%}]")
        
        if ci_lower > 0:
            print(f"       SIGNIFICANT: CI excludes zero")
        else:
            print(f"       NOT SIGNIFICANT: CI includes zero")
        
        # Bootstrap by dataset
        print(f"\n   By dataset:")
        
        dataset_cis = {}
        
        for dataset in large_df['dataset'].unique():
            dataset_df = large_df[large_df['dataset'] == dataset]
            
            if len(dataset_df) < 20:
                continue
            
            bootstrap_ds = []
            
            for i in range(n_iterations):
                sample = dataset_df.sample(n=len(dataset_df), replace=True)
                
                control_acc = sample[sample['condition'] == 'control']['is_correct'].mean()
                brief_acc = sample[sample['condition'] == 'brief']['is_correct'].mean()
                
                bootstrap_ds.append(brief_acc - control_acc)
            
            ci_low = np.percentile(bootstrap_ds, 2.5)
            ci_high = np.percentile(bootstrap_ds, 97.5)
            
            dataset_cis[dataset] = (ci_low, ci_high)
            
            sig = "✅" if ci_low > 0 else "⚠️"
            print(f"      {sig} {dataset:<20}: [{ci_low:+.1%}, {ci_high:+.1%}]")
        
        results = {
            'overall_ci': (ci_lower, ci_upper),
            'point_estimate': point_estimate,
            'significant': ci_lower > 0,
            'dataset_cis': dataset_cis
        }
        
        return results

# ============================================================================
# ABLATION 6: SENSITIVITY TO OUTLIERS
# ============================================================================
class OutlierSensitivityAblation:
    """Test if results are driven by outliers"""
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
    
    def analyze_outlier_sensitivity(self) -> Dict:
        """Test robustness to outliers"""
        print("\n" + "="*80)
        print("ABLATION 6: SENSITIVITY TO OUTLIERS")
        print("="*80)
        print("\n🎯 QUESTION: Are results driven by a few outlier problems?")
        
        large_df = self.df[self.df['size_category'] == 'large']
        
        # Compute problem-level effects
        problem_effects = []
        
        for problem_id in large_df['sample_id'].unique():
            prob_df = large_df[large_df['sample_id'] == problem_id]
            
            control_acc = prob_df[prob_df['condition'] == 'control']['is_correct'].mean()
            brief_acc = prob_df[prob_df['condition'] == 'brief']['is_correct'].mean()
            
            if not np.isnan(control_acc) and not np.isnan(brief_acc):
                problem_effects.append(brief_acc - control_acc)
        
        if len(problem_effects) < 10:
            print("\n⚠️ Insufficient data")
            return {}
        
        problem_effects = np.array(problem_effects)
        
        # Identify outliers (>2 SD from mean)
        mean_effect = np.mean(problem_effects)
        std_effect = np.std(problem_effects)
        
        outliers = np.abs(problem_effects - mean_effect) > 2 * std_effect
        n_outliers = np.sum(outliers)
        
        print(f"\n   Outlier detection (>2 SD from mean):")
        print(f"      Total problems: {len(problem_effects)}")
        print(f"      Outliers: {n_outliers} ({n_outliers/len(problem_effects)*100:.1f}%)")
        
        # Effect with and without outliers
        effect_with = np.mean(problem_effects)
        effect_without = np.mean(problem_effects[~outliers])
        
        print(f"\n   Effect estimates:")
        print(f"      With outliers: {effect_with:+.1%}")
        print(f"      Without outliers: {effect_without:+.1%}")
        print(f"      Difference: {abs(effect_with - effect_without):.1%}")
        
        if abs(effect_with - effect_without) < 0.02:
            print(f"      ✅ ROBUST: Results not driven by outliers")
        else:
            print(f"      ⚠️ SENSITIVE: Outliers have significant impact")
        
        results = {
            'n_problems': len(problem_effects),
            'n_outliers': int(n_outliers),
            'effect_with': effect_with,
            'effect_without': effect_without,
            'robust': abs(effect_with - effect_without) < 0.02
        }
        
        return results


# ============================================================================
# ADDITIONAL CAUSAL ABLATION ANALYSES - WHY BREVITY HELPS
# ============================================================================

class ErrorPatternAnalyzer:
    """
    Analyze what TYPES of errors occur in control vs brief conditions
    """
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
        
    def sample_errors_for_annotation(self, n_per_condition: int = 50):
        """Sample incorrect responses for manual annotation"""
        print("\n📊 SAMPLING ERRORS FOR MANUAL ANNOTATION")
        
        # Get only incorrect responses
        errors = self.df[self.df['is_correct'] == False].copy()
        
        if len(errors) < 20:
            print(f"⚠️ Insufficient errors: {len(errors)}")
            return None
        
        # Sample from each condition
        sampled = []
        
        for condition in ['control', 'brief']:
            condition_errors = errors[errors['condition'] == condition]
            
            if len(condition_errors) > 0:
                sample = condition_errors.sample(
                    n=min(n_per_condition, len(condition_errors)),
                    random_state=42
                )
                sampled.append(sample)
        
        if not sampled:
            return None
        
        annotation_df = pd.concat(sampled)
        
        # Add columns for manual annotation
        annotation_df['error_category'] = ''
        annotation_df['is_cumulative'] = ''
        annotation_df['error_location'] = ''
        
        # Save for annotation
        output_path = InterventionConfig.OUTPUT_DIR / 'errors_for_annotation.csv'
        annotation_df.to_csv(output_path, index=False)
        
        print(f"✅ Sampled {len(annotation_df)} errors for annotation")
        print(f"   Saved to: {output_path}")
        
        return annotation_df


class StepByStepAnalyzer:
    """
    Track accuracy at each reasoning step
    """
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
    
    def parse_reasoning_steps(self, text: str) -> List[str]:
        """Extract individual reasoning steps"""
        import re
        
        if pd.isna(text):
            return []
        
        # Try to find calculations first
        calculations = re.findall(
            r'(\d+\s*[\+\-\*\/\=]\s*\d+(?:\s*[\+\-\*\/\=]\s*\d+)*)',
            str(text)
        )
        
        if calculations:
            return calculations[:10]
        
        # Fallback: split by sentences
        sentences = [s.strip() for s in str(text).split('.') if s.strip()]
        return sentences[:10]
    
    def sample_for_step_annotation(self, n_samples: int = 30):
        """Sample responses for step-by-step annotation"""
        print("\n📊 SAMPLING FOR STEP-BY-STEP ANNOTATION")
        
        # Focus on GSM8K (math problems)
        gsm8k = self.df[self.df['dataset'] == 'gsm8k'].copy()
        
        if len(gsm8k) < 10:
            print("⚠️ Insufficient GSM8K data")
            return None
        
        # Sample from each condition
        sampled = []
        
        for condition in ['control', 'brief']:
            condition_df = gsm8k[gsm8k['condition'] == condition]
            
            if len(condition_df) > 0:
                sample = condition_df.sample(
                    n=min(n_samples, len(condition_df)),
                    random_state=42
                )
                sampled.append(sample)
        
        if not sampled:
            return None
        
        annotation_df = pd.concat(sampled)
        
        # Parse steps
        annotation_df['reasoning_steps'] = annotation_df['full_generation'].apply(
            lambda x: self.parse_reasoning_steps(x)
        )
        annotation_df['num_steps'] = annotation_df['reasoning_steps'].apply(len)
        
        # Add annotation columns
        for i in range(1, 6):
            annotation_df[f'step_{i}_correct'] = ''
        
        # Save
        output_path = InterventionConfig.OUTPUT_DIR / 'steps_for_annotation.csv'
        annotation_df.to_csv(output_path, index=False)
        
        print(f"✅ Sampled {len(annotation_df)} responses for step annotation")
        print(f"   Saved to: {output_path}")
        
        return annotation_df


class ReasoningQualityAnalyzer:
    """
    Automated metrics for reasoning quality
    """
    
    def __init__(self, intervention_df: pd.DataFrame):
        self.df = intervention_df
    
    def compute_quality_metrics(self, text: str) -> Dict:
        """Compute automated quality metrics"""
        import re
        from collections import Counter
        
        if pd.isna(text):
            text = ''
        
        metrics = {}
        words = str(text).lower().split()
        
        # Basic stats
        metrics['word_count'] = len(words)
        metrics['char_count'] = len(str(text))
        
        # Vocabulary diversity
        if words:
            metrics['vocabulary_diversity'] = len(set(words)) / len(words)
        else:
            metrics['vocabulary_diversity'] = 0
        
        # Repetition rate
        word_freq = Counter(words)
        repeated = [w for w, c in word_freq.items() if c >= 3]
        if words:
            metrics['repetition_rate'] = len(repeated) / max(1, len(set(words)))
        else:
            metrics['repetition_rate'] = 0
        
        # Reasoning indicators
        reasoning_words = ['because', 'therefore', 'thus', 'hence', 'so']
        metrics['reasoning_words'] = sum(
            1 for w in reasoning_words if w in str(text).lower()
        )
        
        # Calculations
        calculations = re.findall(r'\d+\s*[\+\-\*\/]\s*\d+', str(text))
        metrics['num_calculations'] = len(calculations)
        
        return metrics
    
    def analyze_quality(self):
        """Compare reasoning quality between conditions"""
        print("\n📊 AUTOMATED REASONING QUALITY ANALYSIS")
        
        # Compute metrics
        metrics_list = []
        
        for _, row in self.df.iterrows():
            metrics = self.compute_quality_metrics(row.get('full_generation', ''))
            metrics['condition'] = row['condition']
            metrics['is_correct'] = row['is_correct']
            metrics['dataset'] = row['dataset']
            metrics_list.append(metrics)
        
        if not metrics_list:
            return None, None
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # Compare by condition
        print("\n   Quality metrics comparison:")
        
        quality_metrics = [
            'word_count',
            'vocabulary_diversity', 
            'repetition_rate',
            'reasoning_words',
            'num_calculations'
        ]
        
        comparison = []
        
        for metric in quality_metrics:
            if metric not in metrics_df.columns:
                continue
                
            control_vals = metrics_df[metrics_df['condition'] == 'control'][metric].dropna()
            brief_vals = metrics_df[metrics_df['condition'] == 'brief'][metric].dropna()
            
            if len(control_vals) > 5 and len(brief_vals) > 5:
                control_mean = control_vals.mean()
                brief_mean = brief_vals.mean()
                
                stat, p_val = mannwhitneyu(control_vals, brief_vals)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                
                comparison.append({
                    'metric': metric,
                    'control': control_mean,
                    'brief': brief_mean,
                    'difference': brief_mean - control_mean,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
                
                arrow = "↑" if brief_mean > control_mean else "↓"
                print(f"   {metric:25s}: Control={control_mean:6.2f}, Brief={brief_mean:6.2f} {arrow} (p={p_val:.4f}){sig}")
        
        comparison_df = pd.DataFrame(comparison)
        
        # Save results
        metrics_df.to_csv(InterventionConfig.OUTPUT_DIR / 'reasoning_quality_metrics.csv', index=False)
        comparison_df.to_csv(InterventionConfig.OUTPUT_DIR / 'quality_comparison.csv', index=False)
        
        print(f"\n✅ Quality analysis complete")
        print(f"   Saved metrics to output directory")
        
        return metrics_df, comparison_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all ablation studies"""
    
    print("\n" + "#"*80)
    print("# CAUSAL INTERVENTION ABLATION STUDIES")
    print("#"*80)
    
    # Create output directory
    InterventionConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = InterventionDataLoader(InterventionConfig.CAUSAL_DIR)
    
    try:
        intervention_df = loader.load_intervention_data()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure intervention data is in:")
        print(f"   {InterventionConfig.CAUSAL_DIR}")
        return
    
    # Run all ablations
    all_results = {}
    
    # Ablation 1: Problem-level heterogeneity
    ablation1 = ProblemLevelAblation(intervention_df)
    results1 = ablation1.analyze_problem_heterogeneity()
    all_results['problem_heterogeneity'] = results1
    
    # Ablation 2: Model-specific effects
    ablation2 = ModelSpecificAblation(intervention_df)
    results2 = ablation2.analyze_model_effects()
    all_results['model_specific'] = results2
    
    # Ablation 3: Dataset robustness
    ablation3 = DatasetRobustnessAblation(intervention_df)
    results3 = ablation3.analyze_dataset_robustness()
    all_results['dataset_robustness'] = results3
    
    # Ablation 4: Dose-response
    ablation4 = DoseResponseAblation(intervention_df)
    results4 = ablation4.analyze_dose_response()
    all_results['dose_response'] = results4
    
    # Ablation 5: Bootstrap CIs
    ablation5 = BootstrapAblation(intervention_df)
    results5 = ablation5.bootstrap_confidence()
    all_results['bootstrap'] = results5
    
    # Ablation 6: Outlier sensitivity
    ablation6 = OutlierSensitivityAblation(intervention_df)
    results6 = ablation6.analyze_outlier_sensitivity()
    all_results['outlier_sensitivity'] = results6
    
    # ========================================================================
    # ABLATION 7-9: WHY ANALYSES - Why does brevity help?
    # ========================================================================
    
    print("\n" + "="*80)
    print("WHY ANALYSES: Why does brevity improve reasoning?")
    print("="*80)
    
    # Prepare data - ensure we have the right column names
    if 'model_output' not in intervention_df.columns and 'full_generation' in intervention_df.columns:
        print("\n📝 Note: Using 'full_generation' column for text analysis")
        # We'll use 'full_generation' directly in the analyzers
    
    # Analysis 7: Error Pattern Categorization
    print("\n📊 ANALYSIS 7: ERROR PATTERN CATEGORIZATION")
    print("   Goal: Are control errors more cumulative?")
    
    error_analyzer = ErrorPatternAnalyzer(intervention_df)
    error_sample = error_analyzer.sample_errors_for_annotation(n_per_condition=30)
    
    if error_sample is not None:
        all_results['error_samples_count'] = len(error_sample)
    
    # Analysis 8: Step-by-Step Accuracy Decay  
    print("\n📊 ANALYSIS 8: STEP-BY-STEP ACCURACY DECAY")
    print("   Goal: Does accuracy decay faster in verbose responses?")
    
    step_analyzer = StepByStepAnalyzer(intervention_df)
    step_sample = step_analyzer.sample_for_step_annotation(n_samples=20)
    
    if step_sample is not None:
        all_results['step_samples_count'] = len(step_sample)
    
    # Analysis 9: Automated Reasoning Quality Metrics
    print("\n📊 ANALYSIS 9: AUTOMATED REASONING QUALITY METRICS")
    print("   Goal: Is control reasoning lower quality?")
    
    quality_analyzer = ReasoningQualityAnalyzer(intervention_df)
    quality_metrics, quality_comparison = quality_analyzer.analyze_quality()
    
    if quality_comparison is not None:
        all_results['quality_analysis'] = {
            'metrics_computed': True,
            'significant_differences': quality_comparison[quality_comparison['significant']].to_dict('records')
        }
    
    print("\n" + "-"*80)
    print("WHY ANALYSES: MANUAL ANNOTATION REQUIRED")
    print("-"*80)
    print("\n📋 Next steps for full 'why' analysis:")
    print("   1. Manually annotate:")
    print("      - errors_for_annotation.csv (error types)")
    print("      - steps_for_annotation.csv (step correctness)")
    print("   2. Run separate analysis scripts on annotated data")
    print("\n💡 Key hypotheses to test:")
    print("   • H1: Control responses have more CUMULATIVE errors")
    print("   • H2: Accuracy decays faster in verbose reasoning chains")
    print("   • H3: Control responses are more repetitive/less focused")
    
    # Save results
    output_file = InterventionConfig.OUTPUT_DIR / 'ablation_results_with_why.json'
    
    # Convert dataframes to dicts for JSON serialization
    save_results = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            save_dict = {}
            for k, v in val.items():
                if isinstance(v, pd.DataFrame):
                    save_dict[k] = v.to_dict('records')
                else:
                    save_dict[k] = v
            save_results[key] = save_dict
        else:
            save_results[key] = val
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n✅ Saved results to: {output_file}")
    
    # ========================================================================
    # EXECUTIVE SUMMARY - UPDATED WITH WHY FINDINGS
    # ========================================================================
    
    print("\n" + "#"*80)
    print("# EXECUTIVE SUMMARY: ABLATION + WHY RESULTS")
    print("#"*80)
    
    print("\n🔬 ROBUSTNESS ASSESSMENT:")
    
    # Check 1: Problem-level consistency
    if results1 and 'consistent' in results1:
        if results1['consistent']:
            print(f"\n   ✅ Problem-level: Effect is CONSISTENT across problems")
            print(f"      - {results1.get('helped_pct', 0):.0f}% of problems show improvement")
        else:
            print(f"\n   ⚠️ Problem-level: Effect is VARIABLE")
            print(f"      - Only {results1.get('helped_pct', 0):.0f}% of problems show improvement")
    
    # Check 2: Dataset robustness
    if results3 and 'robust' in results3:
        if results3['robust']:
            print(f"\n   ✅ Dataset-level: Effect GENERALIZES across datasets")
            print(f"      - {results3.get('positive_count', 0)}/{results3.get('total_datasets', 0)} datasets show improvement")
        else:
            print(f"\n   ⚠️ Dataset-level: Effect is DATASET-SPECIFIC")
            print(f"      - Only {results3.get('positive_count', 0)}/{results3.get('total_datasets', 0)} datasets show improvement")
    
    # Check 3: Statistical significance
    if results5 and 'significant' in results5:
        if results5['significant']:
            print(f"\n   ✅ Statistical: Effect is SIGNIFICANT")
            print(f"      - 95% CI: [{results5.get('overall_ci', (0,0))[0]:+.1%}, {results5.get('overall_ci', (0,0))[1]:+.1%}]")
        else:
            print(f"\n   ⚠️ Statistical: Effect is NOT significant")
            print(f"      - 95% CI includes zero")
    
    # Check 4: Outlier robustness
    if results6 and 'robust' in results6:
        if results6['robust']:
            print(f"\n   ✅ Outliers: Results are ROBUST")
            print(f"      - Effect stable with/without outliers")
        else:
            print(f"\n   ⚠️ Outliers: Results are SENSITIVE")
            print(f"      - {results6.get('n_outliers', 0)} outlier problems have large impact")
    
    print("\n🔍 WHY FINDINGS (Initial Insights):")
    
    # Add quality metrics summary if available
    if 'quality_analysis' in all_results and 'significant_differences' in all_results['quality_analysis']:
        sig_diffs = all_results['quality_analysis']['significant_differences']
        
        print(f"\n   Automated quality analysis found {len(sig_diffs)} significant differences:")
        
        for diff in sig_diffs[:3]:  # Show top 3
            metric = diff.get('metric', '')
            p_val = diff.get('p_value', 1.0)
            control_val = diff.get('control', 0)
            brief_val = diff.get('brief', 0)
            
            if p_val < 0.05:
                direction = "higher" if control_val > brief_val else "lower"
                print(f"      • Control has {direction} {metric} (p={p_val:.4f})")
    
    print("\n   Manual analysis required for:")
    print("      • Error categorization (cumulative vs isolated errors)")
    print("      • Step-by-step accuracy decay")
    print("\n   Files ready for annotation:")
    print("      • errors_for_annotation.csv")
    print("      • steps_for_annotation.csv")
    
    # Update overall verdict
    checks_passed = sum([
        results1.get('consistent', False) if results1 else False,
        results3.get('robust', False) if results3 else False,
        results5.get('significant', False) if results5 else False,
        results6.get('robust', False) if results6 else False
    ])
    
    print(f"\n" + "="*80)
    print(f"OVERALL VERDICT: {checks_passed}/4 robustness checks passed")
    print(f"PLUS: WHY analyses initiated (3 new investigations)")
    print(f"="*80)
    
    if checks_passed >= 3:
        print(f"\n✅ STRONG EVIDENCE: Brevity intervention is ROBUST")
        print(f"   → Ready for Nature submission")
    elif checks_passed >= 2:
        print(f"\n⚠️ MODERATE EVIDENCE: Intervention shows promise but has limitations")
        print(f"   → Suitable for Nature Machine Intelligence")
        print(f"   → Consider additional validation")
    else:
        print(f"\n❌ WEAK EVIDENCE: Effect is fragile")
        print(f"   → More work needed before publication")
    
    print(f"\n🎉 ABLATION STUDIES COMPLETE!")

if __name__ == "__main__":
    main()