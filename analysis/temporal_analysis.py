"""
PUBLICATION-GRADE TEMPORAL SCALING ANALYSIS
============================================
Fixed to work with your exact data structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, pearsonr
from statsmodels.formula.api import mixedlm
from statsmodels.stats.power import TTestIndPower
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import re
import ast
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration for rigorous temporal analysis"""
    
    BASE_DIR = Path("./data/raw/phase1_responses/cross_model_archaeology")
    OUTPUT_DIR = Path("./rigorous_temporal_analysis")
    
    DATASETS = ['gsm8k', 'boolq', 'arc-easy', 'commonsenseqa', 'mmlu-stem']
    
    # Model metadata - EXACT same as original
    MODEL_METADATA = {
        'Qwen/Qwen2.5-0.5B-Instruct': {'size': 0.5, 'tier': 'weak', 'family': 'Qwen'},
        'meta-llama/Llama-3.2-1B-Instruct': {'size': 1.0, 'tier': 'weak', 'family': 'Llama'},
        'google/gemma-3-1b-it': {'size': 1.0, 'tier': 'weak', 'family': 'Gemma'},
        'stabilityai/stablelm-2-1_6b': {'size': 1.6, 'tier': 'weak', 'family': 'StableLM'},
        'google/gemma-2-2b-it': {'size': 2.0, 'tier': 'weak', 'family': 'Gemma'},
        'meta-llama/Llama-3.2-3B-Instruct': {'size': 3.0, 'tier': 'weak', 'family': 'Llama'},
        'Qwen/Qwen2.5-3B-Instruct': {'size': 3.0, 'tier': 'weak', 'family': 'Qwen'},
        'stabilityai/stablelm-zephyr-3b': {'size': 3.0, 'tier': 'weak', 'family': 'StableLM'},
        'microsoft/Phi-3-mini-4k-instruct': {'size': 3.8, 'tier': 'medium', 'family': 'Phi'},
        'microsoft/Phi-3.5-mini-instruct': {'size': 3.8, 'tier': 'medium', 'family': 'Phi'},
        'nvidia/Llama-3.1-Minitron-4B-Width-Base': {'size': 4.0, 'tier': 'medium', 'family': 'Llama'},
        'nvidia/Llama-3.1-Minitron-4B-Depth-Base': {'size': 4.0, 'tier': 'medium', 'family': 'Llama'},
        '01-ai/Yi-1.5-6B-Chat': {'size': 6.0, 'tier': 'medium', 'family': 'Yi'},
        'deepseek-ai/deepseek-llm-7b-base': {'size': 7.0, 'tier': 'strong', 'family': 'DeepSeek'},
        'Qwen/Qwen2.5-7B-Instruct': {'size': 7.0, 'tier': 'strong', 'family': 'Qwen'},
        'mistralai/Mistral-7B-Instruct-v0.3': {'size': 7.0, 'tier': 'strong', 'family': 'Mistral'},
        'meta-llama/Llama-3.1-8B-Instruct': {'size': 8.0, 'tier': 'strong', 'family': 'Llama'},
        'nvidia/Llama-3.1-Nemotron-Nano-8B-v1': {'size': 8.0, 'tier': 'strong', 'family': 'Llama'},
        'google/gemma-2-9b-it': {'size': 9.0, 'tier': 'strong', 'family': 'Gemma'},
        'meta-llama/Llama-2-13b-hf': {'size': 13.0, 'tier': 'strong', 'family': 'Llama'},
        'Qwen/Qwen2.5-14B-Instruct': {'size': 14.0, 'tier': 'strong', 'family': 'Qwen'},
        'openai/gpt-oss-20b': {'size': 20.0, 'tier': 'strong', 'family': 'GPT'},
        'mistralai/Mistral-Small-24B-Instruct-2501': {'size': 24.0, 'tier': 'strong', 'family': 'Mistral'},
        'moonshotai/kimi-k2-instruct': {'size': 32.0, 'tier': 'strong', 'family': 'Kimi'},
        'Qwen/Qwen2.5-32B-Instruct': {'size': 32.0, 'tier': 'strong', 'family': 'Qwen'},
        'deepseek-ai/deepseek-llm-67b-base': {'size': 67.0, 'tier': 'strong', 'family': 'DeepSeek'},
        'meta/llama-3.3-70b-versatile': {'size': 70.0, 'tier': 'strong', 'family': 'Llama'},
        'meta-llama/Meta-Llama-3-70B': {'size': 70.0, 'tier': 'strong', 'family': 'Llama'},
        'meta-llama/Meta-Llama-3-70B-Instruct': {'size': 70.0, 'tier': 'strong', 'family': 'Llama'},
        'databricks-meta-llama-3-1-405b-instruct': {'size': 405.0, 'tier': 'strong', 'family': 'Llama'},
        'gemini-2.0-flash': {'size': 50.0, 'tier': 'strong', 'family': 'Gemini'},
    }
    
    COST_PER_1M_TOKENS = {
        0.5: 0.05, 1.0: 0.10, 2.0: 0.15, 3.0: 0.20,
        3.8: 0.25, 4.0: 0.25, 6.0: 0.30, 7.0: 0.35,
        8.0: 0.40, 9.0: 0.45, 13.0: 0.60, 14.0: 0.65,
        24.0: 1.0, 32.0: 1.5, 50.0: 2.0, 67.0: 2.5,
        70.0: 3.0, 405.0: 15.0
    }
    
    ALPHA = 0.05
    MIN_EFFECT_SIZE = 0.2
    MIN_SAMPLES_PER_GROUP = 30

# ============================================================================
# GROUND TRUTH HORIZON DEFINITIONS
# ============================================================================
class GroundTruthHorizonClassifier:
    """Theory-driven temporal horizon classification"""
    
    @staticmethod
    def classify_gsm8k(question: str, reasoning_trace: List[str]) -> Tuple[int, str]:
        """GSM8K: Count arithmetic operations"""
        numbers = re.findall(r'\d+(?:\.\d+)?', str(question))
        num_operations = len(numbers)
        
        if reasoning_trace:
            ops_in_reasoning = sum(
                1 for step in reasoning_trace
                if any(op in str(step) for op in ['+', '-', '*', '/', '='])
            )
            num_operations = max(num_operations, ops_in_reasoning)
        
        if num_operations <= 2:
            return num_operations, 'short'
        elif num_operations <= 5:
            return num_operations, 'medium'
        else:
            return num_operations, 'long'
    
    @staticmethod
    def classify_boolq(question: str, reasoning_trace: List[str]) -> Tuple[int, str]:
        """BoolQ: Question complexity"""
        clauses = str(question).count(',') + str(question).count(' and ') + str(question).count(' or ')
        conditionals = ['if', 'when', 'unless', 'provided', 'assuming']
        has_conditional = any(cond in str(question).lower() for cond in conditionals)
        complexity = clauses + (2 if has_conditional else 0)
        
        if complexity <= 1:
            return 1, 'short'
        elif complexity <= 3:
            return 2, 'medium'
        else:
            return 3, 'long'
    
    @staticmethod
    def classify_arc(question: str, reasoning_trace: List[str]) -> Tuple[int, str]:
        """ARC: Science reasoning depth"""
        multi_step_indicators = [
            'first', 'then', 'next', 'finally', 'because',
            'causes', 'leads to', 'results in', 'therefore'
        ]
        
        steps = sum(
            1 for indicator in multi_step_indicators
            if indicator in str(question).lower()
        )
        
        if reasoning_trace and len(reasoning_trace) > steps:
            steps = len(reasoning_trace)
        
        if steps <= 1:
            return 1, 'short'
        elif steps <= 3:
            return 2, 'medium'
        else:
            return steps, 'long'
    
    @staticmethod
    def classify_commonsense(question: str, reasoning_trace: List[str]) -> Tuple[int, str]:
        """CommonsenseQA: Inference chain"""
        concepts = re.findall(r'\b[A-Z][a-z]+\b', str(question))
        num_concepts = len(set(concepts))
        
        if num_concepts <= 2:
            return 1, 'short'
        elif num_concepts <= 4:
            return 2, 'medium'
        else:
            return 3, 'long'
    
    @staticmethod
    def classify_mmlu(question: str, reasoning_trace: List[str]) -> Tuple[int, str]:
        """MMLU: Use reasoning trace"""
        if reasoning_trace:
            steps = len(reasoning_trace)
        else:
            steps = min(len(str(question).split()) // 20, 5)
        
        if steps <= 2:
            return steps, 'short'
        elif steps <= 5:
            return steps, 'medium'
        else:
            return steps, 'long'
    
    @classmethod
    def classify(cls, dataset: str, question: str, 
                 reasoning_trace: List[str]) -> Tuple[int, str]:
        """Dispatch to dataset-specific classifier"""
        dataset_lower = str(dataset).lower()
        
        if 'gsm' in dataset_lower:
            return cls.classify_gsm8k(question, reasoning_trace)
        elif 'boolq' in dataset_lower:
            return cls.classify_boolq(question, reasoning_trace)
        elif 'arc' in dataset_lower:
            return cls.classify_arc(question, reasoning_trace)
        elif 'commonsense' in dataset_lower:
            return cls.classify_commonsense(question, reasoning_trace)
        elif 'mmlu' in dataset_lower:
            return cls.classify_mmlu(question, reasoning_trace)
        else:
            steps = len(reasoning_trace) if reasoning_trace else 1
            if steps <= 2:
                return steps, 'short'
            elif steps <= 5:
                return steps, 'medium'
            else:
                return steps, 'long'

# ============================================================================
# DATA LOADER - FOLLOWS YOUR EXACT PATTERN
# ============================================================================
class RigorousDataLoader:
    """Load data using your exact directory structure"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.horizon_classifier = GroundTruthHorizonClassifier()
        self.all_data = {}
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load using EXACT same pattern as original temporal.py"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        model_dirs = [d for d in self.base_dir.iterdir() 
                     if d.is_dir() and d.name.endswith('_model')]
        
        print(f"\nFound {len(model_dirs)} model directories")
        
        for dataset in Config.DATASETS:
            dataset_dfs = []
            
            for model_dir in model_dirs:
                model_name = model_dir.name.replace('_model', '')
                
                # Handle naming variations - EXACT same as original
                if 'Llama-' in model_name and 'meta-llama' not in model_name:
                    model_name = f'meta-llama/{model_name}'
                if 'Qwen' in model_name and 'Qwen/' not in model_name:
                    model_name = f'Qwen/{model_name}'
                
                csv_path = model_dir / "cross_model_archaeology" / dataset / "raw_responses"
                
                if not csv_path.exists():
                    continue
                
                csv_files = list(csv_path.glob("responses_*.csv"))
                if not csv_files:
                    continue
                
                latest_csv = sorted(csv_files)[-1]
                
                try:
                    df = pd.read_csv(latest_csv)
                    
                    # Ensure model_name exists
                    if 'model_name' not in df.columns or df['model_name'].isna().all():
                        df['model_name'] = model_name
                    
                    # Add dataset name
                    df['dataset_name'] = dataset
                    
                    # Convert reasoning_trace if it exists
                    if 'reasoning_trace' in df.columns:
                        df['reasoning_trace'] = df['reasoning_trace'].apply(
                            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                        )
                    
                    dataset_dfs.append(df)
                    
                except Exception as e:
                    print(f"⚠️ Error loading {latest_csv}: {e}")
                    continue
            
            if dataset_dfs:
                combined_df = pd.concat(dataset_dfs, ignore_index=True)
                self.all_data[dataset] = combined_df
                print(f"📂 {dataset}: {len(combined_df)} responses from "
                      f"{combined_df['model_name'].nunique()} models")
        
        return self.all_data
    
    def compute_temporal_metrics(self) -> pd.DataFrame:
        """Compute temporal metrics from loaded data"""
        print("\n" + "="*80)
        print("COMPUTING TEMPORAL METRICS")
        print("="*80)
        
        all_metrics = []
        
        for dataset_name, df in self.all_data.items():
            print(f"\n📊 Processing {dataset_name}...")
            
            for idx, row in df.iterrows():
                metric = self._compute_single_metric(row, dataset_name, idx)
                if metric:
                    all_metrics.append(metric)
        
        metrics_df = pd.DataFrame(all_metrics)
        
        print(f"\n✅ Loaded {len(metrics_df)} validated samples")
        print(f"   Models: {metrics_df['model_name'].nunique()}")
        print(f"   Datasets: {metrics_df['dataset'].nunique()}")
        
        return metrics_df
    
    def _compute_single_metric(self, row: pd.Series, dataset: str, idx: int) -> Optional[Dict]:
        """Compute metric for single row"""
        try:
            model_name = row.get('model_name', '')
            
            if model_name not in Config.MODEL_METADATA:
                return None
            
            metadata = Config.MODEL_METADATA[model_name]
            
            # Get reasoning trace (handle if column doesn't exist)
            reasoning_trace = []
            if 'reasoning_trace' in row.index:
                rt = row['reasoning_trace']
                if isinstance(rt, list):
                    reasoning_trace = rt
                elif isinstance(rt, str) and rt:
                    try:
                        reasoning_trace = ast.literal_eval(rt)
                    except:
                        reasoning_trace = []
            
            # Get question (try multiple column names)
            question = ''
            for col in ['question', 'prompt', 'input']:
                if col in row.index:
                    question = str(row[col])
                    break
            
            # Classify horizon
            required_steps, horizon = self.horizon_classifier.classify(
                dataset, question, reasoning_trace
            )
            
            # Get accuracy (try multiple column names)
            is_correct = False
            for col in ['is_correct', 'correct', 'accuracy']:
                if col in row.index:
                    is_correct = bool(row[col])
                    break
            
            # Get tokens
            output_tokens = 0
            for col in ['output_tokens', 'tokens', 'num_tokens']:
                if col in row.index:
                    output_tokens = int(row[col]) if not pd.isna(row[col]) else 0
                    break
            
            # Get generation time
            generation_time = 0
            for col in ['generation_time', 'time', 'latency']:
                if col in row.index:
                    generation_time = float(row[col]) if not pd.isna(row[col]) else 0
                    break
            
            # Compute cost
            cost_per_1m = Config.COST_PER_1M_TOKENS.get(metadata['size'], 1.0)
            cost_estimate = (output_tokens / 1_000_000) * cost_per_1m
            
            return {
                'model_name': model_name,
                'model_size': metadata['size'],
                'model_family': metadata['family'],
                'sample_id': row.get('sample_id', idx),  # Use idx as fallback
                'dataset': dataset,
                'question': question,
                'required_steps': required_steps,
                'temporal_horizon': horizon,
                'final_accuracy': is_correct,
                'reasoning_trace_length': len(reasoning_trace),
                'output_tokens': output_tokens,
                'generation_time': generation_time,
                'cost_estimate': cost_estimate
            }
        
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            return None

# ============================================================================
# HUMAN VALIDATION
# ============================================================================
class HumanValidationProtocol:
    """Human validation protocol"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.validation_dir = output_dir / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
    
    def sample_for_validation(self, metrics_df: pd.DataFrame, 
                             n_per_dataset: int = 200) -> pd.DataFrame:
        """Sample for validation - only export columns that exist"""
        samples = []
        
        for dataset in metrics_df['dataset'].unique():
            dataset_df = metrics_df[metrics_df['dataset'] == dataset]
            
            for horizon in ['short', 'medium', 'long']:
                horizon_df = dataset_df[dataset_df['temporal_horizon'] == horizon]
                
                n_sample = min(len(horizon_df), n_per_dataset // 3)
                if n_sample > 0:
                    sample = horizon_df.sample(n=n_sample, random_state=42)
                    samples.append(sample)
        
        validation_df = pd.concat(samples, ignore_index=True)
        
        # Export only columns that exist
        export_cols = ['sample_id', 'dataset', 'question', 'temporal_horizon']
        available_cols = [col for col in export_cols if col in validation_df.columns]
        
        annotation_file = self.validation_dir / "for_annotation.csv"
        validation_df[available_cols].to_csv(annotation_file, index=False)
        
        print(f"\n📋 Validation Protocol:")
        print(f"   Sampled {len(validation_df)} examples")
        print(f"   Export: {annotation_file}")
        print(f"\n   NEXT STEPS:")
        print(f"   1. Have 2+ annotators label {annotation_file}")
        print(f"   2. Save as 'annotated_by_[name].csv'")
        print(f"   3. Run compute_agreement() to check κ")
        
        return validation_df
    
    def compute_agreement(self, annotator1_file: Path, annotator2_file: Path) -> Dict:
        """Compute Cohen's κ"""
        df1 = pd.read_csv(annotator1_file)
        df2 = pd.read_csv(annotator2_file)
        
        merged = df1.merge(df2, on='sample_id', suffixes=('_1', '_2'))
        
        agree = (merged['temporal_horizon_1'] == merged['temporal_horizon_2']).mean()
        
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(
            merged['temporal_horizon_1'],
            merged['temporal_horizon_2']
        )
        
        confusion = pd.crosstab(
            merged['temporal_horizon_1'],
            merged['temporal_horizon_2']
        )
        
        print(f"\n📊 Inter-Annotator Agreement:")
        print(f"   Raw agreement: {agree:.1%}")
        print(f"   Cohen's κ: {kappa:.3f}")
        print(f"   {'✅' if kappa >= 0.70 else '❌'} {'Acceptable' if kappa >= 0.70 else 'Too low'}")
        print(f"\n{confusion}")
        
        return {
            'agreement': float(agree),
            'kappa': float(kappa),
            'confusion': confusion.to_dict()
        }

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
class StatisticalAnalyzer:
    """Comprehensive statistical analysis"""
    
    def __init__(self, metrics_df: pd.DataFrame):
        self.metrics_df = metrics_df
    
    def run_analysis(self) -> Dict:
        """Run all statistical tests"""
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
        
        results = {}
        
        # Test 1: Size effect by horizon
        print("\n📊 Test 1: Model Size Effect")
        
        small_models = self.metrics_df[self.metrics_df['model_size'] < 5]
        large_models = self.metrics_df[self.metrics_df['model_size'] >= 20]
        
        for horizon in ['short', 'medium', 'long']:
            small_acc = small_models[
                small_models['temporal_horizon'] == horizon
            ]['final_accuracy'].values
            
            large_acc = large_models[
                large_models['temporal_horizon'] == horizon
            ]['final_accuracy'].values
            
            if len(small_acc) > 0 and len(large_acc) > 0:
                _, p_value = mannwhitneyu(large_acc, small_acc, alternative='greater')
                
                cohen_d = (large_acc.mean() - small_acc.mean()) / np.sqrt(
                    (large_acc.std()**2 + small_acc.std()**2) / 2
                )
                
                results[f'size_effect_{horizon}'] = {
                    'small_mean': float(small_acc.mean()),
                    'large_mean': float(large_acc.mean()),
                    'difference': float(large_acc.mean() - small_acc.mean()),
                    'p_value': float(p_value),
                    'cohen_d': float(cohen_d),
                    'significant': p_value < 0.05
                }
                
                print(f"   {horizon:8s}: Small={small_acc.mean():.3f}, "
                      f"Large={large_acc.mean():.3f}, d={cohen_d:.2f}, p={p_value:.4f}")
        
        # Test 2: R² by horizon
        print("\n📊 Test 2: Size Explanatory Power (R²)")
        
        for horizon in ['short', 'medium', 'long']:
            horizon_df = self.metrics_df[
                self.metrics_df['temporal_horizon'] == horizon
            ]
            
            size_acc = horizon_df.groupby('model_size')['final_accuracy'].mean()
            
            if len(size_acc) >= 3:
                log_sizes = np.log(size_acc.index)
                log_accs = np.log(size_acc.values + 1e-6)
                
                r_squared = np.corrcoef(log_sizes, log_accs)[0, 1]**2
                
                results[f'r_squared_{horizon}'] = float(r_squared)
                
                print(f"   {horizon:8s}: R² = {r_squared:.3f}")
        
        # Test 3: Power analysis
        print("\n📊 Test 3: Statistical Power")
        
        short_acc = self.metrics_df[
            self.metrics_df['temporal_horizon'] == 'short'
        ]['final_accuracy']
        
        long_acc = self.metrics_df[
            self.metrics_df['temporal_horizon'] == 'long'
        ]['final_accuracy']
        
        if len(short_acc) > 0 and len(long_acc) > 0:
            pooled_std = np.sqrt(
                (short_acc.std()**2 + long_acc.std()**2) / 2
            )
            effect_size = abs(short_acc.mean() - long_acc.mean()) / pooled_std
            
            power_analyzer = TTestIndPower()
            power = power_analyzer.solve_power(
                effect_size=effect_size,
                nobs1=len(short_acc),
                ratio=len(long_acc)/len(short_acc),
                alpha=0.05
            )
            
            results['power_analysis'] = {
                'effect_size': float(effect_size),
                'power': float(power)
            }
            
            print(f"   Effect size: {effect_size:.3f}")
            print(f"   Power: {power:.1%}")
            print(f"   {'✅' if power >= 0.80 else '⚠️'} {'Adequate' if power >= 0.80 else 'Low'}")
        
        return results

# ============================================================================
# VISUALIZATIONS
# ============================================================================
class Visualizer:
    """Publication-quality visualizations"""
    
    def __init__(self, metrics_df: pd.DataFrame, output_dir: Path):
        self.metrics_df = metrics_df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_main_figures(self):
        """Create all main figures"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Accuracy vs Size by Horizon
        ax = axes[0, 0]
        for horizon in ['short', 'medium', 'long']:
            horizon_df = self.metrics_df[
                self.metrics_df['temporal_horizon'] == horizon
            ]
            
            size_stats = horizon_df.groupby('model_size')['final_accuracy'].agg([
                'mean', 'std', 'count'
            ])
            
            size_stats['ci'] = 1.96 * size_stats['std'] / np.sqrt(size_stats['count'])
            
            ax.errorbar(
                size_stats.index,
                size_stats['mean'],
                yerr=size_stats['ci'],
                marker='o',
                label=horizon.capitalize(),
                capsize=5
            )
        
        ax.set_xscale('log')
        ax.set_xlabel('Model Size (B)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Model Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Sample Distribution
        ax = axes[0, 1]
        horizon_counts = self.metrics_df['temporal_horizon'].value_counts()
        ax.bar(horizon_counts.index, horizon_counts.values)
        ax.set_xlabel('Horizon')
        ax.set_ylabel('Count')
        ax.set_title('Sample Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy by Dataset
        ax = axes[1, 0]
        dataset_acc = self.metrics_df.groupby(['dataset', 'temporal_horizon'])[
            'final_accuracy'
        ].mean().unstack()
        
        dataset_acc.plot(kind='bar', ax=ax)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Dataset and Horizon')
        ax.legend(title='Horizon')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 4: Model Family Comparison
        ax = axes[1, 1]
        family_acc = self.metrics_df.groupby(['model_family', 'temporal_horizon'])[
            'final_accuracy'
        ].mean().unstack()
        
        if len(family_acc) > 0:
            family_acc.plot(kind='bar', ax=ax)
            ax.set_xlabel('Model Family')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by Family and Horizon')
            ax.legend(title='Horizon')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = self.output_dir / "main_figures.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Saved figures: {output_path}")
        return output_path

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main analysis pipeline"""
    
    print("\n" + "="*80)
    print("PUBLICATION-GRADE TEMPORAL ANALYSIS")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    loader = RigorousDataLoader(Config.BASE_DIR)
    loader.load_all_data()
    
    if not loader.all_data:
        print("\n❌ No data found!")
        return
    
    # Step 2: Compute metrics
    metrics_df = loader.compute_temporal_metrics()
    
    if len(metrics_df) == 0:
        print("\n❌ No metrics computed!")
        return
    
    # Save metrics
    metrics_df.to_csv(Config.OUTPUT_DIR / 'validated_metrics.csv', index=False)
    print(f"\n✅ Saved metrics to: {Config.OUTPUT_DIR / 'validated_metrics.csv'}")
    from parasox_investigation import run_paradox_investigation
    paradox_results = run_paradox_investigation(metrics_df, Config.OUTPUT_DIR)
    
    # Step 3: Validation sampling
    validator = HumanValidationProtocol(Config.OUTPUT_DIR)
    validation_sample = validator.sample_for_validation(metrics_df)
    
    # Step 4: Statistical analysis
    analyzer = StatisticalAnalyzer(metrics_df)
    stats_results = analyzer.run_analysis()
    
    with open(Config.OUTPUT_DIR / 'statistical_results.json', 'w') as f:
        json.dump(stats_results, f, indent=2, default=str)
    
    # Step 5: Visualizations
    visualizer = Visualizer(metrics_df, Config.OUTPUT_DIR)
    visualizer.create_main_figures()
    
    # Step 6: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n📊 DATA:")
    print(f"   Samples: {len(metrics_df):,}")
    print(f"   Models: {metrics_df['model_name'].nunique()}")
    print(f"   Datasets: {metrics_df['dataset'].nunique()}")
    print(f"   Size range: {metrics_df['model_size'].min():.1f}B - {metrics_df['model_size'].max():.1f}B")
    
    print(f"\n ACCURACY BY HORIZON:")
    for horizon in ['short', 'medium', 'long']:
        acc = metrics_df[metrics_df['temporal_horizon'] == horizon]['final_accuracy'].mean()
        n = len(metrics_df[metrics_df['temporal_horizon'] == horizon])
        print(f"   {horizon:8s}: {acc:.1%} (n={n:,})")
    
    print(f"\n All results saved to: {Config.OUTPUT_DIR}")
    print(f"\n  NEXT STEPS:")
    print(f"   1. Complete human validation (see {Config.OUTPUT_DIR}/validation/)")
    print(f"   2. Test more large models (currently have {metrics_df[metrics_df['model_size'] > 100]['model_name'].nunique()})")
    print(f"   3. Run compute_agreement() after annotation")

if __name__ == "__main__":
    main()