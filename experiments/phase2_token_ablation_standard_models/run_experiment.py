"""
Standard Model Token Limit Experiments
"""


import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tqdm import tqdm
from src.phase2.load_dataset import (
    load_gsm8k,
    load_strategyqa,
    load_boolq,
    load_arc_easy,
    load_commonsenseqa
)
from pathlib import Path  
import pandas as pd
from dataclasses import asdict
from src.phase1.cross_model_prober import CrossModelProber  
# ============================================================================
# CONFIGURATION
# ============================================================================


TEST_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
]

# Token limits to test
TOKEN_LIMITS = [300,500,1000,2000,4000]

# Test on these datasets
TEST_DATASETS = {
    'gsm8k': (load_gsm8k, 'math'),
    'boolq': (load_boolq, 'reasoning'),
    'arc-easy': (load_arc_easy, 'commonsense'),
}

SAMPLES_PER_DATASET = 50

OUTPUT_DIR = Path("./token_ablation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# RUN ABLATION
# ============================================================================

def run_token_ablation():
    """Run complete token limit ablation"""
    
    print(f"\n{'#'*80}")
    print(f"# TOKEN LIMIT ABLATION STUDY")
    print(f"# Models: {len(TEST_MODELS)}")
    print(f"# Token limits: {TOKEN_LIMITS}")
    print(f"# Datasets: {len(TEST_DATASETS)}")
    print(f"{'#'*80}\n")
    
    all_results = []
    
    for dataset_name, (loader_func, task_type) in TEST_DATASETS.items():
        
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        # Load samples
        samples, _ = loader_func(num_samples=SAMPLES_PER_DATASET)
        
        # For each token limit
        for token_limit in TOKEN_LIMITS:
            
            print(f"\n Testing token limit: {token_limit}")
            
            #  KEY: Add token_limit to each sample
            for sample in samples:
                sample['token_limit'] = token_limit
            
            # Create prober
            prober = CrossModelProber(
                model_names=TEST_MODELS,
                output_dir=str(OUTPUT_DIR / f"{dataset_name}_{token_limit}"),
                debug=False
            )
            
            # Probe all samples
            all_responses = []
            for sample in tqdm(samples, desc=f"  {dataset_name} @ {token_limit} tokens"):
                try:
                    sample_responses = prober.probe_sample(sample, task_type)
                    all_responses.extend(sample_responses)
                except Exception as e:
                    print(f" Error: {e}")
                    continue
            
            # Convert to DataFrame
            if not all_responses:
                print(f" No responses collected for {dataset_name} @ {token_limit}")
                continue
                
            df = pd.DataFrame([asdict(r) for r in all_responses])
                        
            # Save results
            output_file = OUTPUT_DIR / f"{dataset_name}_{token_limit}_results.csv"
            df.to_csv(output_file, index=False)
            
            # Calculate summary
            for model in df['model_name'].unique():
                model_df = df[df['model_name'] == model]
                accuracy = model_df['is_correct'].mean()
                avg_tokens = model_df['output_tokens'].mean()
                hit_limit = (model_df['output_tokens'] >= (token_limit - 10)).mean()
                
                result = {
                    'dataset': dataset_name,
                    'model': model,
                    'token_limit': token_limit,
                    'accuracy': accuracy,
                    'avg_output_tokens': avg_tokens,
                    'hit_limit_pct': hit_limit * 100,
                    'n_samples': len(model_df)
                }
                all_results.append(result)
            
            print(f" Saved: {output_file}")
            
            # Cleanup
            prober.cleanup_all_models()
    
    # Save aggregate results
    results_df = pd.DataFrame(all_results)
    aggregate_file = OUTPUT_DIR / "aggregate_results.csv"
    results_df.to_csv(aggregate_file, index=False)
    
    print(f"\n{'#'*80}")
    print(f"# ABLATION COMPLETE")
    print(f"{'#'*80}")
    print(f"\n Results saved to: {OUTPUT_DIR}")
    
    # Print summary
    print_summary(results_df)


def print_summary(df):
    """Print summary table"""
    
    print(f"\n{'='*80}")
    print(f" SUMMARY: Accuracy by Token Limit")
    print(f"{'='*80}\n")
    
    for dataset in df['dataset'].unique():
        print(f"\n{dataset.upper()}:")
        dataset_df = df[df['dataset'] == dataset]
        
        for model in dataset_df['model'].unique():
            model_short = model.split('/')[-1]
            print(f"\n  {model_short}:")
            
            model_df = dataset_df[dataset_df['model'] == model].sort_values('token_limit')
            
            for _, row in model_df.iterrows():
                print(f"    {row['token_limit']:4d} tokens: {row['accuracy']*100:5.1f}% "
                      f"(avg tokens: {row['avg_output_tokens']:.0f}, "
                      f"hit limit: {row['hit_limit_pct']:.0f}%)")


if __name__ == "__main__":
    from huggingface_hub import login
    login(token="haggingface_token_goes_here")  # ← Replace with your Hugging Face token
    
    run_token_ablation()