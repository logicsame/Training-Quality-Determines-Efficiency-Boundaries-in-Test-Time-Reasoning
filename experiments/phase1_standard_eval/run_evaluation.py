from pathlib import Path
from typing import List, Dict, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.phase1.load_dataset import (
    load_gsm8k,
    load_commonsenseqa,
    load_boolq,
    load_arc_easy,
    load_mmlu
)
from src.phase1.divergence_analysis import DivergenceAnalyzer
from src.phase1.cross_model_prober import CrossModelProber
# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main cross-model cognitive archaeology pipeline - MULTI-DATASET VERSION"""

    # ========== CONFIGURATION ==========
    MODELS_TO_TEST = [
        # ── Qwen Family ──────────────────────────────────
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        # ── Gemma Family ─────────────────────────────────
        "google/gemma-3-1b-it",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        # ── Llama Family ─────────────────────────────────
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "nvidia/Llama-3.1-Minitron-4B-Width-Base",
        "nvidia/Llama-3.1-Minitron-4B-Depth-Base",
        "meta-llama/Llama-3.1-8B-Instruct",
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama-3.3-70b-versatile",       
        "meta-llama/Llama-3.1-405B", 
        # ── StableLM Family ──────────────────────────────
        "stabilityai/stablelm-2-1_6b",
        "stabilityai/stablelm-zephyr-3b",
        # ── Phi Family ───────────────────────────────────
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        # ── Mistral Family ───────────────────────────────
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        # ── Yi Family ────────────────────────────────────
        "01-ai/Yi-1.5-6B-Chat",
        # ── DeepSeek Family ──────────────────────────────
        "deepseek-ai/deepseek-llm-7b-base",
        "deepseek-ai/deepseek-llm-67b-base",
        # ── Other ────────────────────────────────────────
        "OpenGPT/gpt-oss-20b",
        "moonshotai/Kimi-K2-Instruct",
        "gemini-2.0-flash",              # via google API
           
    ]

    NUM_SAMPLES_PER_DATASET = 300 
    DEBUG = False

    # ========== DATASETS TO TEST ==========
    DATASETS = [
        ("GSM8K", load_gsm8k),
        ("CommonsenseQA", load_commonsenseqa),
        ("BoolQ", load_boolq),
        ("ARC-Easy", load_arc_easy),
        ("MMLU-STEM", load_mmlu),
    ]

    # ========== RUN ON ALL DATASETS ==========
    all_results = {}

    for dataset_name, loader_func in DATASETS:
        print(f"\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset_name}")
        print(f"{'#'*80}\n")

        # Create dataset-specific output directory
        output_dir = f"./cross_model_archaeology/{dataset_name.lower()}"

        prober = CrossModelProber(
            model_names=MODELS_TO_TEST,
            output_dir=output_dir,
            debug=DEBUG
        )

        try:
            # ========== LOAD DATASET ==========
            samples, task_type = loader_func(num_samples=NUM_SAMPLES_PER_DATASET)

            # ========== PROBE ALL MODELS ==========
            responses_df = prober.probe_dataset(samples, task_type)

            # Validate extraction quality
            prober.validate_extraction_quality(responses_df, dataset_name)

            # ========== ANALYZE DIVERGENCE ==========
            analyzer = DivergenceAnalyzer(output_dir=Path(output_dir), debug=DEBUG)
            divergence_points = analyzer.analyze_divergence(responses_df)

            # Store results
            all_results[dataset_name] = {
                'responses': responses_df,
                'divergences': divergence_points,
                'task_type': task_type
            }

            print(f"\n Completed {dataset_name}")
            print(f"   Responses: {len(responses_df)}")
            print(f"   Divergences: {len(divergence_points)}")

        except Exception as e:

            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up models after each dataset
            prober.cleanup_all_models()

    # ========== CROSS-DATASET COMPARISON ==========
    print(f"\n{'#'*80}")
    print(f"# CROSS-DATASET SUMMARY")
    print(f"{'#'*80}\n")

    for dataset_name, results in all_results.items():
        df = results['responses']
        print(f"\n{dataset_name}:")
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            accuracy = model_df['is_correct'].mean()
            print(f"  {model}: {accuracy:.1%}")

    print(f"\n{'#'*80}")
    print(f"# ALL DATASETS COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    from huggingface_hub import login
    login(token="haggingface_token_goes_here")  # ← Replace with your Hugging Face token

    main() 