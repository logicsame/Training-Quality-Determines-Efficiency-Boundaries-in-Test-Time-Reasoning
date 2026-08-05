import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.phase2b.model_manager_random_sample import CrossModelProber
from src.phase2b.load_dataset_random_sample import load_gsm8k, load_arc_easy, load_mmlu
from tqdm import tqdm
from dataclasses import asdict


def main_causal_random_sample():
    """
    RANDOM SAMPLE CAUSAL INTERVENTION
    Fixes the selection bias problem of the original experiment.
    Tests control vs brief vs length_only on randomly sampled problems.
    length_only = same prompt as control, token cap at generation level only.
    This lets us separate prompt framing effect from token budget effect.
    """
    import random
    random.seed(42)

    LARGE_MODELS = [
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "Qwen/Qwen3-14B",
        'Qwen/Qwen3-32B'
    ]

    CONDITIONS = ['control', 'brief', 'length_only']

    DATASETS = [
        ('gsm8k',    load_gsm8k,    'math'),
        ('arc-easy', load_arc_easy, 'commonsense'),
        ('mmlu-stem', load_mmlu, 'commonsense')
    ]

    N_SAMPLES = 100
    DEBUG = False

    all_results = {}

    for dataset_name, loader_func, task_type in DATASETS:

        print(f"\n{'='*80}")
        print(f"RANDOM SAMPLE EXPERIMENT: {dataset_name.upper()}")
        print(f"n={N_SAMPLES} random problems | seed=42 | NOT pre-selected")
        print(f"{'='*80}")

        # Load large pool then take random sample — no selection bias
        all_samples, _ = loader_func(num_samples=1000)
        random_samples = random.sample(all_samples, min(N_SAMPLES, len(all_samples)))
        for i, s in enumerate(random_samples):
            s['id'] = i

        print(f" Random sample: {len(random_samples)} problems (seed=42)")

        output_dir = Path(f"./causal_random_results_prompt_confound/{dataset_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "raw_responses").mkdir(exist_ok=True)

        dataset_results = {}

        for condition in CONDITIONS:
            print(f"\n{'─'*60}")
            print(f" CONDITION: {condition.upper()}")

            prober = CrossModelProber(
                model_names=LARGE_MODELS,
                output_dir=str(output_dir),
                debug=DEBUG
            )

            all_responses = []
            for sample in tqdm(random_samples, desc=f"{dataset_name} - {condition}"):
                try:
                    responses = prober.probe_sample(sample, task_type, condition)
                    all_responses.extend(responses)
                except Exception as e:
                    print(f" Sample {sample['id']}: {e}")
                    continue

            df = pd.DataFrame([asdict(r) for r in all_responses])
            df['condition'] = condition
            df['sample_type'] = 'random'

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = output_dir / "raw_responses" / f"{condition}_{ts}.csv"
            df.to_csv(out_file, index=False)

            acc = df['is_correct'].mean()
            print(f"   Accuracy: {acc:.1%}  (n={len(df)})")
            dataset_results[condition] = df
            prober.cleanup_all_models()

        # ── COMPARISON TABLE ──
        print(f"\n{'='*60}")
        print(f" FINAL RESULT: {dataset_name.upper()}")
        print(f"{'='*60}")

        ctrl  = dataset_results['control']['is_correct'].mean()
        brief = dataset_results['brief']['is_correct'].mean()
        lonly = dataset_results['length_only']['is_correct'].mean()

        print(f"  Control:     {ctrl:.1%}")
        print(f"  Brief:       {brief:.1%}   Δ = {brief - ctrl:+.1%}")
        print(f"  Length-only: {lonly:.1%}   Δ = {lonly - ctrl:+.1%}")

        brief_delta  = brief - ctrl
        lonly_delta  = lonly - ctrl
        framing_effect = brief_delta - lonly_delta

        print(f"\n  Token budget effect (length_only Δ): {lonly_delta:+.1%}")
        print(f"  Prompt framing effect (brief - length_only): {framing_effect:+.1%}")

        if abs(framing_effect) < 0.05:
            print(f"   Framing effect < 5pp → effect is TOKEN BUDGET driven")
            print(f"     Causal claim holds.")
        else:
            print(f"    Framing effect = {framing_effect:+.1%} → PROMPT FRAMING contributes")
            print(f"     Reframe claim: brevity instruction drives accuracy, not token budget alone.")

        all_results[dataset_name] = dataset_results

    print(f"\n Random sample experiment complete.")
    print(f" Results saved to: ./causal_random_results/")
    return all_results


if __name__ == "__main__":
    from huggingface_hub import login
    login(token="haggingface_token")
    main_causal_random_sample()
