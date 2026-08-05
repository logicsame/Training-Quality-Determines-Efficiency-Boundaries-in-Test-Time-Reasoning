# Training Quality Determines Efficiency Boundaries in Test-Time Reasoning

[![Paper](https://img.shields.io/badge/Paper-DOI-b31b1b?style=flat-square&logo=arxiv)](https://doi.org/[YOUR_DOI_HERE])
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-FFD21E?style=flat-square)](https://huggingface.co/datasets/[HF_USERNAME]/[HF_DATASET_NAME])
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/docs/transformers)
![Models](https://img.shields.io/badge/Models%20Evaluated-50-2ea44f?style=flat-square)
![Evaluations](https://img.shields.io/badge/Evaluations-67%2C000%2B-orange?style=flat-square)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## Overview

This repository contains the complete, end-to-end reproducible pipeline for the paper **"Training Quality Determines Efficiency Boundaries in Test-Time Reasoning"** — a systematic empirical study of when additional inference tokens help, hurt, or have no effect on language model reasoning.

The central finding challenges the prevailing test-time compute paradigm: **training quality yields a 5.0–18.8× efficiency advantage at matched parameter scales**, exceeding the gains from ten-fold parameter increases. We evaluate 50 language models (0.5B–685B parameters) across six reasoning benchmarks totalling over 67,000 individual assessments.

> **Note on data:** Mechanistic profiling data for reasoning models is hosted separately on Hugging Face (see [Data](#data) section). All other data required for full reproduction is included in this repository.

---

## Key Findings

<table>
<tr>
<td width="50%">

**🔴 The 1,000-Token Wall**  
Standard models universally plateau at ~1,000 generated tokens regardless of architecture or parameter count. 87.5% of tested model–task conditions show zero accuracy improvement beyond this boundary.

</td>
<td width="50%">

**🟠 Over-Generation, Not Capacity Exhaustion**  
Constraining generation to ≤50 words *improves* accuracy by **+15.4 percentage points** on GSM8K. 76% of failures under unconstrained generation involve models abandoning initially correct reasoning.

</td>
</tr>
<tr>
<td width="50%">

**🟡 Training Quality > Parameter Count**  
Skywork-o1-8B (excellent training, Q = 0.987) achieves 86% GSM8K accuracy using 293 tokens on average. Marco-o1-7B (poor training, Q = 0.243) peaks at 56% while consuming 3,465 tokens — an **18.8× efficiency gap** at the same parameter scale.

</td>
<td width="50%">

**🔵 Standard Benchmarks Are Broken**  
GSM8K performance does not predict GPQA performance (r = −0.027, P = 0.915, n = 18). 92% of reasoning models collapse on graduate-level questions despite ≥80% arithmetic accuracy.

</td>
</tr>
</table>

---

## Repository Structure

```
Training-Quality-Determines-Efficiency/
│
├── analysis/                              # All analysis scripts
│   ├── causal_analysis.py                 # Causal intervention effect analysis
│   ├── compliance_analysis.py             # Token constraint compliance checks
│   ├── cross_dataset_qscore.py            # Q-score cross-dataset generalization
│   ├── efficiency_comparison.py           # 18.8× efficiency gap computation
│   ├── efficiency_vs_Q_score_correlation.py  # Q-score vs. η correlation (r = +0.787)
│   ├── efficiency_vs_Q_score_sensitivity.py  # Weighting sensitivity analysis
│   ├── error_annotation_analysis.py       # Blinded annotation results (κ = 0.886)
│   ├── extraction_bias_check.py           # Answer extraction bias checks
│   ├── gpqa_analysis.py                   # Graduate-level quality collapse (r = −0.027)
│   ├── mechanistic_analysis.py            # Vocabulary diversity, self-regulation, step decomposition
│   ├── qscore_circularity_test.py         # Circularity check (diversity-only r = +0.826)
│   ├── qscore_framework.py                # Q-score construction and tier assignment
│   ├── question_characteristics.py        # Question-level confound analysis
│   ├── random_sample_analysis.py          # Random-sample causal validation
│   ├── reasoning_ablation_analysis_small_models.py
│   ├── step_accuracy_paradox.py           # ρ = −0.093 step-accuracy paradox
│   ├── temporal_analysis.py
│   ├── token_ablation_analysis_large_models.py
│   ├── token_ablation_standard_model.py   # 1,000-token plateau detection
│   └── training_efficiency.py             # Training quality gradient analysis
│
├── configs/                               # Experimental configuration
│   ├── hyperparameters.yaml               # Decoding settings (greedy, T=0.6 replication)
│   ├── models.yaml                        # All 50 model definitions and HuggingFace IDs
│   └── prompts.yaml                       # All prompt templates (control, brief, direct)
│
├── data/
│   ├── annotations/
│   │   ├── annotation_protocol.md         # Blinded annotation protocol
│   │   └── errors_for_annotation.csv      # 58 annotated failure cases
│   └── processed/
│       ├── FUNCTIONAL_ANALYSIS_AGGREGATE_REPORT.md
│       ├── functional_analysis_full_comparison.csv
│       ├── functional_analysis_summary.csv
│       ├── training_quality_gradient_comparison.csv
│       └── inverse_scaling_problem_ids.json  # 115 inverse-scaling problem IDs
│
├── experiments/
│   ├── phase1_standard_eval/
│   │   └── run_evaluation.py              # Phase 1: 31 standard models × 5 benchmarks
│   ├── phase2_token_ablation_standard_models/
│   │   └── run_experiment.py              # Phase 2: Token ablation (300–2,000 tokens)
│   ├── phase2b_causal/
│   │   ├── run_causal_intervention.py     # Causal: brief vs. control conditions
│   │   └── run_random_sample.py           # Causal: random-sample validation
│   └── phase3_reasoning_eval/
│       └── run_reasoning_eval.py          # Phase 3: 19 reasoning models × 6 token limits
│
├── src/                                   # Core library
│   ├── phase1/                            # Standard model evaluation pipeline
│   │   ├── cross_model_prober.py
│   │   ├── divergence_analysis.py
│   │   ├── load_dataset.py
│   │   ├── model_manager.py
│   │   ├── prompt_formater.py
│   │   ├── reasoning_extractor.py
│   │   └── validator.py
│   ├── phase2/                            # Token ablation pipeline
│   │   ├── cross_model.py
│   │   ├── divergence_analysis.py
│   │   ├── load_dataset.py
│   │   ├── model_manager.py
│   │   ├── prompt_formater.py
│   │   ├── reasoning_extractor.py
│   │   └── validator.py
│   ├── phase2b/                           # Causal intervention pipeline
│   │   ├── cross_model_prober_random_sample.py
│   │   ├── cross_model_prober.py
│   │   ├── divergence_analysis.py
│   │   ├── load_dataset_random_sample.py
│   │   ├── load_dataset.py
│   │   ├── model_manager_random_sample.py
│   │   ├── model_manager.py
│   │   ├── prompt_formater_random_sample.py
│   │   ├── prompt_formater.py
│   │   ├── reasoning_extractor.py
│   │   └── validator.py
│   └── phase3/                            # Reasoning model pipeline
│       ├── cross_model_prober.py
│       ├── cross_model.py
│       ├── divergence_analysis.py
│       ├── load_dataset.py
│       ├── model_manager.py
│       ├── prompt_formater.py
│       ├── reasoning_extractor.py
│       └── validator.py
│
├── Q_score_framework_end_to_end_pipeline.py  # Complete Q-score pipeline
├── requirements.txt
├── setup.py
└── token.py                               # HuggingFace authentication
```

---

## Data

Experiment outputs and processed results are organized as follows:

| Data | Location | Description |
|------|----------|-------------|
| Phase 1 raw results | `data/processed/` | 31 models × 5 benchmarks × 300 samples (46,500 evaluations) |
| Phase 2 ablation results | `data/processed/` | 8 models × 3 datasets × 4 token limits (4,800 evaluations) |
| Phase 2b causal results | `data/processed/` | Brief vs. control, 1,441 paired responses |
| Annotation data | `data/annotations/` | 58 manually annotated failures (κ = 0.886) |
| Inverse scaling IDs | `data/inverse_scaling_problem_ids.json` | 115 problem IDs used for causal selection |
| **Phase 3 — full token ablation** | **🤗 Hugging Face** | 13 models × 4 datasets × 6 token limits (288 conditions, Table S4) |
| **Phase 3 — mechanistic dimensions** | **🤗 Hugging Face** | Diversity, repetition, utilization, tokens/step for all models × datasets (Table S3) |
| **Q-score tables** | **🤗 Hugging Face** | Q-scores, tier assignments, cross-dataset generalization (Tables S5, S7, S8) |
| **Phase 1 raw results (full)** | **🤗 Hugging Face** | Complete per-response outputs for all 31 standard models |

> **All Hugging Face data** is in a single dataset repository. Download before running Phase 3 analyses:
> ```bash
> huggingface-cli download [username]/training-quality-reasoning-efficiency \
>     --repo-type dataset \
>     --local-dir data/
> ```
> The dataset is organized into subdirectories mirroring this repository's `data/` structure, so files land in the right place automatically.

---

## Quickstart

### Analysis from Cached Data (No GPU Required)

All paper results can be reproduced from cached data in minutes:

```bash
git clone https://github.com/[username]/Training-Quality-Determines-Efficiency
cd Training-Quality-Determines-Efficiency

pip install -e .

# Reproduce the 18.8× efficiency comparison (Table 1 / Section 3.2)
python analysis/efficiency_comparison.py

# Reproduce the step-accuracy paradox (Figure 1 / Section A.2)
python analysis/step_accuracy_paradox.py

# Reproduce the Q-score framework and tier assignments (Figure 5 / Section 3.3)
python analysis/qscore_framework.py

# Reproduce the GSM8K–GPQA correlation (r = −0.027, Section 3.4)
python analysis/gpqa_analysis.py

# Reproduce the causal intervention analysis (+15.4pp, Section 3.1)
python analysis/causal_analysis.py

# Run the complete Q-score end-to-end pipeline
python Q_score_framework_end_to_end_pipeline.py
```

### Full Reproduction (GPU Required)

See [Reproducing All Results](#reproducing-all-results) below.

---

## Installation

```bash
git clone https://github.com/[username]/Training-Quality-Determines-Efficiency
cd Training-Quality-Determines-Efficiency

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -e .

# Set your HuggingFace token for gated model access
echo "your_hf_token_here" > token.py
# or: export HF_TOKEN=your_hf_token_here
```

**Dependencies** (`requirements.txt`):
```
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
scipy>=1.11.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
huggingface-hub>=0.20.0
```

---

## Hardware Requirements

| Task | Hardware | Estimated Time |
|------|----------|----------------|
| Analysis only (from cached data) | Any CPU | < 5 minutes |
| Phase 1 — 31 standard models | 1× H200 (191GB) or 2× A100 (80GB) | ~48 GPU-hours |
| Phase 2 — Token ablation | 1× H200 or 2× A100 | ~12 GPU-hours |
| Phase 2b — Causal interventions | 1× H200 or 2× A100 | ~8 GPU-hours |
| Phase 3 — 19 reasoning models | 1× H200 or 4× A100 | ~24 GPU-hours |
| **Full reproduction** | 1× H200 or equivalent | **~92 GPU-hours** |

> All experiments use FP16 precision and greedy decoding (`do_sample=False`) unless otherwise specified. Temperature replication experiments use `T=0.6, top_p=0.95`.

---

## Reproducing All Results

### Step 0 — Configuration

Edit `configs/models.yaml` to select which models to evaluate. Edit `configs/hyperparameters.yaml` to adjust token limits, sample sizes, or decoding settings. All prompt templates are in `configs/prompts.yaml`.

### Step 1 — Phase 1: Standard Model Evaluation

Evaluates 31 standard models across 5 benchmarks (GSM8K, BoolQ, ARC-Easy, CommonsenseQA, MMLU-STEM), 300 samples per benchmark.

```bash
python experiments/phase1_standard_eval/run_evaluation.py \
    --config configs/models.yaml \
    --prompts configs/prompts.yaml \
    --output_dir data/raw/phase1 \
    --n_samples 300
```

This produces: per-model accuracy files and reasoning chain extractions used for the step-accuracy paradox analysis.

### Step 2 — Phase 2: Token Ablation (Standard Models)

Evaluates 8 representative models at 4 token limits (300, 500, 1,000, 2,000 tokens), 50 samples per condition.

```bash
python experiments/phase2_token_ablation_standard_models/run_experiment.py \
    --config configs/models.yaml \
    --token_limits 300 500 1000 2000 \
    --datasets gsm8k boolq arc_easy \
    --n_samples 50 \
    --output_dir data/raw/phase2
```

### Step 3 — Phase 2b: Causal Intervention

Runs brief vs. control conditions on inverse-scaling problems and random-sample validation.

```bash
# Main causal intervention (inverse-scaling problems)
python experiments/phase2b_causal/run_causal_intervention.py \
    --problem_ids data/inverse_scaling_problem_ids.json \
    --conditions control brief \
    --output_dir data/raw/phase2b

# Random-sample validation (regression-to-mean exclusion)
python experiments/phase2b_causal/run_random_sample.py \
    --n_samples 100 \
    --conditions control brief length_only \
    --output_dir data/raw/phase2b_random
```

### Step 4 — Phase 3: Reasoning Model Evaluation

Evaluates 19 reasoning-specialized models at up to 6 token limits (300–8,000 tokens), 50 samples per condition, across 4 datasets including GPQA.

```bash
python experiments/phase3_reasoning_eval/run_reasoning_eval.py \
    --config configs/models.yaml \
    --token_limits 300 500 1000 2000 4000 8000 \
    --datasets gsm8k boolq arc_easy gpqa \
    --n_samples 50 \
    --output_dir data/raw/phase3
```

### Step 5 — Run All Analyses

Once all experimental data is collected (or using cached data):

```bash
# Step-accuracy paradox (Figure 1, Appendix A.2)
python analysis/step_accuracy_paradox.py

# Token ablation plateau detection (Figure 2A, Section 3.1)
python analysis/token_ablation_standard_model.py

# Causal intervention analysis (Figure 2B–C, Section 3.1)
python analysis/causal_analysis.py

# Training quality vs. parameter scaling (Figure 3, Section 3.2)
python analysis/training_efficiency.py
python analysis/efficiency_comparison.py

# Mechanistic dimensions (Figure 4, Section 3.3)
python analysis/mechanistic_analysis.py

# Q-score framework construction and validation (Figure 5, Section 3.3)
python analysis/qscore_framework.py
python analysis/efficiency_vs_Q_score_correlation.py
python analysis/efficiency_vs_Q_score_sensitivity.py

# Circularity / structural coupling check (Appendix S5)
python analysis/qscore_circularity_test.py

# GPQA quality collapse and GSM8K–GPQA correlation (Section 3.4)
python analysis/gpqa_analysis.py

# Cross-dataset Q-score generalization (Appendix Table S8)
python analysis/cross_dataset_qscore.py

# Question-level confound analysis (Appendix Table S3)
python analysis/question_characteristics.py

# Compliance analysis (Appendix, DeepSeek-R1-685B)
python analysis/compliance_analysis.py
```

---

## Models Evaluated

<details>
<summary><strong>Phase 1 — Standard Models (31 models, 0.5B–405B)</strong></summary>

| Family | Models |
|--------|--------|
| Llama | Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Llama-3-70B, Llama-3-70B-Instruct, Llama-3.3-70B, Llama-3.1-405B, Llama-2-13B |
| Qwen | Qwen2.5-0.5B, Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B |
| Mistral | Mistral-7B-v0.3, Mistral-Small-24B |
| Gemma | Gemma-3-1B, Gemma-2-2B, Gemma-2-9B |
| Phi | Phi-3-Mini, Phi-3.5-Mini |
| Yi | Yi-1.5-6B |
| DeepSeek | DeepSeek-LLM-7B, DeepSeek-LLM-67B |
| Others | StableLM-2-1.6B, StableLM-Zephyr-3B, Minitron-4B-Width, Minitron-4B-Depth, Nemotron-Nano-8B, GPT-OSS-20B, Kimi-K2-32B, Gemini-2.0-Flash |

</details>

<details>
<summary><strong>Phase 3 — Reasoning-Specialized Models (19 models, 1.5B–685B)</strong></summary>

| Model | Parameters | Training Methodology | Q-Score (GSM8K) |
|-------|-----------|---------------------|-----------------|
| Skywork-o1-8B | 8B | PRM + Q* RL | 0.987 (Tier 1) |
| QwenQwQ-32B | 32B | RL | 0.932 (Tier 1) |
| DeepSeek-R1-70B | 70B | Distillation | 0.873 (Tier 2) |
| Qwen3-32B-O1 | 32B | RL | 0.679 (Tier 3) |
| DeepSeek-R1-685B | 685B | RL + Distillation | 0.684 (Tier 3) |
| DeepSeek-R1-14B | 14B | Distillation | 0.684 (Tier 3) |
| Nemotron-7B | 7B | RL | 0.556 (Tier 4) |
| GLM-4.5 | 355B (32B active) | MoE RL | 0.555 (Tier 4) |
| GLM-4.7 | 355B (32B active) | MoE RL | 0.478 (Tier 4) |
| Kimi2.5-1T | 1T (32B active) | MoE | 0.457 (Tier 4) |
| Marco-o1-7B | 7B | MCTS | 0.243 (Tier 5) |
| DeepSeek-R1-7B | 7B | Distillation | 0.359 (Tier 5) |
| DeepSeek-R1-1.5B | 1.5B | Distillation | 0.082 (Tier 5) |
| + 6 additional models | 7B–229B | Various | — |

</details>

---

## Benchmarks

| Benchmark | Task Type | Samples Used | Phase |
|-----------|-----------|-------------|-------|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Grade-school mathematics | 300 (Ph1), 50 (Ph2/3) | 1, 2, 3 |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | Boolean QA | 300 (Ph1), 50 (Ph2/3) | 1, 2, 3 |
| [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) | Science reasoning | 300 (Ph1), 50 (Ph2/3) | 1, 2, 3 |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | Commonsense reasoning | 300 (Ph1) | 1 |
| [MMLU-STEM](https://huggingface.co/datasets/cais/mmlu) | 19 STEM subjects | 300 (Ph1) | 1 |
| [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | Graduate-level science | 50 (Ph3) | 3 |

---

## The Q-Score Framework

The Q-score is a composite training quality metric computed entirely from inference-time output properties — no access to model internals required.

```
Q = 0.7 × Quality Control + 0.3 × Decomposition Efficiency

where:
  Quality Control     = 0.5 × Diversity Maintenance + 0.5 × Self-Regulation
  Diversity Maintenance = 1 − |ΔD| / 100         (vocabulary type-token ratio change)
  Self-Regulation       = 1 − U                   (token budget utilization)
  Decomposition         = 1 − norm(tokens/step)   (step granularity efficiency)
```

| Tier | Threshold | Label | Example Models |
|------|-----------|-------|----------------|
| 1 | Q ≥ 0.88 | Excellent | Skywork-o1-8B (0.987), QwenQwQ-32B (0.932) |
| 2 | Q ≥ 0.79 | Good | DeepSeek-R1-70B (0.873) |
| 3 | Q ≥ 0.58 | Moderate | Qwen3-32B-O1 (0.679), DS-R1-685B (0.684) |
| 4 | Q ≥ 0.38 | Poor | Nemotron-7B (0.556), GLM-4.5 (0.555) |
| 5 | Q < 0.38 | Insufficient | Marco-o1-7B (0.243), DS-R1-1.5B (0.082) |

The Q-score explains **61.9% of variance** in token efficiency (Pearson r = +0.787, P = 0.001) across models spanning 1.5B–685B parameters. The relationship holds when token utilization is removed from the formula entirely (diversity-only: r = +0.826), ruling out structural circularity.

Run the full Q-score pipeline:
```bash
python Q_score_framework_end_to_end_pipeline.py
```

---

## Reproducing Specific Paper Claims

| Paper Claim | Script | Expected Output |
|-------------|--------|----------------|
| 87.5% of conditions plateau at 1,000 tokens | `analysis/token_ablation_standard_model.py` | 21/24 perfect plateaus (Δ=0.0, P=1.00) |
| +15.4pp from brevity constraints (GSM8K) | `analysis/causal_analysis.py` | t₅₁=2.68, P=0.010 |
| 18.8× efficiency gap (Skywork vs. Marco, 7–8B) | `analysis/efficiency_comparison.py` | η: 0.00294 vs. 0.000156 |
| Q-score vs. efficiency r = +0.787 | `analysis/efficiency_vs_Q_score_correlation.py` | Pearson r=+0.787, P=0.001 |
| GSM8K–GPQA r = −0.027 | `analysis/gpqa_analysis.py` | r=−0.027, P=0.915, n=18 |
| Step-accuracy ρ = −0.093 | `analysis/step_accuracy_paradox.py` | Spearman ρ=−0.093, P<10⁻²⁵ |
| DeepSeek distillation Q-score gradient | `analysis/training_efficiency.py` | 0.873→0.818→0.684→0.359→0.082 |
| Circularity check (diversity-only r = +0.826) | `analysis/qscore_circularity_test.py` | r=+0.826, P=0.0005 |

---

## Decoding Settings

All experiments use **greedy decoding** (`do_sample=False`) as the primary setting. A temperature replication (`T=0.6, top_p=0.95`) is provided in `configs/hyperparameters.yaml` for the token ablation and causal intervention experiments. Temperature sampling produces equal or stronger effects (e.g., +24.0pp for Qwen2.5-32B on GSM8K vs. +15.4pp under greedy), confirming greedy results are a conservative lower bound.

---

## Citation

> 📄 **[Paper](https://doi.org/[YOUR_DOI_HERE])**

```bibtex
@article{training-quality-efficiency-2025,
  title   = {Training Quality Determines Efficiency Boundaries in Test-Time Reasoning},
  year    = {2025},
  doi     = {[YOUR_DOI_HERE]},
  url     = {https://doi.org/[YOUR_DOI_HERE]}
}
```

---

## License

This repository is released under the **MIT License**. See `LICENSE` for details.

Model weights are subject to their respective licenses on Hugging Face. GPQA is subject to its [original license](https://github.com/idavidrein/gpqa).

---

<div align="center">
<sub>If you find this work useful, please consider starring the repository.</sub>
</div>