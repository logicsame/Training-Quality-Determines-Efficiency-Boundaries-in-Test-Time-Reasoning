# Training Quality Determines Efficiency Boundaries in Test-Time Reasoning

> **Anonymous submission to NeurIPS 2026**  
> Do not distribute or share publicly during review.

## Overview

This repository provides complete code and data for reproducing all experiments,
figures, and tables in the paper. The study examines how training methodology
determines token efficiency in language model reasoning across 44 models and
67,000+ evaluations.

## Key Results

| Finding | Section | Reproduction |
|---------|---------|-------------|
| 1,000-token convergence boundary | 3.1 | `experiments/phase2_token_ablation/` |
| +15.4pp causal intervention gain | 3.1 | `experiments/phase2b_causal/` |
| 18.8× efficiency from training quality | 3.2 | `analysis/efficiency_comparison.py` |
| Q-score framework (r = +0.787) | 3.2 | `analysis/qscore_framework.py` |
| GPQA quality collapse (r = −0.03) | 3.4 | `analysis/gpqa_analysis.py` |
| Mechanistic dimensions | 3.3 | `analysis/mechanistic_analysis.py` |

## Repository Structure

```
├── configs/                  # Model configs, prompt templates, hyperparameters
├── data/                     # Raw results, processed data, annotations
├── experiments/              # Experiment runners (Phase 1–3 + causal)
├── analysis/                 # Analysis scripts (figures, tables, statistics)
├── src/                      # Core library (evaluation, extraction, metrics)
├── requirements.txt          # Python dependencies
├── setup.py                  # Package 
```


## Hardware Requirements

- **Analysis only** (from cached data): Any machine, <5 min
- **Full reproduction**: NVIDIA H200 (191GB) or equivalent
  - Phase 1 (31 models): ~48 GPU-hours
  - Phase 2 (token ablation): ~12 GPU-hours
  - Phase 3 (13 reasoning models): ~24 GPU-hours
  - Causal interventions: ~8 GPU-hours

## Dependencies

- Python 3.10+
- PyTorch 2.1+
- HuggingFace Transformers 4.36+
- scipy, numpy, pandas, matplotlib, seaborn



## License

This code is released under the MIT License for research purposes.

```
