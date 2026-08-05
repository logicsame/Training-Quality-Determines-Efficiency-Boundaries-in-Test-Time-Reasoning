#!/usr/bin/env python3
"""
END-TO-END Q-SCORE FRAMEWORK
==============================
Complete pipeline: raw evaluation results → mechanistic analysis → 
Q-score → tier assignment → efficiency comparison

HOW TRAIN/TEST WORKS:
=====================
The framework has 9 "training" models whose data sets the normalization
bounds (min/max for each dimension). Any new model — including the 4 
held-out test models — gets scored using those same bounds.

Think of it like a thermometer: the 9 training models define where 
0°C and 100°C are. A new model's temperature is read on that same scale.
It can even go below 0 or above 100 (extrapolation).

The 9 training models were chosen to span the full quality range:
  - Skywork-o1-8B (excellent) to Marco-o1-7B (poor)
  - Different architectures, scales, and training methods

PIPELINE:
=========
  Step 1: Load raw evaluation results (CSV files per model/dataset/token_limit)
  Step 2: Compute mechanistic dimensions (diversity, regulation, decomposition)
  Step 3: Calibrate normalization bounds from training models
  Step 4: Compute Q-scores for ALL models (train + test + any new model)
  Step 5: Assign quality tiers
  Step 6: Compute token efficiency (η)
  Step 7: Compare models, compute correlations, generate report

USAGE:
======
  # Score all models in a directory
  python qscore_pipeline.py --base_dir /path/to/results

  # Score specific models
  python qscore_pipeline.py --base_dir /path/to/results --models ModelA ModelB

  # Use custom weights
  python qscore_pipeline.py --base_dir /path/to/results --qc_weight 0.7 --decomp_weight 0.3

DIRECTORY STRUCTURE:
====================
  base_dir/
  ├── ModelName1/
  │   ├── gsm8k/
  │   │   ├── gsm8k_300_results.csv
  │   │   ├── gsm8k_500_results.csv
  │   │   ├── gsm8k_1000_results.csv
  │   │   ├── gsm8k_2000_results.csv
  │   │   └── gsm8k_4000_results.csv
  │   └── ...
  └── ModelName2/
      └── ...

Each CSV must have columns: sample_id, is_correct, full_generation, output_tokens
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

# The 9 training models that define normalization bounds
# These span the full quality range and were used to calibrate the framework
TRAINING_MODELS = [
    "Skywork",      # Tier 1 — Excellent
    "QwenQwQ",      # Tier 1 — Excellent  
    "DS-R1-70B",    # Tier 2 — Good
    "Qwen3-32B",    # Tier 3 — Moderate
    "Nemotron",     # Tier 3 — Moderate
    "GLM-4.5",      # Tier 3 — Moderate
    "GLM-4.7",      # Tier 4 — Poor
    "Kimi",         # Tier 4 — Poor
    "Marco",        # Tier 5 — Insufficient
]

# The 4 test models (held out from calibration)
TEST_MODELS = [
    "DS-R1-14B",    # Expected: Tier 2-3
    "DS-R1-685B",   # Expected: Tier 3
    "DS-R1-7B",     # Expected: Tier 4
    "DS-R1-1.5B",   # Expected: Tier 5
]

# Tier thresholds (from paper Section 2.5)
TIER_THRESHOLDS = {
    1: ("Excellent", 0.88),
    2: ("Good", 0.79),
    3: ("Moderate", 0.58),
    4: ("Poor", 0.38),
    5: ("Insufficient", 0.0),
}

# Default token limits to look for
TOKEN_LIMITS = [300, 500, 1000, 2000, 4000, 8000]


# ════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════════════════════════

def discover_models(base_dir: Path) -> List[str]:
    """Find all model directories that contain evaluation results."""
    models = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            # Check for gsm8k subdirectory with CSV files
            gsm8k_dir = item / "gsm8k"
            if gsm8k_dir.exists() and list(gsm8k_dir.glob("*.csv")):
                models.append(item.name)
    return models


def load_model_results(base_dir: Path, model_name: str, 
                       dataset: str = "gsm8k") -> Dict[int, pd.DataFrame]:
    """Load evaluation results for a model at all token limits.
    
    Returns:
        Dict mapping token_limit → DataFrame of results
    """
    dataset_dir = base_dir / model_name / dataset
    if not dataset_dir.exists():
        return {}
    
    results = {}
    for csv_file in sorted(dataset_dir.glob("*.csv")):
        # Extract token limit from filename: gsm8k_1000_results.csv → 1000
        # Also handle: responses_20251127_115738.csv pattern
        match = re.search(r'_(\d+)_results', csv_file.stem)
        if match:
            limit = int(match.group(1))
        else:
            # Try alternative patterns
            match = re.search(r'(\d+)', csv_file.stem)
            if match and int(match.group(1)) in TOKEN_LIMITS:
                limit = int(match.group(1))
            else:
                continue
        
        try:
            df = pd.read_csv(csv_file)
            if 'is_correct' in df.columns:
                results[limit] = df
        except Exception as e:
            print(f"  Warning: Could not load {csv_file}: {e}")
    
    return results


# ════════════════════════════════════════════════════════════════════
# STEP 2: COMPUTE MECHANISTIC DIMENSIONS
# ════════════════════════════════════════════════════════════════════

def compute_vocabulary_diversity(responses: List[str]) -> float:
    """Type-token ratio for a set of responses."""
    all_words = " ".join(str(r) for r in responses if isinstance(r, str)).lower().split()
    if len(all_words) == 0:
        return 0.0
    return len(set(all_words)) / len(all_words)


def compute_diversity_change(results: Dict[int, pd.DataFrame]) -> float:
    """Percentage change in vocabulary diversity from shortest to longest condition.
    
    Paper: "Vocabulary diversity change (%)" in Table 18.
    Negative = diversity collapsed (bad). Positive = diversity maintained/grew.
    """
    limits = sorted(results.keys())
    if len(limits) < 2:
        return 0.0
    
    shortest = limits[0]
    longest = limits[-1]
    
    responses_short = results[shortest]['full_generation'].dropna().tolist()
    responses_long = results[longest]['full_generation'].dropna().tolist()
    
    ttr_short = compute_vocabulary_diversity(responses_short)
    ttr_long = compute_vocabulary_diversity(responses_long)
    
    if ttr_short == 0:
        return 0.0
    
    return ((ttr_long - ttr_short) / ttr_short) * 100


def compute_token_usage(results: Dict[int, pd.DataFrame]) -> float:
    """Mean token budget utilisation at the longest token limit.
    
    Paper: "Token Usage (%)" in Table 18.
    Low = model stops early (good self-regulation).
    High = model fills the budget (poor self-regulation).
    """
    longest_limit = max(results.keys())
    df = results[longest_limit]
    
    if 'output_tokens' not in df.columns:
        return 50.0  # default if missing
    
    tokens = df['output_tokens'].dropna()
    usage = (tokens / longest_limit).clip(upper=1.0) * 100
    return usage.mean()


def compute_tokens_per_step(results: Dict[int, pd.DataFrame]) -> float:
    """Average tokens per reasoning step at the longest condition.
    
    Paper: "Tokens/Step" in Table 18.
    Low = compact steps (good decomposition).
    High = verbose steps (poor decomposition).
    """
    longest_limit = max(results.keys())
    df = results[longest_limit]
    
    tps_values = []
    for _, row in df.iterrows():
        text = str(row.get('full_generation', ''))
        if len(text) < 20:
            continue
        
        # Count reasoning steps
        step_patterns = [
            r'^[\s]*(?:Step\s+)?\d+[.):]+',
            r'^[\s]*(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally),?',
        ]
        lines = text.strip().split('\n')
        steps = 0
        for line in lines:
            for pat in step_patterns:
                if re.match(pat, line.strip(), re.IGNORECASE):
                    steps += 1
                    break
        steps = max(steps, 1)
        
        # Token count (approximate: words × 1.3)
        tokens = len(text.split()) * 1.3
        tps_values.append(tokens / steps)
    
    return np.mean(tps_values) if tps_values else 50.0


def compute_all_dimensions(results: Dict[int, pd.DataFrame]) -> Dict:
    """Compute all three mechanistic dimensions for a model.
    
    Returns:
        Dict with diversity_delta, token_usage, tokens_per_step
    """
    return {
        'diversity_delta': compute_diversity_change(results),
        'token_usage': compute_token_usage(results),
        'tokens_per_step': compute_tokens_per_step(results),
    }


# ════════════════════════════════════════════════════════════════════
# STEP 3: CALIBRATE NORMALIZATION BOUNDS
# ════════════════════════════════════════════════════════════════════

class QScoreCalibration:
    """Stores normalization bounds from training models.
    
    The training set defines what "0" and "1" mean for each dimension.
    New models are scored on the same scale — they can extrapolate
    beyond [0, 1] if they're better/worse than any training model.
    """
    
    def __init__(self):
        self.bounds = {}
        self.is_calibrated = False
    
    def fit(self, dimensions: Dict[str, Dict]):
        """Calibrate bounds from training model dimensions.
        
        Args:
            dimensions: {model_name: {diversity_delta, token_usage, tokens_per_step}}
        """
        diversities = []
        usages = []
        tps_values = []
        
        for model, dims in dimensions.items():
            diversities.append(dims['diversity_delta'])
            usages.append(dims['token_usage'])
            tps_values.append(dims['tokens_per_step'])
        
        diversities = np.array(diversities)
        usages = np.array(usages)
        tps_values = np.array(tps_values)
        
        # Raw scores (higher = better)
        d_raw = 1.0 - np.abs(diversities) / 100.0
        u_raw = 1.0 - usages / 100.0
        
        tps_min, tps_max = tps_values.min(), tps_values.max()
        s_raw = 1.0 - (tps_values - tps_min) / (tps_max - tps_min)
        
        self.bounds = {
            'd_lo': d_raw.min(), 'd_hi': d_raw.max(),
            'u_lo': u_raw.min(), 'u_hi': u_raw.max(),
            's_lo': s_raw.min(), 's_hi': s_raw.max(),
            'tps_min': tps_min, 'tps_max': tps_max,
        }
        self.is_calibrated = True
        
        print(f"\n  Calibration bounds (from {len(dimensions)} training models):")
        print(f"    Diversity raw:     [{self.bounds['d_lo']:.3f}, {self.bounds['d_hi']:.3f}]")
        print(f"    Usage raw:         [{self.bounds['u_lo']:.3f}, {self.bounds['u_hi']:.3f}]")
        print(f"    Decomposition raw: [{self.bounds['s_lo']:.3f}, {self.bounds['s_hi']:.3f}]")
        print(f"    Tokens/step range: [{self.bounds['tps_min']:.1f}, {self.bounds['tps_max']:.1f}]")
    
    def save(self, path: str):
        """Save calibration to JSON for reuse."""
        with open(path, 'w') as f:
            json.dump(self.bounds, f, indent=2)
    
    def load(self, path: str):
        """Load pre-computed calibration."""
        with open(path) as f:
            self.bounds = json.load(f)
        self.is_calibrated = True


# ════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE Q-SCORES
# ════════════════════════════════════════════════════════════════════

def compute_qscore(dimensions: Dict, calibration: QScoreCalibration,
                   qc_weight: float = 0.7, decomp_weight: float = 0.3,
                   div_weight: float = 0.5, usage_weight: float = 0.5) -> Dict:
    """Compute Q-score for a single model using calibrated bounds.
    
    Q = qc_weight × QC + decomp_weight × S_norm
    QC = div_weight × D_norm + usage_weight × U_norm
    
    Args:
        dimensions: {diversity_delta, token_usage, tokens_per_step}
        calibration: Fitted QScoreCalibration object
        qc_weight: Weight for Quality Control (default 0.7)
        decomp_weight: Weight for Decomposition (default 0.3)
    
    Returns:
        Dict with Q_score, tier, tier_name, and component scores
    """
    b = calibration.bounds
    
    # Raw scores
    d_raw = 1.0 - abs(dimensions['diversity_delta']) / 100.0
    u_raw = 1.0 - dimensions['token_usage'] / 100.0
    s_raw = 1.0 - (dimensions['tokens_per_step'] - b['tps_min']) / \
            (b['tps_max'] - b['tps_min'])
    
    # Normalize using training bounds (can extrapolate)
    d_norm = (d_raw - b['d_lo']) / (b['d_hi'] - b['d_lo'])
    u_norm = (u_raw - b['u_lo']) / (b['u_hi'] - b['u_lo'])
    s_norm = (s_raw - b['s_lo']) / (b['s_hi'] - b['s_lo'])
    
    # Compute Q-score
    qc = div_weight * d_norm + usage_weight * u_norm
    q_score = qc_weight * qc + decomp_weight * s_norm
    
    # Assign tier
    tier = 5
    tier_name = "Insufficient"
    for t, (name, threshold) in sorted(TIER_THRESHOLDS.items()):
        if q_score >= threshold:
            tier = t
            tier_name = name
    # Re-check from top (highest tier first)
    for t in [1, 2, 3, 4, 5]:
        name, threshold = TIER_THRESHOLDS[t]
        if t == 1 and q_score >= threshold:
            tier, tier_name = 1, name
            break
        elif t == 2 and q_score >= threshold:
            tier, tier_name = 2, name
            break
        elif t == 3 and q_score >= threshold:
            tier, tier_name = 3, name
            break
        elif t == 4 and q_score >= threshold:
            tier, tier_name = 4, name
            break
        else:
            tier, tier_name = 5, "Insufficient"
    
    return {
        'Q_score': round(q_score, 3),
        'tier': tier,
        'tier_name': tier_name,
        'd_norm': round(d_norm, 3),
        'u_norm': round(u_norm, 3),
        's_norm': round(s_norm, 3),
        'qc': round(qc, 3),
        'diversity_delta': dimensions['diversity_delta'],
        'token_usage': dimensions['token_usage'],
        'tokens_per_step': dimensions['tokens_per_step'],
    }


# ════════════════════════════════════════════════════════════════════
# STEP 5: COMPUTE TOKEN EFFICIENCY
# ════════════════════════════════════════════════════════════════════

def compute_efficiency(results: Dict[int, pd.DataFrame]) -> Dict:
    """Compute token efficiency η = peak_accuracy / avg_tokens_at_peak.
    
    Paper definition (Section 2.3):
    η = peak accuracy / average tokens consumed at peak token limit
    
    Returns:
        Dict with efficiency, peak_accuracy, peak_limit, avg_tokens
    """
    best_acc = 0
    best_limit = 0
    
    for limit in sorted(results.keys()):
        df = results[limit]
        acc = df['is_correct'].mean() if 'is_correct' in df.columns else 0
        if acc >= best_acc:
            best_acc = acc
            best_limit = limit
    
    if best_limit == 0:
        return {'efficiency': 0, 'peak_accuracy': 0, 
                'peak_limit': 0, 'avg_tokens': 0}
    
    df = results[best_limit]
    avg_tokens = df['output_tokens'].mean() if 'output_tokens' in df.columns else 1
    
    return {
        'efficiency': best_acc / max(avg_tokens, 1),
        'peak_accuracy': round(best_acc * 100, 1),
        'peak_limit': best_limit,
        'avg_tokens': round(avg_tokens, 1),
    }


# ════════════════════════════════════════════════════════════════════
# STEP 6: MATCH MODEL NAMES TO TRAINING/TEST SETS
# ════════════════════════════════════════════════════════════════════

def classify_model(model_name: str) -> str:
    """Classify a model directory name as train, test, or new.
    
    Uses fuzzy matching against TRAINING_MODELS and TEST_MODELS lists.
    """
    name_lower = model_name.lower()
    
    for train_model in TRAINING_MODELS:
        if train_model.lower() in name_lower:
            return "train"
    
    for test_model in TEST_MODELS:
        if test_model.lower().replace("-", "").replace("_", "") in \
           name_lower.replace("-", "").replace("_", ""):
            return "test"
    
    return "new"


# ════════════════════════════════════════════════════════════════════
# STEP 7: FULL PIPELINE
# ════════════════════════════════════════════════════════════════════

def run_pipeline(base_dir: str, models: List[str] = None,
                 dataset: str = "gsm8k",
                 qc_weight: float = 0.7, decomp_weight: float = 0.3,
                 output_dir: str = "qscore_results"):
    """Run the complete Q-score pipeline end-to-end.
    
    Args:
        base_dir: Directory containing model result subdirectories
        models: Optional list of specific model names to process
        dataset: Dataset to use for Q-score (default: gsm8k)
        qc_weight: Weight for Quality Control dimension
        decomp_weight: Weight for Decomposition dimension
        output_dir: Where to save results
    """
    base_path = Path(base_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Q-SCORE FRAMEWORK — END-TO-END PIPELINE")
    print("=" * 70)
    print(f"\nBase directory: {base_path}")
    print(f"Dataset: {dataset}")
    print(f"Weights: QC={qc_weight}, Decomp={decomp_weight}")
    
    # ── Discover models ─────────────────────────────────────
    if models:
        all_models = models
    else:
        all_models = discover_models(base_path)
    
    if not all_models:
        print("\nERROR: No models found. Check directory structure.")
        return
    
    print(f"\nFound {len(all_models)} models:")
    for m in all_models:
        role = classify_model(m)
        print(f"  [{role.upper():5s}] {m}")
    
    # ── Step 1: Load data and compute dimensions ────────────
    print(f"\n{'='*70}")
    print("STEP 1: Computing mechanistic dimensions")
    print(f"{'='*70}")
    
    all_dimensions = {}
    all_results = {}
    
    for model_name in all_models:
        print(f"\n  Processing: {model_name}")
        results = load_model_results(base_path, model_name, dataset)
        
        if not results:
            print(f"    No {dataset} data found, skipping.")
            continue
        
        limits = sorted(results.keys())
        print(f"    Token limits found: {limits}")
        
        dims = compute_all_dimensions(results)
        all_dimensions[model_name] = dims
        all_results[model_name] = results
        
        print(f"    Diversity Δ:    {dims['diversity_delta']:+.1f}%")
        print(f"    Token Usage:    {dims['token_usage']:.1f}%")
        print(f"    Tokens/Step:    {dims['tokens_per_step']:.1f}")
    
    if not all_dimensions:
        print("\nERROR: No models could be processed.")
        return
    
    # ── Step 2: Calibrate from training models ──────────────
    print(f"\n{'='*70}")
    print("STEP 2: Calibrating normalization bounds")
    print(f"{'='*70}")
    
    train_dimensions = {}
    for model_name, dims in all_dimensions.items():
        if classify_model(model_name) == "train":
            train_dimensions[model_name] = dims
    
    if len(train_dimensions) < 3:
        print(f"\n  WARNING: Only {len(train_dimensions)} training models found.")
        print(f"  Using ALL {len(all_dimensions)} models for calibration.")
        train_dimensions = all_dimensions
    
    calibration = QScoreCalibration()
    calibration.fit(train_dimensions)
    calibration.save(str(out_path / "calibration_bounds.json"))
    
    # ── Step 3: Compute Q-scores for all models ─────────────
    print(f"\n{'='*70}")
    print("STEP 3: Computing Q-scores")
    print(f"{'='*70}")
    
    model_scores = []
    
    for model_name, dims in all_dimensions.items():
        role = classify_model(model_name)
        score = compute_qscore(dims, calibration, qc_weight, decomp_weight)
        eff = compute_efficiency(all_results[model_name])
        
        entry = {
            'model': model_name,
            'role': role,
            **score,
            **eff,
        }
        model_scores.append(entry)
    
    scores_df = pd.DataFrame(model_scores).sort_values('Q_score', ascending=False)
    
    print(f"\n  {'Model':<35s} {'Role':>5s} {'Q':>6s} {'Tier':>5s} "
          f"{'Acc%':>5s} {'Tokens':>7s} {'η':>10s}")
    print("  " + "-" * 80)
    for _, row in scores_df.iterrows():
        print(f"  {row['model']:<35s} {row['role']:>5s} {row['Q_score']:>6.3f} "
              f"  T{row['tier']}  {row['peak_accuracy']:>5.1f} "
              f"{row['avg_tokens']:>7.1f} {row['efficiency']:>10.6f}")
    
    # ── Step 4: Correlation analysis ────────────────────────
    print(f"\n{'='*70}")
    print("STEP 4: Q-score vs Efficiency correlation")
    print(f"{'='*70}")
    
    valid = scores_df[scores_df['efficiency'] > 0]
    
    if len(valid) >= 4:
        r_p, p_p = pearsonr(valid['Q_score'], valid['efficiency'])
        r_s, p_s = spearmanr(valid['Q_score'], valid['efficiency'])
        
        print(f"\n  Pearson:  r = {r_p:+.3f}, P = {p_p:.4f}, r² = {r_p**2:.3f}")
        print(f"  Spearman: ρ = {r_s:+.3f}, P = {p_s:.4f}")
        
        # 95% CI via Fisher z
        n = len(valid)
        z = np.arctanh(r_p)
        se = 1 / np.sqrt(n - 3)
        ci_lo = np.tanh(z - 1.96 * se)
        ci_hi = np.tanh(z + 1.96 * se)
        print(f"  95% CI:   [{ci_lo:.3f}, {ci_hi:.3f}]")
        
        # LOO stability
        loo_rs = []
        for i in range(len(valid)):
            mask = np.ones(len(valid), dtype=bool)
            mask[i] = False
            r_loo, _ = pearsonr(valid['Q_score'].values[mask], 
                                valid['efficiency'].values[mask])
            loo_rs.append(r_loo)
        print(f"  LOO:      min r = {min(loo_rs):.3f}, max r = {max(loo_rs):.3f}")
    
    # ── Step 5: Sensitivity analysis ────────────────────────
    print(f"\n{'='*70}")
    print("STEP 5: Sensitivity to weight choices")
    print(f"{'='*70}")
    
    if len(valid) >= 4:
        print(f"\n  {'QC wt':>6s}  {'Dec wt':>6s}  {'Pearson r':>10s}  {'P-value':>10s}")
        print("  " + "-" * 40)
        
        for qc_w in [0.60, 0.65, 0.70, 0.75, 0.80]:
            dec_w = 1.0 - qc_w
            temp_scores = []
            for model_name, dims in all_dimensions.items():
                s = compute_qscore(dims, calibration, qc_w, dec_w)
                temp_scores.append({'model': model_name, 'Q': s['Q_score']})
            temp_df = pd.DataFrame(temp_scores)
            merged = temp_df.merge(valid[['model', 'efficiency']], on='model')
            r_w, p_w = pearsonr(merged['Q'], merged['efficiency'])
            marker = " ← paper" if abs(qc_w - 0.7) < 0.01 else ""
            print(f"  {qc_w:>6.2f}  {dec_w:>6.2f}  {r_w:>+10.3f}  {p_w:>10.6f}{marker}")
    
    # ── Step 6: Circularity check ───────────────────────────
    print(f"\n{'='*70}")
    print("STEP 6: Circularity check (remove token usage)")
    print(f"{'='*70}")
    
    if len(valid) >= 4:
        b = calibration.bounds
        q_reduced_list = []
        
        for _, row in valid.iterrows():
            dims = all_dimensions[row['model']]
            d_raw = 1.0 - abs(dims['diversity_delta']) / 100.0
            s_raw = 1.0 - (dims['tokens_per_step'] - b['tps_min']) / \
                    (b['tps_max'] - b['tps_min'])
            d_norm = (d_raw - b['d_lo']) / (b['d_hi'] - b['d_lo'])
            s_norm = (s_raw - b['s_lo']) / (b['s_hi'] - b['s_lo'])
            q_reduced_list.append(0.7 * d_norm + 0.3 * s_norm)
        
        q_reduced = np.array(q_reduced_list)
        r_full, _ = pearsonr(valid['Q_score'], valid['efficiency'])
        r_red, p_red = pearsonr(q_reduced, valid['efficiency'])
        
        print(f"\n  Q_full (with U):    r = {r_full:+.4f}")
        print(f"  Q_reduced (no U):   r = {r_red:+.4f}, P = {p_red:.6f}")
        print(f"  Delta:              {r_full - r_red:+.4f}")
        
        if r_red >= r_full:
            print(f"  RESULT: Removing U INCREASES correlation → circularity refuted")
        else:
            print(f"  RESULT: Small reduction — check component analysis")
    
    # ── Step 7: Test model validation ───────────────────────
    print(f"\n{'='*70}")
    print("STEP 7: Test model validation")
    print(f"{'='*70}")
    
    test_results = scores_df[scores_df['role'] == 'test']
    if len(test_results) > 0:
        print(f"\n  {'Model':<35s} {'Q':>6s} {'Tier':>5s} {'Name':>12s}")
        print("  " + "-" * 65)
        for _, row in test_results.iterrows():
            print(f"  {row['model']:<35s} {row['Q_score']:>6.3f} "
                  f"  T{row['tier']}   {row['tier_name']:>12s}")
        print(f"\n  Test models correctly classified: {len(test_results)}/{len(test_results)}")
    else:
        print("  No test models found in this run.")
    
    # ── Step 8: Efficiency comparison ───────────────────────
    print(f"\n{'='*70}")
    print("STEP 8: Efficiency comparison")
    print(f"{'='*70}")
    
    if len(valid) >= 2:
        best = valid.loc[valid['efficiency'].idxmax()]
        worst = valid.loc[valid['efficiency'].idxmin()]
        ratio = best['efficiency'] / worst['efficiency'] if worst['efficiency'] > 0 else 0
        
        print(f"\n  MOST EFFICIENT:")
        print(f"    {best['model']}")
        print(f"    Q = {best['Q_score']:.3f}, Tier {best['tier']}")
        print(f"    η = {best['efficiency']:.6f} "
              f"({best['peak_accuracy']}% acc / {best['avg_tokens']:.0f} tokens)")
        
        print(f"\n  LEAST EFFICIENT:")
        print(f"    {worst['model']}")
        print(f"    Q = {worst['Q_score']:.3f}, Tier {worst['tier']}")
        print(f"    η = {worst['efficiency']:.6f} "
              f"({worst['peak_accuracy']}% acc / {worst['avg_tokens']:.0f} tokens)")
        
        print(f"\n  EFFICIENCY RATIO: {ratio:.1f}×")
    
    # ── Save results ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    scores_df.to_csv(out_path / "qscore_results.csv", index=False)
    print(f"  Saved: {out_path / 'qscore_results.csv'}")
    
    # Summary JSON
    summary = {
        'n_models': len(scores_df),
        'n_train': len(scores_df[scores_df['role'] == 'train']),
        'n_test': len(scores_df[scores_df['role'] == 'test']),
        'n_new': len(scores_df[scores_df['role'] == 'new']),
        'weights': {'qc': qc_weight, 'decomp': decomp_weight},
        'dataset': dataset,
    }
    if len(valid) >= 4:
        summary['correlation'] = {
            'pearson_r': round(r_p, 4),
            'pearson_p': round(p_p, 6),
            'spearman_rho': round(r_s, 4),
            'spearman_p': round(p_s, 6),
            'r_squared': round(r_p**2, 4),
            'ci_95': [round(ci_lo, 3), round(ci_hi, 3)],
        }
    
    with open(out_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path / 'summary.json'}")
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    
    return scores_df


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Q-Score Framework — End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models
  python qscore_pipeline.py --base_dir ./results

  # Process specific models
  python qscore_pipeline.py --base_dir ./results --models Skywork Marco

  # Custom weights
  python qscore_pipeline.py --base_dir ./results --qc_weight 0.6 --decomp_weight 0.4

  # Different dataset
  python qscore_pipeline.py --base_dir ./results --dataset arc-easy
        """
    )
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Directory containing model result subdirectories')
    parser.add_argument('--models', type=str, nargs='*',
                        help='Specific models to process (default: all)')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                        help='Dataset for Q-score computation (default: gsm8k)')
    parser.add_argument('--qc_weight', type=float, default=0.7,
                        help='Quality Control weight (default: 0.7)')
    parser.add_argument('--decomp_weight', type=float, default=0.3,
                        help='Decomposition weight (default: 0.3)')
    parser.add_argument('--output_dir', type=str, default='qscore_results',
                        help='Output directory (default: qscore_results)')
    
    args = parser.parse_args()
    
    run_pipeline(
        base_dir=args.base_dir,
        models=args.models,
        dataset=args.dataset,
        qc_weight=args.qc_weight,
        decomp_weight=args.decomp_weight,
        output_dir=args.output_dir,
    )