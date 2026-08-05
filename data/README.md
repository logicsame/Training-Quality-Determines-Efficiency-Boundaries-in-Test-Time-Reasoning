# Data Directory

## Structure

```
data/
├── raw/                          # Raw model responses (large, .gitignored)
│   ├── phase1_responses/         # 31 models × 5 datasets
│   ├── phase2_responses/         # 8 models × 3 datasets × 4 token limits
│   ├── phase3_responses/         # 13 models × 4 datasets × 5-6 token limits
│   ├── causal_responses/         # 7 models × 115 problems × 2 conditions
│   └── random_sample_responses/  # 4 models × 2 datasets × 3 conditions
├── processed/                    # Processed results (committed)
│   ├── phase1_accuracy.csv
│   ├── phase2_ablation.csv
│   ├── phase3_reasoning.csv
│   ├── causal_results.csv
│   ├── random_sample_results.csv
│   ├── mechanistic_dimensions.csv
│   ├── qscores.csv
│   └── gpqa_gsm8k_peaks.csv
└── annotations/                  # Human annotations
    ├── error_annotations.csv     # 58 errors, 2 annotators, blinded
    └── annotation_protocol.md    # Annotation guidelines
```

## Downloading Raw Data

Raw model responses are too large for GitHub. Download from:

```bash
# Option 1: Zenodo (permanent DOI)
wget https://zenodo.org/record/XXXXXXX/files/raw_responses.tar.gz
tar -xzf raw_responses.tar.gz -C data/raw/

# Option 2: HuggingFace Datasets
python scripts/download_data.py
```

## Processed Data

All processed CSV files are committed to the repository.
Running `make analysis` regenerates them from raw data.
Running `make figures` and `make tables` uses processed data only.
