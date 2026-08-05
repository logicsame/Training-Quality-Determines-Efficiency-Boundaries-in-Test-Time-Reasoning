# TOKEN LIMIT ABLATION STUDY: SUMMARY REPORT

**Generated:** 2026-02-15 02:10:32

---

## Models Tested (6)

- AIDC-AIMarco-o1_7b
- deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B
- deepseek-aiDeepSeek-R1-Distill-Qwen-14B
- deepseek-aiDeepSeek-R1-Distill-Qwen-7B
- nvidiaOpenReasoning-Nemotron-7B
- SkyworkSkywork-o1-Open-Llama-3.1-8B

---

## Datasets

- GSM8K (grade-school math)
- BoolQ (yes/no questions)
- ARC-Easy (multiple choice science)
- GPQA (Graduate-Level Google-Proof Q&A)

---

## 🎯 KEY FINDINGS

### Finding 1: Plateau Effect (1000→2000 tokens)

| Model | GSM8K | BoolQ | ARC-Easy |
|-------|-------|-------|----------|
| AIDC-AIMarco-o1_7b | -2.0pp (p=0.84) ✗ | +2.0pp (p=0.73) ✗ | +0.0pp (p=1.00) ✓ | -4.0pp (p=0.66) ✗ |
| deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B | +8.0pp (p=0.42) ✗ | +0.0pp (p=1.00) ✓ | +0.0pp (p=1.00) ✓ | -2.0pp (p=0.84) ✗ |
| deepseek-aiDeepSeek-R1-Distill-Qwen-14B | +12.0pp (p=0.16) ✗ | +0.0pp (p=1.00) ✓ | +0.0pp (p=1.00) ✓ | +8.0pp (p=0.39) ✗ |
| deepseek-aiDeepSeek-R1-Distill-Qwen-7B | +14.0pp (p=0.15) ✗ | +0.0pp (p=1.00) ✓ | +2.0pp (p=0.70) ✗ | -2.0pp (p=0.83) ✗ |
| nvidiaOpenReasoning-Nemotron-7B | +12.0pp (p=0.22) ✗ | +2.0pp (p=0.83) ✗ | +6.0pp (p=0.44) ✗ | -12.0pp (p=0.23) ✗ |
| SkyworkSkywork-o1-Open-Llama-3.1-8B | +0.0pp (p=1.00) ✓ | +0.0pp (p=1.00) ✓ | +0.0pp (p=1.00) ✓ | +0.0pp (p=1.00) ✓ |

---

### Finding 2: Constraint Severity @ 300 Tokens

| Model | Dataset | Hit Limit % | Avg Tokens | Accuracy | Severity |
|-------|---------|-------------|------------|----------|----------|
| AIDC-AIMarco-o1_7b | GSM8K | 98.0% | 299 | 16.0% | SEVERE |
| AIDC-AIMarco-o1_7b | BOOLQ | 58.0% | 198 | 92.0% | SEVERE |
| AIDC-AIMarco-o1_7b | ARC-EASY | 100.0% | 300 | 40.0% | SEVERE |
| AIDC-AIMarco-o1_7b | GPQA | 100.0% | 300 | 44.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B | GSM8K | 100.0% | 300 | 16.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B | BOOLQ | 38.0% | 161 | 38.0% | MODERATE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B | ARC-EASY | 40.0% | 236 | 38.0% | MODERATE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B | GPQA | 98.0% | 297 | 50.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-14B | GSM8K | 90.0% | 296 | 42.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-14B | BOOLQ | 6.0% | 44 | 80.0% | MILD |
| deepseek-aiDeepSeek-R1-Distill-Qwen-14B | ARC-EASY | 46.0% | 233 | 70.0% | MODERATE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-14B | GPQA | 94.0% | 295 | 46.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-7B | GSM8K | 100.0% | 300 | 12.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-7B | BOOLQ | 62.0% | 261 | 64.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-7B | ARC-EASY | 86.0% | 290 | 34.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Qwen-7B | GPQA | 100.0% | 300 | 62.0% | SEVERE |
| nvidiaOpenReasoning-Nemotron-7B | GSM8K | 94.0% | 294 | 38.0% | SEVERE |
| nvidiaOpenReasoning-Nemotron-7B | BOOLQ | 90.0% | 292 | 62.0% | SEVERE |
| nvidiaOpenReasoning-Nemotron-7B | ARC-EASY | 90.0% | 294 | 74.0% | SEVERE |
| nvidiaOpenReasoning-Nemotron-7B | GPQA | 100.0% | 300 | 32.0% | SEVERE |
| SkyworkSkywork-o1-Open-Llama-3.1-8B | GSM8K | 48.0% | 258 | 58.0% | MODERATE |
| SkyworkSkywork-o1-Open-Llama-3.1-8B | BOOLQ | 12.0% | 103 | 54.0% | MILD |
| SkyworkSkywork-o1-Open-Llama-3.1-8B | ARC-EASY | 54.0% | 250 | 56.0% | SEVERE |
| SkyworkSkywork-o1-Open-Llama-3.1-8B | GPQA | 96.0% | 297 | 40.0% | SEVERE |

---

## 📊 RECOMMENDATIONS

Based on the analysis:

1. **Optimal token budget:** 1000 tokens for most tasks
2. **Complex tasks (GSM8K):** Need 500+ tokens to avoid severe truncation
3. **Simple tasks (BoolQ):** Can use 300 tokens safely
4. **Cost optimization:** Going beyond 1000 tokens provides minimal benefit
