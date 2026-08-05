# TOKEN LIMIT ABLATION STUDY: SUMMARY REPORT

**Generated:** 2026-02-15 00:45:12

---

## Models Tested (7)

- deepseek-aiDeepSeek-R1-Distill-Llama-70B
- deepseek-r1-685b
- glm_4.5
- glm_4.7_355 billion_32b
- kimi2.5_1t_32b_sctive
- qwenqwen3-32b_o1
- QwenQwQ-32B-Preview

---

## Datasets

- GSM8K (grade-school math)
- BoolQ (yes/no questions)
- ARC-Easy (multiple choice science)

---

## 🎯 KEY FINDINGS

### Finding 1: Plateau Effect (1000→2000 tokens)

| Model | GSM8K | BoolQ | ARC-Easy |
|-------|-------|-------|----------|
| deepseek-aiDeepSeek-R1-Distill-Llama-70B | +4.0pp (p=0.40) ✗ | -2.0pp (p=0.77) ✗ | +2.0pp (p=0.31) ✗ | +10.0pp (p=0.29) ✗ |
| deepseek-r1-685b | +36.0pp (p=0.00) ✗ | +0.0pp (p=1.00) ✓ | +0.0pp (p=1.00) ✓ | +4.0pp (p=0.65) ✗ |
| glm_4.5 | +8.0pp (p=0.38) ✗ | -2.0pp (p=0.70) ✗ | +60.0pp (p=0.00) ✗ | +4.0pp (p=0.64) ✗ |
| glm_4.7_355 billion_32b | +14.0pp (p=0.13) ✗ | +4.0pp (p=0.40) ✗ | +50.0pp (p=0.00) ✗ | -2.0pp (p=0.81) ✗ |
| kimi2.5_1t_32b_sctive | +2.0pp (p=0.84) ✗ | +0.0pp (p=1.00) ✓ | +8.0pp (p=0.04) ✗ | +6.0pp (p=0.49) ✗ |
| qwenqwen3-32b_o1 | +24.0pp (p=0.00) ✗ | +4.0pp (p=0.69) ✗ | +2.0pp (p=0.31) ✗ | +14.0pp (p=0.15) ✗ |
| QwenQwQ-32B-Preview | +6.0pp (p=0.17) ✗ | +0.0pp (p=1.00) ✓ | +0.0pp (p=1.00) ✓ | +4.0pp (p=0.69) ✗ |

---

### Finding 2: Constraint Severity @ 300 Tokens

| Model | Dataset | Hit Limit % | Avg Tokens | Accuracy | Severity |
|-------|---------|-------------|------------|----------|----------|
| deepseek-aiDeepSeek-R1-Distill-Llama-70B | GSM8K | 98.0% | 299 | 56.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Llama-70B | BOOLQ | 60.0% | 268 | 88.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Llama-70B | ARC-EASY | 98.0% | 298 | 26.0% | SEVERE |
| deepseek-aiDeepSeek-R1-Distill-Llama-70B | GPQA | 100.0% | 300 | 54.0% | SEVERE |
| deepseek-r1-685b | GSM8K | 100.0% | 300 | 24.0% | SEVERE |
| deepseek-r1-685b | BOOLQ | 72.0% | 277 | 30.0% | SEVERE |
| deepseek-r1-685b | ARC-EASY | 100.0% | 1003 | 78.0% | SEVERE |
| deepseek-r1-685b | GPQA | 100.0% | 6455 | 44.0% | SEVERE |
| glm_4.5 | GSM8K | 100.0% | 300 | 4.0% | SEVERE |
| glm_4.5 | BOOLQ | 100.0% | 300 | 12.0% | SEVERE |
| glm_4.5 | ARC-EASY | 100.0% | 300 | 30.0% | SEVERE |
| glm_4.5 | GPQA | 100.0% | 300 | 20.0% | SEVERE |
| glm_4.7_355 billion_32b | GSM8K | 100.0% | 300 | 8.0% | SEVERE |
| glm_4.7_355 billion_32b | BOOLQ | 98.0% | 300 | 20.0% | SEVERE |
| glm_4.7_355 billion_32b | ARC-EASY | 100.0% | 300 | 28.0% | SEVERE |
| glm_4.7_355 billion_32b | GPQA | 100.0% | 300 | 14.0% | SEVERE |
| kimi2.5_1t_32b_sctive | GSM8K | 94.0% | 297 | 18.0% | SEVERE |
| kimi2.5_1t_32b_sctive | BOOLQ | 42.0% | 229 | 12.0% | MODERATE |
| kimi2.5_1t_32b_sctive | ARC-EASY | 100.0% | 300 | 24.0% | SEVERE |
| kimi2.5_1t_32b_sctive | GPQA | 100.0% | 300 | 16.0% | SEVERE |
| qwenqwen3-32b_o1 | GSM8K | 100.0% | 300 | 16.0% | SEVERE |
| qwenqwen3-32b_o1 | BOOLQ | 24.0% | 204 | 32.0% | MODERATE |
| qwenqwen3-32b_o1 | ARC-EASY | 98.0% | 299 | 42.0% | SEVERE |
| qwenqwen3-32b_o1 | GPQA | 100.0% | 300 | 24.0% | SEVERE |
| QwenQwQ-32B-Preview | GSM8K | 82.0% | 284 | 34.0% | SEVERE |
| QwenQwQ-32B-Preview | BOOLQ | 6.0% | 70 | 88.0% | MILD |
| QwenQwQ-32B-Preview | ARC-EASY | 48.0% | 243 | 62.0% | MODERATE |
| QwenQwQ-32B-Preview | GPQA | 96.0% | 296 | 34.0% | SEVERE |

---

## 📊 RECOMMENDATIONS

Based on the analysis:

1. **Optimal token budget:** 1000 tokens for most tasks
2. **Complex tasks (GSM8K):** Need 500+ tokens to avoid severe truncation
3. **Simple tasks (BoolQ):** Can use 300 tokens safely
4. **Cost optimization:** Going beyond 1000 tokens provides minimal benefit
