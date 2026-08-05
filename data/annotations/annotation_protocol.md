# Error Annotation Protocol

## Overview
- **N errors annotated**: 58 (29 control, 29 brief)
- **Annotators**: 2 independent annotators
- **Blinding**: Condition labels (control vs brief) hidden from second annotator
- **Agreement**: Cohen's κ = 0.886, 94.7% raw agreement

## Categories

### 1. Premature Termination (Incomplete)
Reasoning abandoned before reaching a solution despite available token budget.
Model stops mid-chain or provides incomplete reasoning.

### 2. Error Propagation (Logical)
Faulty logic propagated through reasoning steps.
Model completes reasoning but arrives at wrong answer due to logical errors.

## Annotation Procedure
1. Annotator 1 labels all 58 errors with full condition information
2. Annotator 2 independently labels same 58 errors with condition labels REMOVED
3. Disagreements resolved by discussion
4. Cohen's κ computed on pre-resolution labels
