# Verification Record

The repository was verified locally before release.

## Checks completed

```text
python -m pytest -q
6 passed

python -m compileall -q src tests
passed

Jupyter notebook execution
passed
```

An additional two-epoch command-line smoke test was completed using a small synthetic transaction table with the same 30-feature structure as the target dataset.

The smoke test exercised:
- data loading
- chronological partitioning
- train-only scaling
- eight-step transaction-window construction
- spatial attention
- temporal attention
- combined classification and contrastive loss
- checkpoint writing
- validation
- test evaluation

## Benchmark status

The public benchmark file creditcard.csv was not available in the verification environment at the time of the repository check. The verification results above establish the integrity of the repository software workflow.
