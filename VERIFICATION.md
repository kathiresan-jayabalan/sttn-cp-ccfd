# Verification Record

The rebuilt repository was checked locally before packaging.

## Checks completed

```text
python -m pytest -q
6 passed

python -m compileall -q src tests
passed

Jupyter notebook execution
passed
```

An additional two-epoch command-line smoke run was completed against a small synthetic transaction table with the same 30-feature shape as the target dataset. That run exercised data loading, chronological partitioning, train-only scaling, eight-step window construction, spatial attention, temporal attention, combined loss, checkpoint writing, validation, and test evaluation.

The synthetic smoke run is a software verification only. It is not a credit-card fraud benchmark and is not reported as a research result.

## Benchmark status

The public benchmark file `creditcard.csv` was not present in the working session and could not be retrieved from the public data hosts available to the execution environment. No paper performance number has therefore been labeled as reproduced by this repository build.
