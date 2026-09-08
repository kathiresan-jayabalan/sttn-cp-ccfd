# Results

This directory is intentionally free of unverified benchmark numbers.

A normal training run writes:

```text
outputs/best_model.pt
outputs/training_summary.json
```

Evaluation writes a JSON file containing accuracy, precision, recall, F1, average precision, ROC-AUC, specificity, and the confusion matrix.

The JATIT paper reports its own Table 4 values separately in the README and notebook for reference. Those values are not copied into this directory as if they had been reproduced by this implementation.
