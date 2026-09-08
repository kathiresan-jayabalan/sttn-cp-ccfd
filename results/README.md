# Results

This directory contains the archived experimental results and execution records for STTN-CP.

## Original Experiment Run

The `original_run/` directory preserves the experiment associated with the original STTN-CP research results.

The original experimental configuration used:

- Min-Max normalization
- SMOTE for class-imbalance handling
- 80% training / 20% testing split
- 8-transaction sequence windows
- 30 input features
- 3 stacked spatial-temporal Transformer blocks
- 4 attention heads
- embedding width of 128
- batch size of 128
- Adam optimizer
- learning rate of 0.001
- maximum of 50 training epochs with early stopping
- InfoNCE contrastive learning with temperature $\tau = 0.07$
- contrastive weight $\lambda = 0.5$

### Files

| File | Contents |
|---|---|
| `run_log.txt` | Archived execution and evaluation log for the original experiment |
| `training_summary.json` | Original experiment configuration, model details, and statistical performance |
| `test_metrics.json` | Recorded testing performance for the original experiment |

### Final test results

| Metric | Value |
|---|---:|
| Accuracy | 99.12% |
| Precision | 99.00% |
| Recall | 98.86% |
| F1 | 98.92% |
| Specificity | 97.96% |

### Statistical performance

| Metric | Mean ± STD |
|---|---:|
| Accuracy | 99.18 ± 0.08% |
| Precision | 99.08 ± 0.11% |
| Recall | 98.92 ± 0.08% |
| F1 | 98.96 ± 0.06% |
| Specificity | 98.24 ± 0.40% |

### Experimental protocol

- **Dataset:** Kaggle Credit Card Fraud Detection
- **Transactions:** 284,807
- **Fraudulent transactions:** 492
- **Legitimate transactions:** 284,315
- **Input features:** `V1`-`V28`, `Time`, `Amount`
- **Feature count:** 30
- **Sequence length:** 8 transactions
- **Train/test split:** 80% / 20%
- **Normalization:** Min-Max normalization
- **Class balancing:** SMOTE
- **Transformer blocks:** 3 stacked spatial-temporal blocks
- **Attention heads:** 4
- **Embedding dimension:** 128
- **Batch size:** 128
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Training limit:** 50 epochs
- **Early stopping:** Enabled
- **Contrastive learning:** InfoNCE
- **Contrastive temperature:** $\tau = 0.07$
- **Contrastive weight:** $\lambda = 0.5$

### Objective

The training objective combines supervised classification with contrastive representation learning:

```math
L_{\mathrm{total}} = L_{\mathrm{classification}} + \lambda L_{\mathrm{contrastive}}
```

where $\lambda$ controls the contribution of the contrastive objective.

## Runtime Outputs

A normal run of the current implementation produces:

```text
outputs/
├── best_model.pt
├── training_summary.json
└── test_metrics.json
```
### Reproducing this run

1. Open `notebook.ipynb` in Google Colab.
2. Set the runtime to a GPU accelerator.
3. Upload `creditcard.csv` to Google Drive at `My Drive/credit card /creditcard.csv`, or update `FILE_PATH` in the notebook to your own path.
4. Install `imbalanced-learn` if it is not already available in the runtime.
5. Run all cells in order.
