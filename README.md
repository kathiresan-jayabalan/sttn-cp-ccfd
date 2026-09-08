# STTN-CP for Credit Card Fraud Detection

STTN-CP is a Spatial-Temporal Transformer Network with Contrastive Pretraining for credit card fraud detection. The implementation models relationships among transaction features with spatial self-attention and sequential dependencies across transaction windows with temporal self-attention.

- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan

## Originality & Novelty

STTN-CP is a unified framework that integrates spatial-temporal Transformer modeling with contrastive pretraining for credit card fraud detection. The model is designed to capture both spatial correlations among transaction features and temporal dependencies across transaction sequences while improving the discriminative quality of learned representations. The evaluation uses accuracy, precision, recall, F1-score, and specificity to assess fraud detection performance.

The architecture uses stacked spatial-temporal Transformer blocks, followed by contrastive representation learning and binary classification. The implementation combines spatial-temporal Transformer modeling with contrastive representation learning as the central design of STTN-CP for credit card fraud detection.

## Model Architecture

The model represents each input as a transaction window with shape `(B, T, D)`, where `B` is the batch size, `T` is the sequence length, and `D` is the number of transaction features.

```text
input transaction window (B, T, D)
            │
            ▼
     input embedding
            │
            ▼
     positional encoding
            │
            ▼
       ┌─────────────┐
       │   ST Block  │
       │             │
       │   Spatial   │  attention across transaction features
       │      │      │
       │   residual  │
       │      │      │
       │   Temporal  │  attention across transaction timesteps
       │      │      │
       │   residual  │
       └─────────────┘
            │
          repeat
            │
            ▼
    stacked ST blocks
            │
            ▼
     mean aggregation
            │
       ┌────┴─────┐
       │          │
       ▼          ▼
 contrastive   classifier
 projection      head
       │          │
       ▼          ▼
    InfoNCE    class logits
       │          │
       └────┬─────┘
            ▼
 L = L_classification + λ L_contrastive
```

The sequential block structure is spatial attention → residual addition → temporal attention → residual addition. Spatial attention operates over the original transaction features at each timestep, while temporal attention operates over the embedded transaction sequence.

## Architecture Components

- **Input Embedding:** Maps the transaction feature vector into the Transformer embedding space.
- **Positional Encoding:** Adds sequence-order information for temporal modeling.
- **Spatial Transformer:** Applies multi-head self-attention across transaction features at each timestep.
- **Temporal Transformer:** Applies multi-head self-attention across consecutive transactions in a window.
- **Residual Connections:** Preserve the incoming representation around the spatial and temporal transformations.
- **Layer Normalization:** Stabilizes intermediate Transformer representations.
- **Representation Aggregation:** Mean pooling across the transaction window produces a fixed-size sequence representation.
- **Contrastive Projection Head:** Projects the aggregated representation into the contrastive space for InfoNCE learning between augmented views.
- **Classifier Head:** Produces two class logits for legitimate and fraudulent transactions.
- **Combined Objective:**

$$
L_{\mathrm{total}}
=
L_{\mathrm{classification}}
+
\lambda L_{\mathrm{contrastive}}
$$

where `λ` controls the contribution of the contrastive objective.

## Model Configuration

The maintained implementation uses the following defaults:

| Component | Default |
|---|---:|
| Window length | 8 transactions |
| Input features | 30 |
| ST blocks | 3 |
| Embedding width | 128 |
| Attention heads | 4 |
| Feed-forward width | 256 |
| Contrastive projection | 64 |
| InfoNCE temperature | 0.07 |
| λ | 0.5 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training limit | 50 epochs |
| Early stopping | Validation F1 |

The current implementation also uses a cosine-annealing learning-rate scheduler and weighted cross-entropy for class imbalance. These are implementation choices in the maintained repository and are separate from the archived original experiment configuration.

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset.

- Source: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`
- File: `creditcard.csv`
- Target: `Class`
- Feature columns: `Time`, `V1`-`V28`, `Amount`
- Number of input features: 30
- Transactions: 284,807
- Fraudulent transactions: 492
- Legitimate transactions: 284,315

Place the dataset at:

```text
 data/creditcard.csv
```

## Repository Protocol

The maintained repository uses a leakage-safe, implementation-oriented workflow:

```text
sort transactions by Time
        ↓
70% train | 10% validation | 20% test
        ↓
fit Min-Max scaler on training partition only
        ↓
transform validation/test with training scaler
        ↓
build 8-step windows independently within each partition
        ↓
weighted classification + contrastive optimization
        ↓
select checkpoint using validation F1
        ↓
evaluate on held-out test partition
```

No sequence crosses a partition boundary. The label of a supervised sequence is the class of its final transaction.

The repository protocol is intentionally documented separately from the archived original experiment. The archived original experiment used an 80/20 train/test configuration with SMOTE. The maintained repository uses chronological validation and test partitions together with its own class-imbalance handling.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m src.train \
  --data-path data/creditcard.csv \
  --output-dir outputs
```

Training writes:

```text
outputs/best_model.pt
outputs/training_summary.json
```

## Evaluation

```bash
python -m src.evaluate \
  --data-path data/creditcard.csv \
  --checkpoint outputs/best_model.pt \
  --output outputs/test_metrics.json
```

Evaluation writes accuracy, precision, recall, F1, average precision, ROC-AUC, specificity, and a 2×2 confusion matrix.

## Tests

The test suite verifies the software contracts that matter for the implementation:

- spatial attention receives exactly the original feature count as its token sequence;
- classification requires a complete `(B, T, D)` transaction window;
- the final transaction in a window supplies its label;
- sequence windows remain within their source partition;
- chronological train/validation/test split sizes and ordering are preserved;
- InfoNCE rejects invalid one-sample batches;
- InfoNCE returns a finite scalar loss for valid inputs;
- the model supports multiple stacked ST blocks;
- the complete train → checkpoint → evaluate workflow runs successfully on a synthetic dataset.

Run the tests with:

```bash
python -m pytest -q
```

Continuous integration is configured under:

```text
.github/workflows/tests.yml
```

The CI workflow runs the test suite and the end-to-end training/evaluation smoke test.

## Results

The archived original experiment reports the following testing performance:

| Metric | Value |
|---|---:|
| Accuracy | 99.12% |
| Precision | 99.00% |
| Recall | 98.86% |
| F1 | 98.92% |
| Specificity | 97.96% |

The associated statistical summary is:

| Metric | Mean ± STD |
|---|---:|
| Accuracy | 99.18 ± 0.08% |
| Precision | 99.08 ± 0.11% |
| Recall | 98.92 ± 0.08% |
| F1 | 98.96 ± 0.06% |
| Specificity | 98.24 ± 0.40% |

The experiment record is preserved under:

```text
results/original_run/
├── run_log.txt
├── training_summary.json
└── test_metrics.json
```

These files describe the original experiment record. They are kept separate from `outputs/`, which contains artifacts generated by the maintained open-source implementation.

## Related Implementations

The repositories below address the same credit-card-fraud-detection problem from different architectural directions:

[kathiresan-jayabalan/trans-fasnet-ccfd](https://github.com/kathiresan-jayabalan/trans-fasnet-ccfd) is a baseline implementation. [kathiresan-jayabalan/sttn-cp-ccfd](https://github.com/kathiresan-jayabalan/sttn-cp-ccfd) implements the STTN-CP design with sequential spatial-temporal Transformer blocks, residual connections, and contrastive representation learning. Within each ST block, spatial attention models relationships among transaction features, followed by temporal attention across consecutive transactions. STTN-CP uses the sequential spatial-temporal design, while [kathiresan-jayabalan/c-sten-ccfd](https://github.com/kathiresan-jayabalan/c-sten-ccfd) develops a later architecture using parallel spatial and temporal branches with gated fusion.

## Publication

**Paper:** STTN-CP: A Spatial-Temporal Transformer with Contrastive Pretraining Model for Credit Card Fraud Detection  
**Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan  
**Journal:** *Journal of Theoretical and Applied Information Technology*, Vol. 104, No. 7, 15 April 2026, pp. 305–324  
**DOI:** `10.5281/zenodo.19593993`

## Citation

See [`CITATION.cff`](CITATION.cff) for software and publication citation metadata.
