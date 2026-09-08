# STTN-CP for Credit Card Fraud Detection (sttn-cp-ccfd)

STTN-CP is a spatial-temporal Transformer model for credit card fraud detection that combines feature-level spatial attention, temporal sequence modeling, residual connections, and contrastive representation learning. The repository provides the model implementation, data preparation, training and evaluation workflow, unit tests, recorded experimental results, and a reproducible Jupyter notebook.

- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan

## Originality & Novelty 

STTN-CP combines spatial feature interactions, temporal transaction dependencies, and contrastive representation learning within a unified Transformer architecture for credit card fraud detection.

The model explicitly separates two forms of information within a transaction sequence: relationships among transaction features and dependencies across consecutive transactions. Sequential spatial-temporal Transformer blocks build the representation, while contrastive learning is used to improve representation quality before binary fraud classification.

## Model Architecture

The model accepts a three-dimensional input tensor with batch, timestep, and transaction-feature axes, represented as $(B, T=8, D=30)$ for the Kaggle Credit Card Fraud Detection dataset.

Each transaction window is processed through a stack of sequential spatial-temporal Transformer blocks. Within each block, spatial attention models relationships across the $D$ transaction features at each timestep, followed by a residual connection. Temporal attention then models dependencies across the $T$ transactions in the window, followed by a second residual connection. The resulting representations are aggregated across the timestep dimension and passed to both the contrastive projection head and the binary classification head.

```
transaction window (B, T=8, D=30)
            │
     input projection
            │
     positional encoding
            │
    ┌───────▼────────┐
    │   ST Block 1   │
    │                │
    │ spatial        │
    │ attention      │
    │ over D         │
    │ features       │
    │      ↓         │
    │   residual     │
    │      ↓         │
    │ temporal       │
    │ attention      │
    │ over T         │
    │      ↓         │
    │   residual     │
    └───────┬────────┘
            │
          ⋮
            │
    ┌───────▼────────┐
    │   ST Block L   │
    └───────┬────────┘
            │
        mean over T
            │
        embedding z
         ┌──┴──┐
         │     │
  projection   classifier
      head        head
         │         │
     InfoNCE    fraud logits
         │         │
         └──┬──────┘
            │
   L = L_class + λ L_contrastive
```

## Architecture Components

- **Input Projection:** Maps the transaction feature vector into the Transformer embedding space.
- **Positional Encoding:** Adds information about transaction order within the fixed-length sequence.
- **Spatial Transformer:** Applies multi-head self-attention across transaction features at each timestep to model inter-feature relationships.
- **Temporal Transformer:** Applies multi-head self-attention across consecutive transactions to model sequential dependencies.
- **Residual Connections:** Preserve the incoming representation around both spatial and temporal Transformer operations.
- **Representation Aggregation:** Mean pooling across the $T$ timesteps produces a fixed-size sequence representation.
- **Contrastive Projection Head:** Maps the aggregated representation into the contrastive embedding space for InfoNCE learning between augmented views.
- **Classifier Head:** Produces binary fraud logits for legitimate and fraudulent transactions.
- **Combined Objective:** The training objective combines supervised classification and contrastive representation learning:
$$
L = L_{\mathrm{class}} + \lambda L_{\mathrm{contrastive}}
$$

## Model Hyperparameters

This implementation uses:

| Component | Default |
|---|----|
| Window length | 8 transactions |
| Input features | 30 for the Kaggle dataset |
| Transformer blocks | 3 |
| Embedding width | 128 |
| Attention heads | 4 |
| Feed-forward width | 256 |
| Contrastive projection | 64 |
| InfoNCE temperature | 0.07 |
| λ for contrastive term | 0.5 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training limit | 50 epochs |
| Early stopping | validation F1 |

- Input: Transaction windows of length 8 (configurable).
- Spatial-Temporal Encoder: 3 stacked ST blocks, with spatial and temporal Transformer components using 4 attention heads (configurable).
- Embedding size: 128 (configurable).
- Contrastive learning: InfoNCE loss with temperature $\tau = 0.07$, using augmented views of transaction sequences.
- Classification: Binary fraud classification using cross-entropy loss, with BorderlineSMOTE applied to the training partition to address class imbalance.
- Optimization: Adam optimizer with a learning rate of 0.001, batch size of 128, and up to 50 epochs with early stopping.
- Combined objective: $L = L_{\mathrm{class}} + \lambda L_{\mathrm{contrastive}}$, with $\lambda = 0.5$.

The training settings retain choices of 50 epochs, batch size 128, Adam at 0.001, and InfoNCE temperature 0.07. 

## Data

The project uses the Kaggle Credit Card Fraud Detection dataset. Min-Max normalization and SMOTE and reports an 80/20 experiment split.

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File name: `creditcard.csv`
- Target column: `Class`
- Time column: `Time`

Download the dataset and place it at `data/creditcard.csv`.

## Repository protocol

- **Split:** Transactions are sorted by Time and divided chronologically into 70% training, 10% validation, and 20% test partitions.
- **Scaling:** `MinMaxScaler` is fitted on the training partition only and then applied to validation and test data.
- **Class balancing:** `BorderlineSMOTE` is applied only to the training partition. Validation and test data remain unsampled.
- **Sequence construction:** Fixed-length windows of 8 consecutive transactions are constructed separately within each partition so that a window never crosses a partition boundary.
- **Sequence labels:** For supervised sequence classification, the label assigned to a window is the class of its final transaction.

## Running the experiments

```text
sort transactions by Time
          ↓
70% train | 10% validation | 20% test
          ↓
fit Min-Max scaler on training partition
          ↓
transform validation/test using training scaler
          ↓
build 8-step windows independently
          ↓
apply BorderlineSMOTE to training partition
          ↓
train spatial-temporal Transformer
          ↓
contrastive + classification optimization
          ↓
select checkpoint using validation F1
          ↓
evaluate on held-out test partition

sort by Time
    ↓
70% train | 10% validation | 20% test
    ↓
fit Min-Max on train only
    ↓
transform validation/test with train scaler
    ↓
build 8-step windows independently in each partition
    ↓
class-weighted supervised loss
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.train \
  --data-path data/creditcard.csv \
  --output-dir outputs
```

Evaluate the trained model:

```bash
python -m src.evaluate \
  --data-path data/creditcard.csv \
  --checkpoint outputs/best_model.pt \
  --output outputs/test_metrics.json
```

## Tests

The test suite checks the structural and data-processing properties that are important to the STTN-CP implementation:

- spatial attention receives exactly the original feature count as its token sequence;
- classification requires the complete `(B, T, D)` window;
- the final row in each window supplies its label;
- windows are partition-local and do not cross train/validation/test boundaries;
- InfoNCE rejects an invalid one-sample batch.

Run the tests with:

```bash
python -m pytest -q
```

Continuous integration is configured under:

```bash
.github/workflows/tests.yml
```
The workflow runs the unit tests and an end-to-end training smoke test on every push to main and pull request targeting main.

## Output

Training writes outputs/best_model.pt and outputs/training_summary.json. Evaluation writes outputs/test_metrics.json. The evaluation output includes accuracy, precision, recall, F1-score, ROC-AUC, average precision, specificity, and the confusion matrix for the fraud class.

## Results

See [`results/README.md`](results/README.md) for the test run, structured metrics, training log, and reproduction details.

The configuration uses:

8-transaction windows
30 input features
3 stacked spatial-temporal Transformer blocks
4 attention heads
embedding width 128
batch size 128
Adam optimizer
learning rate 0.001
InfoNCE temperature $\tau=0.07$
$\lambda=0.5$
maximum of 50 training epochs with early stopping

The recorded run reports 859,458 trainable parameters and completion with early stopping at epoch 42. Reported statistical performance of 99.18% ± 0.08% accuracy, 99.08% ± 0.11% precision, 98.92% ± 0.08% recall, 98.96% ± 0.06% F1-score, and 98.24% ± 0.40% specificity.

| Metric | Value |
|---|---:|
| Accuracy | 99.12% |
| Precision | 99.00% |
| Recall | 98.86% |
| F1 | 98.92% |
| Specificity | 97.96% |

## Status
[kathiresan-jayabalan/trans-fasnet-ccfd](https://github.com/kathiresan-jayabalan/trans-fasnet-ccfd) is a baseline implementation. [kathiresan-jayabalan/sttn-cp-ccfd](https://github.com/kathiresan-jayabalan/sttn-cp-ccfd) extends that baseline by introducing sequential spatial-temporal Transformer blocks with residual connections and contrastive representation learning. Within each ST block, spatial attention models relationships across transaction features, followed by temporal attention across consecutive transactions. STTN-CP represents the sequential spatial-temporal architecture, while [kathiresan-jayabalan/c-sten-ccfd](https://github.com/kathiresan-jayabalan/c-sten-ccfd) develops that design further through parallel spatial and temporal branches with gated fusion.

## Publication
**Paper:** STTN-CP: A Spatial-Temporal Transformer with Contrastive Pretraining Model for Credit Card Fraud Detection
- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan
- **Journal:** *Journal of Theoretical and Applied Information Technology (JATIT)*, Vol. 104, No. 7, April 2026
- **Indexing:** [Scopus-indexed](https://www.scopus.com/sourceid/19700182903) | [SCImago Journal Rank (SJR)](https://www.scimagojr.com/journalsearch.php?q=19700182903&tip=sid)
- **DOI:** [10.5281/zenodo.19593993](https://doi.org/10.5281/zenodo.19593993)
