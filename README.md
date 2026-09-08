# STTN-CP for Credit Card Fraud Detection (sttn-cp-ccfd)

STTN-CP is a Spatial-Temporal Transformer Network with Contrastive Pretraining for credit card fraud detection. The work combines spatial correlations among transaction features, temporal dependencies across transaction sequences, residual Transformer blocks, and contrastive representation learning for fraud classification. The repository provides the model implementation, data preparation, training and evaluation workflow, unit tests, result artifacts, and a Jupyter notebook.

- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan

## Originality & Novelty

The published work proposes STTN-CP as a unified framework that integrates spatial-temporal Transformer modeling with contrastive pretraining for credit card fraud detection. The model is designed to capture both spatial correlations among transaction features and temporal dependencies across transaction sequences while improving the discriminative quality of learned representations. The study also emphasizes imbalance-aware evaluation using accuracy, precision, recall, F1-score, and specificity.

The architecture uses stacked spatial-temporal Transformer blocks, followed by contrastive representation learning and binary classification. The implementation combines spatial-temporal Transformer modeling with contrastive representation learning as the central design of STTN-CP for credit card fraud detection.

## Model Architecture

Formulates the input to each spatial-temporal block as a three-dimensional tensor:

$$
M_l^{sp} \in \mathbb{R}^{A \times T \times d_f}
$$

where $A$ denotes the batch dimension, $T$ is the number of transaction time steps, and $d_f$ is the number of transaction features. The dataset used in the study contains 30 transaction attributes: `V1`-`V28`, `Time`, and `Amount`, with `Class` used as the target label.

Each spatial-temporal block applies the **Spatial Transformer** and **Temporal Transformer** in sequence. The Spatial Transformer captures inter-feature dependencies across transaction attributes. Its output is combined with the block input through a residual connection. The resulting representation is then processed by the Temporal Transformer to capture dependencies across consecutive transactions, followed by a second residual connection.

After the stacked spatial-temporal blocks, the resulting high-level embeddings are aggregated and passed to the contrastive pretraining module. The learned encoder representation is then used by a fully connected binary classification head with sigmoid activation.

The repository implementation uses 8-step transaction windows for its configurable sequence input and 30 transaction features for the Kaggle dataset.

```
input tensor (A, T, d_f)
            │
     input embedding
            │
    positional encoding
            │
    ┌───────▼────────┐
    │   ST Block l   │
    │                │
    │ Spatial        │
    │ Transformer    │
    │ inter-feature  │
    │ dependencies   │
    │       ↓        │
    │   residual     │
    │       ↓        │
    │ Temporal       │
    │ Transformer    │
    │ sequential     │
    │ dependencies   │
    │       ↓        │
    │   residual     │
    └───────┬────────┘
            │
           ⋮
            │
    ┌───────▼────────┐
    │   ST Block L   │
    └───────┬────────┘
            │
       aggregation
            │
   spatial-temporal
      representation
            │
   contrastive pretraining
            │
      learned embedding
            │
     fully connected
     classification head
            │
      sigmoid output
            │
      legitimate / fraud
```

## Architecture Components

- **Input Embedding:** Transforms transaction features into a continuous high-dimensional representation suitable for Transformer processing.
- **Positional Encoding:** Adds sequence-order information so that temporal relationships can be modeled across transactions.
- **Spatial Transformer:** Applies self-attention across transaction features to capture inter-feature dependencies.
- **Temporal Transformer:** Applies self-attention across consecutive transactions to capture temporal and behavioral dependencies.
- **Residual Connections:** Combine the Transformer outputs with their corresponding inputs to support stable gradient flow.
- **Layer Normalization:** Normalizes intermediate Transformer representations and supports stable training.
- **Representation Aggregation:** Aggregates the high-level spatial-temporal embeddings before the contrastive and classification stages.
- **Contrastive Pretraining Module:** Uses augmented transaction samples and cosine-similarity-based InfoNCE learning to improve feature discrimination.
- **Classifier Head:** Uses a fully connected layer with sigmoid activation for binary fraud classification.
- **Combined Objective:** Defines the final objective as the sum of the classification and contrastive losses:

$$
L_{\mathrm{total}} = L_{\mathrm{classification}} + \lambda L_{\mathrm{contrastive}}
$$

where $\lambda$ controls the contribution of the contrastive learning objective.

## Model Hyperparameters

| Component | Value |
|---|---:|
| Train/test split | 80% / 20% |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training limit | 50 epochs |
| Early stopping | Used |
| InfoNCE temperature | 0.07 |
| Preprocessing | Min-Max normalization + SMOTE |

The implementation uses an InfoNCE contrastive stage with a temperature coefficient of 0.07, followed by supervised binary classification using a sigmoid classification head.

The repository implementation additionally exposes configurable architecture defaults of 8 transaction steps, 30 input features, 3 stacked ST blocks, embedding width 128, 4 attention heads, feed-forward width 256, contrastive projection dimension 64, and an implementation-level contrastive weight of $\lambda=0.5$.

## Data

The project uses the Kaggle Credit Card Fraud Detection dataset. The dataset contains 284,807 transactions from European cardholders during September 2013, including 492 fraudulent transactions and 284,315 legitimate transactions. The transaction attributes consist of 28 anonymized PCA components (`V1`–`V28`), `Time`, and `Amount`; `Class` is the binary target label.

The reference configuration uses Min-Max normalization followed by SMOTE to address class imbalance, with an 80/20 train/test split.

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File name: `creditcard.csv`
- Target column: `Class`
- Feature columns: `Time`, `V1`–`V28`, `Amount`
- Number of input features: 30

Download the dataset and place it at `data/creditcard.csv`.

## Repository protocol

The reference configuration uses an 80% training and 20% testing split with Min-Max normalization and SMOTE.

The repository also contains an implementation-oriented workflow for validation and checkpoint handling:

- **Repository split:** Transactions are divided into chronological 70% training, 10% validation, and 20% test partitions.
- **Repository scaling:** `MinMaxScaler` is fitted on the training partition and applied to validation and test data.
- **Repository class balancing:** `BorderlineSMOTE` is applied only to the training partition.
- **Repository sequence construction:** Fixed-length windows are constructed within each partition so that a sequence does not cross a partition boundary.
- **Sequence labels:** For supervised sequence classification, the final transaction in a window supplies the sequence label.

## Running the experiments

The repository training workflow uses its implementation-oriented protocol described above:

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

The workflow runs the unit tests and an end-to-end training smoke test on every push to `main` and on pull requests targeting `main`.

## Output

Training writes `outputs/best_model.pt` and `outputs/training_summary.json`. Evaluation writes `outputs/test_metrics.json`. The evaluation output includes accuracy, precision, recall, F1-score, ROC-AUC, average precision, specificity, and the confusion matrix for the fraud class.

## Results

Experiments results report the following test-set performance:

| Metric | Value |
|---|---:|
| Accuracy | 99.12% |
| Precision | 99.00% |
| Recall | 98.86% |
| F1 | 98.92% |
| Specificity | 97.96% |

These values are the testing-performance values from the experiments and validations. 

Statistical analysis reports the following mean ± standard deviation values:

| Metric | Mean ± STD |
|---|---:|
| Accuracy | 99.18 ± 0.08% |
| Precision | 99.08 ± 0.11% |
| Recall | 98.92 ± 0.08% |
| F1 | 98.96 ± 0.06% |
| Specificity | 98.24 ± 0.40% |

The ablation study reports the effect of the individual components and the full STTN-CP configuration:

| Configuration | Accuracy | Precision | Recall | F1 | Specificity |
|---|---:|---:|---:|---:|---:|
| Spatial Transformer only | 97.25% | 97.06% | 96.85% | 96.94% | 95.80% |
| Temporal Transformer only | 97.10% | 96.88% | 96.52% | 96.70% | 95.60% |
| Spatial + Temporal Transformer without Contrastive Pretraining | 98.46% | 98.21% | 97.98% | 98.09% | 96.90% |
| Spatial + Temporal Transformer + Pretraining without SMOTE | 98.03% | 97.84% | 97.60% | 97.72% | 96.50% |
| **This STTN-CP model** | **99.12%** | **99.00%** | **98.86%** | **98.92%** | **97.96%** |

## Status

[kathiresan-jayabalan/trans-fasnet-ccfd](https://github.com/kathiresan-jayabalan/trans-fasnet-ccfd) is a baseline implementation. [kathiresan-jayabalan/sttn-cp-ccfd](https://github.com/kathiresan-jayabalan/sttn-cp-ccfd) implements the STTN-CP design with sequential spatial-temporal Transformer blocks, residual connections, and contrastive representation learning. Within each ST block, spatial attention models relationships among transaction features, followed by temporal attention across consecutive transactions. STTN-CP uses the sequential spatial-temporal design, while [kathiresan-jayabalan/c-sten-ccfd](https://github.com/kathiresan-jayabalan/c-sten-ccfd) develops a later architecture using parallel spatial and temporal branches with gated fusion.

## Publication

**Paper:** STTN-CP: A Spatial-Temporal Transformer with Contrastive Pretraining Model for Credit Card Fraud Detection

- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan
- **Journal:** *Journal of Theoretical and Applied Information Technology (JATIT)*, Vol. 104, No. 7, 15 April 2026, pp. 305–324
- **Indexing:** [Scopus-indexed](https://www.scopus.com/sourceid/19700182903) | [SCImago Journal Rank (SJR)](https://www.scimagojr.com/journalsearch.php?q=19700182903&tip=sid)
- **DOI:** [10.5281/zenodo.19593993](https://doi.org/10.5281/zenodo.19593993)
