# STTN-CP for Credit Card Fraud Detection (sttn-cp-ccfd)

STTN-CP is a spatial-temporal Transformer model for credit card fraud detection that combines spatial and temporal attention with contrastive representation learning. The repository provides the model implementation, data preparation, training and evaluation workflow, unit tests and a reproducible Jupyter notebook.

The model uses stacked spatial-temporal Transformer blocks to process fixed-length transaction sequences. Each block first applies spatial attention across transaction features, followed by a residual connection, and then applies temporal attention across the sequence, followed by a second residual connection. After the stacked blocks, the resulting representations are aggregated and passed through a contrastive projection head and a classification head. Contrastive pretraining uses an InfoNCE loss with two augmented views of the transaction sequence, while supervised classification uses the fraud labels for binary prediction. BorderlineSMOTE is applied to the training partition to address class imbalance.

- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan

## Originality & Novelty 

This implementation realizes the STTN-CP framework proposed for credit card fraud detection by combining spatial feature interactions, temporal transaction dependencies, and contrastive representation learning within a unified Transformer architecture. The spatial-temporal blocks model relationships among transaction attributes and across consecutive transactions, while contrastive learning is used to improve the representation of transaction behavior before classification.

## Model Architecture

The model accepts a three-dimensional input tensor with batch, timestep, and transaction-feature axes, represented as $(B, T=8, D=30)$. It processes each transaction window through a stack of sequential spatial-temporal Transformer blocks. Within each block, spatial attention models relationships across the $D$ transaction features at each timestep, followed by a residual connection. Temporal attention then models dependencies across the $T$ transactions in the window, followed by a second residual connection. The resulting representations are aggregated across the timestep dimension, with the contrastive module operating on the aggregated representation and the binary classification head using the same representation for fraud prediction.

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

- **Input Projection:** Maps the $D=30$ transaction features at each timestep into the transformer embedding space.
- **Positional Encoding:** Adds information about transaction order within the fixed-length sequence.
- **Sequential Spatial-Temporal Transformer Blocks:** Each block first applies multi-head self-attention across the transaction features at each timestep to capture inter-feature relationships. A residual connection is then applied before temporal self-attention operates across the $T=8$ transactions to capture sequential dependencies. A second residual connection produces the representation passed to the next block.
- **Representation Aggregation:** Mean pooling across the $T$ timesteps produces a fixed-size sequence representation.
- **Contrastive Projection Head:** Maps the aggregated representation into the contrastive embedding space and is optimized using the InfoNCE objective between augmented views of the transaction sequence.
- **Classifier Head:** Maps the learned representation to binary fraud logits for legitimate versus fraudulent transactions.
- **Combined Objective:** The training objective combines the classification and contrastive losses:
$L = L_{\mathrm{class}} + \lambda L_{\mathrm{contrastive}}$.

## Model Hyperparameters

This implementation uses:

| Component | Default |
|---|---:|
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

The training settings above retain the major published choices of 50 epochs, batch size 128, Adam at 0.001, and InfoNCE temperature 0.07. 

## Data

The project uses the Kaggle Credit Card Fraud Detection dataset. Min-Max normalization and SMOTE and reports an 80/20 experiment split.

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File name: `creditcard.csv`
- Target column: `Class`
- Time column: `Time`

Download the dataset and place it at `data/creditcard.csv`.

## Data Split & Preprocessing

- **Split Protocol:** Transactions are sorted by Time and divided chronologically into 70% training, 10% validation, and 20% test - **partitions.
- **Scaling:** `MinMaxScaler` is fitted on the training partition only and then applied to the validation and test partitions.
- **Class Balancing:** `BorderlineSMOTE` is applied exclusively to the training partition to address class imbalance. Synthetic samples are not introduced into the validation or test partitions.
- **Pretraining Windows:** Fixed-length sliding windows of 8 consecutive transactions are constructed separately within each partition, - **preventing windows from crossing partition boundaries.
- **Sequence Labels:** For supervised sequence classification, each window is associated with the class label of its final transaction.

## Running the experiments

```text
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
python -m src.train --data-path data/creditcard.csv --output-dir outputs
```
Evaluate the trained model:
```bash
python -m src.evaluate --data-path data/creditcard.csv --checkpoint outputs/best_model.pt --output outputs/test_metrics.json
```

## Tests

The test suite checks the parts of the implementation that matter most for this model:

- spatial attention receives exactly the original feature count as its token sequence;
- classification requires the complete `(B, T, D)` window;
- the final row in each window supplies its label;
- windows are partition-local and do not cross train/validation/test boundaries;
- InfoNCE rejects an invalid one-sample batch.
  
## Output

Training writes outputs/best_model.pt and outputs/training_summary.json. Evaluation writes outputs/test_metrics.json containing accuracy, precision, recall, F1-score, ROC-AUC, average precision, specificity, and the confusion matrix for the fraud class.

## Results

See [`results/README.md`](results/README.md) for the documented paper-aligned run, structured metrics, training log, and reproduction details. The run uses a transaction window length of 8, 30 input features, 3 stacked spatial-temporal Transformer blocks, 4 attention heads, an embedding size of 128, batch size 128, Adam optimization with a learning rate of 0.001, InfoNCE temperature $\tau=0.07$, and $\lambda=0.5$, with training configured for up to 50 epochs and early stopping.

The recorded run completed successfully with early stopping at epoch 42. The associated training summary records 859,458 trainable parameters and the reported statistical performance of 99.18% ± 0.08% accuracy, 99.08% ± 0.11% precision, 98.92% ± 0.08% recall, 98.96% ± 0.06% F1-score, and 98.24% ± 0.40% specificity.

| Metric | Value |
|---|---:|
| Accuracy | 99.12% |
| Precision | 99.00% |
| Recall | 98.86% |
| F1 | 98.92% |
| Specificity | 97.96% |

## Status
[kathiresan-jayabalan/trans-fasnet-ccfd](https://github.com/kathiresan-jayabalan/trans-fasnet-ccfd) is a baseline implementation. [kathiresan-jayabalan/sttn-cp-ccfd](https://github.com/kathiresan-jayabalan/sttn-cp-ccfd extends that baseline by introducing sequential spatial-temporal Transformer blocks with residual connections and contrastive representation learning. Within each ST block, spatial attention models relationships across transaction features, followed by temporal attention across consecutive transactions. STTN-CP represents the sequential spatial-temporal architecture, while [kathiresan-jayabalan/c-sten-ccfd](https://github.com/kathiresan-jayabalan/c-sten-ccfd) develops that design further through parallel spatial and temporal branches with gated fusion.

```bibtex
@article{jayabalan2026sttncp,
  title   = {STTN-CP: A Spatial-Temporal Transformer with Contrastive Pretraining Model for Credit Card Fraud Detection},
  author  = {Jayabalan, Kathiresan and Radhakrishnan, Sethuraman},
  journal = {Journal of Theoretical and Applied Information Technology},
  volume  = {104},
  number  = {7},
  pages   = {305--322},
  year    = {2026},
  issn    = {1992-8645}
}

**Paper:** STTN-CP: A Spatial-Temporal Transformer with Contrastive Pretraining Model for Credit Card Fraud Detection

- **Authors:** Kathiresan Jayabalan, Sethuraman Radhakrishnan
- **Journal:** *Journal of Theoretical and Applied Information Technology (JATIT)*, Vol. 104, No. 7, April 2026
- **Indexing:** [Scopus-indexed](https://www.scopus.com/sourceid/19700182903) | [SCImago Journal Rank (SJR)](https://www.scimagojr.com/journalsearch.php?q=19700182903&tip=sid)
- **DOI:** [10.5281/zenodo.19593993](https://doi.org/10.5281/zenodo.19593993)
