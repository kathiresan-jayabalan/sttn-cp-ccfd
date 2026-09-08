# Methodology

## Data protocol

The repository works with the public ULB credit-card fraud dataset. Records are ordered by `Time` before partitioning. The default open-source protocol uses 70% of the ordered data for training, 10% for validation, and 20% for testing. Min-Max scaling is fitted on the training partition and then applied unchanged to validation and test data.

Transaction windows contain eight consecutive transactions. The label belongs to the final transaction in each window. Windows are built separately inside each partition, so a test window cannot reach back into training data.

The default implementation does not apply SMOTE to individual transaction rows before window construction. The supervised classifier uses class-weighted cross-entropy instead. This avoids creating synthetic rows that can be interpreted as neighboring transactions in a temporal sequence.

## STTN-CP architecture

The model starts with a linear input projection from the original feature dimension to the Transformer embedding dimension. Each ST block has two stages.

1. The spatial Transformer receives the original transaction features `(B, T, D)`. At each timestep, the `D` scalar feature values become feature tokens. Self-attention is therefore performed across transaction attributes.
2. The spatial output is added to the latent representation through a residual connection.
3. The temporal Transformer receives the resulting `(B, T, E)` representation and attends across the `T` transaction positions.
4. A second residual connection produces the block output.

The same original window is supplied to the spatial component of every stacked block, while the latent representation is passed from one block to the next. This keeps the spatial pathway tied to the transaction feature axis rather than accidentally turning embedding coordinates into pseudo-features.

After the stacked blocks, mean pooling over time produces a window representation. The representation feeds both the contrastive projection head and the binary classifier.

## Contrastive objective

Two lightly perturbed views of the same window are encoded. InfoNCE uses cosine-normalized representations with temperature `0.07`.

The implementation follows the published Equation (10) ordering:

```text
L_total = L_classification + lambda * L_contrastive
```

with `lambda = 0.5` by default.

## Training

The default training configuration is 50 epochs, batch size 128, Adam with learning rate `0.001`, cosine learning-rate scheduling, and early stopping on validation F1.

The repository intentionally keeps the paper's reported experimental settings visible while using an explicitly defined, leakage-safe open-source evaluation protocol. The paper describes an 80/20 split and Min-Max normalization plus SMOTE, but does not fully specify the ordering of those operations. For that reason, the repository does not claim that its default protocol reproduces the paper's Table 4 values.

## Evaluation

The test report contains:

- accuracy
- precision
- recall
- F1 score
- average precision
- ROC-AUC
- specificity
- confusion matrix

Average precision is included because the fraud class is highly imbalanced.
