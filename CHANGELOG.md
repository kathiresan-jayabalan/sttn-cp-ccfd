# Changelog

## 1.0.0 (Initial release)

- Added the STTN-CP architecture featuring stacked sequential Spatial-Temporal Transformer blocks with Spatial Transformer attention, Temporal Transformer attention, residual connections, and layer normalization.
- Added input embedding and positional encoding for transaction sequence representations.
- Added contrastive representation learning using an InfoNCE objective with temperature $\tau = 0.07$.
- Added a fully connected binary classification head with sigmoid activation for legitimate and fraudulent transaction classification.
- Added the combined training objective.
- Added support for the Kaggle Credit Card Fraud Detection dataset with 30 input features (`Time`, `V1`-`V28`, and `Amount`) and `Class` as the target label.
- Added Min-Max normalization and class-imbalance handling through the repository training workflow.
- Added configurable fixed-length transaction windows, with an 8-step sequence configuration used by the reference implementation.
- Added the repository training workflow with chronological 70/10/20 train, validation, and test partitions, training-only scaling, partition-local sequence construction, and training-partition `BorderlineSMOTE`.
- Added training and evaluation scripts for model fitting, checkpoint selection, fraud classification, and metric reporting.
- Added evaluation support for accuracy, precision, recall, F1-score, ROC-AUC, average precision, specificity, and confusion matrix reporting.
- Added the Jupyter notebook for an end-to-end implementation walkthrough.
- Added unit tests covering spatial feature-token handling, complete sequence inputs, sequence labels, partition boundaries, and InfoNCE input validation.
- Added GitHub Actions continuous integration with unit tests and an end-to-end training smoke test.
- Added experiment result artifacts and documentation to support research-oriented reproducibility and provenance.
