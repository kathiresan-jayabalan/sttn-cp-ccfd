# Paper Alignment Notes

| Area | Published paper | Repository implementation |
|---|---|---|
| Model flow | Spatial Transformer -> residual -> Temporal Transformer -> residual | Same |
| Spatial axis | Transaction feature correlations | Original `D` transaction features are tokenized explicitly |
| Temporal axis | Consecutive transaction sequence | Full `T=8` windows are preserved through classification |
| Contrastive temperature | 0.07 | 0.07 |
| Loss | `L_classification + lambda L_contrastive` | Same ordering |
| Epochs | 50 with early stopping | 50 with validation-F1 early stopping |
| Batch size | 128 | 128 |
| Optimizer | Adam, lr 0.001 | Adam, lr 0.001 |
| Paper split | 80/20 | Default repository protocol: chronological 70/10/20 |
| Paper preprocessing | Min-Max + SMOTE | Default repository protocol: train-only Min-Max + class-weighted loss |

## Why the repository differs on the split and balancing protocol

The paper gives the 80/20 split and Min-Max + SMOTE in its experimental setup, but the published description does not explicitly establish whether SMOTE was applied before or after the hold-out split. For a sequence model, creating synthetic transaction rows before window construction can also manufacture temporal neighbors that never existed in the source data.

The open-source repository therefore makes the data boundary explicit and keeps synthetic row generation out of the default temporal pipeline. That choice is a reproducibility safeguard, not a claim that the published paper used this exact protocol.
