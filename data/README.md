# Data

The experiment uses the ULB / Machine Learning Group credit card fraud dataset available through Kaggle.

Download `creditcard.csv` and place it at:

```text
data/creditcard.csv
```

The repository does not redistribute the dataset.

The training pipeline sorts by `Time`, creates chronological 70/10/20 partitions, fits Min-Max scaling on the training partition, and creates transaction windows independently within each partition.
