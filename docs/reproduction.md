# Reproduction

## 1. Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Dataset

Download the ULB credit-card fraud dataset from Kaggle and save the file as:

```text
data/creditcard.csv
```

## 3. Tests

```bash
python -m pytest -q
```

## 4. Training

```bash
python -m src.train \
  --data-path data/creditcard.csv \
  --output-dir outputs
```

The checkpoint and training history are written under `outputs/`.

## 5. Evaluation

```bash
python -m src.evaluate \
  --data-path data/creditcard.csv \
  --checkpoint outputs/best_model.pt \
  --output outputs/test_metrics.json
```

## 6. Notebook

Open:

```text
notebooks/STTN_CP_v2.ipynb
```

The notebook uses the same `src/` implementation as the command-line training path. When `data/creditcard.csv` is not available, its smoke-test section runs on a small deterministic synthetic dataset and clearly labels those numbers as a software check rather than a paper result.
