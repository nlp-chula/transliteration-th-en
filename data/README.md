# Data

Place the source files in this folder (`data_dir` in `src/config.py`; scripts are run from the repo root):

| File | Columns | Description |
|---|---|---|
| `romanization-train.csv` | `name`, `romanize1`, `romanize2`, `romanize3` | Gold standard: 3,269 Thai names, up to three natural Latin spellings each from a native Thai annotator |
| `romanization-silver.csv` | `name`, `romanize`, `trust score` | Silver standard: noisy Thai–Latin pairs (scraped + engine-generated), each with a random-forest trust score |

`python -m src.prepare_data` writes these to `outputs/`:

| File |  size |
|---|---|
| `train_gold.csv` | 2,944 rows / 2,914 unique names |
| `dev_set.csv` | 325 rows / 323 unique names |
| `cleaned_silver.csv` | 3,437,253 rows / 2,846,050 unique names (NaNs and gold names removed) |
| `vocab.json` | character set (gold chars + silver chars seen ≥ 5 times) |

## Splits

The split is deterministic (unique Thai names shuffled with `numpy.random.default_rng(42)`, 10% to dev). To publish the exact split, copy the files here:

```bash
cp outputs/train_gold.csv outputs/dev_set.csv data/splits/
```

To train from committed splits instead of re-splitting, copy them back into `outputs/` and skip the split step.
