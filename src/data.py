import functools
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .config import CFG

ROM_COLS = ["romanize1", "romanize2", "romanize3"]


def load_splits():
    """Read the train/dev split written by `python -m src.prepare_data`."""
    train_gold = pd.read_csv(os.path.join(CFG.out_dir, "train_gold.csv"))
    dev_set = pd.read_csv(os.path.join(CFG.out_dir, "dev_set.csv"))
    return train_gold, dev_set


def gold_to_examples(df):
    ex = []
    for _, r in df.iterrows():
        name = r["name"]
        for c in ROM_COLS:
            rom = r[c]
            if isinstance(rom, str) and rom:
                ex.append((name, "2en", rom))   # TH -> EN
                ex.append((rom, "2th", name))   # EN -> TH
    return ex


def silver_to_examples(df):
    ex = []
    for name, rom in zip(df["name"].values, df["romanize"].values):
        ex.append((name, "2en", rom))
        ex.append((rom, "2th", name))
    return ex


class BiliterationDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.ex = examples; self.tok = tokenizer
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        src_text, direction, tgt_text = self.ex[i]
        return (torch.tensor(self.tok.encode_src(src_text, direction), dtype=torch.long),
                torch.tensor(self.tok.encode_tgt(tgt_text), dtype=torch.long))


def _collate(batch, pad_id):
    srcs, tgts = zip(*batch)
    return (pad_sequence(srcs, batch_first=True, padding_value=pad_id),
            pad_sequence(tgts, batch_first=True, padding_value=pad_id))


def make_collate(pad_id):
    # module-level function + partial (not a closure) so DataLoader workers can pickle it
    # under the "spawn" start method (Windows/macOS)
    return functools.partial(_collate, pad_id=pad_id)


def load_silver_examples(threshold, max_rows, path=None):
    """The 'how much silver' knob: keep trust >= threshold, then if still over max_rows,
    keep the highest-trust rows.
    threshold: keep rows with trust >= threshold (use -1.0 for ALL silver)
    max_rows : cap; keeps the highest-trust rows (None = no cap)"""
    path = path or os.path.join(CFG.out_dir, "cleaned_silver.csv")
    df = pd.read_csv(path)
    df = df[df["trust score"] >= threshold]
    if max_rows is not None and len(df) > max_rows:
        df = df.sort_values("trust score", ascending=False).head(max_rows)
    print(f"silver @ trust>={threshold}, cap={max_rows}: {len(df):,} rows")
    return silver_to_examples(df)
