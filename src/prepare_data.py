import argparse, os
from collections import Counter

import numpy as np
import pandas as pd

from .config import CFG, load_config, setup
from .data import ROM_COLS
from .tokenizer import save_charset


def char_counter(*iterables):
    c = Counter()
    for it in iterables:
        for x in it:
            if isinstance(x, str): c.update(x)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="optional YAML overrides")
    args = ap.parse_args()
    load_config(args.config); setup()

    # --- load gold; clean silver: drop ONLY NaNs and names already present in gold ---
    gold_raw = pd.read_csv(os.path.join(CFG.data_dir, CFG.train_csv))
    gold_name_set = set(gold_raw["name"])

    silver = pd.read_csv(os.path.join(CFG.data_dir, CFG.silver_csv))
    n0 = len(silver)
    silver = silver.dropna(subset=["name", "romanize"])            # drop NaNs
    silver = silver[~silver["name"].isin(gold_name_set)]           # drop names already in gold (reliable eval)
    silver = silver.reset_index(drop=True)
    silver.to_csv(os.path.join(CFG.out_dir, "cleaned_silver.csv"), index=False)
    print(f"silver: {n0:,} -> {len(silver):,} rows kept "
          f"({silver.name.nunique():,} unique names) | only NaNs + gold-name dups removed")
    print("trust range:", round(silver['trust score'].min(),3), "..", round(silver['trust score'].max(),3),
          "| longest name/romanize:", silver.name.str.len().max(), "/", silver.romanize.str.len().max())

    # --- split gold on UNIQUE names ---
    rng = np.random.default_rng(CFG.seed)
    unique_names = gold_raw["name"].drop_duplicates().tolist()
    rng.shuffle(unique_names)
    n_dev = max(1, int(len(unique_names) * CFG.dev_frac))
    dev_names = set(unique_names[:n_dev])
    train_names = set(unique_names[n_dev:])

    train_gold = gold_raw[gold_raw["name"].isin(train_names)].reset_index(drop=True)
    dev_set    = gold_raw[gold_raw["name"].isin(dev_names)].reset_index(drop=True)

    assert train_names.isdisjoint(dev_names), "name leakage between train and dev!"
    train_gold.to_csv(os.path.join(CFG.out_dir, "train_gold.csv"), index=False)
    dev_set.to_csv(os.path.join(CFG.out_dir, "dev_set.csv"), index=False)
    print(f"train_gold: {len(train_gold)} rows / {train_gold.name.nunique()} names")
    print(f"dev_set   : {len(dev_set)} rows / {dev_set.name.nunique()} names")

    # --- build char vocab: all gold chars + silver chars seen >= silver_min_char_freq times ---
    base_cnt   = char_counter(gold_raw["name"], *(gold_raw[c] for c in ROM_COLS))
    silver_cnt = char_counter(silver["name"], silver["romanize"])
    keep_silver = {ch for ch, n in silver_cnt.items() if n >= CFG.silver_min_char_freq}
    CHARSET = set(base_cnt) | keep_silver
    to_unk  = set(silver_cnt) - CHARSET
    print(f"vocab chars: {len(CHARSET)}  (gold base: {len(base_cnt)}, "
          f"+frequent silver: {len(keep_silver - set(base_cnt))})")
    print(f"ultra-rare silver chars -> <unk>: {len(to_unk)}  (their rows are still kept)")
    assert silver.romanize.str.len().max() + 2 <= CFG.max_len, "raise CFG.max_len for the longest sequence"
    print("longest silver seq:", silver.romanize.str.len().max(), "| max_len:", CFG.max_len)
    save_charset(CHARSET, os.path.join(CFG.out_dir, "vocab.json"))
    print("saved:", os.path.join(CFG.out_dir, "vocab.json"))


if __name__ == "__main__":
    main()
