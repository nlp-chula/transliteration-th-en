import argparse, glob, os

import pandas as pd
import torch

from .config import CFG, load_config
from .metrics import evaluate
from .model import load_checkpoint


@torch.no_grad()
def reevaluate(paths=None, dev_csv=None, max_items=None, batch_size=None):
    """Re-evaluate saved checkpoints on a dev CSV. Each checkpoint is decoded with the
    tokenizer saved with it."""
    paths = paths or sorted(glob.glob(os.path.join(CFG.out_dir, "*.pt")))
    dev = pd.read_csv(dev_csv or os.path.join(CFG.out_dir, "dev_set.csv"))
    print(f"dev set: {len(dev)} rows / {dev.name.nunique()} names  ({len(paths)} checkpoints)\n")
    rows = []
    for p in paths:
        mdl, t, _ = load_checkpoint(p)
        r = evaluate(mdl, t, dev, max_items=max_items, batch_size=batch_size)
        a, b = r["TH2EN"], r["EN2TH"]
        print(f"{os.path.basename(p):42s} | TH→EN acc {a['acc']:.4f}  a@3 {a['acc@3']:.4f}  CER {a['CER']:.4f}"
              f"  ||  EN→TH acc {b['acc']:.4f}  a@3 {b['acc@3']:.4f}  CER {b['CER']:.4f}")
        rows.append({"file": os.path.basename(p),
                     "TH2EN_acc": a["acc"], "TH2EN_acc@3": a["acc@3"], "TH2EN_CER": a["CER"],
                     "EN2TH_acc": b["acc"], "EN2TH_acc@3": b["acc@3"], "EN2TH_CER": b["CER"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="optional YAML overrides")
    ap.add_argument("--checkpoints", nargs="*", default=None, help="default: every *.pt in out_dir")
    ap.add_argument("--dev-csv", default=None, help="default: <out_dir>/dev_set.csv")
    ap.add_argument("--out", default=None, help="default: <out_dir>/reeval_summary.csv")
    args = ap.parse_args()
    load_config(args.config)

    results = reevaluate(args.checkpoints, args.dev_csv)
    out = args.out or os.path.join(CFG.out_dir, "reeval_summary.csv")
    results.to_csv(out, index=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
