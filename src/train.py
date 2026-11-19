"""Step 2 · Train the final model (paper §IV-B): mixed -> gold.

1. Pretrain once on the mixed pool (silver at trust >= threshold + gold upsampled),
   and save the checkpoint.
2. Fine-tune on gold from that same checkpoint for every (lr, epochs) in the grid,
   saving each run.
3. Re-evaluate every fine-tuned checkpoint on dev -> <out_dir>/finetune_grid.csv.
   The paper selects the configuration with the highest dev exact-match accuracy.

Runs with the final model's settings by default — no config file needed.

Usage:
  python -m src.train
  python -m src.train --skip-pretrain                       # reuse the saved pretrain checkpoint
  python -m src.train --finetune-lrs 3e-4 --finetune-epochs 10
  python -m src.train --config my_overrides.yaml            # optional YAML overrides
"""
import argparse, os

from .config import CFG, load_config, setup
from .data import gold_to_examples, load_splits
from .evaluate import reevaluate
from .recipes import finetune_gold, pretrain
from .tokenizer import load_tokenizer

# Final-model pipeline. Override with a YAML `pipeline:` section or the command-line flags.
PIPELINE_DEFAULTS = {
    "pretrain_kind": "mixed",          # mixed pool = silver + gold x gold_upsample
    "gold_upsample": 8,
    "pretrain_checkpoint": None,       # None -> "pretrain_<kind>_<threshold>.pt"
    "finetune_lrs": [1e-4, 3e-4],      # gold fine-tuning grid
    "finetune_epochs": [5, 10, 15],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="optional YAML with `config:` / `pipeline:` overrides")
    ap.add_argument("--skip-pretrain", action="store_true",
                    help="reuse <out_dir>/<pretrain_checkpoint> instead of pretraining again")
    ap.add_argument("--pretrain-kind", choices=["mixed", "silver"], default=None)
    ap.add_argument("--gold-upsample", type=int, default=None)
    ap.add_argument("--pretrain-checkpoint", default=None, help="file name in out_dir")
    ap.add_argument("--finetune-lrs", type=float, nargs="+", default=None)
    ap.add_argument("--finetune-epochs", type=int, nargs="+", default=None)
    args = ap.parse_args()
    y = load_config(args.config); setup()

    # precedence: command line > YAML `pipeline:` > PIPELINE_DEFAULTS
    p = {**PIPELINE_DEFAULTS, **(y.get("pipeline") or {})}
    for k in PIPELINE_DEFAULTS:
        v = getattr(args, k)
        if v is not None: p[k] = v
    if not p["pretrain_checkpoint"]:
        p["pretrain_checkpoint"] = f"pretrain_{p['pretrain_kind']}_{CFG.silver_trust_threshold}.pt"
    print("pipeline:", p)

    tok = load_tokenizer(os.path.join(CFG.out_dir, "vocab.json"))
    print("vocab size:", tok.vocab_size, "| sample:", tok.itos[:8])
    train_gold, dev_set = load_splits()
    train_examples_gold = gold_to_examples(train_gold)
    print("gold training examples (both directions):", len(train_examples_gold))

    # 1) pretrain — the expensive part, done once
    pre_name = p["pretrain_checkpoint"]
    if not args.skip_pretrain:
        pretrain(tok, train_examples_gold, dev_set, kind=p["pretrain_kind"],
                 gold_upsample=p["gold_upsample"], save_as=pre_name)
    pre_path = os.path.join(CFG.out_dir, pre_name)

    # 2) gold fine-tuning grid, every trial starts from the same pretrained weights
    stem = os.path.splitext(pre_name)[0]
    ft_paths = []
    for lr in p["finetune_lrs"]:
        lr = float(lr)
        for ep in p["finetune_epochs"]:
            name = f"{stem}_ft_{ep}ep_lr{lr:.0e}.pt"
            finetune_gold(pre_path, train_examples_gold, dev_set, epochs=ep, lr=lr, save_as=name)
            ft_paths.append(os.path.join(CFG.out_dir, name))

    # 3) summary table of the grid on dev
    results = reevaluate(ft_paths)
    out = os.path.join(CFG.out_dir, "finetune_grid.csv")
    results.to_csv(out, index=False)
    print("\n" + results.to_string(index=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
