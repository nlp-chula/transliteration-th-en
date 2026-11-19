"""Run one of the comparison recipes (paper Experiments 1–3).

Usage:
  python -m src.run_experiment --recipe gold_only --size tiny
  python -m src.run_experiment --recipe curriculum --size tiny --threshold 0.927
  python -m src.run_experiment --recipe mixed_with_gold --size small --threshold 0.3 --save run.pt
"""
import argparse, os

from .config import CFG, load_config, setup
from .data import gold_to_examples, load_splits
from .model import SIZES, configure_size, save_checkpoint
from .recipes import (run_curriculum, run_gold_into_silver, run_gold_only,
                      run_mixed, run_mixed_with_gold)
from .tokenizer import load_tokenizer

RECIPES = {
    "gold_only": run_gold_only,
    "curriculum": run_curriculum,            # silver -> gold
    "mixed": run_mixed,                      # gold + silver
    "mixed_with_gold": run_mixed_with_gold,  # mixed -> gold
    "gold_into_silver": run_gold_into_silver,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", choices=RECIPES, required=True)
    ap.add_argument("--config", default=None, help="optional YAML overrides")
    ap.add_argument("--size", choices=SIZES, default=None, help="size preset (default: dims from config)")
    ap.add_argument("--threshold", type=float, default=None, help="silver trust threshold (-1.0 = all)")
    ap.add_argument("--max-rows", type=int, default=None, help="silver cap (keeps highest-trust rows)")
    ap.add_argument("--save", default=None, help="checkpoint file name to save in out_dir")
    args = ap.parse_args()
    load_config(args.config)
    if args.threshold is not None: CFG.silver_trust_threshold = args.threshold
    if args.max_rows is not None: CFG.silver_max_rows = args.max_rows
    if args.size: configure_size(args.size)
    setup()

    tok = load_tokenizer(os.path.join(CFG.out_dir, "vocab.json"))
    train_gold, dev_set = load_splits()
    train_examples_gold = gold_to_examples(train_gold)
    print("gold training examples (both directions):", len(train_examples_gold))
    model = RECIPES[args.recipe](tok, train_examples_gold, dev_set)
    if args.save: save_checkpoint(model, tok, args.save)


if __name__ == "__main__":
    main()
