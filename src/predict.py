import argparse

import pandas as pd

from .config import CFG, load_config
from .decoding import beam_search_batched
from .model import load_checkpoint


def predict(model, tok, names, direction="2en", top_k=3):
    """Top-k candidates for each input. direction: '2en' (Thai -> Latin) or '2th' (Latin -> Thai)."""
    preds = beam_search_batched(model, tok, [(n, direction) for n in names], n_best=top_k)
    rows = [[n] + (p + [""] * top_k)[:top_k] for n, p in zip(names, preds)]
    return pd.DataFrame(rows, columns=["input"] + [f"pred{i+1}" for i in range(top_k)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--direction", choices=["2en", "2th"], default="2en")
    ap.add_argument("--config", default=None, help="optional YAML (decoding settings default to the paper's)")
    ap.add_argument("--top-k", type=int, default=None, help="default: CFG.n_best (3)")
    ap.add_argument("--input", default=None)
    ap.add_argument("--column", default="name", help="column to read when --input is a .csv")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    load_config(args.config)

    names = list(args.names)
    if args.input:
        if args.input.endswith(".csv"):
            names += pd.read_csv(args.input)[args.column].astype(str).tolist()
        else:
            with open(args.input, encoding="utf-8") as f:
                names += [l.strip() for l in f if l.strip()]
    if not names:
        ap.error("give names as arguments or with --input")

    model, tok, _ = load_checkpoint(args.checkpoint)
    out = predict(model, tok, names, args.direction, args.top_k or CFG.n_best)
    if args.output:
        out.to_csv(args.output, index=False)
        print("saved:", args.output)
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
