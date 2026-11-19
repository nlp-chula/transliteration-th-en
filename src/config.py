import json, random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class Config:
    # --- paths ---
    data_dir: str = "data"                    # folder containing the source CSVs (relative to where you run)
    out_dir: str  = "./outputs"
    train_csv: str  = "romanization-train.csv"
    silver_csv: str = "romanization-silver.csv"

    # --- data prep ---
    dev_frac: float = 0.10                     # fraction of unique gold names held out for dev
    seed: int = 42
    silver_min_char_freq: int = 5              # silver chars rarer than this -> <unk> (ROW IS STILL KEPT)

    # --- silver "amount" control ---
    silver_trust_threshold: float = 0.3        # keep rows with trust >= this; use -1.0 for ALL silver
    silver_max_rows: int = None                # hard cap (keeps highest-trust rows); set to None for no cap

    # --- model: "Small" (~7.4M params) ---
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 192                         # max sequence length check (covers full multi-word names, up to ~150 chars)
    tie_embeddings: bool = True

    # --- training ---
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1               # NOTE: not used — T5 computes its own (unsmoothed) loss
    grad_clip: float = 1.0
    silver_epochs: int = 10                    # phase-1 pretrain on silver / mixed pool (lr above)
    gold_epochs: int = 15                      # phase-2 finetune on gold (or gold-only baseline)
    finetune_lr: float = 3e-4                  # lr for the gold phase after pretrain
    # (the final model's gold fine-tuning grid lives in src/train.py: PIPELINE_DEFAULTS)
    amp: bool = True                           # mixed precision on CUDA
    live_plot: bool = True                     # live loss curve + progress bar with ETA (notebook only)
    plot_every: int = 50                       # redraw the loss curve every N steps
    dev_loss_every: int = 50                   # held-out dev loss every N steps (0=off)
    dev_loss_max_items: int = 400              # dev pairs used for that loss
    early_stop: bool = False                   # NOTE: not used by train_phase (no early stopping)
    early_stop_patience: int = 4
    early_stop_min_delta: float = 1e-4
    cer_eval_every: int = 0
    cer_eval_items: int = 200
    cer_eval_beam: int = 1
    # --- decoding / eval ---
    beam: int = 5
    n_best: int = 3
    len_alpha: float = 0.7                     # length penalty for beam ranking
    gen_max_len: int = 24                      # max output tokens (names are short); lower = faster
    eval_batch_size: int = 256                 # how many names decoded together (raise on big GPUs)
    eval_max_items = None                      # cap dev items PER DIRECTION (None = full dev)

    smoke_test: bool = False                   # tiny subset, 1-2 epochs, for a quick dry-run


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_smoke_test():
    # smoke-test overrides
    if CFG.smoke_test:
        CFG.silver_max_rows = 5_000
        CFG.silver_epochs = 1
        CFG.gold_epochs = 2
        CFG.eval_max_items = 50                # keep eval fast in smoke mode


def load_config(path=None):
    """Apply the `config:` section of a YAML file onto the global CFG.
    Returns the whole parsed YAML (so scripts can read their own sections, e.g. `pipeline:`)."""
    y = {}
    if path:
        with open(path, encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        for k, v in (y.get("config") or {}).items():
            if not hasattr(CFG, k):
                raise KeyError(f"unknown config key in {path}: {k}")
            if isinstance(getattr(CFG, k), float) and isinstance(v, (int, str)):
                v = float(v)                   # YAML reads '5e-4' as a string
            setattr(CFG, k, v)
    apply_smoke_test()
    return y


def setup():
    """Seed everything, create out_dir, print the config."""
    random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed)
    Path(CFG.out_dir).mkdir(parents=True, exist_ok=True)
    print("device:", DEVICE)
    print(json.dumps(asdict(CFG), indent=2)[:600], "...")
