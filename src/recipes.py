import random

from .config import CFG
from .data import load_silver_examples
from .metrics import evaluate, print_eval
from .model import build_model, load_checkpoint, save_checkpoint
from .trainer import train_phase


# ---- recipe 1: gold only (baseline) ----
def run_gold_only(tok, train_examples_gold, dev_set):
    model = build_model(CFG, tok)
    train_phase(model, tok, train_examples_gold, CFG.gold_epochs, CFG.lr, dev_set, tag="gold-only")
    print_eval(evaluate(model, tok, dev_set), "gold-only")
    return model


# ---- recipe 2: silver pretrain -> gold finetune (curriculum) ----
def run_curriculum(tok, train_examples_gold, dev_set):
    model = build_model(CFG, tok)
    silver_ex = load_silver_examples(CFG.silver_trust_threshold, CFG.silver_max_rows)
    print('silver training examples', len(silver_ex))
    train_phase(model, tok, silver_ex, CFG.silver_epochs, CFG.lr, dev_set, tag="silver-pretrain")
    train_phase(model, tok, train_examples_gold, CFG.gold_epochs, CFG.finetune_lr, dev_set, tag="gold-finetune")
    print_eval(evaluate(model, tok, dev_set), "curriculum (silver->gold)")
    return model


def run_gold_into_silver(tok, train_examples_gold, dev_set):
    model = build_model(CFG, tok)
    silver_ex = load_silver_examples(CFG.silver_trust_threshold, CFG.silver_max_rows)
    print('silver training examples', len(silver_ex))
    train_phase(model, tok, train_examples_gold, CFG.gold_epochs, 5e-4, dev_set, tag="gold-train")
    train_phase(model, tok, silver_ex, CFG.silver_epochs, 3e-4, dev_set, tag="silver-finetune")
    print_eval(evaluate(model, tok, dev_set), "gold->silver")
    return model


# ---- recipe 3: gold + silver mixed in one pool (gold upsampled to stay influential) ----
def run_mixed(tok, train_examples_gold, dev_set, gold_upsample=7):
    model = build_model(CFG, tok)
    silver_ex = load_silver_examples(CFG.silver_trust_threshold, CFG.silver_max_rows)
    pool = train_examples_gold * gold_upsample + silver_ex
    random.shuffle(pool)
    print(f"mixed pool: {len(pool):,} (gold x{gold_upsample} + silver)")
    train_phase(model, tok, pool, CFG.gold_epochs, CFG.lr, dev_set, tag="mixed")
    print_eval(evaluate(model, tok, dev_set), "mixed (gold+silver)")
    return model


def run_mixed_with_gold(tok, train_examples_gold, dev_set, gold_upsample=5):
    model = build_model(CFG, tok)
    silver_ex = load_silver_examples(CFG.silver_trust_threshold, CFG.silver_max_rows)
    pool = train_examples_gold * gold_upsample + silver_ex
    random.shuffle(pool)
    print(f"mixed pool: {len(pool):,} (gold x{gold_upsample} + silver)")
    train_phase(model, tok, pool, 15, CFG.lr, dev_set, tag="mixed")
    train_phase(model, tok, train_examples_gold, 5, CFG.finetune_lr, dev_set, tag="gold-finetune")
    print_eval(evaluate(model, tok, dev_set), "mixed (gold+silver) with gold fine-tuning")
    return model


# ---- pretrain once, save, then reuse for every finetune trial ----
def pretrain(tok, train_examples_gold, dev_set, kind="mixed", gold_upsample=8, epochs=None, save_as=None):
    model = build_model(CFG, tok)
    silver_ex = load_silver_examples(CFG.silver_trust_threshold, CFG.silver_max_rows)
    print('silver training examples', len(silver_ex))
    epochs = epochs or CFG.silver_epochs
    if kind == "silver":
        train_phase(model, tok, silver_ex, epochs, CFG.lr, dev_set, tag="silver-pretrain")
    elif kind == "mixed":
        pool = train_examples_gold * gold_upsample + silver_ex
        random.shuffle(pool)
        print(f"mixed pool: {len(pool):,} (gold x{gold_upsample} + silver)")
        train_phase(model, tok, pool, epochs, CFG.lr, dev_set, tag="mixed-pretrain")
    else:
        raise ValueError("kind must be 'silver' or 'mixed'")
    print_eval(evaluate(model, tok, dev_set), f"{kind}-pretrain (no finetune)")
    if save_as: save_checkpoint(model, tok, save_as)   # <-- reuse this file for any finetune
    return model


def finetune_gold(pretrained, train_examples_gold, dev_set, tok=None, epochs=None, lr=None, save_as=None):
    """pretrained: a model (then pass `tok`) OR a path to a saved pretrain checkpoint
    (the tokenizer saved with it is used, so vocab always matches the weights)."""
    if isinstance(pretrained, str):
        model, tok, _ = load_checkpoint(pretrained)
    else:
        model = pretrained
    epochs = epochs or CFG.gold_epochs; lr = lr or CFG.finetune_lr
    train_phase(model, tok, train_examples_gold, epochs, lr, dev_set, tag="gold-finetune")
    print_eval(evaluate(model, tok, dev_set), "after gold finetune")
    if save_as: save_checkpoint(model, tok, save_as)
    return model
