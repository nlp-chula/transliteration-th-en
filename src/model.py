import os
from dataclasses import asdict, fields

import torch
from transformers import T5Config, T5ForConditionalGeneration

from .config import CFG, DEVICE, Config
from .tokenizer import tokenizer_from_itos


def build_t5_config(cfg, tok):
    # char-level vocab; dims come from the config / size preset (configure_size)
    return T5Config(
        vocab_size=tok.vocab_size,
        d_model=cfg.d_model,
        d_kv=max(8, cfg.d_model // cfg.nhead),     # per-head dim
        d_ff=cfg.dim_feedforward,
        num_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.nhead,
        dropout_rate=cfg.dropout,
        feed_forward_proj="relu",                  # classic T5; switch to "gated-gelu" for T5 v1.1
        tie_word_embeddings=cfg.tie_embeddings,
        pad_token_id=tok.pad_id,
        eos_token_id=tok.eos_id,
        decoder_start_token_id=tok.pad_id,         # T5 convention: start decoding from pad
        is_encoder_decoder=True,
    )


def build_model(cfg, tok):
    model = T5ForConditionalGeneration(build_t5_config(cfg, tok)).to(DEVICE)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"T5 parameters: {n/1e6:.2f}M")
    return model


# size variants -> T5 dimensions. Apply before build_model.
# Paper names: tiny = "Very Small", small = "Small" (final model), medium = "Medium".
SIZES = {
    "tiny":  dict(d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512),
    "small": dict(d_model=256, nhead=4, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024),
    "medium": dict(d_model=384, nhead=6, num_encoder_layers=5, num_decoder_layers=5, dim_feedforward=1536),
    "base":  dict(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048),
}


def configure_size(name):
    for k, v in SIZES[name].items():
        setattr(CFG, k, v)
    print("size preset:", name, "->", SIZES[name])


def save_checkpoint(model, tok, name="biliteration_t5.pt"):
    """Saves weights + config + tokenizer vocab in one checkpoint under CFG.out_dir."""
    path = os.path.join(CFG.out_dir, name)
    torch.save({"state_dict": model.state_dict(),
                "config": asdict(CFG),
                "itos": tok.itos,
                "t5_config": model.config.to_dict()}, path)
    print("saved:", path)
    return path


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    known = {f.name for f in fields(Config)}           # ignore keys from older configs (e.g. test_csv)
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in known})
    t = tokenizer_from_itos(ckpt["itos"])
    m = T5ForConditionalGeneration(T5Config(**ckpt["t5_config"])).to(DEVICE)
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, t, cfg
