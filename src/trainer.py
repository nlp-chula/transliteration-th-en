"""Training loop. Shows a progress bar with an ETA for the whole phase (tqdm) and a
real-time loss curve (EMA-smoothed, redrawn every CFG.plot_every steps) in notebooks.
Both degrade to plain text outside a notebook."""
import random, time

import torch
from torch.utils.data import DataLoader

from .config import CFG, DEVICE
from .data import BiliterationDataset, gold_to_examples, make_collate

try:
    from torch.amp import autocast, GradScaler
    def make_autocast(enabled): return autocast("cuda", enabled=enabled)
    def make_scaler(enabled):  return GradScaler("cuda", enabled=enabled)
except Exception:  # older torch
    from torch.cuda.amp import autocast as _ac, GradScaler as _gs
    def make_autocast(enabled): return _ac(enabled=enabled)
    def make_scaler(enabled):  return _gs(enabled=enabled)

# progress bar (with live ETA) + live loss plot, both degrade gracefully outside a notebook
try:
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    from IPython.display import display
    _HAS_VIZ = True
except Exception:
    _HAS_VIZ = False


def train_phase(model, tok, examples, epochs, lr, dev_set, tag=""):
    """Train `model` on `examples` for `epochs` at `lr`. `dev_set` (gold dev DataFrame)
    is used only for the held-out dev-loss curve."""
    use_amp = CFG.amp and DEVICE == "cuda"
    collate_fn = make_collate(tok.pad_id)
    ds = BiliterationDataset(examples, tok)
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=2, pin_memory=(DEVICE=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
    # T5ForConditionalGeneration computes the (label-smoothing-free) cross-entropy internally
    scaler = make_scaler(use_amp)
    # --- fixed held-out dev set for a generalization-loss curve ---
    dev_batches = []
    if CFG.dev_loss_every:
        dev_ex = gold_to_examples(dev_set)
        random.Random(0).shuffle(dev_ex)
        dev_ex = dev_ex[:CFG.dev_loss_max_items]
        dl_loader = DataLoader(BiliterationDataset(dev_ex, tok), batch_size=CFG.batch_size, collate_fn=collate_fn)
        dev_batches = [(s.to(DEVICE), t.to(DEVICE)) for s, t in dl_loader]
    dev_steps, dev_hist = [], []
    def _dev_loss():
        model.eval(); tot = 0.0; ntok = 0
        with torch.no_grad():
            for src, tgt in dev_batches:
                attn = (src != tok.pad_id).long()
                labels = tgt.clone(); labels[tgt == tok.pad_id] = -100
                l = model(input_ids=src, attention_mask=attn, labels=labels).loss
                n = (labels != -100).sum().item(); tot += l.item() * n; ntok += n
        model.train(); return tot / max(ntok, 1)

    total_steps = epochs * len(loader)
    bar = tqdm(total=total_steps, desc=f"train[{tag}]", unit="step") if _HAS_VIZ else None  # ETA lives here
    live = _HAS_VIZ and CFG.live_plot
    steps_hist, loss_hist, ema, step = [], [], None, 0
    if live:
        try:
            fig, ax = plt.subplots(figsize=(7, 3)); fig.tight_layout()
            dh = display(fig, display_id=True)
        except Exception:
            live = False
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; ntok = 0
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            attn = (src != tok.pad_id).long()
            labels = tgt.clone(); labels[tgt == tok.pad_id] = -100   # T5 ignores -100
            opt.zero_grad(set_to_none=True)
            with make_autocast(use_amp):
                loss = model(input_ids=src, attention_mask=attn, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(opt); scaler.update()
            l = loss.item(); n = (labels != -100).sum().item()
            tot += l * n; ntok += n; step += 1
            ema = l if ema is None else 0.98 * ema + 0.02 * l
            if bar is not None:
                bar.update(1); bar.set_postfix(epoch=f"{ep}/{epochs}", loss=f"{ema:.3f}")
            if CFG.dev_loss_every and step % CFG.dev_loss_every == 0 and dev_batches:
                d = _dev_loss(); dev_steps.append(step); dev_hist.append(d)
                if bar is not None:
                    bar.set_postfix(epoch=f"{ep}/{epochs}", loss=f"{ema:.3f}", dev=f"{d:.3f}")
            if live and step % CFG.plot_every == 0:
                steps_hist.append(step); loss_hist.append(ema)
                try:
                    ax.clear()
                    ax.plot(steps_hist, loss_hist, label="train (EMA)")
                    if dev_hist: ax.plot(dev_steps, dev_hist, marker="o", markersize=2, label="dev")
                    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(loc="upper right")
                    ax.set_title(f"{tag} — epoch {ep}/{epochs}  |  step {step}/{total_steps}")
                    dh.update(fig)
                except Exception:
                    live = False
        if bar is None:  # plain-text fallback
            print(f"[{tag}] epoch {ep}/{epochs}  loss={tot/max(ntok,1):.4f}  ({time.time()-t0:.1f}s)")
    elapsed = time.time() - t0
    if bar is not None: bar.close()
    if live:
        try:
            steps_hist.append(step); loss_hist.append(ema)
            ax.clear()
            ax.plot(steps_hist, loss_hist, label="train (EMA)")
            if dev_hist: ax.plot(dev_steps, dev_hist, marker="o", label="dev")
            ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(loc="upper right")
            ax.set_title(f"{tag} — done: {epochs} epochs in {elapsed:.0f}s")
            dh.update(fig); plt.close(fig)
        except Exception:
            pass
    print(f"[{tag}] done: {epochs} epochs, final EMA loss {ema:.4f}, {elapsed:.1f}s")
    return model
