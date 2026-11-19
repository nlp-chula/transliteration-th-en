import torch

from .config import CFG
from .data import ROM_COLS
from .decoding import beam_search_batched


def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0: return max(m, n)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def cer(pred, refs):
    return min(levenshtein(pred, r) / max(len(r), 1) for r in refs)


def build_dev_items(dev_df):
    # TH->EN: one item per unique Thai name, refs = all its romanizations
    th2en = {}
    for _, r in dev_df.iterrows():
        refs = [r[c] for c in ROM_COLS if isinstance(r[c], str) and r[c]]
        th2en.setdefault(r["name"], set()).update(refs)
    th2en = [(name, list(refs)) for name, refs in th2en.items()]
    # EN->TH: one item per (romanize, name) pair, single reference
    en2th = []
    for _, r in dev_df.iterrows():
        for c in ROM_COLS:
            if isinstance(r[c], str) and r[c]:
                en2th.append((r[c], r["name"]))
    return th2en, en2th


@torch.no_grad()
def evaluate(model, tok, dev_df, max_items=None, batch_size=None):
    max_items = max_items if max_items is not None else CFG.eval_max_items
    th2en, en2th = build_dev_items(dev_df)
    if max_items:
        th2en, en2th = th2en[:max_items], en2th[:max_items]
    res = {}
    # TH -> EN (multi-reference) -- decode all names in batches
    preds = beam_search_batched(model, tok, [(name, "2en") for name, _ in th2en], batch_size=batch_size)
    c = a1 = a3 = 0
    for (name, refs), p in zip(th2en, preds):
        if not p: continue
        c  += cer(p[0], refs)
        a1 += p[0] in refs
        a3 += any(x in refs for x in p[:3])
    nt = len(th2en)
    res["TH2EN"] = {"CER": c/nt, "acc": a1/nt, "acc@3": a3/nt, "n": nt}
    # EN -> TH (exact)
    preds = beam_search_batched(model, tok, [(rom, "2th") for rom, _ in en2th], batch_size=batch_size)
    c = a1 = a3 = 0
    for (rom, name), p in zip(en2th, preds):
        if not p: continue
        c  += cer(p[0], [name])
        a1 += p[0] == name
        a3 += name in p[:3]
    ne = len(en2th)
    res["EN2TH"] = {"CER": c/ne, "acc": a1/ne, "acc@3": a3/ne, "n": ne}
    return res


def print_eval(res, tag=""):
    print(f"--- eval {tag} ---")
    for d, m in res.items():
        print(f"  {d}: CER={m['CER']:.4f}  acc={m['acc']:.4f}  acc@3={m['acc@3']:.4f}  (n={m['n']})")
