import torch
from torch.nn.utils.rnn import pad_sequence

from .config import CFG, DEVICE


@torch.no_grad()
def beam_search_batched(model, tok, items, beam=None, n_best=None, max_len=None,
                        len_alpha=None, batch_size=None):
    # items: list of (src_text, direction). Returns list of [top n_best strings].
    # Uses T5.generate beam search; num_return_sequences gives the top-k for acc@3.
    model.eval()
    beam = beam or CFG.beam; n_best = n_best or CFG.n_best
    max_len = max_len or CFG.gen_max_len
    len_alpha = CFG.len_alpha if len_alpha is None else len_alpha
    batch_size = batch_size or CFG.eval_batch_size
    n_ret = min(n_best, beam)
    out = [None] * len(items)
    for b0 in range(0, len(items), batch_size):
        chunk = items[b0:b0 + batch_size]; B = len(chunk)
        src = pad_sequence([torch.tensor(tok.encode_src(t, d)) for t, d in chunk],
                           batch_first=True, padding_value=tok.pad_id).to(DEVICE)
        attn = (src != tok.pad_id).long()
        gen = model.generate(
            input_ids=src, attention_mask=attn,
            num_beams=beam, num_return_sequences=n_ret,
            max_length=max_len, early_stopping=True, length_penalty=len_alpha,
            pad_token_id=tok.pad_id, eos_token_id=tok.eos_id,
            decoder_start_token_id=tok.pad_id)
        gen = gen.view(B, n_ret, -1).cpu().tolist()
        for i in range(B):
            seen = []
            for j in range(n_ret):
                txt = tok.decode(gen[i][j])
                if txt and txt not in seen:
                    seen.append(txt)
            out[b0 + i] = seen[:n_best]
    return out


def beam_search(model, tok, src_text, direction, beam=None, n_best=None, max_len=None, len_alpha=None):
    """Single-name convenience wrapper around beam_search_batched."""
    return beam_search_batched(model, tok, [(src_text, direction)], beam=beam, n_best=n_best,
                               max_len=max_len, len_alpha=len_alpha, batch_size=1)[0]
