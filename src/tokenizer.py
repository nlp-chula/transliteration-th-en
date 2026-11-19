"""Character-level tokenizer shared by Thai and Latin, with direction tokens <2en> / <2th>."""
import json


class CharTokenizer:
    def __init__(self, charset, directions=("2en", "2th")):
        self.PAD, self.BOS, self.EOS, self.UNK = "<pad>", "<bos>", "<eos>", "<unk>"
        specials = [self.PAD, self.BOS, self.EOS, self.UNK]
        self.dir_tokens = [f"<{d}>" for d in directions]
        self.itos = specials + self.dir_tokens + sorted(charset)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[self.PAD]; self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]; self.unk_id = self.stoi[self.UNK]

    @property
    def vocab_size(self): return len(self.itos)
    def dir_id(self, d): return self.stoi[f"<{d}>"]

    def encode_src(self, text, direction):
        return [self.dir_id(direction)] + [self.stoi.get(c, self.unk_id) for c in text]

    def encode_tgt(self, text):
        # T5 prepends decoder_start_token_id internally, so NO <bos> here; labels end with <eos>
        return [self.stoi.get(c, self.unk_id) for c in text] + [self.eos_id]

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos[i]
            if t in (self.PAD, self.BOS): continue
            if t == self.EOS: break
            out.append("?" if t == self.UNK else t)
        return "".join(out)


SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>", "<2en>", "<2th>")


def tokenizer_from_itos(itos):
    """Rebuild the tokenizer saved with a checkpoint (vocab must match the weights)."""
    t = CharTokenizer(set(c for c in itos if c not in SPECIALS))
    t.itos = itos; t.stoi = {s: i for i, s in enumerate(t.itos)}
    return t


def save_charset(charset, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(charset), f, ensure_ascii=False)


def load_tokenizer(path):
    with open(path, encoding="utf-8") as f:
        return CharTokenizer(set(json.load(f)))
