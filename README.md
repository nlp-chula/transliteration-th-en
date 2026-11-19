# Thai–English Proper Name Transliteration

Code for the paper **"Thai Proper Name Transliteration with Encoder-Decoder Models Augmented with Phonetically-Aligned Data"**, published at **iSAI-NLP 2026**.

## Pretrained model

The best-performing model is available on Hugging Face:

**https://huggingface.co/Sira-D/thai-name-translit-t5**

One character-level `T5ForConditionalGeneration`, trained from scratch, does both directions. A direction token goes in front of the source:

| Direction | Source                | Target          | Relationship |
|-----------|-----------------------|-----------------|--------------|
| TH→EN     | `<2en>` + Thai name   | Latin spelling  | one-to-many  |
| EN→TH     | `<2th>` + Latin name  | Thai name       | many-to-one  |

## Results (development set)

| Model | TH→EN CER | TH→EN acc | TH→EN acc@3 | EN→TH CER | EN→TH acc | EN→TH acc@3 |
|-------|-----------|-----------|-------------|-----------|-----------|-------------|
| Small (final), mixed→gold, trust ≥ 0.3 | 0.0121 | 0.9350 | 0.9876 | 0.1463 | 0.5649 | 0.8086 |

These are dev-set numbers. The dev set was also used to pick the fine-tuning configuration (see the paper, §IV-A).

## Repository layout

```
├── configs/final.yaml       # final settings for reference (already the code defaults)
├── scripts/reproduce_final.sh
└── src/
    ├── config.py            # Config dataclass, device, seeding, YAML overrides
    ├── tokenizer.py         # shared Thai+Latin char tokenizer
    ├── data.py              # examples (both directions), dataset, silver loader
    ├── model.py             # T5 config/model, size presets, checkpoint save/load
    ├── trainer.py           # training loop (AdamW, AMP, grad clipping, dev-loss curve)
    ├── decoding.py          # batched beam search
    ├── metrics.py           # CER, accuracy, accuracy@3
    ├── recipes.py           # gold-only / silver→gold / mixed / mixed→gold / gold→silver
    ├── prepare_data.py      # step 1: clean silver, split gold 9:1 by name, build vocab
    ├── train.py             # step 2: mixed pretrain + gold fine-tuning grid
    ├── evaluate.py          # step 3: re-evaluate checkpoints on dev
    ├── predict.py           # inference CLI
    └── run_experiment.py    # run a single comparison recipe
```

## Setup

```bash
pip install -r requirements.txt
```

Put `romanization-train.csv`  and `romanization-silver.csv` (unzipped)  in `data/`. 

## Reproduce the final model

```bash
bash scripts/reproduce_final.sh
```

or step by step:

```bash
python -m src.prepare_data   # -> outputs/{cleaned_silver,train_gold,dev_set}.csv, vocab.json
python -m src.train          # -> pretrain + 6 fine-tuned checkpoints, outputs/finetune_grid.csv
python -m src.evaluate       # re-evaluate every *.pt in outputs/
```

No config file is needed: the defaults in `src/config.py` and `src/train.py` are the final model's settings (below).
To try other settings, use command-line flags (`python -m src.train --help`) or copy `configs/final.yaml`,
edit it, and pass it with `--config`.

| Setting | Value |
|---|---|
| Model | T5 (ReLU FFN), d_model 256, 4 heads, 4 encoder + 4 decoder layers, d_ff 1024, dropout 0.1 |
| Data | Gold split 9:1 by unique Thai name (seed 42); silver with trust ≥ 0.3, no row cap |
| Pretraining | Mixed pool (silver + gold ×8), lr 5e-4, 10 epochs |
| Gold fine-tuning | Grid: lr ∈ {1e-4, 3e-4} × epochs ∈ {5, 10, 15}, each from the same pretrained checkpoint |
| Optimisation | AdamW, batch 256, weight decay 0.01, gradient clipping 1.0, AMP on CUDA, seed 42 |
| Decoding | Beam search, 5 beams, top-3 returned, length penalty 0.7, max 24 tokens |


## Inference

```bash
python -m src.predict --checkpoint outputs/<checkpoint>.pt ศิริกานต์ วงศ์เดือน
python -m src.predict --checkpoint outputs/<checkpoint>.pt --direction 2th pachara sirikan
python -m src.predict --checkpoint outputs/<checkpoint>.pt --input names.txt --output preds.csv
```

```python
from src.model import load_checkpoint
from src.predict import predict

model, tok, _ = load_checkpoint("outputs/<checkpoint>.pt")
print(predict(model, tok, ["ศิริกานต์"], direction="2en"))
```

## Other experiments in the paper

```bash
python -m src.run_experiment --recipe gold_only       --size tiny
python -m src.run_experiment --recipe curriculum      --size tiny  --threshold 0.927   # silver -> gold
python -m src.run_experiment --recipe mixed           --size tiny  --threshold 0.927
python -m src.run_experiment --recipe mixed_with_gold --size tiny  --threshold 0.927   # mixed -> gold
python -m src.run_experiment --recipe gold_into_silver --size tiny --threshold 0.927
```

Size presets: `tiny` = "Very Small", `small` = "Small", `medium` = "Medium" in the paper.

## License

The model is released under **CC-BY-4.0**.

## Citation

```bibtex
@inproceedings{daorote2026thai,
  title     = {Thai Proper Name Transliteration with Encoder-Decoder Models Augmented with Phonetically-Aligned Data},
  author    = {Daorote, Sira and Jiamsundutsadee, Ratanon and Wongwarawipatr, Weerin and Lauc, Davor and Rutherford, Attapol},
  booktitle = {2026 International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP)},
  year      = {2026},
  publisher = {IEEE}
}
```
