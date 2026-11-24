# Comparing Summarization Models on CNN/DailyMail

Code and report for **"A Comparative Study of Four Summarization Models on CNN/DailyMail"**
(Ahmed Emad, Fatema Kotb, Amr Elsheshtawy, Salma Hamed — Computer Engineering, Cairo University).

The full report is in [`paper/paper.pdf`](paper/paper.pdf).

## Repository layout

The `models/` folder mirrors the paper's two approaches:

```
.
├── paper/
│   └── paper.pdf                       # the report
└── models/
    ├── recurrent/                      # Approach I  — one package, two RNN models
    │   ├── configs/                    #   base.yaml (PGN), hieratt.yaml (HierAtt)
    │   ├── artifacts/vocab.json        #   shared 50k vocabulary
    │   ├── scripts/                    #   train / eval / build_vocab entrypoints
    │   └── src/                        #   data, model (pgn.py, hieratt.py), utils
    └── transformer/                    # Approach II
        ├── from_scratch/               #   vanilla encoder–decoder transformer (Kaggle notebook)
        └── pretrained_seq2seq/         #   one HF package for mT5+LoRA and Pegasus, config-driven
            ├── configs/                #     Mt5.json, Pegasus.json
            └── outputs/                #     committed ROUGE results
```

| Paper | Model | Code | How it is selected |
|---|---|---|---|
| Approach I | Pointer-Generator + Coverage (LSTM) | [`models/recurrent`](models/recurrent) | `scripts.train` / `configs/base.yaml` |
| Approach I | Hierarchical Attention (BiGRU + LVT) | [`models/recurrent`](models/recurrent) | `scripts.train_hieratt` / `configs/hieratt.yaml` |
| Approach II | Transformer trained from scratch | [`models/transformer/from_scratch`](models/transformer/from_scratch) | notebook |
| Approach II | mT5-base + LoRA (fine-tuned) | [`models/transformer/pretrained_seq2seq`](models/transformer/pretrained_seq2seq) | `--config configs/Mt5.json` |
| Approach II | Pegasus (`google/pegasus-cnn_dailymail`, zero-shot) | [`models/transformer/pretrained_seq2seq`](models/transformer/pretrained_seq2seq) | `--config configs/Pegasus.json` |

Each model folder has its own `README.md` and `requirements.txt` (the packages have very different
dependency footprints, so they are kept separate). **All commands are run from inside that model's folder**
— configs, checkpoints and outputs use paths relative to it.

## Results (paper, Table 1)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| PGN + Coverage, greedy | 0.3668 | 0.1486 | 0.2475 |
| PGN + Coverage, beam-4 | 0.3786 | 0.1629 | 0.2584 |
| Hierarchical Attention BiGRU | 0.2934 | 0.1087 | 0.2037 |
| mT5 + LoRA, beam-4 | 0.3947 | 0.2093 | 0.3118 |
| Pegasus zero-shot, beam-4 | 0.3537 | 0.1442 | 0.2613 |
| Transformer from scratch | 0.1983 | 0.0307 | 0.1859 |

Raw eval files: [`mt5_eval_111.json`](models/transformer/pretrained_seq2seq/outputs/mt5_eval_111.json),
[`pegasus_eval_111.json`](models/transformer/pretrained_seq2seq/outputs/pegasus_eval_111.json).

> **Note:** the committed `mt5_eval_111.json` (0.2947 / 0.1093 / 0.2118) is exactly 0.1 below the mT5
> numbers reported in the paper and the original mT5 README on every metric. Both are kept unchanged here;
> see the [pretrained_seq2seq README](models/transformer/pretrained_seq2seq/README.md#results).

## Dataset

CNN/DailyMail 3.0.0 ([`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail)).
The recurrent models download it automatically via 🤗 Datasets; the pretrained transformer package reads
local parquet files (see its README). Datasets, checkpoints and run logs are git-ignored.
