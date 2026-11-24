# Approach I — Recurrent Summarization Models

One PyTorch package containing **both** recurrent models from the paper. They share the
data pipeline (`src/data/`), vocabulary (`artifacts/vocab.json`), checkpoint/ROUGE utilities
(`src/utils/`) and evaluation protocol.

| Model | Model code | Config | Train | Evaluate |
|---|---|---|---|---|
| Pointer-Generator + Coverage (LSTM decoder) | `src/model/pgn.py` | `configs/base.yaml` | `python -m scripts.train` | `python -m scripts.eval` |
| Hierarchical Attention (BiGRU, sentence + word attention, LVT) | `src/model/hieratt.py` | `configs/hieratt.yaml` | `python -m scripts.train_hieratt` | `python -m scripts.eval_hieratt` |

## Setup

```bash
cd models/recurrent          # every command below runs from here
pip install -r requirements.txt
# optional: better sentence splitting for HierAtt (falls back to a regex splitter if absent)
pip install pysbd
```

Python 3.10+. CNN/DailyMail 3.0.0 ([`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail))
is downloaded automatically by 🤗 Datasets.

> Run the scripts as modules (`python -m scripts.<name>`). They import from `src.…`, so running them
> as file paths (`python scripts/train.py`) fails with `ModuleNotFoundError: No module named 'src'`.

## Vocabulary

Both models load the committed `artifacts/vocab.json` (50k tokens, `min_freq: 2`, lowercased, built from
the training split), so no build step is needed. Rebuilding is optional and **overwrites the committed file**:

```bash
python -m scripts.build_vocab                          # uses configs/base.yaml
python -m scripts.build_vocab --config configs/hieratt.yaml
```

## Model A — Pointer-Generator + Coverage

```bash
python -m scripts.train      # reads configs/base.yaml; best model -> checkpoints/pgn_best.pt
python -m scripts.eval       # loads checkpoints/pgn_best.pt; reports ROUGE-1/2/L
```

`scripts.eval` scores the validation split with beam search. Decoding is controlled from `configs/base.yaml`
(`beam_size`, `len_penalty`, `max_dec_len`, `min_dec_len`); `beam_size: 1` is effectively greedy decoding.
Scores below are ROUGE F1 (`rouge_score`, with stemming) from Table 1 of the paper.

| Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|:---:|:---:|:---:|
| Greedy | 0.3668 | 0.1486 | 0.2475 |
| Beam search (beam 4) | **0.3786** | **0.1629** | **0.2584** |

## Model B — Hierarchical Attention

Word-level encoding + sentence-level encoding, with **sentence attention re-scaling word attention**, and
a Local Vocabulary Table (`lvt_k`) to keep the decoder softmax small.

```bash
python -m scripts.train_hieratt   # reads configs/hieratt.yaml
python -m scripts.eval_hieratt    # loads checkpoints/hieratt_best.pt
```

These scripts take no CLI arguments: the config and checkpoint paths are fixed in code
(`configs/hieratt.yaml`, `checkpoints/hieratt_best.pt`). Training writes `hieratt_latest.pt`,
`hieratt_best.pt` and periodic `hieratt_step*.pt` to `checkpoints/`, and logs a progress line every
fixed number of batches plus periodic GPU memory (Kaggle-friendly).

`configs/hieratt.yaml` controls model dimensions, hierarchical limits (`max_src_sents`, `max_src_len`),
the LVT size, training schedule and decoding limits.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|:---:|:---:|:---:|
| Hierarchical Attention BiGRU | 0.2934 | 0.1087 | 0.2037 |

(Scores from Table 1 of [the paper](../../paper/paper.pdf).)
