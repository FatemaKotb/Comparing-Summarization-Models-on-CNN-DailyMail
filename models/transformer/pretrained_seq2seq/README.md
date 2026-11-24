# Approach II — Pretrained Seq2Seq Transformers (mT5 + LoRA, Pegasus)

One Hugging Face package serving **two** models from the paper. The model is chosen entirely by the
config passed via `--config`; `models/model_factory.py` maps the config's `model_name` to a wrapper class.

| Model | Config | Wrapper | Checkpoint |
|---|---|---|---|
| mT5-base + LoRA (fine-tuned) | `configs/Mt5.json` | `models/mt5.py` | `google/mt5-base` |
| Pegasus, off the shelf (no further training) | `configs/Pegasus.json` | `models/pegasus.py` | `google/pegasus-cnn_dailymail` |

`google/pegasus-cnn_dailymail` has already been fine-tuned by Google on CNN/DailyMail. The paper calls this
setting "zero-shot", but it is really an in-domain baseline that this project uses without further training.

```
pretrained_seq2seq/
├── configs/          Mt5.json, Pegasus.json
├── models/           base_model.py, mt5.py, pegasus.py, model_factory.py
├── training/         train.py (LoRA fine-tuning), dailymail.py, collator.py
├── inference/        generate_summaries.py, test_model.py
├── evaluation/       run_evaluation.py, rouge_eval.py, dataset_reader.py, utils.py
├── preprocessing/    clean_text.py, tokenizer_utils.py
└── outputs/          *_eval_111.json (committed), sample_summaries.md
```

## Setup

```bash
cd models/transformer/pretrained_seq2seq    # every command below runs from here
pip install -r requirements.txt
```

Developed with Python 3.11.9. The pinned `torch` in `requirements.txt` is the CPU build; for CUDA,
install torch separately following https://pytorch.org/get-started/locally/.

**Data.** The code reads the CNN/DailyMail 3.0.0 parquet files (`train`, `validation`, `test` splits)
from a local directory, the `dataset_dir` in both configs (git-ignored):

```
data/raw_data/DailyMail/3.0.0/
```

## Usage

```bash
# Fine-tune mT5 with LoRA (Pegasus is used as-is and is not trained)
python -m training.train --config configs/Mt5.json

# Generate summaries for the first 100 test articles -> the config's output_path
python -m inference.generate_summaries --config configs/Mt5.json
python -m inference.generate_summaries --config configs/Pegasus.json

# Score them with ROUGE -> outputs/<model_name>_eval_111.json
python -m evaluation.run_evaluation --config configs/Mt5.json
python -m evaluation.run_evaluation --config configs/Pegasus.json

# Smoke test on a built-in sample article (defaults to configs/Mt5.json)
python -m inference.test_model --config configs/Pegasus.json
```

`configs/Mt5.json` sets `pretrained_lora_weights` to `checkpoints/mt5_dailymail_2/epoch_1`, which is
**not in the repo** (checkpoints are git-ignored), so `train`, `generate_summaries` and `test_model` fail
with this config on a fresh clone. To train from base mT5, delete that line. To generate summaries with adapters
you trained, point it at your checkpoint (training writes `checkpoints/mt5_dailymail_3/epoch_<N>`).

**Input prefix.** `generate_summaries` and `test_model` prepend the config's `task_prefix` (`summarize: ` for
mT5). The training collator accepts `task_prefix` but does not apply it, so training inputs are unprefixed.

## Results

ROUGE F1 (`rouge_score`, with stemming) on the **first 100 articles of the CNN/DailyMail test split**.
Summaries come from `inference/generate_summaries.py`: beam search with 5 beams, `no_repeat_ngram_size=2`,
length penalty 1.0.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Source |
|---|:---:|:---:|:---:|---|
| mT5 + LoRA (beam 5) | 0.2947 | 0.1093 | 0.2118 | [`outputs/mt5_eval_111.json`](outputs/mt5_eval_111.json) |
| Pegasus, off the shelf (beam 5) | 0.3537 | 0.1442 | 0.2613 | [`outputs/pegasus_eval_111.json`](outputs/pegasus_eval_111.json) |

The paper's Table 1 reports different mT5 scores; the committed evaluation outputs above are the reference.

Qualitative outputs for both models: [`outputs/sample_summaries.md`](outputs/sample_summaries.md).
