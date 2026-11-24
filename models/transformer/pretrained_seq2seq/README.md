# Approach II — Pretrained Seq2Seq Transformers (mT5 + LoRA, Pegasus)

One Hugging Face package serving **two** models from the paper. The model is chosen entirely by the
config passed via `--config`; `models/model_factory.py` maps the config's `model_name` to a wrapper class.

| Model | Config | Wrapper | Checkpoint |
|---|---|---|---|
| mT5-base + LoRA (fine-tuned) | `configs/Mt5.json` | `models/mt5.py` | `google/mt5-base` |
| Pegasus, zero-shot | `configs/Pegasus.json` | `models/pegasus.py` | `google/pegasus-cnn_dailymail` |

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
# Fine-tune mT5 with LoRA (Pegasus is evaluated zero-shot and is not trained)
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

`configs/Mt5.json` resumes from LoRA weights at `checkpoints/mt5_dailymail_2/epoch_1`
(`pretrained_lora_weights`) and writes new checkpoints to `checkpoints/mt5_dailymail_3`.
Remove `pretrained_lora_weights` to start from the base model.

## Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Source |
|---|:---:|:---:|:---:|---|
| mT5 + LoRA (beam 4) | 0.3947 | 0.2093 | 0.3118 | paper Table 1; original mT5 README |
| mT5 + LoRA | 0.2947 | 0.1093 | 0.2118 | [`outputs/mt5_eval_111.json`](outputs/mt5_eval_111.json) |
| Pegasus zero-shot (beam 4) | 0.3537 | 0.1442 | 0.2613 | [`outputs/pegasus_eval_111.json`](outputs/pegasus_eval_111.json); paper Table 1 |

> The committed mT5 eval JSON is exactly 0.1 below the paper on all three metrics (the original mT5
> README labelled the eval "after 2 epochs"). Both values are kept unchanged — reconcile before citing.

Qualitative outputs for both models: [`outputs/sample_summaries.md`](outputs/sample_summaries.md).
