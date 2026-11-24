# Approach II — Transformer Trained from Scratch

A standard encoder–decoder transformer (6 encoder + 6 decoder layers, multi-head self-attention,
sinusoidal positional encodings) trained from scratch on CNN/DailyMail, with no pretrained weights.
Everything lives in a single notebook: [`vanilla_transformer.ipynb`](vanilla_transformer.ipynb).

- Tokenizer: `T5Tokenizer` (vocabulary only — model weights are randomly initialised)
- Inputs truncated to 400 tokens; teacher forcing with cross-entropy; 3 epochs; greedy decoding
- Final validation loss: 3.4865

## Running

The notebook was developed on **Kaggle**: it uses Kaggle paths (`/kaggle/input/...`, `/kaggle/working`)
and `kagglehub` to fetch the dataset. Run it on Kaggle as-is, or edit those paths to run locally.

```bash
pip install -r requirements.txt
```

## Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|:---:|:---:|:---:|
| Transformer (scratch) | 0.1983 | 0.0307 | 0.1859 |

(Table 1 of [the paper](../../../paper/paper.pdf).)
