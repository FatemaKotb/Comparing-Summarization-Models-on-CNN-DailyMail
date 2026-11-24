# Approach II — Transformer Trained from Scratch

A standard encoder–decoder transformer (6 encoder + 6 decoder layers, multi-head self-attention,
sinusoidal positional encodings) trained from scratch on CNN/DailyMail, with no pretrained weights.
Everything lives in a single notebook: [`vanilla_transformer.ipynb`](vanilla_transformer.ipynb).

- Data: CNN/DailyMail CSVs from Kaggle (`gowrishankarp/newspaper-text-summarization-cnn-dailymail`, via `kagglehub`)
- Tokenizer: `t5-base` `T5Tokenizer` (vocabulary only; model weights are randomly initialised)
- Articles truncated to 400 tokens, summaries to 100; teacher forcing with cross-entropy; 3 epochs; greedy decoding
- Final validation loss: 3.4865

## Running

The notebook was written for **Kaggle**: it reads and writes hardcoded `/kaggle/input/...` and
`/kaggle/working` paths, including a checkpoint it resumes from
(`/kaggle/input/vanilla-transformer/.../checkpoint_batch_15500.pth`). Run it on Kaggle, or edit those
paths before running it locally.

```bash
pip install -r requirements.txt               # includes jupyter
jupyter notebook vanilla_transformer.ipynb    # train + evaluate by running all cells
```

## Results

ROUGE F1 computed in the notebook on **50 random articles from the test split** (`random_state=42`) with the
`rouge` package. That is a different ROUGE implementation and evaluation set from the other models
(`rouge_score` with stemming), so cross-model comparisons are indicative only.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|:---:|:---:|:---:|
| Transformer (scratch, greedy) | 0.1983 | 0.0307 | 0.1859 |

(As reported in Table 1 of [the paper](../../../paper/paper.pdf).)
