# Comparing Summarization Models on CNN-DailyMail

This project compares five abstractive summarizers on the CNN/DailyMail benchmark, spanning two families: recurrent models (a pointer-generator and a hierarchical-attention model) and transformers (one used off the shelf, one fine-tuned, and one trained from scratch). Every model is scored with ROUGE-1/2/L, but the evaluation sets and ROUGE implementations are not identical across models (see [Results](#results)), so cross-family comparisons are indicative rather than exact.

Summaries are generated with one of two decoding strategies: greedy decoding picks the single most likely word at each step, while beam search keeps the *k* most likely partial summaries at each step and returns the best complete one, which usually scores higher but is slower.

On the reported numbers, the beam-search pointer-generator scores highest on ROUGE-1 and ROUGE-2, while the off-the-shelf Pegasus transformer is close behind and best on ROUGE-L.

This was a course project for CMP639 Advanced Natural Language Processing, Cairo University. The paper (PDF) is in `paper/`.

## Structure

```
Comparing-Summarization-Models-on-CNN-DailyMail/
├── paper/
│   └── paper.pdf                               # full write-up (methods, results, refs)
└── models/
    ├── recurrent/                              # Recurrent (one package, both models)
    │   ├── configs/                            # base.yaml (pointer-generator), hieratt.yaml
    │   ├── scripts/                            # build_vocab, train, eval, train_hieratt, eval_hieratt
    │   ├── src/
    │   │   ├── data/ utils/
    │   │   └── model/                          # pgn.py (pointer-generator), hieratt.py (hierarchical attention)
    │   └── artifacts/vocab.json                # shared vocabulary (committed)
    └── transformer/                            # Transformer
        ├── from_scratch/                       # trained from scratch
        │   └── vanilla_transformer.ipynb
        └── pretrained_seq2seq/                 # fine-tuned and off-the-shelf (one package)
            ├── configs/                        # Mt5.json (fine-tuned), Pegasus.json (off-the-shelf)
            ├── models/                         # base_model, model_factory, mt5, pegasus
            ├── training/ inference/ evaluation/ preprocessing/
            └── outputs/                        # mt5_eval_111.json, pegasus_eval_111.json, sample_summaries.md
```

## Models

### Recurrent

Both recurrent models read CNN/DailyMail 3.0.0 from HuggingFace (`abisee/cnn_dailymail`) and share a 50k vocabulary built from the training split (lowercased, minimum frequency 2). The vocabulary is committed as `artifacts/vocab.json`, so no build step is needed; rebuilding it overwrites that file:

```bash
cd models/recurrent
pip install -r requirements.txt     # shared by Pointer-Generator and Hierarchical Attention
python -m scripts.build_vocab       # optional: regenerates artifacts/vocab.json
```

#### Pointer-Generator

Each summary word is generated from the vocabulary or copied from the article; tracking already-attended words (coverage) discourages repetition.

```mermaid
flowchart LR
    I[Input<br/>source article] --> E[Encode<br/>BiLSTM]
    E --> A[Attend<br/>coverage-aware attention]
    A --> D[Decode<br/>LSTM<br/>copy or generate each word]
    D --> O[Output<br/>summary]
    classDef io fill:#d9d9d9,stroke:#999999,color:#000000
    class I,O io
```

```bash
python -m scripts.train             # config: configs/base.yaml
python -m scripts.eval
```

#### Hierarchical Attention

It attends to key sentences, then to words within them, and decodes from a small per-batch vocabulary (LVT).

```mermaid
flowchart LR
    I[Input<br/>source article] --> E[Encode<br/>word + sentence BiGRU]
    E --> A[Attend<br/>sentence-then-word attention]
    A --> D[Decode<br/>GRU + LVT]
    D --> O[Output<br/>summary]
    classDef io fill:#d9d9d9,stroke:#999999,color:#000000
    class I,O io
```

```bash
python -m scripts.train_hieratt     # config: configs/hieratt.yaml
python -m scripts.eval_hieratt
```

### Transformer

The three transformers are ordered by how much training this project performs on them: none (off the shelf), then fine-tuning, then training from scratch.

#### Off-the-Shelf (Pegasus)

Pegasus is run as-is, but this checkpoint was already fine-tuned on CNN/DailyMail, so it isn't truly zero-shot.

```mermaid
flowchart LR
    I[Input<br/>source article] --> E[Encode<br/>Pegasus encoder, frozen]
    E --> D[Decode<br/>Pegasus decoder, frozen]
    D --> O[Output<br/>summary]
    classDef io fill:#d9d9d9,stroke:#999999,color:#000000
    class I,O io
```

```bash
cd models/transformer/pretrained_seq2seq
pip install -r requirements.txt                                         # shared with Fine-Tuned (mT5)
# expects CNN/DailyMail Parquet files in data/raw_data/DailyMail/3.0.0
# no training step
python -m inference.generate_summaries --config configs/Pegasus.json    # -> outputs/pegasus_predictions.json
python -m evaluation.run_evaluation --config configs/Pegasus.json       # ROUGE: outputs/pegasus_eval_111.json
```

#### Fine-Tuned (mT5 + LoRA)

Multilingual mT5 is fine-tuned with LoRA adapters that update under 1% of its parameters.

```mermaid
flowchart LR
    I[Input<br/>source article] --> E[Encode<br/>mT5 encoder, frozen + LoRA]
    E --> D[Decode<br/>mT5 decoder, frozen + LoRA]
    D --> O[Output<br/>summary]
    classDef io fill:#d9d9d9,stroke:#999999,color:#000000
    class I,O io
```

`configs/Mt5.json` sets `pretrained_lora_weights` to `checkpoints/mt5_dailymail_2/epoch_1`, which is not in the repo. To train from base mT5, delete that line. To generate summaries with adapters you trained, point it at your checkpoint (training writes `checkpoints/mt5_dailymail_3/epoch_<N>`).

```bash
cd models/transformer/pretrained_seq2seq
pip install -r requirements.txt                                         # shared with Off-the-Shelf (Pegasus)
# expects CNN/DailyMail Parquet files in data/raw_data/DailyMail/3.0.0
python -m training.train --config configs/Mt5.json
python -m inference.generate_summaries --config configs/Mt5.json        # -> outputs/mt5_dailymail_predictions.json
python -m evaluation.run_evaluation --config configs/Mt5.json           # ROUGE: outputs/mt5_eval_111.json
```

#### Trained from Scratch

This baseline is a 6-layer encoder-decoder transformer trained from scratch, with no pre-training, inside a Kaggle notebook.

```mermaid
flowchart LR
    I[Input<br/>source article] --> E[Encode<br/>6-layer encoder]
    E --> D[Decode<br/>6-layer decoder]
    D --> O[Output<br/>summary]
    classDef io fill:#d9d9d9,stroke:#999999,color:#000000
    class I,O io
```

The notebook was written for Kaggle: it reads and writes hardcoded `/kaggle/input/...` and `/kaggle/working` paths, including a checkpoint it resumes from. Run it on Kaggle, or edit those paths before running it locally.

```bash
cd models/transformer/from_scratch
pip install -r requirements.txt               # includes jupyter
jupyter notebook vanilla_transformer.ipynb    # train + evaluate by running all cells
```

## Results

ROUGE F1 on CNN/DailyMail. The models were not all scored on the same examples or with the same ROUGE library (see the note below the table).

The decoding strategy (greedy or beam-*k*, explained in the introduction) is given in brackets.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Recurrent — Pointer-Generator (greedy) | 0.3668 | 0.1486 | 0.2475 |
| Recurrent — Pointer-Generator (beam-4) | 0.3786 | 0.1629 | 0.2584 |
| Recurrent — Hierarchical Attention (beam-4) | 0.2934 | 0.1087 | 0.2037 |
| Transformer — Off-the-Shelf Pegasus (beam-5) | 0.3537 | 0.1442 | 0.2613 |
| Transformer — Fine-Tuned mT5 + LoRA (beam-5) | 0.2947 | 0.1093 | 0.2118 |
| Transformer — Trained from Scratch (greedy) | 0.1983 | 0.0307 | 0.1859 |

Recurrent and from-scratch scores are from the paper's Table 1; Pegasus and mT5 scores are from the committed [outputs](models/transformer/pretrained_seq2seq/outputs/) (first 100 test articles), and the mT5 row differs from the paper. The recurrent models are scored on the validation split, and the from-scratch notebook on 50 random test articles with the `rouge` package rather than `rouge_score`.

On these numbers, a compact recurrent model beats the much larger fine-tuned mT5 on every metric and the off-the-shelf Pegasus on ROUGE-1 and ROUGE-2, falling just short of Pegasus on ROUGE-L.

## License

This project is licensed under the MIT License.

## Contributors

| [<img src="https://github.com/ahmedemad81.png" width="80"/><br/>Ahmed Emad](https://github.com/ahmedemad81) | [<img src="https://github.com/FatemaKotb.png" width="80"/><br/>Fatema Kotb](https://github.com/FatemaKotb) | Amr Elsheshtawy | Salma Hamed |
|:-:|:-:|:-:|:-:|
