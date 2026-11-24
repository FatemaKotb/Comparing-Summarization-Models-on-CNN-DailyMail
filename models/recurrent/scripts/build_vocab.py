import argparse
import os
import yaml
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset

from src.data.textproc import tokenize
from src.data.vocab import build_vocab, save_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml", help="Config yaml to use for vocab build")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    os.makedirs(cfg["artifacts_dir"], exist_ok=True)

    # Load dataset (e.g., CNN/DailyMail)
    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"])
    train = ds["train"]

    # Source and target text columns
    src_col = "article"
    tgt_col = "highlights"

    # Count token frequencies across source and target texts
    counter = Counter()
    for ex in tqdm(train, desc="Counting tokens"):
        counter.update(tokenize(ex.get(src_col, ""), lowercase=cfg["lowercase"]))
        counter.update(tokenize(ex.get(tgt_col, ""), lowercase=cfg["lowercase"]))

    # Build vocabulary with size and frequency constraints
    vocab = build_vocab(
        token_counter=counter,
        vocab_size=cfg["vocab_size"],
        min_freq=cfg["min_freq"],
    )

    # Save vocabulary to disk
    vocab_path = os.path.join(cfg["artifacts_dir"], "vocab.json")
    save_vocab(vocab, vocab_path)

    print("Saved vocab:", vocab_path)
    print("Vocab size:", vocab.size)


if __name__ == "__main__":
    main()
