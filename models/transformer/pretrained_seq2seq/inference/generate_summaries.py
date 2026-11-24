import sys
import json
from typing import List, Dict
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Dataset


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.model_factory import get_model
from preprocessing.clean_text import clean_text


def generate_summaries(
    model,
    tokenizer,
    texts: List[str],
    device,
    max_length: int,
    max_input_length: int
) -> List[str]:

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_input_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            length_penalty=1.0
        )

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        params = json.load(f)

    model_name = params["model_name"]
    data_dir = params["dataset_dir"]
    output_path = params["output_path"]

    task_prefix = params.get("task_prefix", "")
    task_suffix = params.get("task_suffix", "")

    max_tokens = params.get("max_tokens", 1024)
    max_output_tokens = params.get("max_output_tokens", 256)
    batch_size = params.get("batch_size", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_wrapper, tokenizer = get_model(
        model_name,
        pretrained_lora_weights=params.get("pretrained_lora_weights")
    )

    model = model_wrapper.model.to(device)

    dataset = load_dataset(
        "parquet",
        data_dir=data_dir,
        split="test",
    )

    assert isinstance(dataset, Dataset)

    predictions: Dict[str, str] = {}

    MAX_SAMPLES = 100
    num_samples = min(MAX_SAMPLES, len(dataset))

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating summaries"):
        batch = dataset[i:i + batch_size]

        batch_texts = [
            f"{task_prefix} {clean_text(article)} {task_suffix}".strip()
            for article in batch["article"]
        ]

        summaries = generate_summaries(
            model,
            tokenizer,
            batch_texts,
            device,
            max_length=max_output_tokens,
            max_input_length=max_tokens
        )

        for j, summary in enumerate(summaries):
            predictions[str(i + j)] = summary

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
