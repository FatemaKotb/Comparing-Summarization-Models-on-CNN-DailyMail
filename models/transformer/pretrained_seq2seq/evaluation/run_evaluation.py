
import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset, Dataset
from evaluation.rouge_eval import RougeEvaluator


def load_predictions(path: str) -> List[str]:
    """
    Load predictions JSON and return summaries in index order.
    Assumes keys are 'sample_0', 'sample_1', ...
    """
    with open(path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    return [preds[k] for k in sorted(preds, key=lambda x: int(x.split("_")[-1]))]


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization predictions with ROUGE.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # -----------------------
    # Load config
    # -----------------------
    with open(args.config, "r", encoding="utf-8") as f:
        params = json.load(f)

    dataset_dir = params["dataset_dir"]
    predictions_path = Path(params["output_path"])
    model_name = params.get("model_name", "model")

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    # -----------------------
    # Load predictions
    # -----------------------
    predictions = load_predictions(str(predictions_path))
    num_preds = len(predictions)

    print(f"Loaded {num_preds} predictions")

    # -----------------------
    # Load references from parquet
    # -----------------------
    dataset = load_dataset(
        "parquet",
        data_dir=dataset_dir,
        split="test"
    )

    assert isinstance(dataset, Dataset)

    references = dataset.select(range(num_preds))["highlights"]


    for ref in references:
        print("ref: ", ref)
        print(" ")

    # -----------------------
    # ROUGE evaluation
    # -----------------------
    evaluator = RougeEvaluator()
    scores = evaluator.evaluate(predictions, references)

    print("\nROUGE Scores")
    print("-" * 30)
    for k, v in scores.items():
        print(f"{k.upper():8s}: {v:.4f}")

    # -----------------------
    # Save scores
    # -----------------------
    out_dir = predictions_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = out_dir / f"{model_name}_eval_111.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print(f"\nSaved ROUGE scores to {scores_path}")


if __name__ == "__main__":
    main()
