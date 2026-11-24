import json
from typing import List, Tuple


def load_predictions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [x["prediction"] for x in data]


def load_references(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [x["title"] for x in data]


def load_pred_ref(pred_path: str, ref_path: str) -> Tuple[List[str], List[str]]:
    preds = load_predictions(pred_path)
    refs = load_references(ref_path)
    assert len(preds) == len(refs)
    return preds, refs
