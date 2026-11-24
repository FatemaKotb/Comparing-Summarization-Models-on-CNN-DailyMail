from datasets import load_dataset
from typing import Tuple

_dataset_cache = {}


def _get_dataset(parquet_dir: str, split: str):
    key = (parquet_dir, split)
    if key not in _dataset_cache:
        _dataset_cache[key] = load_dataset(
            "parquet",
            data_dir=parquet_dir,
            split=split
        )
    return _dataset_cache[key]


def load_example(
    parquet_dir: str,
    index: int,
    split: str = "test"
) -> Tuple[str, str]:

    dataset = _get_dataset(parquet_dir, split)

    item = dataset[index]
    return item["article"].strip(), item["highlights"].strip()
