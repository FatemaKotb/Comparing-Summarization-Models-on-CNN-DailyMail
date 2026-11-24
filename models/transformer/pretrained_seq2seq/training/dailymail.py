from typing import Optional, Dict, Any
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, load_dataset
import random

class DailyMailDataset(Dataset):
    """
    PyTorch Dataset wrapper for the DailyMail dataset (or any parquet dataset).

    Supports:
    - Random sampling of a maximum number of examples
    - Type-safe access for static checkers
    """

    def __init__(self, data_dir: str, split: str = "train", max_samples: Optional[int] = 10000):
        # Load dataset
        dataset = load_dataset(
            "parquet",
            data_dir=data_dir,
            split=split
        )

        # Ensure it's a HuggingFace Dataset (not DatasetDict or IterableDataset)
        if not isinstance(dataset, HFDataset):
            raise TypeError(f"Expected a Dataset for split '{split}', got {type(dataset)}")

        # Randomly select max_samples if needed
        if max_samples is not None and max_samples < len(dataset):
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)

        self.dataset: HFDataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        return {
            "article": item["article"].strip(),
            "title": item["highlights"].strip()
        }
