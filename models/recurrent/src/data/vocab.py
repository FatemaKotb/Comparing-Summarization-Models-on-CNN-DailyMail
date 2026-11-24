import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @property
    def pad_id(self) -> int: return self.token_to_id["<pad>"]
    @property
    def unk_id(self) -> int: return self.token_to_id["<unk>"]
    @property
    def bos_id(self) -> int: return self.token_to_id["<s>"]
    @property
    def eos_id(self) -> int: return self.token_to_id["</s>"]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, tok: str) -> int:
        return self.token_to_id.get(tok, self.unk_id)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return "<oov>"

def build_vocab(
    token_counter: Counter,
    vocab_size: int,
    min_freq: int = 1
) -> Vocab:
    token_to_id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    id_to_token: List[str] = list(SPECIAL_TOKENS)

    for tok, freq in token_counter.most_common():
        if tok in token_to_id:
            continue
        if freq < min_freq:
            continue
        token_to_id[tok] = len(id_to_token)
        id_to_token.append(tok)
        if len(id_to_token) >= vocab_size:
            break

    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)

def save_vocab(vocab: Vocab, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"id_to_token": vocab.id_to_token},
            f,
            ensure_ascii=False,
            indent=2
        )

def load_vocab(path: str) -> Vocab:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    id_to_token = obj["id_to_token"]
    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)
