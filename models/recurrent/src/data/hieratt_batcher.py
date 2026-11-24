import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.textproc import tokenize, truncate
from src.data.vocab import Vocab


# Optional: better sentence segmentation, fallback to simple regex.
try:
    import pysbd  # type: ignore
    _SEG = pysbd.Segmenter(language="en", clean=True)

    def split_sentences(text: str) -> List[str]:
        text = (text or "").strip()
        return _SEG.segment(text) if text else []
except Exception:
    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def split_sentences(text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        return [s for s in _SENT_RE.split(text) if s]


def encode_article_hier(
    article_text: str,
    vocab: Vocab,
    lowercase: bool,
    max_src_len: int,
    max_src_sents: int,
) -> Tuple[List[int], List[int], int]:
    """Encode an article into (src_ids, sent_ids, sent_count) using the SAME word tokenization as PGN."""
    sents = split_sentences(article_text)[:max_src_sents]
    src_ids: List[int] = []
    sent_ids: List[int] = []

    for s_idx, sent in enumerate(sents):
        toks = tokenize(sent, lowercase=lowercase)
        for tok in toks:
            if max_src_len > 0 and len(src_ids) >= max_src_len:
                break
            src_ids.append(vocab.encode(tok))
            sent_ids.append(s_idx)
        if max_src_len > 0 and len(src_ids) >= max_src_len:
            break

    if not src_ids:
        src_ids = [vocab.unk_id]
        sent_ids = [0]
        sent_count = 1
    else:
        sent_count = max(sent_ids) + 1

    return src_ids, sent_ids, sent_count


def encode_summary(
    summary_text: str,
    vocab: Vocab,
    lowercase: bool,
    max_tgt_len: int,
) -> Tuple[List[int], List[int]]:
    toks = tokenize(summary_text, lowercase=lowercase)
    toks = truncate(toks, max_tgt_len - 1)  # room for EOS in target
    ids = [vocab.encode(t) for t in toks]
    dec_in = [vocab.bos_id] + ids
    dec_tgt = ids + [vocab.eos_id]
    return dec_in, dec_tgt


@dataclass
class HierAttExample:
    src_ids: torch.LongTensor
    sent_ids: torch.LongTensor
    sent_count: int
    dec_in: torch.LongTensor
    dec_tgt: torch.LongTensor


class HierAttDataset(torch.utils.data.Dataset):
    """HF dataset wrapper producing hierarchical fields + normal decoder ids (global)."""
    def __init__(
        self,
        hf_split,
        vocab: Vocab,
        src_col: str = "article",
        tgt_col: str = "highlights",
        lowercase: bool = True,
        max_src_len: int = 800,
        max_src_sents: int = 60,
        max_tgt_len: int = 120,
    ) -> None:
        self.data = hf_split
        self.vocab = vocab
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.lowercase = lowercase
        self.max_src_len = max_src_len
        self.max_src_sents = max_src_sents
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> HierAttExample:
        ex = self.data[i]
        src_ids, sent_ids, sent_count = encode_article_hier(
            str(ex[self.src_col]),
            self.vocab,
            lowercase=self.lowercase,
            max_src_len=self.max_src_len,
            max_src_sents=self.max_src_sents,
        )
        dec_in, dec_tgt = encode_summary(
            str(ex[self.tgt_col]),
            self.vocab,
            lowercase=self.lowercase,
            max_tgt_len=self.max_tgt_len,
        )
        return HierAttExample(
            src_ids=torch.tensor(src_ids, dtype=torch.long),
            sent_ids=torch.tensor(sent_ids, dtype=torch.long),
            sent_count=int(sent_count),
            dec_in=torch.tensor(dec_in, dtype=torch.long),
            dec_tgt=torch.tensor(dec_tgt, dtype=torch.long),
        )


def build_common_pool(
    hf_train_split,
    vocab: Vocab,
    tgt_col: str = "highlights",
    lowercase: bool = True,
    max_samples: int = 20000,
) -> List[int]:
    """Frequent token IDs (excluding specials) used to fill the LVT when K isn't full."""
    c = Counter()
    n = min(int(max_samples), len(hf_train_split))
    for i in range(n):
        toks = tokenize(str(hf_train_split[i][tgt_col]), lowercase=lowercase)
        c.update(vocab.encode(t) for t in toks)

    specials = {vocab.pad_id, vocab.unk_id, vocab.bos_id, vocab.eos_id}
    return [tid for tid, _ in c.most_common() if tid not in specials]


def _build_lvt(
    src_ids: torch.LongTensor,   # (B,Tsrc)
    dec_tgt: torch.LongTensor,   # (B,Tdec)
    src_mask: torch.Tensor,      # (B,Tsrc) bool
    dec_mask: torch.Tensor,      # (B,Tdec) bool
    common_pool: List[int],
    vocab: Vocab,
    K: int,
) -> Tuple[torch.LongTensor, Dict[int, int]]:
    """Paper-style LVT: specials + unique(batch src+target) + fill from common_pool."""
    specials = [vocab.pad_id, vocab.unk_id, vocab.bos_id, vocab.eos_id]
    local: List[int] = []
    seen = set()

    def add(x: int) -> None:
        if x not in seen and len(local) < K:
            seen.add(x)
            local.append(x)

    for s in specials:
        add(int(s))

    # 1) Always include ALL target tokens first (so dec_tgt_local is not corrupted)
    tgt_tokens = [int(x) for x in dec_tgt[dec_mask].tolist()]
    tgt_unique = []
    seen_t = set()
    for tid in tgt_tokens:
        if tid not in seen_t:
            seen_t.add(tid)
            tgt_unique.append(tid)

    for tid in tgt_unique:
        add(tid)
        if len(local) >= K:
            break

    # 2) Fill remaining slots using source frequency
    if len(local) < K:
        src_freq = Counter(int(x) for x in src_ids[src_mask].tolist())
        for tid, _ in src_freq.most_common():
            add(int(tid))
            if len(local) >= K:
                break

    if len(local) < K:
        for tid in common_pool:
            add(int(tid))
            if len(local) >= K:
                break

    while len(local) < K:
        add(int(vocab.unk_id))

    local_vocab_ids = torch.tensor(local, dtype=torch.long)
    mapping = {gid: i for i, gid in enumerate(local)}
    return local_vocab_ids, mapping


def _map_global_to_local(ids: torch.LongTensor, mapping: Dict[int, int], unk_global: int) -> torch.LongTensor:
    unk_local = mapping.get(int(unk_global), 0)
    flat = ids.flatten().tolist()
    mapped = [mapping.get(int(g), unk_local) for g in flat]
    return torch.tensor(mapped, dtype=torch.long, device=ids.device).view_as(ids)


def collate_hieratt(
    batch: List[HierAttExample],
    vocab: Vocab,
    common_pool: List[int],
    K: int,
) -> Dict[str, Any]:
    # pad source + sentence ids
    src = [b.src_ids for b in batch]
    sent = [b.sent_ids for b in batch]
    sent_counts = torch.tensor([b.sent_count for b in batch], dtype=torch.long)

    # pad decoder
    dec_in = [b.dec_in for b in batch]
    dec_tgt = [b.dec_tgt for b in batch]

    src_lens = torch.tensor([len(x) for x in src], dtype=torch.long)
    dec_lens = torch.tensor([len(x) for x in dec_in], dtype=torch.long)

    src_ids = pad_sequence(src, batch_first=True, padding_value=vocab.pad_id)
    sent_ids = pad_sequence(sent, batch_first=True, padding_value=0)

    dec_in_ids = pad_sequence(dec_in, batch_first=True, padding_value=vocab.pad_id)
    dec_tgt_ids = pad_sequence(dec_tgt, batch_first=True, padding_value=vocab.pad_id)

    src_mask = (src_ids != vocab.pad_id)
    dec_mask = (dec_in_ids != vocab.pad_id)
    dec_tgt_mask = (dec_tgt_ids != vocab.pad_id)

    # build LVT (local vocab) for this batch
    local_vocab_ids, mapping = _build_lvt(
        src_ids=src_ids,
        dec_tgt=dec_tgt_ids,
        src_mask=src_mask,
        dec_mask=dec_tgt_mask,
        common_pool=common_pool,
        vocab=vocab,
        K=int(K),
    )
    local_vocab_ids = local_vocab_ids.to(src_ids.device)

    # map decoder targets to local ids (for loss)
    dec_in_local = _map_global_to_local(dec_in_ids, mapping, vocab.unk_id)
    dec_tgt_local = _map_global_to_local(dec_tgt_ids, mapping, vocab.unk_id)

    return {
        "src_ids": src_ids,
        "src_lens": src_lens,
        "src_mask": src_mask,
        "sent_ids": sent_ids,
        "sent_counts": sent_counts,
        "dec_in": dec_in_ids,                 # global ids (for embedding)
        "dec_tgt": dec_tgt_ids,               # global ids (debug/inspection)
        "dec_in_local": dec_in_local,         # local ids (optional)
        "dec_tgt_local": dec_tgt_local,       # local ids (for loss)
        "dec_lens": dec_lens,
        "dec_mask": dec_mask,
        "local_vocab_ids": local_vocab_ids,   # [K] global ids
    }
