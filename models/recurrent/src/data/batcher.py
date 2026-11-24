from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch

from .textproc import tokenize, truncate
from .vocab import Vocab

def make_extended_vocab(
    src_tokens: List[str],
    vocab: Vocab
) -> Tuple[List[int], List[int], List[str], Dict[str, int]]:
    """
    Returns:
      src_ids: base vocab ids (OOV -> <unk>)
      src_ext_ids: extended ids (OOV -> vocab.size + oov_index)
      oovs: list of OOV tokens (unique, in order of first appearance)
      oov_to_ext: mapping token -> extended id
    """
    src_ids: List[int] = []
    src_ext_ids: List[int] = []
    oovs: List[str] = []
    oov_to_ext: Dict[str, int] = {}

    for tok in src_tokens:
        vid = vocab.encode(tok)
        src_ids.append(vid)
        if vid != vocab.unk_id:
            src_ext_ids.append(vid)
        else:
            if tok not in oov_to_ext:
                oov_to_ext[tok] = vocab.size + len(oovs)
                oovs.append(tok)
            src_ext_ids.append(oov_to_ext[tok])

    return src_ids, src_ext_ids, oovs, oov_to_ext

def tgt_to_extended_ids(
    tgt_tokens: List[str],
    vocab: Vocab,
    oov_to_ext: Dict[str, int]
) -> List[int]:
    ids: List[int] = []
    for tok in tgt_tokens:
        vid = vocab.encode(tok)
        if vid != vocab.unk_id:
            ids.append(vid)
        else:
            # If OOV but appears in source, use its extended id so it can be copied
            if tok in oov_to_ext:
                ids.append(oov_to_ext[tok])
            else:
                ids.append(vocab.unk_id)
    return ids

class PGNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_split,
        vocab: Vocab,
        src_col: str = "article",
        tgt_col: str = "highlights",
        lowercase: bool = True,
        max_src_len: int = 600,
        max_tgt_len: int = 120
    ):
        self.data = hf_split
        self.vocab = vocab
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.lowercase = lowercase
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.data[idx]
        src = truncate(tokenize(ex.get(self.src_col, ""), self.lowercase), self.max_src_len)
        tgt = truncate(tokenize(ex.get(self.tgt_col, ""), self.lowercase), self.max_tgt_len)

        src_ids, src_ext_ids, oovs, oov_to_ext = make_extended_vocab(src, self.vocab)

        # Decoder input uses base ids (unknown is fine); target uses extended ids.
        dec_in = [self.vocab.bos_id] + [self.vocab.encode(t) for t in tgt]
        dec_tgt = tgt_to_extended_ids(tgt, self.vocab, oov_to_ext) + [self.vocab.eos_id]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "src_ext_ids": torch.tensor(src_ext_ids, dtype=torch.long),
            "dec_in": torch.tensor(dec_in, dtype=torch.long),
            "dec_tgt": torch.tensor(dec_tgt, dtype=torch.long),
            "oovs": oovs,
        }

def collate_fn(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    bsz = len(batch)

    src_lens = torch.tensor([len(x["src_ids"]) for x in batch], dtype=torch.long)
    dec_lens = torch.tensor([len(x["dec_in"]) for x in batch], dtype=torch.long)

    max_src = int(src_lens.max().item())
    max_dec = int(dec_lens.max().item())

    src_ids = torch.full((bsz, max_src), pad_id, dtype=torch.long)
    src_ext_ids = torch.full((bsz, max_src), pad_id, dtype=torch.long)

    dec_in = torch.full((bsz, max_dec), pad_id, dtype=torch.long)
    # dec_tgt is same length as dec_in (because we appended eos to tgt, bos to in)
    dec_tgt = torch.full((bsz, max_dec), pad_id, dtype=torch.long)

    oovs_list = []
    max_oov = 0

    for i, x in enumerate(batch):
        s = x["src_ids"]
        se = x["src_ext_ids"]
        di = x["dec_in"]
        dt = x["dec_tgt"]
        src_ids[i, : len(s)] = s
        src_ext_ids[i, : len(se)] = se
        dec_in[i, : len(di)] = di
        dec_tgt[i, : len(dt)] = dt
        oovs_list.append(x["oovs"])
        max_oov = max(max_oov, len(x["oovs"]))

    src_mask = (src_ids != pad_id).float()
    dec_mask = (dec_in != pad_id).float()

    return {
        "src_ids": src_ids,
        "src_ext_ids": src_ext_ids,
        "src_lens": src_lens,
        "src_mask": src_mask,
        "dec_in": dec_in,
        "dec_tgt": dec_tgt,
        "dec_lens": dec_lens,
        "dec_mask": dec_mask,
        "oovs": oovs_list,
        "max_oov": max_oov,
    }
