import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial

import sys
from pathlib import Path

from src.data.vocab import load_vocab
from src.data.hieratt_batcher import HierAttDataset, collate_hieratt, build_common_pool
from src.model.hieratt import HierAttnSeq2Seq
from src.utils.device import get_device
from src.utils.checkpoint import load_ckpt
from src.utils.rouge import compute_rouge
from src.utils.logging import should_log
from src.utils.diag_hieratt import debug_hieratt_batch, env_debug_enabled

def ids_to_text(ids, vocab) -> str:
    toks = []
    for i in ids:
        if i == vocab.eos_id:
            break
        tok = vocab.decode(int(i))
        if tok not in ("<s>", "</s>", "<pad>"):
            toks.append(tok)
    return " ".join(toks)


def length_penalty(length: int, alpha: float) -> float:
    return ((5.0 + float(length)) / 6.0) ** alpha


@torch.no_grad()
def beam_decode_single(
    model: HierAttnSeq2Seq,
    batch,
    vocab,
    max_len: int,
    beam_size: int = 4,
    alpha: float = 0.6,
    min_len: int = 35,
):
    """Beam search for ONE example (batch size = 1). Returns list of GLOBAL token ids (no BOS)."""
    device = next(model.parameters()).device

    enc_cache, dec_state0 = model.encode(batch)
    local_vocab_ids = batch["local_vocab_ids"].to(device)  # [K]

    # beam item = (tokens_so_far_global, sum_logp, dec_state(h,ctx), ended)
    beams = [([vocab.bos_id], 0.0, (dec_state0[0].clone(), dec_state0[1].clone()), False)]
    completed = []

    for t in range(int(max_len)):
        new_beams = []
        for tokens, sum_logp, state, ended in beams:
            if ended:
                new_beams.append((tokens, sum_logp, state, True))
                continue

            y_prev = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
            log_probs, next_state = model.decode_step(y_prev=y_prev, dec_state=state, enc_cache=enc_cache)
            lp = log_probs.squeeze(0)  # [K]

            cur_len = len(tokens) - 1
            if cur_len < int(min_len):
                eos_local = (local_vocab_ids == vocab.eos_id).nonzero(as_tuple=False)
                if eos_local.numel() > 0:
                    lp[int(eos_local[0].item())] = -1e9

            topk_logp, topk_ids = torch.topk(lp, k=int(beam_size))

            for k in range(int(beam_size)):
                tok_local = int(topk_ids[k].item())
                tok_global = int(local_vocab_ids[tok_local].item())
                tok_logp = float(topk_logp[k].item())

                new_tokens = tokens + [tok_global]
                new_sum = sum_logp + tok_logp
                new_ended = (tok_global == vocab.eos_id)

                h, ctx = next_state
                cloned_state = (h.clone(), ctx.clone())

                new_beams.append((new_tokens, new_sum, cloned_state, new_ended))

        def score_fn(b):
            toks, s, *_ = b
            L = max(1, len(toks) - 1)
            return s / length_penalty(L, float(alpha))

        new_beams.sort(key=score_fn, reverse=True)
        beams = new_beams[:int(beam_size)]

        completed = [b for b in beams if b[3]]
        if len(completed) == int(beam_size):
            break

    final_pool = completed if len(completed) > 0 else beams
    final_pool.sort(key=lambda b: b[1] / length_penalty(max(1, len(b[0]) - 1), float(alpha)), reverse=True)
    best_tokens = final_pool[0][0]

    return best_tokens[1:]  # drop BOS


@torch.no_grad()
def beam_decode_batch(model, batch, vocab, cfg):
    B = batch["src_ids"].size(0)
    outs = []
    for i in range(B):
        one = {k: (v[i:i+1] if torch.is_tensor(v) else v) for k, v in batch.items()}
        ids = beam_decode_single(
            model=model,
            batch=one,
            vocab=vocab,
            max_len=int(cfg.get("max_dec_len", 120)),
            beam_size=int(cfg.get("beam_size", 4)),
            alpha=float(cfg.get("len_penalty", 0.6)),
            min_len=int(cfg.get("min_dec_len", 35)),
        )
        outs.append(ids)
    return outs


def main():
    cfg = yaml.safe_load(open("configs/hieratt.yaml", "r", encoding="utf-8"))
    device = get_device()
    print("Device:", device)

    vocab = load_vocab(os.path.join(cfg["artifacts_dir"], "vocab.json"))
    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"])

    common_pool = build_common_pool(
        ds["train"],
        vocab,
        tgt_col="highlights",
        lowercase=bool(cfg["lowercase"]),
        max_samples=int(cfg.get("common_pool_max_samples", 20000)),
    )

    val_set = HierAttDataset(
        ds["validation"],
        vocab,
        src_col="article",
        tgt_col="highlights",
        lowercase=bool(cfg["lowercase"]),
        max_src_len=int(cfg["max_src_len"]),
        max_src_sents=int(cfg["max_src_sents"]),
        max_tgt_len=int(cfg["max_tgt_len"]),
    )

    collate = partial(
        collate_hieratt,
        vocab=vocab,
        common_pool=common_pool,
        K=int(cfg["lvt_k"]),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )

    model = HierAttnSeq2Seq(
        vocab_size=vocab.size,
        emb_dim=int(cfg["emb_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        attn_dim=int(cfg["attn_dim"]),
        dropout=float(cfg["dropout"]),
        pad_id=vocab.pad_id,
        lvt_k=int(cfg["lvt_k"]),
        pos_dim=int(cfg["pos_dim"]),
        max_pos=int(cfg["max_pos"]),
    ).to(device)

    ckpt_path = os.path.join(cfg["ckpt_dir"], "hieratt_best.pt")
    print("Loading:", ckpt_path)
    load_ckpt(ckpt_path, model, optimizer=None)

    preds, refs = [], []
    total = len(val_loader) if hasattr(val_loader, "__len__") else None
    for i, batch in enumerate(val_loader, start=1):
        if total is not None and should_log(i, total, every=50):
            print(f"[eval]  {i}/{total}")
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        if env_debug_enabled() and i == 1:
            debug_hieratt_batch(batch, vocab, cfg, title="EVAL/FIRST-BATCH", max_tokens=60)

        out_ids = beam_decode_batch(model, batch, vocab, cfg)[0]
        pred = ids_to_text(out_ids, vocab)

        ref_ids = batch["dec_tgt"][0].tolist()
        ref = ids_to_text(ref_ids, vocab)

        preds.append(pred)
        refs.append(ref)

    scores = compute_rouge(preds, refs)
    print(scores)


if __name__ == "__main__":
    main()
