# src/utils/diag_hieratt.py
import os
import math
import torch


def _tok(vocab, idx: int) -> str:
    try:
        return vocab.decode(int(idx))
    except Exception:
        return f"<id:{int(idx)}>"


def _is_unk(vocab, idx: int) -> bool:
    return _tok(vocab, idx) == "<unk>"


def _estimate_unk_rate(vocab, ids: torch.Tensor, max_sample: int = 5000) -> float:
    if not torch.is_tensor(ids):
        return float("nan")
    flat = ids.detach().flatten()
    if flat.numel() == 0:
        return 0.0
    n = min(max_sample, flat.numel())
    # sample evenly spaced indices for speed
    if n < flat.numel():
        step = max(1, flat.numel() // n)
        sample = flat[::step][:n]
    else:
        sample = flat
    unk = 0
    for x in sample.tolist():
        if _is_unk(vocab, x):
            unk += 1
    return unk / max(1, len(sample))


def _local_to_global(batch, local_ids_1d: torch.Tensor, local_vocab_ids: torch.Tensor) -> torch.Tensor:
    """
    local_ids_1d: [T] local indices
    local_vocab_ids: [K] or [B,K]
    returns global ids: [T]
    """
    if local_vocab_ids.dim() == 1:
        K = int(local_vocab_ids.size(0))
        safe = local_ids_1d.clamp(min=0, max=K - 1)
        return local_vocab_ids.index_select(0, safe)
    else:
        # per-example local vocab [B,K] – use first example for debugging
        K = int(local_vocab_ids.size(1))
        safe = local_ids_1d.clamp(min=0, max=K - 1)
        return local_vocab_ids[0].index_select(0, safe)


def debug_hieratt_batch(batch: dict, vocab, cfg: dict, *, title: str = "DEBUG", max_tokens: int = 60) -> None:
    """
    Prints:
      - key tensor shapes
      - UNK rates in src_ids / dec_in
      - local vocab stats (K, unique count, how many <unk>)
      - decoded snippets for src / teacher-forced tgt / LVT-mapped tgt
    """
    print("\n" + "=" * 90)
    print(f"[{title}] artifacts_dir={cfg.get('artifacts_dir')} vocab_size_cfg={cfg.get('vocab_size')} lowercase={cfg.get('lowercase')}")
    print(f"[{title}] vocab.size={getattr(vocab, 'size', None)} bos_id={getattr(vocab, 'bos_id', None)} eos_id={getattr(vocab, 'eos_id', None)} pad_id={getattr(vocab, 'pad_id', None)}")

    # basic batch tensor inventory
    for k, v in batch.items():
        if torch.is_tensor(v):
            shape = tuple(v.shape)
            print(f"[{title}] batch[{k}]: shape={shape} dtype={v.dtype} device={v.device}")

    # UNK rates
    if "src_ids" in batch and torch.is_tensor(batch["src_ids"]):
        r = _estimate_unk_rate(vocab, batch["src_ids"])
        print(f"[{title}] approx UNK rate src_ids: {r:.3f}")

    if "dec_in" in batch and torch.is_tensor(batch["dec_in"]):
        r = _estimate_unk_rate(vocab, batch["dec_in"])
        print(f"[{title}] approx UNK rate dec_in (teacher-forced input): {r:.3f}")

    # local vocab stats
    local_vocab_ids = batch.get("local_vocab_ids", None)
    if torch.is_tensor(local_vocab_ids):
        if local_vocab_ids.dim() == 1:
            K = int(local_vocab_ids.numel())
            ids_list = local_vocab_ids.tolist()
        else:
            K = int(local_vocab_ids.size(1))
            ids_list = local_vocab_ids[0].tolist()

        uniq = len(set(ids_list))
        unk_cnt = sum(1 for x in ids_list if _is_unk(vocab, x))
        print(f"[{title}] local_vocab_ids: K={K} unique={uniq} (<unk> count in local vocab={unk_cnt})")

        preview = " ".join(_tok(vocab, x) for x in ids_list[:30])
        print(f"[{title}] local_vocab preview (first 30): {preview}")

        # check special tokens presence (by id, not by string)
        for name in ("bos_id", "eos_id", "pad_id"):
            sid = getattr(vocab, name, None)
            if sid is not None:
                present = sid in set(ids_list)
                print(f"[{title}] local_vocab contains {name}={sid} ({_tok(vocab, sid)}): {present}")

    # decode snippets
    def _decode_global_ids(global_ids_1d: torch.Tensor) -> str:
        out = []
        for x in global_ids_1d.tolist():
            if getattr(vocab, "eos_id", None) is not None and int(x) == int(vocab.eos_id):
                break
            t = _tok(vocab, x)
            if t not in ("<s>", "</s>", "<pad>"):
                out.append(t)
            if len(out) >= max_tokens:
                break
        return " ".join(out)

    if "src_ids" in batch and torch.is_tensor(batch["src_ids"]):
        s = _decode_global_ids(batch["src_ids"][0])
        print(f"[{title}] decoded src_ids[0]: {s}")

    if "dec_in" in batch and torch.is_tensor(batch["dec_in"]):
        s = _decode_global_ids(batch["dec_in"][0])
        print(f"[{title}] decoded dec_in[0]: {s}")

    # If you have global target ids:
    if "dec_tgt" in batch and torch.is_tensor(batch["dec_tgt"]):
        s = _decode_global_ids(batch["dec_tgt"][0])
        print(f"[{title}] decoded dec_tgt (global)[0]: {s}")
        r = _estimate_unk_rate(vocab, batch["dec_tgt"])
        print(f"[{title}] approx UNK rate dec_tgt (global): {r:.3f}")

    # If you only have local target ids, map them through local vocab:
    if "dec_tgt_local" in batch and torch.is_tensor(batch["dec_tgt_local"]) and torch.is_tensor(local_vocab_ids):
        local_1d = batch["dec_tgt_local"][0]
        # range sanity
        max_local = int(local_1d.max().item()) if local_1d.numel() else -1
        K = int(local_vocab_ids.numel()) if local_vocab_ids.dim() == 1 else int(local_vocab_ids.size(1))
        print(f"[{title}] dec_tgt_local range check: max_local={max_local} vs K={K} (should be < K)")

        mapped = _local_to_global(batch, local_1d, local_vocab_ids)
        s = _decode_global_ids(mapped)
        print(f"[{title}] decoded dec_tgt_local mapped->global[0]: {s}")

        r = _estimate_unk_rate(vocab, mapped)
        print(f"[{title}] approx UNK rate (mapped targets): {r:.3f}")

    print("=" * 90 + "\n")


def env_debug_enabled() -> bool:
    return os.environ.get("DEBUG_HIERATT", "0") == "1"
