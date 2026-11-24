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
from src.utils.seed import set_seed
from src.utils.checkpoint import save_ckpt, load_ckpt
from src.utils.rouge import compute_rouge
from src.utils.logging import should_log, fmt_cuda_mem
from src.utils.diag_hieratt import debug_hieratt_batch, env_debug_enabled


def masked_nll_loss(log_probs: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Mean NLL over non-PAD tokens. log_probs: [B,T,V], targets: [B,T]."""
    B, T, V = log_probs.shape
    lp = log_probs.reshape(B * T, V)
    tgt = targets.reshape(B * T)
    nll = F.nll_loss(lp, tgt, reduction="none", ignore_index=pad_id)
    # nll_loss returns 0 for ignored; compute mean over non-pad
    mask = (tgt != pad_id).float()
    denom = mask.sum().clamp(min=1.0)
    return (nll * mask).sum() / denom


def ids_to_text(ids, vocab) -> str:
    toks = []
    for i in ids:
        if i == vocab.eos_id:
            break
        tok = vocab.decode(int(i))
        if tok not in ("<s>", "</s>", "<pad>"):
            toks.append(tok)
    return " ".join(toks)


@torch.no_grad()
def greedy_decode_autoreg(model, batch, vocab, max_len: int, min_len: int = 35) -> str:
    """Greedy autoregressive decode for batch size = 1."""
    device = next(model.parameters()).device

    enc_cache, dec_state = model.encode(batch)
    local_vocab_ids = batch["local_vocab_ids"].to(device)  # [K]

    tokens = [vocab.bos_id]
    y_prev = torch.tensor([vocab.bos_id], device=device, dtype=torch.long)

    for t in range(int(max_len)):
        log_probs, dec_state = model.decode_step(y_prev=y_prev, dec_state=dec_state, enc_cache=enc_cache)
        lp = log_probs.squeeze(0)  # [K]

        # prevent early EOS
        if len(tokens) - 1 < int(min_len):
            eos_local = (local_vocab_ids == vocab.eos_id).nonzero(as_tuple=False)
            if eos_local.numel() > 0:
                lp[int(eos_local[0].item())] = -1e9

        y_local = int(torch.argmax(lp).item())
        y_global = int(local_vocab_ids[y_local].item())

        tokens.append(y_global)
        y_prev = torch.tensor([y_global], device=device, dtype=torch.long)

        if y_global == vocab.eos_id:
            break

    return ids_to_text(tokens[1:], vocab)  # drop BOS


@torch.no_grad()
def run_validation(model, val_loader, vocab, cfg, device):
    was_training = model.training
    model.eval()
    try:
        preds, refs = [], []
        max_batches = int(cfg.get("max_val_batches", 200))
        max_len = int(cfg.get("max_dec_len", 120))
        min_len = int(cfg.get("min_dec_len", 35))

        total = min(max_batches, len(val_loader)) if hasattr(val_loader, "__len__") else max_batches
        printed_sample = False
        for i, batch in enumerate(val_loader, start=1):
            if i > max_batches:
                break
            if should_log(i, total, every=50):
                print(f"[val]   {i}/{total}")

            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = greedy_decode_autoreg(model, batch, vocab, max_len=max_len, min_len=min_len)
            ref_ids = batch["dec_tgt"][0].tolist()
            ref = ids_to_text(ref_ids, vocab)
            preds.append(pred)
            refs.append(ref)

            # --- print ONE sample per validation run (for debugging) ---
            if not printed_sample:
                src_ids = batch["src_ids"][0].tolist()
                article = ids_to_text(src_ids, vocab)

                # (optional) truncate article so logs aren't huge
                article_preview = " ".join(article.split()[:120])

                print("\n[sample] article:", article_preview)
                print("[sample] ref:", ref)
                print("[sample] pred:", pred)
                print()
                printed_sample = True

        scores = compute_rouge(preds, refs)
        return scores
    finally:
        if was_training:
            model.train()

def get_metric(scores: dict, metric_name: str):
    if metric_name not in scores:
        raise KeyError(f"Unknown metric '{metric_name}'. Available: {list(scores.keys())}")
    return float(scores[metric_name])


def main():
    cfg = yaml.safe_load(open("configs/hieratt.yaml", "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    print("Device:", device)

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)

    vocab = load_vocab(os.path.join(cfg["artifacts_dir"], "vocab.json"))

    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"])

    common_pool = build_common_pool(
        ds["train"],
        vocab,
        tgt_col="highlights",
        lowercase=bool(cfg["lowercase"]),
        max_samples=int(cfg.get("common_pool_max_samples", 20000)),
    )

    train_set = HierAttDataset(
        ds["train"],
        vocab,
        src_col="article",
        tgt_col="highlights",
        lowercase=bool(cfg["lowercase"]),
        max_src_len=int(cfg["max_src_len"]),
        max_src_sents=int(cfg["max_src_sents"]),
        max_tgt_len=int(cfg["max_tgt_len"]),
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

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        collate_fn=collate,
        pin_memory=True,
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

    optim = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    metric_name = str(cfg.get("early_stop_metric", "rouge2"))
    patience = int(cfg.get("early_stop_patience", 3))
    min_delta = float(cfg.get("early_stop_min_delta", 1e-4))
    eval_every = int(cfg.get("eval_every", 2000))

    sched_kwargs = dict(
    mode="max",
    factor=float(cfg.get("lr_factor", 0.5)),
    patience=int(cfg.get("lr_patience", 1)),
    min_lr=float(cfg.get("min_lr", 1e-6)),
)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **sched_kwargs, verbose=True)
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **sched_kwargs)


    latest_ckpt = os.path.join(cfg["ckpt_dir"], "hieratt_latest.pt")
    best_ckpt = os.path.join(cfg["ckpt_dir"], "hieratt_best.pt")

    step = 0
    best_metric = float("-inf")
    bad_evals = 0

    # ---- Auto-resume ----
    if os.path.exists(latest_ckpt):
        print(f"Resuming from: {latest_ckpt}")
        obj = load_ckpt(latest_ckpt, model, optim)
        if isinstance(obj, dict):
            step = int(obj.get("step", 0))
            best_metric = float(obj.get("best_metric", best_metric))
            bad_evals = int(obj.get("bad_evals", bad_evals))
        print(f"Resumed at step={step} | best_{metric_name}={best_metric:.6f} | bad_evals={bad_evals}")

    model.train()

    for epoch in range(int(cfg["epochs"])):
        num_batches = len(train_loader)
        print(f"===== EPOCH {epoch+1}/{cfg['epochs']} | batches={num_batches} =====")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for b_idx, batch in enumerate(train_loader, start=1):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            if env_debug_enabled() and epoch == 0 and b_idx == 1 and step == 0:
                debug_hieratt_batch(batch, vocab, cfg, title="TRAIN/FIRST-BATCH", max_tokens=60)

            log_probs = model(batch)  # [B,T,K]
            nll = masked_nll_loss(log_probs, batch["dec_tgt_local"], pad_id=0)  # local PAD is always index 0

            optim.zero_grad(set_to_none=True)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optim.step()

            step += 1

            # progress log every 50 batches + final batch
            if should_log(b_idx, num_batches, every=50):
                lr = optim.param_groups[0]["lr"]
                print(f"[train] epoch {epoch+1}/{cfg['epochs']} | {b_idx}/{num_batches} | step={step} | loss={float(nll.item()):.4f} | lr={lr:.2e}")

            # memory log every 200 batches + final batch
            if should_log(b_idx, num_batches, every=200):
                print(f"[mem]   epoch {epoch+1}/{cfg['epochs']} | {b_idx}/{num_batches} | step={step} | {fmt_cuda_mem()}")

            # ---- periodic eval + best saving + early stopping + lr decay ----
            if step % eval_every == 0:
                scores = run_validation(model, val_loader, vocab, cfg, device)
                cur = get_metric(scores, metric_name)
                print(f"\n[step={step}] val scores: {scores} | {metric_name}={cur:.6f}")

                scheduler.step(cur)

                improved = (cur > best_metric + min_delta)
                if improved:
                    best_metric = cur
                    bad_evals = 0
                    save_ckpt(best_ckpt, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
                    print(f"[step={step}] ✅ New best {metric_name}={best_metric:.6f} -> saved {best_ckpt}")
                else:
                    bad_evals += 1
                    print(f"[step={step}] no improvement. bad_evals={bad_evals}/{patience}")

                save_ckpt(latest_ckpt, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})

                if bad_evals >= patience:
                    print(f"\n⛔ Early stopping: {patience} evals without improvement in {metric_name}.")
                    final_path = os.path.join(cfg["ckpt_dir"], "hieratt_final.pt")
                    save_ckpt(final_path, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
                    return

            # keep step checkpoints too (same as colleague)
            if step % 2000 == 0:
                ckpt_path = os.path.join(cfg["ckpt_dir"], f"hieratt_step{step}.pt")
                save_ckpt(ckpt_path, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
                save_ckpt(latest_ckpt, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})

    final_path = os.path.join(cfg["ckpt_dir"], "hieratt_final.pt")
    save_ckpt(final_path, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
    save_ckpt(latest_ckpt, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
    print("Saved:", final_path)


if __name__ == "__main__":
    main()
