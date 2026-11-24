import os
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial

from src.data.vocab import load_vocab
from src.data.batcher import PGNDataset, collate_fn
from src.model.pgn import PGN
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.checkpoint import save_ckpt, load_ckpt
from src.utils.rouge import compute_rouge


def masked_nll_loss(log_probs, targets, pad_id):
    # Compute token-level NLL loss while ignoring PAD tokens.
    # log_probs: [B, T, ExtV] (log-softmax over extended vocab), targets: [B, T]
    B, T, V = log_probs.shape
    flat_lp = log_probs.reshape(B * T, V)
    flat_t = targets.reshape(B * T)
    loss = F.nll_loss(flat_lp, flat_t, ignore_index=pad_id, reduction="mean")
    return loss


def ids_to_text(ids, vocab, oovs):
    # Convert a sequence of token IDs (may include extended-vocab IDs) back to text.
    tokens = []
    for i in ids:
        if i == vocab.eos_id:
            break
        if i < vocab.size:
            tok = vocab.decode(i)
        else:
            oov_idx = i - vocab.size
            tok = oovs[oov_idx] if 0 <= oov_idx < len(oovs) else "<oov>"
        if tok not in ("<s>", "</s>", "<pad>"):
            tokens.append(tok)
    return " ".join(tokens)


@torch.no_grad()
def greedy_decode_autoreg(model, batch, vocab, max_len, min_len=35):
    # Greedy autoregressive decoding for a whole batch (fast baseline, no beam search).
    src_ids = batch["src_ids"]
    src_ext_ids = batch["src_ext_ids"]
    src_lens = batch["src_lens"]
    src_mask = batch["src_mask"]
    max_oov = batch["max_oov"]  # per-batch max OOV count for extended vocab size

    enc_out, dec_state, coverage = model.encode(src_ids, src_lens, src_mask)

    B = src_ids.size(0)
    y_prev = torch.full((B,), vocab.bos_id, dtype=torch.long, device=src_ids.device)

    finished = torch.zeros(B, dtype=torch.bool, device=src_ids.device)
    outputs = []

    for _ in range(int(max_len)):
        log_probs, dec_state, coverage = model.decode_step(
            y_prev=y_prev,
            dec_state=dec_state,
            enc_out=enc_out,
            src_mask=src_mask,
            src_ext_ids=src_ext_ids,
            max_oov=max_oov,
            coverage=coverage,
        )
        
        # Prevent premature stopping before reaching min_len.
        if len(outputs) < int(min_len):
            log_probs[:, vocab.eos_id] = -1e9
        
        y_next = torch.argmax(log_probs, dim=-1)
        outputs.append(y_next)
        
        # For already-finished sequences, keep emitting EOS.
        y_prev = torch.where(finished, torch.full_like(y_next, vocab.eos_id), y_next)
        finished = finished | (y_next == vocab.eos_id)
        if bool(finished.all()):
            break

    if len(outputs) == 0:
        return [[] for _ in range(B)]

    return torch.stack(outputs, dim=1).cpu().tolist()


@torch.no_grad()
def run_validation(model, val_loader, vocab, cfg, device):
    # Run limited validation and compute ROUGE over decoded predictions vs references.
    model.eval()

    preds, refs = [], []

    max_val_batches = int(cfg.get("max_val_batches", 200))  # limit for speed
    min_len = int(cfg.get("min_dec_len", 35))

    for b_idx, batch in enumerate(tqdm(val_loader, desc="val", leave=False)):
        if b_idx >= max_val_batches:
            break

        oovs_list = batch["oovs"]  # keep list on CPU
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        pred_ids = greedy_decode_autoreg(model, batch, vocab, cfg["max_dec_len"], min_len=min_len)
        gold_ids = batch["dec_tgt"].cpu().tolist()

        for i in range(len(pred_ids)):
            preds.append(ids_to_text(pred_ids[i], vocab, oovs_list[i]))
            refs.append(ids_to_text(gold_ids[i], vocab, oovs_list[i]))

    scores = compute_rouge(preds, refs)
    model.train()
    return scores


def get_metric(scores: dict, metric_name: str):
    # scores like {"rouge1":..., "rouge2":..., "rougeL":...}
    if metric_name not in scores:
        raise KeyError(f"Unknown metric '{metric_name}'. Available: {list(scores.keys())}")
    return float(scores[metric_name])


def main():
    cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))
    set_seed(42)
    device = get_device()
    print("Device:", device)

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)

    vocab_path = os.path.join(cfg["artifacts_dir"], "vocab.json")
    vocab = load_vocab(vocab_path)

    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"])

    # Build PGN datasets that create both normal IDs and extended-vocab IDs (+ OOV lists).
    train_set = PGNDataset(
        ds["train"],
        vocab,
        src_col="article",
        tgt_col="highlights",
        lowercase=cfg["lowercase"],
        max_src_len=cfg["max_src_len"],
        max_tgt_len=cfg["max_tgt_len"],
    )

    val_set = PGNDataset(
        ds["validation"],
        vocab,
        src_col="article",
        tgt_col="highlights",
        lowercase=cfg["lowercase"],
        max_src_len=cfg["max_src_len"],
        max_tgt_len=cfg["max_tgt_len"],
    )

    collate = partial(collate_fn, pad_id=vocab.pad_id)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg["num_workers"] > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg["num_workers"] > 0),
    )

    # Coverage settings
    use_coverage = bool(cfg.get("use_coverage", True))
    cov_lambda = float(cfg.get("cov_lambda", 1.0))

    model = PGN(
        vocab_size=vocab.size,
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        pad_id=vocab.pad_id,
        unk_id=vocab.unk_id,
        use_coverage=use_coverage,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ---- Scheduling / Early stopping config ----
    eval_every = int(cfg.get("eval_every", 2000))  # steps
    metric_name = str(cfg.get("early_stop_metric", "rouge2"))  # "rouge1" / "rouge2" / "rougeL"
    patience = int(cfg.get("early_stop_patience", 3))  # number of evals without improvement
    min_delta = float(cfg.get("early_stop_min_delta", 1e-4))

    # LR decay on plateau (maximize ROUGE => use mode="max")
    lr_factor = float(cfg.get("lr_factor", 0.5))
    lr_patience = int(cfg.get("lr_patience", 1))
    min_lr = float(cfg.get("min_lr", 1e-6))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=lr_factor, patience=lr_patience, min_lr=min_lr, verbose=True
    )

    # ---- Auto-resume ----
    latest_ckpt = os.path.join(cfg["ckpt_dir"], "pgn_latest.pt")
    best_ckpt = os.path.join(cfg["ckpt_dir"], "pgn_best.pt")

    step = 0
    best_metric = float("-inf")
    bad_evals = 0

    if os.path.exists(latest_ckpt):
        print(f"Resuming from: {latest_ckpt}")
        obj = load_ckpt(latest_ckpt, model, optim)
        if isinstance(obj, dict):
            step = int(obj.get("step", 0))
            best_metric = float(obj.get("best_metric", best_metric))
            bad_evals = int(obj.get("bad_evals", bad_evals))
        print(f"Resumed at step={step} | best_{metric_name}={best_metric:.6f} | bad_evals={bad_evals}")

    model.train()

    for epoch in range(cfg["epochs"]):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg['epochs']}")
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            out = model(batch)  # forward: returns log_probs and optionally cov_loss
            if isinstance(out, tuple):
                log_probs, cov_loss = out
            else:
                log_probs, cov_loss = out, None
            
            nll = masked_nll_loss(log_probs, batch["dec_tgt"], pad_id=vocab.pad_id)
            
            # Total loss = NLL + coverage penalty (if enabled and provided).
            if use_coverage and cov_loss is not None:
                loss = nll + cov_lambda * cov_loss
                pbar.set_postfix(
                    {"loss": float(loss.item()), "nll": float(nll.item()), "cov": float(cov_loss.item()), "step": step, "lr": optim.param_groups[0]["lr"]}
                )
            else:
                loss = nll
                pbar.set_postfix({"loss": float(loss.item()), "step": step, "lr": optim.param_groups[0]["lr"]})

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optim.step()

            step += 1

            # ---- periodic eval + best saving + early stopping + lr decay ----
            if step % eval_every == 0:
                scores = run_validation(model, val_loader, vocab, cfg, device)
                cur = get_metric(scores, metric_name)
                print(f"\n[step={step}] val scores: {scores} | {metric_name}={cur:.6f}")

                # scheduler step (maximize)
                scheduler.step(cur)

                improved = (cur > best_metric + min_delta)
                if improved:
                    best_metric = cur
                    bad_evals = 0
                    # save best checkpoint
                    save_ckpt(best_ckpt, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
                    print(f"[step={step}] ✅ New best {metric_name}={best_metric:.6f} -> saved {best_ckpt}")
                else:
                    bad_evals += 1
                    print(f"[step={step}] no improvement. bad_evals={bad_evals}/{patience}")

                # always save latest
                save_ckpt(latest_ckpt, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})

                # early stop
                if bad_evals >= patience:
                    print(f"\n⛔ Early stopping: {patience} evals without improvement in {metric_name}.")
                    final_path = os.path.join(cfg["ckpt_dir"], "pgn_final.pt")
                    save_ckpt(final_path, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
                    return

            # keep your step checkpoints too
            if step % 2000 == 0:
                ckpt_path = os.path.join(cfg["ckpt_dir"], f"pgn_step{step}.pt")
                save_ckpt(ckpt_path, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
                save_ckpt(latest_ckpt, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})

    # Save final state after all epochs.
    final_path = os.path.join(cfg["ckpt_dir"], "pgn_final.pt")
    save_ckpt(final_path, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
    save_ckpt(latest_ckpt, model, optim, step, {**cfg, "best_metric": best_metric, "bad_evals": bad_evals})
    print("Saved:", final_path)


if __name__ == "__main__":
    main()
