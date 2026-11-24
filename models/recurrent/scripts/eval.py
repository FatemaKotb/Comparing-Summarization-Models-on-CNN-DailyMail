import os
import yaml
from tqdm import tqdm
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial

from src.data.vocab import load_vocab
from src.data.batcher import PGNDataset, collate_fn
from src.model.pgn import PGN
from src.utils.device import get_device
from src.utils.checkpoint import load_ckpt
from src.utils.rouge import compute_rouge


def ids_to_text(ids, vocab, oovs):
    """Convert predicted token ids (incl. extended-vocab ids) into a readable string."""
    tokens = []
    for i in ids:
        if i == vocab.eos_id:
            break
        if i < vocab.size:
            tok = vocab.decode(i)
        else:
            # Extended-vocab token: map back to per-example OOV list
            oov_idx = i - vocab.size
            tok = oovs[oov_idx] if 0 <= oov_idx < len(oovs) else "<oov>"
        # Drop special tokens from final text
        if tok not in ("<s>", "</s>", "<pad>"):
            tokens.append(tok)
    return " ".join(tokens)


def length_penalty(length: int, alpha: float) -> float:
    """Google NMT length penalty: ((5 + len) / 6)^alpha."""
    return ((5.0 + float(length)) / 6.0) ** alpha


@torch.no_grad()
def beam_decode_single(
    model,
    src_ids, src_ext_ids, src_lens, src_mask,
    vocab,
    max_len: int,
    beam_size: int = 4,
    alpha: float = 0.6,
    min_len: int = 35,
):
    """
    Beam search for ONE example (batch size = 1).
    Supports pointer-generator extended vocabulary and optional coverage.
    Returns: List[int] token ids (may include extended ids).
    """
    device = src_ids.device
    
    # Encode once; decoding will attend to enc_out each step
    enc_out, dec_state, coverage = model.encode(src_ids, src_lens, src_mask)
    
    # Infer max_oov for this example from src_ext_ids (how far ext ids go beyond vocab)
    max_oov = int(torch.max(src_ext_ids).item() - (vocab.size - 1)) if torch.max(src_ext_ids).item() >= vocab.size else 0
    
    # Beam item = (tokens_so_far, sum_logp, decoder_state, coverage, ended_flag)
    beams = [([vocab.bos_id], 0.0, dec_state, coverage, False)]
    completed = []

    for t in range(max_len):
        new_beams = []

        for tokens, sum_logp, state, cov, ended in beams:
            if ended:
                # keep ended beams
                new_beams.append((tokens, sum_logp, state, cov, True))
                continue
            
            # Previous emitted token id (batch size = 1)
            y_prev = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
            
            # One decoding step: returns log-probs over extended vocab
            log_probs, next_state, next_cov = model.decode_step(
                y_prev=y_prev,
                dec_state=state,
                enc_out=enc_out,
                src_mask=src_mask,
                src_ext_ids=src_ext_ids,
                max_oov=max_oov,
                coverage=cov,
            )  # [1, extV]

            lp = log_probs.squeeze(0)  # [extV]

            # Prevent early EOS to avoid too-short summaries
            cur_len = len(tokens) - 1
            if cur_len < int(min_len):
                lp[vocab.eos_id] = -1e9
            
            # Expand current beam by top-k candidates
            topk_logp, topk_ids = torch.topk(lp, k=beam_size)

            for k in range(beam_size):
                tok_id = int(topk_ids[k].item())
                tok_logp = float(topk_logp[k].item())

                new_tokens = tokens + [tok_id]
                new_sum = sum_logp + tok_logp
                new_ended = (tok_id == vocab.eos_id)

                # Clone state/coverage to avoid reference sharing across beam branches
                h, c = next_state
                cloned_state = (h.clone(), c.clone())
                cloned_cov = next_cov.clone() if next_cov is not None else None

                new_beams.append((new_tokens, new_sum, cloned_state, cloned_cov, new_ended))

        # Rank beams using length-penalized score
        def score_fn(b):
            tokens, sum_logp, *_ = b
            L = max(1, len(tokens) - 1) # exclude BOS
            return sum_logp / length_penalty(L, alpha)
        
        new_beams.sort(key=score_fn, reverse=True)
        beams = new_beams[:beam_size]

        # Track completed beams; stop early if all beams ended
        completed = [b for b in beams if b[4]]
        if len(completed) == beam_size:
            break

    # Choose best completed beam if any, otherwise best alive beam
    final_pool = completed if len(completed) > 0 else beams
    final_pool.sort(key=lambda b: b[1] / length_penalty(max(1, len(b[0]) - 1), alpha), reverse=True)
    best_tokens = final_pool[0][0]

    # Drop BOS before returning
    return best_tokens[1:]


@torch.no_grad()
def beam_decode_batch(model, batch, vocab, cfg):
    """
    Beam decode each example independently (slower, but simplest + correct).
    Returns: List[List[int]] length = batch_size
    """
    beam_size = int(cfg.get("beam_size", 4))
    alpha = float(cfg.get("len_penalty", 0.6))
    min_len = int(cfg.get("min_dec_len", 35))
    max_len = int(cfg.get("max_dec_len", 120))

    src_ids = batch["src_ids"]
    src_ext_ids = batch["src_ext_ids"]
    src_lens = batch["src_lens"]
    src_mask = batch["src_mask"]
    
    B = src_ids.size(0)
    outputs = []
    for i in range(B):
        out_ids = beam_decode_single(
            model,
            src_ids=src_ids[i:i+1],
            src_ext_ids=src_ext_ids[i:i+1],
            src_lens=src_lens[i:i+1],
            src_mask=src_mask[i:i+1],
            vocab=vocab,
            max_len=max_len,
            beam_size=beam_size,
            alpha=alpha,
            min_len=min_len,
        )
        outputs.append(out_ids)
    return outputs


def main():
    # Load eval config and choose device (cuda if available)
    cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))
    device = get_device()
    print("Device:", device)
    
    # Load saved vocabulary
    vocab = load_vocab(os.path.join(cfg["artifacts_dir"], "vocab.json"))
    
    # Load dataset and wrap validation split with PGN dataset/batching (adds extended vocab fields)
    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"])
    val_set = PGNDataset(
        ds["validation"],
        vocab,
        src_col="article",
        tgt_col="highlights",
        lowercase=cfg["lowercase"],
        max_src_len=cfg["max_src_len"],
        max_tgt_len=cfg["max_tgt_len"],
    )
    
    # Collate pads sequences and builds tensors (incl. src_ext_ids/oovs for pointer-generator)
    collate = partial(collate_fn, pad_id=vocab.pad_id)
    
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg["num_workers"] > 0),
    )
    
    # Recreate model and load best checkpoint
    model = PGN(
        vocab_size=vocab.size,
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        pad_id=vocab.pad_id,
        unk_id=vocab.unk_id,
        use_coverage=bool(cfg.get("use_coverage", True)),
    ).to(device)
    
    ckpt_path = os.path.join(cfg["ckpt_dir"], "pgn_best.pt")
    load_ckpt(ckpt_path, model)
    model.eval()

    preds, refs = [], []
    printed = False

    for batch in tqdm(val_loader, desc="eval"):
        # Keep Python OOV lists on CPU; only tensors go to GPU/CPU device
        oovs_list = batch["oovs"]
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        
        # Predict with beam search and decode gold targets for ROUGE
        pred_ids = beam_decode_batch(model, batch, vocab, cfg)
        gold_ids = batch["dec_tgt"].cpu().tolist()
        
        # Print a small qualitative sample once
        if not printed:
            for j in range(min(3, len(pred_ids))):
                print("\nPRED:", ids_to_text(pred_ids[j], vocab, oovs_list[j])[:400])
                print("REF :", ids_to_text(gold_ids[j], vocab, oovs_list[j])[:400])
            printed = True
        
        # Accumulate strings for corpus-level ROUGE
        for i in range(len(pred_ids)):
            preds.append(ids_to_text(pred_ids[i], vocab, oovs_list[i]))
            refs.append(ids_to_text(gold_ids[i], vocab, oovs_list[i]))
    # Compute ROUGE over the whole validation set
    scores = compute_rouge(preds, refs)
    print(scores)


if __name__ == "__main__":
    main()
