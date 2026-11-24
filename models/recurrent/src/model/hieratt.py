"""Hierarchical attention seq2seq components used by the PGN experiments."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Standard additive attention layer operating over encoder states."""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        dec_state: torch.Tensor,    # [B,D]
        enc_states: torch.Tensor,   # [B,T,E]
        enc_mask: torch.Tensor,     # [B,T] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context and attention weights given decoder and encoder states.

        Args:
            dec_state: Decoder hidden state of shape [batch, dec_dim].
            enc_states: Encoder representations of shape [batch, time, enc_dim].
            enc_mask: Boolean mask indicating valid encoder positions.

        Returns:
            ctx: Weighted encoder context vector [batch, enc_dim].
            attn: Attention distribution over encoder steps [batch, time].
        """
        # energy for each encoder step conditioned on current decoder state
        scores = self.v(torch.tanh(self.W_enc(enc_states) + self.W_dec(dec_state).unsqueeze(1))).squeeze(-1)  # [B,T]
        # remove padded positions so they do not receive probability mass
        scores = scores.masked_fill(~enc_mask.bool(), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)  # [B,T]
        # context is the expectation of encoder states under the attention
        ctx = torch.bmm(attn.unsqueeze(1), enc_states).squeeze(1)  # [B,E]
        return ctx, attn


class HierEncoder(nn.Module):
    """Two-level encoder that builds sentence-aware representations."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        pad_id: int,
        pos_dim: int,
        max_pos: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)

        self.word_rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.sent_rnn = nn.GRU(2 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.pos_emb = nn.Embedding(max_pos, pos_dim)
        self.init_proj = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode token-level and sentence-level structure.

        Args:
            batch: Mini-batch dictionary containing keys ``src_ids``, ``src_lens``,
                ``src_mask``, ``sent_ids``, and ``sent_counts``.

        Returns:
            Dict with encoder caches (word states, sentence states with position
            info, masks, and initial decoder state).
        """
        # unpack the fields needed to perform hierarchical encoding
        src_ids = batch["src_ids"]
        src_lens = batch["src_lens"]
        src_mask = batch["src_mask"]
        sent_ids = batch["sent_ids"]
        sent_counts = batch["sent_counts"]

        # embed tokens and encode them with a bidirectional GRU
        x = self.drop(self.emb(src_ids))  # [B,T,E]
        packed = nn.utils.rnn.pack_padded_sequence(x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.word_rnn(packed)
        word_states, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B,T,2H]

        B, T, E2 = word_states.shape
        Smax = int(sent_counts.max().item())

        # map every word position to its sentence index and mask padded items
        word_sent_ids = sent_ids[:, :T].clamp(min=0, max=max(Smax - 1, 0))
        m = src_mask[:, :T].float()

        # accumulate word states and counts to derive average sentence vectors
        sent_sum = torch.zeros(B, Smax, E2, device=word_states.device, dtype=word_states.dtype)
        sent_cnt = torch.zeros(B, Smax, device=word_states.device, dtype=word_states.dtype)
        sent_sum.scatter_add_(1, word_sent_ids.unsqueeze(-1).expand(-1, -1, E2), word_states * m.unsqueeze(-1))
        sent_cnt.scatter_add_(1, word_sent_ids, m)
        sent_vecs = sent_sum / sent_cnt.clamp(min=1.0).unsqueeze(-1)  # [B,S,2H]

        # sentence mask tells us which slots correspond to real sentences
        sent_mask = torch.arange(Smax, device=sent_vecs.device).unsqueeze(0) < sent_counts.unsqueeze(1)  # [B,S]

        # encode the sentence sequence and keep both outputs and final states
        packed_s = nn.utils.rnn.pack_padded_sequence(sent_vecs, sent_counts.cpu(), batch_first=True, enforce_sorted=False)
        packed_s_out, h_n_s = self.sent_rnn(packed_s)
        sent_states, _ = nn.utils.rnn.pad_packed_sequence(packed_s_out, batch_first=True)  # [B,S,2H]

        # append learned absolute sentence position embeddings
        pos = torch.arange(sent_states.size(1), device=sent_states.device).unsqueeze(0).expand(B, -1)
        pos = pos.clamp(max=self.pos_emb.num_embeddings - 1)
        pos_vec = self.pos_emb(pos)
        sent_states_pos = torch.cat([sent_states, pos_vec], dim=-1)  # [B,S,2H+pos]

        # initial decoder state comes from concatenated final sentence RNN states
        init = torch.cat([h_n_s[0], h_n_s[1]], dim=-1)  # [B,2H]
        init = torch.tanh(self.init_proj(init))

        return {
            "word_states": word_states,              # [B,T,2H]
            "word_mask": src_mask[:, :T],            # [B,T]
            "sent_states_pos": sent_states_pos,      # [B,S,2H+pos]
            "sent_mask": sent_mask,                  # [B,S]
            "word_sent_ids": word_sent_ids,          # [B,T]
            "init_h": init,                          # [B,2H]
        }


class HierAttnSeq2Seq(nn.Module):
    """Hierarchical attention seq2seq with a local vocabulary table (LVT)."""
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        attn_dim: int,
        dropout: float,
        pad_id: int,
        lvt_k: int,
        pos_dim: int,
        max_pos: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.K = int(lvt_k)

        self.enc = HierEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            pad_id=pad_id,
            pos_dim=pos_dim,
            max_pos=max_pos,
            dropout=dropout,
        )

        dec_dim = 2 * hidden_dim
        self.dec_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dec_cell = nn.GRUCell(emb_dim + dec_dim, dec_dim)

        self.sent_attn = AdditiveAttention(enc_dim=dec_dim + pos_dim, dec_dim=dec_dim, attn_dim=attn_dim)
        self.word_attn = AdditiveAttention(enc_dim=dec_dim, dec_dim=dec_dim, attn_dim=attn_dim)

        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dec_dim + dec_dim, vocab_size)

    def encode(self, batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Run the hierarchical encoder and prepare the initial decoder states."""
        cache = self.enc(batch)
        # LVT: store the per-batch (or per-example) list of global vocab ids
        if "local_vocab_ids" in batch:
            cache["local_vocab_ids"] = batch["local_vocab_ids"].to(cache["word_states"].device)
        B = cache["init_h"].size(0)
        # decoder begins with encoder-derived hidden state and zero context
        ctx0 = torch.zeros(B, cache["word_states"].size(-1), device=cache["word_states"].device, dtype=cache["word_states"].dtype)
        dec_state = (cache["init_h"], ctx0)  # (h, ctx)
        return cache, dec_state

    def decode_step(
        self,
        y_prev: torch.LongTensor,                   # [B]
        dec_state: Tuple[torch.Tensor, torch.Tensor],
        enc_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One autoregressive decoding step over the local vocabulary.

        Args:
            y_prev: Previous decoder input tokens (global ids) [batch].
            dec_state: Tuple of current hidden state and context vector.
            enc_cache: Cached encoder outputs from :meth:`encode`.

        Returns:
            log_probs: Local vocabulary log-probabilities [batch, K].
            Tuple of the next decoder hidden state and context vector.
        """
        h, ctx = dec_state

        # embed the previous target token and run the GRUCell update
        emb = self.drop(self.dec_emb(y_prev))  # [B,E]
        h_next = self.dec_cell(torch.cat([emb, ctx], dim=-1), h)  # [B,2H]

        word_states = enc_cache["word_states"]
        word_mask = enc_cache["word_mask"]
        sent_states_pos = enc_cache["sent_states_pos"]
        sent_mask = enc_cache["sent_mask"]
        word_sent_ids = enc_cache["word_sent_ids"]

        # hierarchical attention first selects sentences, then redistributes over words
        _, beta = self.sent_attn(h_next, sent_states_pos, sent_mask)   # [B,S]
        _, alpha = self.word_attn(h_next, word_states, word_mask)      # [B,T]

        beta_for_words = beta.gather(1, word_sent_ids.clamp(min=0))    # [B,T]
        w = alpha * beta_for_words
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # context mixes word states using the combined sentence+word attention
        ctx_next = torch.bmm(w.unsqueeze(1), word_states).squeeze(1)   # [B,2H]

        # project the decoder state and context into scores over the *local* vocabulary
        # IMPORTANT: local-vocab indices must map to stable global vocab parameters.
        feats = torch.cat([h_next, ctx_next], dim=-1)  # [B,F]
        local_vocab_ids = enc_cache.get("local_vocab_ids", None)
        if local_vocab_ids is None:
            raise KeyError("Missing 'local_vocab_ids' in enc_cache. Provide batch['local_vocab_ids'] for LVT decoding.")
        local_vocab_ids = local_vocab_ids.to(feats.device)

        if local_vocab_ids.dim() == 1:
            # per-batch local vocab: [K]
            W_sub = self.out.weight.index_select(0, local_vocab_ids)  # [K,F]
            b_sub = self.out.bias.index_select(0, local_vocab_ids)    # [K]
            logits = F.linear(feats, W_sub, b_sub)                    # [B,K]
        else:
            # per-example local vocab: [B,K]
            # gather token-specific projection rows for each example
            W_sub = self.out.weight[local_vocab_ids]                  # [B,K,F]
            b_sub = self.out.bias[local_vocab_ids]                    # [B,K]
            logits = torch.bmm(W_sub, feats.unsqueeze(-1)).squeeze(-1) + b_sub  # [B,K]

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, (h_next, ctx_next)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Teacher-forced decoding producing log-probs for each target step."""
        enc_cache, dec_state = self.encode(batch)

        dec_in = batch["dec_in"]  # global ids (for embeddings)
        B, Tdec = dec_in.shape
        K = enc_cache.get("local_vocab_ids", None).size(-1) if enc_cache.get("local_vocab_ids", None) is not None else self.K
        log_probs_all = torch.zeros(B, Tdec, K, device=dec_in.device)

        # roll out teacher forcing one token at a time
        y_prev = dec_in[:, 0]
        for t in range(Tdec):
            lp, dec_state = self.decode_step(y_prev=y_prev, dec_state=dec_state, enc_cache=enc_cache)
            log_probs_all[:, t, :] = lp
            if t + 1 < Tdec:
                y_prev = dec_in[:, t + 1]

        return log_probs_all
