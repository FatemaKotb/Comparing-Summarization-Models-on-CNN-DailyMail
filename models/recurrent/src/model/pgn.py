from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        # src_ids: [B,S]
        x = self.dropout(self.emb(src_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_out_packed, (h, c) = self.lstm(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True, total_length=src_ids.size(1))
        # enc_out: [B,S,2H]
        return enc_out, (h, c)


class BahdanauAttention(nn.Module):
    """
    Coverage-aware Bahdanau attention.
    If coverage is None, behaves like standard attention.
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.W_c = nn.Linear(1, attn_dim, bias=False)  # coverage
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out, dec_state, src_mask, coverage: Optional[torch.Tensor] = None):
        """
        enc_out:  [B,S,enc_dim]
        dec_state:[B,dec_dim]
        src_mask: [B,S] float (1 valid, 0 pad)
        coverage: [B,S] float or None
        """
        features = self.W_h(enc_out) + self.W_s(dec_state).unsqueeze(1)  # [B,S,A]
        if coverage is not None:
            features = features + self.W_c(coverage.unsqueeze(-1))  # [B,S,A]

        e = self.v(torch.tanh(features)).squeeze(-1)  # [B,S]
        e = e.masked_fill(src_mask == 0, -1e9)
        attn = F.softmax(e, dim=-1)  # [B,S]
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B,enc_dim]
        return ctx, attn


class PGN(nn.Module):
    """
    Pointer-Generator Network with optional coverage loss.

    Training:
      - forward(batch) returns log_probs OR (log_probs, cov_loss) depending on use_coverage

    Decoding:
      - encode(...) returns (enc_out, dec_state, coverage)
      - decode_step(..., coverage) returns (log_probs, dec_state, coverage)
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        pad_id: int,
        unk_id: int,
        use_coverage: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.use_coverage = use_coverage

        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, dropout, pad_id)

        # Decoder hidden is 2H to match bi-encoder output size
        self.dec_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dec_lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim * 2,
            num_layers=1,
            batch_first=True,
        )

        self.attn = BahdanauAttention(enc_dim=hidden_dim * 2, dec_dim=hidden_dim * 2, attn_dim=hidden_dim)

        # [decoder_state; context] -> vocab logits
        self.vocab_proj = nn.Linear(hidden_dim * 2 + hidden_dim * 2, vocab_size)

        # p_gen uses [context; decoder_state; decoder_input_emb]
        self.p_gen_lin = nn.Linear(hidden_dim * 2 + hidden_dim * 2 + emb_dim, 1)

        self.dropout = nn.Dropout(dropout)

        # Initialize decoder state from pooled encoder outputs
        self.init_h = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.init_c = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def _init_decoder_state(self, enc_out: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        enc_out:  [B,S,2H]
        src_mask: [B,S] float
        returns: (h0,c0) each [1,B,2H]
        """
        mask = src_mask.unsqueeze(-1)  # [B,S,1]
        pooled = (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B,2H]
        h0 = torch.tanh(self.init_h(pooled)).unsqueeze(0)  # [1,B,2H]
        c0 = torch.tanh(self.init_c(pooled)).unsqueeze(0)
        return (h0, c0)

    def forward(self, batch: dict):
        """
        batch fields used:
          src_ids      [B,S]
          src_ext_ids  [B,S]
          src_lens     [B]
          src_mask     [B,S]
          dec_in       [B,T]
          max_oov      int

        Returns:
          if use_coverage:
            (final_log_probs [B,T,ext], cov_loss scalar)
          else:
            final_log_probs [B,T,ext]
        """
        src_ids = batch["src_ids"]
        src_ext_ids = batch["src_ext_ids"].long()
        src_lens = batch["src_lens"]
        src_mask = batch["src_mask"]

        dec_in = batch["dec_in"]
        max_oov = int(batch["max_oov"])

        enc_out, _ = self.encoder(src_ids, src_lens)  # [B,S,2H]
        dec_state = self._init_decoder_state(enc_out, src_mask)  # (h,c) each [1,B,2H]

        B, T = dec_in.shape
        dec_emb = self.dropout(self.dec_emb(dec_in))  # [B,T,E]

        ext_size = self.vocab_size + max_oov
        final_dists = []

        coverage = torch.zeros_like(src_mask) if self.use_coverage else None  # [B,S]
        cov_losses = []  # list of [B]

        for t in range(T):
            x_t = dec_emb[:, t:t + 1, :]  # [B,1,E]
            dec_out, dec_state = self.dec_lstm(x_t, dec_state)  # [B,1,2H]
            s_t = dec_out.squeeze(1)  # [B,2H]

            ctx_t, attn_t = self.attn(enc_out, s_t, src_mask, coverage=coverage)  # [B,2H], [B,S]

            if self.use_coverage:
                # covloss_t = sum_i min(attn_i, coverage_i)
                cov_losses.append(torch.sum(torch.min(attn_t, coverage), dim=1))  # [B]
                coverage = coverage + attn_t

            # generator vocab dist
            vocab_logits = self.vocab_proj(torch.cat([s_t, ctx_t], dim=-1))  # [B,V]
            p_vocab = F.softmax(vocab_logits, dim=-1)  # [B,V]

            # p_gen
            p_gen = torch.sigmoid(self.p_gen_lin(torch.cat([ctx_t, s_t, x_t.squeeze(1)], dim=-1)))  # [B,1]

            # extended vocab
            p_vocab_ext = torch.zeros(B, ext_size, device=src_ids.device)
            p_vocab_ext[:, : self.vocab_size] = p_vocab

            # copy dist -> scatter add attention weights into ext ids
            p_copy_ext = torch.zeros(B, ext_size, device=src_ids.device)
            p_copy_ext.scatter_add_(dim=1, index=src_ext_ids, src=attn_t)

            final = p_gen * p_vocab_ext + (1.0 - p_gen) * p_copy_ext
            final = final.clamp_min(1e-12)
            final_dists.append(final.unsqueeze(1))  # [B,1,ext]

        final_dists = torch.cat(final_dists, dim=1)  # [B,T,ext]
        log_probs = torch.log(final_dists)

        if not self.use_coverage:
            return log_probs

        # Mean over time then batch
        cov_loss = torch.stack(cov_losses, dim=1).mean() if len(cov_losses) > 0 else torch.tensor(0.0, device=src_ids.device)
        return log_probs, cov_loss

    # ---------- Autoregressive helpers ----------

    def encode(self, src_ids: torch.Tensor, src_lens: torch.Tensor, src_mask: torch.Tensor):
        """
        Encode once for autoregressive decoding.
        Returns:
          enc_out   [B,S,2H]
          dec_state (h,c) each [1,B,2H]
          coverage  [B,S] or None
        """
        enc_out, _ = self.encoder(src_ids, src_lens)
        dec_state = self._init_decoder_state(enc_out, src_mask)
        coverage = torch.zeros_like(src_mask) if self.use_coverage else None
        return enc_out, dec_state, coverage

    def decode_step(
        self,
        y_prev: torch.Tensor,
        dec_state: Tuple[torch.Tensor, torch.Tensor],
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
        src_ext_ids: torch.Tensor,
        max_oov: int,
        coverage: Optional[torch.Tensor] = None,
    ):
        """
        One autoregressive step.

        y_prev: [B] previous token ids (may be extended ids >= vocab_size)
        Returns:
          log_probs: [B, vocab_size + max_oov]
          dec_state: updated (h,c)
          coverage:  updated [B,S] or None
        """
        B = y_prev.size(0)
        max_oov = int(max_oov)
        src_ext_ids = src_ext_ids.long()

        # Map extended ids to <unk> for embedding lookup
        y_prev_base = torch.where(
            y_prev < self.vocab_size,
            y_prev,
            torch.full_like(y_prev, self.unk_id),
        )

        x_t = self.dropout(self.dec_emb(y_prev_base)).unsqueeze(1)  # [B,1,E]
        dec_out, dec_state = self.dec_lstm(x_t, dec_state)          # [B,1,2H]
        s_t = dec_out.squeeze(1)                                    # [B,2H]

        ctx_t, attn_t = self.attn(enc_out, s_t, src_mask, coverage=coverage)  # [B,2H], [B,S]

        if self.use_coverage:
            if coverage is None:
                coverage = torch.zeros_like(src_mask)
            coverage = coverage + attn_t

        vocab_logits = self.vocab_proj(torch.cat([s_t, ctx_t], dim=-1))  # [B,V]
        p_vocab = F.softmax(vocab_logits, dim=-1)                        # [B,V]

        p_gen = torch.sigmoid(self.p_gen_lin(torch.cat([ctx_t, s_t, x_t.squeeze(1)], dim=-1)))  # [B,1]

        ext_size = self.vocab_size + max_oov

        p_vocab_ext = torch.zeros(B, ext_size, device=enc_out.device)
        p_vocab_ext[:, : self.vocab_size] = p_vocab

        p_copy_ext = torch.zeros(B, ext_size, device=enc_out.device)
        p_copy_ext.scatter_add_(dim=1, index=src_ext_ids, src=attn_t)

        final = p_gen * p_vocab_ext + (1.0 - p_gen) * p_copy_ext
        final = final.clamp_min(1e-12)
        log_probs = torch.log(final)
        return log_probs, dec_state, coverage
