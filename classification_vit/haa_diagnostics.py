"""
haa_diagnostics.py — Per-layer HAA telemetry and deep geometric diagnostics.

Two entry points:
  log_haa_epoch_metrics     — called every validation epoch; reads stored
                              telemetry fields from mha attributes (no extra pass).
  log_haa_deep_diagnostics  — called at epochs
                              {1,5,10,20,30,40,50,60,70,80,90,final}; registers
                              temporary hooks and runs one full val pass.
"""

import math
import warnings

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Streaming aggregators (copied verbatim from run_phase_0.py)
# ---------------------------------------------------------------------------

class WelfordAggregator:
    """Online mean and variance using Welford's parallel / batch combination."""

    def __init__(self) -> None:
        self.n:    int   = 0
        self.mean: float = 0.0
        self._M2:  float = 0.0

    def update(self, values: torch.Tensor) -> None:
        flat   = values.detach().float().cpu().reshape(-1)
        n_b    = flat.numel()
        if n_b == 0:
            return
        mean_b = flat.mean().item()
        var_b  = flat.var(unbiased=False).item() if n_b > 1 else 0.0
        n_combined = self.n + n_b
        delta  = mean_b - self.mean
        self.mean  = (self.n * self.mean + n_b * mean_b) / n_combined
        self._M2  += var_b * n_b + delta ** 2 * self.n * n_b / n_combined
        self.n     = n_combined

    @property
    def variance(self) -> float:
        return self._M2 / (self.n - 1) if self.n > 1 else 0.0


_HIST_BINS = 50
_HIST_MIN  = -1.0
_HIST_MAX  =  1.0


class HistogramAccumulator:
    """Fixed-grid histogram over [-1, 1] for KL divergence of Z distribution."""

    _EPS = 1e-10

    def __init__(self, n_bins: int = _HIST_BINS,
                 min_val: float = _HIST_MIN,
                 max_val: float = _HIST_MAX) -> None:
        self.n_bins  = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.counts  = torch.zeros(n_bins, dtype=torch.long)
        self._n_total: int   = 0
        self._sum_z:   float = 0.0

    def update(self, z_safe: torch.Tensor) -> None:
        flat = z_safe.detach().float().cpu().reshape(-1)
        h    = torch.histc(flat, bins=self.n_bins, min=self.min_val, max=self.max_val)
        self.counts   += h.long()
        self._n_total += flat.numel()
        self._sum_z   += flat.sum().item()

    @property
    def z_mean(self) -> float:
        return self._sum_z / self._n_total if self._n_total > 0 else 0.0

    def _prob_vector(self) -> torch.Tensor:
        p = self.counts.float() + self._EPS
        return p / p.sum()

    def kl_divergence(self) -> float:
        """KL(empirical ‖ Discrete-Uniform) with Laplace smoothing."""
        p = self._prob_vector()
        q = torch.full_like(p, 1.0 / self.n_bins)
        return (p * (p / q).log()).sum().item()


# ---------------------------------------------------------------------------
# Geometric formulas (copied verbatim from run_phase_0.py)
# ---------------------------------------------------------------------------

def compute_Z(q: torch.Tensor, k: torch.Tensor, K: float) -> torch.Tensor:
    """Angular signal Z for every query-key pair. Values in [-1, 1]."""
    q_time  = q[..., 0:1]
    q_space = q[..., 1:]
    k_time  = k[..., 0:1]
    k_space = k[..., 1:]
    sqrt_k  = math.sqrt(K)

    inner_QK  = (-q_time  @ k_time.transpose(-1, -2)
                 + q_space @ k_space.transpose(-1, -2))
    inner_OQ  = -sqrt_k * q_time
    inner_OK  = -sqrt_k * k_time.transpose(-1, -2)
    norm_sq_QK = (inner_QK.pow(2) / K - K).clamp_min(0.0)
    norm_sq_OQ = (inner_OQ.pow(2) / K - K).clamp_min(0.0)
    numer_Z    = inner_OK + (inner_QK * inner_OQ) / K
    denom = torch.sqrt(norm_sq_QK + 1e-12) * torch.sqrt(norm_sq_OQ + 1e-12)
    Z_raw = numer_Z / denom
    # A5 parity: symmetric in Q and K (must mirror transformer_blocks.py).
    mask_valid = ((norm_sq_QK > 1e-6) & (norm_sq_OQ > 1e-6)).expand_as(Z_raw)
    nan_mask = torch.isnan(Z_raw)
    return torch.where(mask_valid & ~nan_mask,
                       Z_raw.clamp(-1.0, 1.0),
                       torch.zeros_like(Z_raw))


def compute_B(q_time: torch.Tensor, K: float, beta: float) -> torch.Tensor:
    """Exact aperture surrogate B for each query position."""
    sqrt_k  = math.sqrt(K)
    c_tilde = torch.acosh((q_time / sqrt_k).clamp_min(1.0 + 1e-3))
    sinh_c  = torch.sinh(c_tilde)
    arg_B   = 1.0 - (beta / sinh_c).pow(2)
    return torch.sqrt(arg_B.clamp_min(0.0) + 1e-8)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _make_log_fn(writer, epoch: int):
    """Return a (key, value) -> None callable that writes to the best available logger."""
    try:
        import wandb
        if wandb.run is not None:
            return lambda key, val: wandb.log({key: val}, step=epoch)
    except ImportError:
        pass
    if writer is not None:
        return lambda key, val: writer.add_scalar(key, val, epoch)
    return lambda key, val: print(f"[epoch {epoch}] {key}: {val:.6f}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_haa_epoch_metrics(model, epoch: int, writer) -> None:
    """
    Read per-layer HAA telemetry stored during the last eval forward pass
    and emit to TensorBoard / wandb / stdout.

    Call this immediately after evaluate() while the model is still in eval mode.
    """
    from lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention

    base = model.module if hasattr(model, 'module') else model
    log_fn = _make_log_fn(writer, epoch)

    haa_mhas = sorted(
        [(mod.layer_idx, mod)
         for _, mod in base.named_modules()
         if isinstance(mod, LorentzMultiHeadAttention) and mod.use_haa],
        key=lambda x: x[0]
    )

    first_haa = haa_mhas[0][0] if haa_mhas else None
    last_haa  = haa_mhas[-1][0] if haa_mhas else None

    for l, mha in haa_mhas:
        log_fn(f"haa/layer_{l}/beta",             mha.haa_alpha)
        log_fn(f"haa/layer_{l}/tau",              mha.haa_tau)
        log_fn(f"haa/layer_{l}/lambda",           mha.haa_lambda)
        log_fn(f"haa/layer_{l}/cone_sparsity",    mha.haa_cone_sparsity)
        log_fn(f"haa/layer_{l}/z_mean",           mha.haa_mean_Z)
        log_fn(f"haa/layer_{l}/frac_near_origin", mha.haa_frac_near_origin)

        if l in (first_haa, last_haa) and mha._grad_norms:
            for param_name, norm_val in mha._grad_norms.items():
                log_fn(f"haa/layer_{l}/grad_norm_{param_name}", norm_val)
                if not (1e-3 <= norm_val <= 1.0):
                    warnings.warn(
                        f"[HAA layer {l}] grad_norm_{param_name}={norm_val:.3e} "
                        f"outside expected range [1e-3, 1e0]")

        # Step 11: in-attention alpha_q telemetry, if active.
        if getattr(mha, 'use_q_depth_mlp', False) and mha._last_alpha_q is not None:
            _a = mha._last_alpha_q.float()
            log_fn(f"haa/layer_{l}/alpha_q_mean", _a.mean().item())
            log_fn(f"haa/layer_{l}/alpha_q_std",  _a.std().item() if _a.numel() > 1 else 0.0)
            log_fn(f"haa/layer_{l}/alpha_q_min",  _a.min().item())
            log_fn(f"haa/layer_{l}/alpha_q_max",  _a.max().item())
            log_fn(f"haa/layer_{l}/alpha_q_grad_norm",
                   float(getattr(mha, '_last_alpha_q_grad', 0.0)))

        # P4: log all three Z-failure rates against the same denominator
        # (total Z elements seen this epoch) and reset every counter so
        # the throttled-print modulus does not diverge across epochs.
        elem_rate = mha._z_nan_element_count / max(1, mha._z_nan_element_total)
        log_fn(f"haa/layer_{l}/z_nan_element_rate", elem_rate)
        nearzero_qk_rate = mha._z_nearzero_qk_count / max(1, mha._z_nan_element_total)
        log_fn(f"haa/layer_{l}/z_nearzero_qk_rate", nearzero_qk_rate)
        origin_rate = mha._z_origin_count / max(1, mha._z_nan_element_total)
        log_fn(f"haa/layer_{l}/z_origin_rate", origin_rate)

        # Reset ALL counters per epoch — fixes the throttled-print divergence.
        mha._z_nan_batch_count = 0
        mha._z_nan_total_calls = 0
        mha._z_nan_element_count = 0
        mha._z_nan_element_total = 0
        mha._z_nearzero_qk_count = 0
        mha._z_origin_count = 0

        # ---- CLS-row-only telemetry (Phase-3-E) ----
        # CLS is token index 0 in the query dimension. These five metrics
        # isolate the row that determines classification, since global
        # metrics are dominated by the 98.5% patch-to-patch pairs.
        # Guarded against div-by-zero via clamp_min on the valid count.
        if (getattr(mha, '_last_Z', None) is not None
                and getattr(mha, '_last_valid_mask', None) is not None
                and getattr(mha, '_last_B', None) is not None):
            with torch.no_grad():
                _Z_cls = mha._last_Z[:, :, 0, :].detach()             # [B, h, n_k]
                _M_cls = mha._last_valid_mask[:, :, 0, :].detach()    # [B, h, n_k] bool
                _B_cls = mha._last_B[:, :, 0, :].detach().expand_as(_Z_cls)  # [B, h, n_k]

                _valid_f = _M_cls.float()
                _n_valid = _valid_f.sum(-1).clamp_min(2.0)            # [B, h] >=2.0
                _z_sum = (_Z_cls * _valid_f).sum(-1)                  # [B, h]
                _z_mean_cls = _z_sum / _n_valid                       # [B, h]
                _z_centered = (_Z_cls - _z_mean_cls.unsqueeze(-1)) * _valid_f
                _z_var_cls = (_z_centered.pow(2)).sum(-1) / _n_valid  # [B, h]
                _cone_cls = (((_B_cls + _Z_cls) <= 0).float() * _valid_f).sum(-1) / _n_valid  # [B, h]

                # Guard: if ALL pairs masked in a row, _n_valid was clamped to 2.0
                # but _z_sum is 0, giving a spurious zero. Mask those rows out of
                # the batch-mean by reweighting with a row-validity indicator.
                _row_has_valid = (_M_cls.any(dim=-1)).float()         # [B, h]
                _row_w = _row_has_valid.sum().clamp_min(1.0)

                _z_mean_cls_avg = (_z_mean_cls * _row_has_valid).sum() / _row_w
                _z_var_cls_avg  = (_z_var_cls  * _row_has_valid).sum() / _row_w
                _cone_cls_avg   = (_cone_cls   * _row_has_valid).sum() / _row_w

                log_fn(f"haa/layer_{l}/cls_row_zmean",         _z_mean_cls_avg.item())
                log_fn(f"haa/layer_{l}/cls_row_zvar",          _z_var_cls_avg.item())
                log_fn(f"haa/layer_{l}/cls_row_cone_sparsity", _cone_cls_avg.item())

                # Per-row attention entropy: CLS row vs patches.
                _score = getattr(mha, '_last_score', None)
                if _score is not None:
                    _s_cls   = _score[:, :, 0, :]       # [B, h, n_k]
                    _s_patch = _score[:, :, 1:, :]      # [B, h, n_q-1, n_k]
                    _H_cls   = -(_s_cls   * _s_cls.clamp_min(1e-12).log()).sum(-1).mean()
                    _H_patch = -(_s_patch * _s_patch.clamp_min(1e-12).log()).sum(-1).mean()
                    log_fn(f"haa/layer_{l}/cls_row_entropy",   _H_cls.item())
                    log_fn(f"haa/layer_{l}/patch_row_entropy", _H_patch.item())

    # Stage 2.1 Path B telemetry — CLS depth residual.
    # Lives on the inner ViT (base.encoder for ViTClassifier wrapper).
    cls_resid = getattr(base, 'cls_depth_residual', None)
    if cls_resid is None:
        cls_resid = getattr(getattr(base, 'encoder', None),
                            'cls_depth_residual', None)
    if cls_resid is not None and cls_resid._last_alpha is not None:
        a = cls_resid._last_alpha.float()
        log_fn("cls_residual/alpha_mean", a.mean().item())
        log_fn("cls_residual/alpha_std",  a.std().item() if a.numel() > 1 else 0.0)
        log_fn("cls_residual/alpha_min",  a.min().item())
        log_fn("cls_residual/alpha_max",  a.max().item())
        log_fn("cls_residual/alpha_grad_norm",
               float(getattr(cls_resid, '_last_alpha_grad', 0.0)))


def log_haa_deep_diagnostics(model, val_loader, device: str,
                              epoch: int, K: float, writer) -> None:
    """
    Register temporary hooks on all HAA layers, run one full val pass, and log:
      - sigma2_c_tilde  (radial depth variance)
      - z_mean_diag     (mean angular signal across all query-key pairs)
      - kl_z            (KL divergence of Z distribution vs uniform)
      - spatial_cv      (coefficient of variation of spatial norms)
      - attention_entropy (mean entropy of the attention distribution)

    This is an additional forward pass; model is put into eval mode internally.
    """
    from lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention

    base = model.module if hasattr(model, 'module') else model
    log_fn = _make_log_fn(writer, epoch)

    haa_mhas = sorted(
        [(mod.layer_idx, mod)
         for _, mod in base.named_modules()
         if isinstance(mod, LorentzMultiHeadAttention) and mod.use_haa],
        key=lambda x: x[0]
    )

    if not haa_mhas:
        return

    sqrt_K = math.sqrt(K)

    # Per-layer streaming accumulators
    welford_ctilde = {l: WelfordAggregator() for l, _ in haa_mhas}
    hist_z         = {l: HistogramAccumulator() for l, _ in haa_mhas}
    welford_spatial = {l: WelfordAggregator() for l, _ in haa_mhas}
    entropy_agg    = {l: WelfordAggregator() for l, _ in haa_mhas}

    q_buffer: dict = {}
    handles: list  = []

    for layer_idx, mha in haa_mhas:

        def make_q_hook(lidx: int):
            def q_hook(module, inp, out):
                q_buffer[lidx] = out.detach()
            return q_hook

        def make_k_hook(lidx: int, wc: WelfordAggregator,
                        hz: HistogramAccumulator, ws: WelfordAggregator):
            def k_hook(module, inp, out):
                q = q_buffer.pop(lidx, None)
                if q is None:
                    return
                k = out.detach()

                # Radial depth σ²_c̃ (M1)
                q_time = q[..., 0:1]
                c_tilde = torch.acosh((q_time / sqrt_K).clamp_min(1.0 + 1e-3))
                wc.update(c_tilde)

                # Angular signal Z distribution (M2)
                z_safe = compute_Z(q, k, K)
                hz.update(z_safe)

                # Spatial norm CV (M4)
                q_space = q[..., 1:]
                ws.update(q_space.norm(dim=-1))

            return k_hook

        def make_softmax_hook(lidx: int, ea: WelfordAggregator):
            def softmax_hook(module, inp, out):
                p = out.detach()
                # Mean entropy per query: H = -Σ p·log(p)
                H = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
                ea.update(H.unsqueeze(0))
            return softmax_hook

        h1 = mha.q.register_forward_hook(make_q_hook(layer_idx))
        h2 = mha.k.register_forward_hook(
            make_k_hook(layer_idx,
                        welford_ctilde[layer_idx],
                        hist_z[layer_idx],
                        welford_spatial[layer_idx]))
        h3 = mha.softmax.register_forward_hook(
            make_softmax_hook(layer_idx, entropy_agg[layer_idx]))
        handles.extend([h1, h2, h3])

    was_training = model.training
    model.eval()
    deep_metrics = {}
    try:
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                model(x)

        for layer_idx, _ in haa_mhas:
            wc = welford_ctilde[layer_idx]
            hz = hist_z[layer_idx]
            ws = welford_spatial[layer_idx]
            ea = entropy_agg[layer_idx]

            sigma2_c  = wc.variance
            z_mean    = hz.z_mean
            kl_z      = hz.kl_divergence()
            spatial_cv = (math.sqrt(ws.variance) / (abs(ws.mean) + 1e-8))
            attn_ent  = ea.mean

            log_fn(f"deep/layer_{layer_idx}/sigma2_c_tilde",    sigma2_c)
            log_fn(f"deep/layer_{layer_idx}/z_mean_diag",       z_mean)
            log_fn(f"deep/layer_{layer_idx}/kl_z",              kl_z)
            log_fn(f"deep/layer_{layer_idx}/spatial_cv",        spatial_cv)
            log_fn(f"deep/layer_{layer_idx}/attention_entropy", attn_ent)

            deep_metrics[f"deep_layer{layer_idx}_sigma2_c_tilde"]    = sigma2_c
            deep_metrics[f"deep_layer{layer_idx}_z_mean_diag"]       = z_mean
            deep_metrics[f"deep_layer{layer_idx}_kl_z"]              = kl_z
            deep_metrics[f"deep_layer{layer_idx}_spatial_cv"]        = spatial_cv
            deep_metrics[f"deep_layer{layer_idx}_attention_entropy"] = attn_ent
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return deep_metrics
