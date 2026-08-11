#!/usr/bin/env python3
"""
run_phase_0.py  —  Phase 0 geometric diagnostics for a pretrained HexFormer.

Zero-shot evaluation on the full CIFAR-100 validation set (10 000 images).
All tensor interception is done via PyTorch forward hooks registered from
this external script; no source file is modified.

Run from the HexFormer project root:
    python run_phase_0.py --checkpoint path/to/checkpoint.pth
"""

import argparse
import os
import sys
import json
import math
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as tvd
import matplotlib
matplotlib.use("Agg")   # headless; no display required
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path bootstrap  (mirrors classification_vit/train.py chdir/sys.path logic)
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.realpath(__file__))   # HexFormer project root
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "classification_vit"))

from utils.initialize import select_model                                      # noqa: E402
from lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention   # noqa: E402

# ===========================================================================
# Configuration and command-line interface
# ===========================================================================

VAL_BATCH_SIZE   = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the historical Phase-0 geometric diagnostics on a CIFAR-100 checkpoint.")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint produced by the training entry point. No checkpoint is bundled with the repository.")
    parser.add_argument(
        "--data-root", default="data",
        help="CIFAR-100 root. Relative paths are resolved from the current working directory.")
    parser.add_argument(
        "--output-dir", default=os.path.join("output", "phase0"),
        help="Directory for diagnostic JSON and PNG files. Relative paths are resolved from the current working directory.")
    parser.add_argument(
        "--device", default="cuda:0",
        help="CUDA device in cuda:N form. CPU execution is not supported.")
    args = parser.parse_args()

    if not args.device.startswith("cuda:"):
        parser.error("--device must be a CUDA device in cuda:N form; CPU execution is not supported.")
    try:
        device_idx = int(args.device.split(":", 1)[1])
    except (TypeError, ValueError):
        parser.error("--device must be a CUDA device in cuda:N form, for example cuda:0.")
    if device_idx < 0:
        parser.error("--device CUDA index must be non-negative.")

    return args

# Histogram grid for M2
HIST_BINS        = 50
HIST_MIN         = -1.0
HIST_MAX         =  1.0

# Bootstrap for M2 p-value
N_BOOTSTRAP      = 1_000
BOOTSTRAP_SEED   = 42
# Cap samples drawn per bootstrap iteration to stay tractable.
# The null KL is evaluated at the same sample count as the empirical data,
# but never more than this limit (KL converges well below 1 M samples).
BOOTSTRAP_N_CAP  = 100_000

# Compatibility thresholds
THR_SIGMA2       = 0.10   # M1: σ²_c̃  > THR_SIGMA2
THR_SPARSITY_LO  = 0.15   # M3: cone_sparsity in [LO, HI]
THR_SPARSITY_HI  = 0.35
THR_PVALUE       = 0.05   # M2: bootstrap p < THR_PVALUE  AND  z_mean < 0


# ===========================================================================
# Streaming aggregators
# ===========================================================================

class WelfordAggregator:
    """
    Online mean and variance using Welford's parallel / batch combination.

    Accepts tensors of any shape; they are flattened to scalars internally.
    After all updates, read .mean and .variance.

    Batch-combination formula (Chan et al.):
        δ  = mean_b - mean_a
        n  = n_a + n_b
        M2 = M2_a + M2_b + δ² · n_a · n_b / n
    """

    def __init__(self) -> None:
        self.n:     int   = 0
        self.mean:  float = 0.0
        self._M2:   float = 0.0   # accumulated sum of squared deviations

    def update(self, values: torch.Tensor) -> None:
        """Incorporate a new batch of values (any device / dtype / shape)."""
        flat  = values.detach().float().cpu().reshape(-1)
        n_b   = flat.numel()
        if n_b == 0:
            return

        mean_b = flat.mean().item()
        # Population variance of the incoming batch (biased, avoids n-1=0)
        var_b  = flat.var(unbiased=False).item() if n_b > 1 else 0.0

        n_combined = self.n + n_b
        delta      = mean_b - self.mean

        # Parallel Welford combination
        self.mean  = (self.n * self.mean + n_b * mean_b) / n_combined
        self._M2  += var_b * n_b + delta ** 2 * self.n * n_b / n_combined
        self.n     = n_combined

    @property
    def variance(self) -> float:
        """Sample variance (unbiased, denominator n-1)."""
        return self._M2 / (self.n - 1) if self.n > 1 else 0.0


class HistogramAccumulator:
    """
    Fixed-grid histogram over [HIST_MIN, HIST_MAX] for KL divergence (M2).

    Call .update(z_safe) per batch.
    After all batches call .kl_divergence() and .bootstrap_p_value(rng).

    Laplace smoothing (ε = 1e-10) is applied at query time, not during
    accumulation, so integer counts stay exact throughout.
    """

    _EPS = 1e-10

    def __init__(self,
                 n_bins:   int   = HIST_BINS,
                 min_val:  float = HIST_MIN,
                 max_val:  float = HIST_MAX) -> None:
        self.n_bins  = n_bins
        self.min_val = min_val
        self.max_val = max_val
        # Accumulate as long integers to avoid floating-point drift.
        self.counts  = torch.zeros(n_bins, dtype=torch.long)
        self._n_total: int   = 0
        self._sum_z:   float = 0.0   # running sum for empirical mean (M2)

    def update(self, z_safe: torch.Tensor) -> None:
        """
        Add Z values of any shape into the histogram.
        Values beyond [min_val, max_val] fall into the edge bins
        (torch.histc clamps implicitly).
        """
        flat = z_safe.detach().float().cpu().reshape(-1)
        h    = torch.histc(flat, bins=self.n_bins,
                           min=self.min_val, max=self.max_val)
        self.counts   += h.long()
        self._n_total += flat.numel()
        self._sum_z   += flat.sum().item()

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    @property
    def z_mean(self) -> float:
        return self._sum_z / self._n_total if self._n_total > 0 else 0.0

    def _prob_vector(self) -> torch.Tensor:
        """Laplace-smoothed probability vector from accumulated counts."""
        p = self.counts.float() + self._EPS
        return p / p.sum()

    def kl_divergence(self) -> float:
        """
        KL(empirical ‖ Discrete-Uniform) using Laplace-smoothed counts.

        D_KL(p ‖ q) = Σ p_i · log(p_i / q_i),   q_i = 1/n_bins  ∀ i.
        """
        p = self._prob_vector()
        q = torch.full_like(p, 1.0 / self.n_bins)
        return (p * (p / q).log()).sum().item()

    def bootstrap_p_value(self, rng: np.random.Generator) -> float:
        """
        One-sided bootstrap p-value against the null H₀: Z ~ Uniform[-1,1].

        Algorithm:
          1. Draw N_BOOTSTRAP null histograms, each from
             min(_n_total, BOOTSTRAP_N_CAP) Uniform[-1,1] samples.
          2. Compute KL of each null histogram vs uniform.
          3. p = fraction of null KLs >= empirical KL.

        Returns 1.0 if no Z values have been accumulated yet.
        """
        if self._n_total == 0:
            return 1.0

        empirical_kl = self.kl_divergence()
        n_samples    = min(self._n_total, BOOTSTRAP_N_CAP)
        null_kls     = np.empty(N_BOOTSTRAP, dtype=np.float64)

        for i in range(N_BOOTSTRAP):
            samples = rng.uniform(self.min_val, self.max_val,
                                  size=n_samples).astype(np.float32)
            t     = torch.from_numpy(samples)
            h     = torch.histc(t, bins=self.n_bins,
                                min=self.min_val, max=self.max_val)
            p_null = (h.float() + self._EPS)
            p_null = p_null / p_null.sum()
            q      = torch.full_like(p_null, 1.0 / self.n_bins)
            null_kls[i] = (p_null * (p_null / q).log()).sum().item()

        return float((null_kls >= empirical_kl).mean())


# ===========================================================================
# Placeholder geometric formulas  ← formulas to be provided in next message
# ===========================================================================

def compute_Z(q: torch.Tensor, k: torch.Tensor, K: float) -> torch.Tensor:
    """
    Compute the angular signal Z (cosine of the entailment angle) for
    every query–key pair in a single batch.

    Args:
        q:  Query tensor,  shape (batch, heads, seq_q, D).
            Lorentz feature vector; q[..., 0] is the time coordinate x₀,
            q[..., 1:] are the spatial components x_s.
        k:  Key tensor,    shape (batch, heads, seq_k, D).
        K:  Manifold curvature scalar (manifold.k.item(), positive float).

    Returns:
        z_safe: Tensor of shape (batch, heads, seq_q, seq_k),
                values in [-1, 1] (clamped via torch.clamp).

    Extracted verbatim from haa.py steps 1 and 3.
    """
    q_time  = q[..., 0:1]          # (B, H, Nq, 1)
    q_space = q[..., 1:]           # (B, H, Nq, head_dim)
    k_time  = k[..., 0:1]          # (B, H, Nk, 1)
    k_space = k[..., 1:]           # (B, H, Nk, head_dim)

    sqrt_k = math.sqrt(K)

    # Step 1: pairwise Lorentz inner product  (B, H, Nq, Nk)
    inner_QK = (- q_time  @ k_time.transpose(-1, -2)
                + q_space @ k_space.transpose(-1, -2))

    # Step 3: direct inner-product angle formulation
    inner_OQ = -sqrt_k * q_time                               # (B, H, Nq, 1)
    inner_OK = -sqrt_k * k_time.transpose(-1, -2)             # (B, H, 1,  Nk)

    norm_sq_QK = (inner_QK.pow(2) / K - K).clamp_min(0.0)    # (B, H, Nq, Nk)
    norm_sq_OQ = (inner_OQ.pow(2) / K - K).clamp_min(0.0)    # (B, H, Nq, 1)

    numer_Z = inner_OK + (inner_QK * inner_OQ) / K            # (B, H, Nq, Nk)

    EPS_Z   = 5e-3
    denom_Z = torch.sqrt((norm_sq_QK * norm_sq_OQ).clamp_min(EPS_Z ** 2))

    Z_raw  = numer_Z / denom_Z
    z_safe = Z_raw.clamp(-1.0, 1.0)
    return z_safe


def compute_B(q_time: torch.Tensor, K: float, beta: float) -> torch.Tensor:
    """
    Compute the exact aperture surrogate B for each query position.

    Args:
        q_time: Time coordinate of Q, shape (batch, heads, seq_q).
                Obtained as q[..., 0].
        K:      Manifold curvature scalar (manifold.k.item()).
        beta:   Aperture parameter (scalar float or matching tensor).

    Returns:
        B: Aperture surrogate tensor, shape (batch, heads, seq_q).

    Extracted verbatim from haa.py step 4.
    beta is passed as a plain positive float — do NOT apply softplus.
    Uses clamp_min(0.0) in place of F.relu to avoid importing F.
    """
    sqrt_k  = math.sqrt(K)
    # c_tilde: radial depth of each query token; clamp matches haa.py exactly
    c_tilde = torch.acosh((q_time / sqrt_k).clamp_min(1.0 + 1e-3))
    sinh_c  = torch.sinh(c_tilde)
    arg_B   = 1.0 - (beta / sinh_c).pow(2)
    B       = torch.sqrt(arg_B.clamp_min(0.0) + 1e-8)
    return B


# ===========================================================================
# Hook factory
# ===========================================================================

def register_hooks(
    model:            torch.nn.Module,
    welford_ctilde:   list[WelfordAggregator],
    hist_z:           list[HistogramAccumulator],
    cone_true:        list,
    cone_total:       list,
    welford_spatial:  list[WelfordAggregator],
    K:                float,
) -> list:
    """
    Register one pair of forward hooks on mha.q / mha.k per
    LorentzMultiHeadAttention module discovered in forward-traversal order.

    Q–K interception strategy
    ─────────────────────────
    •  A hook on mha.q captures the Q output tensor and stores it in
       q_buffer[layer_idx] (a dict local to this function).
    •  A hook on mha.k reads q_buffer[layer_idx], runs all per-layer
       metric updates (M1–M4), then deletes both tensors explicitly.

    This guarantees that at most ONE batch's Q tensor is live at any time
    per layer, in strict accordance with the memory-safety constraint.

    The hook returns None in both cases so the forward pass is not altered.

    Args:
        model:           The ViTClassifier (not DataParallel-wrapped).
        welford_ctilde:  Per-layer Welford aggregators for M1 (σ²_c̃).
        hist_z:          Per-layer histogram accumulators for M2 (KL of Z).
        cone_true:       Per-layer int counters for M3 (True cone entries).
        cone_total:      Per-layer int counters for M3 (total entries).
        welford_spatial: Per-layer Welford aggregators for M4 (CV ‖x_s‖).
        K:               Manifold curvature (read from checkpoint at runtime).

    Returns:
        List of hook handles; call handle.remove() on each to deregister.
    """
    q_buffer: dict[int, torch.Tensor] = {}
    handles:  list = []

    # Walk named_modules in definition order — matches forward-pass layer order.
    mha_modules = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, LorentzMultiHeadAttention)
    ]

    for layer_idx, (name, mha) in enumerate(mha_modules):

        # ---- Q hook: buffer the output until K hook fires ----
        def make_q_hook(lidx: int):
            def q_hook(module, inp, output):
                # output shape: (batch, heads, seq, head_dim+1)
                q_buffer[lidx] = output.detach()
                return None
            return q_hook

        # ---- K hook: process (Q, K) pair, update aggregators, release ----
        def make_k_hook(lidx: int):
            def k_hook(module, inp, output):
                q = q_buffer.pop(lidx, None)
                k = output.detach()

                if q is None:
                    return None

                z_safe: torch.Tensor | None = None

                try:
                    # -- M1: Radial depth variance  σ²_c̃  (Welford on Q) ------
                    # Time coordinate x₀ = q[..., 0];  shape (B, H, N)
                    q_time = q[..., 0]
                    arg    = (q_time / math.sqrt(K)).clamp(min=1.0 + 1e-3)
                    c_tilde = torch.acosh(arg)              # (B, H, N)
                    welford_ctilde[lidx].update(c_tilde)

                    # -- M4: Spatial norm CV  (Welford on Q spatial norms) -----
                    x_s          = q[..., 1:]               # (B, H, N, head_dim)
                    spatial_norm = x_s.norm(dim=-1)         # (B, H, N)
                    welford_spatial[lidx].update(spatial_norm)

                    # -- M2 & M3: Z and B  (placeholders, skip until provided) -
                    try:
                        z_safe = compute_Z(q, k, K)         # (B, H, Nq, Nk)

                        # M2: accumulate histogram
                        hist_z[lidx].update(z_safe)

                        # M3: zero-shot cone sparsity
                        # Call path that guarantees cone_total > 0 after batch 1:
                        #   q[..., 0:1]  → shape (B, H, N, 1)
                        #   compute_B    → returns B of shape (B, H, N, 1)
                        #   z_safe       → shape (B, H, N, N)
                        #   b_val + z_safe broadcasts to (B, H, N, N) via dim-3
                        #   cone_mask.numel() = B * H * N * N > 0  ✓
                        _beta     = 1.0
                        b_val     = compute_B(q[..., 0:1], K, _beta)   # (B,H,N,1)
                        cone_mask = (b_val + z_safe) <= 0.0
                        cone_true[lidx]  += int(cone_mask.sum().item())
                        cone_total[lidx] += cone_mask.numel()

                    except NotImplementedError:
                        pass   # formulas arrive in next message — silently skip

                finally:
                    # Explicit release regardless of exceptions.
                    del q, k
                    if z_safe is not None:
                        del z_safe
                    torch.cuda.empty_cache()

                return None
            return k_hook

        h_q = mha.q.register_forward_hook(make_q_hook(layer_idx))
        h_k = mha.k.register_forward_hook(make_k_hook(layer_idx))
        handles.extend([h_q, h_k])

    return handles


# ===========================================================================
# Model and data loading
# ===========================================================================

def load_model_and_K(
    checkpoint_path: str,
    device: str,
) -> tuple[torch.nn.Module, float, object]:
    """
    Reconstruct the model architecture from the saved args, load weights,
    and read the manifold curvature K.

    The checkpoint was saved as model.module.state_dict() (DataParallel),
    so we load directly into the non-wrapped model without key remapping.

    Returns:
        model:     ViTClassifier in eval mode, on `device`.
        K:         Manifold curvature (float).
        ckpt_args: The argparse.Namespace saved inside the checkpoint.
    """
    ckpt      = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = ckpt["args"]

    # Resolve model-size presets — mirrors train.py main()
    model_configs = {
        "tiny":  dict(num_layers=9,  hidden_dim=192, mlp_dim=384,  num_heads=12),
        "small": dict(num_layers=12, hidden_dim=384, mlp_dim=768,  num_heads=6),
        "base":  dict(num_layers=12, hidden_dim=768, mlp_dim=3072, num_heads=12),
    }
    size = getattr(ckpt_args, "model_size", "tiny")
    for key, value in model_configs[size].items():
        if getattr(ckpt_args, key, None) is None:
            setattr(ckpt_args, key, value)

    model = select_model(img_dim=[3, 32, 32], num_classes=100, args=ckpt_args)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # enc_manifold is set in ViTClassifier.__init__ as self.encoder.manifold
    K = float(model.enc_manifold.k.item())
    return model, K, ckpt_args


def build_val_loader(data_root: str) -> torch.utils.data.DataLoader:
    """
    Return a DataLoader over the CIFAR-100 test split (10 000 images).
    Uses the same normalisation parameters as initialize.py.
    """
    mean = (0.5070, 0.4865, 0.4409)
    std  = (0.2673, 0.2564, 0.2762)
    tf   = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    ds = tvd.CIFAR100(data_root, train=False, download=False, transform=tf)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


# ===========================================================================
# Post-processing and serialisation helpers
# ===========================================================================

def _spatial_cv(w: WelfordAggregator) -> float:
    """CV = std(‖x_s‖) / mean(‖x_s‖).  Returns 0.0 on degenerate input."""
    if w.mean < 1e-12:
        return 0.0
    return math.sqrt(w.variance) / w.mean


def _json_safe(v: float) -> float | None:
    """Convert NaN / Inf to None so json.dump produces valid JSON."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return v


def compute_results(
    n_layers:        int,
    welford_ctilde:  list[WelfordAggregator],
    hist_z:          list[HistogramAccumulator],
    cone_true:       list,
    cone_total:      list,
    welford_spatial: list[WelfordAggregator],
    rng:             np.random.Generator,
) -> dict:
    """
    Compute final per-layer metrics, determine k_star_candidate, and
    find k_star (shallowest layer satisfying all three mandatory thresholds).
    """
    results: dict = {}
    k_star:  int | None = None

    for l in range(n_layers):
        label = f"layer_{l + 1}"

        sigma2   = welford_ctilde[l].variance
        kl_div   = hist_z[l].kl_divergence()
        z_mean   = hist_z[l].z_mean
        p_val    = hist_z[l].bootstrap_p_value(rng)
        sparsity = (cone_true[l] / cone_total[l]) if cone_total[l] > 0 else None
        cv       = _spatial_cv(welford_spatial[l])

        m1_ok = sigma2 > THR_SIGMA2
        m2_ok = (p_val < THR_PVALUE) and (z_mean < 0.0)
        m3_ok = (sparsity is not None and
                 THR_SPARSITY_LO <= sparsity <= THR_SPARSITY_HI)
        candidate = m1_ok and m2_ok and m3_ok

        if candidate and k_star is None:
            k_star = l + 1   # 1-indexed

        results[label] = {
            "sigma2_c_tilde":      _json_safe(sigma2),
            "kl_divergence_Z":     _json_safe(kl_div),
            "z_mean":              _json_safe(z_mean),
            "z_bootstrap_p_value": _json_safe(p_val),
            "cone_sparsity":       _json_safe(sparsity),
            "spatial_cv":          _json_safe(cv),
            "k_star_candidate":    bool(candidate),
        }

    results["k_star"] = k_star
    return results


# ===========================================================================
# Figure generation
# ===========================================================================

def save_figure(results: dict, n_layers: int, path: str) -> None:
    """
    2×2 subplot grid of M1–M4 as layer-indexed line plots.
    Compatibility thresholds are drawn as horizontal dashed red lines.
    """
    layers = list(range(1, n_layers + 1))

    def _get(metric: str) -> list:
        return [results[f"layer_{l}"][metric] for l in layers]

    sigma2   = _get("sigma2_c_tilde")
    kl_div   = _get("kl_divergence_Z")
    sparsity = _get("cone_sparsity")
    cv       = _get("spatial_cv")

    # Replace None with NaN so matplotlib skips them gracefully
    def _to_float(seq: list) -> list:
        return [float("nan") if v is None else v for v in seq]

    sigma2   = _to_float(sigma2)
    kl_div   = _to_float(kl_div)
    sparsity = _to_float(sparsity)
    cv       = _to_float(cv)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Phase 0 — HexFormer geometric diagnostics (CIFAR-100 val, zero-shot)",
        fontsize=13,
    )

    # ---- M1 ----
    ax = axes[0, 0]
    ax.plot(layers, sigma2, marker="o", color="tab:blue")
    ax.axhline(THR_SIGMA2, color="red", linestyle="--",
               linewidth=1.2, label=f"threshold = {THR_SIGMA2}")
    ax.set_title("M1 — Radial depth variance  σ²_c̃")
    ax.set_xlabel("Layer")
    ax.set_ylabel("σ²")
    ax.legend(fontsize=8)
    ax.set_xticks(layers)

    # ---- M2 ----
    ax = axes[0, 1]
    ax.plot(layers, kl_div, marker="o", color="tab:orange")
    ax.set_title("M2 — Angular KL divergence  D_KL(p_Z ‖ Uniform)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL divergence")
    ax.set_xticks(layers)

    # ---- M3 ----
    ax = axes[1, 0]
    ax.plot(layers, sparsity, marker="o", color="tab:green")
    ax.axhline(THR_SPARSITY_LO, color="red",     linestyle="--",
               linewidth=1.2, label=f"lo = {THR_SPARSITY_LO}")
    ax.axhline(THR_SPARSITY_HI, color="darkred", linestyle="--",
               linewidth=1.2, label=f"hi = {THR_SPARSITY_HI}")
    ax.set_title("M3 — Zero-shot cone sparsity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sparsity ratio")
    ax.legend(fontsize=8)
    ax.set_xticks(layers)

    # ---- M4 ----
    ax = axes[1, 1]
    ax.plot(layers, cv, marker="o", color="tab:purple")
    ax.set_title("M4 — Spatial norm CV  ‖x_s‖")
    ax.set_xlabel("Layer")
    ax.set_ylabel("CV = std / mean")
    ax.set_xticks(layers)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[phase0] figure saved → {path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    device_idx = int(args.device.split(":", 1)[1])

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Supply an existing file with --checkpoint.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by run_phase_0.py, but no CUDA device is available.")
    if device_idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {args.device}, but only {torch.cuda.device_count()} CUDA "
            "device(s) are visible.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "phase0_diagnostics.json")
    output_png = os.path.join(args.output_dir, "phase0_curves.png")
    torch.cuda.set_device(args.device)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[phase0] loading checkpoint: {args.checkpoint}")
    model, K, ckpt_args = load_model_and_K(args.checkpoint, args.device)
    print(f"[phase0] manifold curvature K = {K:.6f}")

    # ── Discover LorentzMultiHeadAttention layers in forward order ───────────
    mha_list = [
        m for m in model.modules()
        if isinstance(m, LorentzMultiHeadAttention)
    ]
    n_layers = len(mha_list)
    print(f"[phase0] LorentzMultiHeadAttention layers discovered: {n_layers}")

    # ── Initialise per-layer aggregators ────────────────────────────────────
    welford_ctilde  = [WelfordAggregator()    for _ in range(n_layers)]
    hist_z          = [HistogramAccumulator() for _ in range(n_layers)]
    cone_true       = [0                      for _ in range(n_layers)]
    cone_total      = [0                      for _ in range(n_layers)]
    welford_spatial = [WelfordAggregator()    for _ in range(n_layers)]

    # ── Register hooks ───────────────────────────────────────────────────────
    handles = register_hooks(
        model,
        welford_ctilde, hist_z,
        cone_true, cone_total,
        welford_spatial,
        K,
    )
    print(f"[phase0] registered {len(handles)} hook handles "
          f"({n_layers} layers × 2 = {n_layers * 2} expected)")

    # ── Validation loader ────────────────────────────────────────────────────
    val_loader = build_val_loader(args.data_root)
    n_batches  = len(val_loader)
    print(f"[phase0] val batches: {n_batches}  "
          f"(batch_size={VAL_BATCH_SIZE}, dataset={args.data_root})")

    # ── Inference loop  (single torch.no_grad() context, model.eval()) ───────
    print("[phase0] running inference...")
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(args.device)
            _ = model(x)   # forward hooks fire here, aggregators update in-place
            del x
            # Empty cache once per batch to release any hook-side allocations.
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == n_batches:
                print(f"  batch {batch_idx + 1:>3}/{n_batches}")

    # ── Remove all hooks ─────────────────────────────────────────────────────
    for handle in handles:
        handle.remove()
    handles.clear()
    print("[phase0] hooks removed")

    # ── Compute final metrics (bootstrap run here, after all batches) ────────
    print(f"[phase0] computing bootstrap p-values "
          f"(N={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}) ...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = compute_results(
        n_layers,
        welford_ctilde, hist_z,
        cone_true, cone_total,
        welford_spatial,
        rng,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n[phase0] ── per-layer summary ─────────────────────────────────")
    print(f"  {'Layer':>5}  {'σ²_c̃':>8}  {'KL':>8}  "
          f"{'z_mean':>8}  {'sparsity':>9}  {'CV':>8}  k*?")
    for l in range(1, n_layers + 1):
        r = results[f"layer_{l}"]
        sp = r["cone_sparsity"]
        print(
            f"  {l:>5}  "
            f"{r['sigma2_c_tilde']:>8.4f}  "
            f"{r['kl_divergence_Z']:>8.4f}  "
            f"{r['z_mean']:>8.4f}  "
            f"{('—' if sp is None else f'{sp:.4f}'):>9}  "
            f"{r['spatial_cv']:>8.4f}  "
            f"{r['k_star_candidate']}"
        )
    print(f"\n[phase0] k_star = {results['k_star']}")

    # ── Write JSON ────────────────────────────────────────────────────────────
    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[phase0] results written → {output_json}")

    # ── Save figure ───────────────────────────────────────────────────────────
    save_figure(results, n_layers, output_png)


if __name__ == "__main__":
    main()
