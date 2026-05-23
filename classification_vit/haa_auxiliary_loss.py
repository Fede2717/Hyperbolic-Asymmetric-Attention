"""Auxiliary losses for breaking the HAA geometric deadlock.

Implements the Loss Factory pattern of Master_Execution_Pipeline.md Rule 2.
All auxiliary loss logic is contained in this module; the training loop
only calls ``build_aux_losses(args)`` once and iterates the returned dict.

Modules:
    RampSchedule              — warmup/ramp/plateau weight scheduler
    DirectionalAngularLoss    — band-hinge surrogate for angular ordering
    HyperbolicHierarchyLoss   — depth-stratification metric loss (HHL)
    HyperbolicPrototypeLoss   — Lorentz-distance to fixed fine prototypes
    RadialVarianceLoss        — one-sided hinge on radial-depth variance
    BetaCapLoss               — cap β to the geometric envelope of the batch
    build_hyperbolic_prototypes — fixed-prototype constructor for L_proto
    build_aux_losses          — factory mapping CLI flags to loss instances
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from hierarchy_loader import load_hierarchy


# ---------------------------------------------------------------------------
# Prototype constructor (P5 / P6)
# ---------------------------------------------------------------------------
def build_hyperbolic_prototypes(num_super: int,
                                num_fine: int,
                                hidden_dim: int,
                                fine_to_super_lut: torch.Tensor,
                                K: float = 1.0,
                                seed: int = 42,
                                d_s: float = 0.3,
                                d_f_mid: float = 1.175) -> torch.Tensor:
    """Construct fixed Lorentz prototypes for L_proto.

    Returns a tensor of shape [num_super + num_fine, hidden_dim].
    Indices [0 : num_super] are super-prototypes (shallow, depth d_s).
    Indices [num_super : num_super + num_fine] are fine-prototypes
    (deep, depth d_s + d_f_mid, inside parent's tan(pi/8) angular cap).

    Depths are DETERMINISTIC per level: every super sits exactly at
    depth d_s, every fine sits exactly at depth d_s + d_f_mid. Sibling
    fines therefore share identical Lorentz norms (the variance over
    fine depths is zero).

    Super angular positions: closed-form simplex ETF. C = num_super vectors
    in spatial space with all pairwise cosines = -1/(C-1). Maximally and
    uniformly separated by construction. Dataset-independent: depends only
    on (num_super, num_fine, spatial_dim, fine_to_super_lut).

    Fine angular positions: for each fine class c with super(c) = s, sample
    delta in R^spatial_dim, project onto the orthogonal complement of v_s,
    scale to tan(pi/8) magnitude, renormalise (v_s + delta_perp) to unit
    norm. Each fine prototype lies inside its parent's angular cap.

    Final spatial = unit_vec * sinh(depth); time = sqrt(K) * cosh(depth).
    """
    g = torch.Generator().manual_seed(seed)
    sqrt_K = K ** 0.5
    spatial_dim = hidden_dim - 1

    # --- super angles via closed-form simplex ETF ---
    # Mettes 2019 / neural-collapse canonical maximum-separation construction.
    C = num_super
    if spatial_dim < C - 1:
        raise ValueError(
            f"spatial_dim={spatial_dim} < num_super-1={C-1}; "
            "simplex ETF requires spatial_dim >= num_super - 1.")
    U_full = torch.randn(spatial_dim, C, generator=g)
    U, _ = torch.linalg.qr(U_full)                           # [spatial_dim, C], semi-orthogonal
    M_simplex = (math.sqrt(C / (C - 1.0)) *
                 (torch.eye(C) - (1.0 / C) * torch.ones(C, C)))
    super_angles = (U @ M_simplex).T                         # [C, spatial_dim]
    super_angles = super_angles / super_angles.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # --- fine angles within parent super's angular cap ---
    cap_tan = torch.tan(torch.tensor(math.pi / 8.0))
    fine_angles = torch.zeros(num_fine, spatial_dim)
    for c in range(num_fine):
        s = int(fine_to_super_lut[c].item())
        v_s = super_angles[s]
        delta = torch.randn(spatial_dim, generator=g)
        delta_perp = delta - (delta @ v_s) * v_s
        delta_perp = delta_perp / delta_perp.norm().clamp_min(1e-8)
        w = v_s + cap_tan.item() * delta_perp
        w = w / w.norm().clamp_min(1e-8)
        fine_angles[c] = w

    # --- depths (deterministic per level) ---
    super_depths = torch.full((num_super,), float(d_s))
    fine_depths  = torch.full((num_fine,),  float(d_s) + float(d_f_mid))

    # --- assemble Lorentz points ---
    def assemble(angles, depths):
        sinh_d = torch.sinh(depths).unsqueeze(-1)
        cosh_d = torch.cosh(depths).unsqueeze(-1)
        spatial = angles * sinh_d
        time = sqrt_K * cosh_d
        return torch.cat([time, spatial], dim=-1)

    super_protos = assemble(super_angles, super_depths)
    fine_protos = assemble(fine_angles, fine_depths)
    assert torch.allclose(super_protos[..., 0], super_protos[0:1, 0].expand_as(super_protos[..., 0]), atol=1e-6), "super depth check failed"
    assert torch.allclose(fine_protos[..., 0], fine_protos[0:1, 0].expand_as(fine_protos[..., 0]), atol=1e-6), "fine depth check failed"
    return torch.cat([super_protos, fine_protos], dim=0)


# ---------------------------------------------------------------------------
# Helper: locate the deepest HAA layer's MultiHeadAttention module.
# ---------------------------------------------------------------------------
def _get_last_haa_mha(model):
    """Return the LorentzMultiHeadAttention with use_haa=True at the
    highest layer_idx in the (possibly DataParallel-wrapped) model."""
    base = model.module if hasattr(model, 'module') else model
    last_mha = None
    last_idx = -1
    for _, m in base.named_modules():
        if getattr(m, 'use_haa', False):
            idx = getattr(m, 'layer_idx', -1)
            if idx > last_idx:
                last_mha = m
                last_idx = idx
    return last_mha


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
class RampSchedule:
    """Warmup → linear ramp → plateau weight schedule.

    Behaviour:
        epoch < warmup                              -> 0.0
        warmup <= epoch < warmup + ramp             -> linear ramp
        epoch >= warmup + ramp                      -> plateau
    """
    def __init__(self, warmup: int, ramp: int, plateau: float):
        self.warmup = warmup
        self.ramp = ramp
        self.plateau = plateau

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 0.0
        if epoch < self.warmup + self.ramp:
            return self.plateau * (epoch - self.warmup) / max(1, self.ramp)
        return self.plateau


# ---------------------------------------------------------------------------
# Cone Occupancy Loss
# ---------------------------------------------------------------------------
class DirectionalAngularLoss(nn.Module):
    """Smooth surrogate for angular ordering with band-hinge target.

    Despite this loss being historically labelled "cone occupancy", the
    forward computation does NOT measure volumetric cone occupancy: it does
    not reference the aperture B at all. It measures angular ordering on the
    hyperboloid — the soft fraction of (Q, K) pairs in which K lies in the
    angular half-space "deeper" than Q (i.e., Z < 0 at vertex Q), regardless
    of the half-angle β subtended by Q's entailment cone.

    Surrogate:
        s_angular = mean( σ( κ · ( -Z - m_smooth ) ) )

    A pair contributes ≈ 1 when Z < 0 (K geometrically deeper than Q at
    vertex Q) and ≈ 0 when Z > 0 (sibling/shallower geometry). The earlier
    formulation cone_score = -(B + Z) - m_smooth was replaced by -Z to block
    a β-collapse loophole: the optimizer was satisfying the constraint by
    inflating β rather than moving tokens, producing trivial-looking
    occupancy with no geometric content.

    The band [s_lo, s_hi] is enforced on this angular fraction: if it falls
    outside the target window the squared hinge penalty kicks in.

    P1: Reduction is computed over geometrically valid Q-K pairs only
    (those with norm_sq_QK > 1e-6 AND norm_sq_OQ > 1e-6). Masked pairs are
    excluded entirely; they do not drag the mean toward the band-satisfying
    value σ(−0.2) ≈ 0.45 that would otherwise auto-satisfy the loss when
    the masking rate is high.
    """
    def __init__(self,
                 s_lo: float = 0.55,
                 s_hi: float = 0.85,
                 kappa: float = 4.0,
                 m_smooth: float = 0.05,
                 warmup: int = 15,
                 ramp: int = 25,
                 plateau: float = 0.3):
        super().__init__()
        self.s_lo = s_lo
        self.s_hi = s_hi
        self.kappa = kappa
        self.m_smooth = m_smooth
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_Z', None) is None or getattr(mha, '_last_valid_mask', None) is None:
            return torch.zeros((), device=device)
        Z = mha._last_Z
        valid = getattr(mha, '_last_valid_mask', None)
        if valid is None or not valid.any():
            return torch.zeros((), device=device)
        # REPLACED -(B+Z) WITH -Z: Breaks the cheap beta-collapse gradient path.
        # Occupancy must be satisfied by moving Z structurally, not inflating beta.
        cone_score = -Z - self.m_smooth
        sigmoid_vals = torch.sigmoid(self.kappa * cone_score)
        valid_f = valid.float()
        n_valid = valid_f.sum().clamp_min(1.0)
        s_cone = (sigmoid_vals * valid_f).sum() / n_valid
        loss_lo = F.relu(self.s_lo - s_cone).pow(2)
        loss_hi = F.relu(s_cone - self.s_hi).pow(2)
        return loss_lo + loss_hi


# ---------------------------------------------------------------------------
# CLS-Row Direction Loss (Phase-3-E)
# ---------------------------------------------------------------------------
class CLSRowDirectionLoss(nn.Module):
    """One-sided hinge on the mean of Z restricted to the CLS row at the
    deepest HAA layer. Replaces L_angular in Phase-3-E configurations.

    Formula:
        L = mean_{b,h} ReLU(z_mean_cls[b, h] - z_star)^2
    where z_mean_cls is the valid-mask-weighted mean of Z over keys at
    query-index 0 (the CLS row). Achievable target (z_star < 0) makes
    the loss satisfiable at equilibrium, eliminating the standing
    unsatisfied gradient of L_angular's band-hinge.

    Mask handling: pairs flagged invalid by mha._last_valid_mask are
    excluded from the per-row mean. Rows with zero valid pairs are
    excluded from the batch mean.
    """
    def __init__(self,
                 z_star: float = -0.05,
                 warmup: int = 15,
                 ramp: int = 25,
                 plateau: float = 0.15):
        super().__init__()
        self.z_star = z_star
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None \
                or getattr(mha, '_last_Z', None) is None \
                or getattr(mha, '_last_valid_mask', None) is None:
            return torch.zeros((), device=device)
        Z_cls = mha._last_Z[:, :, 0, :]                # [B, h, n_k]
        M_cls = mha._last_valid_mask[:, :, 0, :]       # [B, h, n_k] bool
        valid_f = M_cls.float()
        n_valid = valid_f.sum(-1)                      # [B, h]
        row_has_valid = (n_valid >= 2.0).float()       # [B, h]
        if row_has_valid.sum().item() < 1.0:
            return torch.zeros((), device=device)
        n_valid_safe = n_valid.clamp_min(2.0)
        z_mean_cls = (Z_cls * valid_f).sum(-1) / n_valid_safe   # [B, h]
        hinge = F.relu(z_mean_cls - self.z_star).pow(2)         # [B, h]
        loss = (hinge * row_has_valid).sum() / row_has_valid.sum().clamp_min(1.0)
        return loss


# ---------------------------------------------------------------------------
# CLS-Row Variance Loss (Phase-3-E)
# ---------------------------------------------------------------------------
class CLSRowVarianceLoss(nn.Module):
    """One-sided floor on the K-variance of Z restricted to the CLS row.

    Formula:
        sigma2_cls[b, h] = Var_k[Z(0, k)]  over valid pairs
        L = ReLU(sigma2_star - mean_{b,h} sigma2_cls)^2

    Direct expression of the discrimination-capacity signal HAA provides
    at the CLS row. Initial target sigma2_star = 0.05 is a placeholder
    calibrated from sigma_Z ~ 0.27 in the non-degenerate regime
    (Var_Z ~ 0.073, target ~70%). Replace with a probe-calibrated value
    once measured on an existing checkpoint.

    Mask handling: same as CLSRowDirectionLoss.
    """
    def __init__(self,
                 sigma2_star: float = 0.05,
                 warmup: int = 10,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        self.sigma2_star = sigma2_star
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None \
                or getattr(mha, '_last_Z', None) is None \
                or getattr(mha, '_last_valid_mask', None) is None:
            return torch.zeros((), device=device)
        Z_cls = mha._last_Z[:, :, 0, :]                # [B, h, n_k]
        M_cls = mha._last_valid_mask[:, :, 0, :]       # [B, h, n_k] bool
        valid_f = M_cls.float()
        n_valid = valid_f.sum(-1)                      # [B, h]
        row_has_valid = (n_valid >= 2.0).float()       # [B, h]
        if row_has_valid.sum().item() < 1.0:
            return torch.zeros((), device=device)
        n_valid_safe = n_valid.clamp_min(2.0)
        z_mean_cls = (Z_cls * valid_f).sum(-1) / n_valid_safe          # [B, h]
        z_centered = (Z_cls - z_mean_cls.unsqueeze(-1)) * valid_f
        sigma2_cls = z_centered.pow(2).sum(-1) / n_valid_safe          # [B, h]
        sigma2_batch = (sigma2_cls * row_has_valid).sum() / row_has_valid.sum().clamp_min(1.0)
        return F.relu(self.sigma2_star - sigma2_batch).pow(2)


# ---------------------------------------------------------------------------
# Cone Occupancy Loss (Phase 2) — β-detach guarded
# ---------------------------------------------------------------------------
class ConeOccupancyLoss(nn.Module):
    """Soft cone-occupation loss with mathematically correct β-collapse guard.

    cone_score = -(B_det + Z) - m_smooth, with B_det = B.detach(). The
    aperture B is held detached so the optimizer cannot satisfy the
    occupation target by inflating β — gradient must flow through Z only.
    The loss is a one-sided hinge on the soft-occupation fraction, pulling
    it up to s_target.
    """
    def __init__(self,
                 s_target: float = 0.10,
                 kappa: float = 6.0,
                 m_smooth: float = 0.05,
                 warmup: int = 15,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        self.s_target = s_target
        self.kappa = kappa
        self.m_smooth = m_smooth
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_Z', None) is None \
                       or getattr(mha, '_last_B', None) is None \
                       or getattr(mha, '_last_valid_mask', None) is None:
            return torch.zeros((), device=device)
        Z = mha._last_Z                             # gradient-attached
        B_det = mha._last_B.detach()                # β-collapse guard
        assert not B_det.requires_grad, "B_det must be detached (β-collapse guard)"
        valid = mha._last_valid_mask
        if not valid.any():
            return torch.zeros((), device=device)
        cone_score = -(B_det + Z) - self.m_smooth
        sig = torch.sigmoid(self.kappa * cone_score)
        valid_f = valid.float()
        n_valid = valid_f.sum().clamp_min(1.0)
        s_cone = (sig * valid_f).sum() / n_valid
        return F.relu(self.s_target - s_cone).pow(2)


# ---------------------------------------------------------------------------
# Hyperbolic Hierarchy Loss (HHL)
# ---------------------------------------------------------------------------
class HyperbolicHierarchyLoss(nn.Module):
    """Enforces super-class shallower than fine-class via margin hinge.

    Depth d_L(x, O) on the Lorentz hyperboloid with curvature K is
        sqrt(K) · acosh( x_0 / sqrt(K) ).
    Computed on the CLS token's time coordinate at the final HAA layer.
    Uses EMA centroid smoothing across batches for stability.
    """
    def __init__(self,
                 K: float,
                 dataset_name: str = 'CIFAR-100',
                 margin: float = 0.3,
                 ema: float = 0.9,
                 warmup: int = 5,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        FINE_TO_SUPER, NUM_FINE, NUM_SUPER = load_hierarchy(dataset_name)
        self.NUM_SUPER = NUM_SUPER
        self.NUM_FINE = NUM_FINE
        self.K = K
        self.sqrt_K = K ** 0.5
        self.margin = margin
        self.ema = ema
        self.schedule = RampSchedule(warmup, ramp, plateau)

        # EMA centroid buffers (non-persistent — reset on checkpoint reload).
        self.register_buffer('super_centroids',
                             torch.zeros(NUM_SUPER), persistent=False)
        self.register_buffer('fine_centroids',
                             torch.zeros(NUM_FINE), persistent=False)
        # Lazy-init flag (Python attribute, not persistent).
        self._initialized = False

        # Fine -> super lookup table as a buffer so it moves with .to(device).
        _lut = torch.zeros(NUM_FINE, dtype=torch.long)
        for fine, sup in FINE_TO_SUPER.items():
            _lut[fine] = sup
        self.register_buffer('fine_to_super_lut', _lut, persistent=False)

    def _depth(self, time_coord: torch.Tensor) -> torch.Tensor:
        return self.sqrt_K * torch.acosh(
            (time_coord / self.sqrt_K).clamp_min(1.0 + 1e-3))

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_cls_time', None) is None:
            return torch.zeros((), device=device)

        cls_time = mha._last_cls_time  # [b, 1]
        if cls_time.dim() > 2:
            cls_time = cls_time.reshape(cls_time.shape[0], -1)[:, 0:1]
        depths = self._depth(cls_time).squeeze(-1)  # [b]

        super_labels = self.fine_to_super_lut[y]

        # Per-class mean depth this batch.
        super_mean = torch.zeros(self.NUM_SUPER, device=depths.device)
        fine_mean = torch.zeros(self.NUM_FINE, device=depths.device)
        super_count = torch.zeros(self.NUM_SUPER, device=depths.device)
        fine_count = torch.zeros(self.NUM_FINE, device=depths.device)
        super_mean.scatter_add_(0, super_labels, depths)
        fine_mean.scatter_add_(0, y, depths)
        super_count.scatter_add_(0, super_labels, torch.ones_like(depths))
        fine_count.scatter_add_(0, y, torch.ones_like(depths))
        super_mean = super_mean / super_count.clamp_min(1.0)
        fine_mean = fine_mean / fine_count.clamp_min(1.0)

        # EMA update — first call seeds centroids only on observed classes.
        with torch.no_grad():
            mask_super = super_count > 0
            mask_fine = fine_count > 0
            if not self._initialized:
                self.super_centroids[mask_super] = super_mean[mask_super].detach()
                self.fine_centroids[mask_fine] = fine_mean[mask_fine].detach()
                self._initialized = True
            else:
                self.super_centroids[mask_super] = (
                    self.ema * self.super_centroids[mask_super]
                    + (1.0 - self.ema) * super_mean[mask_super].detach()
                )
                self.fine_centroids[mask_fine] = (
                    self.ema * self.fine_centroids[mask_fine]
                    + (1.0 - self.ema) * fine_mean[mask_fine].detach()
                )

        # NOTE: This fix produces non-zero gradient but does NOT guarantee
        # directional depth stratification. Both tensors are derived from the
        # same batch with no external reference; the gradient is non-zero but
        # stochastic in sign across batches. Directional radial supervision
        # requires the prototype loss (L_proto) introduced in the next iteration.
        # This fix is diagnostic — it confirms whether HHL contributes anything.
        #
        # Map per-fine-class mean depths (computed above) to their superclasses,
        # then average. This is class-weighted, mathematically distinct from
        # super_mean (which is token-weighted across the whole superclass).
        observed_fine_idx = mask_fine.nonzero(as_tuple=False).squeeze(-1)
        super_of_observed_fine = self.fine_to_super_lut[observed_fine_idx]
        fine_mean_per_super  = torch.zeros(self.NUM_SUPER, device=depths.device)
        fine_classes_per_super = torch.zeros(self.NUM_SUPER, device=depths.device)
        fine_mean_per_super.scatter_add_(0, super_of_observed_fine,
                                         fine_mean[observed_fine_idx])
        fine_classes_per_super.scatter_add_(
            0, super_of_observed_fine,
            torch.ones_like(fine_mean[observed_fine_idx]))
        fine_mean_per_super = (fine_mean_per_super
                               / fine_classes_per_super.clamp_min(1.0))

        # Hinge: super centroid must be shallower than fine centroid
        # by at least margin. Active only for superclasses present in batch.
        mask = (super_count > 0) & (fine_classes_per_super > 0)
        if not mask.any():
            return torch.zeros((), device=device)

        loss = F.relu(
            super_mean[mask] - fine_mean_per_super[mask] + self.margin
        ).pow(2).mean()
        return loss


# ---------------------------------------------------------------------------
# Hyperbolic Prototype Loss (P5)
# ---------------------------------------------------------------------------
class HyperbolicPrototypeLoss(nn.Module):
    """Squared Lorentz distance from post-attention CLS to fixed FINE prototype.

    FIX-PROTO-DEPTH (REVISED, Alternative 1): pull CLS toward the
    fine-grained prototype at depth d_s + d_f[y], NOT the superclass
    prototype at d_s. Reasoning: the classification head needs CLS deep
    enough to discriminate ``num_fine`` classes; pulling CLS to the
    shallow superclass depth compresses the classification angular
    budget. The HAA depth-ordering requirement (Q-CLS shallower than
    K-patch) is handled separately by an in-attention alpha_q control;
    L_proto's job here is to supervise the post-encoder CLS for
    classification, not to drive HAA geometry.

    Math (validated):
      inner = -<cls, p_y>_L
      arg   = inner / K
      u     = max(arg - 1, 0)
      acosh = log1p(u + sqrt(u·(2+u) + 1e-12))   [closed-form, no torch.acosh]
      L     = mean(acosh²)

    The closed-form identity arccosh(1+u) = log1p(u + sqrt(u(2+u))) avoids
    the 1/sqrt(arg-1) singularity in the standard arccosh derivative.
    Combined with the outer ()², the gradient at coincident points is
    exactly zero (validated under static gradient checks).
    """
    def __init__(self,
                 K: float,
                 prototypes_lorentz: torch.Tensor,
                 num_super: int,
                 warmup: int = 5,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        self.K = K
        self.num_super = num_super
        self.register_buffer('prototypes', prototypes_lorentz, persistent=False)
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_cls_lorentz', None) is None:
            return torch.zeros((), device=device)
        cls = mha._last_cls_lorentz                          # [B, hidden_dim+1]
        # FIX-PROTO-DEPTH (REVISED, Alternative 1):
        # Pull post-attention CLS toward the fine-grained prototype at
        # depth d_s + d_f[y], NOT the superclass prototype at d_s.
        # Reasoning: the classification head needs CLS deep enough to
        # discriminate 100 fine classes. Pulling CLS to the shallow
        # superclass depth (0.3) compresses the classification angular
        # budget. The HAA depth-ordering requirement (Q-CLS shallower than
        # K-patch) is handled separately by the in-attention alpha_q
        # control planned for Step 11; L_proto's job is to supervise the
        # post-encoder CLS for classification, not to drive HAA geometry.
        # Fine-prototypes occupy indices [num_super : num_super + num_fine]
        # in self.prototypes.
        target_proto = self.prototypes[self.num_super + y]   # [B, hidden_dim+1]
        inner = (-cls[..., 0:1] * target_proto[..., 0:1]
                 + (cls[..., 1:] * target_proto[..., 1:]).sum(-1, keepdim=True))
        arg = -inner / self.K
        u = (arg - 1.0).clamp_min(0.0)
        sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
        acosh_val = torch.log1p(u + sqrt_term)
        # S-7b: full Lorentz geodesic distance squared d_L^2 = K * acosh^2(arg).
        # K=1.0 in all past Phase-1 experiments (Step 8 v3.2 etc.), so this
        # change is numerically identity for prior runs. The fix matches the
        # K factor convention used by RadialVarianceLoss, BetaCapLoss, and the
        # spatial-penalty path in transformer_blocks.py.
        return (self.K * acosh_val.pow(2)).mean()


# ---------------------------------------------------------------------------
# Radial Variance Loss (P5)
# ---------------------------------------------------------------------------
class RadialVarianceLoss(nn.Module):
    """One-sided hinge on within-image radial-depth variance.

    Now supervises POST-W_Q Q tensor at the HAA layer (the tensor the score
    formula reads), not the pre-MHA input. Step 8 v3.2 finding: pre-MHA
    supervision was structurally decoupled from HAA's consumed geometry.

    Math (validated):
      c_tilde = sqrt(K) · log1p(u + sqrt(u·(2+u) + 1e-12))   with u = max(x_0/sqrt(K) - 1, 0)
      σ²_per_image_per_head = Var_n(c_tilde)   [population variance over tokens]
      σ²_batch              = mean over images and heads
      L                     = ReLU(σ²_target - σ²_batch)²

    Population variance (unbiased=False) avoids the 1/(n-1) blowup at
    n=2. Below target: linear-in-deficit gradient. Above target: zero
    (saturated). Token count < 2 is guarded.
    """
    def __init__(self,
                 K: float,
                 sigma2_target: float = 0.10,
                 warmup: int = 5,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        self.K = K
        self.sqrt_K = K ** 0.5
        self.sigma2_target = sigma2_target
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_q_post_wq', None) is None:
            return torch.zeros((), device=device)
        # Post-W_Q Q tensor: [B, heads, n, head_dim+1]. We compute the
        # within-image variance of c_tilde across tokens, averaged over heads
        # and the batch. This is the tensor HAA's score formula actually
        # consumes — supervising the input is necessary but not sufficient,
        # as documented in the Step 8 v3.2 report (post-W_Q sigma2 collapses
        # 50x to 5000x relative to input sigma2).
        q = mha._last_q_post_wq                                # [B, h, n, d+1]
        if q.shape[0] < 1 or q.shape[2] < 2:
            return torch.zeros((), device=device)
        time = q[..., 0]                                       # [B, h, n]
        arg = time / self.sqrt_K
        u = (arg - 1.0).clamp_min(0.0)
        sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
        c_tilde = self.sqrt_K * torch.log1p(u + sqrt_term)     # [B, h, n]
        sigma2_per_image_per_head = c_tilde.var(dim=-1, unbiased=False)  # [B, h]
        sigma2_batch = sigma2_per_image_per_head.mean()
        return F.relu(self.sigma2_target - sigma2_batch).pow(2)


# ---------------------------------------------------------------------------
# Spread Loss (anti-collapse floor on post-W_Q radial AND spatial variance)
# ---------------------------------------------------------------------------
class SpreadLoss(nn.Module):
    """Anti-collapse spread floor on post-W_Q tensors at the HAA layer.

    Penalises representation collapse — the trivial-solution failure mode
    that L_occ alone admits (E3 and E4 reports). Two floor terms:
      (i)  radial:  ReLU(sigma2_target_rad - sigma2_c_tilde_post_W_Q)^2
      (ii) spatial: ReLU(cv_target_spat   - spatial_cv_post_W_Q)^2
    where sigma2_c_tilde is the within-image variance of c_tilde across
    tokens of the post-W_Q Q tensor (averaged over heads), and
    spatial_cv is the coefficient of variation of post-W_Q spatial norms
    across tokens within an image (averaged over heads).
    """
    def __init__(self,
                 K: float,
                 sigma2_target_rad: float = 0.10,
                 cv_target_spat: float = 0.30,
                 weight_rad: float = 1.0,
                 weight_spat: float = 1.0,
                 warmup: int = 5,
                 ramp: int = 25,
                 plateau: float = 0.5):
        super().__init__()
        self.K = K
        self.sqrt_K = K ** 0.5
        self.sigma2_target_rad = sigma2_target_rad
        self.cv_target_spat = cv_target_spat
        self.weight_rad = weight_rad
        self.weight_spat = weight_spat
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_q_post_wq', None) is None:
            return torch.zeros((), device=device)
        q = mha._last_q_post_wq                                  # [B, h, n, d+1]
        if q.shape[0] < 1 or q.shape[2] < 2:
            return torch.zeros((), device=device)
        time = q[..., 0]
        arg = time / self.sqrt_K
        u = (arg - 1.0).clamp_min(0.0)
        sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
        c_tilde = self.sqrt_K * torch.log1p(u + sqrt_term)       # [B, h, n]
        sigma2_per = c_tilde.var(dim=-1, unbiased=False)         # [B, h]
        sigma2_batch = sigma2_per.mean()
        loss_rad = F.relu(self.sigma2_target_rad - sigma2_batch).pow(2)

        space = q[..., 1:]                                       # [B, h, n, d]
        norms = space.norm(dim=-1)                               # [B, h, n]
        mean_n = norms.mean(dim=-1, keepdim=True)                # [B, h, 1]
        std_n  = norms.std(dim=-1, unbiased=False, keepdim=True) # [B, h, 1]
        cv_per = (std_n / (mean_n.abs() + 1e-8)).squeeze(-1)     # [B, h]
        cv_batch = cv_per.mean()
        loss_spat = F.relu(self.cv_target_spat - cv_batch).pow(2)

        return self.weight_rad * loss_rad + self.weight_spat * loss_spat


# ---------------------------------------------------------------------------
# Beta-Cap Loss (P5)
# ---------------------------------------------------------------------------
class BetaCapLoss(nn.Module):
    """Cap β to the geometric envelope of the batch.

    Math (validated):
      target = sinh(quantile(c_tilde_batch, q))   [DETACHED — no grad through]
      L = mean over heads of ReLU(β_h - target)²

    Path A (dynamic, default): target recomputed from each batch's c̃ percentile.
    Path B (static): target frozen at value passed via static_target argument.

    Gradient flows only into β_raw via softplus parameterisation; the
    quantile is detached so the non-smooth order-statistic gradient
    never propagates. Saturated (β ≤ target): grad = 0. batch < 4 guarded.
    """
    def __init__(self,
                 K: float,
                 percentile: float = 0.25,
                 static_target: float = None,
                 warmup: int = 20,
                 ramp: int = 20,
                 plateau: float = 0.3):
        super().__init__()
        self.K = K
        self.sqrt_K = K ** 0.5
        self.percentile = percentile
        self.static_target = static_target
        self.schedule = RampSchedule(warmup, ramp, plateau)

    def forward(self, model, x, y, device):
        mha = _get_last_haa_mha(model)
        if mha is None or getattr(mha, '_last_all_tokens_lorentz', None) is None:
            return torch.zeros((), device=device)

        beta = F.softplus(mha.beta_raw)  # [H] after Stage 1.1, else [1]

        if self.static_target is not None:
            target = torch.sinh(torch.tensor(self.static_target, device=device))
        else:
            tokens = mha._last_all_tokens_lorentz  # [B, n, hidden_dim]
            time = tokens[..., 0:1]
            arg = time / self.sqrt_K
            u = (arg - 1.0).clamp_min(0.0)
            sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
            c_tilde = self.sqrt_K * torch.log1p(u + sqrt_term)
            c_flat = c_tilde.detach().flatten()
            if c_flat.numel() < 4:
                return torch.zeros((), device=device)
            target = torch.sinh(torch.quantile(c_flat, self.percentile))

        return F.relu(beta - target).pow(2).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_aux_losses(args):
    """Construct auxiliary loss modules controlled by CLI flags.

    Returns a dict with keys 'angular', 'hhl', 'proto', 'radvar', 'betacap'
    where each entry is either an ``nn.Module`` or ``None`` if disabled
    (its corresponding *_max weight is 0). The training loop iterates the
    dict and skips None entries.
    """
    out = {'angular': None, 'hhl': None, 'proto': None,
           'radvar': None, 'betacap': None, 'occ': None,
           'spread': None, 'cls_dir': None, 'cls_var': None}

    gamma_max = float(getattr(args, 'gamma_angular_max', 0.0))
    if gamma_max > 0:
        out['angular'] = DirectionalAngularLoss(
            warmup=int(getattr(args, 'gamma_angular_warmup', 15)),
            ramp=25,
            plateau=gamma_max,
        )

    eta_max = float(getattr(args, 'eta_max', 0.0))
    if eta_max > 0:
        K = float(getattr(args, 'encoder_k', 1.0))
        out['hhl'] = HyperbolicHierarchyLoss(
            K=K,
            dataset_name=str(getattr(args, 'dataset', 'CIFAR-100')),
            warmup=int(getattr(args, 'eta_warmup', 5)),
            ramp=25,
            plateau=eta_max,
        )

    eta_proto_max = float(getattr(args, 'eta_proto_max', 0.0))
    if eta_proto_max > 0:
        if bool(getattr(args, 'use_proto_softmax', False)):
            # PHASE2: L_proto provides only attraction to p_y; prototype-softmax CE
            # provides attraction PLUS logsumexp repulsion from non-targets. Running
            # both double-counts the attraction component. L_proto is disabled here.
            import warnings as _w
            _w.warn("L_proto disabled because --use_proto_softmax is set "
                    "(prototype-softmax CE supersedes L_proto).")
        else:
            K = float(getattr(args, 'encoder_k', 1.0))
            prototypes = getattr(args, 'hyperbolic_prototypes', None)
            if prototypes is None:
                raise RuntimeError(
                    "L_proto enabled but args.hyperbolic_prototypes not set. "
                    "Build prototypes in the model setup before constructing aux losses.")
            _, _NUM_FINE, _NUM_SUPER = load_hierarchy(
                str(getattr(args, 'dataset', 'CIFAR-100')))
            out['proto'] = HyperbolicPrototypeLoss(
                K=K,
                prototypes_lorentz=prototypes,
                num_super=int(getattr(args, 'num_super', _NUM_SUPER)),
                warmup=int(getattr(args, 'eta_proto_warmup', 5)),
                ramp=25,
                plateau=eta_proto_max,
            )

    zeta_radvar_max = float(getattr(args, 'zeta_radvar_max', 0.0))
    if zeta_radvar_max > 0:
        K = float(getattr(args, 'encoder_k', 1.0))
        out['radvar'] = RadialVarianceLoss(
            K=K,
            sigma2_target=float(getattr(args, 'sigma2_target', 0.10)),
            warmup=int(getattr(args, 'zeta_radvar_warmup', 5)),
            ramp=25,
            plateau=zeta_radvar_max,
        )

    xi_betacap_max = float(getattr(args, 'xi_betacap_max', 0.0))
    if xi_betacap_max > 0:
        K = float(getattr(args, 'encoder_k', 1.0))
        static_t = getattr(args, 'betacap_static_target', None)
        out['betacap'] = BetaCapLoss(
            K=K,
            percentile=float(getattr(args, 'betacap_percentile', 0.25)),
            static_target=float(static_t) if static_t is not None else None,
            warmup=int(getattr(args, 'xi_betacap_warmup', 20)),
            ramp=20,
            plateau=xi_betacap_max,
        )

    phi_occ_max = float(getattr(args, 'phi_occ_max', 0.0))
    if phi_occ_max > 0:
        out['occ'] = ConeOccupancyLoss(
            s_target=float(getattr(args, 'occ_s_target', 0.10)),
            kappa=float(getattr(args, 'occ_kappa', 6.0)),
            m_smooth=float(getattr(args, 'occ_m_smooth', 0.05)),
            warmup=int(getattr(args, 'phi_occ_warmup', 15)),
            ramp=25,
            plateau=phi_occ_max,
        )

    omega_spread_max = float(getattr(args, 'omega_spread_max', 0.0))
    if omega_spread_max > 0:
        K = float(getattr(args, 'encoder_k', 1.0))
        out['spread'] = SpreadLoss(
            K=K,
            sigma2_target_rad=float(getattr(args, 'spread_sigma2_target', 0.10)),
            cv_target_spat=float(getattr(args, 'spread_cv_target', 0.30)),
            weight_rad=float(getattr(args, 'spread_weight_rad', 1.0)),
            weight_spat=float(getattr(args, 'spread_weight_spat', 1.0)),
            warmup=int(getattr(args, 'omega_spread_warmup', 5)),
            ramp=25,
            plateau=omega_spread_max,
        )

    chi_cls_dir_max = float(getattr(args, 'chi_cls_dir_max', 0.0))
    if chi_cls_dir_max > 0:
        out['cls_dir'] = CLSRowDirectionLoss(
            z_star=float(getattr(args, 'cls_dir_z_star', -0.05)),
            warmup=int(getattr(args, 'chi_cls_dir_warmup', 15)),
            ramp=25,
            plateau=chi_cls_dir_max,
        )

    psi_cls_var_max = float(getattr(args, 'psi_cls_var_max', 0.0))
    if psi_cls_var_max > 0:
        out['cls_var'] = CLSRowVarianceLoss(
            sigma2_star=float(getattr(args, 'cls_var_sigma2_star', 0.05)),
            warmup=int(getattr(args, 'psi_cls_var_warmup', 10)),
            ramp=25,
            plateau=psi_cls_var_max,
        )

    return out
