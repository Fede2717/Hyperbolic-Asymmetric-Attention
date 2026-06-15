"""L_HEC — Ganea-style entailment cone loss for Hyperbolic Asymmetric Attention.

Supervises post-W_Q CLS-vs-patch pairs at the HAA layer. Hard hinge on
per-pair entailment angle theta vs cone half-aperture psi.

Two variants:
  - 'naive':  all cross-image patches are negatives
  - 'supcon': mask same-class patches from negatives (avoids representation tearing)

Math:
    theta(u, v) = acos(-Z(u, v))       # entailment angle at u toward v
    psi(u)      = asin(B(u))            # cone half-aperture at u
    positives:  pairs (CLS_i, patch in image i)        ; loss = relu(theta - psi)
    negatives:  pairs (CLS_i, patch in image j != i)   ; loss = relu(psi - theta + gamma)

For SupCon variant, negatives further exclude patches whose source image shares
the class label with image i. Required: per-image class labels in the batch.

Numerical safeguards:
    - Z is clamped to [-1 + eps, 1 - eps] before acos
    - B is clamped to [eps, 1 - eps] before asin
    - Returns 0.0 with detached tensor in eval mode for safety
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_acos(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.acos(z.clamp(-1.0 + eps, 1.0 - eps))


def _safe_asin(b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.asin(b.clamp(eps, 1.0 - eps))


class HECLoss(nn.Module):
    """Hyperbolic Entailment Cone loss applied to post-W_Q tensors at the HAA layer.

    Reads tensors captured by the MHA module during forward:
      - mha._last_q_post_wq:  shape (B, H, N+1, D_head) — post-W_Q Q (CLS at idx 0)
      - mha._last_k_post_wq:  shape (B, H, N+1, D_head) — post-W_Q K
      - mha._last_q_x0_per_token: shape (B, H, N+1) — temporal coord x0 of Q (for B-aperture)
      - K (manifold curvature) is read from mha or passed at init.

    Args:
        mha_module: the MultiHeadAttention instance at the HAA layer (terminal layer).
                    The forward() method below reads its _last_q_post_wq, _last_k_post_wq,
                    _last_q_x0_per_token at loss-evaluation time.
        beta: scalar β learnable from the HAA layer (same β used in score formula).
        curvature_K: float, manifold curvature (default 1.0 for Lorentz with K=1).
        margin: float, the γ hinge margin for negatives. Default 0.1 rad.
        negative_mode: 'naive' or 'supcon'. supcon requires labels.
        head_reduce: 'mean' or 'first' — collapse multi-head into single Q/K. mean = average
                     across heads (cheap, captures aggregate cone), first = head 0 only
                     (matches HAA telemetry convention). Default 'mean'.
        warmup_epochs: int, linear warmup of loss weight to 1.0 over this many epochs.
    """

    def __init__(self,
                 beta_provider,
                 curvature_K: float = 1.0,
                 margin: float = 0.1,
                 negative_mode: str = 'naive',
                 head_reduce: str = 'mean',
                 warmup_epochs: int = 5,
                 eps: float = 1e-5):
        super().__init__()
        assert negative_mode in ('naive', 'supcon'), \
            f"negative_mode must be 'naive' or 'supcon', got {negative_mode!r}"
        assert head_reduce in ('mean', 'first'), \
            f"head_reduce must be 'mean' or 'first', got {head_reduce!r}"
        self.beta_provider = beta_provider  # callable () -> scalar tensor
        self.K = float(curvature_K)
        self.margin = float(margin)
        self.negative_mode = negative_mode
        self.head_reduce = head_reduce
        self.warmup_epochs = int(warmup_epochs)
        self.eps = float(eps)

        # Buffers for telemetry — populated each forward
        self.last_loss = torch.tensor(0.0)
        self.last_n_pos = 0
        self.last_n_neg = 0
        self.last_mean_theta_pos = 0.0
        self.last_mean_theta_neg = 0.0
        self.last_mean_psi = 0.0

    def _reduce_heads(self, q: torch.Tensor, k: torch.Tensor,
                      q_x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q,k: (B, H, T, D)   q_x0: (B, H, T)  ->  (B, T, D), (B, T, D), (B, T)"""
        if self.head_reduce == 'mean':
            return q.mean(dim=1), k.mean(dim=1), q_x0.mean(dim=1)
        else:
            return q[:, 0], k[:, 0], q_x0[:, 0]

    def _compute_Z(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute angular signal Z(u, v) using HAA's inner-product formulation.

        u, v: (B, P, D) Lorentz vectors. Returns Z of shape (B, P) for paired (u_i, v_i)
        or use broadcasting for cross-pair computations externally.

        For now compute pairwise per-batch: u (B, 1, D) vs v (B, P, D).
        """
        # Lorentz inner products
        # Component 0 is timelike: <x,y>_L = -x0*y0 + sum_i xi*yi
        ux0 = u[..., 0:1]
        us = u[..., 1:]
        vx0 = v[..., 0:1]
        vs = v[..., 1:]
        inner_uv = (-ux0 * vx0) + (us * vs).sum(dim=-1, keepdim=True)
        inner_uo = -ux0  # u with origin (1, 0, ..., 0)*sqrt(K=1) ; <u, O>_L = -ux0
        # ||u_QK_tangent||^2 = inner_uv^2 / K - K  (HAA inner-product formulation)
        sqnorm_QK = (inner_uv ** 2) / self.K - self.K
        # ||u_QO_tangent||^2 = ||us||^2 (spatial-norm-squared)
        sqnorm_QO = (us ** 2).sum(dim=-1, keepdim=True)
        denom = torch.sqrt(sqnorm_QK.clamp(min=0.0) * sqnorm_QO.clamp(min=0.0) + (5e-3) ** 2)
        inner_tangent = inner_uv * inner_uo  # tangent inner product
        Z = inner_tangent / denom
        return Z.squeeze(-1)  # (B, P)

    def _compute_psi(self, q_x0: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Compute cone half-aperture psi(u) = asin(B(u)) from q_x0 (temporal coord)
        and current beta. Uses softplus-smoothed B, matching HAA score formula.
        """
        # c_tilde = acosh(q_x0 / sqrt(K))
        arg_acosh = (q_x0 / (self.K ** 0.5)).clamp(min=1.0 + 1e-3)
        c_tilde = torch.acosh(arg_acosh)
        # arg_B = 1 - beta^2 / sinh^2(c_tilde)
        sinh_c = torch.sinh(c_tilde)
        arg_B = 1.0 - (beta ** 2) / (sinh_c ** 2 + 1e-8)
        # Smooth via softplus to match HAA convention
        B_inner = F.softplus(4.0 * arg_B) / 4.0
        B = torch.sqrt(B_inner + 1e-8)
        psi = _safe_asin(B, eps=self.eps)
        return psi  # (B,) — psi for CLS at each batch item

    def forward(self,
                mha_module,
                labels: Optional[torch.Tensor] = None,
                epoch: int = 0) -> torch.Tensor:
        """Compute L_HEC. Returns scalar loss tensor (with grad).

        Args:
            mha_module: the HAA-active MHA from which to read post-W_Q tensors.
            labels: (B,) tensor of class indices. Required if negative_mode='supcon'.
            epoch: current training epoch (for warmup weight).
        """
        # Gate by warmup: weight linearly ramps 0 -> 1 over warmup_epochs
        if self.warmup_epochs > 0:
            w = min(1.0, max(0.0, epoch / float(self.warmup_epochs)))
        else:
            w = 1.0

        # Read captured tensors
        q = getattr(mha_module, '_last_q_post_wq', None)
        k = getattr(mha_module, '_last_k_post_wq', None)
        q_x0 = getattr(mha_module, '_last_q_x0_per_token', None)
        if q is None or k is None or q_x0 is None:
            # HAA layer didn't run (e.g., baseline mode) — return zero
            zero = torch.zeros((), device=next(mha_module.parameters()).device,
                               requires_grad=False)
            self.last_loss = zero.detach()
            self.last_n_pos = 0
            self.last_n_neg = 0
            return zero

        # Reduce heads — q, k: (B, T, D); q_x0: (B, T)
        q, k, q_x0 = self._reduce_heads(q, k, q_x0)
        B, T, D = q.shape
        if T < 2:
            zero = torch.zeros((), device=q.device, requires_grad=False)
            self.last_loss = zero.detach()
            return zero

        # Slice CLS and patches
        cls_q = q[:, 0:1, :]            # (B, 1, D)
        patches_k = k[:, 1:, :]         # (B, P, D)  P = T-1
        cls_q_x0 = q_x0[:, 0]           # (B,)
        P = patches_k.shape[1]

        # Get current beta
        beta = self.beta_provider()  # scalar tensor

        # Compute psi for each CLS (one per batch item)
        psi = self._compute_psi(cls_q_x0, beta)  # (B,)

        # ----------------- POSITIVES: (CLS_i, patch in image i) -----------------
        # For each batch item i, compute Z(cls_q_i, patches_k_i) — shape (B, P)
        # Vectorise: cls_q (B,1,D), patches_k (B,P,D). _compute_Z handles broadcasting.
        Z_pos = self._compute_Z(cls_q, patches_k)  # (B, P)
        theta_pos = _safe_acos(-Z_pos, eps=self.eps)  # (B, P)
        psi_pos = psi.unsqueeze(1).expand(-1, P)  # (B, P)
        loss_pos = F.relu(theta_pos - psi_pos)
        loss_pos_mean = loss_pos.mean()

        # ----------------- NEGATIVES: (CLS_i, patch in image j != i) -----------------
        # Build (B, B-1, P) tensor of cross-image patches for each anchor i.
        # Use index_select with a roll-style index to avoid materialising (B, B, P, D).
        # Simpler: compute Z for all (B, B, P) anchor-target combos, mask out i==j.
        if B > 1:
            # cls_q: (B, 1, 1, D)   patches_k: (1, B, P, D)
            cls_q_exp = cls_q.unsqueeze(1)        # (B, 1, 1, D)
            patches_k_exp = patches_k.unsqueeze(0)  # (1, B, P, D)
            Z_all = self._compute_Z(cls_q_exp.expand(B, B, 1, D).reshape(B * B, 1, D),
                                    patches_k_exp.expand(B, B, P, D).reshape(B * B, P, D))
            Z_all = Z_all.reshape(B, B, P)
            # Anchor-target mask: keep (i, j) where j != i
            eye = torch.eye(B, device=Z_all.device, dtype=torch.bool)
            same_image_mask = eye.unsqueeze(2).expand(B, B, P)  # True at i==j

            if self.negative_mode == 'supcon' and labels is not None:
                # Also mask out (i, j) where labels[i] == labels[j]
                same_class_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B, B)
                same_class_mask = same_class_mask.unsqueeze(2).expand(B, B, P)
                valid_neg_mask = ~(same_image_mask | same_class_mask)
            else:
                valid_neg_mask = ~same_image_mask

            theta_neg_all = _safe_acos(-Z_all, eps=self.eps)
            psi_neg = psi.unsqueeze(1).unsqueeze(2).expand(B, B, P)  # anchor's psi
            margin_term = F.relu(psi_neg - theta_neg_all + self.margin)
            # Apply mask and average over valid pairs
            valid_neg_mask_f = valid_neg_mask.float()
            n_neg = valid_neg_mask_f.sum().clamp(min=1.0)
            loss_neg_mean = (margin_term * valid_neg_mask_f).sum() / n_neg
            self.last_n_neg = int(valid_neg_mask.sum().item())
        else:
            loss_neg_mean = torch.zeros((), device=q.device)
            self.last_n_neg = 0

        # Telemetry
        self.last_n_pos = int(loss_pos.numel())
        self.last_mean_theta_pos = float(theta_pos.mean().item())
        self.last_mean_psi = float(psi.mean().item())

        loss = w * (loss_pos_mean + loss_neg_mean)
        self.last_loss = loss.detach()
        return loss
