import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lib.geoopt import ManifoldParameter

from lib.utils.drop_path import DropPath

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import (
    LorentzFullyConnected,
    LorentzLayerNorm,
    LorentzProjection,
LorentzAct
)


class LorentzEmbedding(nn.Module):
    def __init__(self, manifold: CustomLorentz, hidden_dim, patch_dim, num_tokens):
        super(LorentzEmbedding, self).__init__()
        self.manifold = manifold
        self.patch_embed = LorentzFullyConnected(self.manifold, patch_dim, hidden_dim)
        self.cls_token = ManifoldParameter(self.manifold.random_normal(1, 1, hidden_dim), manifold=self.manifold)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim-1))

    def forward(self, x):
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))

        x = self.patch_embed(x)

        if torch.isnan(x).sum()>0 or torch.isinf(x).sum()>0:
            print("break")

        x = torch.cat([self.cls_token.repeat(x.size(0),1,1), x], dim=1)
        x = x.narrow(-1, 1, x.shape[-1]-1) + self.pos_embed

        return self.manifold.add_time(x)


class LorentzTransformerEncoder(nn.Module):
    def __init__(self, manifold: CustomLorentz, hidden, mlp_hidden, num_patches, heads, dropout,
                 stochastic_depth=0.1, use_haa=False, beta_init_val=None,
                 tau_init=1.0, lambda_init=1.0, learn_lambda=True,
                 B_smooth='softplus', B_softplus_temp=4.0,
                 use_q_depth_mlp=False, disable_B: bool = False):
        super(LorentzTransformerEncoder, self).__init__()

        self.manifold = manifold

        self.hidden = hidden
        self.mlp_hidden = mlp_hidden
        self.num_patches = num_patches
        self.heads = heads
        self.dropout = dropout

        self.ln1 = LorentzLayerNorm(manifold, hidden)
        self.mha = LorentzMultiHeadAttention(manifold, hidden, num_patches, heads, dropout, use_haa=use_haa, beta_init_val=beta_init_val, tau_init=tau_init, lambda_init=lambda_init,
                                             learn_lambda=learn_lambda,
                                             B_smooth=B_smooth, B_softplus_temp=B_softplus_temp,
                                             use_q_depth_mlp=use_q_depth_mlp,
                                             disable_B=disable_B)
        self.ln2 = LorentzLayerNorm(manifold, hidden)
        self.mlp = nn.Sequential(
            LorentzFullyConnected(manifold, hidden, mlp_hidden, activation=nn.GELU(), dropout=dropout),
            LorentzFullyConnected(manifold, mlp_hidden, hidden, dropout=dropout),
        )

        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        out = self.mha(self.ln1(x))
        out = self.drop_path(out.narrow(-1, 1, x.shape[-1]-1)) + x.narrow(-1, 1, x.shape[-1]-1)

        # Post-attention, pre-MLP CLS capture for L_proto.
        # `out` here is spatial-only (the first residual operates on spatial
        # coordinates). Re-project to Lorentz via add_time to produce a valid
        # manifold point. add_time is differentiable; capture is gradient-
        # preserving (no .detach()). This capture is correct pre-Step-11:
        # L_proto's gradient flows back through W_Q at layer 8 with one
        # residual split, providing the strongest HAA-depth supervision
        # available before alpha_q is introduced. Move to post-block when
        # Step 11 (alpha_q) lands.
        if hasattr(self.mha, 'use_haa') and self.mha.use_haa:
            _post_attn_full = self.manifold.add_time(out)
            self.mha._last_cls_lorentz = _post_attn_full[:, 0, :]
            # NEW: capture full post-attention token tensor for σ² diagnostic.
            # Detached: this is for measurement only, not loss.
            self.mha._last_post_attn_all_tokens = _post_attn_full.detach()
            # POST-ATTENTION σ² accumulation (what L_proto supervises CLS at,
            # extended to all tokens for the diagnostic).
            self.mha._accumulate_sigma2(_post_attn_full,
                                        'train_post' if self.mha.training else 'val_post')

        out = self.drop_path(self.mlp(self.ln2(self.manifold.add_time(out))).narrow(-1, 1, x.shape[-1]-1)) + out
        out = self.manifold.add_time(out)
        return out


class LorentzMultiHeadAttention(nn.Module):
    def __init__(self, manifold: CustomLorentz, num_features, num_patches, heads, dropout=0.0,
                 learn_scale=False, use_haa=False, beta_init_val=None,
                 tau_init=1.0, lambda_init=1.0, learn_lambda=True,
                 B_smooth='softplus', B_softplus_temp=4.0,
                 use_q_depth_mlp=False, disable_B: bool = False):
        super(LorentzMultiHeadAttention, self).__init__()
        # CHANGE-2: aperture-gradient regime selector (relu = legacy, softplus = fixed)
        self.B_smooth = B_smooth
        self.B_softplus_temp = B_softplus_temp
        self.disable_B = disable_B

        self.manifold = manifold

        self.num_features = num_features
        self.num_patches = num_patches
        self.heads = heads
        self.head_dim = (num_features-1)//heads
        self.temperature = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(self.head_dim**(-0.5)*torch.ones((1, heads, 1, 1)), requires_grad=learn_scale)

        self.softmax = nn.Softmax(dim=-1)

        self.q = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.k = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)
        self.v = LorentzFullyConnected(manifold, num_features, num_features, nheads=heads, bias=False)

        self.o = LorentzFullyConnected(manifold, num_features, num_features, dropout=dropout)

        self.use_haa = use_haa
        self.layer_idx = -1
        self.max_layer_idx = 8
        self._grad_norms = {}

        if use_haa:
            # τ_init=1.0 calibrated to spatial penalty dynamic range.
            # Spatial range ≈ [-λ·1.98, 0]; entailment range ≈ [-τ·1.40, +τ·0.36].
            # With λ_init=1.0, setting τ_init=1.0 gives magnitude ratio ≈ 1:1,
            # ensuring the entailment term contributes meaningful score-matrix
            # variance from epoch 0. Lower τ_init values produce sub-softmax-noise
            # entailment signal and prevent β/τ from receiving informative gradient
            # — a structural cause of the deadlock.
            if beta_init_val is not None and beta_init_val > 1e-6:
                _beta_raw_init = math.log(math.exp(beta_init_val) - 1.0)
            elif beta_init_val is not None:
                _beta_raw_init = -10.0   # softplus(-10) ≈ 4.5e-5 ≈ 0
            else:
                _beta_raw_init = math.log(math.exp(1.0) - 1.0)
            self.beta_raw   = nn.Parameter(torch.tensor([_beta_raw_init]))
            self.tau_raw    = nn.Parameter(torch.tensor([math.log(math.exp(tau_init) - 1.0)]))
            self.lambda_raw = nn.Parameter(torch.tensor([math.log(math.exp(lambda_init) - 1.0)]))
            if not learn_lambda:
                self.lambda_raw.requires_grad = False

            self.haa_alpha            = 0.0
            self.haa_tau              = 0.0
            self.haa_lambda           = 0.0
            self.haa_mean_c_tilde     = 0.0
            self.haa_mean_B           = 0.0
            self.haa_mean_Z           = 0.0
            self.haa_cone_sparsity    = 0.0
            self.haa_frac_near_origin = 0.0

            # NaN counters for Z telemetry (STEP 0 Action A / Item 8)
            self._z_nan_batch_count   = 0
            self._z_nan_total_calls   = 0
            self._z_nan_element_count = 0
            self._z_nan_element_total = 0
            # A5: sentry for the close-pair regime (norm_sq_QK ≤ 1e-6).
            self._z_nearzero_qk_count = 0

            # STEP 3 / CHANGE-4: live tensors exposed for auxiliary losses
            # (DirectionalAngularLoss reads Z; HHL reads CLS time coord;
            # B is retained for diagnostics/future use).
            self._last_B = None
            self._last_Z = None
            self._last_cls_time = None
            self._last_valid_mask = None
            self._last_cls_lorentz = None
            self._last_all_tokens_lorentz = None
            self._last_score = None
            # P4: origin-only counter (norm_sq_OQ ≤ 1e-6) — distinct failure
            # mode from generic Q-K close-pair masking.
            self._z_origin_count = 0

            # Sigma^2 per-image accumulators (per-epoch sum + image count).
            # Reset by train.py at the start of each epoch's train and val passes.
            self._sigma2_train_pre_sum   = 0.0
            self._sigma2_train_pre_count = 0
            self._sigma2_val_pre_sum     = 0.0
            self._sigma2_val_pre_count   = 0
            self._sigma2_train_post_sum  = 0.0
            self._sigma2_train_post_count = 0
            self._sigma2_val_post_sum    = 0.0
            self._sigma2_val_post_count  = 0
            # Post-attention all-tokens cache (set by encoder forward, detached).
            self._last_post_attn_all_tokens = None
            # K cached at construction so the σ² helper does not require a closure.
            self._K_cache = abs(float(manifold.k.item())) if hasattr(manifold, 'k') else 1.0
            self._sqrt_K_cache = self._K_cache ** 0.5

            import math as _math
            # Per-head Q-CLS depth scaling MLP (Step 11). Operates on each head's
            # spatial slice of the CLS Q vector and produces a per-head alpha_q in
            # (0.1, 1.5) used to rescale CLS-Q spatial component before the score
            # formula. alpha_q = 1.0 at init via bias = logit(0.6) = log(0.6/0.4).
            self.use_q_depth_mlp = False  # toggled by an external flag at construction time
            _spatial_per_head = (num_features - 1) // heads
            self.q_depth_mlp = torch.nn.Sequential(
                torch.nn.Linear(_spatial_per_head, 32),
                torch.nn.GELU(),
                torch.nn.Linear(32, 1),
            )
            torch.nn.init.zeros_(self.q_depth_mlp[-1].weight)
            torch.nn.init.constant_(self.q_depth_mlp[-1].bias, _math.log(0.6 / 0.4))
            torch.nn.init.normal_(self.q_depth_mlp[0].weight, std=0.01)
            torch.nn.init.zeros_(self.q_depth_mlp[0].bias)
            self._last_alpha_q = None       # [B, heads] float, detached
            self._last_alpha_q_grad = 0.0

            # NEW: post-W_Q full token tensor for L_radvar / L_spread supervision.
            # Captured AFTER self.q(x) and AFTER any q_depth_mlp scaling so that the
            # losses supervise the exact tensor the score formula reads.
            self._last_q_post_wq = None     # [B, heads, n, head_dim+1]
            self._last_k_post_wq = None     # [B, heads, n, head_dim+1]

            self.use_q_depth_mlp = use_q_depth_mlp

            assert num_features - 1 == heads * ((num_features - 1) // heads), \
                f"head_dim*heads != num_features-1: {heads}*{(num_features-1)//heads} != {num_features-1}"

    def _accumulate_sigma2(self, tokens, where: str):
        """Push per-image c_tilde variance into the appropriate accumulator.
        tokens: [B, n, hidden_dim+1] Lorentz points.
        where:  one of 'train_pre', 'train_post', 'val_pre', 'val_post'.
        Detached, no gradient impact."""
        with torch.no_grad():
            time = tokens[..., 0]
            arg = (time / self._sqrt_K_cache).clamp_min(1.0 + 1e-3)
            c_tilde = torch.acosh(arg)
            sigma2_pi = c_tilde.var(dim=-1, unbiased=False)   # [B]
            s = sigma2_pi.sum().item()
            n = sigma2_pi.numel()
            if where == 'train_pre':
                self._sigma2_train_pre_sum   += s
                self._sigma2_train_pre_count += n
            elif where == 'train_post':
                self._sigma2_train_post_sum   += s
                self._sigma2_train_post_count += n
            elif where == 'val_pre':
                self._sigma2_val_pre_sum   += s
                self._sigma2_val_pre_count += n
            elif where == 'val_post':
                self._sigma2_val_post_sum   += s
                self._sigma2_val_post_count += n

    def reset_sigma2_train(self):
        self._sigma2_train_pre_sum   = 0.0
        self._sigma2_train_pre_count = 0
        self._sigma2_train_post_sum   = 0.0
        self._sigma2_train_post_count = 0

    def reset_sigma2_val(self):
        self._sigma2_val_pre_sum   = 0.0
        self._sigma2_val_pre_count = 0
        self._sigma2_val_post_sum   = 0.0
        self._sigma2_val_post_count = 0

    def get_sigma2_train(self):
        pre  = (self._sigma2_train_pre_sum  / self._sigma2_train_pre_count)  if self._sigma2_train_pre_count  > 0 else 0.0
        post = (self._sigma2_train_post_sum / self._sigma2_train_post_count) if self._sigma2_train_post_count > 0 else 0.0
        return pre, post

    def get_sigma2_val(self):
        pre  = (self._sigma2_val_pre_sum  / self._sigma2_val_pre_count)  if self._sigma2_val_pre_count  > 0 else 0.0
        post = (self._sigma2_val_post_sum / self._sigma2_val_post_count) if self._sigma2_val_post_count > 0 else 0.0
        return pre, post

    def lorentz_expmap_aggregation(self, v, score):
        v_tangent = self.manifold.logmap0(v)
        weighted_v_tangent = torch.matmul(score, v_tangent)
        sum_weights = score.sum(dim=-1, keepdim=True)
        mean_v_tangent = weighted_v_tangent / (sum_weights + 1e-8)
        mean_v = self.manifold.expmap0(mean_v_tangent)
        return mean_v

    def forward(self, x):
        b, n, l = x.size()

        if self.use_haa:
            # P5: full pre-attention token tensor — gradient-preserving
            # reference used by L_betacap to derive the geometric envelope.
            self._last_all_tokens_lorentz = x
            # PRE-MHA σ² accumulation (input tokens — what L_radvar supervises).
            self._accumulate_sigma2(x, 'train_pre' if self.training else 'val_pre')

        q = self.q(x)
        if self.use_haa and self.use_q_depth_mlp:
            # q shape after self.q is [B, heads, n, head_dim+1] Lorentz.
            # Operate on token 0 (CLS) only, per head, on the spatial slice.
            _cls_q_space = q[:, :, 0:1, 1:]                                # [B, h, 1, head_dim]
            _raw = self.q_depth_mlp(_cls_q_space.squeeze(2)).squeeze(-1)   # [B, h]
            _alpha_q = 0.1 + 1.4 * torch.sigmoid(_raw)                     # [B, h] in (0.1, 1.5)
            self._last_alpha_q = _alpha_q.detach()
            if self.training and _alpha_q.requires_grad:
                _alpha_q.register_hook(
                    lambda g: setattr(self, '_last_alpha_q_grad',
                                      g.detach().abs().mean().item()))
            # Apply per-head scaling to CLS spatial; reproject onto Lorentz.
            _cls_space_scaled = _alpha_q[:, :, None, None] * _cls_q_space   # [B, h, 1, head_dim]
            _patches_space = q[:, :, 1:, 1:]                                 # [B, h, n-1, head_dim]
            _q_space_full = torch.cat([_cls_space_scaled, _patches_space], dim=2)  # [B, h, n, head_dim]
            q = self.manifold.add_time(_q_space_full)                        # back to [B, h, n, head_dim+1]
        k = self.k(x)
        v = self.v(x)

        if self.use_haa:
            # Capture post-W_Q tensors for L_radvar / L_spread. Gradient-attached
            # (no .detach()) — these tensors feed losses that must drive W_Q.
            self._last_q_post_wq = q
            self._last_k_post_wq = k
            q_time  = q.narrow(-1, 0, 1)
            k_time  = k.narrow(-1, 0, 1)
            q_space = q.narrow(-1, 1, q.shape[-1] - 1)
            k_space = k.narrow(-1, 1, k.shape[-1] - 1)

            # STEP 3 / CHANGE-4: capture CLS token time coord (with gradient)
            # for HHL. q_time shape: [b, h, n, 1]; CLS is token index 0.
            self._last_cls_time = q_time[:, :, 0:1, :]

            sqrt_k = torch.sqrt(self.manifold.k)
            K = self.manifold.k

            # --- Lorentz inner products ---
            inner_QK = (-q_time @ k_time.transpose(-1, -2)
                        + q_space @ k_space.transpose(-1, -2))
            inner_OQ = -sqrt_k * q_time                    # [b,h,n,1]
            inner_OK = -sqrt_k * k_time.transpose(-1, -2)  # [b,h,1,n]

            # --- Angular signal Z (inner-product formulation, no HLoC) ---
            # CHANGE-1: Additive ε on each norm separately, eliminating the bias
            # toward sign(numer_Z) when ‖q_space‖ → 0. Original ε=5e-3 inside
            # sqrt forced Z toward sign(numer_Z) for tokens near origin (Phase 0).
            EPS_Z_INNER = 1e-12
            norm_sq_QK  = (inner_QK.pow(2) / K - K).clamp_min(0.0)
            norm_sq_OQ  = (inner_OQ.pow(2) / K - K).clamp_min(0.0)
            numer_Z     = inner_OK + (inner_QK * inner_OQ) / K
            norm_QK     = torch.sqrt(norm_sq_QK + EPS_Z_INNER)
            norm_OQ     = torch.sqrt(norm_sq_OQ + EPS_Z_INNER)
            denom_Z     = norm_QK * norm_OQ
            Z_raw       = numer_Z / denom_Z

            # STEP 0 Action A / Item 8: element-wise NaN telemetry — never
            # full-matrix zeroing (full zeroing collapses entailment to depth-only,
            # destroying K-structure for the entire attention row).
            nan_mask = torch.isnan(Z_raw)
            if nan_mask.any():
                self._z_nan_batch_count   += 1
                self._z_nan_element_count += nan_mask.sum().item()
            self._z_nan_element_total += Z_raw.numel()
            self._z_nan_total_calls   += 1

            if not self.training and nan_mask.any():
                raise RuntimeError(
                    f"[HAA L{self.layer_idx}] NaN in Z during eval — "
                    f"{nan_mask.sum().item()}/{Z_raw.numel()} elements. "
                    "Eval metrics would be fabricated. Diagnose inner_QK.")

            # CHANGE-1 + A5: Validity mask (degenerate spatial component → Z=0,
            # geometrically neutral) combined with NaN rescue into one torch.where.
            # A5: mask is symmetric in Q and K — close-pair regime drives
            # norm_sq_QK toward 0, blowing up the denominator before the NaN
            # check fires. Catching it here keeps Z_safe finite.
            mask_valid = ((norm_sq_QK > 1e-6) & (norm_sq_OQ > 1e-6)).expand_as(Z_raw)
            nearzero_qk = (norm_sq_QK <= 1e-6)
            if nearzero_qk.any():
                self._z_nearzero_qk_count += nearzero_qk.sum().item()
            # P4: origin-only failure (norm_sq_OQ near 0) tracked separately.
            self._z_origin_count += (norm_sq_OQ <= 1e-6).sum().item()
            Z_safe = torch.where(
                mask_valid & ~nan_mask,
                Z_raw.clamp(-1.0, 1.0),
                torch.zeros_like(Z_raw))
            # P1: expose joint validity mask so DirectionalAngularLoss can
            # restrict its mean to geometrically valid pairs only.
            self._last_valid_mask = (mask_valid & ~nan_mask).detach()

            if self._z_nan_batch_count % 100 == 1 and nan_mask.any():
                print(f"[HAA L{self.layer_idx}] Z NaN rescue: "
                      f"cumulative element rate: "
                      f"{100*self._z_nan_element_count/max(1,self._z_nan_element_total):.4f}%",
                      flush=True)

            # --- Radial depth and aperture B ---
            c_tilde = torch.acosh((q_time / sqrt_k).clamp_min(1.0 + 1e-3))
            beta    = F.softplus(self.beta_raw)
            sinh_c  = torch.sinh(c_tilde)
            arg_B = 1.0 - (beta / sinh_c).pow(2)
            # CHANGE-2: Softplus floor restores β-gradient in the shallow regime.
            # F.relu kills ∂B/∂β when arg_B < 0 (all shallow tokens → β frozen).
            # Softplus provides exponentially small but non-zero gradient everywhere,
            # allowing β to receive aggregate signal even when no individual token
            # is in the active (arg_B > 0) regime.
            # scale=4.0 → softplus(4·x)/4 ≈ relu(x) for |x|>1, smooth for |x|<1.
            if self.B_smooth == 'softplus':
                arg_B_smooth = F.softplus(self.B_softplus_temp * arg_B) / self.B_softplus_temp
                B = torch.sqrt(arg_B_smooth + 1e-8)
            else:  # legacy relu (backward compatibility)
                B = torch.sqrt(F.relu(arg_B) + 1e-8)

            # --disable_B: force B to the neutral constant 1.0. Preserves the
            # entail_pen formula tau*Phi(B + Z - m) in the linear branch for
            # all Z in [-1, 1]. beta_raw remains a Parameter for checkpoint
            # compatibility but receives zero gradient.
            if self.disable_B:
                B = torch.ones_like(B)

            # STEP 3 / CHANGE-4: expose B and Z for DirectionalAngularLoss
            # (uses Z only; B kept for diagnostics).
            self._last_B = B
            self._last_Z = Z_safe

            # Gradient hooks for diagnostic layers only (first and last HAA layer)
            if self.training and self.layer_idx in (0, self.max_layer_idx):
                c_tilde.register_hook(
                    lambda g: self._grad_norms.update({'c_tilde': g.norm().item()}))
                B.register_hook(
                    lambda g: self._grad_norms.update({'B': g.norm().item()}))

            # --- Spatial penalty (soft-clamped log-cosh, δ₀=15) ---
            lam         = F.softplus(self.lambda_raw)
            safe_cosh_d = (-inner_QK / K).clamp_min(1.0 + 1e-3)
            dist_raw    = sqrt_k * torch.acosh(safe_cosh_d)
            d_soft      = 40.0 * torch.tanh(dist_raw / 40.0)
            scaled_d    = d_soft / 15.0
            H           = scaled_d + F.softplus(-2.0 * scaled_d) - math.log(2.0)
            spatial_pen = -lam * H

            # --- Entailment penalty (margin-softplus, m=0.1 fixed) ---
            tau       = F.softplus(self.tau_raw)
            _m        = 0.1
            _sp_neg_m = math.log(1.0 + math.exp(-_m))  # precomputed scalar constant
            Phi       = F.softplus((B + Z_safe) - _m) - _sp_neg_m
            entail_pen = -tau * Phi

            # --- Score and attention weights ---
            score_matrix = spatial_pen + entail_pen
            score = self.softmax(score_matrix / self.temperature)
            self._last_score = score.detach()  # [B, h, n_q, n_k] for CLS-row telemetry

            if not self.training:
                frac_near_origin          = (arg_B < 0).float().mean().item()
                self.haa_frac_near_origin = frac_near_origin
                self.haa_alpha            = beta.item()
                self.haa_tau              = tau.item()
                self.haa_lambda           = lam.item()
                self.haa_mean_c_tilde     = c_tilde.mean().item()
                self.haa_mean_B           = B.mean().item()
                self.haa_mean_Z           = Z_safe.mean().item()
                self.haa_cone_sparsity    = ((B + Z_safe) <= 0).float().mean().item()
        else:
            dists = -self.manifold.csqdist(q, k) * self.scale.expand((b, self.heads, 1, 1))
            score = self.softmax(dists / self.temperature)

        attn = self.lorentz_expmap_aggregation(v, score).permute(0, 2, 1, 3)

        # Lorentz direct concatenation of heads
        attn_space = attn.narrow(-1, 1, attn.shape[-1]-1).reshape(b, n, -1)
        attn_time = attn.narrow(-1, 0, 1).reshape(b, n, -1)
        time_sq_arg = torch.sum(attn_time ** 2, dim=-1, keepdim=True) - ((self.heads - 1) * self.manifold.k)
        # CHANGE-6: Detect FP32 drift in time-coordinate accumulation (eval only).
        # If this fires, downstream HAA measurements on these tokens are corrupted.
        if (not self.training) and time_sq_arg.min() <= 0:
            import warnings
            warnings.warn(
                f"[L{self.layer_idx}] FP32 drift detected: time_sq_arg.min()="
                f"{time_sq_arg.min().item():.2e}. Manifold constraint violated. "
                f"HAA geometric measurements unreliable on this batch.")
        attn_time_rescaled = torch.sqrt(time_sq_arg.clamp_min(1e-8))
        attn = torch.concat((attn_time_rescaled, attn_space), dim=-1)

        o = self.o(attn)
        return o
