import torch
import torch.nn as nn
import math

from lib.Euclidean.blocks.transformer_blocks import TransformerEncoder, Embedding

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzMLR, LorentzLayerNorm, LorentzFullyConnected
from lib.lorentz.blocks.transformer_blocks import LorentzTransformerEncoder, LorentzEmbedding


import math

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (LorentzFullyConnected)):
        nn.init.xavier_normal_(m.weight.weight)
        if m.weight.bias is not None:
            nn.init.constant_(m.weight.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class ViT(nn.Module):
    def __init__(
        self,
        manifold=None,
        num_layers=9,
        img_dim=[3,32,32],
        num_classes=100,
        patch_size=4,
        heads=12,
        hidden_dim=192,
        mlp_dim=384,
        dropout=0.0,
        remove_linear=False,
        active_haa_layers=None,
        beta_proportional=False,
        tau_init=1.0,
        lambda_init=1.0,
        learn_lambda=True,
        beta_init_override=None,
        B_smooth='softplus',
        B_softplus_temp=4.0,
        use_cls_depth_residual=False,
        use_q_depth_mlp=False,
        disable_B=False,
    ):
        super(ViT, self).__init__()
        self.manifold = manifold

        if active_haa_layers is None:
            active_haa_layers = []

        c,h,w = img_dim

        self.patch_size = patch_size
        self.num_patches = (h//patch_size) * (w//patch_size)

        self.patch_dim = self.patch_size**2 * c
        self.hidden_dim = hidden_dim

        self.num_tokens = self.num_patches+1 # +1 because of CLS token

        self.embed = self._get_embedding()

        max_haa_idx = max(active_haa_layers) if active_haa_layers else num_layers - 1
        enc_list = []
        for idx in range(num_layers):
            use_haa = (idx in active_haa_layers)
            if beta_proportional:
                _beta_max = 1.0
                _beta_val = _beta_max * idx / max(num_layers - 1, 1)
                _beta_init = _beta_val
            else:
                _beta_init = None
            # STEP 4 / CHANGE-3: --beta_init_override forces a single beta init
            # value across ALL HAA layers regardless of mode (overrides beta_proportional).
            if beta_init_override is not None:
                _beta_init = beta_init_override
            layer = self._get_transformerEncoder(hidden_dim, mlp_dim, self.num_patches, heads, dropout, use_haa=use_haa, beta_init_val=_beta_init, tau_init=tau_init, lambda_init=lambda_init,
                                                 learn_lambda=learn_lambda,
                                                 B_smooth=B_smooth, B_softplus_temp=B_softplus_temp,
                                                 use_q_depth_mlp=use_q_depth_mlp,
                                                 disable_B=disable_B)
            if hasattr(layer, 'mha'):
                layer.mha.layer_idx = idx
                layer.mha.max_layer_idx = max_haa_idx
            enc_list.append(layer)
        self.encoder = nn.Sequential(*enc_list)

        # STEP 0 Action B: Verify layer_idx assignment for all HAA layers.
        if active_haa_layers:
            n_haa = sum(1 for blk in self.encoder
                        if hasattr(blk, 'mha') and blk.mha.use_haa)
            assert n_haa > 0, "No HAA layers found despite non-empty active_haa_layers"
            for i, blk in enumerate(self.encoder):
                if hasattr(blk, 'mha') and blk.mha.use_haa:
                    assert blk.mha.layer_idx >= 0, \
                        f"Block {i}: layer_idx not assigned (still -1)"
                    assert blk.mha.max_layer_idx == max(active_haa_layers), \
                        (f"Block {i}: max_layer_idx={blk.mha.max_layer_idx}, "
                         f"expected {max(active_haa_layers)}")
            print(f"[HAA INIT] Verified {n_haa} HAA layers with valid layer_idx assignment.",
                  flush=True)

        # Stage 2.1 Path B (revised): per-CLS radial scaling residual.
        # Decouples CLS depth from spatial norm so L_proto / L_radvar operate
        # on an actual depth DOF. Active only when use_cls_depth_residual=True.
        self.use_cls_depth_residual = use_cls_depth_residual
        if use_cls_depth_residual:
            if type(self.manifold) is not CustomLorentz:
                raise RuntimeError(
                    "CLS depth residual requires the Lorentz manifold; "
                    f"got {type(self.manifold)}")
            from classification_vit.cls_depth_residual import CLSDepthResidual
            self.cls_depth_residual = CLSDepthResidual(
                manifold=self.manifold,
                hidden_dim_lorentz=hidden_dim + 1,
            )
        else:
            self.cls_depth_residual = None

        self.final_ln = self._get_layerNorm(hidden_dim)

        if remove_linear:
            self.predictor = None
        else:
            self.predictor = self._get_predictor(hidden_dim, num_classes)

        self.apply(init_weights)

        for blk in self.encoder:
            if hasattr(blk, 'mha') and getattr(blk.mha, 'use_q_depth_mlp', False):
                qdm = blk.mha.q_depth_mlp
                nn.init.zeros_(qdm[-1].weight)
                nn.init.constant_(qdm[-1].bias, math.log(0.6 / 0.4))
                nn.init.normal_(qdm[0].weight, std=0.01)
                nn.init.zeros_(qdm[0].bias)

        # Re-zero the CLS depth residual MLP final layer AFTER apply(init_weights),
        # which would otherwise xavier-init it and break the alpha=1.0 init guarantee.
        # FIX-ALPHA-SIGMOID: under alpha = 0.1 + 1.4 * sigmoid(raw), bias must
        # equal logit(0.6) = log(0.6 / 0.4) ≈ 0.405 to keep alpha=1.0 at init.
        if self.cls_depth_residual is not None:
            nn.init.zeros_(self.cls_depth_residual.mlp[-1].weight)
            nn.init.constant_(self.cls_depth_residual.mlp[-1].bias,
                              math.log(0.6 / 0.4))   # FIX-ALPHA-SIGMOID: gives α=1.0 at init
            nn.init.normal_(self.cls_depth_residual.mlp[0].weight, std=0.01)
            nn.init.zeros_(self.cls_depth_residual.mlp[0].bias)

    def forward(self, x):
        x = self.patchify(x)
        x = self.embed(x)
        x = self.encoder(x)[:, 0]            # [B, hidden_dim+1] CLS Lorentz
        if self.cls_depth_residual is not None:
            x = self.cls_depth_residual(x)   # decouples depth from spatial norm
        x = self.final_ln(x)

        if self.predictor is not None:
            x = self.predictor(x)

        return x

    def patchify(self, x):
        """ Divide image into patches: (bs x C x H x W) -> (bs x N x L) """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.num_patches, -1)

        return out

    def _get_embedding(self):
        if self.manifold is None:
            return Embedding(self.hidden_dim, self.patch_dim, self.num_tokens)

        elif type(self.manifold) is CustomLorentz:
            return LorentzEmbedding(self.manifold, self.hidden_dim+1, self.patch_dim+1, self.num_tokens)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ViT.")

    def _get_transformerEncoder(self, hidden_dim, mlp_dim, num_patches, heads, dropout, use_haa=False, beta_init_val=None, tau_init=1.0, lambda_init=1.0,
                                learn_lambda=True,
                                B_smooth='softplus', B_softplus_temp=4.0,
                                use_q_depth_mlp=False, disable_B=False):
        if self.manifold is None:
            return TransformerEncoder(hidden_dim, mlp_dim, num_patches, heads, dropout)

        elif type(self.manifold) is CustomLorentz:
            return LorentzTransformerEncoder(self.manifold, hidden_dim+1, mlp_dim+1, num_patches, heads, dropout, use_haa=use_haa, beta_init_val=beta_init_val, tau_init=tau_init, lambda_init=lambda_init,
                                              learn_lambda=learn_lambda,
                                              B_smooth=B_smooth, B_softplus_temp=B_softplus_temp,
                                              use_q_depth_mlp=use_q_depth_mlp,
                                              disable_B=disable_B)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ViT.")

    def _get_layerNorm(self, num_features):
        if self.manifold is None:
            return nn.LayerNorm(num_features)

        elif type(self.manifold) is CustomLorentz:
            return LorentzLayerNorm(self.manifold, num_features+1)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ViT.")

    def _get_predictor(self, in_features, num_classes):
        if self.manifold is None:
            return nn.Linear(in_features, num_classes)

        elif type(self.manifold) is CustomLorentz:
            return LorentzMLR(self.manifold, in_features+1, num_classes)

        else:
            raise RuntimeError(f"Manifold {type(self.manifold)} not supported in ViT.")


def vit(**kwargs):
    " Constructs a Vision Transformer (ViT) "
    model = ViT(**kwargs)
    return model

def lorentz_vit(k=1, learn_k=False, manifold=None, **kwargs):
    " Constructs fully hyperbolic Vision Transformer (ViT) "
    if manifold is None:
        manifold = CustomLorentz(k=k, learnable=learn_k)
    model = ViT(manifold, **kwargs)
    return model
