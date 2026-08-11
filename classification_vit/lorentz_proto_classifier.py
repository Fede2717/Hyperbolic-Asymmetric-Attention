import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from haa_auxiliary_loss import build_hyperbolic_prototypes
from hierarchy_loader import load_hierarchy


class LorentzPrototypeClassifier(nn.Module):
    """Prototype-softmax classification head on the Lorentz hyperboloid.
    Logits = -d_L^2(cls, p_c) / T, c in fine prototypes.
    Replaces LorentzMLR for Design 2 of the Phase 2 pipeline.

    Dataset constraint (S-1): currently CIFAR-100 only. The fine prototypes
    are constructed against `cifar100_hierarchy.FINE_TO_SUPER`, which is a
    100-class fine -> 20-class super mapping specific to CIFAR-100. Running
    with `num_classes != NUM_FINE` raises NotImplementedError. To extend to
    other datasets, generalise build_hyperbolic_prototypes and provide an
    equivalent fine -> super hierarchy module.

    Curvature constraint (S-2): self.K is a Python float captured at init
    and used in forward() to compute the prototype distance. Constructing
    this classifier under a learnable curvature (manifold.k is a Parameter
    with requires_grad=True) raises NotImplementedError, since prototypes
    are fixed at init while manifold.k would evolve, breaking the
    geometric consistency between prototypes and queries.
    """
    def __init__(self,
                 manifold,
                 hidden_dim_lorentz: int,
                 num_classes: int,
                 K: float = 1.0,
                 proto_seed: int = 42,
                 d_s: float = 0.3,
                 d_f_mid: float = 1.175,
                 T_init: float = 1.0,
                 dataset_name: str = 'CIFAR-100',
                 hierarchy_path: str = None):
        super().__init__()
        FINE_TO_SUPER, NUM_FINE, NUM_SUPER = load_hierarchy(
            dataset_name, hierarchy_path=hierarchy_path)
        if num_classes != NUM_FINE:
            raise NotImplementedError(
                f"LorentzPrototypeClassifier expects num_classes == NUM_FINE "
                f"for dataset '{dataset_name}' (NUM_FINE={NUM_FINE}; got "
                f"{num_classes}). Generalise build_hyperbolic_prototypes "
                f"and supply an equivalent fine->super hierarchy module "
                f"if you need a different layout.")
        # S-2 guard: self.K is captured as a Python float here and used in
        # forward(). If manifold.k is a learnable Parameter, manifold.k evolves
        # while self.K does not, producing inconsistent geometry between
        # prototypes (built at init with fixed K) and queries. Refuse the
        # combination explicitly rather than silently diverging.
        _k_attr = getattr(manifold, 'k', None)
        if isinstance(_k_attr, torch.nn.Parameter) and _k_attr.requires_grad:
            raise NotImplementedError(
                "LorentzPrototypeClassifier does not support learnable "
                "curvature (manifold.k is a Parameter with requires_grad=True). "
                "Prototypes are constructed at init with fixed K and cannot "
                "track manifold.k as it evolves. Pass --learn_k=False, or "
                "extend the classifier to rebuild prototypes per forward().")
        self.manifold = manifold
        self.K = K
        self.num_super = NUM_SUPER
        self.num_fine = NUM_FINE

        lut = torch.zeros(NUM_FINE, dtype=torch.long)
        for f, s in FINE_TO_SUPER.items():
            lut[f] = s
        protos = build_hyperbolic_prototypes(
            num_super=NUM_SUPER,
            num_fine=NUM_FINE,
            hidden_dim=hidden_dim_lorentz,
            fine_to_super_lut=lut,
            K=K,
            seed=proto_seed,
            d_s=d_s,
            d_f_mid=d_f_mid,
        )
        self.register_buffer('prototypes', protos, persistent=False)

        # Positive temperature via softplus(log_T).
        # log_T = log(exp(T_init) - 1) gives softplus(log_T) = T_init at init.
        # S-5 guard: log(exp(T_init) - 1) -> -inf as T_init -> 0 and is undefined
        # for T_init <= 0. Any T_init at or below 1e-3 is rejected explicitly
        # rather than producing NaN at construction time.
        if T_init <= 1e-3:
            raise ValueError(
                f"proto_T_init must be > 1e-3 to keep the softplus inverse "
                f"finite; got T_init={T_init}.")
        _log_T_init = math.log(math.exp(T_init) - 1.0)
        self.log_T = nn.Parameter(torch.tensor([_log_T_init]))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_T) + 1e-3

    def forward(self, cls_lorentz: torch.Tensor) -> torch.Tensor:
        # cls_lorentz: [B, hidden_dim+1] post-final_ln Lorentz CLS.
        fine_protos = self.prototypes[self.num_super:
                                      self.num_super + self.num_fine]   # [F, D]
        # Lorentz inner product <cls, p>_L = -t_cls*t_p + s_cls·s_p
        inner = (-cls_lorentz[..., 0:1] * fine_protos[..., 0:1].T
                 + cls_lorentz[..., 1:] @ fine_protos[..., 1:].T)        # [B, F]
        arg = -inner / self.K
        u = (arg - 1.0).clamp_min(0.0)
        sqrt_term = torch.sqrt(u * (2.0 + u) + 1e-12)
        # S-7a: full Lorentz geodesic distance squared d_L^2 = K * acosh^2(arg).
        # For K=1.0 (all past and current experiments) this is numerically
        # identical to the previous `acosh^2` form. The K factor matters only
        # when curvature is varied, in which case it keeps the geometry correct
        # and the temperature in a meaningful absolute scale.
        d2 = self.K * torch.log1p(u + sqrt_term).pow(2)                  # [B, F]
        T = self.temperature
        return -d2 / T
