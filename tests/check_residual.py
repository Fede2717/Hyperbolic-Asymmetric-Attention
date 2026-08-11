"""Manifold-constraint validation for CLSDepthResidual.

Verifies:
  1. alpha = 1.0 exactly at init (zero-weight final layer with calibrated bias).
  2. Output equals input at init (no-op residual).
  3. Lorentz constraint <x,x>_L = -K preserved after a perturbed forward.
  4. alpha stays within the current (0.1, 1.5) bounds.
  5. Gradient flows through alpha (telemetry hook fires).

Run manually as a lightweight validation after training-environment setup.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lib.lorentz.manifold import CustomLorentz
from classification_vit.cls_depth_residual import CLSDepthResidual


def main():
    K = 1.0
    manifold = CustomLorentz(k=K, learnable=False)
    resid = CLSDepthResidual(manifold, hidden_dim_lorentz=193)

    # Build a batch of valid Lorentz points.
    torch.manual_seed(0)
    spatial = torch.randn(4, 192) * 0.5
    time = torch.sqrt(K + (spatial ** 2).sum(-1, keepdim=True))
    cls_in = torch.cat([time, spatial], dim=-1)              # [4, 193]

    # Sanity: input is on the manifold.
    inner_in = -cls_in[:, 0] ** 2 + (cls_in[:, 1:] ** 2).sum(-1)
    assert (inner_in + K).abs().max() < 1e-4, "input off-manifold"

    # Init: alpha must be exactly 1.0 -> output equals input.
    resid.eval()
    cls_out = resid(cls_in)
    assert torch.allclose(cls_out, cls_in, atol=1e-5), \
        f"alpha=1 init failed; max diff {(cls_out - cls_in).abs().max().item()}"
    assert (resid._last_alpha - 1.0).abs().max() < 1e-6, \
        "alpha telemetry not 1.0 at init"

    # Manifold preserved after a perturbed forward.
    resid.train()
    for p in resid.parameters():
        p.data.add_(0.05 * torch.randn_like(p.data))  # break the zero init
    cls_out = resid(cls_in)
    inner_out = -cls_out[:, 0] ** 2 + (cls_out[:, 1:] ** 2).sum(-1)
    assert (inner_out + K).abs().max() < 1e-4, \
        f"output off-manifold; max violation {(inner_out + K).abs().max().item()}"

    # Alpha bounded in the current implementation's (0.1, 1.5) range.
    a = resid._last_alpha
    assert (a > 0.1).all() and (a < 1.5).all(), \
        f"alpha out of bounds: min={a.min()}, max={a.max()}"

    # Gradient flows through alpha.
    cls_out = resid(cls_in)
    loss = cls_out[:, 0].sum()  # any scalar loss touching the output
    loss.backward()
    grad = resid._last_alpha_grad
    assert grad > 0, f"alpha gradient hook did not fire; got {grad}"

    print("CLS depth residual checks OK")


if __name__ == "__main__":
    main()
