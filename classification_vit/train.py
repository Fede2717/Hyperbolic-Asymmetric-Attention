# -----------------------------------------------------
# Change working directory to parent 
import os
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import torch
from torch.nn import DataParallel
import argparse
import configargparse
from tqdm import tqdm
import random
import numpy as np
import csv  # Import CSV module for saving the metrics
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import datetime

from utils.initialize import (
    select_dataset, select_model, select_optimizer, load_checkpoint,
    attach_hyperbolic_prototypes,
)
from lib.utils.utils import AverageMeter, accuracy
from lib.utils.mix import cutmix_data, mixup_data, mixup_criterion
from lib.utils.losses import LabelSmoothingCrossEntropy
from haa_diagnostics import log_haa_epoch_metrics, log_haa_deep_diagnostics
from haa_auxiliary_loss import build_aux_losses

DEEP_EPOCHS = frozenset({1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90})


def _haa_mhas_iter(model):
    # Walk submodules (instead of iterating base.encoder directly) because
    # base = ViTClassifier exposes `.encoder` as the inner ViT module — an
    # nn.Module without __iter__. The Sequential of blocks lives one level
    # deeper. named_modules() reaches them regardless of wrapping depth.
    from lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention
    base = model.module if hasattr(model, 'module') else model
    mhas = [m for _, m in base.named_modules()
            if isinstance(m, LorentzMultiHeadAttention) and getattr(m, 'use_haa', False)]
    mhas.sort(key=lambda m: m.layer_idx)
    for m in mhas:
        yield m


def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='Image classification training', add_help=True)

    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str,
                        help="Path to config file.")

    # Output settings
    parser.add_argument('--exp_name', default="test", type=str,
                        help="Name of the experiment.")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Path for output files. Relative paths are resolved from the current working directory.")
    parser.add_argument('--log_dir', default="runs", type=str,
                        help="Base directory for TensorBoard logs. Relative paths are resolved from the current working directory.")

    # General settings
    parser.add_argument('--device', default="cuda:0",
                        type=lambda s: [str(item) for item in s.replace(' ', '').split(',')],
                        help="CUDA device or comma-separated CUDA devices (e.g. cuda:0 or cuda:0,cuda:1). CPU execution is not supported.")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"],
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int,
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help="Path to model checkpoint (weights, optimizer, epoch).")
    parser.add_argument('--compile', action='store_true',
                        help="Compile model for faster inference (requires PyTorch 2).")
    parser.add_argument('--histogram', action='store_true',
                        help="Have a gradients histogram in tensorboard.")

    # General training parameters
    parser.add_argument('--num_epochs', default=200, type=int,
                        help="Number of training epochs.")
    parser.add_argument('--warmup', default=10, type=int,
                        help="Number of warmup epochs.")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="Training batch size.")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Training learning rate.")
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--optimizer', default="RiemannianAdamW", type=str,
                        choices=["RiemannianAdamW", "RiemannianSGD", "AdamW", "SGD"],
                        help="Optimizer for training.")

    # General validation/testing hyperparameters
    parser.add_argument('--batch_size_test', default=128, type=int,
                        help="Validation/Testing batch size.")

    # Transformer settings
    parser.add_argument('--patch_size', default=16, type=int,
                        help="Number of encoder layers in ViT.")
    # Add model size option
    parser.add_argument(
        "--model_size",
        choices=["tiny", "small", "base"],
        default="tiny",
        help="Choose model size: tiny, small, or base."
    )
    parser.add_argument('--num_layers', default=None, type=int,
                        help="Number of encoder layers in ViT.")
    parser.add_argument('--hidden_dim', default=None, type=int,
                        help="Dimensionality of hidden dimensionality")
    parser.add_argument('--mlp_dim', default=None, type=int,
                        help="Dimensionality of hidden dimensionality")
    parser.add_argument('--num_heads', default=None, type=int,
                        help="Number of heads in Transformer attention")
    parser.add_argument('--encoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz", "poincare"],
                        help="Select conv model encoder manifold.")
    parser.add_argument('--decoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz", "poincare"],
                        help="Select conv model decoder manifold.")

    # Hyperbolic geometry settings
    parser.add_argument('--learn_k', action='store_true',
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--encoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--decoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).")
    parser.add_argument('--clip_features', default=1.0, type=float,
                        help="Clipping parameter for hybrid HNNs proposed by Guo et al. (2022)")

    # Dataset settings
    parser.add_argument('--dataset', default='CIFAR-100', type=str,
                        choices=["CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "ImageNet", "tieredImageNet"],
                        help="Select a dataset.")
    parser.add_argument('--data_root', default='data', type=str,
                        help="Root directory for CIFAR data or the unfinished tieredImageNet ImageFolder layout. Relative paths are resolved from the current working directory.")
    parser.add_argument('--tiered_lmdb_root', default=None, type=str,
                        help="Optional root containing tieredImageNet train/val/test LMDB files. Relative paths are resolved from the current working directory.")

    # HAA ablation mode
    parser.add_argument('--haa_mode', default='baseline', type=str,
                        choices=['baseline', 'terminal', 'full_uniform', 'continuous', 'terminal_aggressive'],
                        help="HAA injection mode: baseline=no HAA, terminal=last layer only, full_uniform=all layers, continuous=all layers with depth-proportional beta init, terminal_aggressive=last layer with high tau/low lambda.")
    parser.add_argument('--haa_tau_init', default=1.0, type=float,
        help="Initial tau value for HAA entailment weight. Default 1.0 "
             "(P3: calibrated to spatial penalty dynamic range). "
             "Use 3.0 for terminal_aggressive.")
    parser.add_argument('--haa_lambda_init', default=1.0, type=float,
        help="Initial lambda value for HAA spatial weight. Default 1.0. Use 0.3 for terminal_aggressive.")
    parser.add_argument('--learn_lambda', default=True,
        action=argparse.BooleanOptionalAction,
        help="If set, lambda_raw is a trainable nn.Parameter (default). "
             "Pass --no-learn_lambda to freeze the spatial penalty parameter "
             "at its init value (requires_grad=False) for ablation studies.")
    parser.add_argument('--deep_diagnostics', action='store_true',
        help="Run an extra evaluation-loader pass at epochs "
             "{1,5,10,20,30,40,50,60,70,80,90,final} for geometric diagnostics. "
             "Disable for RNG-clean training runs.")
    parser.add_argument('--eval_only', action='store_true',
        help="Skip training; load checkpoint and run a single validation pass + HAA telemetry.")

    # STEP 2 / CHANGE-2: aperture-gradient regime
    parser.add_argument('--B_smooth', choices=['relu', 'softplus'], default='softplus',
        help="Smoothing for aperture B argument: 'softplus' restores β-gradient in "
             "the shallow regime (recommended); 'relu' is the legacy clamp.")
    parser.add_argument('--B_softplus_temp', type=float, default=4.0,
        help="Temperature for softplus B smoothing. softplus(temp·x)/temp ≈ relu(x) "
             "for |x|>1, smooth for |x|<1.")

    # STEP 4 / CHANGE-3: β init override
    parser.add_argument('--beta_init_override', type=float, default=None,
        help="Override β init for ALL HAA layers (replaces mode default and "
             "beta_proportional). None = use mode default.")

    # STEP 3 / CHANGE-4 + CHANGE-5: auxiliary loss weights
    parser.add_argument('--gamma_angular_max', type=float, default=0.0,
        help="Max plateau weight for DirectionalAngularLoss (0 disables).")
    parser.add_argument('--eta_max', type=float, default=0.0,
        help="Max plateau weight for HyperbolicHierarchyLoss (0 disables).")
    parser.add_argument('--gamma_angular_warmup', type=int, default=15,
        help="Epoch at which the angular-ordering ramp begins.")
    parser.add_argument('--eta_warmup', type=int, default=5,
        help="Epoch at which the HHL ramp begins.")
    # A3 deprecation shim — accept the old flag names for one release.
    parser.add_argument('--gamma_max', type=float, default=None,
        help="DEPRECATED: use --gamma_angular_max. Retained for one release.")
    parser.add_argument('--gamma_warmup', type=int, default=None,
        help="DEPRECATED: use --gamma_angular_warmup. Retained for one release.")

    # P5: L_proto / L_radvar / L_betacap weights and configuration.
    parser.add_argument('--eta_proto_max', type=float, default=0.0)
    parser.add_argument('--eta_proto_warmup', type=int, default=5)
    parser.add_argument('--zeta_radvar_max', type=float, default=0.0)
    parser.add_argument('--zeta_radvar_warmup', type=int, default=5)
    parser.add_argument('--sigma2_target', type=float, default=0.10)
    parser.add_argument('--xi_betacap_max', type=float, default=0.0)
    parser.add_argument('--xi_betacap_warmup', type=int, default=20)
    parser.add_argument('--betacap_percentile', type=float, default=0.25)
    parser.add_argument('--betacap_static_target', type=float, default=None)
    parser.add_argument('--proto_seed', type=int, default=42)
    parser.add_argument('--d_s', type=float, default=0.3)
    parser.add_argument('--d_f_mid', type=float, default=1.175,
        help="Deterministic per-class fine-prototype depth offset (d_f_mid). "
             "Final fine-prototype depth = d_s + d_f_mid. Was previously "
             "sampled uniform[d_f_low, d_f_high]; now fixed.")

    parser.add_argument('--phi_occ_max', type=float, default=0.0,
        help="Max plateau weight for ConeOccupancyLoss (β-detach guarded). 0 disables.")
    parser.add_argument('--phi_occ_warmup', type=int, default=15)
    parser.add_argument('--occ_s_target', type=float, default=0.10,
        help="Target soft cone-occupation fraction for ConeOccupancyLoss.")
    parser.add_argument('--occ_kappa', type=float, default=6.0)
    parser.add_argument('--occ_m_smooth', type=float, default=0.05)

    parser.add_argument('--omega_spread_max', type=float, default=0.0,
        help="Max plateau weight for SpreadLoss (post-W_Q radial+spatial anti-collapse). 0 disables.")
    parser.add_argument('--omega_spread_warmup', type=int, default=5)
    parser.add_argument('--spread_sigma2_target', type=float, default=0.10)
    parser.add_argument('--spread_cv_target', type=float, default=0.30)
    parser.add_argument('--spread_weight_rad', type=float, default=1.0)
    parser.add_argument('--spread_weight_spat', type=float, default=1.0)

    parser.add_argument('--chi_cls_dir_max', type=float, default=0.0,
        help="Max plateau weight for CLSRowDirectionLoss (Phase-3-E). 0 disables.")
    parser.add_argument('--chi_cls_dir_warmup', type=int, default=15,
        help="Epoch at which CLSRowDirectionLoss ramp begins.")
    parser.add_argument('--cls_dir_z_star', type=float, default=-0.05,
        help="Target ceiling for CLS-row z_mean in CLSRowDirectionLoss. "
             "One-sided hinge: zero gradient when z_mean_cls <= z_star.")

    parser.add_argument('--psi_cls_var_max', type=float, default=0.0,
        help="Max plateau weight for CLSRowVarianceLoss (Phase-3-E). 0 disables.")
    parser.add_argument('--psi_cls_var_warmup', type=int, default=10,
        help="Epoch at which CLSRowVarianceLoss ramp begins.")
    parser.add_argument('--cls_var_sigma2_star', type=float, default=0.05,
        help="Target floor for CLS-row K-variance of Z. One-sided hinge: "
             "zero gradient when sigma2_cls >= sigma2_star. Initial value "
             "0.05 is a placeholder; recalibrate from probe measurement.")

    parser.add_argument('--use_cls_depth_residual', action='store_true',
        help="Stage 2.1 Path B: enable per-CLS radial scaling residual "
             "alpha = 0.1 + 1.4*sigmoid(MLP(cls_spatial)) before the classifier. "
             "Decouples CLS depth from spatial norm so L_proto / L_radvar "
             "operate on an actual depth degree of freedom.")
    parser.add_argument('--use_q_depth_mlp', action='store_true',
        help="Step 11: enable in-attention per-head q_depth_mlp that scales CLS-Q "
             "spatial norm before the HAA score is computed.")
    parser.add_argument('--disable_B', action='store_true',
        help="Force the HAA aperture B to the neutral constant 1.0, removing "
             "the cone-aperture mechanism from the score formula. beta_raw "
             "remains a Parameter for checkpoint compatibility but receives "
             "zero gradient. Use to ablate the B/cone machinery cleanly.")

    parser.add_argument('--use_hec_loss', action='store_true',
                        help='Enable L_HEC entailment cone loss at the HAA layer.')
    parser.add_argument('--hec_weight', type=float, default=1.0,
                        help='Weight multiplier on L_HEC. Default 1.0.')
    parser.add_argument('--hec_margin', type=float, default=0.1,
                        help='Hinge margin gamma for negative pairs in L_HEC.')
    parser.add_argument('--hec_negative_mode', type=str, default='naive',
                        choices=['naive', 'supcon'],
                        help='Negative selection: naive=all cross-image; supcon=mask same-class.')
    parser.add_argument('--hec_warmup', type=int, default=5,
                        help='Linear warmup epochs for L_HEC weight.')
    parser.add_argument('--hec_head_reduce', type=str, default='mean',
                        choices=['mean', 'first'],
                        help='Reduce multi-head into single Q/K: mean or first head.')

    parser.add_argument('--use_proto_softmax', action='store_true',
        help="Replace LorentzMLR with LorentzPrototypeClassifier (Design 2).")
    parser.add_argument('--proto_T_init', type=float, default=1.0,
        help="Initial temperature for prototype-softmax classifier (positive).")

    args = parser.parse_args()

    if args.gamma_max is not None:
        import warnings as _warnings
        _warnings.warn(
            "--gamma_max is deprecated; use --gamma_angular_max instead.",
            DeprecationWarning, stacklevel=2)
        args.gamma_angular_max = args.gamma_max
    if args.gamma_warmup is not None:
        import warnings as _warnings
        _warnings.warn(
            "--gamma_warmup is deprecated; use --gamma_angular_warmup instead.",
            DeprecationWarning, stacklevel=2)
        args.gamma_angular_warmup = args.gamma_warmup

    if args.eval_only and args.load_checkpoint is None:
        parser.error("--eval_only requires --load_checkpoint PATH; evaluation without trained weights is not supported.")

    for requested_device in args.device:
        if not requested_device.startswith('cuda:'):
            parser.error(
                f"Unsupported device '{requested_device}'. This training entry point requires CUDA devices such as cuda:0.")
        try:
            device_idx = int(requested_device.split(':', 1)[1])
        except (TypeError, ValueError):
            parser.error(
                f"Invalid CUDA device '{requested_device}'. Use cuda:N or a comma-separated list such as cuda:0,cuda:1.")
        if device_idx < 0:
            parser.error(f"Invalid CUDA device index in '{requested_device}'; CUDA indices must be non-negative.")

    return args

def main(args):
    # Define presets
    model_configs = {
        "tiny":  dict(num_layers=9,  hidden_dim=192, mlp_dim=384,  num_heads=12),
        "small": dict(num_layers=12, hidden_dim=384, mlp_dim=768,  num_heads=6),
        "base":  dict(num_layers=12, hidden_dim=768, mlp_dim=3072, num_heads=12),
    }

    # Apply defaults if not explicitly set
    for key, value in model_configs[args.model_size].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Map --haa_mode to the list of layer indices where HAA scoring is active
    _haa_mode_map = {
        'baseline':             [],
        'terminal':             [args.num_layers - 1],
        'full_uniform':         list(range(args.num_layers)),
        'continuous':           list(range(args.num_layers)),
        'terminal_aggressive':  [args.num_layers - 1],
    }
    args.active_haa_layers = _haa_mode_map[args.haa_mode]
    args.beta_proportional = (args.haa_mode == 'continuous')

    if args.haa_mode == 'terminal_aggressive':
        if args.haa_tau_init == 1.0:
            args.haa_tau_init = 3.0
        if args.haa_lambda_init == 1.0:
            args.haa_lambda_init = 0.3

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the supported training path, but no CUDA device is available.")

    device_ids = [int(d.split(':', 1)[1]) for d in args.device]
    available_device_count = torch.cuda.device_count()
    unavailable_device_ids = [idx for idx in device_ids
                              if idx >= available_device_count]
    if unavailable_device_ids:
        raise RuntimeError(
            f"Requested CUDA device indices {unavailable_device_ids}, but only "
            f"{available_device_count} CUDA device(s) are visible.")

    device = args.device[0]
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    print("Running experiment: " + args.exp_name)

    print("Arguments:")
    print(args)

    print("Loading dataset...")
    train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(args)

    print("Creating model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, len(train_loader), args)
    criterion = LabelSmoothingCrossEntropy()

    start_epoch = 0
    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(model, optimizer, lr_scheduler, args)

    model = DataParallel(model, device_ids=device_ids)

    if args.compile:
        model = torch.compile(model)

    # Build a self-identifying experiment name that embeds the HAA mode
    if args.haa_mode != 'baseline':
        _suffix = args.haa_mode
        if args.haa_mode == 'terminal_aggressive':
            _suffix = (f"terminal_aggressive"
                       f"_tau{args.haa_tau_init}"
                       f"_lam{args.haa_lambda_init}")
        _run_exp_name = args.exp_name + "_haa_" + _suffix
    else:
        _run_exp_name = args.exp_name

    # PN: append active-aux-loss tokens (proto, radvar, betacap, angular, hhl)
    # so concurrent ablations never collide on a single checkpoint / CSV /
    # TensorBoard logdir. Resolution runs exactly once, before any path is built.
    from run_naming import resolve_run_name
    _run_exp_name = resolve_run_name(_run_exp_name, args)
    args.run_name = _run_exp_name

    # Initialize TensorBoard writer
    # Set up TensorBoard logging directory with a unique name for each experiment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, _run_exp_name + "_" + timestamp)
    writer = SummaryWriter(log_dir=log_dir)


    # Initialize lists to store the metrics
    train_losses, val_losses = [], []
    train_acc1s, val_acc1s = [], []
    train_acc5s, val_acc5s = [], []

    # ------------------------------------------------------------------
    # Eval-only shortcut: load checkpoint, run one val pass, log HAA
    # telemetry, and exit — no training loop entered at all.
    # Used for STEP 1 checkpoint verification (z_mean measurement after
    # CHANGE-1 numerics fix).
    # ------------------------------------------------------------------
    if args.eval_only:
        with torch.no_grad():
            loss_val, acc1_val, acc5_val = evaluate(model, val_loader, criterion, device)
        print(f"[eval_only] Val: Loss={loss_val:.4f}, "
              f"Acc@1={acc1_val:.4f}, Acc@5={acc5_val:.4f}")
        log_haa_epoch_metrics(model, epoch=start_epoch, writer=writer)
        # S-3: also log prototype-softmax temperature in eval_only.
        _decoder = (model.module.decoder if hasattr(model, 'module')
                    else model.decoder)
        from classification_vit.lorentz_proto_classifier import LorentzPrototypeClassifier
        if isinstance(_decoder, LorentzPrototypeClassifier):
            _T = _decoder.temperature.detach().item()
            writer.add_scalar('proto_softmax/temperature', _T, start_epoch)
            print(f"[eval_only] proto_softmax temperature = {_T:.6f}")
        if args.deep_diagnostics:
            _base_m = model.module if hasattr(model, 'module') else model
            _K = _base_m.enc_manifold.k.item()
            _cuda_rng = torch.cuda.get_rng_state_all()
            _cpu_rng  = torch.get_rng_state()
            _ = log_haa_deep_diagnostics(model, val_loader, device, start_epoch, _K, writer)
            torch.cuda.set_rng_state_all(_cuda_rng)
            torch.set_rng_state(_cpu_rng)
        writer.close()
        return

    print("Training...")
    global_step = start_epoch * len(train_loader)

    # STEP 3 / CHANGE-4: build auxiliary loss modules (Loss Factory pattern).
    # Both losses are no-ops when their weight is 0 (default), so this call is
    # safe for baseline/legacy runs.
    # P5: build fixed Lorentz prototypes when L_proto is active. Must run
    # before build_aux_losses so the factory can pick up args.hyperbolic_prototypes.
    attach_hyperbolic_prototypes(model, args)
    aux_losses = build_aux_losses(args)
    for _name, _loss_fn in aux_losses.items():
        if _loss_fn is not None:
            _loss_fn.to(device)
            print(f"[AUX LOSS] Enabled '{_name}' "
                  f"(plateau={_loss_fn.schedule.plateau}, "
                  f"warmup={_loss_fn.schedule.warmup}, "
                  f"ramp={_loss_fn.schedule.ramp})", flush=True)

    best_acc = 0.0
    best_epoch = 0

    from lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention
    _base = model.module if hasattr(model, 'module') else model
    _haa_layers = sorted(
        [(m.layer_idx, m)
         for _, m in _base.named_modules()
         if isinstance(m, LorentzMultiHeadAttention) and m.use_haa],
        key=lambda x: x[0]
    )

    # L_HEC entailment cone loss — bespoke aux path (not part of build_aux_losses).
    # Reads post-W_Q tensors captured by the terminal HAA MHA on every forward.
    hec_loss_fn = None
    haa_mha = None
    if getattr(args, 'use_hec_loss', False):
        if args.haa_mode == 'baseline' or not getattr(args, 'active_haa_layers', []):
            raise ValueError("--use_hec_loss requires HAA active (--haa_mode terminal or similar).")
        if not _haa_layers:
            raise ValueError("--use_hec_loss set but no HAA-active LorentzMultiHeadAttention found.")
        import torch.nn.functional as F
        from hec_loss import HECLoss
        # Terminal (last) HAA layer's MHA module — same discovery scheme as telemetry.
        haa_mha = _haa_layers[-1][1]

        def _beta_provider():
            return F.softplus(haa_mha.beta_raw) if hasattr(haa_mha, 'beta_raw') \
                else torch.tensor(1.0, device=next(haa_mha.parameters()).device)

        hec_loss_fn = HECLoss(
            beta_provider=_beta_provider,
            curvature_K=float(getattr(args, 'encoder_k', 1.0)),
            margin=float(args.hec_margin),
            negative_mode=args.hec_negative_mode,
            head_reduce=args.hec_head_reduce,
            warmup_epochs=int(args.hec_warmup),
        ).to(device)
        print(f"[L_HEC] enabled: negative_mode={args.hec_negative_mode}, "
              f"margin={args.hec_margin}, warmup={args.hec_warmup}, "
              f"head_reduce={args.hec_head_reduce}, layer={_haa_layers[-1][0]}", flush=True)
    # Collect per-epoch HAA scalars — use last HAA layer only for CSV
    # (full per-layer data remains in TensorBoard)
    _haa_epoch_data = {
        'beta':             [],
        'tau':              [],
        'lambda':           [],
        'cone_sparsity':    [],
        'z_mean':           [],
        'frac_near_origin': [],
        'grad_norm_B':      [],
        'grad_norm_c_tilde':[],
        'alpha_mean':       [],
        'alpha_grad_norm':  [],
    }
    _haa_epoch_data.update({
        'beta_first':             [],
        'tau_first':              [],
        'lambda_first':           [],
        'cone_sparsity_first':    [],
        'z_mean_first':           [],
        'frac_near_origin_first': [],
    })
    _haa_epoch_data['proto_T'] = []

    # Build dynamic σ² and deep-diagnostic CSV column names per HAA layer.
    sigma2_col_names = []
    deep_col_names   = []
    if any(True for _ in _haa_mhas_iter(model)):
        for _mha in _haa_mhas_iter(model):
            l = _mha.layer_idx
            sigma2_col_names += [
                f"train_sigma2_pre_attn_layer{l}",
                f"train_sigma2_post_attn_layer{l}",
                f"val_sigma2_pre_attn_layer{l}",
                f"val_sigma2_post_attn_layer{l}",
            ]
            if args.deep_diagnostics:
                for stat in ('sigma2_c_tilde', 'z_mean_diag', 'kl_z',
                             'spatial_cv', 'attention_entropy'):
                    deep_col_names.append(f"deep_layer{l}_{stat}")
    # Per-epoch storage for σ² readings and deep diagnostics.
    _sigma2_epoch_data = []        # list[dict] indexed by epoch offset
    _deep_metrics_epoch_data = []  # list[dict] indexed by epoch offset

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        losses = AverageMeter("Loss", ":.4e")
        acc1 = AverageMeter("Acc@1", ":6.2f")
        acc5 = AverageMeter("Acc@5", ":6.2f")

        # Reset per-epoch σ² accumulators on every HAA MHA before the train pass.
        for _mha in _haa_mhas_iter(model):
            _mha.reset_sigma2_train()

        for i, (x, y) in tqdm(enumerate(train_loader)):
            # ------- Start iteration -------
            x = x.to(device)
            y = y.to(device)
            y_original = y.clone()  # preserve for HHL aux loss

            # Track mix state so aux losses can use the correct labels.
            mixing_occurred = False
            y_a, y_b, lam = y_original, y_original, 1.0

            r = np.random.rand(1)
            mix_prob = 0.5
            if r < mix_prob:
                switching_prob = np.random.rand(1)
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(x, y)
                    x[:, :, slicing_idx[0]:slicing_idx[2],
                          slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(x)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                    mixing_occurred = True
                else:
                    x, y_a, y_b, lam = mixup_data(x, y)
                    output = model(x)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                    mixing_occurred = True
            else:
                output = model(x)
                loss = criterion(output, y)

            # Mixup-aware aux-loss accumulation.
            # Label-conditioned losses (proto, hhl) receive the lam-weighted sum
            # over the two mixed labels. Label-independent losses (angular, radvar,
            # betacap) ignore y entirely and are called once with y_original.
            LABEL_CONDITIONED = {'proto', 'hhl'}
            loss_total = loss
            for _aux_name, _aux_fn in aux_losses.items():
                if _aux_fn is None:
                    continue
                _w = _aux_fn.schedule(epoch)
                if _w <= 0:
                    continue
                if mixing_occurred and _aux_name in LABEL_CONDITIONED:
                    _lam_t = float(lam) if not torch.is_tensor(lam) else lam.item()
                    _loss_aux = (_lam_t * _aux_fn(model, x, y_a, device)
                                 + (1.0 - _lam_t) * _aux_fn(model, x, y_b, device))
                else:
                    _loss_aux = _aux_fn(model, x, y_original, device)
                loss_total = loss_total + _w * _loss_aux

            # L_HEC entailment cone loss (bespoke aux path). Reads post-W_Q tensors
            # captured during the model(x) forward above. y_original are the
            # un-mixed per-image class labels (used by the supcon negative mask).
            if hec_loss_fn is not None:
                loss_hec = hec_loss_fn(haa_mha, labels=y_original, epoch=epoch)
                loss_total = loss_total + args.hec_weight * loss_hec

            optimizer.zero_grad()
            loss_total.backward()
            
            if args.histogram:
                if (i > 0 and i % 300 == 0) or i in [1, 5, 10, 25, 50, 75, 100, 150, 200]:
                    # Collect combined gradients
                    combined_gradients = []
                    for name, param in model.module.named_parameters():
                        if param.grad is not None:
                            combined_gradients.extend(param.grad.abs().add_(1e-9).log().view(-1).tolist())
                    
                    # Convert to tensor
                    combined_gradients_tensor = torch.tensor(combined_gradients, device=device)
                    
                    # Log close-up histogram for specific iterations in the list
                    if epoch == start_epoch:
                        if i in [1, 5, 10, 25, 50, 75, 100, 150, 200]:
                            writer.add_histogram('grad/combined_closeup', combined_gradients_tensor, global_step)
                    
                    # Log complete histogram for all iterations matching the condition
                    writer.add_histogram('grad/combined_complete', combined_gradients_tensor, global_step)


            optimizer.step()
            lr_scheduler.step()

            with torch.no_grad():
                top1, top5 = accuracy(output, y, topk=(1, 5))
                losses.update(loss.item())
                acc1.update(top1.item())
                acc5.update(top5.item())

            global_step += 1
            
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            # ------- End iteration -------
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', losses.avg, epoch)
        writer.add_scalar('Accuracy/train_top1', acc1.avg, epoch)
        writer.add_scalar('Accuracy/train_top5', acc5.avg, epoch)

        # Append the training metrics for this epoch
        train_losses.append(losses.avg)
        train_acc1s.append(acc1.avg)
        train_acc5s.append(acc5.avg)

        # ------- Start validation and logging -------
        # Reset per-epoch σ² accumulators on every HAA MHA before the val pass.
        for _mha in _haa_mhas_iter(model):
            _mha.reset_sigma2_val()
        with torch.no_grad():
            loss_val, acc1_val, acc5_val = evaluate(model, val_loader, criterion, device)

            print(
                "Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}, Validation: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
                    epoch + 1, args.num_epochs, losses.avg, acc1.avg, acc5.avg, loss_val, acc1_val, acc5_val))

            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val', loss_val, epoch)
            writer.add_scalar('Accuracy/val_top1', acc1_val, epoch)
            writer.add_scalar('Accuracy/val_top5', acc5_val, epoch)

            # Log per-layer HAA telemetry (reads values stored during evaluate())
            log_haa_epoch_metrics(model, epoch, writer)

            # Phase 2: prototype-softmax temperature telemetry.
            _decoder = (model.module.decoder if hasattr(model, 'module')
                        else model.decoder)
            from classification_vit.lorentz_proto_classifier import LorentzPrototypeClassifier
            if isinstance(_decoder, LorentzPrototypeClassifier):
                _T = _decoder.temperature.detach().item()
                writer.add_scalar('proto_softmax/temperature', _T, epoch)
                _haa_epoch_data['proto_T'].append(_T)
            else:
                _haa_epoch_data['proto_T'].append('')

            # Read per-layer σ² accumulators (per-image variance of c̃ across
            # the n=65 tokens of one image; mean over images in the pass).
            sigma2_readings = {}
            for _mha in _haa_mhas_iter(model):
                _train_pre, _train_post = _mha.get_sigma2_train()
                _val_pre,   _val_post   = _mha.get_sigma2_val()
                sigma2_readings[f"train_sigma2_pre_attn_layer{_mha.layer_idx}"]  = _train_pre
                sigma2_readings[f"train_sigma2_post_attn_layer{_mha.layer_idx}"] = _train_post
                sigma2_readings[f"val_sigma2_pre_attn_layer{_mha.layer_idx}"]    = _val_pre
                sigma2_readings[f"val_sigma2_post_attn_layer{_mha.layer_idx}"]   = _val_post

            for _key, _val in sigma2_readings.items():
                if writer is not None:
                    writer.add_scalar(f"sigma2/{_key}", _val, epoch)
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({f"sigma2/{_key}": _val}, step=epoch)
                except ImportError:
                    pass
            _sigma2_epoch_data.append(sigma2_readings)

            # Collect HAA scalars from the last HAA layer for CSV
            if _haa_layers:
                _, _last_mha = _haa_layers[-1]
                _haa_epoch_data['beta'].append(_last_mha.haa_alpha)
                _haa_epoch_data['tau'].append(_last_mha.haa_tau)
                _haa_epoch_data['lambda'].append(_last_mha.haa_lambda)
                _haa_epoch_data['cone_sparsity'].append(_last_mha.haa_cone_sparsity)
                _haa_epoch_data['z_mean'].append(_last_mha.haa_mean_Z)
                _haa_epoch_data['frac_near_origin'].append(_last_mha.haa_frac_near_origin)
                _haa_epoch_data['grad_norm_B'].append(_last_mha._grad_norms.get('B', ''))
                _haa_epoch_data['grad_norm_c_tilde'].append(_last_mha._grad_norms.get('c_tilde', ''))

            if _haa_layers:
                _, _first_mha = _haa_layers[0]
                _haa_epoch_data['beta_first'].append(_first_mha.haa_alpha)
                _haa_epoch_data['tau_first'].append(_first_mha.haa_tau)
                _haa_epoch_data['lambda_first'].append(_first_mha.haa_lambda)
                _haa_epoch_data['cone_sparsity_first'].append(_first_mha.haa_cone_sparsity)
                _haa_epoch_data['z_mean_first'].append(_first_mha.haa_mean_Z)
                _haa_epoch_data['frac_near_origin_first'].append(_first_mha.haa_frac_near_origin)

            # Stage 2.1 Path B: per-CLS radial scaling residual telemetry.
            _cls_resid = getattr(_base, 'cls_depth_residual', None)
            if _cls_resid is None:
                _cls_resid = getattr(getattr(_base, 'encoder', None),
                                     'cls_depth_residual', None)
            if _cls_resid is not None and _cls_resid._last_alpha is not None:
                _haa_epoch_data['alpha_mean'].append(
                    _cls_resid._last_alpha.float().mean().item())
                _haa_epoch_data['alpha_grad_norm'].append(
                    float(getattr(_cls_resid, '_last_alpha_grad', 0.0)))
            else:
                _haa_epoch_data['alpha_mean'].append('')
                _haa_epoch_data['alpha_grad_norm'].append('')

            # Append the validation metrics for this epoch
            val_losses.append(loss_val)
            val_acc1s.append(acc1_val)
            val_acc5s.append(acc5_val)

            # Testing for best model
            if acc1_val > best_acc:
                best_acc = acc1_val
                best_epoch = epoch + 1
                if args.output_dir is not None:
                    save_path = os.path.join(args.output_dir, f"best_{_run_exp_name}.pth")
                    torch.save({
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'epoch': epoch,
                        'args': args,
                    }, save_path)

        # Deep diagnostics at epochs in DEEP_EPOCHS {1,5,10,20,30,40,50,60,70,80,90}
        # plus the final epoch.
        if args.deep_diagnostics and ((epoch+1) in DEEP_EPOCHS
                                       or (epoch+1) == args.num_epochs):
            _K = model.module.enc_manifold.k.item()
            _cuda_rng = torch.cuda.get_rng_state_all()
            _cpu_rng  = torch.get_rng_state()
            deep_metrics_this_epoch = log_haa_deep_diagnostics(model, val_loader, device, epoch, _K, writer) or {}
            torch.cuda.set_rng_state_all(_cuda_rng)
            torch.set_rng_state(_cpu_rng)
        else:
            deep_metrics_this_epoch = {}
        _deep_metrics_epoch_data.append(deep_metrics_this_epoch)
        # ------- End validation and logging -------

    print("-----------------\nTraining finished\n-----------------")
    print("Best epoch = {}, with Acc@1={:.4f}".format(best_epoch, best_acc))

    if args.output_dir is not None:
        save_path = os.path.join(args.output_dir, f"final_{_run_exp_name}.pth")
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epoch': epoch,
            'args': args,
        }, save_path)
        print("Model saved to " + save_path)

        # Save metrics to CSV
        metrics_file = os.path.join(args.output_dir, f"{_run_exp_name}_metrics.csv")
        with open(metrics_file, mode='w', newline='') as file:
            writer_csv = csv.writer(file)
            base_header = [
                'Epoch', 'Train Loss', 'Val Loss', 'Train Acc@1', 'Val Acc@1',
                'Train Acc@5', 'Val Acc@5', 'haa_beta', 'haa_tau', 'haa_lambda',
                'haa_cone_sparsity', 'haa_z_mean', 'haa_frac_near_origin',
                'haa_grad_norm_B', 'haa_grad_norm_c_tilde',
                'cls_alpha_mean', 'cls_alpha_grad_norm',
            ]
            base_header += [
                'haa_beta_first', 'haa_tau_first', 'haa_lambda_first',
                'haa_cone_sparsity_first', 'haa_z_mean_first',
                'haa_frac_near_origin_first',
            ]
            base_header += ['proto_T']
            full_header = base_header + sigma2_col_names + deep_col_names
            writer_csv.writerow(full_header)
            for i in range(args.num_epochs):
                base_row = [
                    i+1,
                    train_losses[i], val_losses[i],
                    train_acc1s[i], val_acc1s[i],
                    train_acc5s[i], val_acc5s[i],
                    _haa_epoch_data['beta'][i]             if i < len(_haa_epoch_data['beta']) else '',
                    _haa_epoch_data['tau'][i]              if i < len(_haa_epoch_data['tau']) else '',
                    _haa_epoch_data['lambda'][i]           if i < len(_haa_epoch_data['lambda']) else '',
                    _haa_epoch_data['cone_sparsity'][i]    if i < len(_haa_epoch_data['cone_sparsity']) else '',
                    _haa_epoch_data['z_mean'][i]           if i < len(_haa_epoch_data['z_mean']) else '',
                    _haa_epoch_data['frac_near_origin'][i] if i < len(_haa_epoch_data['frac_near_origin']) else '',
                    _haa_epoch_data['grad_norm_B'][i]      if i < len(_haa_epoch_data['grad_norm_B']) else '',
                    _haa_epoch_data['grad_norm_c_tilde'][i] if i < len(_haa_epoch_data['grad_norm_c_tilde']) else '',
                    _haa_epoch_data['alpha_mean'][i]      if i < len(_haa_epoch_data['alpha_mean']) else '',
                    _haa_epoch_data['alpha_grad_norm'][i] if i < len(_haa_epoch_data['alpha_grad_norm']) else '',
                    _haa_epoch_data['beta_first'][i]             if i < len(_haa_epoch_data['beta_first']) else '',
                    _haa_epoch_data['tau_first'][i]              if i < len(_haa_epoch_data['tau_first']) else '',
                    _haa_epoch_data['lambda_first'][i]           if i < len(_haa_epoch_data['lambda_first']) else '',
                    _haa_epoch_data['cone_sparsity_first'][i]    if i < len(_haa_epoch_data['cone_sparsity_first']) else '',
                    _haa_epoch_data['z_mean_first'][i]           if i < len(_haa_epoch_data['z_mean_first']) else '',
                    _haa_epoch_data['frac_near_origin_first'][i] if i < len(_haa_epoch_data['frac_near_origin_first']) else '',
                    _haa_epoch_data['proto_T'][i] if i < len(_haa_epoch_data['proto_T']) else '',
                ]
                _sigma2_i = _sigma2_epoch_data[i] if i < len(_sigma2_epoch_data) else {}
                _deep_i   = _deep_metrics_epoch_data[i] if i < len(_deep_metrics_epoch_data) else {}
                sigma2_row = [_sigma2_i.get(name, '') for name in sigma2_col_names]
                deep_row   = [_deep_i.get(name, '')   for name in deep_col_names]
                writer_csv.writerow(base_row + sigma2_row + deep_row)
        print(f"Metrics saved to {metrics_file}")

        # Save best accuracy and epoch to a text file
        best_metrics_file = os.path.join(args.output_dir, f"{_run_exp_name}_best_metrics.txt")
        with open(best_metrics_file, mode='w') as file:
            file.write(f"Best Epoch: {best_epoch}\n")
            file.write(f"Best Accuracy@1: {best_acc:.4f}\n\n")
            file.write(f"Arguments: {args}\n")
        print(f"Best metrics saved to {best_metrics_file}")
    else:
        print("Model and metrics not saved.")

    print("Testing final model...")
    loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

    print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
        loss_test, acc1_test, acc5_test))

    print("Testing best model...")
    if args.output_dir is not None:
        print("Loading best model...")
        save_path = os.path.join(args.output_dir, f"best_{_run_exp_name}.pth")
        checkpoint = torch.load(save_path, map_location=device)
        model.module.load_state_dict(checkpoint['model'], strict=True)

        loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

        print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
            loss_test, acc1_test, acc5_test))
    else:
        print("Best model not saved, because no output_dir given.")

    writer.close()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """ Evaluates model performance """
    model.eval()
    model.to(device)

    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)

        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item())
        acc1.update(top1.item(), x.shape[0])
        acc5.update(top5.item(), x.shape[0])

    return losses.avg, acc1.avg, acc5.avg

if __name__ == '__main__':
    args = getArguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + args.dtype

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            print("Create missing output directory...")
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
