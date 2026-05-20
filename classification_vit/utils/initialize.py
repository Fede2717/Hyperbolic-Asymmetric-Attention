import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from lib.geoopt import ManifoldParameter
from lib.geoopt.optim import RiemannianAdamW, RiemannianSGD

from lib.utils.imagenet import ImageNet

from models.classifier import ViTClassifier

from lib.utils.scheduler import build_scheduler

from lib.utils.autoaug import CIFAR10Policy
from lib.utils.random_erasing import RandomErasing
from lib.utils.sampler import RASampler


def attach_hyperbolic_prototypes(model, args):
    """P5: build the fixed-prototype tensor for L_proto and stash it on
    ``args.hyperbolic_prototypes`` so ``build_aux_losses`` can pick it up.

    No-op when ``args.eta_proto_max`` is not strictly positive.
    """
    if float(getattr(args, 'eta_proto_max', 0.0)) <= 0:
        return
    from haa_auxiliary_loss import build_hyperbolic_prototypes
    from hierarchy_loader import load_hierarchy
    FINE_TO_SUPER, NUM_FINE, NUM_SUPER = load_hierarchy(
        str(getattr(args, 'dataset', 'CIFAR-100')))

    lut = torch.zeros(NUM_FINE, dtype=torch.long)
    for f, s in FINE_TO_SUPER.items():
        lut[f] = s

    protos = build_hyperbolic_prototypes(
        num_super=NUM_SUPER,
        num_fine=NUM_FINE,
        hidden_dim=args.hidden_dim + 1,
        fine_to_super_lut=lut,
        K=float(getattr(args, 'encoder_k', 1.0)),
        seed=int(getattr(args, 'proto_seed', 42)),
        d_s=float(getattr(args, 'd_s', 0.3)),
        d_f_mid=float(getattr(args, 'd_f_mid', 1.175)),
    )
    protos = protos.to(next(model.parameters()).device)
    args.hyperbolic_prototypes = protos


def load_checkpoint(model, optimizer, lr_scheduler, args):
    """ Loads a checkpoint from file-system. """

    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint:
        if checkpoint['args'].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr

            if (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(args.optimizer, checkpoint['args'].optimizer))

    epoch = 0
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1

    return model, optimizer, lr_scheduler, epoch

def load_model_checkpoint(model, checkpoint_path):
    """ Loads a checkpoint from file-system. """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model

def select_model(img_dim, num_classes, args):
    """ Selects and sets up an available model and returns it. """

    if getattr(args, 'use_q_depth_mlp', False) and getattr(args, 'use_cls_depth_residual', False):
        raise RuntimeError("--use_q_depth_mlp and --use_cls_depth_residual are mutually exclusive.")

    enc_args = {
        'num_layers' : args.num_layers,
        'img_dim' : img_dim,
        'num_classes' : num_classes,
        'patch_size' : args.patch_size,
        'heads' : args.num_heads,
        'hidden_dim' : args.hidden_dim,
        'mlp_dim' : args.mlp_dim,
        'active_haa_layers'  : getattr(args, 'active_haa_layers', []),
        'beta_proportional'  : getattr(args, 'beta_proportional', False),
    }
    enc_args['tau_init']    = getattr(args, 'haa_tau_init',    1.0)
    enc_args['lambda_init'] = getattr(args, 'haa_lambda_init', 1.0)
    enc_args['learn_lambda'] = getattr(args, 'learn_lambda', True)
    # STEP 2 / CHANGE-2: aperture-gradient regime (relu = legacy, softplus = fixed)
    enc_args['B_smooth']        = getattr(args, 'B_smooth',        'softplus')
    enc_args['B_softplus_temp'] = getattr(args, 'B_softplus_temp', 4.0)
    # STEP 4 / CHANGE-3: optional override of β init for all HAA layers
    enc_args['beta_init_override'] = getattr(args, 'beta_init_override', None)
    # Stage 2.1 Path B (revised): per-CLS radial scaling residual toggle.
    enc_args['use_cls_depth_residual'] = getattr(
        args, 'use_cls_depth_residual', False)
    enc_args['use_q_depth_mlp'] = getattr(args, 'use_q_depth_mlp', False)

    if (args.encoder_manifold=="lorentz") or (args.encoder_manifold=="poincare"):
        enc_args['learn_k'] = args.learn_k
        enc_args['k'] = args.encoder_k

    dec_args = {
        'embed_dim'       : args.hidden_dim,
        'num_classes'     : num_classes,
        'k'               : args.decoder_k,
        'learn_k'         : args.learn_k,
        'type'            : 'mlr',
        'clip_r'          : args.clip_features,
        'use_proto_softmax': bool(getattr(args, 'use_proto_softmax', False)),
        'proto_seed'      : int(getattr(args, 'proto_seed', 42)),
        'd_s'             : float(getattr(args, 'd_s', 0.3)),
        'd_f_mid'         : float(getattr(args, 'd_f_mid', 1.175)),
        'T_init'          : float(getattr(args, 'proto_T_init', 1.0)),
        'dataset_name'    : str(getattr(args, 'dataset', 'CIFAR-100')),
    }

    model = ViTClassifier(
        enc_type=args.encoder_manifold,
        dec_type=args.decoder_manifold,
        enc_kwargs=enc_args,
        dec_kwargs=dec_args
    )

    return model

def select_optimizer(model, len_train_loader, args):
    """ Selects and sets up an available optimizer and returns it. """

    model_parameters = get_param_groups(model, args.lr, args.weight_decay)

    if args.optimizer == "RiemannianAdamW":
        optimizer = RiemannianAdamW(model_parameters, lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, stabilize=1)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError(
            "Optimizer not found. Wrong optimizer in configuration -> "
            f"{args.optimizer}")
      
    lr_scheduler = build_scheduler(args, optimizer, len_train_loader)

    return optimizer, lr_scheduler

def get_param_groups(model, lr_manifold, weight_decay_manifold):
    # Historical exclusion list — matched parameters are removed from
    # every group below. `scale` is non-trainable by default
    # (requires_grad=False), so this is effectively a no-op for it,
    # but the exclusion is kept for backward compatibility.
    no_include = ["scale"]
    # G-1: parameters that should be optimized but with no weight decay.
    # The prototype-softmax temperature `log_T` (LorentzPrototypeClassifier)
    # is a 1-D scalar driving softmax sharpness; weight decay would
    # bias it toward softplus(0) ≈ 0.69 and confound the learned
    # temperature curve.
    no_wd_optimized = ["log_T"]
    k_params = [".k"]

    parameters = [
        {
            # Default group: trainable, decay-eligible.
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_include)
                and not any(nw in n for nw in no_wd_optimized)
                and not isinstance(p, ManifoldParameter)
                and not any(nd in n for nd in k_params)
            ],
            "name": "1"
        },
        {
            # G-1: trainable, weight_decay = 0.
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(nw in n for nw in no_wd_optimized)
                and not isinstance(p, ManifoldParameter)
                and not any(nd in n for nd in k_params)
            ],
            "weight_decay": 0.0,
            "name": "no_wd"
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and isinstance(p, ManifoldParameter)
            ],
            'lr': lr_manifold,
            "weight_decay": weight_decay_manifold,
            "name": "manifold"
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in k_params)
            ],
            "weight_decay": weight_decay_manifold,
            "lr": 1e-1,
            "name": "k_group"
        }
    ]

    return parameters

def select_dataset(args, validation_split=False):
    """ Selects an available dataset and returns PyTorch dataloaders for training, validation and testing. """

    # --- Per-dataset DataLoader performance overrides (tieredImageNet only) ---
    # All other datasets fall into the `else` branches below and use the
    # original hardcoded values byte-for-byte. _is_tiered gates BOTH the LMDB
    # dataset selection (later in this function) and the DataLoader knobs
    # (at the end of this function).
    _TIERED_LOADER_OVERRIDES = {
        'num_workers':        16,
        'persistent_workers': True,
        'prefetch_factor':    4,
        'ra_len_factor':      1,
    }
    _is_tiered = (args.dataset == 'tieredImageNet')

    if args.dataset == 'CIFAR-10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=mean)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_set = datasets.CIFAR10('/media/pinas/datasets/', train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR10('/media/pinas/datasets/', train=False, download=False, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-100':
        mean = (0.5070, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=mean)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_set = datasets.CIFAR100('/media/pinas/datasets/', train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR100('/media/pinas/datasets/', train=False, download=False, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 100

    elif args.dataset == 'Tiny-ImageNet':
        root_dir = "/add/path/here/" 
        train_dir = root_dir + "train"
        val_dir = root_dir + "val"
        test_dir = root_dir + "val" # No labels for test were given, so treat validation as test

        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)

        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=mean)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, test_transform)
        test_set = datasets.ImageFolder(test_dir, test_transform)

        img_dim = [3, 64, 64]
        num_classes = 200

    elif args.dataset == 'tieredImageNet':
        root_dir = "/media/hdd/usr/forner/tieredImageNet/"
        train_dir = root_dir + "train"
        val_dir   = root_dir + "val"
        test_dir  = root_dir + "test" if os.path.isdir(root_dir + "test") else val_dir

        # tieredImageNet native resolution is 84x84.
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(84, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=mean),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # If LMDB files exist locally, use the LMDB-backed dataset to skip
        # network I/O. Otherwise fall back to ImageFolder (current behavior).
        _lmdb_root = "/media/hdd/usr/forner/tieredImageNet_lmdb/"
        _train_lmdb = os.path.join(_lmdb_root, "train.lmdb")
        _val_lmdb   = os.path.join(_lmdb_root, "val.lmdb")
        _test_lmdb  = os.path.join(_lmdb_root, "test.lmdb")
        _use_lmdb = (os.path.isdir(_train_lmdb)
                     and os.path.isfile(_train_lmdb + ".meta.json")
                     and os.path.isdir(_val_lmdb)
                     and os.path.isfile(_val_lmdb + ".meta.json"))

        if _use_lmdb:
            try:
                from classification_vit.lmdb_dataset import LMDBImageFolder
            except ImportError:
                from lmdb_dataset import LMDBImageFolder
            train_set = LMDBImageFolder(_train_lmdb, transform=train_transform)
            val_set   = LMDBImageFolder(_val_lmdb,   transform=test_transform)
            _has_test_lmdb = (os.path.isdir(_test_lmdb)
                              and os.path.isfile(_test_lmdb + ".meta.json"))
            test_set = (LMDBImageFolder(_test_lmdb, transform=test_transform)
                        if _has_test_lmdb else val_set)
            print(f"[DATASET] tieredImageNet via LMDB at {_lmdb_root}", flush=True)
        else:
            train_set = datasets.ImageFolder(train_dir, train_transform)
            val_set   = datasets.ImageFolder(val_dir,   test_transform)
            test_set  = datasets.ImageFolder(test_dir,  test_transform)
            print(f"[DATASET] tieredImageNet via ImageFolder at {root_dir} "
                  f"(LMDB not found at {_lmdb_root})", flush=True)

        img_dim = [3, 84, 84]
        num_classes = len(train_set.classes)

    elif args.dataset == 'ImageNet':
        root_dir = "classification/data/imagenet/"

        train_transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_set = ImageNet(root_dir, split='train', transform=train_transform)
        val_set = ImageNet(root_dir, split='val', transform=test_transform)
        test_set = ImageNet(root_dir, split='val', transform=test_transform)

        img_dim = [3, 224, 224]
        num_classes = 1000

    else:
        raise RuntimeError(
            f"Selected dataset '{args.dataset}' not available.")
    
    # Dataloader
    if _is_tiered:
        train_loader = DataLoader(train_set,
            num_workers=_TIERED_LOADER_OVERRIDES['num_workers'],
            pin_memory=True,
            persistent_workers=_TIERED_LOADER_OVERRIDES['persistent_workers'],
            prefetch_factor=_TIERED_LOADER_OVERRIDES['prefetch_factor'],
            batch_sampler=RASampler(len(train_set),
                batch_size=args.batch_size,
                repetitions=1,
                len_factor=_TIERED_LOADER_OVERRIDES['ra_len_factor'],
                shuffle=True,
                drop_last=True
            )
        )
    else:
        train_loader = DataLoader(train_set,
            num_workers=4, 
            pin_memory=True, 
            batch_sampler=RASampler(len(train_set), 
                batch_size=args.batch_size, 
                repetitions=1,
                len_factor=3,
                shuffle=True, 
                drop_last=True
            )
        )
    if _is_tiered:
        test_loader = DataLoader(test_set,
            batch_size=args.batch_size_test,
            num_workers=_TIERED_LOADER_OVERRIDES['num_workers'],
            pin_memory=True,
            persistent_workers=_TIERED_LOADER_OVERRIDES['persistent_workers'],
            prefetch_factor=_TIERED_LOADER_OVERRIDES['prefetch_factor'],
            shuffle=False
        )
    else:
        test_loader = DataLoader(test_set, 
            batch_size=args.batch_size_test, 
            num_workers=4, 
            pin_memory=True, 
            shuffle=False
        ) 
    
    if validation_split:
        if _is_tiered:
            val_loader = DataLoader(val_set,
                batch_size=args.batch_size_test,
                num_workers=_TIERED_LOADER_OVERRIDES['num_workers'],
                pin_memory=True,
                persistent_workers=_TIERED_LOADER_OVERRIDES['persistent_workers'],
                prefetch_factor=_TIERED_LOADER_OVERRIDES['prefetch_factor'],
                shuffle=False
            )
        else:
            val_loader = DataLoader(val_set, 
                batch_size=args.batch_size_test, 
                num_workers=4, 
                pin_memory=True, 
                shuffle=False
            )
    else:
        val_loader = test_loader
        
    return train_loader, val_loader, test_loader, img_dim, num_classes
