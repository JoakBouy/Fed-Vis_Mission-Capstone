"""Train the Attention U-Net on a single site's data.

This is the non-federated training script. Useful for
debugging the pipeline or getting a single-site baseline
before running the federated version.

Usage:
    python -m fedvis.scripts.train_local --data fets --site 1 --epochs 50
    python -m fedvis.scripts.train_local --data prostate --site BIDMC
    python -m fedvis.scripts.train_local --data lung --site 1
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedvis.models.attention_unet import AttentionUNet3D
from fedvis.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# each dataset has its own set of hospital sites
DATASETS = {
    'fets': {
        'sites': ['1', '6', '18', '21'],
        'default_path': 'data/FeTS2022_processed',
        'vol_shape': (64, 128, 128),
    },
    'prostate': {
        'sites': ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL'],
        'default_path': 'data/Prostate_processed',
        'vol_shape': (64, 128, 128),
    },
    'lung': {
        'sites': ['1', '2', '3', '4', '5'],
        'default_path': 'data/CTLung_processed',
        'vol_shape': (64, 128, 128),
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='Fed-Vis single-site training')
    p.add_argument('--data', type=str, default='fets',
                   choices=list(DATASETS.keys()))
    p.add_argument('--site', type=str, default='1')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--features', type=int, default=32,
                   help='base channel count in the encoder')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=str, default='outputs')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--data_root', type=str, default=None)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_loaders(args):
    """Create train/val DataLoaders for the chosen dataset + site."""
    from fedvis.data import FeTSDataset, ProstateDataset, LungDataset
    from omegaconf import OmegaConf

    ds_info = DATASETS[args.data]
    root = args.data_root or ds_info['default_path']
    d, h, w = ds_info['vol_shape']

    cfg = OmegaConf.create({
        'data': {
            'processed_path': root,
            'volume_size': {'depth': d, 'height': h, 'width': w},
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'sites': ds_info['sites'],
            'use_modality': 'T1',
            'binary_labels': True,
            'num_classes': 1,
        },
        'paths': {'data_root': root},
    })

    cls_map = {
        'fets': FeTSDataset,
        'prostate': ProstateDataset,
        'lung': LungDataset,
    }
    Cls = cls_map[args.data]

    train_ds = Cls(cfg, split='train', site=args.site)
    val_ds = Cls(cfg, split='val', site=args.site)

    logger.info(f"{args.data} site={args.site}  train={len(train_ds)} val={len(val_ds)}")

    def init_worker(wid):
        random.seed(args.seed + wid)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, worker_init_fn=init_worker,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader


def main():
    args = parse_args()
    seed_everything(args.seed)

    logger.info(f"Fed-Vis local training | {args.data} site {args.site}")

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
        logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Running on CPU")

    # model
    model = AttentionUNet3D(in_channels=1, out_channels=1, base_features=args.features)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    # data
    train_loader, val_loader = build_loaders(args)

    # output
    run_name = f"{args.data}_{args.site}_f{args.features}"
    out_dir = os.path.join(args.output, run_name)
    os.makedirs(out_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(out_dir, 'train.log'))
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(fh)

    # train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg={'lr': args.lr, 'epochs': args.epochs, 'dice_weight': 1.0, 'bce_weight': 1.0},
        output_dir=out_dir,
        device=device,
    )

    if args.resume:
        trainer.resume(args.resume)

    best = trainer.train(num_epochs=args.epochs)
    logger.info(f"Best dice: {best:.4f}")


if __name__ == '__main__':
    main()
