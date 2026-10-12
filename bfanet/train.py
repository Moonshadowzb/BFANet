"""Train / evaluate BFANet on BCI-IV 2a (within-subject, session T -> E).

Every run appends one row to ``--csv`` with the columns
    experiment, model, dataset, subject, seed, budget, acc, kappa
and writes run metadata (parameter counts, wall time, GPU, full config) to
``<out_dir>/meta/<run_id>.json``.

The evaluation session is touched exactly once, after the last optimisation
step.  ``--curve-every`` additionally logs evaluation accuracy/loss for the
convergence figure; those values are never used for model selection.

Examples
--------
  python -m bfanet.train --subject 1 --model bfa --seed 0
  python -m bfanet.train --subject 1 --model vanilla --seed 0
  python -m bfanet.train --subject 1 --model bfa --decay-mode mul_logits --causal   # literal paper
  python -m bfanet.train --subject 1 --model bfa --csp                             # CSP ablation
  python -m bfanet.train --all-subjects --seeds 0 1 2 3 4 --model bfa
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

from . import data as D
from .model import BFAConfig, BFANet

CSV_FIELDS = ['experiment', 'model', 'dataset', 'subject', 'seed', 'budget', 'acc', 'kappa']


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def lr_at(step, total, warm, lr0=5e-5, lr1=5e-4, lr_min=1e-6):
    if step < warm:
        return lr0 + (lr1 - lr0) * step / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    return lr_min + 0.5 * (lr1 - lr_min) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, X, y, device, bs=32):
    model.eval()
    logits = torch.cat([model(X[i:i + bs].to(device)).float().cpu()
                        for i in range(0, len(X), bs)])
    loss = F.cross_entropy(logits, y).item()
    pred = logits.argmax(-1).numpy()
    acc = float((pred == y.numpy()).mean() * 100)
    kappa = float(cohen_kappa_score(y.numpy(), pred))
    return acc, kappa, loss


def train_one(cfg: BFAConfig, Xtr, ytr, Xte, yte, args, device,
              spatial_prior_factory=None, curve_log=None):
    model = BFANet(cfg, spatial_prior_factory).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999),
                            weight_decay=0.01)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xte_t, yte_t = torch.from_numpy(Xte), torch.from_numpy(yte)
    n = len(Xtr_t); bs = min(args.batch_size, n)
    total = args.steps; warm = int(round(total * args.warmup_frac))
    step = 0; t0 = time.time()
    while step < total:
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            if step >= total:
                break
            idx = perm[i:i + bs]
            x, y = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            model.train()
            if random.random() < args.noise_prob:
                x = x + args.noise_std * torch.randn_like(x)
            for g in opt.param_groups:
                g['lr'] = lr_at(step, total, warm)
            if random.random() < args.mixup_prob and len(idx) > 1:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                j = torch.randperm(len(idx), device=device)
                out = model(lam * x + (1 - lam) * x[j])
                loss = lam * F.cross_entropy(out, y, label_smoothing=0.1) + \
                    (1 - lam) * F.cross_entropy(out, y[j], label_smoothing=0.1)
            else:
                loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if curve_log is not None and args.curve_every and step % args.curve_every == 0:
                a, _, l = evaluate(model, Xte_t, yte_t, device, args.eval_bs)
                curve_log.append({'step': step, 'train_loss': loss.item(),
                                  'eval_acc': a, 'eval_loss': l})
    wall = time.time() - t0
    acc, kappa, _ = evaluate(model, Xte_t, yte_t, device, args.eval_bs)
    return model, acc, kappa, wall


def build_cfg(args, channels, T, n_classes) -> BFAConfig:
    prior = args.model
    return BFAConfig(
        channels=list(channels), n_classes=n_classes, T=T,
        d=args.d, depth=args.depth, heads=args.heads, dropout=args.dropout,
        patch=args.patch, prior=prior, decay_mode=args.decay_mode,
        causal=args.causal, use_geo=not args.no_geo,
        use_affinity=not args.no_affinity, learn_affinity=not args.hard_prior,
        affinity_param=args.affinity_param, shared_gamma=args.shared_gamma,
        spatial_prior=not args.csp, bfac_kind=args.bfac_kind,
        psi_drop=tuple(args.psi_drop), distill=not args.no_distill,
        distill_seed=args.distill_seed, sh_lmax=args.sh_lmax)


def append_csv(path, row):
    new = not Path(path).exists()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in CSV_FIELDS})


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor() or 'cpu'
    subjects = range(1, 10) if args.all_subjects else [args.subject]
    budgets = args.budgets or [None]
    for s in subjects:
        Xtr_all, ytr_all, Xte, yte = D.load_2a(s, args.cache_dir)
        for seed in args.seeds:
            subsets = (D.nested_budget_indices(ytr_all, [b for b in budgets if b], seed)
                       if budgets != [None] else {None: np.arange(len(ytr_all))})
            for b, idx in subsets.items():
                set_seed(seed)
                Xtr, ytr = Xtr_all[idx], ytr_all[idx]
                Xtr_s, Xte_s = D.standardise(Xtr, Xte, args.standardise)
                channels = D.CHANNELS_2A
                if args.csp:
                    W = D.fit_csp_ovr(Xtr_s, ytr, args.csp_filters)
                    Xtr_s, Xte_s = D.apply_spatial_filter(Xtr_s, W), D.apply_spatial_filter(Xte_s, W)
                    channels = [f'CSP{k}' for k in range(W.shape[1])]
                cfg = build_cfg(args, channels, Xtr_s.shape[-1], 4)
                curve = [] if args.curve_every else None
                model, acc, kappa, wall = train_one(cfg, Xtr_s.astype(np.float32), ytr,
                                                    Xte_s.astype(np.float32), yte,
                                                    args, device, curve_log=curve)
                budget = len(ytr) if b is None else b
                row = dict(experiment=args.experiment, model=args.tag or args.model,
                           dataset='BCI-IV-2a', subject=s, seed=seed, budget=budget,
                           acc=round(acc, 4), kappa=round(kappa, 5))
                append_csv(args.csv, row)
                run_id = f'{args.experiment}_{row["model"]}_S{s}_seed{seed}_n{budget}'
                meta = dict(row, params=sum(p.numel() for p in model.parameters()),
                            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            prior_params=sum(p.numel() for p in model.prior_parameters()),
                            bfac_distill=getattr(model, 'bfac_info', None),
                            wall_sec=round(wall, 1), device=gpu, tokens=model.tok.C * model.tok.Tp,
                            cfg=asdict(cfg) | {'channels': list(cfg.channels)},
                            args=vars(args), curve=curve)
                mdir = Path(args.out_dir) / 'meta'; mdir.mkdir(parents=True, exist_ok=True)
                (mdir / f'{run_id}.json').write_text(json.dumps(meta, indent=1, default=str))
                print(f'[{run_id}] acc={acc:.2f} kappa={kappa:.4f} wall={wall:.0f}s', flush=True)


def get_parser():
    p = argparse.ArgumentParser(description='BFANet on BCI-IV 2a')
    p.add_argument('--subject', type=int, default=1)
    p.add_argument('--all-subjects', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--budgets', type=int, nargs='*', default=None,
                   help='labelled trials per subject (class-balanced, nested)')
    p.add_argument('--model', choices=['bfa', 'vanilla', 'bfac'], default='bfa')
    p.add_argument('--tag', default=None, help='name written to the model column')
    p.add_argument('--experiment', default='main')
    p.add_argument('--csv', default='results/results.csv')
    p.add_argument('--out-dir', default='results')
    p.add_argument('--cache-dir', default='./cache_2a')
    # input
    p.add_argument('--csp', action=argparse.BooleanOptionalAction, default=False,
                   help='CSP front-end (ablation; disables the spatial prior). '
                        'Default --no-csp: raw 22 electrodes.')
    p.add_argument('--csp-filters', type=int, default=4)
    p.add_argument('--standardise', choices=['train_stats', 'trial'], default='train_stats')
    # architecture
    p.add_argument('--d', type=int, default=64)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--patch', type=int, default=25)
    # prior ablations
    p.add_argument('--decay-mode', choices=['logadd', 'mul_logits'], default='logadd')
    p.add_argument('--causal', action='store_true')
    p.add_argument('--no-geo', action='store_true')
    p.add_argument('--no-affinity', action='store_true')
    p.add_argument('--hard-prior', action='store_true')
    p.add_argument('--affinity-param', choices=['exp', 'sigmoid'], default='exp')
    p.add_argument('--shared-gamma', action='store_true')
    # BFA-C
    p.add_argument('--bfac-kind', choices=['mlp', 'sh'], default='mlp')
    p.add_argument('--psi-drop', nargs='*', default=[],
                   choices=['delta', 'delta_bar', 'xx', 'yy'])
    p.add_argument('--no-distill', action='store_true')
    p.add_argument('--distill-seed', type=int, default=0)
    p.add_argument('--sh-lmax', type=int, default=5)
    # optimisation (paper section 3.5)
    p.add_argument('--steps', type=int, default=5400,
                   help='fixed optimisation budget (600 epochs x 9 batches at n=288)')
    p.add_argument('--warmup-frac', type=float, default=20 / 600)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--mixup-prob', type=float, default=0.5)
    p.add_argument('--mixup-alpha', type=float, default=0.2)
    p.add_argument('--noise-prob', type=float, default=0.3)
    p.add_argument('--noise-std', type=float, default=0.01)
    p.add_argument('--curve-every', type=int, default=0)
    p.add_argument('--eval-bs', type=int, default=32)
    return p


if __name__ == '__main__':
    run(get_parser().parse_args())
