"""BCI Competition IV 2a: loading, preprocessing, optional CSP, label budgets.

Protocol (paper section 4.1-4.2, with the corrections of the design report)
--------------------------------------------------------------------------
* 22 EEG electrodes (EOG dropped), 250 Hz.
* 4-40 Hz 4th-order Butterworth, zero-phase, applied to each *continuous*
  run before epoching (avoids filter edge effects inside the trial).
* Window [0.5, 4] s after cue onset -> T = 875 samples.
* Standardisation: per-channel mean/std estimated on the *training* trials
  only and applied to both sessions ('train_stats', default).  The literal
  per-trial z-score ('trial') is available for ablation; it removes the
  per-trial channel variance, i.e. the band-power (ERD/ERS) information.
* Session '0train' (T) for training, '1test' (E) for evaluation.  No model
  selection is ever performed on the evaluation session.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from .montage import CHANNELS_2A

FS = 250.0
CUE_WINDOW = (0.5, 4.0)          # seconds after cue onset
BAND = (4.0, 40.0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _epoch_raw(raw, cue_offset: float, window=CUE_WINDOW, band=BAND):
    import mne
    raw = raw.copy().pick(CHANNELS_2A + ['stim'])
    raw.load_data()
    raw.filter(*band, picks=CHANNELS_2A, method='iir',
               iir_params=dict(order=4, ftype='butter'), phase='zero', verbose=False)
    ev = mne.find_events(raw, stim_channel='stim', verbose=False)
    ev = ev[np.isin(ev[:, 2], [1, 2, 3, 4])]
    X = raw.get_data(picks=CHANNELS_2A) * 1e6            # V -> uV
    a = int(round((cue_offset + window[0]) * FS))
    n = int(round((window[1] - window[0]) * FS))
    trials = np.stack([X[:, s + a: s + a + n] for s in ev[:, 0]])
    return trials.astype(np.float32), (ev[:, 2] - 1).astype(np.int64)


def load_2a(subject: int, cache_dir: str | os.PathLike = './cache_2a'):
    """Return (X_train, y_train, X_eval, y_eval) in uV, shapes (n, 22, 875).

    Uses MOABB's BNCI2014_001 (downloads the official files on first use).
    Labels: 0 left hand, 1 right hand, 2 feet, 3 tongue.
    Results are cached as npz.
    """
    cache = Path(cache_dir) / f'A{subject:02d}.npz'
    if cache.exists():
        z = np.load(cache)
        return z['Xtr'], z['ytr'], z['Xte'], z['yte']
    from moabb.datasets import BNCI2014_001
    ds = BNCI2014_001()
    cue_offset = float(ds.interval[0])       # event marks trial start; cue at +2 s
    sessions = ds.get_data(subjects=[subject])[subject]
    out = {}
    for sess, runs in sessions.items():
        Xs, ys = zip(*[_epoch_raw(r, cue_offset) for r in runs.values()])
        out[sess] = (np.concatenate(Xs), np.concatenate(ys))
    keys = sorted(out)                        # ['0train', '1test']
    (Xtr, ytr), (Xte, yte) = out[keys[0]], out[keys[1]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------
def standardise(Xtr, Xte, mode: str = 'train_stats'):
    if mode == 'train_stats':
        m = Xtr.mean(axis=(0, 2), keepdims=True)
        s = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
        return (Xtr - m) / s, (Xte - m) / s
    if mode == 'trial':
        f = lambda X: (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-6)
        return f(Xtr), f(Xte)
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# CSP (ablation only; corrected implementation)
# ---------------------------------------------------------------------------
def fit_csp_ovr(X, y, filters_per_class: int = 4, reg: float = 1e-6):
    """One-versus-rest CSP via the generalised eigenproblem
    Sigma_c w = lambda (Sigma_c + Sigma_rest) w.  For each class the
    filters_per_class/2 largest and smallest eigenvalues are kept.
    Returns W of shape (C, K * filters_per_class)."""
    def cov(Z):
        S = np.einsum('nct,ndt->ncd', Z, Z)
        S /= np.trace(S, axis1=1, axis2=2)[:, None, None]
        return S.mean(0)
    C = X.shape[1]; W = []
    for c in np.unique(y):
        Sa, Sb = cov(X[y == c]), cov(X[y != c])
        lam, V = eigh(Sa, Sa + Sb + reg * np.eye(C))      # ascending
        h = filters_per_class // 2
        top = np.arange(C - (filters_per_class - h), C)[::-1]   # max class-c variance
        bottom = np.arange(h)                                     # max rest variance
        W.append(V[:, np.r_[top, bottom]])
    return np.concatenate(W, axis=1)


def apply_spatial_filter(X, W):
    return np.einsum('nct,ck->nkt', X, W).astype(np.float32)


# ---------------------------------------------------------------------------
# Label budgets (task 3)
# ---------------------------------------------------------------------------
def nested_budget_indices(y, budgets, seed: int):
    """Class-balanced, nested subsets: indices[n] is contained in every
    larger budget.  One random order per class and seed."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y); K = len(classes)
    order = {c: rng.permutation(np.flatnonzero(y == c)) for c in classes}
    out = {}
    for n in sorted(budgets):
        assert n % K == 0, f'budget {n} not divisible by {K} classes'
        per = n // K
        assert all(len(order[c]) >= per for c in classes), f'budget {n} too large'
        out[n] = np.sort(np.concatenate([order[c][:per] for c in classes]))
    return out
