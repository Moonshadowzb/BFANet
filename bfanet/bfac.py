"""BFA-C: montage-agnostic continuous spatial prior (paper section 3.3).

    gamma^C_geo(i, j) = 1 / (1 + alpha * delta_ij^2),  delta_ij = arccos(p_i . p_j)   (eq. 7)
    psi_ij = [delta_ij, delta_bar_ij, x_i x_j, y_i^2 + y_j^2],
             delta_bar_ij = arccos(p_i . M p_j),  M = diag(-1, 1, 1)               (eq. 8)
    A_phi(i, j) = exp(f_phi(psi_ij)),  f_phi: 4 -> 16 -> 16 -> 1, SiLU           (eq. 9)
    phi_0 = argmin sum_{i,j in R} (f_phi(psi_ij) - log A~(i, j))^2                (eq. 10)

f_phi (369 parameters) is shared by all encoder layers; alpha stays per layer,
so each layer adds 1 + H scalars and the whole prior adds 369 + L(1 + H).

Spherical-harmonic variant
--------------------------
    A(i, j) = exp(e(p_i)^T W e(p_j)),  e = real spherical harmonics up to degree l.
W is symmetric (exchange symmetry).  Real SH have definite parity under the
mid-sagittal reflection, e(M p) = diag(s) e(p) with s in {+1, -1}; restricting
W to entries with s_a s_b = +1 makes A invariant to reflecting the montage,
the same symmetry the MLP inherits from psi.  Because the bilinear form can
interpolate the 16 x 16 reference matrix exactly once K = (l+1)^2 >= 16,
distillation uses a Laplace-Beltrami (smoothness) penalty (lambda = 0.1, chosen by
leave-one-mirror-pair-out cross-validation on R)
lambda * sum_ab [l_a(l_a+1) + l_b(l_b+1)] W_ab^2, i.e. the smoothest field on
the sphere that reproduces the discrete prior; it is solved in closed form.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import montage as mt

PSI_NAMES = ['delta', 'delta_bar', 'xx', 'yy']
LOG_BOUND = math.log(2.0)   # |log A| < ln 2: A in (0.5, 2), brackets the discrete range [0.7, 1.5]


def soft_bound(z, b=LOG_BOUND):
    return b * torch.tanh(z / b) if b else z
MIRROR = np.diag([-1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
def pair_descriptor(P: np.ndarray) -> np.ndarray:
    """(C, C, 4) psi_ij for unit positions P (C, 3)."""
    delta = mt.great_circle(P)
    delta_bar = mt.great_circle(P, P @ MIRROR)
    x, y = P[:, 0], P[:, 1]
    xx = x[:, None] * x[None, :]
    yy = y[:, None] ** 2 + y[None, :] ** 2
    return np.stack([delta, delta_bar, xx, yy], axis=-1)


# ---------------------------------------------------------------------------
# MLP affinity
# ---------------------------------------------------------------------------
class MLPAffinity(nn.Module):
    """f_phi: 4 -> 16 -> 16 -> 1 with SiLU; 369 parameters.

    ``drop`` zeroes descriptor entries (ablation) while keeping the parameter
    count fixed, so every ablation arm has the same capacity."""

    def __init__(self, hidden: int = 16, drop=(), bound: float = LOG_BOUND):
        super().__init__()
        self.bound = bound
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        mask = torch.ones(4)
        for name in drop:
            mask[PSI_NAMES.index(name)] = 0.0
        self.register_buffer('mask', mask)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:      # (..., 4) -> (...)
        return soft_bound(self.net(psi * self.mask).squeeze(-1), self.bound)


# ---------------------------------------------------------------------------
# Spherical-harmonic affinity
# ---------------------------------------------------------------------------
def real_sph_harm(P: np.ndarray, lmax: int):
    """Real SH basis (C, K), degrees l_a (K,), mirror parities s_a (K,).
    Polar angle from +z (vertex), azimuth from +x (right ear)."""
    from scipy.special import sph_harm_y
    theta = np.arccos(np.clip(P[:, 2], -1, 1))
    phi = np.arctan2(P[:, 1], P[:, 0])
    cols, degs, par = [], [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m == 0:
                v, s = Y.real, 1.0
            elif m > 0:        # ~ cos(m phi): phi -> pi - phi gives (-1)^m
                v, s = math.sqrt(2) * (-1) ** m * Y.real, (-1.0) ** m
            else:              # ~ sin(|m| phi): gives -(-1)^m
                v, s = math.sqrt(2) * (-1) ** m * Y.imag, -(-1.0) ** abs(m)
            cols.append(v); degs.append(l); par.append(s)
    return np.stack(cols, 1), np.array(degs), np.array(par)


class SHAffinity(nn.Module):
    """log A(i, j) = e_i^T W e_j, W symmetric and mirror-block-diagonal."""

    def __init__(self, lmax: int = 5, bound: float = LOG_BOUND):
        super().__init__()
        self.lmax, self.bound = lmax, bound
        K = (lmax + 1) ** 2
        _, degs, par = real_sph_harm(np.array([[0., 0., 1.]]), lmax)
        self.register_buffer('allowed', torch.tensor(np.outer(par, par) > 0, dtype=torch.float32))
        self.register_buffer('lap', torch.tensor(degs * (degs + 1), dtype=torch.float32))
        self.U = nn.Parameter(torch.zeros(K, K))

    @property
    def W(self):
        return 0.5 * (self.U + self.U.T) * self.allowed

    def n_free(self) -> int:
        a = self.allowed.cpu().numpy()
        return int(np.triu(a).sum())

    def log_A(self, E: torch.Tensor) -> torch.Tensor:          # E: (C, K)
        return soft_bound(E @ self.W @ E.T, self.bound)


# ---------------------------------------------------------------------------
# Per-layer spatial prior used inside BFANet
# ---------------------------------------------------------------------------
class ContinuousSpatialPrior(nn.Module):
    """log S = -log(1 + alpha delta^2) + log A_phi.  ``affinity`` is the
    module shared by all layers; ``alpha`` is per layer."""

    def __init__(self, channels, affinity: nn.Module, use_geo=True, use_affinity=True):
        super().__init__()
        P = mt.unit_sphere_positions(channels)
        self.register_buffer('d2', torch.tensor(mt.great_circle(P) ** 2, dtype=torch.float32))
        self.register_buffer('psi', torch.tensor(pair_descriptor(P), dtype=torch.float32))
        if isinstance(affinity, SHAffinity):
            E, _, _ = real_sph_harm(P, affinity.lmax)
            self.register_buffer('E', torch.tensor(E, dtype=torch.float32))
        self.affinity = affinity
        self.use_geo, self.use_aff = use_geo, use_affinity
        self.a = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    @property
    def alpha(self):
        return F.softplus(self.a)

    def log_affinity(self):
        if isinstance(self.affinity, SHAffinity):
            return self.affinity.log_A(self.E)
        return self.affinity(self.psi)

    def log_S(self):
        out = torch.zeros_like(self.d2)
        if self.use_geo:
            out = out - torch.log1p(self.alpha * self.d2)
        if self.use_aff:
            out = out + self.log_affinity()
        return out


# ---------------------------------------------------------------------------
# Distillation (eq. 10)
# ---------------------------------------------------------------------------
def reference_targets(ref=None):
    ref = ref or mt.REFERENCE_MONTAGE
    P = mt.unit_sphere_positions(ref)
    return ref, P, np.log(mt.base_affinity(ref))


def distill_mlp(aff: MLPAffinity, ref=None, seed: int = 0, adam_steps: int = 3000,
                lbfgs_steps: int = 300) -> dict:
    """Fit f_phi to log A~ on the reference montage (all ordered pairs,
    diagonal included).  Deterministic given ``seed``."""
    g = torch.Generator().manual_seed(seed)
    for p in aff.parameters():                       # reproducible init
        with torch.no_grad():
            if p.dim() > 1:
                bound = 1 / math.sqrt(p.shape[1]); p.uniform_(-bound, bound, generator=g)
            else:
                p.uniform_(-0.1, 0.1, generator=g)
    ref, P, T = reference_targets(ref)
    psi = torch.tensor(pair_descriptor(P), dtype=torch.float32)
    tgt = torch.tensor(T, dtype=torch.float32)
    opt = torch.optim.Adam(aff.parameters(), lr=1e-2)
    for _ in range(adam_steps):
        opt.zero_grad(); loss = F.mse_loss(aff(psi), tgt); loss.backward(); opt.step()
    opt = torch.optim.LBFGS(aff.parameters(), lr=0.5, max_iter=lbfgs_steps,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); l = F.mse_loss(aff(psi), tgt); l.backward(); return l
    opt.step(closure)
    with torch.no_grad():
        pred = aff(psi).numpy()
    return fidelity(pred, T)


def distill_sh(aff: SHAffinity, ref=None, lam: float = 0.1) -> dict:
    """Closed-form penalised least squares for the free entries of W."""
    ref, P, T = reference_targets(ref)
    E, _, _ = real_sph_harm(P, aff.lmax)
    K = E.shape[1]
    allowed = aff.allowed.cpu().numpy() > 0
    lap = aff.lap.cpu().numpy()
    idx = [(a, b) for a in range(K) for b in range(a, K) if allowed[a, b]]
    # design: e_i^T W e_j = sum_{a<=b} W_ab (E_ia E_jb + E_ib E_ja) [a != b]
    rows = []
    for a, b in idx:
        G = np.outer(E[:, a], E[:, b])
        rows.append((G + G.T).ravel() if a != b else G.ravel())
    Xd = np.stack(rows, 1)                                  # (C*C, n_free)
    pen = np.array([(lap[a] + lap[b]) * (2 if a != b else 1) for a, b in idx])
    b = aff.bound
    Tz = b * np.arctanh(T / b) if b else T             # invert the soft bound
    w = np.linalg.solve(Xd.T @ Xd + lam * np.diag(pen), Xd.T @ Tz.ravel())
    W = np.zeros((K, K))
    for (a, b), v in zip(idx, w):
        W[a, b] = W[b, a] = v
    with torch.no_grad():
        aff.U.copy_(torch.tensor(W, dtype=torch.float32))
    with torch.no_grad():
        pred = aff.log_A(torch.tensor(E, dtype=torch.float32)).numpy()
    return fidelity(pred, T)


def fidelity(pred: np.ndarray, target: np.ndarray) -> dict:
    """Pearson r between log A_phi0 and log A~ over all ordered pairs of R,
    and over unordered off-diagonal pairs (the non-redundant set)."""
    iu = np.triu_indices(target.shape[0], 1)
    r_all = float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    r_off = float(np.corrcoef(pred[iu], target[iu])[0, 1])
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return dict(r_all=r_all, r_offdiag=r_off, rmse=rmse,
                max_abs=float(np.abs(pred - target).max()))


def distill_mlp_restarts(aff: MLPAffinity, seed: int = 0, restarts: int = 5) -> dict:
    """Distil from ``restarts`` initialisations and keep the lowest-error fit.
    Deterministic, and performed before any EEG is seen."""
    best, best_state = None, None
    for r in range(restarts):
        info = distill_mlp(aff, seed=seed * 100 + r)
        if best is None or info['rmse'] < best['rmse']:
            best, best_state = info, {k: v.clone() for k, v in aff.state_dict().items()}
    aff.load_state_dict(best_state)
    return best


def build_affinity(kind: str = 'mlp', drop=(), distill: bool = True, seed: int = 0,
                   lmax: int = 5):
    """Create (and by default distil) the shared affinity module."""
    if kind == 'mlp':
        aff = MLPAffinity(drop=drop)
        info = distill_mlp_restarts(aff, seed=seed) if distill else {}
    elif kind == 'sh':
        aff = SHAffinity(lmax)
        info = distill_sh(aff) if distill else {}
    else:
        raise ValueError(kind)
    return aff, info
