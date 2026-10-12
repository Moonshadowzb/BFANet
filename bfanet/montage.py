"""Electrode geometry and the discrete functional prior of BFANet.

All spatial quantities are computed from the MNE ``standard_1005`` template,
so that BFA (section 3.2) and BFA-C (section 3.3) share exactly the same
coordinates and differ only in how the affinity A is parameterised.

Coordinate frame (paper, section 3.3)
-------------------------------------
Unit sphere, x -> right pre-auricular point, y -> nasion, z -> vertex.
The template positions are (i) centred on a least-squares sphere fitted to
all 10-05 electrodes and (ii) rotated about the origin so that the central
coronal great circle T7-C3-Cz-C4-T8 lies in the plane y = 0.  Step (ii) makes
the pair-descriptor term y_i^2 + y_j^2 a genuine distance to the central
plane, as the paper states.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Montages
# ---------------------------------------------------------------------------
# BCI Competition IV 2a, 22 EEG channels in the order used by MOABB
# (BNCI2014_001) and by the original GDF files.
CHANNELS_2A = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz',
]

# Five-region partition of the paper (section 3.2).  Electrodes that are not
# listed (for 2a: C5, C6, P1, Pz, P2, POz) receive the identity multiplier.
FUNCTIONAL_REGIONS = {
    'motor_left':    ['C3', 'CP3', 'FC3'],
    'motor_right':   ['C4', 'CP4', 'FC4'],
    'motor_central': ['Cz', 'FCz', 'CPz'],
    'premotor':      ['FC1', 'FC2', 'C1', 'C2', 'CP1', 'CP2'],
    'frontal':       ['Fz'],
}
REGION_NAMES = list(FUNCTIONAL_REGIONS)
A_SAME, A_CONTRA, A_OTHER = 1.5, 0.7, 1.0

# Reference montage R used for distillation (eq. 10): the listed electrodes.
REFERENCE_MONTAGE = [c for r in REGION_NAMES for c in FUNCTIONAL_REGIONS[r]]


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def _rotation_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix R with R @ a = b (unit vectors)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


@functools.lru_cache(maxsize=1)
def _template_unit_positions() -> dict:
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mont = mne.channels.make_standard_montage('standard_1005')
    pos = mont.get_positions()['ch_pos']
    names = list(pos)
    P = np.array([pos[n] for n in names], dtype=np.float64)
    # (i) least-squares sphere: |p|^2 = 2 c.p + (r^2 - |c|^2)
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    U = P - sol[:3]
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    # (ii) rotate the central great circle into y = 0
    idx = {n: i for i, n in enumerate(names)}
    ring = U[[idx[k] for k in ('T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8')]]
    normal = np.linalg.svd(ring, full_matrices=False)[2][-1]
    if normal[1] < 0:
        normal = -normal
    U = U @ _rotation_to(normal, np.array([0., 1., 0.])).T
    # (iii) enforce exact left/right symmetry of the template: average each
    # electrode with the reflection of its homologue (midline: x = 0), so the
    # priors carry no hemispheric bias from template digitisation noise.
    M = np.diag([-1., 1., 1.])
    out = {}
    for n in names:
        h = mirror_name(n)
        q = U[idx[n]] if h is None or h not in idx else 0.5 * (U[idx[n]] + U[idx[h]] @ M)
        if h == n:
            q = q.copy(); q[0] = 0.0
        out[n] = q / np.linalg.norm(q)
    return out


def mirror_name(name: str):
    """Homologous 10-05 label: odd <-> even number (1<->2, 3<->4, ...),
    'z' is its own mirror.  Returns None for labels without a number."""
    import re
    m = re.fullmatch(r'([A-Za-z]+?)(z|\d+)(h?)', name)
    if m is None:
        return None
    pre, num, h = m.groups()
    if num == 'z':
        return name
    k = int(num)
    return f'{pre}{k + 1 if k % 2 else k - 1}{h}'


def unit_sphere_positions(channels) -> np.ndarray:
    """(C, 3) float64 array of unit vectors for the given channel names."""
    tpl = _template_unit_positions()
    lut = {k.lower(): v for k, v in tpl.items()}
    missing = [c for c in channels if c.lower() not in lut]
    if missing:
        raise KeyError(f'no 10-05 template position for {missing}')
    return np.stack([lut[c.lower()] for c in channels])


def great_circle(P: np.ndarray, Q: np.ndarray | None = None) -> np.ndarray:
    """Pairwise great-circle distance arccos(p_i . q_j) on the unit sphere."""
    Q = P if Q is None else Q
    return np.arccos(np.clip(P @ Q.T, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Discrete prior
# ---------------------------------------------------------------------------
def region_index(channels) -> np.ndarray:
    """r(c) for each channel; -1 for electrodes outside the partition."""
    lut = {c.lower(): k for k, r in enumerate(REGION_NAMES)
           for c in FUNCTIONAL_REGIONS[r]}
    return np.array([lut.get(c.lower(), -1) for c in channels], dtype=np.int64)


def base_affinity(channels) -> np.ndarray:
    """Discrete base affinity A~ (C, C): 1.5 same region, 0.7 contralateral
    motor, 1.0 otherwise and for any pair involving an unlisted electrode."""
    r = region_index(channels)
    L, R = REGION_NAMES.index('motor_left'), REGION_NAMES.index('motor_right')
    A = np.full((len(channels),) * 2, A_OTHER, dtype=np.float64)
    listed = r >= 0
    same = (r[:, None] == r[None, :]) & listed[:, None] & listed[None, :]
    contra = ((r[:, None] == L) & (r[None, :] == R)) | ((r[:, None] == R) & (r[None, :] == L))
    A[same] = A_SAME
    A[contra] = A_CONTRA
    return A
