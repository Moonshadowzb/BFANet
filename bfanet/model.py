"""BFANet: brain-functional decay attention (paper section 3).

Token layout
------------
A trial X in R^{C x T} is tokenised *per channel*, giving T_p patches per
channel and N = C * T_p tokens ordered channel-major:

    u = c * T_p + t,   pi_c(u) = u // T_p,   pi_t(u) = u % T_p.

Readout is the mean of the N token representations (default).  With a class
token (``pool='cls'``, as written in v3) the last layer's decay would receive
no gradient, because only the CLS query reaches the classifier and all decay
factors involving CLS equal one.

Decay modes
-----------
``logadd`` (default, recommended)
    softmax(Q K^T / sqrt(d_h) + log D + B) V
    i.e. D multiplies the *unnormalised attention weights* exp(logit);
    D = 1 recovers vanilla attention, D -> 0 suppresses a pair completely.
``mul_logits`` (literal eq. 2 of bfanet_iop_v3.tex, kept for ablation)
    softmax((Q K^T / sqrt(d_h)) * D + B) V

The log-decay decomposes additively (eq. 3 in log form):
    log D^{(h)}[u, v] = log g_geo(c_u, c_v) + log A(c_u, c_v)
                        + log rho^{(h)}(t_u, t_v).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import montage as mt


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _logit(p):
    p = torch.as_tensor(p, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


# ============================================================================
# Priors
# ============================================================================
class TemporalDecay(nn.Module):
    """Per-head multi-scale temporal decay rho^{(h)} (eq. 6).

    rho^{(h)}(t_u, t_v) = (gamma_h) ** (|t_u - t_v| * patch_sec / ref_sec)

    gamma_h in (g_lo, g_hi) through a sigmoid, initialised uniformly in
    ``init_range``.  With ``ref_sec`` equal to the patch duration the rate is
    per patch, exactly as in the paper; expressing it per ``ref_sec`` seconds
    makes the learned rates comparable across tokenisers (needed when the
    factor is transplanted into CBraMod in task 3).

    causal=True reproduces the one-sided form of the paper (future pairs get
    rho = 0, i.e. log rho = -inf).
    """

    def __init__(self, num_heads: int, init_range=(0.85, 0.99),
                 bounds=(0.5, 0.995), shared: bool = False,
                 causal: bool = False, ref_sec: float = 0.1):
        super().__init__()
        self.lo, self.hi = bounds
        self.causal, self.ref_sec, self.shared = causal, ref_sec, shared
        g0 = torch.linspace(*init_range, num_heads)
        if shared:  # ablation "shared gamma_t": one rate for all heads
            g0 = g0.mean().expand(1).clone()
        self.theta = nn.Parameter(_logit((g0 - self.lo) / (self.hi - self.lo)))
        self.num_heads = num_heads

    @property
    def gamma(self) -> torch.Tensor:
        g = self.lo + (self.hi - self.lo) * torch.sigmoid(self.theta)
        return g.expand(self.num_heads) if self.shared else g

    def log_rho(self, Tp: int, patch_sec: float) -> torch.Tensor:
        """(H, Tp, Tp) log-decay (<= 0, -inf where causally masked)."""
        t = torch.arange(Tp, device=self.theta.device, dtype=torch.float32)
        diff = t[:, None] - t[None, :]                      # t_u - t_v
        steps = diff.abs() * (patch_sec / self.ref_sec)
        out = torch.log(self.gamma)[:, None, None] * steps  # (H, Tp, Tp)
        if self.causal:
            out = out.masked_fill((diff < 0)[None], float('-inf'))
        return out


class DiscreteSpatialPrior(nn.Module):
    """Geometric decay (eq. 4) times functional affinity (eq. 5), in log form.

    log S(i, j) = -log(1 + alpha * d_ij^2)
                  + log A~(i, j) + w_{r(i)} + w_{r(j)}

    * d_ij is the great-circle distance on the unit sphere (identical to the
      BFA-C geometric factor, eq. 7), so BFA and BFA-C differ only in A.
    * alpha = softplus(a), initialised to 1.
    * Affinity rescaling: ``exp`` (default) uses exp(w_r(i) + w_r(j)), which
      equals A~ exactly at w = 0.  ``sigmoid`` is the literal
      sigma(w_r(i)) sigma(w_r(j)) of the paper, which at w = 0 scales every
      *listed* pair by 0.25 while unlisted pairs keep 1 (see report).
    * Electrodes outside the partition have A~ = 1 and no w (identity).
    Learnable scalars: alpha + 5 region weights = 6 per layer.
    """

    def __init__(self, channels, use_geo=True, use_affinity=True,
                 learn_affinity=True, affinity_param='exp'):
        super().__init__()
        P = mt.unit_sphere_positions(channels)
        self.register_buffer('d2', torch.tensor(mt.great_circle(P) ** 2,
                                                dtype=torch.float32))
        self.register_buffer('logA0', torch.tensor(np.log(mt.base_affinity(channels)),
                                                   dtype=torch.float32))
        r = torch.as_tensor(mt.region_index(channels))
        self.register_buffer('region', r.clamp(min=0))
        self.register_buffer('listed', (r >= 0).float())
        self.use_geo, self.use_aff = use_geo, use_affinity
        self.affinity_param = affinity_param
        self.a = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
        self.w = nn.Parameter(torch.zeros(len(mt.REGION_NAMES)),
                              requires_grad=learn_affinity)

    @property
    def alpha(self):
        return F.softplus(self.a)

    def log_affinity(self) -> torch.Tensor:
        wc = self.w[self.region]
        if self.affinity_param == 'exp':
            s = wc * self.listed
            return self.logA0 + s[:, None] + s[None, :]
        # literal: sigma(w_i) sigma(w_j) for listed, 1 for unlisted
        ls = F.logsigmoid(wc) * self.listed
        return self.logA0 + ls[:, None] + ls[None, :]

    def log_S(self) -> torch.Tensor:
        out = torch.zeros_like(self.d2)
        if self.use_geo:
            out = out - torch.log1p(self.alpha * self.d2)
        if self.use_aff:
            out = out + self.log_affinity()
        return out


class RelativeBias(nn.Module):
    """Learnable relative-position bias B, factorised to stay O(C^2 + T_p).

    B^{(h)}[u, v] = b_t^{(h)}[t_u - t_v] + b_c^{(h)}[c_u, c_v]
    plus one scalar per head for query->CLS and CLS->key pairs.
    The dense H x N x N bias of the paper would need ~6e7 parameters per layer
    for N ~ 2700 and ~5e6 for N = 770; this factorisation has
    H (2 T_p - 1 + C^2 + 2) parameters.  b_c depends on the montage and is
    re-initialised at cross-montage transfer (paper section 4.9).
    """

    def __init__(self, num_heads: int, C: int, Tp: int):
        super().__init__()
        self.bt = nn.Parameter(torch.zeros(num_heads, 2 * Tp - 1))
        self.bc = nn.Parameter(torch.zeros(num_heads, C, C))
        self.bcls = nn.Parameter(torch.zeros(num_heads, 2))
        self.Tp, self.C = Tp, C

    def forward(self, pc, pt, with_cls: bool = True) -> torch.Tensor:
        H = self.bt.shape[0]
        rel = (pt[:, None] - pt[None, :]) + self.Tp - 1
        core = self.bt[:, rel] + self.bc[:, pc][:, :, pc]           # (H, N, N)
        if not with_cls:
            return core
        N = core.shape[-1]
        out = core.new_zeros(H, N + 1, N + 1)
        out[:, 1:, 1:] = core
        out[:, 1:, 0] = self.bcls[:, 0:1]
        out[:, 0, 1:] = self.bcls[:, 1:2]
        return out


# ============================================================================
# Attention and encoder
# ============================================================================
class BFAttention(nn.Module):
    def __init__(self, d: int, H: int, dropout: float, mode: str = 'logadd'):
        super().__init__()
        assert d % H == 0 and mode in ('logadd', 'mul_logits')
        self.H, self.dh, self.mode, self.p = H, d // H, mode, dropout
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)

    def forward(self, x, log_D, B, return_attn=False):
        Bsz, N, d = x.shape
        q, k, v = self.qkv(x).view(Bsz, N, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        p = self.p if self.training else 0.0
        if self.mode == 'logadd' and not return_attn:
            bias = B if log_D is None else B + log_D
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(0),
                                               dropout_p=p)
            w = None
        else:
            logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            if log_D is not None:
                if self.mode == 'logadd':
                    logits = logits + log_D
                else:  # literal: logits * D, with D = exp(log D) (0 where masked)
                    logits = logits * torch.exp(log_D)
            w = torch.softmax(logits + B, dim=-1)
            o = F.dropout(w, p, self.training) @ v
        o = o.transpose(1, 2).reshape(Bsz, N, d)
        return (self.proj(o), w) if return_attn else self.proj(o)


class Block(nn.Module):
    def __init__(self, d, H, dropout, mode, ffn_ratio=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = BFAttention(d, H, dropout, mode)
        h = ffn_ratio * d
        self.ffn = nn.Sequential(nn.Linear(d, 2 * h), nn.GLU(dim=-1),
                                 nn.Dropout(dropout), nn.Linear(h, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, log_D, B, return_attn=False):
        a = self.attn(self.n1(x), log_D, B, return_attn)
        a, w = a if return_attn else (a, None)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.n2(x)))
        return (x, w) if return_attn else x


class ChannelwiseTokenizer(nn.Module):
    """Per-channel temporal convolution -> patch tokens (section 3.4).

    Shared across channels, so every token keeps its electrode identity.
      Conv1d(1, F, k, same padding) -> BN -> GELU      temporal filtering
      Conv1d(F, d, P, stride P)                        non-overlapping patches
    With T = 875 (0.5-4 s at 250 Hz) and P = 25 (100 ms): T_p = 35, N = 770.
    """

    def __init__(self, d: int, C: int, T: int, kernel: int = 25,
                 filters: int = 16, patch: int = 25):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(1, filters, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(filters), nn.GELU())
        self.patch = nn.Conv1d(filters, d, patch, stride=patch)
        self.Tp = (T - patch) // patch + 1
        self.C = C
        self.chan_embed = nn.Parameter(torch.zeros(C, d))
        nn.init.trunc_normal_(self.chan_embed, std=0.02)
        pc = torch.arange(C).repeat_interleave(self.Tp)
        pt = torch.arange(self.Tp).repeat(C)
        self.register_buffer('pi_c', pc, persistent=False)
        self.register_buffer('pi_t', pt, persistent=False)

    def forward(self, x):                                 # x: (B, C, T)
        Bsz, C, T = x.shape
        z = self.patch(self.temporal(x.reshape(Bsz * C, 1, T)))   # (B*C, d, Tp)
        z = z.reshape(Bsz, C, -1, self.Tp).transpose(2, 3)        # (B, C, Tp, d)
        z = z + self.chan_embed[None, :, None, :]
        return z.reshape(Bsz, C * self.Tp, -1)                    # u = c*Tp + t


# ============================================================================
# Full model
# ============================================================================
@dataclass
class BFAConfig:
    channels: list = field(default_factory=lambda: list(mt.CHANNELS_2A))
    n_classes: int = 4
    T: int = 875
    fs: float = 250.0
    d: int = 64
    depth: int = 4
    heads: int = 8
    dropout: float = 0.2
    kernel: int = 25
    filters: int = 16
    patch: int = 25
    # prior switches
    prior: str = 'bfa'             # 'bfa' | 'vanilla' | 'bfac' (task 2)
    decay_mode: str = 'logadd'     # 'logadd' | 'mul_logits' (literal eq. 2)
    causal: bool = False           # True: literal one-sided eq. 6
    use_geo: bool = True
    use_affinity: bool = True
    learn_affinity: bool = True    # False: hard prior A~ only
    affinity_param: str = 'exp'    # 'exp' | 'sigmoid' (literal eq. 5)
    shared_gamma: bool = False
    gamma_init: tuple = (0.85, 0.99)
    spatial_prior: bool = True     # set False automatically for CSP input
    pool: str = 'mean'             # 'mean' (default) | 'cls' (paper v3)
    # BFA-C (prior='bfac')
    bfac_kind: str = 'mlp'         # 'mlp' | 'sh'
    psi_drop: tuple = ()           # subset of ('delta','delta_bar','xx','yy')
    distill: bool = True
    distill_seed: int = 0
    sh_lmax: int = 5


class BFANet(nn.Module):
    def __init__(self, cfg: BFAConfig, spatial_prior_factory=None):
        super().__init__()
        self.cfg = cfg
        C = len(cfg.channels)
        self.tok = ChannelwiseTokenizer(cfg.d, C, cfg.T, cfg.kernel,
                                        cfg.filters, cfg.patch)
        Tp = self.tok.Tp
        self.patch_sec = cfg.patch / cfg.fs
        self.with_cls = cfg.pool == 'cls'
        if self.with_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d))
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(cfg.d, cfg.heads, cfg.dropout, cfg.decay_mode) for _ in range(cfg.depth)])
        self.rel = nn.ModuleList([RelativeBias(cfg.heads, C, Tp) for _ in range(cfg.depth)])
        self.use_prior = cfg.prior != 'vanilla'
        if self.use_prior:
            self.temporal = nn.ModuleList([
                TemporalDecay(cfg.heads, cfg.gamma_init, shared=cfg.shared_gamma,
                              causal=cfg.causal, ref_sec=self.patch_sec)
                for _ in range(cfg.depth)])
            if cfg.spatial_prior and (cfg.use_geo or cfg.use_affinity):
                if cfg.prior == 'bfa':
                    mk = lambda: DiscreteSpatialPrior(
                        cfg.channels, cfg.use_geo, cfg.use_affinity,
                        cfg.learn_affinity, cfg.affinity_param)
                elif spatial_prior_factory is not None:
                    mk = spatial_prior_factory
                else:  # BFA-C: one affinity shared by all layers, alpha per layer
                    from .bfac import build_affinity, ContinuousSpatialPrior
                    aff, self.bfac_info = build_affinity(
                        cfg.bfac_kind, cfg.psi_drop, cfg.distill, cfg.distill_seed, cfg.sh_lmax)
                    self.shared_affinity = aff
                    mk = lambda: ContinuousSpatialPrior(cfg.channels, aff, cfg.use_geo,
                                                        cfg.use_affinity)
                self.spatial = nn.ModuleList([mk() for _ in range(cfg.depth)])
            else:
                self.spatial = None
        self.norm = nn.LayerNorm(cfg.d)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.d, cfg.n_classes))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---- decay matrix --------------------------------------------------------
    def log_decay(self, layer: int) -> torch.Tensor | None:
        """(H, N', N') log D for one layer; N' = N (+1 with CLS, row/col = 0)."""
        if not self.use_prior:
            return None
        pc, pt, Tp = self.tok.pi_c, self.tok.pi_t, self.tok.Tp
        core = self.temporal[layer].log_rho(Tp, self.patch_sec)[:, pt][:, :, pt]
        if self.spatial is not None:
            S = self.spatial[layer].log_S()
            core = core + S[pc][:, pc][None]
        if not self.with_cls:
            return core
        H, N, _ = core.shape
        out = core.new_zeros(H, N + 1, N + 1)
        out[:, 1:, 1:] = core
        return out

    def forward(self, x, return_attn=False):
        z = self.tok(x)
        if self.with_cls:
            z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], dim=1)
        maps = []
        for i, blk in enumerate(self.blocks):
            B = self.rel[i](self.tok.pi_c, self.tok.pi_t, self.with_cls)
            out = blk(z, self.log_decay(i), B, return_attn)
            if return_attn:
                z, w = out; maps.append(w)
            else:
                z = out
        z = self.norm(z)
        feat = z[:, 0] if self.with_cls else z.mean(1)
        logits = self.head(feat)
        return (logits, maps) if return_attn else logits

    # ---- bookkeeping ---------------------------------------------------------
    def prior_parameters(self):
        ps = {}
        if self.use_prior:
            mods = [self.temporal] + ([self.spatial] if self.spatial is not None else [])
            for mod in mods:
                for p in mod.parameters():      # deduplicates the shared f_phi
                    ps[id(p)] = p
        return list(ps.values())
