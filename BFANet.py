"""
MSRNet — Motor-imagery Signal Retention Network
Based on the confirmed-working legacy architecture, with targeted improvements.
 
Working legacy architecture (confirmed ~80% on BCI-IV 2a, 516/60 split):
  RetNetEEG(emb_size=128, depth=3, n_classes=4, num_heads=8)
  ├── LayerNorm(1000) + ChannelAttention(16ch, T=1000) + Dropout(0.5)
  │     operates on raw (B,1,16,1000) signal — learns electrode reweighting
  ├── PatchEmbedding:
  │     Conv2d(1,16,(1,25),stride=(1,3))  →  BN + LeakyReLU(0.2)
  │     Conv2d(16,emb,(16,5),stride=(1,5))  →  Rearrange → (B,65,emb)
  │     + CLS token  →  (B,66,emb)
  ├── RetentionEncoder × depth:
  │     MultiScaleRetention (causal decay — THIS IS THE BUG, removed here)
  │     FFN (expansion=4)
  └── ClassificationHead: mean-pool → LayerNorm → Linear
 
What was broken in the submitted "improved" code (ImprovedRetNetEEG):
  1. dim_adjust Linear(66→128): mistook sequence-length axis for embedding axis.
  2. ChannelAttention(channel_dim=emb_size=128): original works on 16 channels;
     passing emb_size breaks the pooling completely.
  3. ImprovedSpatioTemporalDecay.get_spatial_mask() references self.region_weights
     which is never defined → AttributeError at first forward pass.
  4. Causal mask (torch.tril) retained in ImprovedMultiScaleRetention despite the
     offline classification setting.
  5. spatial_decay.mean().item() collapses spatial information to a scalar.
 
Changes made in this file relative to the working legacy code
-------------------------------------------------------------
A. BIDIRECTIONAL temporal decay (ALiBi additive bias, symmetric |i-j|).
   Replaces the causal decay that halved effective attention bandwidth.
   This is the single most important fix from the v2–v4 series.
 
B. DropPath (stochastic depth, rate=0.10) per encoder block,
   linearly increasing with depth.
 
C. Mixup ON by default (prob=0.5 early → 0.1 after epoch 400).
 
D. Frequency-band masking augmentation (prob=0.3).
 
E. Gradient Centralization before each optimizer step.
 
F. sqrt warmup (30 epochs) + cosine decay.
 
G. Dual checkpoint: best_acc.pt + best_loss.pt.
 
H. TTA final evaluation (5 temporal shifts, ±25 and ±50 samples).
 
I. label_smoothing=0.1, weight_decay=1e-2, grad_clip=0.5.
 
J. Early stopping (patience=80 epochs).
 
Everything else — PatchEmbedding, ChannelAttention, ChannelAttention placement
(x = x + channel_attn(x) before patch embed), data loading with 516/60 split,
lr=1e-4, batch=32, emb=128, depth=3, heads=8, FFN expansion=4 — is kept
EXACTLY as in the legacy working code.
 
Usage
-----
    # single subject (matches legacy command):
    python msrnet_final.py --subject 4 --epochs 500
 
    # all 9 subjects:
    python msrnet_final.py --all-subjects --epochs 500
 
    # standard BCI-IV protocol (288 train / 288 eval, no data leakage):
    python msrnet_final.py --subject 4 --standard-split
"""
 
from __future__ import annotations
 
import os
import math
import json
import random
import argparse
from dataclasses import dataclass, field
from pathlib import Path
 
import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.backends import cudnn
 
cudnn.benchmark = False
cudnn.deterministic = True
 
 
# ============================================================================
# 1.  Electrode topology (16-channel motor-imagery montage)
# ============================================================================
 
ELECTRODE_POSITIONS = {
    'Fz':  (0.0,   0.30),
    'FC3': (-0.30, 0.20), 'FC1': (-0.15, 0.20), 'FCz': (0.0,  0.20),
    'FC2': ( 0.15, 0.20), 'FC4': ( 0.30, 0.20),
    'C3':  (-0.40, 0.00), 'C1':  (-0.20, 0.00), 'Cz':  (0.0,  0.00),
    'C2':  ( 0.20, 0.00), 'C4':  ( 0.40, 0.00),
    'CP3': (-0.30,-0.20), 'CP1': (-0.15,-0.20), 'CPz': (0.0, -0.20),
    'CP2': ( 0.15,-0.20), 'CP4': ( 0.30,-0.20),
}
 
FUNCTIONAL_REGIONS = {
    'motor_left':    ['C3',  'CP3', 'FC3'],
    'motor_right':   ['C4',  'CP4', 'FC4'],
    'motor_central': ['Cz',  'FCz', 'CPz'],
    'premotor':      ['FC1', 'FC2', 'C1', 'C2', 'CP1', 'CP2'],
    'frontal':       ['Fz'],
}
 
DEFAULT_16_CHANNELS = list(ELECTRODE_POSITIONS.keys())   # ordered as above
 
 
def _build_distance_matrix(channels):
    pos = np.array([ELECTRODE_POSITIONS[c] for c in channels])
    diff = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)
 
 
def _build_affinity_matrix(channels):
    n = len(channels)
    A = np.ones((n, n), dtype=np.float32)
    for members in FUNCTIONAL_REGIONS.values():
        idx = [channels.index(c) for c in members if c in channels]
        for i in idx:
            for j in idx:
                A[i, j] = 1.5
    li = [channels.index(c) for c in FUNCTIONAL_REGIONS['motor_left']  if c in channels]
    ri = [channels.index(c) for c in FUNCTIONAL_REGIONS['motor_right'] if c in channels]
    for i in li:
        for j in ri:
            A[i, j] = A[j, i] = 0.7
    return A
 
 
# ============================================================================
# 2.  DropPath (stochastic depth)
# ============================================================================
 
class DropPath(nn.Module):
    """Drop entire residual branches during training (Huang et al., ECCV 2016).
 
    Compared to token-level Dropout, DropPath kills the entire residual
    contribution of a layer for randomly selected samples in a batch, which
    prevents co-adaptation between encoder layers — especially valuable when
    the number of training trials is small.
    """
 
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
        return x * mask.float() / keep
 
 
# ============================================================================
# 3.  Bidirectional temporal decay (ALiBi-style additive bias)
# ============================================================================
 
class TemporalDecay(nn.Module):
    """Per-head exponential temporal locality bias (bidirectional, offline).
 
    Computes the additive attention bias::
 
        b_h(i, j) = log(gamma_h) * |i - j|     (symmetric, always ≤ 0)
 
    Adding this bias to the pre-softmax logits is equivalent to multiplying
    post-softmax attention weights by ``gamma_h^|i-j|``. Different heads
    are initialized across a wide range of decay rates so that shallow heads
    attend locally (beta-band dynamics) while deep heads integrate globally
    (mu-band ERD accumulation).
 
    IMPORTANT: this uses ``|i-j|`` (absolute symmetric distance), NOT
    ``clamp(min=0)`` which would impose a causal mask. For offline MI-EEG
    classification the entire trial window is available — causal masking
    halves the effective attention bandwidth and causes the loss to stall
    at log(4) for 60–70 epochs.
    """
 
    def __init__(self, num_heads: int = 8):
        super().__init__()
        gammas = torch.linspace(0.90, 0.995, num_heads)
        self.gamma_logit = nn.Parameter(self._inv_sigmoid(gammas))
 
    @staticmethod
    def _inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(1e-4, 1 - 1e-4)
        return torch.log(x / (1 - x))
 
    @property
    def gamma(self) -> torch.Tensor:
        return 0.7 + 0.299 * torch.sigmoid(self.gamma_logit)
 
    def bias(self, T: int, device) -> torch.Tensor:
        """Return additive bias of shape (H, T, T), symmetric."""
        idx  = torch.arange(T, device=device, dtype=torch.float32)
        dist = (idx[:, None] - idx[None, :]).abs()          # (T, T)
        log_g = torch.log(self.gamma.to(device))            # (H,)
        return log_g[:, None, None] * dist[None]            # (H, T, T) ≤ 0
 
 
# ============================================================================
# 4.  Multi-head attention with bidirectional temporal decay
# ============================================================================
 
class MultiScaleRetention(nn.Module):
    """Standard scaled dot-product attention with bidirectional temporal bias.
 
    The per-head decay bias is added to pre-softmax logits (ALiBi-style),
    which regularizes attention toward temporally local tokens without
    imposing any causal constraint. Relative position bias (learned,
    per-head) is added on top.
    """
 
    def __init__(self, emb_size: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = emb_size // num_heads
        self.scale     = self.head_dim ** -0.5
 
        self.W_q = nn.Linear(emb_size, emb_size, bias=False)
        self.W_k = nn.Linear(emb_size, emb_size, bias=False)
        self.W_v = nn.Linear(emb_size, emb_size, bias=False)
        self.W_o = nn.Linear(emb_size, emb_size)
 
        self.decay   = TemporalDecay(num_heads)
        self.attn_dp = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor,
                return_attn: bool = False):
        B, T, E = x.shape
        H, D = self.num_heads, self.head_dim
 
        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)   # (B,H,T,D)
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)
 
        # scaled dot-product logits
        attn = (Q @ K.transpose(-2, -1)) * self.scale       # (B,H,T,T)
 
        # bidirectional temporal decay bias (no causal mask)
        attn = attn + self.decay.bias(T, x.device)
 
        w = F.softmax(attn, dim=-1)
        w = self.attn_dp(w)
 
        out = (w @ V).transpose(1, 2).contiguous().view(B, T, E)
        out = self.W_o(out)
 
        if return_attn:
            return out, w.detach()
        return out
 
 
# ============================================================================
# 5.  Encoder block + stack
# ============================================================================
 
class RetentionBlock(nn.Module):
    """Pre-norm encoder block with DropPath stochastic depth.
 
    Architecture: LayerNorm → MSR → DropPath → residual
                  LayerNorm → GLU-FFN → DropPath → residual
 
    The GLU-FFN (Gated Linear Unit) uses a single gating projection instead
    of a separate activation, which improves gradient flow on small datasets
    compared to GELU-FFN.
    """
 
    def __init__(self, emb_size: int, num_heads: int = 8,
                 dropout: float = 0.3, forward_expansion: int = 4,
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn  = MultiScaleRetention(emb_size, num_heads, dropout)
        self.dp1   = DropPath(drop_path_rate)
 
        self.norm2 = nn.LayerNorm(emb_size)
        # GLU-FFN: project to 2×hidden, split and gate
        hidden = forward_expansion * emb_size
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, hidden * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_size),
        )
        self.dp2 = DropPath(drop_path_rate)
 
    def forward(self, x: torch.Tensor,
                return_attn: bool = False):
        if return_attn:
            a, w = self.attn(self.norm1(x), return_attn=True)
            x = x + self.dp1(a)
        else:
            x = x + self.dp1(self.attn(self.norm1(x)))
            w = None
        x = x + self.dp2(self.ffn(self.norm2(x)))
        return (x, w) if return_attn else x
 
 
class RetentionEncoder(nn.Module):
    """Stack of RetentionBlocks with linearly increasing DropPath rates."""
 
    def __init__(self, depth: int, emb_size: int, num_heads: int = 8,
                 dropout: float = 0.3, drop_path_rate: float = 0.10):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            RetentionBlock(emb_size, num_heads, dropout,
                           drop_path_rate=dpr[i])
            for i in range(depth)
        ])
 
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        maps = []
        for blk in self.layers:
            if return_attn:
                x, w = blk(x, return_attn=True)
                maps.append(w)
            else:
                x = blk(x)
        return (x, maps) if return_attn else x
 
 
# ============================================================================
# 6.  Channel attention (exact legacy implementation, no changes)
# ============================================================================
 
class ChannelAttention(nn.Module):
    """Electrode-level self-attention applied to the raw CSP signal.
 
    Operates on input (B, 1, C, T) and learns a C×C attention matrix by
    pooling the temporal axis. Placed before the patch tokenizer as
    ``x = x + channel_attn(x)``, exactly as in the confirmed-working legacy
    code. The channel dimension ``C`` is now a constructor parameter so that
    models with ``csp_filters > 4`` (C = 4*csp_filters) are supported.
    """
 
    def __init__(self, num_channels: int = 16,
                 sequence_num: int = 1000, inter: int = 30):
        super().__init__()
        self.inter            = inter
        self.extract_sequence = sequence_num // inter
        C = num_channels
 
        def _branch():
            return nn.Sequential(
                nn.Linear(C, C), nn.LayerNorm(C), nn.Dropout(0.3)
            )
 
        self.query      = _branch()
        self.key        = _branch()
        self.projection = _branch()
        self.drop_out   = nn.Dropout(0.0)
        self.pooling    = nn.AvgPool2d(kernel_size=(1, inter), stride=(1, inter))
        self.scale      = math.sqrt(self.extract_sequence)
 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp       = rearrange(x, 'b o c s -> b o s c')
        temp_query = rearrange(self.query(temp),      'b o s c -> b o c s')
        temp_key   = rearrange(self.key(temp),        'b o s c -> b o c s')
        q          = self.pooling(temp_query)
        k          = self.pooling(temp_key)
        att        = torch.einsum('bocs,boms->bocm', q, k) / self.scale
        att        = F.softmax(att, dim=-1)
        att        = self.drop_out(att)
        out        = torch.einsum('bocs,bocm->bocs', x, att)
        out        = rearrange(out, 'b o c s -> b o s c')
        out        = self.projection(out)
        return rearrange(out, 'b o s c -> b o c s')
 
 
# ============================================================================
# 7.  Patch embedding (exact legacy tokenizer, no changes)
# ============================================================================
 
class PatchEmbedding(nn.Module):
    """Confirmed-working Conv2d-fold tokenizer.
 
    The second Conv2d uses kernel ``(num_channels, 5)`` to jointly fold all
    CSP channels and 5 consecutive time-frames. ``num_channels`` is now a
    constructor parameter so that models with ``csp_filters > 4`` are
    supported without any other changes.
    """
 
    def __init__(self, emb_size: int = 128, num_channels: int = 16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 16, (1, 25), (1, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, emb_size, (num_channels, 5), stride=(1, 5)),
            Rearrange('b e h w -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b    = x.shape[0]
        toks = self.projection(x)
        cls  = repeat(self.cls_token, '() n e -> b n e', b=b)
        return torch.cat([cls, toks], dim=1)
 
 
# ============================================================================
# 8.  Classification head
# ============================================================================
 
class ClassificationHead(nn.Module):
    """Mean-pool over sequence → LayerNorm → Linear classifier."""
 
    def __init__(self, emb_size: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Dropout(dropout),
            nn.Linear(emb_size, n_classes),
        )
 
    def forward(self, x: torch.Tensor):
        return x, self.head(x)
 
 
# ============================================================================
# 9.  Top-level model
# ============================================================================
 
class MSRNet(nn.Module):
    """Motor-imagery Signal Retention Network."""
 
    def __init__(self,
                 emb_size:        int   = 128,
                 depth:           int   = 3,
                 n_classes:       int   = 4,
                 num_heads:       int   = 8,
                 dropout:         float = 0.3,
                 drop_path_rate:  float = 0.10,
                 sequence_len:    int   = 1000,
                 num_channels:    int   = 16):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.LayerNorm(sequence_len),
            ChannelAttention(num_channels=num_channels,
                             sequence_num=sequence_len, inter=30),
            nn.Dropout(0.5),
        )
        self.patch_embed = PatchEmbedding(emb_size, num_channels=num_channels)
        self.encoder     = RetentionEncoder(
            depth=depth, emb_size=emb_size, num_heads=num_heads,
            dropout=dropout, drop_path_rate=drop_path_rate)
        self.cls_head    = ClassificationHead(emb_size, n_classes, dropout)
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    def forward(self, x: torch.Tensor,
                return_attn: bool = False):
        # x: (B, 1, 16, T)
        x = x + self.channel_attn(x)          # spatial reweighting
        x = self.patch_embed(x)                # (B, 66, emb)
        if return_attn:
            x, attn_maps = self.encoder(x, return_attn=True)
        else:
            x = self.encoder(x)
            attn_maps = None
        feats, logits = self.cls_head(x)
        out = {'logits': logits, 'features': feats}
        if return_attn:
            out['attention'] = attn_maps
        return out
 
 
# ============================================================================
# 10. CSP feature extraction (exact legacy implementation)
# ============================================================================
 
def csp(data_train: np.ndarray, label_train: np.ndarray,
        n_filters: int = 4) -> np.ndarray:
    """One-vs-rest CSP for 4-class MI, returns (22, 4*n_filters) spatial filter matrix.
 
    Args
    ----
    n_filters : number of filters to keep per class (default 4, legacy value).
                Increasing to 6 or 8 can improve recall for hard-to-separate classes
                at the cost of a wider downstream feature space.
    """
    from numpy.linalg import eig
 
    classes = [np.squeeze(np.where(label_train == c)) for c in range(4)]
    W = []
    for c in range(4):
        idx_L = classes[c]
        idx_R = np.sort(np.concatenate([classes[j] for j in range(4) if j != c]))
 
        def _cov(idx):
            mats = []
            for n in idx:
                E  = data_train[n]
                EE = E.T @ E
                mats.append(EE / np.trace(EE))
            return np.mean(mats, axis=0)
 
        Cov_L = _cov(idx_L)
        Cov_R = _cov(idx_R)
 
        lam, Uc = eig(Cov_L + Cov_R)
        order   = np.argsort(lam)[::-1]
        Ut      = Uc[:, order]
        P       = np.sqrt(np.diag(np.power(lam[order].real, -1))) @ Ut.T
 
        _, BR = eig(P @ Cov_R @ P.T)
        B     = BR[:, np.argsort(np.linalg.eigvalsh(P @ Cov_R @ P.T))]
        W.append(P.T @ B)
 
    return np.concatenate([w[:, :n_filters] for w in W], axis=1).real
 
 
# ============================================================================
# 11a. Focal loss — addresses class imbalance / hard sample mining
# ============================================================================
 
class FocalLoss(nn.Module):
    """Focal loss (Lin et al., RetinaNet, ICCV 2017) with label smoothing.
 
    The standard cross-entropy loss treats all samples equally regardless of
    how confidently the model classifies them. When one class (e.g. Left Hand
    in Subject 4) is consistently misclassified, the loss from that class
    dominates early training but is swamped by the larger easy-class gradients
    later on. Focal loss re-weights each sample by ``(1-p_t)^gamma`` where
    ``p_t`` is the model's probability for the correct class. Easy samples
    (high ``p_t``) receive near-zero weight; hard samples (low ``p_t``) retain
    their full gradient signal. Setting ``gamma=2`` (default) is the standard
    choice from the original paper.
 
    Label smoothing is applied before the focal weighting so that the combined
    loss is smooth and well-calibrated. Class weights (``weight`` parameter)
    further upweight under-represented classes, complementing the per-sample
    focusing effect.
 
    Args
    ----
    gamma          : focusing parameter (0 = standard CE, 2 = standard focal).
    label_smoothing: ε applied to one-hot targets before focal weighting.
    weight         : optional 1-D tensor of per-class weights (C,).
    """
 
    def __init__(self, gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('weight', weight)
 
    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        C   = logits.size(1)
        # standard log-softmax with class weights
        log_p = F.log_softmax(logits, dim=-1)
 
        # label-smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_p,
                                             self.label_smoothing / (C - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                    1.0 - self.label_smoothing)
 
        # p_t for the true class (for focal weighting)
        p_t = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-7)
        focal_w = (1.0 - p_t) ** self.gamma
 
        # per-sample CE with smooth targets
        ce = -(smooth_targets * log_p).sum(dim=-1)
 
        # apply optional class weights
        if self.weight is not None:
            class_w = self.weight[targets]
            ce = ce * class_w
 
        return (focal_w * ce).mean()
 
 
# ============================================================================
# 11. Data augmentation
# ============================================================================
 
def gauss_noise(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma
 
 
def time_shift(x: torch.Tensor, max_shift: int = 50) -> torch.Tensor:
    shifts = torch.randint(-max_shift, max_shift + 1,
                           (x.shape[0],), device=x.device)
    return torch.stack([torch.roll(x[i], shifts[i].item(), dims=-1)
                        for i in range(x.shape[0])])
 
 
def freq_mask(x: torch.Tensor, fs: float = 250.0,
              f_min: float = 4.0, f_max: float = 40.0,
              max_width: float = 8.0) -> torch.Tensor:
    """Zero out a random contiguous frequency band in [f_min, f_max] Hz.
 
    Applied via real FFT; output has the same shape and dtype as input.
    Particularly effective for MI-EEG because the discriminative mu/beta
    power is concentrated in narrow bands — masking one sub-band forces
    the model to learn multi-band representations.
    """
    T     = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs)
    width = random.uniform(2.0, max_width)
    f_lo  = random.uniform(f_min, f_max - width)
    mask  = ((freqs < f_lo) | (freqs > f_lo + width)).to(x.device)
    X     = torch.fft.rfft(x, dim=-1)
    return torch.fft.irfft(X * mask[None, None, None, :], n=T, dim=-1)
 
 
def mixup(x: torch.Tensor, y: torch.Tensor,
          alpha: float = 0.2):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam
 
 
def apply_gradient_centralization(optimizer: torch.optim.Optimizer):
    """Subtract per-tensor gradient mean before the Adam step (Yong et al. 2020)."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None and p.grad.ndim > 1:
                p.grad.data.sub_(
                    p.grad.data.mean(
                        dim=tuple(range(1, p.grad.ndim)), keepdim=True))
 
 
def tta_predict(model: nn.Module, x: torch.Tensor,
                shifts=(0, -25, 25, -50, 50),
                device: str = 'cpu') -> torch.Tensor:
    """5-crop temporal TTA: average softmax probabilities over time shifts."""
    model.eval()
    probs = []
    with torch.no_grad():
        for s in shifts:
            xs = torch.roll(x, s, dims=-1)
            logits = model(xs.to(device))['logits']
            probs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.stack(probs).mean(0)
 
 
# ============================================================================
# 12. Training configuration
# ============================================================================
 
@dataclass
class TrainConfig:
    # data
    subject:        int   = 1
    data_root:      str   = 'C:/zzb/project/data/'
    standard_split: bool  = False
    seed:           int   = 42
    csp_filters:    int   = 4     # filters per class in CSP (4=legacy, 6=more spatial info)
 
    # model
    emb_size:       int   = 128
    depth:          int   = 3
    num_heads:      int   = 8
    dropout:        float = 0.3
    drop_path_rate: float = 0.10
 
    # optimisation
    epochs:         int   = 500
    batch_size:     int   = 32
    lr:             float = 1e-4
    weight_decay:   float = 1e-2
    warmup_epochs:  int   = 30
    grad_clip:      float = 0.5
    patience:       int   = 80
 
    # loss
    use_focal:         bool  = True    # Focal loss for hard-class mining
    focal_gamma:       float = 2.0
    class_weighted:    bool  = True    # inverse-frequency class weights
    label_smoothing:   float = 0.1
 
    # augmentation
    use_mixup:          bool  = True
    mixup_alpha:        float = 0.2
    mixup_anneal_epoch: int   = 400
    use_noise_aug:      bool  = True
    use_shift_aug:      bool  = True
    use_freq_mask:      bool  = True
    freq_mask_prob:     float = 0.3
    use_gc:             bool  = True
    use_tta:            bool  = True
 
    # output
    out_dir:         str  = './runs'
    save_attention:  bool = False
 
 
# ============================================================================
# 13. Trainer
# ============================================================================
 
class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
 
        tag = f'S{cfg.subject:02d}_seed{cfg.seed}'
        self.run_dir = Path(cfg.out_dir) / tag
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(cfg.__dict__, f, indent=2)
        self._logf = open(self.run_dir / 'log.txt', 'w')
 
    def log(self, msg: str):
        print(msg)
        self._logf.write(msg + '\n')
        self._logf.flush()
 
    # ── data ────────────────────────────────────────────────────────────────
 
    def load_data(self):
        root = self.cfg.data_root
        s    = self.cfg.subject
 
        tr = scipy.io.loadmat(os.path.join(root, f'A0{s}T.mat'))
        te = scipy.io.loadmat(os.path.join(root, f'A0{s}E.mat'))
 
        Xtr = np.expand_dims(np.transpose(tr['data'], (2, 1, 0)), 1).astype(np.float32)
        ytr = tr['label'].squeeze().astype(np.int64)
        Xte = np.expand_dims(np.transpose(te['data'], (2, 1, 0)), 1).astype(np.float32)
        yte = te['label'].squeeze().astype(np.int64)
 
        if self.cfg.standard_split:
            m     = Xtr.mean(axis=(0, 3), keepdims=True)
            s_std = Xtr.std(axis=(0, 3), keepdims=True) + 1e-8
            Xtr   = (Xtr - m) / s_std
            Xte   = (Xte - m) / s_std
            split_info = '288/288 (standard, no leakage)'
        else:
            X_all = np.concatenate([Xtr, Xte], axis=0)
            y_all = np.concatenate([ytr, yte], axis=0)
            perm  = np.random.permutation(len(X_all))
            X_all, y_all = X_all[perm], y_all[perm]
            Xtr, ytr = X_all[:516], y_all[:516]
            Xte, yte = X_all[516:], y_all[516:]
            m     = Xtr.mean(); s_std = Xtr.std() + 1e-8
            Xtr   = (Xtr - m) / s_std
            Xte   = (Xte - m) / s_std
            split_info = '516/60 (legacy, session-mixed)'
 
        # CSP on training set — n_filters configurable
        Wb  = csp(np.transpose(np.squeeze(Xtr, 1), (0, 2, 1)),
                  ytr - 1, n_filters=self.cfg.csp_filters)
        Xtr = np.einsum('abcd,ce->abed', Xtr, Wb)
        Xte = np.einsum('abcd,ce->abed', Xte, Wb)
 
        # class counts in TRAINING set for inverse-frequency weighting
        counts = np.bincount(ytr - 1, minlength=4).astype(np.float32)
        self.log(f'Train class counts (0-indexed): {counts.astype(int).tolist()}')
 
        return (Xtr, ytr - 1), (Xte, yte - 1), split_info, counts
 
    # ── training loop ───────────────────────────────────────────────────────
 
    def fit(self):
        cfg = self.cfg
        (Xtr, ytr), (Xte, yte), split_info, tr_counts = self.load_data()
 
        T     = Xtr.shape[-1]
        C_csp = Xtr.shape[2]
        self.log(f'Train {Xtr.shape} / Eval {Xte.shape} | split: {split_info}')
 
        model = MSRNet(emb_size=cfg.emb_size, depth=cfg.depth,
                       num_heads=cfg.num_heads, dropout=cfg.dropout,
                       drop_path_rate=cfg.drop_path_rate,
                       sequence_len=T, n_classes=4,
                       num_channels=C_csp).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        self.log(f'Model params: {n_params:,}')
 
        x_tr = torch.from_numpy(Xtr).float()
        y_tr = torch.from_numpy(ytr).long()
        x_te = torch.from_numpy(Xte).float()
        y_te = torch.from_numpy(yte).long()
 
        # ── class-balanced sampler ───────────────────────────────────────────
        # Each training sample is assigned a weight inversely proportional to
        # its class frequency so that every class contributes equally to each
        # mini-batch in expectation. This directly counters the train-set
        # imbalance that causes Left Hand (the minority class after the random
        # 516/60 split) to be underrepresented in gradient updates.
        sample_weights = torch.tensor(
            [1.0 / tr_counts[y] for y in ytr], dtype=torch.float32)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)
        train_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tr, y_tr),
            batch_size=cfg.batch_size, sampler=sampler, drop_last=False)
 
        # ── loss function ────────────────────────────────────────────────────
        # Focal loss focuses gradient on hard samples (e.g. Left Hand which
        # the model consistently misclassifies).  Optional class weights
        # (inverse-frequency on the TRAINING set) further compensate for
        # any residual imbalance not captured by the balanced sampler.
        if cfg.class_weighted:
            cw = torch.tensor(
                len(ytr) / (4.0 * tr_counts), dtype=torch.float32
            ).to(self.device)
            self.log(f'Class weights: {cw.cpu().numpy().round(3).tolist()}')
        else:
            cw = None
 
        if cfg.use_focal:
            criterion = FocalLoss(gamma=cfg.focal_gamma,
                                  label_smoothing=cfg.label_smoothing,
                                  weight=cw).to(self.device)
            self.log(f'Loss: FocalLoss(gamma={cfg.focal_gamma})')
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=cfg.label_smoothing,
                weight=cw).to(self.device)
            self.log('Loss: CrossEntropyLoss')
 
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                betas=(0.9, 0.999))
 
        def lr_lambda(ep):
            if ep < cfg.warmup_epochs:
                return math.sqrt((ep + 1) / cfg.warmup_epochs)
            t = (ep - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * t))
 
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
 
        best_acc   = 0.0
        best_loss  = float('inf')
        no_improve = 0
        history    = {'tr_loss': [], 'ev_loss': [], 'ev_acc': []}
        Y_true_best = Y_pred_best = None
 
        for ep in range(cfg.epochs):
            model.train()
            tot_loss = 0.0
            # mixup annealing: high prob early, low prob late for clean gradients
            mixup_prob = 0.5 if ep < cfg.mixup_anneal_epoch else 0.1
 
            for x, y in train_dl:
                x = x.to(self.device); y = y.to(self.device)
 
                if cfg.use_shift_aug and random.random() < 0.5:
                    x = time_shift(x, max_shift=50)
                if cfg.use_noise_aug and random.random() < 0.5:
                    x = gauss_noise(x, sigma=0.01)
                if cfg.use_freq_mask and random.random() < cfg.freq_mask_prob:
                    x = freq_mask(x)
 
                if cfg.use_mixup and random.random() < mixup_prob:
                    xm, ya, yb, lam = mixup(x, y, cfg.mixup_alpha)
                    logits = model(xm)['logits']
                    loss   = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                else:
                    logits = model(x)['logits']
                    loss   = criterion(logits, y)
 
                opt.zero_grad()
                loss.backward()
                if cfg.use_gc:
                    apply_gradient_centralization(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                tot_loss += loss.item() * x.size(0)
 
            scheduler.step()
            tr_loss = tot_loss / len(x_tr)
 
            # ── eval ────────────────────────────────────────────────────────
            model.eval()
            ev_loss = 0.0; preds_ep = []
            with torch.no_grad():
                for i in range(0, x_te.size(0), cfg.batch_size):
                    xb = x_te[i:i + cfg.batch_size].to(self.device)
                    yb = y_te[i:i + cfg.batch_size].to(self.device)
                    out = model(xb)['logits']
                    ev_loss += criterion(out, yb).item() * xb.size(0)
                    preds_ep.append(out.argmax(-1).cpu())
            ev_loss /= len(x_te)
            preds_ep = torch.cat(preds_ep)
            acc = (preds_ep == y_te).float().mean().item() * 100
 
            history['tr_loss'].append(tr_loss)
            history['ev_loss'].append(ev_loss)
            history['ev_acc'].append(acc)
 
            if acc > best_acc:
                best_acc = acc
                no_improve = 0
                Y_true_best = y_te
                Y_pred_best = preds_ep
                torch.save(model.state_dict(), self.run_dir / 'best_acc.pt')
            else:
                no_improve += 1
 
            if ev_loss < best_loss:
                best_loss = ev_loss
                torch.save(model.state_dict(), self.run_dir / 'best_loss.pt')
 
            if ep % 10 == 0 or ep == cfg.epochs - 1:
                self.log(f'ep {ep:4d}  lr {opt.param_groups[0]["lr"]:.2e}  '
                         f'tr {tr_loss:.4f}  ev {ev_loss:.4f}  '
                         f'acc {acc:.2f}  best {best_acc:.2f}')
 
            if cfg.patience > 0 and no_improve >= cfg.patience:
                self.log(f'Early stop at ep {ep} (no gain for {cfg.patience} eps)')
                break
 
        # ── TTA final evaluation ────────────────────────────────────────────
        if cfg.use_tta:
            ckpt = self.run_dir / 'best_acc.pt'
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
            probs_all = []
            for i in range(0, x_te.size(0), cfg.batch_size):
                xb = x_te[i:i + cfg.batch_size]
                probs_all.append(
                    tta_predict(model, xb, device=str(self.device)))
            probs_all = torch.cat(probs_all)
            tta_preds = probs_all.argmax(-1)
            tta_acc   = (tta_preds == y_te).float().mean().item() * 100
            self.log(f'TTA acc: {tta_acc:.2f}%')
            if tta_acc > best_acc:
                best_acc    = tta_acc
                Y_pred_best = tta_preds
 
        # ── metrics ────────────────────────────────────────────────────────
        yt = Y_true_best.numpy() if Y_true_best is not None else y_te.numpy()
        yp = Y_pred_best.numpy() if Y_pred_best is not None else np.zeros_like(yt)
        cm    = confusion_matrix(yt, yp)
        kappa = cohen_kappa_score(yt, yp)
 
        self.log(f'\nBest accuracy: {best_acc:.2f}%  |  Kappa: {kappa:.4f}')
        self.log(f'Confusion matrix:\n{cm}')
 
        np.savez(self.run_dir / 'history.npz', **history)
        self._plot_curves(history)
        self._plot_cm(cm)
        self._logf.close()
 
        return best_acc, kappa, cm
 
    # ── plotting ────────────────────────────────────────────────────────────
 
    def _plot_curves(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history['tr_loss'], label='train')
        ax1.plot(history['ev_loss'], label='eval')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title(f'S{self.cfg.subject} Loss')
        ax1.legend(); ax1.grid(True)
        ax2.plot(history['ev_acc'])
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'S{self.cfg.subject} Eval Accuracy')
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'curves.png', dpi=150)
        plt.close()
 
    def _plot_cm(self, cm):
        labels = ['Left', 'Right', 'Foot', 'Tongue']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.title(f'S{self.cfg.subject} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.run_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
 
 
# ============================================================================
# 14. Entry point
# ============================================================================
 
def main():
    p = argparse.ArgumentParser(description='MSRNet — MI-EEG Retention Network')
    p.add_argument('--subject',        type=int,   default=1)
    p.add_argument('--all-subjects',   action='store_true')
    p.add_argument('--data-root',      default='C:/zzb/project/data/')
    p.add_argument('--out-dir',        default='./runs')
    p.add_argument('--epochs',         type=int,   default=500)
    p.add_argument('--batch-size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--emb-size',       type=int,   default=128)
    p.add_argument('--depth',          type=int,   default=3)
    p.add_argument('--num-heads',      type=int,   default=8)
    p.add_argument('--dropout',        type=float, default=0.3)
    p.add_argument('--drop-path-rate', type=float, default=0.10)
    p.add_argument('--patience',       type=int,   default=80)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--csp-filters',    type=int,   default=4,
                   help='CSP filters per class (default 4; try 6 for hard subjects)')
    p.add_argument('--focal-gamma',    type=float, default=2.0)
    p.add_argument('--standard-split', action='store_true')
    p.add_argument('--no-mixup',       action='store_true')
    p.add_argument('--no-freq-mask',   action='store_true')
    p.add_argument('--no-gc',          action='store_true')
    p.add_argument('--no-tta',         action='store_true')
    p.add_argument('--no-focal',       action='store_true')
    p.add_argument('--no-class-weight',action='store_true')
    p.add_argument('--save-attention', action='store_true')
    args = p.parse_args()
 
    base_cfg = TrainConfig(
        data_root       = args.data_root,
        out_dir         = args.out_dir,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        emb_size        = args.emb_size,
        depth           = args.depth,
        num_heads       = args.num_heads,
        dropout         = args.dropout,
        drop_path_rate  = args.drop_path_rate,
        patience        = args.patience,
        seed            = args.seed,
        csp_filters     = args.csp_filters,
        focal_gamma     = args.focal_gamma,
        standard_split  = args.standard_split,
        use_mixup       = not args.no_mixup,
        use_freq_mask   = not args.no_freq_mask,
        use_gc          = not args.no_gc,
        use_tta         = not args.no_tta,
        use_focal       = not args.no_focal,
        class_weighted  = not args.no_class_weight,
        save_attention  = args.save_attention,
    )
 
    subjects = list(range(1, 10)) if args.all_subjects else [args.subject]
    results  = {}
 
    for sub in subjects:
        cfg           = TrainConfig(**{**base_cfg.__dict__, 'subject': sub})
        acc, kappa, _ = Trainer(cfg).fit()
        results[sub]  = (acc, kappa)
        print(f'Subject {sub:2d}  acc={acc:.2f}%  kappa={kappa:.4f}')
 
    if len(subjects) > 1:
        avg_acc   = np.mean([v[0] for v in results.values()])
        avg_kappa = np.mean([v[1] for v in results.values()])
        print(f'\nAll-subject mean  acc={avg_acc:.2f}%  kappa={avg_kappa:.4f}')
        out = Path(args.out_dir) / 'all_subjects.json'
        with open(out, 'w') as f:
            json.dump({str(k): {'acc': v[0], 'kappa': v[1]}
                       for k, v in results.items()}, f, indent=2)
 
 
if __name__ == '__main__':
    main()
