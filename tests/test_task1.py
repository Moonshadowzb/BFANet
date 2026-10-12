"""Task-1 checks: BFA module vs. bfanet_iop_v3.tex eqs. (2)-(6) and sec. 3.4."""
import math, sys, numpy as np, torch
sys.path.insert(0, '.')
from bfanet import montage as mt, data as D
from bfanet.model import BFAConfig, BFANet, BFAttention, TemporalDecay, DiscreteSpatialPrior

torch.manual_seed(0)
ok = lambda c, m: print(('PASS ' if c else 'FAIL ') + m) or c
res = []

# ---- montage / discrete prior -------------------------------------------
ch = mt.CHANNELS_2A; i = {c: k for k, c in enumerate(ch)}
A = mt.base_affinity(ch)
res.append(ok(np.allclose(A, A.T), 'A~ symmetric'))
res.append(ok(A[i['C3'], i['C4']] == 0.7 and A[i['FC3'], i['CP4']] == 0.7, 'contralateral motor pairs = 0.7'))
res.append(ok(A[i['C3'], i['CP3']] == 1.5 and A[i['C1'], i['CP2']] == 1.5, 'same-region pairs = 1.5'))
res.append(ok(np.all(A[i['C5']] == 1.0) and np.all(A[:, i['POz']] == 1.0), 'unlisted electrodes -> identity'))
P = mt.unit_sphere_positions(ch)
res.append(ok(np.allclose(np.linalg.norm(P, axis=1), 1), 'positions on unit sphere'))
res.append(ok(abs(P[i['Cz'], 1]) < 0.02 and abs(P[i['C3'], 1]) < 0.02, f'central plane y~0 (Cz y={P[i["Cz"],1]:.3f}, C3 y={P[i["C3"],1]:.3f})'))
res.append(ok(P[i['C4'], 0] > 0 > P[i['C3'], 0] and P[i['Fz'], 1] > 0 and P[i['Cz'], 2] > 0.95, 'axes: x->right, y->nasion, z->vertex'))

# ---- tokenizer: channel identity and pi maps ------------------------------
cfg = BFAConfig(); m = BFANet(cfg).eval()
Tp, C = m.tok.Tp, m.tok.C
res.append(ok((Tp, C, C * Tp) == (35, 22, 770), f'T=875, P=25 -> T_p={Tp}, N={C*Tp}'))
u = torch.arange(C * Tp)
res.append(ok(torch.equal(m.tok.pi_c, u // Tp) and torch.equal(m.tok.pi_t, u % Tp), 'pi_c(u)=u//T_p, pi_t(u)=u%T_p'))
with torch.no_grad():
    m.tok.chan_embed.zero_(); m.tok.patch.bias.zero_()
    x = torch.zeros(2, C, 875); x[:, 7] = torch.randn(2, 875)       # only C3 active
    z = m.tok(x).abs().sum(-1)                                      # (B, N)
    active = z.reshape(2, C, Tp).sum(-1) > 1e-6
res.append(ok(active[:, 7].all() and active.sum() == 2, 'tokens of channel c depend only on channel c'))

# ---- decay factors at initialisation ---------------------------------------
m = BFANet(BFAConfig()).eval()
sp, td = m.spatial[0], m.temporal[0]
res.append(ok(abs(sp.alpha.item() - 1) < 1e-6, 'alpha initialised to 1 (softplus)'))
res.append(ok(torch.allclose(sp.log_affinity(), torch.tensor(np.log(A), dtype=torch.float32)), 'A = A~ exactly at w=0 (exp parameterisation)'))
res.append(ok(torch.allclose(td.gamma, torch.linspace(0.85, 0.99, 8), atol=1e-5), f'gamma_t init = linspace(0.85,0.99): {td.gamma.detach().numpy().round(3)}'))
n_prior = sum(p.numel() for p in list(sp.parameters()) + list(td.parameters()))
res.append(ok(n_prior == 6 + 8, f'prior scalars per layer = 6+H = {n_prior}'))
lit = DiscreteSpatialPrior(ch, affinity_param='sigmoid')
la = lit.log_affinity().detach()
res.append(ok(abs(math.exp(la[i['C3'], i['CP3']]) - 1.5 * 0.25) < 1e-5 and abs(math.exp(la[i['C5'], i['C6']]) - 1) < 1e-6,
              'literal sigmoid form: listed pairs x0.25, unlisted x1 at init (documented flaw)'))

# ---- lifting (eq. 3) -------------------------------------------------------
with torch.no_grad():
    for p in m.prior_parameters(): p.add_(0.1 * torch.randn_like(p))
    L = m.log_decay(0); S = m.spatial[0].log_S(); R = m.temporal[0].log_rho(Tp, m.patch_sec)
    uu, vv = torch.randint(0, C * Tp, (2, 500))
    ref = R[:, uu % Tp, vv % Tp] + S[uu // Tp, vv // Tp]
res.append(ok(torch.allclose(L[:, uu, vv], ref, atol=1e-5), 'log D[u,v] = log rho(t_u,t_v) + log g_geo(c_u,c_v) + log A(c_u,c_v)'))
mcls = BFANet(BFAConfig(pool='cls'))
Lc = mcls.log_decay(0)
res.append(ok(Lc.shape[-1] == 771 and torch.all(Lc[:, 0, :] == 0) and torch.all(Lc[:, :, 0] == 0), "pool='cls': CLS row/column D = 1"))
same_t = (u[:, None] % Tp) == (u[None, :] % Tp)
res.append(ok(torch.allclose(R[:, u % Tp][:, :, u % Tp][:, same_t], torch.zeros(1)), 'same time index -> rho = 1'))

# ---- attention semantics ---------------------------------------------------
att = BFAttention(64, 8, 0.0, 'logadd').eval(); x = torch.randn(3, 50, 64)
Bz = torch.zeros(8, 50, 50); logD = -torch.rand(8, 50, 50) * 3
with torch.no_grad():
    o1 = att(x, None, Bz); o0 = att(x, torch.zeros(8, 50, 50), Bz)
    o_sdpa = att(x, logD, Bz); o_man, w = att(x, logD, Bz, return_attn=True)
    q, k, v = att.qkv(x).view(3, 50, 3, 8, 8).permute(2, 0, 3, 1, 4)
    e = torch.exp((q @ k.transpose(-2, -1)) / math.sqrt(8)) * torch.exp(logD)
    w_ref = e / e.sum(-1, keepdim=True)
res.append(ok(torch.allclose(o1, o0, atol=1e-6), 'D = 1 recovers vanilla attention'))
res.append(ok(torch.allclose(o_sdpa, o_man, atol=1e-5), 'fused SDPA path == explicit path'))
res.append(ok(torch.allclose(w, w_ref, atol=1e-6), 'logadd == D multiplies exp(logits), then normalise'))

# causal: logadd masks the future, literal mul_logits does not
tdc = TemporalDecay(8, causal=True); Rc = tdc.log_rho(10, 0.1)
lD = Rc.repeat_interleave(1, 0)
x = torch.randn(2, 10, 64)
a_add = BFAttention(64, 8, 0.0, 'logadd').eval(); a_mul = BFAttention(64, 8, 0.0, 'mul_logits').eval()
a_mul.load_state_dict(a_add.state_dict())
with torch.no_grad():
    _, w_add = a_add(x, lD, torch.zeros(8, 10, 10), True)
    _, w_mul = a_mul(x, lD, torch.zeros(8, 10, 10), True)
fut = torch.triu(torch.ones(10, 10, dtype=torch.bool), 1)
res.append(ok(w_add[..., fut].abs().max() < 1e-7, 'causal logadd: zero weight on future tokens'))
res.append(ok(w_mul[..., fut].sum(-1).mean() > 0.1, f'literal causal mul_logits: future still receives {w_mul[..., fut].sum()/w_mul.sum():.0%} of attention mass (eq. 6 does not mask)'))

# ---- gradients flow into every prior parameter ----------------------------
m = BFANet(BFAConfig()).train()
loss = torch.nn.functional.cross_entropy(m(torch.randn(4, 22, 875)), torch.tensor([0, 1, 2, 3]))
loss.backward()
g = {n: p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters()
     if n.startswith(('spatial', 'temporal'))}
res.append(ok(all(g.values()), f'gradients reach alpha, w_r, gamma_t in all layers ({len(g)} tensors)'))

# ---- CSP mode disables the spatial prior ----------------------------------
mc = BFANet(BFAConfig(channels=[f'CSP{k}' for k in range(16)], spatial_prior=False))
res.append(ok(mc.spatial is None and mc.log_decay(0) is not None, 'CSP input: spatial factors off, temporal kept'))

# ---- vanilla arm identical except the prior -------------------------------
mv = BFANet(BFAConfig(prior='vanilla')); mb = BFANet(BFAConfig())
nv, nb = sum(p.numel() for p in mv.parameters()), sum(p.numel() for p in mb.parameters())
res.append(ok(nb - nv == 4 * (6 + 8), f'params: vanilla {nv:,} / BFA {nb:,} (diff = L(6+H) = {nb-nv})'))

# ---- CSP implementation ---------------------------------------------------
rng = np.random.default_rng(0)
Xs = rng.standard_normal((200, 6, 300)); ys = np.repeat(np.arange(4), 50)
for c in range(4): Xs[ys == c, c] *= 3.0                       # class c: channel c has high variance
W = D.fit_csp_ovr(Xs, ys, 4)
res.append(ok(W.shape == (6, 16) and all(np.argmax(np.abs(W[:, 4 * c])) == c for c in range(4)),
              'CSP: first filter of class c peaks on its discriminative channel'))

# ---- nested budgets ---------------------------------------------------------
yb = np.repeat(np.arange(4), 72)
S = D.nested_budget_indices(yb, [8, 16, 32, 64, 128, 288], seed=3)
nested = all(set(S[a]) <= set(S[b]) for a, b in zip([8, 16, 32, 64, 128], [16, 32, 64, 128, 288]))
bal = all(np.all(np.bincount(yb[S[n]]) == n // 4) for n in S)
res.append(ok(nested and bal, 'budgets nested and class-balanced'))

# CLS readout starves the last layer's prior of gradient; mean pooling does not
mc2 = BFANet(BFAConfig(pool='cls')).train()
torch.nn.functional.cross_entropy(mc2(torch.randn(4, 22, 875)), torch.tensor([0, 1, 2, 3])).backward()
res.append(ok(mc2.temporal[3].theta.grad.abs().sum() == 0, "pool='cls': last-layer prior gets zero gradient (why mean pooling is default)"))
print(f'\n{sum(res)}/{len(res)} checks passed')
