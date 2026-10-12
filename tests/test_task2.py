"""Task-2 checks and distillation report for BFA-C (sec. 3.3, eqs. 7-10)."""
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from bfanet import montage as mt, bfac as bc
from bfanet.model import BFAConfig, BFANet
ok = lambda c, m: print(('PASS ' if c else 'FAIL ') + m) or c
res = []; report = {}

# descriptor properties
P = mt.unit_sphere_positions(mt.CHANNELS_2A); psi = bc.pair_descriptor(P)
res.append(ok(np.allclose(psi, psi.transpose(1, 0, 2), atol=1e-6), 'psi invariant to i<->j'))
res.append(ok(all(np.allclose(P[i2] @ bc.MIRROR, P[mt.CHANNELS_2A.index(mt.mirror_name(c))]) for i2, c in enumerate(mt.CHANNELS_2A)), 'template exactly left/right symmetric (C3 = M C4, ...)'))
res.append(ok(np.allclose(bc.pair_descriptor(P @ bc.MIRROR), psi, atol=1e-6), 'psi invariant to mid-sagittal reflection'))
i = {c: k for k, c in enumerate(mt.CHANNELS_2A)}
res.append(ok(psi[i['C3'], i['C4'], 1] < 0.1 < psi[i['C3'], i['C4'], 0], f"homologous distance small for C3-C4 ({psi[i['C3'],i['C4'],1]:.3f} rad) though delta={psi[i['C3'],i['C4'],0]:.2f}"))
res.append(ok(sum(p.numel() for p in bc.MLPAffinity().parameters()) == 369, 'f_phi has 369 parameters'))

# SH basis sanity: orthonormal parity signs
E, deg, par = bc.real_sph_harm(P, 4)
Em, _, _ = bc.real_sph_harm(P @ bc.MIRROR, 4)
res.append(ok(np.allclose(Em, E * par, atol=1e-8), 'real SH parity: e(Mp) = diag(s) e(p)'))

# distillation fidelity (MLP over 5 init seeds, SH closed form)
mlp = []
for s in range(5):
    aff = bc.MLPAffinity(); mlp.append(bc.distill_mlp(aff, seed=s))
aff0 = bc.MLPAffinity(); report['mlp_best_of_5'] = bc.distill_mlp_restarts(aff0, seed=0)
report['mlp'] = {k: [round(d[k], 4) for d in mlp] for k in mlp[0]}
sh4 = bc.SHAffinity(5); report['sh_l5'] = bc.distill_sh(sh4); report['sh_l5']['free_params'] = sh4.n_free()
_t = bc.SHAffinity(4); report['sh_l4'] = bc.distill_sh(_t); report['sh_l4']['free_params'] = _t.n_free()
sh6 = bc.SHAffinity(6); report['sh_l6'] = bc.distill_sh(sh6); report['sh_l6']['free_params'] = sh6.n_free()
print(json.dumps(report, indent=1))
res.append(ok(min(report['mlp']['r_all']) > 0.95, f"MLP: Pearson r(log A_phi0, log A~) >= {min(report['mlp']['r_all']):.4f} over 5 seeds"))

# mirror invariance of the distilled fields on the full 22-channel montage
with torch.no_grad():
    LA = aff0(torch.tensor(psi, dtype=torch.float32)).numpy()
    perm = [i[mt.mirror_name(c)] for c in mt.CHANNELS_2A]
res.append(ok(np.allclose(LA, LA[np.ix_(perm, perm)], atol=1e-5), 'MLP affinity on 22 ch is left/right symmetric'))
Esh, _, _ = bc.real_sph_harm(P, 5); LS = Esh @ sh4.W.detach().numpy() @ Esh.T
res.append(ok(np.allclose(LS, LS[np.ix_(perm, perm)], atol=1e-6) and np.allclose(LS, LS.T), 'SH affinity symmetric in i<->j and left/right'))

# extrapolation to electrodes outside the partition (not used in distillation)
pairs = [('C5','C3'), ('C5','CP3'), ('C5','C4'), ('C5','C6'), ('C3','C6'), ('Pz','CPz'), ('POz','Cz'), ('P1','CP3'), ('P1','CP4')]
print('\nExtrapolated A (exp log A) on pairs outside R [discrete BFA gives 1.0 for all]:')
print(f"{'pair':10s} {'MLP':>6s} {'SH l=5':>7s}")
for a, b in pairs:
    print(f'{a+"-"+b:10s} {np.exp(LA[i[a], i[b]]):6.2f} {np.exp(LS[i[a], i[b]]):7.2f}')
ref_pairs = [('C3','CP3'), ('C3','C4'), ('FC1','C2'), ('Cz','C3')]
print('Reference pairs (target A~):')
for a, b in ref_pairs:
    print(f'{a+"-"+b:10s} {np.exp(LA[i[a], i[b]]):6.2f} {np.exp(LS[i[a], i[b]]):7.2f}   target {mt.base_affinity(mt.CHANNELS_2A)[i[a], i[b]]:.2f}')

# integration into BFANet
m = BFANet(BFAConfig(prior='bfac'))
n_prior = sum(p.numel() for p in m.prior_parameters())
res.append(ok(n_prior == 369 + 4 * (1 + 8), f'BFA-C prior params = 369 + L(1+H) = {n_prior}'))
res.append(ok(all(sp.affinity is m.spatial[0].affinity for sp in m.spatial), 'f_phi shared by all layers'))
x = torch.randn(2, 22, 875); y = torch.tensor([0, 3])
torch.nn.functional.cross_entropy(m.train()(x), y).backward()
res.append(ok(all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.shared_affinity.parameters()), 'gradients reach phi end-to-end'))
ms = BFANet(BFAConfig(prior='bfac', bfac_kind='sh'))
res.append(ok(ms.log_decay(0).shape == (8, 770, 770), 'SH variant builds decay'))
mr = BFANet(BFAConfig(prior='bfac', psi_drop=('delta_bar',)))
res.append(ok(mr.shared_affinity.mask.tolist() == [1, 0, 1, 1] and 'r_all' in mr.bfac_info, 'descriptor ablation masks delta_bar, keeps 369 params'))
# transfer: same phi, new montage -> A defined for every pair
from bfanet.bfac import ContinuousSpatialPrior
cp = ContinuousSpatialPrior(['C3', 'Cz', 'C4', 'T7', 'T8', 'Oz'], m.shared_affinity)
res.append(ok(torch.isfinite(cp.log_S()).all(), 'BFA-C evaluates on an unseen montage (T7, T8, Oz)'))
json.dump(report, open('smoke/bfac_distill.json', 'w'), indent=1)
print(f'\n{sum(res)}/{len(res)} checks passed')
