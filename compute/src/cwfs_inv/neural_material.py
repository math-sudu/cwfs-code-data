"""Objective CWFS neural material for the existing plane-strain FE interface.

The learned quantity is the increment of accumulated plastic shear used to
evaluate the evolving strength. A differentiable, fixed-strength MC layer
returns stress on a face, either edge, or the apex. Plastic strain and history
are recovered from that SAME stress. No CWFS root solver is called at inference.
The tangent differentiates the actual neural stress map, including its strength
prediction, and the spectral rotation back to element coordinates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fe_materials import (Material, MaterialUpdate, elastic_matrix_4,
                           _principal_inplane, _rotation_matrices)


def strengths(gamma, theta):
    c = theta[:, 0] + (theta[:, 1]-theta[:, 0])*(gamma/theta[:, 4]).clamp(0, 1)
    phi = torch.deg2rad(theta[:, 2] + (theta[:, 3]-theta[:, 2])*
                        (gamma/theta[:, 5]).clamp(0, 1))
    return c, phi


def principal_span(values):
    """Exact max-minus-min, with an isotropic subgradient at repeated values."""
    hi, lo = values.amax(1), values.amin(1)
    tol = 1e-10*values.abs().amax(1).clamp_min(1e-12)
    high = ((values-hi[:, None]).abs() <= tol[:, None]).to(values.dtype)
    low = ((values-lo[:, None]).abs() <= tol[:, None]).to(values.dtype)
    weights = high/high.sum(1, keepdim=True)-low/low.sum(1, keepdim=True)
    return (hi-lo).detach() + (weights*(values-values.detach())).sum(1)


def fixed_strength_return(trial, c, phi, C, psi_deg):
    """Batched closed MC face/edge/apex map; trial rows are sorted descending."""
    sp = torch.sin(phi)
    N = (1+sp)/(1-sp)
    sc = 2*c*torch.cos(phi)/(1-sp)
    ss = np.sin(np.deg2rad(psi_deg))
    Np = (1+ss)/(1-ss)
    b = trial.new_tensor([[1, 0, -Np], [0, 1, -Np], [1, -Np, 0]])
    r = b @ C
    z = torch.zeros_like(N)
    a = torch.stack((torch.stack((torch.ones_like(N), z, -N), 1),
                     torch.stack((z, torch.ones_like(N), -N), 1),
                     torch.stack((torch.ones_like(N), -N, z), 1)), 1)
    f = (a*trial[:, None, :]).sum(2)-sc[:, None]
    dl = f[:, 0].clamp_min(0)/(a[:, 0]*r[0]).sum(1)
    out = trial-dl[:, None]*r[0]
    v12, v23 = out[:, 1] > out[:, 0], out[:, 2] > out[:, 1]
    apex = v12 & v23
    region = torch.where(f[:, 0] > 0, 1, 0)
    for mask, pair, rid in ((v12 & ~apex, (0, 1), 2), (v23 & ~apex, (0, 2), 3)):
        ae, re = a[:, pair, :], r[list(pair)]
        mat = ae @ re.T
        rhs = f[:, pair]
        det = mat[:, 0, 0]*mat[:, 1, 1]-mat[:, 0, 1]*mat[:, 1, 0]
        multipliers = torch.stack(((rhs[:, 0]*mat[:, 1, 1]-rhs[:, 1]*mat[:, 0, 1])/det,
                                  (rhs[:, 1]*mat[:, 0, 0]-rhs[:, 0]*mat[:, 1, 0])/det), 1)
        edge = trial-multipliers @ re
        out = torch.where(mask[:, None], edge, out)
        region = torch.where(mask, rid, region)
        bad = edge[:, 2] > edge[:, 0] if rid == 2 else edge[:, 1] > edge[:, 0]
        apex = apex | (mask & bad)
    out = torch.where(apex[:, None], (-c/torch.tan(phi))[:, None], out)
    region = torch.where(apex, 4, region)
    return out, region


class PrincipalCWFS(nn.Module):
    """Learn evolving strength; enforce tensor decomposition and MC region flow."""

    def __init__(self, lo, hi, E=560., nu=.30, psi_deg=5., hidden=(128, 128, 128)):
        super().__init__()
        self.E, self.nu, self.psi_deg = float(E), float(nu), float(psi_deg)
        self.hidden = tuple(hidden)
        self.register_buffer('C', torch.tensor(elastic_matrix_4(E, nu)[:3, :3], dtype=torch.float32))
        self.register_buffer('Cinv', torch.linalg.inv(self.C))
        self.register_buffer('lo', torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer('hi', torch.as_tensor(hi, dtype=torch.float32))
        layers = []
        dim = 12
        for width in hidden:
            layers += [nn.Linear(dim, width), nn.SiLU()]
            dim = width
        layers += [nn.Linear(dim, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, trial, gamma_old, theta):
        c0, phi0 = strengths(gamma_old, theta)
        base, _ = fixed_strength_return(trial, c0, phi0, self.C, self.psi_deg)
        dp0 = (trial-base) @ self.Cinv.T
        dg0 = principal_span(dp0).clamp_min(0)
        scale = theta[:, 4:6].amax(1)
        mean = trial.mean(1)
        dev = trial-mean[:, None]
        q = torch.sqrt(1.5*dev.square().sum(1).clamp_min(1e-24))
        lode = 13.5*dev.prod(1)/q.pow(3)
        invariants = torch.stack((mean/30, q/30, lode), 1)
        features = torch.cat((invariants,
                              (gamma_old[:, None]/theta[:, 4:6]).clamp(0, 1),
                              torch.log((dg0/scale).clamp_min(1e-12))[:, None]/6,
                              2*(theta-self.lo)/(self.hi-self.lo)-1), 1)
        log_ratio = self.net(features).squeeze(1)
        guess = gamma_old + dg0*torch.exp(log_ratio)
        c, phi = strengths(guess, theta)
        proposed, region = fixed_strength_return(trial, c, phi, self.C, self.psi_deg)
        f0 = trial[:, 0]*(1-torch.sin(phi0))-trial[:, 2]*(1+torch.sin(phi0))-2*c0*torch.cos(phi0)
        active = f0 > 0
        stress = torch.where(active[:, None], proposed, trial)
        dp = (trial-stress) @ self.Cinv.T
        gamma = gamma_old + principal_span(dp)
        region = torch.where(active, region, 0)
        return stress, dp, gamma, region, log_ratio, dg0


def load_principal_checkpoint(path, E, nu, psi_deg, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved['format'] != 'principal_cwfs_v1':
        raise ValueError('A general principal-state checkpoint is required')
    requested = np.array([E, nu, psi_deg], float)
    stored = np.array([saved['material'][k] for k in ('E', 'nu', 'psi_deg')])
    if not np.allclose(requested, stored, rtol=0, atol=1e-12):
        raise ValueError('Checkpoint and FE elastic/dilatancy parameters differ')
    model = PrincipalCWFS(saved['lo'], saved['hi'], **saved['material'], hidden=saved['hidden'])
    model.load_state_dict(saved['state_dict'])
    if not np.allclose(model.C.detach().numpy(), elastic_matrix_4(E, nu)[:3, :3], rtol=1e-6):
        raise ValueError('Checkpoint elastic operator differs from material metadata')
    model = model.to(device=device, dtype=torch.float64).eval()
    # Reassemble the SAME material in FE precision; checkpoint training is float32.
    with torch.no_grad():
        model.C.copy_(torch.as_tensor(elastic_matrix_4(E, nu)[:3, :3],dtype=model.C.dtype,device=device))
        model.Cinv.copy_(torch.linalg.inv(model.C))
    return model, saved


class NeuralCWFSMaterial(Material):
    """Batched neural stress/history update, with an actual-map consistent tangent."""
    name = 'neural_cwfs'
    n_state = 3
    state_names = ('gamma_p', 'region', 'eps_p_vol')

    def __init__(self, E, nu, c_peak, c_res, phi_peak_deg, phi_res_deg,
                 gp_c, gp_phi, psi_deg, checkpoint, device='cpu'):
        self.E, self.nu, self.psi_deg = float(E), float(nu), float(psi_deg)
        self.theta = np.array([c_peak, c_res, phi_peak_deg, phi_res_deg, gp_c, gp_phi])
        self.C = elastic_matrix_4(E, nu)
        self.G = self.C[3, 3]
        self.checkpoint = Path(checkpoint).resolve()
        self.model, saved = load_principal_checkpoint(self.checkpoint, E, nu, psi_deg, device)
        if np.any(self.theta < np.asarray(saved['lo'])-1e-8) or np.any(self.theta > np.asarray(saved['hi'])+1e-8):
            raise ValueError('FE material is outside the trained parameter box')
        self.device = device
        self.calls = self.points = self.transition_points = 0

    def elastic_tangent(self):
        return self.C.copy()

    def plastic_mask(self, state):
        return state[:, 0] > 0

    def equivalent_plastic_strain(self, state):
        return state[:, 0]

    def describe(self):
        return {'name': self.name, 'E': self.E, 'nu': self.nu, 'psi_deg': self.psi_deg,
                'theta': self.theta.tolist(), 'checkpoint': self.checkpoint.name,
                'checkpoint_sha256': hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                'material_calls': self.calls, 'integration_point_updates': self.points,
                'neural_transition_updates': self.transition_points,
                'stress_update': 'neural strength prediction and differentiable MC region layer',
                'tangent': 'derivative of the neural stress map with spectral rotation'}

    def update(self, deps, sig_old, state_old, dt=0.):
        n = len(deps)
        trial = sig_old + deps @ self.C.T
        sa, sb, angle = _principal_inplane(trial[:, 0], trial[:, 1], trial[:, 3])
        principal = np.column_stack((sa, sb, trial[:, 2]))
        order = np.argsort(-principal, axis=1, kind='stable')
        sorted_trial = np.take_along_axis(principal, order, 1)
        inp = torch.tensor(sorted_trial, dtype=torch.float64, device=self.device, requires_grad=True)
        gp = torch.as_tensor(state_old[:, 0], dtype=inp.dtype, device=self.device)
        theta = torch.as_tensor(self.theta, dtype=inp.dtype, device=self.device).expand(n, -1)
        with torch.enable_grad():
            out, dp, gn, reg, _, _ = self.model(inp, gp, theta)
            jac = torch.stack([torch.autograd.grad(out[:, k].sum(), inp, retain_graph=k<2)[0]
                               for k in range(3)], 1)
        out, dp, gn, reg, jac = [a.detach().cpu().numpy() for a in (out, dp, gn, reg, jac)]
        inverse = np.argsort(order, axis=1)
        local = np.take_along_axis(out, inverse, 1)
        ar = np.arange(n)
        J = np.zeros((n, 3, 3))
        J[ar[:, None, None], order[:, :, None], order[:, None, :]] = jac
        gap = principal[:, 0]-principal[:, 1]
        small = abs(gap) < 1e-9*(abs(principal).sum(1)+self.theta[0])
        shear = np.divide(local[:, 0]-local[:, 1], gap, out=np.zeros(n), where=~small)
        shear[small] = .5*(J[small, 0, 0]-J[small, 0, 1]-J[small, 1, 0]+J[small, 1, 1])
        D = np.zeros((n, 4, 4))
        D[:, :3, :3] = J @ self.C[:3, :3]
        D[:, 3, 3] = shear*self.G
        Ts, Te = _rotation_matrices(angle)
        tangent = Ts @ D @ Te
        # Same apex stiffness stabilization used by the FE's other materials.
        tangent[reg == 4] += 1e-3*self.C
        sig_local = np.column_stack((local, np.zeros(n)))
        stress = np.einsum('nij,nj->ni', Ts, sig_local)
        state = state_old.copy()
        state[:, 0], state[:, 1] = gn, reg
        state[:, 2] += dp.sum(1)
        active = reg > 0
        transitions = active & (state_old[:, 0] < max(self.theta[4:6]))
        self.calls += 1
        self.points += n
        self.transition_points += int(transitions.sum())
        return MaterialUpdate(stress, state, tangent,
                              {'n_plastic': int(active.sum()), 'n_neural_transition': int(transitions.sum()),
                               'n_edge': int(((reg == 2)|(reg == 3)).sum()), 'n_apex': int((reg == 4).sum())})
