"""Batched integration-point materials for the plane-strain FE framework.

Material interface (contract with ``horseshoe_fe``)
----------------------------------------------------
The solver never looks inside a material.  It owns, for every integration point
of a material block, a stress row and a state row, and once per Newton iteration
it hands the whole block to the material in ONE call::

    upd = material.update(deps, sig_old, state_old, dt)

    deps      (n_ip, 4)  total strain increment since the last CONVERGED state
                         (not since the last iteration), compression positive,
                         Voigt order (xx, yy, zz, xy) with ENGINEERING shear
                         gamma_xy; the solver always passes zz = 0 (plane strain)
    sig_old   (n_ip, 4)  stress at the last converged state, compression positive,
                         (sxx, syy, szz, txy) -- szz is a full stress component
                         that takes part in the principal ordering
    state_old (n_ip, n_state) internal variables at the last converged state
    dt        float      time increment of the current load step (for rate
                         regularizations; 0.0 in rate-independent runs)

    upd.sig     (n_ip, 4)     updated stress
    upd.state   (n_ip, n_state) updated internal variables
    upd.tangent (n_ip, 4, 4)  algorithmic (consistent) tangent d sig / d eps,
                              compression-positive on both sides, acting on the
                              engineering-shear Voigt vector; may be unsymmetric
    upd.info    dict          free-form diagnostics (counts of plastic points ...)

Every call is a pure function of (deps, sig_old, state_old, dt): the solver
calls it repeatedly with growing ``deps`` inside one increment and commits
``upd.sig`` / ``upd.state`` only when the increment has converged.  A material
therefore must not keep per-point memory of its own.

A material also provides::

    n_state, state_names   layout of the state row (documented per class)
    initial_state(n_ip)    (n_ip, n_state) virgin state
    elastic_tangent()      (4, 4) elastic operator (initial stiffness, lining)
    plastic_mask(state)    (n_ip,) bool, points that have yielded (diagnostics)
    equivalent_plastic_strain(state)  (n_ip,) scalar plastic measure for output

The state row is the only place a strain-softening (CWFS) material keeps its
equivalent plastic shear strain and a viscoplastic regularization keeps its
over-stress history; both are plugged in by subclassing :class:`Material` and
implementing :meth:`update` with the same signature, so the solver is untouched.
The solver keeps one material per element block (rock, lining) and calls each
with all integration points of its block.

Sign and component conventions
------------------------------
Compression positive for stress AND strain (the solver negates the kinematic
strain increment ``B du`` before calling the material and negates the stress
again when forming internal forces).  Principal stresses are sorted
``s1 >= s2 >= s3`` (s1 the largest compression).  Mohr--Coulomb in that
convention reads ``f = s1 - N s3 - sig_c`` with ``N = (1 + sin phi)/(1 - sin phi)``
and ``sig_c = 2 c cos phi/(1 - sin phi)``; the potential is ``g = s1 - N_psi s3``.

Materials provided
------------------
* :class:`LinearElastic` -- isotropic Hooke, no state.
* :class:`MohrCoulombEPP` -- elastic--perfectly-plastic Mohr--Coulomb, non-
  associated (constant dilation), plane strain with the out-of-plane stress in
  the principal ordering.  Exact return in principal-stress space to the yield
  plane, to the two edges (triaxial compression s2 = s3, triaxial extension
  s1 = s2) or to the apex, following the region logic of Clausen, Damkilde &
  Andersen (Comput. Struct. 85, 2007): a trial point returns to the plane, and
  if the plane return breaks the principal ordering it returns to the violated
  edge, and if the edge return passes the apex it returns to the apex.  The
  consistent tangent is the exact derivative of that map (constant per region in
  the principal frame, completed by the (s_a - s_b)/(s_a^tr - s_b^tr) shear term
  of the isotropic-function derivative and rotated to the xy frame).
  State row: ``gamma_p`` (accumulated |max - min principal plastic strain
  increment|, the equivalent plastic shear strain the CWFS laws are driven by),
  ``region`` (last return region: 0 elastic, 1 plane, 2 edge s1=s2, 3 edge
  s2=s3, 4 apex), ``eps_p_vol`` (accumulated plastic volumetric strain).
* :class:`TrescaEPP` -- :class:`MohrCoulombEPP` at ``phi = psi = 0``, i.e. the
  pressure-independent limit ``s1 - s3 <= sigma_lim`` with associated,
  incompressible flow.  Used for a support ring that must reach a load plateau
  and shed load when the only read strength numbers are uniaxial: it introduces
  no friction angle and no tension cut-off.
* :class:`CWFSMaterial` -- cohesion-weakening friction-strengthening Mohr--
  Coulomb: the same region logic and non-associated flow, with piecewise linear
  strengths driven by an equivalent plastic shear strain, a fully implicit return
  map (a bracketed scalar accumulation solve around admissible fixed-strength
  face, edge and apex returns), the consistent tangent of that map
  including the softening term, and an optional Duvaut--Lions viscoplastic
  relaxation.  Softening is LOCAL: there is no internal length, so a softening
  band collapses onto the element size.  See the class docstring for the state
  layout, the treatment of the vanishing and negative return denominator and the
  ``dt`` convention.  Optional CRACK-BAND (fracture-energy) regularisation makes
  the two softening thresholds inversely proportional to each integration point's
  own characteristic length, so that the energy dissipated per unit area of band
  is mesh-independent and ``gp_c`` / ``gp_phi`` are properties of a declared
  reference width instead of the mesh; see :meth:`CWFSMaterial._point_thresholds`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# equilateral triangle of edge h with three integration points: each point's
# tributary area is A/3 = (sqrt(3)/12) h^2, so its characteristic length is
CRACK_BAND_SHAPE_FACTOR = math.sqrt(math.sqrt(3.0) / 12.0)      # 0.37997


def crack_band_reference(h_wall: float) -> float:
    """Characteristic length of a T6 integration point on a mesh of target size ``h_wall``.

    The FE hands each integration point its OWN characteristic length
    ``sqrt(w_q det J)``, the square root of its tributary area; this function is
    the same quantity for an ideal equilateral element, and is what a crack-band
    reference width is quoted against so that a reference is a definition rather
    than a measurement of one particular mesh.
    """
    return CRACK_BAND_SHAPE_FACTOR * float(h_wall)


def _take(x, m):
    """``x[m]`` for a per-point array, ``None`` passed through (scalar thresholds)."""
    return None if x is None else x[m]


@dataclass
class MaterialUpdate:
    sig: np.ndarray
    state: np.ndarray
    tangent: np.ndarray
    info: dict


def lame_constants(E: float, nu: float):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    return lam, G


def elastic_matrix_4(E: float, nu: float) -> np.ndarray:
    """4x4 plane-strain-compatible isotropic operator on (xx, yy, zz, gamma_xy)."""
    lam, G = lame_constants(E, nu)
    C = np.zeros((4, 4))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * G
    C[3, 3] = G
    return C


class Material:
    """Base class; see the module docstring for the contract."""

    name = "base"
    n_state = 0
    state_names: tuple = ()

    def initial_state(self, n_ip: int) -> np.ndarray:
        return np.zeros((n_ip, self.n_state))

    def elastic_tangent(self) -> np.ndarray:
        raise NotImplementedError

    def update(self, deps, sig_old, state_old, dt: float = 0.0) -> MaterialUpdate:
        raise NotImplementedError

    def plastic_mask(self, state) -> np.ndarray:
        return np.zeros(state.shape[0], dtype=bool)

    def equivalent_plastic_strain(self, state) -> np.ndarray:
        return np.zeros(state.shape[0])

    def describe(self) -> dict:
        return {"name": self.name}


class LinearElastic(Material):
    name = "linear_elastic"
    n_state = 0
    state_names = ()

    def __init__(self, E: float, nu: float):
        self.E = float(E)
        self.nu = float(nu)
        self.C = elastic_matrix_4(self.E, self.nu)

    def elastic_tangent(self):
        return self.C.copy()

    def update(self, deps, sig_old, state_old, dt: float = 0.0):
        sig = sig_old + deps @ self.C.T
        tangent = np.broadcast_to(self.C, (deps.shape[0], 4, 4))
        return MaterialUpdate(sig=sig, state=state_old, tangent=tangent,
                              info={"n_plastic": 0})

    def describe(self):
        return {"name": self.name, "E": self.E, "nu": self.nu}


# ---------------------------------------------------------------------------
# Mohr-Coulomb elastic-perfectly-plastic, non-associated
# ---------------------------------------------------------------------------

def _principal_inplane(sxx, syy, txy):
    """Larger/smaller in-plane principal stress and the angle of the larger one."""
    m = 0.5 * (sxx + syy)
    dd = 0.5 * (sxx - syy)
    R = np.sqrt(dd * dd + txy * txy)
    theta = 0.5 * np.arctan2(txy, dd)
    return m + R, m - R, theta


def _rotation_matrices(theta):
    """T_sig (frame -> xy for stress-like Voigt vectors) and T_eps (xy -> frame for
    engineering-strain Voigt vectors), frame = (a, b, z, ab) with a at angle theta."""
    c = np.cos(theta)
    s = np.sin(theta)
    c2, s2, sc = c * c, s * s, s * c
    n = theta.shape[0]
    Ts = np.zeros((n, 4, 4))
    Ts[:, 0, 0] = c2; Ts[:, 0, 1] = s2; Ts[:, 0, 3] = -2.0 * sc
    Ts[:, 1, 0] = s2; Ts[:, 1, 1] = c2; Ts[:, 1, 3] = 2.0 * sc
    Ts[:, 2, 2] = 1.0
    Ts[:, 3, 0] = sc; Ts[:, 3, 1] = -sc; Ts[:, 3, 3] = c2 - s2
    Te = np.zeros((n, 4, 4))
    Te[:, 0, 0] = c2; Te[:, 0, 1] = s2; Te[:, 0, 3] = sc
    Te[:, 1, 0] = s2; Te[:, 1, 1] = c2; Te[:, 1, 3] = -sc
    Te[:, 2, 2] = 1.0
    Te[:, 3, 0] = -2.0 * sc; Te[:, 3, 1] = 2.0 * sc; Te[:, 3, 3] = c2 - s2
    return Ts, Te


class MohrCoulombEPP(Material):
    """Mohr--Coulomb elastic--perfectly-plastic, non-associated, plane strain.

    Parameters: ``E``, ``nu``, ``c`` (cohesion, stress units of the run),
    ``phi_deg`` (friction), ``psi_deg`` (dilation, ``0 <= psi <= phi``).
    """

    name = "mohr_coulomb_epp"
    n_state = 3
    state_names = ("gamma_p", "region", "eps_p_vol")

    def __init__(self, E: float, nu: float, c: float, phi_deg: float, psi_deg: float,
                 apex_tangent_fraction: float = 1e-3):
        self.E, self.nu, self.c = float(E), float(nu), float(c)
        self.phi_deg, self.psi_deg = float(phi_deg), float(psi_deg)
        # the exact apex tangent is zero; a small multiple of the elastic operator keeps
        # the global matrix regular where whole element patches sit at the apex (the
        # residual, hence the converged solution, is unaffected)
        self.apex_tangent_fraction = float(apex_tangent_fraction)
        if not (0.0 <= psi_deg <= phi_deg):
            raise ValueError("need 0 <= psi <= phi")
        self.C = elastic_matrix_4(self.E, self.nu)
        lam, G = lame_constants(self.E, self.nu)
        self.G = G
        self.Cn = self.C[:3, :3].copy()
        self.Cn_inv = np.linalg.inv(self.Cn)
        sp = math.sin(math.radians(phi_deg))
        cp = math.cos(math.radians(phi_deg))
        ss = math.sin(math.radians(psi_deg))
        self.N = (1.0 + sp) / (1.0 - sp)
        self.Npsi = (1.0 + ss) / (1.0 - ss)
        self.sig_c = 2.0 * self.c * cp / (1.0 - sp)
        # apex (hydrostatic tension c cot phi, negative in compression-positive)
        self.sig_apex = -self.sig_c / (self.N - 1.0) if self.N > 1.0 else -np.inf
        N, Np = self.N, self.Npsi
        # yield gradients (a_i) and potential gradients (b_i) of the three planes
        # meeting at the two edges of the sorted sextant s1 >= s2 >= s3
        a1 = np.array([1.0, 0.0, -N]); b1 = np.array([1.0, 0.0, -Np])   # s1 - N s3
        a2 = np.array([0.0, 1.0, -N]); b2 = np.array([0.0, 1.0, -Np])   # s2 - N s3
        a3 = np.array([1.0, -N, 0.0]); b3 = np.array([1.0, -Np, 0.0])   # s1 - N s2
        r1, r2, r3 = self.Cn @ b1, self.Cn @ b2, self.Cn @ b3
        I3 = np.eye(3)
        # plane return: constants
        self._a1, self._r1 = a1, r1
        self._den1 = float(a1 @ r1)
        self._J_plane = I3 - np.outer(r1, a1) / self._den1
        # edge s1 = s2 (planes 1 and 2)
        R12 = np.column_stack([r1, r2]); A12 = np.vstack([a1, a2])
        M12 = A12 @ R12
        self._R12, self._A12, self._Minv12 = R12, A12, np.linalg.inv(M12)
        self._J_edge12 = I3 - R12 @ self._Minv12 @ A12
        # edge s2 = s3 (planes 1 and 3)
        R13 = np.column_stack([r1, r3]); A13 = np.vstack([a1, a3])
        M13 = A13 @ R13
        self._R13, self._A13, self._Minv13 = R13, A13, np.linalg.inv(M13)
        self._J_edge13 = I3 - R13 @ self._Minv13 @ A13

    def describe(self):
        return {"name": self.name, "E": self.E, "nu": self.nu, "c": self.c,
                "phi_deg": self.phi_deg, "psi_deg": self.psi_deg,
                "N_phi": self.N, "N_psi": self.Npsi, "sig_c": self.sig_c,
                "apex_tangent_fraction": self.apex_tangent_fraction}

    def elastic_tangent(self):
        return self.C.copy()

    def plastic_mask(self, state):
        return state[:, 0] > 0.0

    def equivalent_plastic_strain(self, state):
        return state[:, 0]

    # -- core -------------------------------------------------------------
    def yield_function(self, sig4: np.ndarray) -> np.ndarray:
        """f = s1 - N s3 - sig_c on sorted principals (>0: outside)."""
        sa, sb, _ = _principal_inplane(sig4[:, 0], sig4[:, 1], sig4[:, 3])
        tri = np.column_stack([sa, sb, sig4[:, 2]])
        s = -np.sort(-tri, axis=1)
        return s[:, 0] - self.N * s[:, 2] - self.sig_c

    def update(self, deps, sig_old, state_old, dt: float = 0.0):
        n = deps.shape[0]
        sig_tr = sig_old + deps @ self.C.T
        sa, sb, theta = _principal_inplane(sig_tr[:, 0], sig_tr[:, 1], sig_tr[:, 3])
        tri = np.column_stack([sa, sb, sig_tr[:, 2]])          # frame (a, b, z)
        order = np.argsort(-tri, axis=1, kind="stable")        # sorted slot -> frame idx
        s_tr = np.take_along_axis(tri, order, axis=1)          # s1 >= s2 >= s3
        f_tr = s_tr[:, 0] - self.N * s_tr[:, 2] - self.sig_c
        # a point that was yielding at the last converged state and whose trial state
        # sits on the surface (zero strain increment at the first Newton iteration) is
        # treated as plastic: the stress is unchanged, but the tangent is the
        # elastoplastic one, which keeps the first iterate from an elastic overshoot
        scale_f = np.abs(s_tr).sum(axis=1) + self.sig_c + 1e-30
        plastic = (f_tr > 0.0) | ((state_old[:, 1] > 0.5) & (f_tr > -1e-10 * scale_f))

        sig_new = sig_tr.copy()
        state_new = state_old.copy()
        tangent = np.empty((n, 4, 4))
        tangent[:] = self.C
        region = np.zeros(n, dtype=np.int8)
        info = {"n_plastic": int(plastic.sum()), "n_apex": 0, "n_edge": 0}
        if not plastic.any():
            state_new[:, 1] = 0.0
            return MaterialUpdate(sig=sig_new, state=state_new, tangent=tangent, info=info)

        idx = np.nonzero(plastic)[0]
        st = s_tr[idx]                                           # (m, 3)
        m = idx.shape[0]
        # 1. plane return
        dlam = np.maximum(f_tr[idx], 0.0) / self._den1
        s_p = st - dlam[:, None] * self._r1[None, :]
        J = np.broadcast_to(self._J_plane, (m, 3, 3)).copy()
        reg = np.ones(m, dtype=np.int8)
        viol12 = s_p[:, 1] > s_p[:, 0]
        viol23 = s_p[:, 2] > s_p[:, 1]
        apex = viol12 & viol23
        e12 = viol12 & ~apex
        e23 = viol23 & ~apex
        s_new = s_p
        # 2. edge returns
        if e12.any():
            fe = np.column_stack([st[e12, 0] - self.N * st[e12, 2] - self.sig_c,
                                  st[e12, 1] - self.N * st[e12, 2] - self.sig_c])
            dl = fe @ self._Minv12.T
            se = st[e12] - dl @ self._R12.T
            s_new[e12] = se
            J[e12] = self._J_edge12
            reg[e12] = 2
            bad = se[:, 2] > se[:, 0]
            if bad.any():
                ii = np.nonzero(e12)[0][bad]
                apex[ii] = True
        if e23.any():
            fe = np.column_stack([st[e23, 0] - self.N * st[e23, 2] - self.sig_c,
                                  st[e23, 0] - self.N * st[e23, 1] - self.sig_c])
            dl = fe @ self._Minv13.T
            se = st[e23] - dl @ self._R13.T
            s_new[e23] = se
            J[e23] = self._J_edge13
            reg[e23] = 3
            bad = se[:, 1] > se[:, 0]
            if bad.any():
                ii = np.nonzero(e23)[0][bad]
                apex[ii] = True
        # 3. apex
        if apex.any():
            s_new[apex] = self.sig_apex
            J[apex] = 0.0
            reg[apex] = 4
        info["n_apex"] = int(apex.sum())
        info["n_edge"] = int(((reg == 2) | (reg == 3)).sum())
        # plastic strain increment (sorted frame) and state
        deps_p = (st - s_new) @ self.Cn_inv.T
        dgam = deps_p.max(axis=1) - deps_p.min(axis=1)
        state_new[idx, 0] = state_old[idx, 0] + dgam
        state_new[idx, 2] = state_old[idx, 2] + deps_p.sum(axis=1)
        region[idx] = reg
        # un-sort to the (a, b, z) frame
        ordp = order[idx]
        inv = np.argsort(ordp, axis=1)
        tri_new = np.take_along_axis(s_new, inv, axis=1)
        ar = np.arange(m)
        Jf = np.zeros((m, 3, 3))
        Jf[ar[:, None, None], ordp[:, :, None], ordp[:, None, :]] = J
        # shear term of the isotropic-function derivative
        da = tri[idx, 0] - tri[idx, 1]
        scale = np.abs(tri[idx]).sum(axis=1) + self.sig_c + 1e-30
        small = np.abs(da) <= 1e-9 * scale
        g = np.empty(m)
        with np.errstate(divide="ignore", invalid="ignore"):
            g[~small] = (tri_new[~small, 0] - tri_new[~small, 1]) / da[~small]
        g[small] = 0.5 * (Jf[small, 0, 0] - Jf[small, 0, 1] - Jf[small, 1, 0] + Jf[small, 1, 1])
        Df = np.zeros((m, 4, 4))
        Df[:, :3, :3] = Jf @ self.Cn
        Df[:, 3, 3] = g * self.G
        Ts, Te = _rotation_matrices(theta[idx])
        tangent[idx] = Ts @ Df @ Te
        if apex.any():
            tangent[idx[apex]] += self.apex_tangent_fraction * self.C
        # stress back to xy
        c, s = np.cos(theta[idx]), np.sin(theta[idx])
        pa, pb, pz = tri_new[:, 0], tri_new[:, 1], tri_new[:, 2]
        sig_new[idx, 0] = c * c * pa + s * s * pb
        sig_new[idx, 1] = s * s * pa + c * c * pb
        sig_new[idx, 2] = pz
        sig_new[idx, 3] = s * c * (pa - pb)
        state_new[:, 1] = region
        return MaterialUpdate(sig=sig_new, state=state_new, tangent=tangent, info=info)


# ---------------------------------------------------------------------------
# Tresca: pressure-independent strength limit (yielding lining ring)
# ---------------------------------------------------------------------------

class TrescaEPP(MohrCoulombEPP):
    """Elastic--perfectly-plastic with a pressure-INDEPENDENT strength limit.

    ``f = s1 - s3 - sigma_lim``: the largest principal stress difference the
    material can carry is ``sigma_lim``, which for a uniaxial path is exactly the
    uniaxial strength.  It is :class:`MohrCoulombEPP` at ``phi = psi = 0``, so
    ``N = 1``, ``sig_c = 2 c = sigma_lim``, the flow is associated and
    incompressible, and the whole return map, consistent tangent and state layout
    are inherited unchanged.

    With ``N = 1`` the apex sits at hydrostatic tension infinity and is never
    reached: the plane return gives ``s_p = (s1 - f/2, s2, s3 + f/2)``, so the two
    ordering violations ``s2 > s1 - f/2`` and ``s3 + f/2 > s2`` cannot hold at
    once (adding them gives ``0 > sig_c``), and on either edge the returned state
    keeps ``s1 - s3 = sig_c > 0``.  The material therefore never loses its
    tangent to the apex regularisation.

    What it deliberately does NOT carry: any dependence of the limit on the mean
    stress (a confined band is given no strength gain, an unconfined one no
    loss), and any tension cut-off -- ``f`` is blind to hydrostatic tension, so
    the band cannot crack.  Both are choices for a material whose only read
    strength numbers are uniaxial.
    """

    name = "tresca_epp"

    def __init__(self, E: float, nu: float, sigma_lim: float,
                 apex_tangent_fraction: float = 1e-3):
        sigma_lim = float(sigma_lim)
        if sigma_lim <= 0.0:
            raise ValueError("sigma_lim must be > 0")
        super().__init__(E, nu, 0.5 * sigma_lim, 0.0, 0.0,
                         apex_tangent_fraction=apex_tangent_fraction)
        self.sigma_lim = sigma_lim

    def describe(self):
        return {"name": self.name, "E": self.E, "nu": self.nu,
                "sigma_lim": self.sigma_lim,
                "yield_function": "s1 - s3 - sigma_lim (compression positive, sorted)",
                "flow": "associated, incompressible (psi = 0)",
                "apex_tangent_fraction": self.apex_tangent_fraction}


# ---------------------------------------------------------------------------
# CWFS: cohesion-weakening friction-strengthening Mohr-Coulomb
# ---------------------------------------------------------------------------

class CWFSMaterial(Material):
    """Cohesion-weakening friction-strengthening Mohr--Coulomb, plane strain.

    Strengths vary piecewise linearly with the accumulated plastic shear::

        c(gp)   = c_peak   + (c_res   - c_peak)   * min(gp / gp_c,   1)
        phi(gp) = phi_peak + (phi_res - phi_peak) * min(gp / gp_phi, 1)

    The general principal-state map solves for the accumulated increment x.
    At each candidate gp_base + x, _fixed_strength_map selects the admissible
    face, edge or apex at that strength. The plastic increment follows from
    dp = Cn^-1 (s_trial - s_return), and the scalar residual is
    span(dp) - x. This active-set evaluation avoids continuing an edge equality
    into negative plastic multipliers during friction strengthening.

    A doubling bracket is scanned at the strength breakpoints and at
    SCAN_SUBDIVISIONS subintervals per segment. When multiple crossings are
    detected, the first downward crossing is retained. Safeguarded secant
    steps and bisection solve the accumulated-strain residual. Stress, plastic
    shear and plastic volume are computed from the same accepted return.

    _tangent_blocks differentiates the face/edge consistency equations with
    their evolving strength. On a face, J = I - r (x) a / den, where
    den = a.r - kappa df/dgp. On an edge the corresponding two-plane matrix
    gives J = I - R M^-1 A. The apex uses p = -c cot(phi) and the principal
    plastic-strain span. Spectral rotation supplies the spatial tangent.

    softening_tangent_floor (default 0.05) caps the strength derivative when
    the implicit consistency denominator becomes small or negative. This
    changes the tangent, not the returned stress or state. The apex receives
    apex_tangent_fraction times the elastic operator. These stabilizations
    belong to this conventional material; the neural material differentiates
    its own explicit stress map.

    The optional crack-band setting scales the strength thresholds with the
    integration-point characteristic length. Without it, the law is local.
    Optional Duvaut--Lions relaxation, with r = dt/eta, blends the inviscid
    return and elastic trial as (trial + r*return)/(1+r), including the
    internal variables and tangent. eta = 0 disables this relaxation.

    State columns:
        0 gp_local   accumulated max-minus-min principal plastic increment
        1 region     0 elastic, 1 face, 2 edge s1=s2, 3 edge s2=s3, 4 apex
        2 eps_p_vol  accumulated plastic volume, compression positive

    flow="plane_strain" uses the full principal-region map and flow
    b1=(1,0,-N_psi). The retained flow="triaxial" interface uses the separate
    _map_split multiplier solve with b1=(1,-N_psi/2,-N_psi/2); it supports
    reproduction of the historical fixed-axis scalar model.
    """

    name = "cwfs"
    n_state = 3
    state_names = ("gp_local", "region", "eps_p_vol")
    CRACK_BAND_STATE_NAMES = ("gp_local", "region", "eps_p_vol", "h_char")

    PLANES = ((0, 2), (1, 2), (0, 1))    # (major slot, minor slot) of f_1, f_2, f_3
    FLOWS = ("plane_strain", "triaxial")
    SCAN_SUBDIVISIONS = 8                # residual samples per strength segment (root selection)

    def __init__(self, E: float, nu: float, c_peak: float, c_res: float,
                 phi_peak_deg: float, phi_res_deg: float, gp_c: float, gp_phi: float,
                 psi_deg: float, flow: str = "plane_strain",
                 viscoplastic: dict = None, crack_band: dict = None,
                 apex_tangent_fraction: float = 1e-3,
                 softening_tangent_floor: float = 0.05,
                 max_iter: int = 60, tol: float = 1e-14):
        self.E, self.nu = float(E), float(nu)
        self.c_peak, self.c_res = float(c_peak), float(c_res)
        self.phi_peak_deg, self.phi_res_deg = float(phi_peak_deg), float(phi_res_deg)
        self.gp_c, self.gp_phi = float(gp_c), float(gp_phi)
        self.psi_deg = float(psi_deg)
        self.flow = str(flow)
        self.apex_tangent_fraction = float(apex_tangent_fraction)
        self.softening_tangent_floor = float(softening_tangent_floor)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if self.flow not in self.FLOWS:
            raise ValueError(f"flow must be one of {self.FLOWS}")
        if self.gp_c <= 0.0 or self.gp_phi <= 0.0:
            raise ValueError("gp_c and gp_phi must be > 0")
        if min(self.c_peak, self.c_res) < 0.0:
            raise ValueError("cohesions must be >= 0")
        for p in (self.phi_peak_deg, self.phi_res_deg):
            if not (0.0 < p < 89.0):
                raise ValueError("friction angles must be in (0, 89) deg")
        if not (0.0 <= self.psi_deg <= min(self.phi_peak_deg, self.phi_res_deg)):
            raise ValueError("need 0 <= psi <= min(phi_peak, phi_res)")
        if not (0.0 < self.softening_tangent_floor <= 1.0):
            raise ValueError("softening_tangent_floor must be in (0, 1]")
        self.phi_peak = math.radians(self.phi_peak_deg)
        self.phi_res = math.radians(self.phi_res_deg)
        self.C = elastic_matrix_4(self.E, self.nu)
        lam, G = lame_constants(self.E, self.nu)
        self.G = G
        self.Cn = self.C[:3, :3].copy()
        self.Cn_inv = np.linalg.inv(self.Cn)
        ss = math.sin(math.radians(self.psi_deg))
        self.Npsi = (1.0 + ss) / (1.0 - ss)
        Np = self.Npsi
        b = np.zeros((3, 3))
        for k, (maj, mino) in enumerate(self.PLANES):
            b[k, maj] = 1.0
            b[k, mino] = -Np
        if self.flow == "triaxial":
            maj, mino = self.PLANES[0]
            third = 3 - maj - mino
            b[0] = 0.0
            b[0, maj] = 1.0
            b[0, mino] = -0.5 * Np
            b[0, third] = -0.5 * Np
        self.b = b
        r = b @ self.Cn                                    # r_m = Cn b_m (Cn symmetric)
        for k in range(3):                                 # exact symmetry of equal slots
            for i in range(3):
                for j in range(i + 1, 3):
                    if b[k, i] == b[k, j]:
                        r[k, j] = r[k, i]
        self.r = r
        self.kappa = b.max(axis=1) - b.min(axis=1)         # dgp per unit multiplier
        self.eta = 0.0
        if viscoplastic:
            cfg = dict(viscoplastic)
            self.eta = float(cfg.pop("eta", 0.0))
            if cfg:
                raise ValueError(f"unknown viscoplastic keys {sorted(cfg)}")
            if self.eta < 0.0:
                raise ValueError("viscoplastic eta must be >= 0")
        self.crack_band = None
        if crack_band:
            cfg = dict(crack_band)
            h_ref = float(cfg.pop("h_ref"))
            scale_phi = bool(cfg.pop("scale_phi", True))
            if cfg:
                raise ValueError(f"unknown crack_band keys {sorted(cfg)}")
            if h_ref <= 0.0:
                raise ValueError("crack_band h_ref must be > 0")
            self.crack_band = {
                "h_ref": h_ref, "scale_phi": scale_phi,
                # the mesh-independent invariants: the plastic slip across a band
                "w_c": self.gp_c * h_ref,
                "w_phi": (self.gp_phi * h_ref) if scale_phi else None,
            }
            self.n_state = 4
            self.state_names = self.CRACK_BAND_STATE_NAMES

    # -- crack band -------------------------------------------------------
    def initial_state(self, n_ip: int) -> np.ndarray:
        """Virgin state.  With crack-band scaling ``h_char`` starts at the REFERENCE
        width, so a caller that never writes the column gets the reference law
        exactly; the finite-element driver overwrites it with each point's own
        characteristic length."""
        st = np.zeros((n_ip, self.n_state))
        if self.crack_band is not None:
            st[:, 3] = self.crack_band["h_ref"]
        return st

    def _point_thresholds(self, state_old, info=None):
        """Per-point ``(gp_c, gp_phi)`` from the band width in the state row.

        Crack-band (fracture-energy) regularisation: the energy a softening band
        dissipates per unit AREA is the energy per unit VOLUME times the band
        width, and a local law with no internal length puts that band in one
        integration point's tributary strip.  Holding the areal energy fixed makes
        the softening THRESHOLD inversely proportional to that strip's width,

            gp_c(h) = w_c / h ,    w_c = gp_c^ref h_ref ,

        so the mesh-independent quantity is the plastic SLIP ``w_c`` across the
        band, in metres, and ``gp_c`` becomes a property of a declared reference
        width rather than of the mesh.  Returns ``(None, None)`` when the
        regularisation is off, which is the plain local law.
        """
        cb = self.crack_band
        if cb is None:
            return None, None
        h = np.asarray(state_old[:, 3], dtype=float)
        if h.size and not np.all(h > 0.0):
            raise ValueError("crack-band h_char must be > 0 at every point; the driver "
                             "has to write the characteristic length into state column 3")
        gpc = cb["w_c"] / h
        gpphi = (cb["w_phi"] / h) if cb["scale_phi"] else np.full(h.shape, self.gp_phi)
        if info is not None and h.size:
            info["crack_band_scale_min"] = float((cb["h_ref"] / h).min())
            info["crack_band_scale_max"] = float((cb["h_ref"] / h).max())
        return gpc, gpphi

    def snap_back_limit(self) -> dict:
        """Band width above which the softening law releases more elastic strain
        than the plastic strain it softens over -- i.e. it snaps back.

        Softening from peak to residual drops the uniaxial strength by
        ``d sigma_c = sigma_c(c_peak, phi_peak) - sigma_c(c_res, phi_res)`` with
        ``sigma_c = 2 c cos phi / (1 - sin phi)``, and unloading that drop
        elastically recovers ``eps_el = d sigma_c / E``.  The drop is spread over
        the plastic strain ``gp_c``, so the response of a band is monotonic only
        while ``gp_c >= eps_el``.

        With crack-band scaling ``gp_c(h) = w_c / h``, so that condition is a
        condition on the BAND WIDTH,

            h <= h_snap = w_c / eps_el = w_c E / d sigma_c ,

        which is a length the material sets: it needs no fit and no solve, and it
        is usable as a mesh criterion -- an element carrying yielding rock wider
        than ``h_snap`` cannot dissipate the band's energy without the response
        snapping back.  ``h_snap_m`` is ``None`` when the criterion is not a
        length: without crack-band scaling the threshold does not depend on ``h``
        at all (``snap_back_at_every_width`` then says whether the local law snaps
        back, at every width equally), and a law that gains strength between peak
        and residual (``d sigma_c <= 0``) never snaps back at any width.

        The measure is uniaxial and per point: it compares the strength drop with
        the elastic strain that drop releases, and says nothing about the
        confinement the point actually sits at, nor about whether the structure
        as a whole can find the softened branch.
        """
        def sig_c(c, phi_deg):
            sp = math.sin(math.radians(phi_deg))
            return 2.0 * c * math.cos(math.radians(phi_deg)) / (1.0 - sp)

        d_sig = (sig_c(self.c_peak, self.phi_peak_deg)
                 - sig_c(self.c_res, self.phi_res_deg))
        eps_el = d_sig / self.E
        cb = self.crack_band
        w_c = None if cb is None else float(cb["w_c"])
        h_snap = None
        if cb is not None and eps_el > 0.0:
            h_snap = w_c / eps_el
        local_snap = cb is None and eps_el > 0.0 and self.gp_c < eps_el
        return {
            "uniaxial_strength_drop_MPa": float(d_sig),
            "elastic_strain_released": float(eps_el),
            "gp_c_at_h_ref": float(self.gp_c),
            "w_c_m": w_c,
            "h_snap_m": h_snap,
            "snap_back_at_every_width": bool(local_snap),
            "crack_band": cb is not None,
            "formula": "h_snap = w_c / (d sigma_c / E), sigma_c = 2 c cos phi / (1 - sin phi)",
            "note": ("h_snap_m is None when the criterion is not a length: no crack-band "
                     "scaling (the threshold does not depend on h) or no strength drop "
                     "(d sigma_c <= 0, never snaps back)"),
        }

    # -- contract ---------------------------------------------------------
    def describe(self):
        return {"name": self.name, "E": self.E, "nu": self.nu,
                "c_peak": self.c_peak, "c_res": self.c_res,
                "phi_peak_deg": self.phi_peak_deg, "phi_res_deg": self.phi_res_deg,
                "gp_c": self.gp_c, "gp_phi": self.gp_phi, "psi_deg": self.psi_deg,
                "flow": self.flow, "N_psi": self.Npsi,
                "apex_tangent_fraction": self.apex_tangent_fraction,
                "softening_tangent_floor": self.softening_tangent_floor,
                "viscoplastic": {"eta": self.eta} if self.eta > 0.0 else None,
                "crack_band": (None if self.crack_band is None else
                               dict(self.crack_band,
                                    gp_c_at_h_ref=self.gp_c, gp_phi_at_h_ref=self.gp_phi,
                                    note=("gp_c and gp_phi are the thresholds AT h_ref; "
                                          "each integration point uses w/h with its own "
                                          "characteristic length h"))),
                "n_state": self.n_state, "state_names": list(self.state_names)}

    def elastic_tangent(self):
        return self.C.copy()

    def plastic_mask(self, state):
        return state[:, 0] > 0.0

    def equivalent_plastic_strain(self, state):
        return state[:, 0]

    # -- strengths --------------------------------------------------------
    def thresholds(self, gpc=None, gpphi=None):
        """Resolve the softening thresholds: the material's own, or per-point ones.

        ``gpc`` / ``gpphi`` are ``None`` for the plain local law (the scalar
        ``gp_c`` and ``gp_phi`` of the constructor) and per-point arrays when
        crack-band scaling makes them a function of each point's band width.
        """
        return (self.gp_c if gpc is None else gpc,
                self.gp_phi if gpphi is None else gpphi)

    def strengths(self, gp, gpc=None, gpphi=None):
        """``c, phi, sin phi, cos phi, N, sig_c`` and the gp-derivatives ``dN, dsig_c``.

        The strength laws are piecewise linear, so ``dc/dgp`` and ``dphi/dgp`` are
        one-sided at ``gp = 0``, ``gp_c`` and ``gp_phi``; the right derivative is
        used (zero at and beyond a threshold), which is the branch a continuation
        from the current state follows.
        """
        gp = np.asarray(gp, dtype=float)
        gc, gf = self.thresholds(gpc, gpphi)
        N, sig_c, c, sp, cp = self.strength_values(gp, gpc, gpphi)
        phi = self.phi_peak + (self.phi_res - self.phi_peak) * np.clip(gp / gf, 0.0, 1.0)
        dc = np.where((gp >= 0.0) & (gp < gc), (self.c_res - self.c_peak) / gc, 0.0)
        dphi = np.where((gp >= 0.0) & (gp < gf),
                        (self.phi_res - self.phi_peak) / gf, 0.0)
        om = 1.0 - sp
        dN = (2.0 * cp / (om * om)) * dphi
        dsig_c = 2.0 * dc * cp / om + (2.0 * c / om) * dphi
        return c, phi, sp, cp, N, sig_c, dN, dsig_c

    def strength_values(self, gp, gpc=None, gpphi=None):
        """``N, sig_c, c, sin phi, cos phi`` only (no derivatives): the hot path."""
        gc, gf = self.thresholds(gpc, gpphi)
        c = self.c_peak + (self.c_res - self.c_peak) * np.clip(gp / gc, 0.0, 1.0)
        phi = self.phi_peak + (self.phi_res - self.phi_peak) * np.clip(gp / gf, 0.0, 1.0)
        sp = np.sin(phi)
        cp = np.cos(phi)
        om = 1.0 - sp
        return (1.0 + sp) / om, 2.0 * c * cp / om, c, sp, cp

    def yield_function(self, sig4: np.ndarray, gp, gpc=None, gpphi=None) -> np.ndarray:
        """``f = s1 - N(gp) s3 - sig_c(gp)`` on sorted principals (> 0: outside)."""
        sa, sb, _ = _principal_inplane(sig4[:, 0], sig4[:, 1], sig4[:, 3])
        s = -np.sort(-np.column_stack([sa, sb, sig4[:, 2]]), axis=1)
        gp = np.broadcast_to(np.asarray(gp, dtype=float), (sig4.shape[0],))
        N, sig_c, _, _, _ = self.strength_values(gp, gpc, gpphi)
        return s[:, 0] - N * s[:, 2] - sig_c

    def apex_stress(self, gp, gpc=None, gpphi=None):
        """Hydrostatic apex ``p = -c cot phi`` (negative in compression positive)."""
        _, _, c, sp, cp = self.strength_values(gp, gpc, gpphi)
        return -c * cp / np.maximum(sp, 1e-12)

    # -- region kernels ---------------------------------------------------
    def _first_crossing_bracket(self, value, hi, kinks):
        """Tighten the bracket ``[0, hi]`` onto the FIRST downward zero crossing.

        ``value(x)`` is the vectorised return residual (positive outside the
        surface), ``hi`` an upper end found by doubling, at which the residual is
        non-positive, and ``kinks`` an ``(n, k)`` array of the multiplier values at
        which the strength laws change slope; entries outside ``(0, hi)`` are
        inert.  Between consecutive kinks the residual is smooth and beyond the
        last one it is linear, so its sign at ``0``, at the kinks, at
        ``SCAN_SUBDIVISIONS - 1`` interior points of every kink segment and at
        ``hi`` locates the first interval on which it crosses from positive to
        negative.

        Returns ``(lo, hi, x0)``: the bracket and the Newton start.  Where the
        residual, positive at ``0``, comes back above zero after a non-positive
        sample -- three roots below ``hi`` -- the bracket is the first ``(+, -)``
        scan interval and the start is its low end.  Everywhere else the scan
        shows at most one sign change, the residual has one non-negative root,
        and the bracket is ``[0, hi]`` with the start at ``hi``: exactly the
        iteration the class ran before the scan existed, so those points return
        bit for bit what they did (the class docstring, **Root selection**).
        """
        n = hi.shape[0]
        m = self.SCAN_SUBDIVISIONS
        bounds = np.column_stack([np.zeros(n), np.sort(kinks, axis=1)])
        bounds = np.maximum.accumulate(np.clip(bounds, 0.0, hi[:, None]), axis=1)
        pts = [bounds[:, 0]]
        for j in range(1, bounds.shape[1]):              # each kink segment: interior + its end
            a, b = bounds[:, j - 1], bounds[:, j]
            for i in range(1, m):
                pts.append(a + (b - a) * (i / m))
            pts.append(b)
        pts.append(hi)                                   # the linear tail needs no interior
        pts = np.column_stack(pts)
        pos = np.column_stack([value(pts[:, k]) > 0.0 for k in range(pts.shape[1])])
        neg_before = np.zeros_like(pos)
        neg_before[:, 1:] = np.logical_or.accumulate(~pos, axis=1)[:, :-1]
        three_roots = pos[:, 0] & (pos & neg_before).any(axis=1)
        down = pos[:, :-1] & ~pos[:, 1:]
        k = np.argmax(down, axis=1)
        rows = np.arange(n)
        lo_t, hi_t = pts[rows, k], pts[rows, k + 1]
        lo = np.where(three_roots, lo_t, 0.0)
        x0 = np.where(three_roots, lo_t, hi)
        hi = np.where(three_roots, hi_t, hi)
        return lo, hi, x0

    def _plane_return(self, s_tr, gp_base, f_tr, gpc=None, gpphi=None):
        """Smallest ``dlam >= 0`` solving ``f_1 = 0``.

        The upper end of the bracket is found by doubling the perfectly-plastic
        estimate ``f_tr / den_pp`` until the residual is non-positive; the bracket
        is then narrowed onto the first downward zero crossing by
        :meth:`_first_crossing_bracket` (kinks at ``(gp_phi - gp_base)/kappa`` and
        ``(gp_c - gp_base)/kappa``), and the root is found there by a Newton
        iteration with a bisection safeguard.
        """
        maj, mino = self.PLANES[0]
        r1 = self.r[0]
        kap = self.kappa[0]
        scale = np.abs(s_tr).sum(axis=1) + self.c_peak + 1.0
        tol = self.tol * scale

        def value(x):
            N, sig_c, _, _, _ = self.strength_values(gp_base + kap * x, gpc, gpphi)
            return (s_tr[:, maj] - x * r1[maj]) - N * (s_tr[:, mino] - x * r1[mino]) - sig_c

        def resid(x):
            _, _, _, _, N, sig_c, dN, dsig_c = self.strengths(gp_base + kap * x, gpc, gpphi)
            smj = s_tr[:, maj] - x * r1[maj]
            smn = s_tr[:, mino] - x * r1[mino]
            f = smj - N * smn - sig_c
            dfdx = -(r1[maj] - N * r1[mino]) + (-dN * smn - dsig_c) * kap
            return f, dfdx

        n = s_tr.shape[0]
        N0, _, _, _, _ = self.strength_values(gp_base, gpc, gpphi)
        den0 = r1[maj] - N0 * r1[mino]
        hi = np.where(den0 > 0.0, np.maximum(f_tr, 0.0) / np.where(den0 > 0.0, den0, 1.0), 1.0)
        hi = np.maximum(hi, 1e-14)
        for _ in range(120):
            need = value(hi) > 0.0
            if not need.any():
                break
            hi = np.where(need, 2.0 * hi, hi)
        gc, gf = self.thresholds(gpc, gpphi)
        kinks = np.column_stack([np.broadcast_to(gf, (n,)) - gp_base,
                                 np.broadcast_to(gc, (n,)) - gp_base]) / kap
        lo, hi, x = self._first_crossing_bracket(value, hi, kinks)
        n_it = 0
        for n_it in range(1, self.max_iter + 1):
            f, df = resid(x)
            lo = np.where(f > 0.0, x, lo)
            hi = np.where(f <= 0.0, x, hi)
            done = (np.abs(f) <= tol) | ((hi - lo) <= 1e-15 * (1.0 + hi))
            if done.all():
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                xn = x - f / df
            ok = np.isfinite(xn) & (xn > lo) & (xn < hi)
            x = np.where(done, x, np.where(ok, xn, 0.5 * (lo + hi)))
        return x, gp_base + kap * x, n_it, np.abs(value(x)) / scale

    def _edge_return(self, s_tr, gp_base, pa, pb, gpc=None, gpphi=None):
        """Edge formed by yield planes ``pa`` and ``pb``, reduced to one scalar root.

        For a FROZEN internal variable the two consistency conditions are linear:
        ``f_m(s_tr - dl_a r_a - dl_b r_b, gp) = 0`` is exactly
        ``M_pp(gp) dl = f(s_tr, gp)`` with ``M_pp[m][k] = a_m r_k``.  The only
        nonlinear condition left is the internal variable itself, so the 2x2
        system collapses to the scalar residual

            R(x) = kappa(x) . dl(x) - x ,   gp = gp_base + x

        in the plastic-shear-strain increment ``x``, with ``kappa`` the
        (semismooth) max/min slots of ``dl_a b_a + dl_b b_b``.  ``R(0) >= 0`` at a
        point that violated the ordering, and the strengths saturate so ``R``
        becomes ``-x`` asymptotically: the root is always bracketed.  The bracket
        is narrowed onto the first downward zero crossing of ``R`` by
        :meth:`_first_crossing_bracket` (kinks at ``gp_phi - gp_base`` and
        ``gp_c - gp_base``), exactly as on the plane, and the bracketed Newton
        below returns the smallest root.
        """
        n = s_tr.shape[0]
        ra, rb = self.r[pa], self.r[pb]
        ba, bb = self.b[pa], self.b[pb]
        maj_a, min_a = self.PLANES[pa]
        maj_b, min_b = self.PLANES[pb]
        scale = np.abs(s_tr).sum(axis=1) + self.c_peak + 1.0

        def solve2(M, rhs):
            det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
            bad = np.abs(det) <= 1e-300
            det = np.where(bad, 1.0, det)
            out = np.empty_like(rhs)
            out[:, 0] = (M[:, 1, 1] * rhs[:, 0] - M[:, 0, 1] * rhs[:, 1]) / det
            out[:, 1] = (-M[:, 1, 0] * rhs[:, 0] + M[:, 0, 0] * rhs[:, 1]) / det
            out[bad] = 0.0
            return out

        def state_at(x):
            """dl, N, sig_c and the internal-variable image xh at the guess x."""
            N, sig_c, _, _, _ = self.strength_values(gp_base + x, gpc, gpphi)
            M = np.empty((n, 2, 2))
            M[:, 0, 0] = ra[maj_a] - N * ra[min_a]
            M[:, 0, 1] = rb[maj_a] - N * rb[min_a]
            M[:, 1, 0] = ra[maj_b] - N * ra[min_b]
            M[:, 1, 1] = rb[maj_b] - N * rb[min_b]
            F = np.column_stack([s_tr[:, maj_a] - N * s_tr[:, min_a] - sig_c,
                                 s_tr[:, maj_b] - N * s_tr[:, min_b] - sig_c])
            dl = solve2(M, F)
            ep = dl[:, 0:1] * ba[None, :] + dl[:, 1:2] * bb[None, :]
            imax = np.argmax(ep, axis=1)
            imin = np.argmin(ep, axis=1)
            kap = np.column_stack([ba[imax] - ba[imin], bb[imax] - bb[imin]])
            xh = kap[:, 0] * dl[:, 0] + kap[:, 1] * dl[:, 1]
            return dl, N, sig_c, xh, M, kap

        def dR_at(x, dl, M, kap):
            """dR/dx = d(xh)/dgp - 1 with the max/min slots frozen (semismooth)."""
            _, _, _, _, _, _, dN, dsig_c = self.strengths(gp_base + x, gpc, gpphi)
            r0 = (-dN * s_tr[:, min_a] - dsig_c) + dN * (ra[min_a] * dl[:, 0] + rb[min_a] * dl[:, 1])
            r1_ = (-dN * s_tr[:, min_b] - dsig_c) + dN * (ra[min_b] * dl[:, 0] + rb[min_b] * dl[:, 1])
            ddl = solve2(M, np.column_stack([r0, r1_]))
            return kap[:, 0] * ddl[:, 0] + kap[:, 1] * ddl[:, 1] - 1.0

        def value(x):
            return state_at(x)[3] - x

        xh0 = state_at(np.zeros(n))[3]
        hi = np.maximum(xh0, 1e-16)
        for _ in range(120):
            need = value(hi) > 0.0
            if not need.any():
                break
            hi = np.where(need, 2.0 * hi, hi)
        gc, gf = self.thresholds(gpc, gpphi)
        kinks = np.column_stack([np.broadcast_to(gf, (n,)) - gp_base,
                                 np.broadcast_to(gc, (n,)) - gp_base])
        lo, hi, x = self._first_crossing_bracket(value, hi, kinks)
        n_it = 0
        for n_it in range(1, self.max_iter + 1):
            dl, N, sig_c, xh, M, kap = state_at(x)
            R = xh - x
            lo = np.where(R > 0.0, x, lo)
            hi = np.where(R <= 0.0, x, hi)
            done = (np.abs(R) <= self.tol * (1.0 + np.abs(x))) | ((hi - lo) <= 1e-15 * (1.0 + hi))
            if done.all():
                break
            dR = dR_at(x, dl, M, kap)
            with np.errstate(divide="ignore", invalid="ignore"):
                xn = x - R / dR
            ok = np.isfinite(xn) & (xn > lo) & (xn < hi)
            x = np.where(done, x, np.where(ok, xn, 0.5 * (lo + hi)))
        dl, N, sig_c, xh, _, _ = state_at(x)
        dl = np.where((xh0 <= 0.0)[:, None], 0.0, dl)        # degenerate: no return
        s = s_tr - dl[:, 0:1] * ra[None, :] - dl[:, 1:2] * rb[None, :]
        gp = gp_base + (xh if (xh0 > 0.0).all() else np.where(xh0 > 0.0, xh, 0.0))
        fa = s[:, maj_a] - N * s[:, min_a] - sig_c
        fb = s[:, maj_b] - N * s[:, min_b] - sig_c
        res = np.maximum(np.abs(fa), np.abs(fb)) / scale
        return dl, s, gp, n_it, res

    # -- update -----------------------------------------------------------
    def update(self, deps, sig_old, state_old, dt: float = 0.0):
        n = deps.shape[0]
        sig_tr = sig_old + deps @ self.C.T
        sig_new = sig_tr.copy()
        state_new = state_old.copy()
        tangent = np.empty((n, 4, 4))
        tangent[:] = self.C
        info = {"n_plastic": 0, "n_edge": 0, "n_apex": 0, "n_softening_capped": 0,
                "plane_iterations": 0, "edge_iterations": 0,
                "max_return_residual": 0.0, "viscoplastic_ratio": 0.0}
        if n == 0:
            return MaterialUpdate(sig=sig_new, state=state_new, tangent=tangent, info=info)

        sa, sb, theta = _principal_inplane(sig_tr[:, 0], sig_tr[:, 1], sig_tr[:, 3])
        tri = np.column_stack([sa, sb, sig_tr[:, 2]])
        order = np.argsort(-tri, axis=1, kind="stable")
        s_tr = np.take_along_axis(tri, order, axis=1)
        gp_loc_old = state_old[:, 0]
        region_old = state_old[:, 1]
        gpc, gpphi = self._point_thresholds(state_old, info)

        out = self._map(s_tr, gp_loc_old, region_old, info, gpc, gpphi)
        region = out["region"]
        gp_loc_new = gp_loc_old + out["dgam"]
        eps_p_vol_new = state_old[:, 2] + out["dep_vol"]

        idx = out["idx"]
        if idx.size:
            m = idx.shape[0]
            inv = np.argsort(order[idx], axis=1)
            tri_new = np.take_along_axis(out["s_new"][idx], inv, axis=1)
            ar = np.arange(m)
            ordp = order[idx]
            Jf = np.zeros((m, 3, 3))
            Jf[ar[:, None, None], ordp[:, :, None], ordp[:, None, :]] = out["J"]
            da = tri[idx, 0] - tri[idx, 1]
            scale = np.abs(tri[idx]).sum(axis=1) + self.c_peak + 1e-30
            small = np.abs(da) <= 1e-9 * scale
            g = np.empty(m)
            with np.errstate(divide="ignore", invalid="ignore"):
                g[~small] = (tri_new[~small, 0] - tri_new[~small, 1]) / da[~small]
            g[small] = 0.5 * (Jf[small, 0, 0] - Jf[small, 0, 1]
                              - Jf[small, 1, 0] + Jf[small, 1, 1])
            Df = np.zeros((m, 4, 4))
            Df[:, :3, :3] = Jf @ self.Cn
            Df[:, 3, 3] = g * self.G
            Ts, Te = _rotation_matrices(theta[idx])
            tangent[idx] = Ts @ Df @ Te
            apex = region[idx] == 4
            if apex.any():
                tangent[idx[apex]] += self.apex_tangent_fraction * self.C
            c_, s_ = np.cos(theta[idx]), np.sin(theta[idx])
            pa, pb, pz = tri_new[:, 0], tri_new[:, 1], tri_new[:, 2]
            sig_new[idx, 0] = c_ * c_ * pa + s_ * s_ * pb
            sig_new[idx, 1] = s_ * s_ * pa + c_ * c_ * pb
            sig_new[idx, 2] = pz
            sig_new[idx, 3] = s_ * c_ * (pa - pb)

        # Duvaut-Lions relaxation (rate regularisation; an exact no-op for eta = 0)
        if self.eta > 0.0 and dt > 0.0:
            rr = float(dt) / self.eta
            fac = 1.0 / (1.0 + rr)
            sig_new = (sig_tr + rr * sig_new) * fac
            gp_loc_new = (gp_loc_old + rr * gp_loc_new) * fac
            eps_p_vol_new = (state_old[:, 2] + rr * eps_p_vol_new) * fac
            tangent = (self.C[None, :, :] + rr * tangent) * fac
            info["viscoplastic_ratio"] = rr

        state_new[:, 0] = gp_loc_new
        state_new[:, 1] = region
        state_new[:, 2] = eps_p_vol_new
        return MaterialUpdate(sig=sig_new, state=state_new, tangent=tangent, info=info)

    def _fixed_strength_map(self, st, gp, gpc=None, gpphi=None):
        """Admissible MC return at supplied strengths, including active-set changes.

        Unlike an unconstrained two-plane solve, this map cannot continue an
        edge with negative plastic multipliers when strengthening moves the
        trial stress inside that candidate surface.
        """
        N, sc, _, _, _ = self.strength_values(gp, gpc, gpphi)
        f = st[:, 0]-N*st[:, 2]-sc
        dl = np.maximum(f, 0)/(self.r[0, 0]-N*self.r[0, 2])
        out = st-dl[:, None]*self.r[0]
        reg = np.where(f > 0, 1, 0).astype(np.int8)
        v12, v23 = out[:, 1] > out[:, 0], out[:, 2] > out[:, 1]
        apex = v12 & v23
        for mask, (pa, pb), rid in ((v12 & ~apex, (0, 1), 2), (v23 & ~apex, (0, 2), 3)):
            if not mask.any():
                continue
            ne, se, te = N[mask], sc[mask], st[mask]
            ra, rb = self.r[pa], self.r[pb]
            ma, mi = self.PLANES[pa]
            mb, mj = self.PLANES[pb]
            aa, ab = ra[ma]-ne*ra[mi], rb[ma]-ne*rb[mi]
            ba, bb = ra[mb]-ne*ra[mj], rb[mb]-ne*rb[mj]
            fa, fb = te[:, ma]-ne*te[:, mi]-se, te[:, mb]-ne*te[:, mj]-se
            det = aa*bb-ab*ba
            la, lb = (bb*fa-ab*fb)/det, (aa*fb-ba*fa)/det
            edge = te-la[:, None]*ra-lb[:, None]*rb
            out[mask] = edge
            reg[mask] = rid
            bad = edge[:, 2] > edge[:, 0] if rid == 2 else edge[:, 1] > edge[:, 0]
            apex[np.flatnonzero(mask)[bad]] = True
        if apex.any():
            out[apex] = self.apex_stress(gp[apex], _take(gpc, apex), _take(gpphi, apex))[:, None]
            reg[apex] = 4
        dp = (st-out) @ self.Cn_inv
        return out, dp, reg

    def _map(self, s_tr, gp_base, region_old, info, gpc=None, gpphi=None):
        """Solve plastic accumulation with the admissible active set at every iterate."""
        if self.flow == "triaxial":
            return self._map_split(s_tr, gp_base, region_old, info, gpc, gpphi)
        n = len(s_tr)
        N, sc, _, _, _ = self.strength_values(gp_base, gpc, gpphi)
        f = s_tr[:, 0]-N*s_tr[:, 2]-sc
        plastic = (f > 0) | ((region_old > .5) & (f > -1e-10*(abs(s_tr).sum(1)+sc+1e-30)))
        idx = np.flatnonzero(plastic)
        sn = s_tr.copy()
        region = np.zeros(n, np.int8)
        dg, dv = np.zeros(n), np.zeros(n)
        if not len(idx):
            return dict(s_new=sn, J=np.zeros((0,3,3)), idx=idx, region=region,dgam=dg,dep_vol=dv)
        st, gb = s_tr[idx], gp_base[idx]
        gc, gf = _take(gpc,idx), _take(gpphi,idx)

        def state(x):
            return self._fixed_strength_map(st,gb+x,gc,gf)

        def value(x):
            _, dp, _ = state(x)
            return np.ptp(dp,axis=1)-x

        zero = np.zeros(len(idx))
        r0 = value(zero)
        hi = np.maximum(r0,1e-16)
        for _ in range(120):
            need = value(hi)>0
            if not need.any():
                break
            hi = np.where(need,2*hi,hi)
        tc, tf = self.thresholds(gc,gf)
        kinks = np.column_stack((np.broadcast_to(tc,gb.shape)-gb,np.broadcast_to(tf,gb.shape)-gb))
        lo,hi,_ = self._first_crossing_bracket(value,hi,kinks)
        rl,rh = value(lo),value(hi)
        x = hi.copy()
        done = r0 <= self.tol
        x[done]=0
        for iteration in range(1,self.max_iter+1):
            rx = value(x)
            done |= (abs(rx)<=self.tol*(1+x)) | ((hi-lo)<=1e-15*(1+hi))
            if done.all():
                break
            pos = rx>0
            lo,rl = np.where(pos & ~done,x,lo),np.where(pos & ~done,rx,rl)
            hi,rh = np.where(~pos & ~done,x,hi),np.where(~pos & ~done,rx,rh)
            denom = rh-rl
            candidate = lo-rl*(hi-lo)/np.where(abs(denom)>1e-300,denom,-1.)
            central = (candidate>lo+.25*(hi-lo)) & (candidate<hi-.25*(hi-lo))
            x = np.where(done,x,np.where(central,candidate,.5*(lo+hi)))
        if not done.all():
            raise RuntimeError(f'CWFS accumulation solve failed: residual={abs(value(x)[~done]).max():.3g}, '
                               f'bracket_width={(hi-lo)[~done].max():.3g}')
        sp,dp,reg = state(x)
        gamma = gb+np.ptp(dp,axis=1)
        apex = reg==4
        ep = st[apex]@self.Cn_inv
        imax,imin = np.argmax(ep,axis=1),np.argmin(ep,axis=1)
        J,ncap = self._tangent_blocks(st,sp,gamma,reg,imax,imin,apex,gc,gf)
        sn[idx],region[idx],dg[idx],dv[idx] = sp,reg,np.ptp(dp,axis=1),dp.sum(1)
        Nr,scr,_,_,_ = self.strength_values(gamma,gc,gf)
        residual = sp[:,0]-Nr*sp[:,2]-scr
        info.update(n_plastic=len(idx),n_edge=int(((reg==2)|(reg==3)).sum()),n_apex=int(apex.sum()),
                    n_softening_capped=ncap,plane_iterations=iteration,edge_iterations=iteration,
                    max_return_residual=float(np.maximum(residual,0).max()/(abs(st).max()+self.c_peak+1)))
        return dict(s_new=sn,J=J,idx=idx,region=region,dgam=dg,dep_vol=dv)

    def _map_split(self, s_tr, gp_base, region_old, info, gpc=None, gpphi=None):
        """One rate-independent return of the whole block from the converged state."""
        n = s_tr.shape[0]
        N0, sc0, _, _, _ = self.strength_values(gp_base, gpc, gpphi)
        f_tr = s_tr[:, 0] - N0 * s_tr[:, 2] - sc0
        scale_f = np.abs(s_tr).sum(axis=1) + sc0 + 1e-30
        # a point that was yielding at the last converged state and whose trial state
        # sits on the surface (zero strain increment at the first Newton iteration) is
        # kept plastic: the stress is unchanged but the tangent is the elastoplastic one
        plastic = (f_tr > 0.0) | ((region_old > 0.5) & (f_tr > -1e-10 * scale_f))
        s_new = s_tr.copy()
        region = np.zeros(n, dtype=np.int8)
        dgam = np.zeros(n)
        dep_vol = np.zeros(n)
        info["n_plastic"] = int(plastic.sum())
        info["n_edge"] = 0
        info["n_apex"] = 0
        info["n_softening_capped"] = 0
        if not plastic.any():
            return {"s_new": s_new, "J": np.zeros((0, 3, 3)), "idx": np.zeros(0, np.int64),
                    "region": region, "dgam": dgam, "dep_vol": dep_vol}

        idx = np.nonzero(plastic)[0]
        st = s_tr[idx]
        gb = gp_base[idx]
        gc_i, gf_i = _take(gpc, idx), _take(gpphi, idx)
        m = idx.shape[0]

        # ---- 1. plane return --------------------------------------------
        dlam, gp_reg, n_it, res = self._plane_return(st, gb, f_tr[idx], gc_i, gf_i)
        info["plane_iterations"] = max(info["plane_iterations"], int(n_it))
        info["max_return_residual"] = max(info["max_return_residual"], float(res.max()))
        s_reg = st - dlam[:, None] * self.r[0][None, :]
        reg = np.ones(m, dtype=np.int8)
        tol_ord = 1e-12 * (np.abs(st).sum(axis=1) + self.c_peak + 1e-30)
        viol12 = s_reg[:, 1] > s_reg[:, 0] + tol_ord
        viol23 = s_reg[:, 2] > s_reg[:, 1] + tol_ord
        apex = viol12 & viol23
        edge_planes = np.zeros((m, 2), dtype=np.int8)

        # ---- 2. edge returns ---------------------------------------------
        for mask, (pa, pb), rid in ((viol12 & ~apex, (0, 1), 2),
                                    (viol23 & ~apex, (0, 2), 3)):
            if not mask.any():
                continue
            dl, se, gpe, n_it, res = self._edge_return(st[mask], gb[mask], pa, pb,
                                                       _take(gc_i, mask), _take(gf_i, mask))
            info["edge_iterations"] = max(info["edge_iterations"], int(n_it))
            info["max_return_residual"] = max(info["max_return_residual"], float(res.max()))
            s_reg[mask] = se
            gp_reg[mask] = gpe
            reg[mask] = rid
            edge_planes[mask, 0] = pa
            edge_planes[mask, 1] = pb
            bad = (se[:, 2] > se[:, 0] + tol_ord[mask]) if rid == 2 else \
                  (se[:, 1] > se[:, 0] + tol_ord[mask])
            if bad.any():
                apex[np.nonzero(mask)[0][bad]] = True

        # ---- 3. apex ------------------------------------------------------
        iamax = iamin = None
        if apex.any():
            e_tr = st[apex] @ self.Cn_inv
            iamax = np.argmax(e_tr, axis=1)
            iamin = np.argmin(e_tr, axis=1)
            ra = np.arange(e_tr.shape[0])
            gp_a = gb[apex] + (e_tr[ra, iamax] - e_tr[ra, iamin])
            s_reg[apex] = self.apex_stress(gp_a, _take(gc_i, apex),
                                           _take(gf_i, apex))[:, None]
            gp_reg[apex] = gp_a
            reg[apex] = 4

        # ---- state increments, exact from the returned stress --------------
        deps_p = (st - s_reg) @ self.Cn_inv
        dgam[idx] = deps_p.max(axis=1) - deps_p.min(axis=1)
        dep_vol[idx] = deps_p.sum(axis=1)
        s_new[idx] = s_reg
        region[idx] = reg

        # ---- analytic consistent tangent -----------------------------------
        Jm, n_cap = self._tangent_blocks(st, s_reg, gp_reg, reg, iamax, iamin, apex,
                                         gc_i, gf_i)
        info["n_softening_capped"] = int(n_cap)
        info["n_edge"] = int(((reg == 2) | (reg == 3)).sum())
        info["n_apex"] = int((reg == 4).sum())
        return {"s_new": s_new, "J": Jm, "idx": idx, "region": region, "dgam": dgam,
                "dep_vol": dep_vol}

    def _tangent_blocks(self, s_tr, s_new, gp, reg, iamax, iamin, apex,
                        gpc=None, gpphi=None):
        """Analytic consistent tangent d s_new / d s_trial in the sorted principal frame."""
        m = s_tr.shape[0]
        I3 = np.eye(3)
        J = np.broadcast_to(I3, (m, 3, 3)).copy()
        _, _, _, _, N, _, dN, dsig_c = self.strengths(gp, gpc, gpphi)
        beta = self.softening_tangent_floor
        n_cap = 0

        pl = reg == 1
        if pl.any():
            maj, mino = self.PLANES[0]
            r1 = self.r[0]
            k = int(pl.sum())
            a = np.zeros((k, 3))
            a[:, maj] = 1.0
            a[:, mino] = -N[pl]
            den_pp = r1[maj] - N[pl] * r1[mino]
            fg = -dN[pl] * s_new[pl, mino] - dsig_c[pl]
            kap = self.kappa[0]
            chi = kap * fg / den_pp
            theta = np.ones(k)
            cap = (1.0 - chi) < beta
            theta[cap] = (1.0 - beta) / chi[cap]
            n_cap += int(cap.sum())
            den = den_pp - theta * kap * fg
            J[pl] = I3[None] - (r1[None, :, None] * a[:, None, :]) / den[:, None, None]

        for rid, (pa, pb) in ((2, (0, 1)), (3, (0, 2))):
            eg = reg == rid
            if not eg.any():
                continue
            k = int(eg.sum())
            ra, rb = self.r[pa], self.r[pb]
            ba, bb = self.b[pa], self.b[pb]
            maj_a, min_a = self.PLANES[pa]
            maj_b, min_b = self.PLANES[pb]
            Ne = N[eg]
            A = np.zeros((k, 2, 3))
            A[:, 0, maj_a] = 1.0
            A[:, 0, min_a] = -Ne
            A[:, 1, maj_b] = 1.0
            A[:, 1, min_b] = -Ne
            RM = np.empty((k, 3, 2))
            RM[:, :, 0] = ra
            RM[:, :, 1] = rb
            Mpp = np.empty((k, 2, 2))
            Mpp[:, 0, 0] = ra[maj_a] - Ne * ra[min_a]
            Mpp[:, 0, 1] = rb[maj_a] - Ne * rb[min_a]
            Mpp[:, 1, 0] = ra[maj_b] - Ne * ra[min_b]
            Mpp[:, 1, 1] = rb[maj_b] - Ne * rb[min_b]
            ep = (s_tr[eg] - s_new[eg]) @ self.Cn_inv
            imax = np.argmax(ep, axis=1)
            imin = np.argmin(ep, axis=1)
            kap = np.column_stack([ba[imax] - ba[imin], bb[imax] - bb[imin]])
            fg = np.column_stack([-dN[eg] * s_new[eg, min_a] - dsig_c[eg],
                                  -dN[eg] * s_new[eg, min_b] - dsig_c[eg]])
            det = Mpp[:, 0, 0] * Mpp[:, 1, 1] - Mpp[:, 0, 1] * Mpp[:, 1, 0]
            z0 = (Mpp[:, 1, 1] * fg[:, 0] - Mpp[:, 0, 1] * fg[:, 1]) / det
            z1 = (-Mpp[:, 1, 0] * fg[:, 0] + Mpp[:, 0, 0] * fg[:, 1]) / det
            chi = kap[:, 0] * z0 + kap[:, 1] * z1     # kap^T Mpp^-1 fg, explicit 2x2
            theta = np.ones(k)
            cap = (1.0 - chi) < beta
            theta[cap] = np.where(chi[cap] > 0.0, (1.0 - beta) / chi[cap], 1.0)
            n_cap += int(cap.sum())
            M = Mpp - theta[:, None, None] * fg[:, :, None] * kap[:, None, :]
            dM = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
            Minv = np.empty_like(M)
            Minv[:, 0, 0] = M[:, 1, 1] / dM
            Minv[:, 0, 1] = -M[:, 0, 1] / dM
            Minv[:, 1, 0] = -M[:, 1, 0] / dM
            Minv[:, 1, 1] = M[:, 0, 0] / dM
            J[eg] = I3[None] - RM @ (Minv @ A)

        if apex is not None and apex.any():
            gpa = gp[apex]
            gca, gfa = self.thresholds(_take(gpc, apex), _take(gpphi, apex))
            c_a, _, sp, cp, _, _, _, _ = self.strengths(gpa, _take(gpc, apex),
                                                        _take(gpphi, apex))
            dc = np.where((gpa >= 0.0) & (gpa < gca),
                          (self.c_res - self.c_peak) / gca, 0.0)
            dphi = np.where((gpa >= 0.0) & (gpa < gfa),
                            (self.phi_res - self.phi_peak) / gfa, 0.0)
            sp = np.maximum(sp, 1e-12)
            dp_dgp = -dc * cp / sp + c_a * dphi / (sp * sp)     # d(-c cot phi)/d gp
            k = int(apex.sum())
            u = np.zeros((k, 3))
            rr = np.arange(k)
            u[rr, iamax] += 1.0
            u[rr, iamin] -= 1.0
            v = u @ self.Cn_inv
            J[apex] = dp_dgp[:, None, None] * np.ones((1, 3, 1)) * v[:, None, :]
        return J, n_cap


MATERIALS = {"elastic": LinearElastic, "mc": MohrCoulombEPP, "cwfs": CWFSMaterial,
             "tresca": TrescaEPP}

CWFS_KEYS = ("E", "nu", "c_peak", "c_res", "phi_peak_deg", "phi_res_deg",
             "gp_c", "gp_phi", "psi_deg")


def make_material(spec: dict) -> Material:
    """Build a material from a plain dict.

    ``{"model": "elastic", E, nu}``
    ``{"model": "mc", E, nu, c, phi_deg, psi_deg}``
    ``{"model": "tresca", E, nu, sigma_lim, [apex_tangent_fraction]}``
    ``{"model": "cwfs", E, nu, c_peak, c_res, phi_peak_deg, phi_res_deg, gp_c,
       gp_phi, psi_deg, [flow], [viscoplastic: {eta}],
       [crack_band: {h_ref, [scale_phi]}], [apex_tangent_fraction],
       [softening_tangent_floor], [max_iter], [tol]}``
    """
    spec = dict(spec)
    model = spec.pop("model", "elastic")
    if model == "neural_cwfs":
        from .neural_material import NeuralCWFSMaterial
        return NeuralCWFSMaterial(**spec)
    if model == "elastic":
        return LinearElastic(spec["E"], spec["nu"])
    if model == "mc":
        return MohrCoulombEPP(spec["E"], spec["nu"], spec["c"], spec["phi_deg"], spec["psi_deg"])
    if model == "tresca":
        kw = {k: spec.pop(k) for k in ("E", "nu", "sigma_lim")}
        if "apex_tangent_fraction" in spec:
            kw["apex_tangent_fraction"] = spec.pop("apex_tangent_fraction")
        if spec:
            raise ValueError(f"unknown tresca material keys {sorted(spec)}")
        return TrescaEPP(**kw)
    if model == "cwfs":
        kw = {k: spec.pop(k) for k in CWFS_KEYS}
        kw["flow"] = spec.pop("flow", "plane_strain")
        kw["viscoplastic"] = spec.pop("viscoplastic", None)
        kw["crack_band"] = spec.pop("crack_band", None)
        for opt in ("apex_tangent_fraction", "softening_tangent_floor", "max_iter", "tol"):
            if opt in spec:
                kw[opt] = spec.pop(opt)
        if spec:
            raise ValueError(f"unknown cwfs material keys {sorted(spec)}")
        return CWFSMaterial(**kw)
    raise ValueError(f"unknown material model '{model}'")
