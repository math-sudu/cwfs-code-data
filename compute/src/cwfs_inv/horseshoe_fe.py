"""Plane-strain staged-excavation FE framework for the inclined-shaft case study.

Public entry points (importable with ``PYTHONPATH=compute/src``)::

    RunConfig, StageSpec, default_config(...)   run description (plain dicts inside)
    build_model(cfg) -> StagedModel              geometry + mesh + solver
    StagedModel.run() -> dict                    staged solve, returns the run record
    run_case(cfg, out_json=None) -> dict         convenience wrapper
    python -m cwfs_inv.horseshoe_fe --help       CLI (arguments or a JSON file)

Mechanics
---------
Small strain, plane strain, compression positive for stresses, uniform initial
stress ``(sigma_h = K sigma_v, sigma_v, sigma_z, 0)`` with no gravity gradient.
Half domain (x >= 0) with ``u_x = 0`` on the axis and ``u = 0`` on the far
half-circle of radius ``r_far`` (the far-boundary reactions carry the initial
stress; the truncation error of the fixed boundary is O((a/r_far)^2) in the
wall displacement).  Six-node triangles, three Gauss points each.

Excavation of a stage removes its elements and releases the unbalanced
boundary force progressively.  With ``f_int(S, sig) = sum_S int B^T (-sig)``
(internal force of element set ``S`` in the tension-positive FE convention),
``F_0 = f_int(all, sigma_0)`` the discrete far-field load (non-zero only on the
fixed far boundary) and ``f_R = f_int(removed, sig_at_removal)`` the force the
removed elements exerted on the remaining domain, the load during the stage is
``F_ext(lambda) = F_0 - (1 - lambda) f_R`` with the release factor ``lambda``
stepping from 0 to ``lambda_end`` in user-set increments; at ``lambda = 1`` the
new boundary is traction-free.  A stage that ends below 1 hands its remainder
``(1 - lambda_end) f_R`` to the next stage, whose ``f_R`` is then the force of
its own removed elements plus that remainder (``StageSpec.lambda_end``), so a
partly released heading can be followed by the bench and the invert and the
whole profile is traction-free once the last stage reaches 1.  Nodes that
belong to no active element leave the system.  At ``lambda_install`` the stage's lining-ring elements are activated
with zero stress and the lining material of ``RunConfig.lining``; the ring is a
continuum band bonded to the rock, so subsequent load increments strain it
from that instant on.  Readings are stored in total and relative to the
install instant of the stage that contains the reading point.

The ring material is whatever ``RunConfig.lining`` describes (see
:func:`lining_material`): ``{"model": "elastic", E, nu}`` -- the default and the
historical behaviour -- is a band that can only stiffen the opening, while
``{"model": "tresca", E, nu, sigma_lim}`` gives it a strength limit, so it
reaches a load plateau and sheds load once the section is at its limit.  How
much of the installed ring is at that limit is reported with the run
(``lining.yield``).

``StageSpec.install_gap_m`` turns the space between rock and ring into a real
CLOSING GAP: the ring of a stage engages at the first converged increment at or
after ``lambda_install`` at which the inward normal convergence of that stage's
excavation line has reached the gap, and if the gap never closes within the
stage the ring is never installed (reported in ``lining.rings_not_installed``).
Without it the ring is in contact from ``lambda_install`` on, which is what the
geometric offset ``HorseshoeParams.reserved_deformation`` alone gives.

``RunConfig.ring_upgrade`` is a single RING-UPGRADE EVENT, off by default:
``{"enabled": True, "stage": k, "lambda": l, "material": spec}`` switches, at the
first converged increment of stage ``k`` at or after release factor ``l``, every
ring element installed and active at that instant to the material ``spec``.
The switch keeps the element's stress and its state row (the accumulated
plastic strain of the primary ring is carried; a material without state drops
the row), so the band continues from the state the primary ring had reached
with a new stiffness and a new strength limit -- this is a secondary lining
cast inside a primary ring that is at its plateau.  Rings installed AFTER the
event keep the primary material.  :func:`secondary_lining_band` derives the
band material of a primary ring plus a secondary lining from the two sections.
With the event disabled the model is bit-identical to one without the field.

The rock material may carry CRACK-BAND (fracture-energy) regularisation.  A
local softening law has no internal length, so its band collapses onto one row
of integration points and the energy it dissipates per unit area of band falls
with the mesh.  With ``rock["crack_band"] = {"h_ref": ...}`` each integration
point is handed its OWN characteristic length -- the square root of its
tributary area, written into the state row by
:meth:`StagedModel._write_characteristic_length` -- and the softening thresholds
become ``w/h`` with ``w = gp h_ref``, so that the plastic slip across the band,
not the plastic strain in it, is the material property.  The realised spread of
``h`` over the mesh is reported with the run under ``materials.crack_band``.

Systematic rock bolts are the second component of the primary support.  They
are not discretised: a fully grouted radial pattern is smeared into the rock of
the annulus it reinforces, as the confinement one bolt spreads over its
tributary area and the Mohr--Coulomb cohesion that confinement is worth (see
:func:`bolt_reinforcement` for the derivation and for what the representation
cannot carry).  The annulus elements are switched to the reinforced material at
the ``lambda_install`` of the stage whose profile they face, i.e. the bolts and
the ring of a stage start to act at the same instant.

The release factor doubles as the PSEUDO-TIME of the run: the increment handed
to the materials is ``dt = solver["time_scale"] * d lambda`` (``time_scale``
default 1), which is what a rate regularization such as the Duvaut--Lions
relaxation of ``fe_materials.CWFSMaterial`` relaxes towards.  Rate-independent
materials ignore it, so ``dt`` changes nothing for them.

Equilibrium at every increment is solved by Newton--Raphson on the residual
``F_ext - f_int`` with the material's algorithmic tangent (unsymmetric sparse
LU) and a backtracking line search; when no Newton step lowers the residual
(loading/unloading switches, non-associated tangents losing definiteness) a
block of modified-Newton iterations with the elastic stiffness and Irons--Tuck
relaxation is interposed, and if the increment still fails the release
increment is halved (adaptive stepping back up to the user increment after two
fast increments).  Convergence: residual norm below ``tol_rel`` (default 1e-3)
times the norm of the stage's release force.  Materials implement the batched interface of
``cwfs_inv.fe_materials`` (see that module's docstring): one call per material
block per iteration with all its integration points.

Outputs (also written to the run JSON)
--------------------------------------
crown settlement; chord shortenings between mirrored points on the explicitly
selected observation boundary (excavation line by default, support inner line
for the CAD S2 binding), at depths below the reference crown; installation
stages, zero references, coordinates and initial chord lengths are recorded.
Total inner-support nodal displacement retains the FE activation reference;
only its installation-referenced increment is a target observation.
Other outputs include
lining normal pressure at the rock--lining interface and mean hoop stress at
five stations (crown, shoulder, haunch, foot and invert centre). CAD station
depths are explicit; the historical profile uses a 45-degree shoulder ray,
bench-line haunch and foot 0.5 m above its wall base. These are obtained from the
installed lining integration points within an arclength window around the
station by a linear fit of the normal stress through the thickness (value at
the interface) and the weighted mean of the tangential stress; plastic-zone
depth normal to the excavation line at crown, haunch, foot and invert (largest
normal distance of an integration point with ``gamma_p > 0`` inside a band
around the station's normal); maximum equivalent plastic shear strain; solver
diagnostics (iterations, residual norms, wall time per increment).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from cwfs_inv.fe_materials import Material, LinearElastic, MohrCoulombEPP, make_material
from cwfs_inv.horseshoe_geometry import (
    HorseshoeParams, horseshoe_net_profile, cad_reference_profile, circle_profile, offset_profile,
    point_at_y, profile_top, profile_bottom, ProfileCurve, build_mesh, MeshData,
)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class StageSpec:
    """One excavation stage.

    regions         mesh stage ids (1-based, top to bottom) whose core + ring are
                    removed at the start of this stage
    n_increments    number of equal lambda increments from 0 to lambda_end
    lambda_install  release factor at which the rings of ``install_regions`` are
                    installed stress-free (None = never installed)
    lambda_end      final release factor of the stage (1.0 = full release; e.g.
                    1 - P_i/P_0 to stop at a residual support pressure).  On a
                    stage that is not the last one, the unreleased remainder
                    ``(1 - lambda_end) f_R`` is carried into the release force
                    of the next stage, which then releases it together with its
                    own excavation force from 0 to its own ``lambda_end``: the
                    next stage is excavated while this one is still partly
                    held, as a bench that follows the heading before the face
                    effect has vanished.  The final state at ``lambda_end = 1``
                    of the last stage is traction-free on the whole profile
                    whatever the intermediate values.
    install_regions defaults to ``regions``
    install_gap_m   radial CLOSING GAP (m) between the rock and the ring, None =
                    none.  The ring of this stage engages at the first converged
                    increment at or after ``lambda_install`` whose inward normal
                    convergence of the stage's own excavation line has reached
                    the gap; if the gap never closes within the stage the ring is
                    never installed.  ``lambda_install`` therefore keeps its
                    meaning (the crew cannot spray before it) and the gap adds
                    the second condition (the ring cannot carry before it
                    touches).  A single instant for the whole ring is an
                    approximation: a real gap closes point by point, and the
                    criterion used here is the LARGEST inward normal
                    displacement of the stage's excavation line, i.e. the
                    earliest instant at which any part of the profile has
                    reached the ring.
    """
    regions: list
    n_increments: int = 10
    lambda_install: Optional[float] = 0.5
    lambda_end: float = 1.0
    install_regions: Optional[list] = None
    extra_lambdas: list = field(default_factory=list)
    install_gap_m: Optional[float] = None


@dataclass
class RunConfig:
    geometry: str = "horseshoe"              # historical "horseshoe" | "cad_reference" | "circle"
    horseshoe: dict = field(default_factory=lambda: asdict(HorseshoeParams()))
    cad_section: dict = field(default_factory=dict)
    circle_radius: float = 4.9
    lining: dict = field(default_factory=lambda: {"enabled": True, "E": 37000.0, "nu": 0.2})
    ring_upgrade: dict = field(default_factory=lambda: {"enabled": False})
    bolts: dict = field(default_factory=lambda: {"enabled": False})
    rock: dict = field(default_factory=lambda: {"model": "elastic", "E": 560.0, "nu": 0.3})
    stress: dict = field(default_factory=lambda: {"sigma_v": 6.0, "K": 1.5, "sigma_z": 10.0})
    mesh: dict = field(default_factory=lambda: {"h_wall": 0.5, "r_far": 80.0, "h_far": None,
                                                "h_ring": None, "dist_min": None,
                                                "growth_length": None})
    stages: list = field(default_factory=lambda: [asdict(StageSpec([1])),
                                                  asdict(StageSpec([2])),
                                                  asdict(StageSpec([3]))])
    solver: dict = field(default_factory=lambda: {"tol_rel": 1e-3, "tol_abs": 1e-10,
                                                  "max_iter": 25, "modified_max_iter": 400,
                                                  "relax_block": 30, "max_stalls": 10,
                                                  "max_bisect": 8, "line_search": True,
                                                  "adaptive": True, "divergence_factor": 1e3,
                                                  "on_failure": "raise", "time_scale": 1.0})
    readings: dict = field(default_factory=lambda: {"chord_depths": [2.5, 4.7, None],
                                                    "station_window": 0.5,
                                                    "plastic_band": 0.5})
    label: str = ""

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        cfg = RunConfig()
        for k, v in d.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key '{k}'")
            cur = getattr(cfg, k)
            if isinstance(cur, dict) and isinstance(v, dict):
                cur = dict(cur)
                cur.update(v)
                setattr(cfg, k, cur)
            else:
                setattr(cfg, k, copy.deepcopy(v))
        cfg.stages = [asdict(StageSpec(**s)) if isinstance(s, dict) else asdict(s)
                      for s in cfg.stages]
        if cfg.geometry == "cad_reference":
            cfg.horseshoe = {}  # No inactive legacy geometry in a CAD replay record.
        return cfg

    def to_dict(self) -> dict:
        return copy.deepcopy(asdict(self))


def default_config(**overrides) -> RunConfig:
    return RunConfig.from_dict(overrides)


# Keys of ``RunConfig.lining`` that describe the RING rather than its material:
# whether it exists at all, its thickness and stage cuts (circle geometry only,
# the horseshoe takes both from ``HorseshoeParams``) and free-text provenance.
LINING_NON_MATERIAL_KEYS = ("enabled", "thickness", "cut_levels", "source",
                            "strength_source", "note")


def lining_material(lining: dict) -> Material:
    """Material of the lining ring from ``RunConfig.lining``.

    Everything that is not a ring descriptor (:data:`LINING_NON_MATERIAL_KEYS`)
    is a ``fe_materials.make_material`` spec.  ``model`` defaults to ``elastic``,
    so a config written before the ring could yield builds the same linear
    elastic band it always did; ``{"model": "tresca", E, nu, sigma_lim}`` gives
    the band a strength limit, i.e. a load plateau and the ability to shed load.
    """
    spec = {k: v for k, v in lining.items() if k not in LINING_NON_MATERIAL_KEYS}
    spec.setdefault("model", "elastic")
    return make_material(spec)


RING_UPGRADE_KEYS = ("enabled", "stage", "lambda", "material", "rule", "source", "note",
                     "label")


def ring_upgrade_spec(ring_upgrade: dict, n_stages: int, stages: list) -> Optional[dict]:
    """Validated ``RunConfig.ring_upgrade``: ``None`` when the event is off.

    When on it needs ``stage`` (1-based index into ``RunConfig.stages``),
    ``lambda`` (a release factor inside that stage, ``0 < lambda <=
    lambda_end``) and ``material`` (a ``fe_materials.make_material`` spec).
    ``rule`` / ``source`` / ``note`` / ``label`` are free provenance.
    """
    ru = dict(ring_upgrade or {})
    unknown = sorted(k for k in ru if k not in RING_UPGRADE_KEYS)
    if unknown:
        raise ValueError(f"unknown ring_upgrade keys {unknown}")
    if not ru.get("enabled", False):
        return None
    missing = [k for k in ("stage", "lambda", "material") if ru.get(k) is None]
    if missing:
        raise ValueError(f"ring_upgrade is enabled but lacks {missing}")
    stage = int(ru["stage"])
    if not (1 <= stage <= n_stages):
        raise ValueError(f"ring_upgrade.stage {stage} is not one of 1..{n_stages}")
    lam = float(ru["lambda"])
    lam_end = float(stages[stage - 1].get("lambda_end", 1.0))
    if not (0.0 < lam <= lam_end + 1e-12):
        raise ValueError(f"ring_upgrade.lambda {lam} is outside (0, lambda_end = {lam_end}] "
                         f"of stage {stage}")
    if not isinstance(ru["material"], dict):
        raise ValueError("ring_upgrade.material must be a material spec dict")
    return {"stage": stage, "lambda": lam, "material": dict(ru["material"]),
            "rule": ru.get("rule"), "source": ru.get("source"), "note": ru.get("note"),
            "label": ru.get("label")}


def secondary_lining_band(primary: dict, band_thickness_m: float, secondary: dict,
                          strength: str = "tresca") -> dict:
    """Band material once a secondary lining is cast inside the primary ring.

    The model has ONE ring band of thickness ``t_b`` (the primary ring's), and
    the secondary lining is a new ring of thickness ``t_s``, modulus ``E_s`` and
    uniaxial strength ``f_s`` cast stress-free inside it.  The band keeps the
    primary's stress at the switch (:class:`StagedModel` keeps stress and state
    through a ring upgrade), so what the upgraded band must carry in ADDITION to
    the primary's stress is what the secondary lining adds, smeared over the
    band:

        E_band     = E_s t_s / t_b
        sigma_band = sigma_lim_primary + f_s t_s / t_b        (strength = "tresca")

    The strength is the parallel sum of the two rings at their limits: a
    perfectly plastic primary ring (Tresca) keeps delivering ``sigma_lim_primary``
    while the secondary lining goes from zero to ``f_s``, so the band's
    plateau is the sum, per unit band thickness.  The stiffness is the
    secondary lining's alone, because a primary ring at its plateau has no
    tangent stiffness left; where the primary is still elastic (haunch, foot,
    invert in the S2 runs) this understates the band's tangent by the primary's
    share, which is the conservative side for the post-lining increments.
    ``strength = "elastic"`` gives the upgraded band no limit at all -- an
    upper bound on what a lining can hold.

    What the rule does not carry: the secondary lining's own bending stiffness
    (the band is axial-only, as the smeared primary ring is), the shear
    transfer between the two rings, and any loss of the primary's plateau at
    the strains the record implies (its plateau is whatever ``sigma_lim_primary``
    says).  ``primary`` is ``RunConfig.lining``; it must be a Tresca band for
    ``strength = "tresca"`` because an elastic primary has no plateau to add to.
    Returns ``{"material": spec, "rule": {...numbers and the formulae...}}``.
    """
    t_b = float(band_thickness_m)
    t_s = float(secondary["thickness_m"])
    e_s = float(secondary["E"])
    nu_s = float(secondary["nu"])
    if min(t_b, t_s, e_s) <= 0.0:
        raise ValueError("band thickness, secondary thickness and E must be > 0")
    scale = t_s / t_b
    e_band = e_s * scale
    rule = {"band_thickness_m": t_b, "secondary_thickness_m": t_s,
            "secondary_E_MPa": e_s, "secondary_nu": nu_s, "thickness_ratio": scale,
            "E_band_MPa": e_band, "stiffness": "E_band = E_s t_s / t_b (secondary lining alone)",
            "strength": strength}
    if strength == "tresca":
        if primary.get("model") != "tresca" or primary.get("sigma_lim") is None:
            raise ValueError("secondary_lining_band(strength='tresca') needs a Tresca primary "
                             "ring with sigma_lim: an elastic primary has no plateau to add to")
        f_s = float(secondary["sigma_lim"])
        if f_s <= 0.0:
            raise ValueError("secondary sigma_lim must be > 0")
        sig_p = float(primary["sigma_lim"])
        sig_band = sig_p + f_s * scale
        rule.update({"sigma_lim_primary_MPa": sig_p, "secondary_sigma_lim_MPa": f_s,
                     "sigma_lim_band_MPa": sig_band,
                     "strength_formula": "sigma_band = sigma_lim_primary + f_s t_s / t_b"})
        mat = {"model": "tresca", "E": e_band, "nu": nu_s, "sigma_lim": sig_band}
    elif strength == "elastic":
        rule.update({"strength_formula": "no limit: an elastic upgraded band"})
        mat = {"model": "elastic", "E": e_band, "nu": nu_s}
    else:
        raise ValueError("strength must be 'tresca' or 'elastic'")
    return {"material": mat, "rule": rule}


BOLT_KEYS = ("diameter_mm", "spacing_circumferential_m", "spacing_longitudinal_m",
             "arch_length_m", "wall_length_m", "steel_E_MPa", "steel_yield_MPa",
             "mobilisation")


def bolt_reinforcement(bolts: dict, rock: dict) -> dict:
    """Smeared confinement of a systematic radial bolt pattern, as a strength gain.

    A fully grouted bolt at circumferential spacing ``s_c`` and longitudinal spacing
    ``s_l`` can carry at most ``T_y = f_y A_s``; spread over the tributary area
    ``s_c s_l`` of one bolt that is an average radial confinement

        d_sigma_b = mobilisation * f_y * A_s / (s_c * s_l) .

    Adding a confinement ``d_sigma`` to the minor principal stress of the
    Mohr--Coulomb criterion ``sigma_1 = N sigma_3 + 2 c sqrt(N)`` is exactly a
    cohesion gain ``d_c = sqrt(N) d_sigma / 2``.  The reinforced rock is therefore
    the same material with ``c`` raised by that amount -- for CWFS, ``c_peak`` raised
    at the peak friction angle and ``c_res`` at the residual one, so both ends of the
    softening ramp are exact and the ramp between them stays linear in the softening
    variable, as the unreinforced law is.

    ``mobilisation`` is the fraction of the bar yield load the pattern is assumed to
    have picked up.  It is a DECLARED modelling parameter, not a measured one: no
    bolt load is instrumented at this site, so 1.0 is an upper bound on what the
    steel can give and not a statement that the bolts are yielding.

    What the representation does not carry: the axial stiffness of the steel (it acts
    along the bolt alone and is not an isotropic modulus), the distribution of axial
    force along a grouted bar and its neutral point, the anchorage length beyond the
    plastic front, and pull-out or grout failure.  It assumes the bolt axis is the
    minor principal direction, which holds near the wall of a yielded annulus and
    degrades away from it.  The reinforced annulus also ends in a strength step at
    the bolt length, which a band can localise against.  No bolt force is an output.

    ``steel_E_MPa`` is required and recorded but deliberately not applied.  Its
    smeared value ``E_s A_s / (s_c s_l)`` is what an isotropic modulus increase would
    have to stand in for, and it is not small next to a weak rock mass -- for the S2
    pattern (phi 32 at 1.2 x 0.6 m) it is 230 MPa against a rock modulus of 560 MPa.
    Applying it isotropically would stiffen the annulus tangentially, where the bolts
    contribute nothing, so it is left out; the displacements of a bolted annulus are
    therefore upper bounds, while the limit load, which is a strength question, is
    the quantity this representation is built for.
    """
    cfg = dict(bolts)
    cfg.pop("enabled", None)
    cfg.pop("source", None)
    missing = [k for k in BOLT_KEYS if cfg.get(k) is None]
    if missing:
        raise ValueError(f"bolt pattern is missing {missing}; every value must come from "
                         f"the section's own support record, none may be defaulted")
    unknown = [k for k in cfg if k not in BOLT_KEYS]
    if unknown:
        raise ValueError(f"unknown bolt keys {sorted(unknown)}")
    d = float(cfg["diameter_mm"]) * 1e-3
    area = math.pi * 0.25 * d * d                            # m2
    trib = float(cfg["spacing_circumferential_m"]) * float(cfg["spacing_longitudinal_m"])
    mob = float(cfg["mobilisation"])
    fy = float(cfg["steel_yield_MPa"])
    if min(d, trib, fy) <= 0.0 or mob < 0.0:
        raise ValueError("bolt diameter, spacings and yield must be > 0 and mobilisation >= 0")
    dsig = mob * fy * area / trib                            # MPa (MN/m2)
    spec = dict(rock)
    model = spec.get("model", "elastic")
    out = {"enabled": True, **{k: float(cfg[k]) for k in BOLT_KEYS},
           "source": bolts.get("source"),
           "steel_area_mm2": area * 1e6, "tributary_area_m2": trib,
           "bar_yield_load_kN": fy * area * 1e3,
           "confinement_MPa": dsig,
           "axial_stiffness_not_modelled_MPa": float(cfg["steel_E_MPa"]) * area / trib}
    if model in ("cwfs", "neural_cwfs"):
        dc_peak = _cohesion_gain(dsig, float(spec["phi_peak_deg"]))
        dc_res = _cohesion_gain(dsig, float(spec["phi_res_deg"]))
        spec["c_peak"] = float(spec["c_peak"]) + dc_peak
        spec["c_res"] = float(spec["c_res"]) + dc_res
        out["delta_c_peak_MPa"] = dc_peak
        out["delta_c_res_MPa"] = dc_res
    elif model == "mc":
        dc = _cohesion_gain(dsig, float(spec["phi_deg"]))
        spec["c"] = float(spec["c"]) + dc
        out["delta_c_MPa"] = dc
    else:
        raise ValueError(f"bolt reinforcement needs a frictional rock, got model '{model}'")
    out["reinforced_rock"] = spec
    return out


def _cohesion_gain(d_sigma: float, phi_deg: float) -> float:
    """``d_c = sqrt(N) d_sigma / 2`` with ``N = (1 + sin phi)/(1 - sin phi)``."""
    sp = math.sin(math.radians(float(phi_deg)))
    return 0.5 * math.sqrt((1.0 + sp) / (1.0 - sp)) * float(d_sigma)


# ---------------------------------------------------------------------------
# T6 element kinematics
# ---------------------------------------------------------------------------

GAUSS_T6 = (np.array([[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]]),
            np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]))


def t6_shape(xi, eta):
    """Shape functions and (xi, eta)-gradients of the 6-node triangle, gmsh order."""
    L1, L2, L3 = 1.0 - xi - eta, xi, eta
    N = np.array([L1 * (2 * L1 - 1), L2 * (2 * L2 - 1), L3 * (2 * L3 - 1),
                  4 * L1 * L2, 4 * L2 * L3, 4 * L3 * L1])
    dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    dN = np.array([(4 * L1 - 1) * dL[0], (4 * L2 - 1) * dL[1], (4 * L3 - 1) * dL[2],
                   4 * (L2 * dL[0] + L1 * dL[1]), 4 * (L3 * dL[1] + L2 * dL[2]),
                   4 * (L1 * dL[2] + L3 * dL[0])])
    return N, dN


def edge3_shape(xi):
    """Quadratic edge (end, end, mid) shape functions at xi in [-1, 1]."""
    return np.array([0.5 * xi * (xi - 1.0), 0.5 * xi * (xi + 1.0), 1.0 - xi * xi])


class Kinematics:
    """Precomputed B matrices and weights for all elements."""

    def __init__(self, mesh: MeshData):
        nodes, elems = mesh.nodes, mesh.elems
        pts, w = GAUSS_T6
        n_el, n_ip = elems.shape[0], pts.shape[0]
        self.n_el, self.n_ip = n_el, n_ip
        X = nodes[elems]                                    # (n_el, 6, 2)
        self.B = np.zeros((n_el, n_ip, 4, 12))
        self.wdet = np.zeros((n_el, n_ip))
        self.ip_xy = np.zeros((n_el, n_ip, 2))
        for q in range(n_ip):
            N, dN = t6_shape(*pts[q])                       # dN (6, 2)
            J = np.einsum("eia,ib->eab", X, dN)             # (n_el, 2, 2): J[a,b] = d x_a / d xi_b
            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            if np.any(detJ <= 0):
                raise RuntimeError(f"{int((detJ <= 0).sum())} elements with non-positive Jacobian")
            Jinv = np.empty_like(J)
            Jinv[:, 0, 0] = J[:, 1, 1] / detJ
            Jinv[:, 1, 1] = J[:, 0, 0] / detJ
            Jinv[:, 0, 1] = -J[:, 0, 1] / detJ
            Jinv[:, 1, 0] = -J[:, 1, 0] / detJ
            dNdx = np.einsum("ib,eba->eia", dN, Jinv)       # dN_i/dx_a = dN_i/dxi_b * dxi_b/dx_a
            self.B[:, q, 0, 0::2] = dNdx[:, :, 0]
            self.B[:, q, 1, 1::2] = dNdx[:, :, 1]
            self.B[:, q, 3, 0::2] = dNdx[:, :, 1]
            self.B[:, q, 3, 1::2] = dNdx[:, :, 0]
            self.wdet[:, q] = w[q] * detJ
            self.ip_xy[:, q] = np.einsum("i,eia->ea", N, X)
        dofs = np.empty((n_el, 12), dtype=np.int64)
        dofs[:, 0::2] = 2 * elems
        dofs[:, 1::2] = 2 * elems + 1
        self.dofs = dofs
        self.rows = np.repeat(dofs, 12, axis=1)             # (n_el, 144)
        self.cols = np.tile(dofs, (1, 12))
        self.area = self.wdet.sum(axis=1)

    def strain(self, u, el_idx):
        """Tension-positive strain (n_sel, n_ip, 4) of elements ``el_idx``."""
        ue = u[self.dofs[el_idx]]                            # (n_sel, 12)
        return np.einsum("eqkl,el->eqk", self.B[el_idx], ue)

    def internal_force(self, sig_c, el_idx, n_dof):
        """f_int = sum int B^T (-sig_c) over elements el_idx (sig_c compression positive)."""
        fe = -np.einsum("eq,eqkl,eqk->el", self.wdet[el_idx], self.B[el_idx], sig_c)
        f = np.zeros(n_dof)
        np.add.at(f, self.dofs[el_idx].ravel(), fe.ravel())
        return f

    def stiffness_entries(self, D, el_idx):
        """Element stiffness matrices (n_sel, 12, 12) from tangents D (n_sel, n_ip, 4, 4)."""
        B = self.B[el_idx]
        DB = np.einsum("eqkm,eqml->eqkl", D, B)
        return np.einsum("eq,eqkl,eqkm->elm", self.wdet[el_idx], B, DB)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class SolverError(RuntimeError):
    pass


class StagedModel:
    def __init__(self, cfg: RunConfig, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.t_build = time.perf_counter()
        self._build_geometry()
        self._build_mesh()
        self._build_fields()
        self.t_build = time.perf_counter() - self.t_build

    # -- construction -------------------------------------------------------
    def _build_geometry(self):
        cfg = self.cfg
        if cfg.geometry == "cad_reference":
            self.hp = None
            self.reference_prims = cad_reference_profile(cfg.cad_section["geometry"])
            if cfg.cad_section.get("reference_boundary") != "initial_support_inner":
                raise ValueError("CAD width belongs to the primary-support inner boundary")
            t = float(cfg.lining["thickness"])
            if t <= 0 or not cfg.lining.get("enabled"):
                raise ValueError("The CAD S2 calculation requires its primary support ring")
            self.inner_prims = self.reference_prims
            self.exc_prims = offset_profile(self.reference_prims, t)
            self.cut_levels = list(cfg.cad_section["cut_levels_y_m"])
            if (len(self.cut_levels) != 2 or self.cut_levels != sorted(self.cut_levels, reverse=True)
                    or not all(profile_bottom(self.inner_prims)[1] < y < 0 for y in self.cut_levels)):
                raise ValueError("Two ordered construction cuts must cross the reference interior")
            self.lining_thickness = t
        elif cfg.geometry == "horseshoe":
            hp = HorseshoeParams(**cfg.horseshoe)
            self.hp = hp
            net = horseshoe_net_profile(hp)
            self.exc_prims = offset_profile(net, hp.offset, hp.snap_deg)
            self.inner_prims = (offset_profile(net, hp.reserved_deformation, hp.snap_deg)
                                if cfg.lining.get("enabled", True) and hp.lining_thickness > 0 else None)
            self.cut_levels = [-hp.bench_depth, -hp.base_depth]
            self.lining_thickness = hp.lining_thickness if self.inner_prims is not None else 0.0
        elif cfg.geometry == "circle":
            a = float(cfg.circle_radius)
            self.hp = None
            self.exc_prims = circle_profile(a, cy=-a)      # crown at (0, 0) like the horseshoe
            t = float(cfg.lining.get("thickness", 0.25))
            self.inner_prims = circle_profile(a - t, cy=-a) if cfg.lining.get("enabled", False) else None
            self.cut_levels = list(cfg.lining.get("cut_levels", []))
            self.lining_thickness = t if self.inner_prims is not None else 0.0
        else:
            raise ValueError("geometry must be 'cad_reference', 'horseshoe' or 'circle'")
        self.exc_curve = ProfileCurve(self.exc_prims)
        self.y_top = profile_top(self.exc_prims)[1]
        self.y_bot = profile_bottom(self.exc_prims)[1]

    def _build_mesh(self):
        m = self.cfg.mesh
        self.mesh = build_mesh(self.exc_prims, self.inner_prims, self.cut_levels,
                               r_far=float(m["r_far"]), h_wall=float(m["h_wall"]),
                               h_far=m.get("h_far"), h_ring=m.get("h_ring"),
                               dist_min=m.get("dist_min"), growth_length=m.get("growth_length"),
                               algorithm=int(m.get("algorithm", 6)))
        self.kin = Kinematics(self.mesh)
        self.n_dof = 2 * self.mesh.n_nodes

    def _build_fields(self):
        cfg, mesh, kin = self.cfg, self.mesh, self.kin
        self.rock: Material = make_material(cfg.rock)
        self.lining: Optional[Material] = None
        if self.inner_prims is not None:
            self.lining = lining_material(cfg.lining)
        self.bolts = (bolt_reinforcement(cfg.bolts, cfg.rock)
                      if cfg.bolts.get("enabled") else None)
        self.bolted: Optional[Material] = (make_material(self.bolts["reinforced_rock"])
                                           if self.bolts is not None else None)
        # the ring-upgrade event (secondary lining): validated here, fired in run()
        self.ring_upgrade = ring_upgrade_spec(cfg.ring_upgrade, len(cfg.stages), cfg.stages)
        self.upgraded: Optional[Material] = None
        if self.ring_upgrade is not None:
            if self.lining is None:
                raise ValueError("ring_upgrade needs a lining ring to upgrade")
            self.upgraded = make_material(self.ring_upgrade["material"])
        self.upgrade_applied = None          # {"stage", "lambda", "n_ring_elements"} once fired
        self.u_upgrade = None
        self.materials = {0: self.rock, 1: self.lining, 2: self.bolted, 3: self.upgraded}
        n_state = max(self.rock.n_state, self.lining.n_state if self.lining else 0,
                      self.bolted.n_state if self.bolted else 0,
                      self.upgraded.n_state if self.upgraded else 0, 1)
        n_el, n_ip = kin.n_el, kin.n_ip
        sv = float(cfg.stress["sigma_v"])
        sh = float(cfg.stress["K"]) * sv
        sz = float(cfg.stress["sigma_z"])
        self.sigma0 = np.array([sh, sv, sz, 0.0])
        self.sig = np.tile(self.sigma0, (n_el, n_ip, 1))
        self.state = np.zeros((n_el, n_ip, n_state))
        self.state[:, :, :self.rock.n_state] = self.rock.initial_state(n_el * n_ip).reshape(n_el, n_ip, -1)
        self.elem_mat = np.zeros(n_el, dtype=np.int8)
        self.active = np.ones(n_el, dtype=bool)
        self.installed = np.zeros(n_el, dtype=bool)
        self.u = np.zeros(self.n_dof)
        # boundary conditions
        fixed = np.zeros(self.n_dof, dtype=bool)
        fixed[2 * mesh.outer_nodes] = True
        fixed[2 * mesh.outer_nodes + 1] = True
        fixed[2 * mesh.axis_nodes] = True
        self.fixed = fixed
        # discrete far-field load
        self.F0 = kin.internal_force(self.sig, np.arange(n_el), self.n_dof)
        self.F_ext = self.F0.copy()
        # release force left unreleased by the stages solved so far (see run())
        self.fR_carry = np.zeros(self.n_dof)
        self._refresh_dofs()
        # integration-point geometry relative to the excavation line
        xy = kin.ip_xy.reshape(-1, 2)
        s, dist, proj, nrm = self.exc_curve.project(xy)
        self.ip_s = s.reshape(n_el, n_ip)
        self.ip_dist = dist.reshape(n_el, n_ip)
        self.ip_nrm = nrm.reshape(n_el, n_ip, 2)
        self._assign_bolt_annulus(proj.reshape(n_el, n_ip, 2))
        self._write_characteristic_length()
        self.u_install = {}
        self.stage_install_lambda = {}
        self.stage_install_convergence = {}
        self.rings_not_installed = []
        self.history = []
        self.increments = []
        self.increment_attempts = []
        # pseudo-time increment of the current load step, handed to the materials.
        # Pseudo-time IS the release factor lambda, so dt = time_scale * d lambda;
        # only rate-regularised materials look at it (see fe_materials).
        self.dt = 0.0
        self._define_readings()

    def _assign_bolt_annulus(self, ip_proj):
        """Rock elements inside the bolted annulus and the stage that installs them.

        The bolts are drilled normal to the excavation line, so an element is inside
        the annulus when its mean distance from that line is within the bolt length at
        the profile point it projects onto (arch above the bench line, wall below it),
        and it is installed with the ring of the stage that contains that point.  The
        annulus boundary is therefore resolved to the element size; the realised
        annulus is reported with the run.
        """
        n_el = self.kin.n_el
        self.bolt_elems = np.zeros(n_el, dtype=bool)
        self.bolt_stage = np.zeros(n_el, dtype=np.int8)
        self.bolt_installed = np.zeros(n_el, dtype=bool)
        if self.bolted is None:
            return
        y_proj = ip_proj[:, :, 1].mean(axis=1)
        d_el = self.ip_dist.mean(axis=1)
        split = self.cut_levels[0] if self.cut_levels else -np.inf
        L = np.where(y_proj >= split, float(self.bolts["arch_length_m"]),
                     float(self.bolts["wall_length_m"]))
        self.bolt_elems = (self.mesh.elem_kind == 0) & (d_el > 0.0) & (d_el <= L)
        stage = np.array([self._stage_of_y(y) for y in y_proj], dtype=np.int8)
        self.bolt_stage = np.where(self.bolt_elems, stage, 0).astype(np.int8)

    def _write_characteristic_length(self):
        """Give every crack-band point its own band width, and report the spread.

        The characteristic length of an integration point is the square root of
        its own tributary area, ``h = sqrt(w_q det J)``.  For a six-node triangle
        with three Gauss points that is ``sqrt(A_e / 3)`` -- about 0.38 h_wall on
        a near-equilateral mesh -- and it is the area the point's dissipation is
        actually integrated over, so it is the width a localised band occupies
        when the band collapses onto one row of integration points.  Taking the
        element instead would overstate the band by sqrt(3) and would ignore mesh
        grading, which this measure follows exactly because it is read from the
        point's own weight.

        What it cannot carry: it is a SCALAR with no direction, so a band
        inclined to the element edges is resolved no better than the tributary
        area itself, and a strongly stretched element is represented by an
        equivalent square.
        """
        self.h_char = np.sqrt(self.kin.wdet)                 # (n_el, n_ip), metres
        hc = self.h_char.ravel()
        # the rock within 2 m of the excavation line: the band that localises.  Ring
        # and core points are excluded -- the ring is not a CWFS material and the core
        # is excavated away, so neither carries a softening band.
        rock_ip = np.repeat(self.mesh.elem_kind == 0, self.kin.n_ip)
        near = rock_ip & (self.ip_dist.ravel() <= 2.0)

        def stats(v):
            return {"min": float(v.min()), "median": float(np.median(v)),
                    "mean": float(v.mean()), "max": float(v.max()), "n": int(v.size)}

        # reported for EVERY run, regularised or not: a plastic shear strain is a
        # strain in a band of this width, so it is only comparable across meshes once
        # multiplied by it
        self.characteristic_length = {
            "definition": ("h = sqrt(w_q det J), the square root of the integration "
                           "point's tributary area; A_e/3 for a T6 with three Gauss "
                           "points"),
            "h_char_m": stats(hc),
            "h_char_in_rock_within_2m_of_the_excavation_line_m": stats(hc[near]),
        }
        self.crack_band = None
        mats = [m for m in (self.rock, self.bolted)
                if getattr(m, "crack_band", None) is not None]
        if not mats:
            return
        cols = {m.state_names.index("h_char") for m in mats}
        if len(cols) != 1:
            raise ValueError("rock and bolted rock disagree on the h_char state column")
        col = cols.pop()
        self.state[:, :, col] = self.h_char
        h_ref = float(mats[0].crack_band["h_ref"])
        self.crack_band = {
            "h_ref_m": h_ref,
            "scale_phi": bool(mats[0].crack_band["scale_phi"]),
            "w_c_m": float(mats[0].crack_band["w_c"]),
            "w_phi_m": (None if mats[0].crack_band["w_phi"] is None
                        else float(mats[0].crack_band["w_phi"])),
            "state_column": int(col),
            "definition": self.characteristic_length["definition"],
            "h_char_m": self.characteristic_length["h_char_m"],
            "h_char_in_rock_within_2m_of_the_excavation_line_m":
                self.characteristic_length[
                    "h_char_in_rock_within_2m_of_the_excavation_line_m"],
            "threshold_scale_h_ref_over_h": stats(h_ref / hc),
            "threshold_scale_near_excavation": stats(h_ref / hc[near]),
        }

    def _refresh_dofs(self):
        self._dof_version = getattr(self, "_dof_version", 0) + 1
        self._elastic_lu_cache = None
        act_nodes = np.zeros(self.mesh.n_nodes, dtype=bool)
        act_nodes[np.unique(self.mesh.elems[self.active])] = True
        act_dof = np.repeat(act_nodes, 2) & ~self.fixed
        self.free = np.nonzero(act_dof)[0]
        self.dof_map = np.full(self.n_dof, -1, dtype=np.int64)
        self.dof_map[self.free] = np.arange(self.free.shape[0])
        self.active_idx = np.nonzero(self.active)[0]
        rows = self.dof_map[self.kin.rows[self.active_idx]]
        cols = self.dof_map[self.kin.cols[self.active_idx]]
        self._entry_mask = (rows >= 0) & (cols >= 0)
        self._rows = rows[self._entry_mask]
        self._cols = cols[self._entry_mask]

    # -- readings definitions -------------------------------------------------
    def _stage_of_y(self, y):
        k = 1
        for y0 in self.cut_levels:
            if y < y0 - 1e-9:
                k += 1
        return k

    def _define_readings(self):
        cfg = self.cfg
        exc = self.exc_prims
        boundary = cfg.readings.get("boundary", "excavation")
        if boundary not in ("excavation", "initial_support_inner"):
            raise ValueError("Unrecognized observation boundary")
        profile = self.inner_prims if boundary == "initial_support_inner" else exc
        if profile is None:
            raise ValueError("Observation boundary does not exist")
        self.reading_boundary = boundary
        self.points = {}
        self.points["crown"] = {"xy": profile_top(profile), "stage": 1,
                                "kind": "settlement", "channel": "GD", "boundary": boundary}
        depths = list(cfg.readings.get("chord_depths", []))
        self.chord_specs = []
        for i, dep in enumerate(depths):
            if dep is None:
                if self.hp is None:
                    continue
                dep = self.hp.base_depth - 0.5
            y = -float(dep)                                   # depths below the net crown (y = 0)
            if not (profile_bottom(profile)[1] < y < profile_top(profile)[1]):
                raise ValueError(f"Chord {i+1} at y={y} is outside {boundary}")
            pt = point_at_y(profile, y)
            name = f"chord_{i + 1}"
            self.points[name] = {"xy": pt, "stage": self._stage_of_y(y), "kind": "chord",
                                 "depth": float(dep), "channel": f"SL{i+1}", "boundary": boundary}
            self.chord_specs.append(name)
        # lining / plastic stations
        self.stations = {}
        crown = np.array([0.0, self.y_top])
        self.stations["crown"] = crown
        if cfg.geometry == "cad_reference":
            depths = cfg.cad_section["station_depths_m"]
            for name in ("shoulder", "haunch", "foot"):
                self.stations[name] = np.array(point_at_y(exc, -float(depths[name])))
        elif self.hp is not None:
            hp = self.hp
            c1 = np.array([0.0, -hp.crown_radius])
            self.stations["shoulder"], _ = self.exc_curve.ray_intersection(
                c1, np.array([math.sin(math.radians(45.0)), math.cos(math.radians(45.0))]))
            self.stations["haunch"] = np.array(point_at_y(exc, -hp.bench_depth))
            self.stations["foot"] = np.array(point_at_y(exc, -hp.base_depth + 0.5))
        else:
            a = float(cfg.circle_radius)
            c1 = np.array([0.0, -a])
            self.stations["shoulder"] = c1 + a * np.array([math.sin(math.radians(45.0)),
                                                           math.cos(math.radians(45.0))])
            self.stations["haunch"] = c1 + a * np.array([1.0, 0.0])
            self.stations["foot"] = c1 + a * np.array([math.sin(math.radians(135.0)),
                                                       math.cos(math.radians(135.0))])
        self.stations["invert"] = np.array([0.0, self.y_bot])
        self.station_s = {}
        for name, pt in self.stations.items():
            s, _, _, _ = self.exc_curve.project(np.asarray(pt, float)[None, :])
            self.station_s[name] = float(s[0])
        # excavation-line edges for point interpolation
        self.exc_edges = self.mesh.exc_edges_all()
        self.reading_edges = (np.vstack(list(self.mesh.inner_edges.values()))
                              if boundary == "initial_support_inner" else self.exc_edges)
        # excavation-line nodes with their outward normal and stage, for the
        # convergence criterion of a closing install gap
        nd = np.unique(self.exc_edges)
        xy = self.mesh.nodes[nd]
        _, _, _, nrm = self.exc_curve.project(xy)
        self._exc_nodes = (nd, nrm,
                           np.array([self._stage_of_y(y) for y in xy[:, 1]], dtype=np.int8))

    # -- evaluation helpers ---------------------------------------------------
    def eval_u_on_exc_line(self, xy, u=None):
        """Displacement at a point of the excavation line by quadratic edge interpolation."""
        return self.eval_u_on_edges(xy, self.exc_edges, u)

    def eval_u_on_edges(self, xy, edges, u=None):
        """Quadratic displacement interpolation on an explicitly identified boundary."""
        u = self.u if u is None else u
        nodes = self.mesh.nodes
        E = edges
        P = nodes[E]                                          # (n_edge, 3, 2)
        target = np.asarray(xy, float)
        xis = np.linspace(-1.0, 1.0, 21)
        best = (np.inf, -1, 0.0)
        for xi in xis:
            Nq = edge3_shape(xi)
            q = np.einsum("i,eia->ea", Nq, P)
            d = np.sum((q - target) ** 2, axis=1)
            j = int(np.argmin(d))
            if d[j] < best[0]:
                best = (d[j], j, xi)
        _, j, xi = best
        Pe = P[j]
        for _ in range(20):                                   # Newton on the edge parameter
            Nq = edge3_shape(xi)
            dN = np.array([xi - 0.5, xi + 0.5, -2.0 * xi])
            q = Nq @ Pe
            dq = dN @ Pe
            ddq = np.array([1.0, 1.0, -2.0]) @ Pe
            r = q - target
            g = r @ dq
            h = dq @ dq + r @ ddq
            if abs(h) < 1e-30:
                break
            step = -g / h
            xi = float(np.clip(xi + step, -1.0, 1.0))
            if abs(step) < 1e-12:
                break
        Nq = edge3_shape(xi)
        ue = u[2 * E[j]] , u[2 * E[j] + 1]
        return np.array([Nq @ ue[0], Nq @ ue[1]]), float(np.hypot(*(Nq @ Pe - target)))

    def exc_convergence(self, regions=None, u=None):
        """Largest INWARD normal displacement of the excavation line (m).

        ``self.exc_curve``'s normal points outward into the rock, so the inward
        convergence of a node is ``-u . n``.  ``regions`` restricts the nodes to
        the stages given (the profile the stage's own ring faces); ``None`` uses
        the whole line.  This is the quantity a radial closing gap has to reach
        before the ring can carry, and the maximum is the earliest instant at
        which any part of the profile has come into contact.
        """
        u = self.u if u is None else u
        nd, nrm, stage = self._exc_nodes
        sel = (np.ones(nd.shape[0], dtype=bool) if regions is None
               else np.isin(stage, np.asarray(list(regions), dtype=np.int8)))
        if not sel.any():
            return 0.0
        n = nd[sel]
        uu = np.column_stack([u[2 * n], u[2 * n + 1]])
        return float((-(uu * nrm[sel]).sum(axis=1)).max())

    def readings(self, u=None):
        u = self.u if u is None else u
        out = {}
        for name, spec in self.points.items():
            ux, uy = self.eval_u_on_edges(spec["xy"], self.reading_edges, u)[0]
            if spec["kind"] == "settlement":
                out[name] = -uy
            else:
                out[name] = -2.0 * ux
        return out

    def readings_post_install(self):
        tot = self.readings()
        out = {}
        for name, spec in self.points.items():
            k = spec["stage"]
            if k in self.u_install:
                base = self.readings(self.u_install[k])[name]
                out[name] = tot[name] - base
            else:
                out[name] = None
        return tot, out

    def _ring_materials(self):
        """``(material id, material)`` of the ring band's materials that exist."""
        return [(mid, m) for mid, m in ((1, self.lining), (3, self.upgraded)) if m is not None]

    def _ring_at_limit(self, ridx):
        """Per integration point of ``ridx``: returned to its OWN material's yield
        surface at the last converged update.  ``None`` when no ring material
        has a limit (an elastic band has no state and nothing to reach)."""
        mats = [(mid, m) for mid, m in self._ring_materials() if m.n_state >= 2]
        if not mats:
            return None
        n_ip = self.kin.n_ip
        at = np.zeros(ridx.shape[0] * n_ip, dtype=bool)
        region = self.state[ridx, :, 1].ravel()
        for mid, _ in mats:
            sel = np.repeat(self.elem_mat[ridx] == mid, n_ip)
            at[sel] = region[sel] > 0.5
        return at

    def lining_stations(self):
        """Interface pressure, mean hoop stress and yielded fraction per station.

        ``fraction_at_limit`` is the area-weighted fraction of the station window's
        ring integration points whose last converged update returned to the yield
        surface of the material they carry (primary or upgraded); it is ``None``
        when no ring material has state (an elastic band has no limit to reach).
        Everything is ``None`` where no ring is installed.
        """
        w = float(self.cfg.readings.get("station_window", 0.5))
        out = {}
        ring = self.installed & self.active
        empty = {"pressure": None, "hoop_mean": None, "fraction_at_limit": None}
        if not ring.any():
            return {n: dict(empty) for n in self.stations}
        ridx = np.nonzero(ring)[0]
        s = self.ip_s[ridx].ravel()
        dist = self.ip_dist[ridx].ravel()
        nrm = self.ip_nrm[ridx].reshape(-1, 2)
        sig = self.sig[ridx].reshape(-1, 4)
        wq = self.kin.wdet[ridx].ravel()
        at_limit = self._ring_at_limit(ridx)
        nx, ny = nrm[:, 0], nrm[:, 1]
        tx, ty = -ny, nx
        s_nn = sig[:, 0] * nx * nx + sig[:, 1] * ny * ny + 2.0 * sig[:, 3] * nx * ny
        s_tt = sig[:, 0] * tx * tx + sig[:, 1] * ty * ty + 2.0 * sig[:, 3] * tx * ty
        for name, s0 in self.station_s.items():
            sel = np.abs(s - s0) <= w
            if sel.sum() < 3:
                out[name] = dict(empty)
                continue
            z = -dist[sel]                                    # 0 at the interface, t at the inner face
            A = np.column_stack([np.ones(sel.sum()), z])
            W = np.sqrt(wq[sel])
            coef, *_ = np.linalg.lstsq(A * W[:, None], s_nn[sel] * W, rcond=None)
            out[name] = {"pressure": float(coef[0]),
                         "hoop_mean": float(np.sum(s_tt[sel] * wq[sel]) / np.sum(wq[sel])),
                         "fraction_at_limit":
                             (None if at_limit is None
                              else float(wq[sel][at_limit[sel]].sum() / wq[sel].sum())),
                         "n_ip": int(sel.sum())}
        return out

    def plastic_zone(self):
        """Plastic-zone depth per station, the peak plastic shear strain, and the
        peak plastic SLIP across the band.

``max_gamma_p`` is a STRAIN in a band one integration-point strip wide, so
        it is not comparable across meshes on its own: halving the element size
        roughly doubles it whether or not the softening law is regularised.  Two
        mesh-comparable measures of the same thing are reported beside it.

        ``max_plastic_slip_m = max_gamma_p * h_char`` is the plastic displacement
        across the band at the hottest point.  It has the right units but it is
        read at ONE point, and which point that is depends on where the mesher put
        its smallest element -- a local minimum that does not scale with the
        nominal mesh size -- so it is noisy.

        ``gamma_p_integral_m2 = sum gamma_p * w_q`` over the active rock is the
        robust one: for a band of width ``h`` and length ``L`` carrying a slip
        ``u`` it is ``(u/h) (h L) = u L``, the slip times the band length, with no
        ``h`` left in it.  It integrates instead of sampling, so no single element
        can dominate it.

        The depths are the largest normal distance of a plastic integration point,
        so their own resolution is the point spacing.
        """
        band = float(self.cfg.readings.get("plastic_band", 0.5))
        rock = self.active & ((self.elem_mat == 0) | (self.elem_mat == 2))
        ridx = np.nonzero(rock)[0]
        n_pts = ridx.shape[0] * self.kin.n_ip
        st = self.state[ridx, :, :self.rock.n_state].reshape(n_pts, self.rock.n_state)
        pm = self.rock.plastic_mask(st) if self.rock.n_state else np.zeros(n_pts, bool)
        gp = self.rock.equivalent_plastic_strain(st) if self.rock.n_state else np.zeros(n_pts)
        s = self.ip_s[ridx].ravel()
        dist = self.ip_dist[ridx].ravel()
        h_ip = self.h_char[ridx].ravel() if ridx.size else np.zeros(0)
        w_ip = self.kin.wdet[ridx].ravel() if ridx.size else np.zeros(0)
        i_max = int(np.argmax(gp)) if gp.size else None
        yielding = pm & (dist > 0)
        out = {"depth": {}, "max_gamma_p": float(gp.max()) if gp.size else 0.0,
               "depth_max_m": (float(dist[yielding].max()) if yielding.any() else 0.0),
               "depth_max_note": ("the largest normal distance of ANY yielding "
                                  "integration point, over the whole excavation line "
                                  "rather than in the four station bands: it is the "
                                  "depth a mesh has to stay adequate to"),
               "snap_back": self._snap_back_exposure(ridx, pm, h_ip, w_ip, dist),
               "max_plastic_slip_m": (float(gp[i_max] * h_ip[i_max])
                                      if i_max is not None else 0.0),
               "h_char_at_max_gamma_p_m": (float(h_ip[i_max]) if i_max is not None
                                           else None),
               "gamma_p_integral_m2": float(np.sum(gp * w_ip)) if gp.size else 0.0,
               "plastic_area_m2": float(w_ip[pm].sum()) if gp.size else 0.0,
               "slip_note": ("max_plastic_slip_m is gamma_p times the band width at the "
                             "hottest point -- right units, one sample; "
                             "gamma_p_integral_m2 is sum gamma_p * w_q over the rock, "
                             "which for a band of width h, length L and slip u is u L "
                             "with no h in it, and is the robust mesh-comparable measure"),
               "n_plastic_ip": int(pm.sum()), "n_rock_ip": int(st.shape[0])}
        for name in ("crown", "haunch", "foot", "invert"):
            s0 = self.station_s[name]
            sel = pm & (np.abs(s - s0) <= band) & (dist > 0)
            out["depth"][name] = float(dist[sel].max()) if sel.any() else 0.0
        return out

    def _snap_back_exposure(self, ridx, pm, h_ip, w_ip, dist):
        """How much of the YIELDING rock sits in bands wider than ``h_snap``.

        ``fe_materials.CWFSMaterial.snap_back_limit`` gives the width above which
        the softening law releases more elastic strain than the plastic strain it
        softens over.  It is a mesh criterion -- no element carrying yielding rock
        should have a characteristic length above it -- and the model can MEASURE
        compliance instead of assuming it, because every integration point already
        carries its own characteristic length.

        The mask is the points that are actually yielding NOW, in the rock, at the
        state the run reached.  That is the population the criterion is about: a
        coarse element in elastic rock never has to dissipate a band's energy, so
        the domain-wide area fraction -- which on a graded mesh out to r_far is
        dominated by far field that never yields -- is not the measure.

        ``None`` when the rock is not a softening material with a snap-back
        length (see :meth:`~cwfs_inv.fe_materials.CWFSMaterial.snap_back_limit`).
        """
        lim = {}
        for code, mat in ((0, self.rock), (2, self.bolted)):
            fn = getattr(mat, "snap_back_limit", None)
            if fn is not None:
                lim[code] = fn()
        if not lim:
            return None
        h_snap = np.full(h_ip.shape, np.inf)
        mat_of = np.repeat(self.elem_mat[ridx], self.kin.n_ip)
        for code, d in lim.items():
            v = d["h_snap_m"]
            h_snap[mat_of == code] = np.inf if v is None else float(v)
        over = pm & (h_ip > h_snap)
        n_p = int(pm.sum())
        w_p = float(w_ip[pm].sum()) if n_p else 0.0
        return {
            "rock": lim[0],
            "bolted_rock": lim.get(2),
            "n_plastic_ip": n_p,
            "n_plastic_ip_above_h_snap": int(over.sum()),
            "fraction_of_plastic_ip_above_h_snap": (float(over.sum() / n_p) if n_p else 0.0),
            "fraction_of_plastic_area_above_h_snap": (float(w_ip[over].sum() / w_p)
                                                      if w_p > 0.0 else 0.0),
            "max_h_char_at_a_plastic_ip_m": (float(h_ip[pm].max()) if n_p else None),
            "median_h_char_at_a_plastic_ip_m": (float(np.median(h_ip[pm])) if n_p else None),
            "deepest_plastic_ip_above_h_snap_m": (float(dist[over].max()) if over.any()
                                                  else None),
            "shallowest_plastic_ip_above_h_snap_m": (float(dist[over].min()) if over.any()
                                                     else None),
            "compliant": bool(n_p == 0 or not over.any()),
            "note": ("the fraction of the points that are YIELDING whose own "
                     "characteristic length exceeds the material's snap-back limit; "
                     "the population is the yielding rock, not the whole domain"),
        }

    def bolt_annulus_info(self) -> dict:
        """The annulus the mesh actually realises, and how much of it is installed."""
        if self.bolted is None:
            return {}
        el = np.nonzero(self.bolt_elems)[0]
        area = float(self.kin.wdet[el].sum()) if el.size else 0.0
        d = self.ip_dist[el].ravel() if el.size else np.zeros(0)
        return {"n_annulus_elements": int(el.size),
                "n_annulus_installed": int(self.bolt_installed.sum()),
                "annulus_area_m2_half_section": area,
                "annulus_max_distance_m": float(d.max()) if d.size else 0.0,
                "n_annulus_elements_per_stage":
                    {str(k): int((self.bolt_stage == k).sum()) for k in (1, 2, 3)
                     if (self.bolt_stage == k).any()}}

    def lining_yield_state(self) -> dict:
        """How much of the installed ring is at its strength limit.

        ``fraction_at_limit`` is the area-weighted fraction of the installed
        ring's integration points whose LAST CONVERGED update returned to the
        yield surface (state column ``region`` > 0), i.e. that are carrying their
        limit stress now; ``fraction_yielded_ever`` is the area-weighted fraction
        that has ever yielded (accumulated plastic strain > 0), which also counts
        points that have since unloaded.  Both are ``None`` for an elastic ring,
        which has no state and cannot reach a limit.
        """
        ring = self.installed & self.active
        n_ip = int(ring.sum()) * self.kin.n_ip
        out = {"fraction_at_limit": None, "fraction_yielded_ever": None,
               "n_ring_integration_points": n_ip, "n_at_limit": None,
               "max_gamma_p": None}
        if self.lining is None or not ring.any():
            return out
        ridx = np.nonzero(ring)[0]
        at = self._ring_at_limit(ridx)
        if at is None:
            out["note"] = ("the ring material has no state: a linear elastic band has no "
                           "strength limit to reach")
            return out
        w = self.kin.wdet[ridx].ravel()
        ever = np.zeros(at.shape[0], dtype=bool)
        gp = np.zeros(at.shape[0])
        for mid, mat in self._ring_materials():
            if mat.n_state == 0:
                continue
            sel = np.repeat(self.elem_mat[ridx] == mid, self.kin.n_ip)
            if not sel.any():
                continue
            st = self.state[ridx, :, :mat.n_state].reshape(-1, mat.n_state)[sel]
            ever[sel] = mat.plastic_mask(st)
            gp[sel] = mat.equivalent_plastic_strain(st)
        wsum = float(w.sum())
        out.update({"fraction_at_limit": float(w[at].sum() / wsum),
                    "fraction_yielded_ever": float(w[ever].sum() / wsum),
                    "n_at_limit": int(at.sum()),
                    "max_gamma_p": float(gp.max())})
        if self.upgraded is not None:
            up = np.repeat(self.elem_mat[ridx] == 3, self.kin.n_ip)
            out["upgraded_fraction_of_ring_area"] = float(w[up].sum() / wsum)
            out["upgraded_fraction_at_limit"] = (float(w[up & at].sum() / w[up].sum())
                                                 if up.any() else None)
        return out

    def lining_stress_extrema(self):
        ring = self.installed & self.active
        if not ring.any():
            return {"max_abs_stress": 0.0, "max_abs_pressure_fit": 0.0}
        sig = self.sig[ring].reshape(-1, 4)
        return {"max_abs_stress": float(np.abs(sig).max())}

    # -- assembly / solve -----------------------------------------------------
    def _material_update(self, du):
        """Update all active integration points for the trial increment du."""
        kin = self.kin
        sig_new = self.sig.copy()
        state_new = self.state.copy()
        D = np.zeros((self.active_idx.shape[0], kin.n_ip, 4, 4))
        info = {}
        act = self.active_idx
        deps_all = -kin.strain(du, act)                        # compression positive
        for mid, mat in self.materials.items():
            if mat is None:
                continue
            sel = np.nonzero(self.elem_mat[act] == mid)[0]
            if sel.size == 0:
                continue
            el = act[sel]
            ns = mat.n_state
            n_pts = el.shape[0] * kin.n_ip
            upd = mat.update(deps_all[sel].reshape(n_pts, 4), self.sig[el].reshape(n_pts, 4),
                             self.state[el, :, :ns].reshape(n_pts, ns), self.dt)
            sig_new[el] = upd.sig.reshape(-1, kin.n_ip, 4)
            if ns:
                state_new[el, :, :ns] = upd.state.reshape(-1, kin.n_ip, ns)
            D[sel] = upd.tangent.reshape(-1, kin.n_ip, 4, 4)
            info[mat.name] = upd.info
        return sig_new, state_new, D, info

    def _residual(self, sig_new, F_ext):
        f_int = self.kin.internal_force(sig_new[self.active_idx], self.active_idx, self.n_dof)
        return (F_ext - f_int)[self.free]

    def _assemble(self, D):
        Ke = self.kin.stiffness_entries(D, self.active_idx).reshape(self.active_idx.shape[0], -1)
        data = Ke[self._entry_mask]
        n = self.free.shape[0]
        K = sps.coo_array((data, (self._rows, self._cols)), shape=(n, n)).tocsc()
        return K

    def _elastic_lu(self):
        """LU of the elastic stiffness of the current active set (cached)."""
        if self._elastic_lu_cache is None:
            kin = self.kin
            D = np.empty((self.active_idx.shape[0], kin.n_ip, 4, 4))
            for mid, mat in self.materials.items():
                if mat is None:
                    continue
                sel = self.elem_mat[self.active_idx] == mid
                D[sel] = mat.elastic_tangent()
            self._elastic_lu_cache = spla.splu(self._assemble(D), permc_spec="MMD_AT_PLUS_A",
                                               options={"SymmetricMode": True})
        return self._elastic_lu_cache

    def _activation_load_norm(self, F_ext, regions):
        """Residual carried by DOFs about to enter with a stress-free ring.

        Existing internal forces do not change at installation. Newly free
        DOFs have zero internal force, so their external-load norm adds in
        quadrature to the residual on the currently free DOFs.
        """
        if self.lining is None:
            return 0.0
        ring = self._regions_elements(regions, kinds=(2,)) & ~self.installed
        if not ring.any():
            return 0.0
        future_nodes = np.zeros(self.mesh.n_nodes, dtype=bool)
        future_nodes[np.unique(self.mesh.elems[self.active | ring])] = True
        added = np.repeat(future_nodes, 2) & ~self.fixed
        added[self.free] = False
        return float(np.linalg.norm(F_ext[added]))

    def _acceptance_budget(self, f_ref):
        """Reserve the new-DOF residual within the unchanged full stage tolerance."""
        full = float(self.cfg.solver["tol_rel"]) * f_ref + float(self.cfg.solver["tol_abs"])
        added = float(getattr(self, "activation_load_norm", 0.0))
        active = 0.0 if full == 0.0 else full * math.sqrt(max(0.0, 1.0 - (added / full) ** 2))
        return {"full_residual_tolerance": full,
                "newly_active_load_norm": added,
                "active_residual_tolerance": active}

    def _solve_increment(self, F_ext, f_ref, label=""):
        """Equilibrium for the load F_ext from the last converged state.

        Newton--Raphson with the algorithmic tangent and backtracking (a step is
        accepted only if it lowers the residual norm).  When no backtracked Newton
        step lowers the residual (loading/unloading switches of many integration
        points, indefinite non-associated tangents), a block of modified-Newton
        iterations with the elastic stiffness and Irons--Tuck relaxation is run
        until the residual has dropped by a factor 3 (or ``relax_block``
        iterations), after which Newton is resumed.  Budgets: ``max_iter`` Newton
        iterations, ``modified_max_iter`` relaxed iterations, ``max_stalls`` blocks.
        Returns (converged, du, sig, state, residual history, n_iterations, info).
        """
        s = self.cfg.solver
        budget = self._acceptance_budget(f_ref)
        tol = budget["active_residual_tolerance"]
        max_iter = int(s["max_iter"])
        max_mod = int(s.get("modified_max_iter", 400))
        relax_block = int(s.get("relax_block", 30))
        max_stalls = int(s.get("max_stalls", 10))
        div = float(s.get("divergence_factor", 1e3))
        line_search = bool(s.get("line_search", True))
        line_search_steps = int(s.get("line_search_steps", 6))
        du = np.zeros(self.n_dof)
        sig_new, state_new, D, info = self._material_update(du)
        r = self._residual(sig_new, F_ext)
        rn = float(np.linalg.norm(r))
        r0 = max(rn, tol)
        res = [rn]
        if budget["newly_active_load_norm"] > budget["full_residual_tolerance"]:
            info = dict(info, scheme={"newton": 0, "modified": 0, "stalls": 0,
                                     "stop": "activation load exceeds residual tolerance"})
            return False, du, sig_new, state_new, res, 0, info
        best = (rn, du, sig_new, state_new, D, info, r)
        n_newton = n_modified = n_stalls = 0
        stop = ""
        lu_e = None
        while True:
            if rn <= tol:
                stop = "converged"
                break
            if not np.isfinite(rn) or rn > div * r0:
                stop = "diverged"
                break
            if n_newton >= max_iter:
                stop = "newton budget"
                break
            # --- Newton step with backtracking
            accepted = None
            try:
                # MMD on A + A^T with symmetric-mode pivoting: 2-3x faster than COLAMD on
                # these matrices; the unsymmetric plastic tangent is still factorized exactly
                lu = spla.splu(self._assemble(D), permc_spec="MMD_AT_PLUS_A",
                               options={"SymmetricMode": True})
                ddu = np.zeros(self.n_dof)
                ddu[self.free] = lu.solve(r)
                if np.all(np.isfinite(ddu)):
                    alpha = 1.0
                    for ls in range(line_search_steps):
                        du_t = du + alpha * ddu
                        sig_t, state_t, D_t, info_t = self._material_update(du_t)
                        r_t = self._residual(sig_t, F_ext)
                        rn_t = float(np.linalg.norm(r_t))
                        if np.isfinite(rn_t) and (rn_t < rn or not line_search):
                            accepted = (rn_t, du_t, sig_t, state_t, D_t, info_t, r_t)
                            break
                        alpha *= 0.5
            except RuntimeError:
                accepted = None          # singular tangent: fall through to the relaxed block
            n_newton += 1
            if accepted is not None:
                rn, du, sig_new, state_new, D, info, r = accepted
                res.append(rn)
                if rn < best[0]:
                    best = accepted
                continue
            # --- stalled: block of elastic-stiffness iterations
            n_stalls += 1
            if n_stalls > max_stalls or n_modified >= max_mod:
                stop = "stalled"
                break
            if lu_e is None:
                try:
                    lu_e = self._elastic_lu()
                except RuntimeError as exc:
                    stop = f"elastic factorization failed ({exc})"
                    break
            omega, d_prev = 1.0, None
            rn_block0 = rn
            for k in range(relax_block):
                if rn <= tol or rn < rn_block0 / 3.0 or n_modified >= max_mod:
                    break
                d = np.zeros(self.n_dof)
                d[self.free] = lu_e.solve(r)
                if d_prev is not None:
                    diff = d - d_prev
                    den = float(diff @ diff)
                    if den > 0.0:
                        omega = float(np.clip(-omega * (d_prev @ diff) / den, 0.05, 2.0))
                du = du + omega * d
                d_prev = d
                sig_new, state_new, D, info = self._material_update(du)
                r = self._residual(sig_new, F_ext)
                rn = float(np.linalg.norm(r))
                res.append(rn)
                n_modified += 1
                if not np.isfinite(rn):
                    break
                if rn < best[0]:
                    best = (rn, du, sig_new, state_new, D, info, r)
        if rn > best[0] or not np.isfinite(rn):
            rn, du, sig_new, state_new, D, info, r = best
            res.append(rn)
        converged = rn <= tol
        info = dict(info)
        info["scheme"] = {"newton": n_newton, "modified": n_modified, "stalls": n_stalls,
                          "stop": "converged" if converged else stop}
        return converged, du, sig_new, state_new, res, n_newton + n_modified, info

    # -- staged run -------------------------------------------------------------
    def _regions_elements(self, regions, kinds=(1, 2)):
        m = np.zeros(self.mesh.n_elems, dtype=bool)
        for k in regions:
            for kd in kinds:
                m |= (self.mesh.elem_stage == k) & (self.mesh.elem_kind == kd)
        return m

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _install(self, regions, stage_no, lam, convergence=None):
        """Activate the stage's lining ring and switch its bolted annulus on.

        Returns ``(n_ring_elements, n_annulus_elements)``.  The ring enters the
        active set stress-free; the annulus keeps its stress and state and only
        changes material, which is what installing a bolt into strained ground does.
        ``convergence`` is the inward normal convergence the excavation line had
        reached at this instant, recorded when a closing gap set the instant.
        """
        n_ring = 0
        if self.lining is not None:
            m = self._regions_elements(regions, kinds=(2,)) & ~self.installed
            if m.any():
                self.active[m] = True
                self.installed[m] = True
                self.elem_mat[m] = 1
                self.sig[m] = 0.0
                self.state[m] = 0.0
                n_ring = int(m.sum())
        n_bolt = 0
        if self.bolted is not None:
            b = (self.bolt_elems & ~self.bolt_installed
                 & np.isin(self.bolt_stage, np.asarray(regions, dtype=np.int8)))
            if b.any():
                self.elem_mat[b] = 2
                self.bolt_installed |= b
                n_bolt = int(b.sum())
        if n_ring == 0 and n_bolt == 0:
            return 0, 0
        self._refresh_dofs()
        self.activation_load_norm = 0.0
        for k in regions:
            self.u_install[k] = self.u.copy()
            self.stage_install_lambda[k] = lam
            self.stage_install_convergence[k] = convergence
        return n_ring, n_bolt

    def _upgrade_ring(self, stage_no, lam):
        """Switch every installed, active ring element to the upgraded material.

        Stress and state are kept: the band continues from the state the
        primary ring had reached.  The state row is the Mohr--Coulomb layout
        ``(gamma_p, region, eps_p_vol)`` for both Tresca materials, so the
        accumulated plastic strain and the last return region carry over (a
        region flag that says 'at the limit' is the primary's limit until the
        next converged update re-evaluates it against the new one); an
        upgraded material without state drops the row.  Returns the number of
        elements switched.
        """
        m = self.installed & self.active & (self.elem_mat == 1)
        n = int(m.sum())
        if n:
            self.elem_mat[m] = 3
            if self.upgraded.n_state == 0:
                self.state[m] = 0.0
            self._elastic_lu_cache = None       # the elastic stiffness changed
        self.upgrade_applied = {"stage": stage_no, "lambda": float(lam), "n_ring_elements": n}
        self.u_upgrade = self.u.copy()
        return n

    def _record(self, stage_no, lam, extra):
        tot, post = self.readings_post_install()
        rec = {"stage": stage_no, "lambda": lam, "readings_total": tot,
               "readings_post_install": post}
        rec.update(extra)
        self.history.append(rec)

    def run(self, resume=False) -> dict:
        """Advance the excavation, optionally continuing a saved stopped state."""
        t_run = time.perf_counter()
        sol = self.cfg.solver
        restart = self.stopped if resume else None
        if resume and restart is None:
            raise SolverError("resume requires a stopped calculation")
        elapsed = self.wall_time if resume else 0.0
        self.stopped = None
        self._log(f"[horseshoe_fe] mesh: {self.mesh.n_nodes} nodes, {self.mesh.n_elems} T6, "
                  f"{self.free.shape[0]} free dofs (build {self.t_build:.1f} s)")
        for si, sd in enumerate(self.cfg.stages):
            sp = StageSpec(**sd)
            stage_no = si + 1
            if restart is not None and stage_no < restart["stage"]:
                continue
            regions = list(sp.regions)
            remove = self._regions_elements(regions) & self.active
            continuing = restart is not None and stage_no == restart["stage"]
            if not continuing and not remove.any():
                raise SolverError(f"stage {stage_no}: nothing to excavate for regions {regions}")
            ridx = np.nonzero(remove)[0]
            # The release force of this stage is the force the removed elements
            # exerted on the remaining domain PLUS whatever the previous stages
            # left unreleased (their lambda_end < 1).  With the previous stage in
            # equilibrium at F_0 - (1 - lambda_end) f_R^prev, removing this
            # stage's elements leaves f_int = F_0 - (1 - lambda_end) f_R^prev - f_R,
            # so the unbalanced load that this stage releases from 0 to 1 is the
            # sum; when every earlier stage was fully released the carried
            # remainder is exactly zero and nothing changes.
            fR = (self.fR_last if continuing else
                  self.kin.internal_force(self.sig[ridx], ridx, self.n_dof) + self.fR_carry)
            carried = float(np.linalg.norm(self.fR_carry))
            if not continuing:
                self.active[remove] = False
                self._refresh_dofs()
            f_ref = (self.increment_attempts[-1]["release_force_norm_free"] if continuing
                     else float(np.linalg.norm(fR[self.free])))
            tol = float(sol["tol_rel"]) * f_ref
            # mandatory stops: the user grid, the install factor, extra factors, the end
            stops = set(round(float(l), 12) for l in np.linspace(0.0, sp.lambda_end, sp.n_increments + 1)[1:])
            for extra in list(sp.extra_lambdas) + ([sp.lambda_install] if sp.lambda_install is not None else []):
                if extra is not None and 0.0 < extra <= sp.lambda_end:
                    stops.add(round(float(extra), 12))
            upgrade_here = (self.ring_upgrade is not None and self.upgrade_applied is None
                            and self.ring_upgrade["stage"] == stage_no)
            if upgrade_here:
                stops.add(round(float(self.ring_upgrade["lambda"]), 12))
            stops = sorted(stops)
            dlam_max = sp.lambda_end / sp.n_increments
            dlam = (restart["lambda_target"] - restart["lambda_reached"]
                    if continuing else dlam_max)
            install_regions = list(sp.install_regions) if sp.install_regions is not None else regions
            installed_this_stage = (continuing and any(k in self.u_install for k in install_regions))
            lam = restart["lambda_reached"] if continuing else 0.0
            n_fast = 0
            if not continuing:
                self._record(stage_no, 0.0, {"event": "excavate", "n_removed": int(remove.sum()),
                                             "carried_release_norm": carried,
                                             "release_norm": float(np.linalg.norm(fR))})
            while lam < sp.lambda_end - 1e-12:
                next_stop = next(st for st in stops if st > lam + 1e-12)
                target = min(lam + dlam, next_stop)
                if next_stop - target < 1e-9:
                    target = next_stop
                F_ext = self.F0 - (1.0 - target) * fR
                self.dt = float(sol.get("time_scale", 1.0)) * (target - lam)
                self.activation_load_norm = 0.0
                # For a scheduled stress-free activation, test the total norm
                # after the added DOFs enter, without changing the stage's
                # stated tolerance, loads, material update or installation zero.
                if (not installed_this_stage and sp.lambda_install is not None
                        and target >= sp.lambda_install - 1e-12
                        and sp.install_gap_m is None):
                    self.activation_load_norm = self._activation_load_norm(F_ext, install_regions)
                acceptance_budget = self._acceptance_budget(f_ref)
                t0 = time.perf_counter()
                ok, du, sig_new, state_new, res, n_it, info = self._solve_increment(F_ext, f_ref)
                dt = time.perf_counter() - t0
                self.increment_attempts.append({
                    "stage": stage_no, "lambda_from": lam, "lambda_target": target,
                    "accepted": bool(ok), "iterations": n_it,
                    "release_force_norm_free": f_ref,
                    "residual_tolerance": tol + float(sol["tol_abs"]),
                    "acceptance_budget": acceptance_budget,
                    "residuals": [float(v) for v in res], "info": info})
                if not ok:
                    if (target - lam) < dlam_max / 2 ** int(sol["max_bisect"]) + 1e-15:
                        msg = (f"stage {stage_no}: no convergence at lambda {target:.5f} "
                               f"(step {target - lam:.2e}) after {n_it} iterations "
                               f"(active residual {res[-1]:.3e}, active tol "
                               f"{acceptance_budget['active_residual_tolerance']:.3e}, "
                               f"full tol {acceptance_budget['full_residual_tolerance']:.3e}, "
                               f"{info.get('scheme')})")
                        if str(sol.get("on_failure", "raise")) != "partial":
                            raise SolverError(msg)
                        # corpus mode: keep the last converged state and report where it stopped
                        self.stopped = {"stage": stage_no, "lambda_reached": lam,
                                        "lambda_target": target, "message": msg,
                                        "residual": float(res[-1]),
                                        "tolerance": acceptance_budget["active_residual_tolerance"],
                                        "full_residual_tolerance": acceptance_budget["full_residual_tolerance"],
                                        "newly_active_load_norm": acceptance_budget["newly_active_load_norm"],
                                        "combined_residual": math.hypot(float(res[-1]), acceptance_budget["newly_active_load_norm"])}
                        self._log(f"  {msg}\n  -> stopping (on_failure=partial)")
                        self.fR_last = fR
                        self.wall_time = elapsed + time.perf_counter() - t_run
                        return self.results()
                    dlam = 0.5 * (target - lam)
                    n_fast = 0
                    self._log(f"  stage {stage_no} lambda {lam:.4f}->{target:.4f}: not converged "
                              f"({res[-1]:.2e}, {info.get('scheme')}), step -> {dlam:.2e}")
                    continue
                self.u += du
                self.sig = sig_new
                self.state = state_new
                self.increments.append({"stage": stage_no, "lambda_from": lam, "lambda_to": target,
                                        "iterations": n_it, "residuals": [float(x) for x in res],
                                        "release_force_norm_free": f_ref,
                                        "residual_tolerance": tol + float(sol["tol_abs"]),
                                        "acceptance_budget": acceptance_budget,
                                        "wall_time": dt, "info": info})
                pinfo = info.get(self.rock.name, {})
                sch = info.get("scheme", {})
                self._log(f"  stage {stage_no} lambda {target:.4f}: {n_it} it "
                          f"(N{sch.get('newton', 0)}/M{sch.get('modified', 0)}), r={res[-1]:.2e} "
                          f"(tol {tol:.1e}), {dt:.2f} s"
                          + (f", plastic {pinfo.get('n_plastic', 0)} (edge {pinfo.get('n_edge', 0)}, "
                             f"apex {pinfo.get('n_apex', 0)})" if pinfo else ""))
                lam = target
                # adaptive step: grow back after two consecutive fast Newton increments
                if bool(sol.get("adaptive", True)):
                    if sch.get("stalls", 0) == 0 and sch.get("newton", 0) <= 6:
                        n_fast += 1
                    else:
                        n_fast = 0
                    if n_fast >= 2 and dlam < dlam_max:
                        dlam = min(2.0 * dlam, dlam_max)
                        n_fast = 0
                ev = {}
                if (not installed_this_stage and sp.lambda_install is not None
                        and lam >= sp.lambda_install - 1e-12):
                    # a closing gap adds a second condition to the release factor:
                    # the ring cannot carry before the rock has moved onto it
                    conv = (self.exc_convergence(regions)
                            if sp.install_gap_m is not None else None)
                    if conv is None or conv >= float(sp.install_gap_m) - 1e-12:
                        n_ring, n_bolt = self._install(install_regions, stage_no, lam, conv)
                        installed_this_stage = True
                        ev = {"event": "install", "n_ring_elements": n_ring,
                              "n_annulus_elements": n_bolt,
                              "convergence_at_install_m": conv}
                        self._log(f"  stage {stage_no}: support installed at lambda {lam:.4f} "
                                  f"({n_ring} ring elements, {n_bolt} bolted-annulus "
                                  f"elements)"
                                  + ("" if conv is None
                                     else f", gap {sp.install_gap_m:.3f} m closed at "
                                          f"convergence {conv:.4f} m"))
                if (upgrade_here and self.upgrade_applied is None
                        and lam >= self.ring_upgrade["lambda"] - 1e-12):
                    # the ring-upgrade event (secondary lining): after the install
                    # check, so a ring installed at this same instant is upgraded too
                    n_up = self._upgrade_ring(stage_no, lam)
                    ev = dict(ev, event=("install+ring_upgrade" if ev else "ring_upgrade"),
                              n_upgraded_ring_elements=n_up)
                    self._log(f"  stage {stage_no}: ring upgraded at lambda {lam:.4f} "
                              f"({n_up} ring elements -> {self.upgraded.name})")
                committed_residual = float(np.linalg.norm(self._residual(self.sig, F_ext)))
                ev["equilibrium"] = {
                    "residual_norm": committed_residual,
                    "release_force_norm_free": f_ref,
                    "tolerance": tol + float(sol["tol_abs"]),
                    "residual_over_tolerance": committed_residual / (tol + float(sol["tol_abs"]))}
                self._record(stage_no, lam, ev)
            if (sp.lambda_install is not None and sp.install_gap_m is not None
                    and not installed_this_stage):
                # the gap never closed inside this stage: the ring is not there
                reached = self.exc_convergence(regions)
                self.rings_not_installed.append(
                    {"stage": stage_no, "regions": regions,
                     "install_gap_m": float(sp.install_gap_m),
                     "max_convergence_reached_m": reached,
                     "reason": "the closing gap did not close before the end of the stage"})
                self._log(f"  stage {stage_no}: ring NOT installed -- gap "
                          f"{sp.install_gap_m:.3f} m, convergence reached {reached:.4f} m")
            self.fR_last = fR
            # what this stage leaves unreleased is carried into the next stage's
            # release force (zero for a fully released stage)
            self.fR_carry = (1.0 - lam) * fR
        self.wall_time = elapsed + time.perf_counter() - t_run
        return self.results()

    def ring_upgrade_record(self) -> dict:
        """The ring-upgrade event as configured and as it fired (or did not)."""
        if self.ring_upgrade is None:
            return {"enabled": False, "applied": False}
        out = {"enabled": True, "applied": self.upgrade_applied is not None,
               "stage": self.ring_upgrade["stage"], "lambda": self.ring_upgrade["lambda"],
               "material": self.upgraded.describe(), "rule": self.ring_upgrade["rule"],
               "source": self.ring_upgrade["source"], "note": self.ring_upgrade["note"],
               "label": self.ring_upgrade["label"],
               "n_ring_elements": None, "readings_at_upgrade": None,
               "readings_post_upgrade": None}
        if self.upgrade_applied is None:
            out["why_not"] = (f"the run did not reach lambda {self.ring_upgrade['lambda']} "
                              f"of stage {self.ring_upgrade['stage']}")
            return out
        out["n_ring_elements"] = self.upgrade_applied["n_ring_elements"]
        at = self.readings(self.u_upgrade)
        now = self.readings()
        out["readings_at_upgrade"] = at
        out["readings_post_upgrade"] = {k: now[k] - at[k] for k in now}
        return out

    def zero_load_resolve(self):
        """Re-solve the current state under unchanged load (round-off check)."""
        F_ext = self.F0 - (1.0 - 1.0) * self.fR_last if hasattr(self, "fR_last") else self.F0
        f_ref = float(np.linalg.norm(self.fR_last[self.free])) if hasattr(self, "fR_last") else 1.0
        self.dt = 0.0                                   # no pseudo-time passes in a re-solve
        ok, du, sig_new, state_new, res, n_it, info = self._solve_increment(self.F0, f_ref)
        self.u += du
        self.sig, self.state = sig_new, state_new
        return {"converged": bool(ok), "iterations": n_it, "residuals": [float(x) for x in res]}

    # -- results ------------------------------------------------------------------
    def results(self) -> dict:
        tot, post = self.readings_post_install()
        chords = []
        for name in self.chord_specs:
            spec = self.points[name]
            chords.append({"name": name, "channel": spec["channel"],
                           "boundary": spec["boundary"], "depth_below_crown": spec["depth"],
                           "initial_chord_length_m": 2 * float(spec["xy"][0]),
                           "point": [float(v) for v in spec["xy"]], "stage": spec["stage"],
                           "total": tot[name], "post_install": post[name]})
        lin = self.lining_stations()
        pz = self.plastic_zone()
        inc = self.increments
        out = {
            "label": self.cfg.label,
            "config": self.cfg.to_dict(),
            "mesh": {"n_nodes": int(self.mesh.n_nodes), "n_elements": int(self.mesh.n_elems),
                     "n_free_dof": int(self.free.shape[0]), "h_wall": self.mesh.h_wall,
                     "r_far": self.mesh.r_far, **{k: (float(v) if isinstance(v, (int, float)) else v)
                                                   for k, v in self.mesh.info.items()}},
            "materials": {"rock": self.rock.describe(),
                          "characteristic_length": self.characteristic_length,
                          "crack_band": self.crack_band,
                          "lining": self.lining.describe() if self.lining else None,
                          "lining_thickness": self.lining_thickness,
                          "upgraded_lining": (self.upgraded.describe() if self.upgraded
                                              else None),
                          "bolted_rock": self.bolted.describe() if self.bolted else None,
                          "bolts": (dict(self.bolts, **self.bolt_annulus_info())
                                    if self.bolts else None)},
            "initial_stress": {"sigma_h": float(self.sigma0[0]), "sigma_v": float(self.sigma0[1]),
                               "sigma_z": float(self.sigma0[2])},
            "readings": {
                "crown_settlement": {"total": tot["crown"], "post_install": post["crown"],
                                     "stage": self.points["crown"]["stage"],
                                     "point": [float(v) for v in self.points["crown"]["xy"]],
                                     "channel": "GD", "boundary": self.reading_boundary},
                "chord_shortening": chords,
                "install_lambda": {str(k): v for k, v in self.stage_install_lambda.items()},
                "install_convergence_m": {str(k): v for k, v
                                          in self.stage_install_convergence.items()},
                "exc_line_max_convergence_m": self.exc_convergence(),
            },
            "lining": {"stations": {name: {"point": [float(v) for v in self.stations[name]],
                                           **lin[name]} for name in self.stations},
                       "extrema": self.lining_stress_extrema(),
                       "yield": self.lining_yield_state(),
                       "rings_not_installed": list(self.rings_not_installed),
                       "ring_upgrade": self.ring_upgrade_record()},
            "plastic_zone": pz,
            "completed": getattr(self, "stopped", None) is None,
            "stopped": getattr(self, "stopped", None),
            "diagnostics": {
                "n_increments": len(inc),
                "iterations_per_increment": [d["iterations"] for d in inc],
                "final_residuals": [d["residuals"][-1] for d in inc],
                "increments": inc,
                "increment_attempts": self.increment_attempts,
                "wall_time_solve": getattr(self, "wall_time", None),
                "wall_time_build": self.t_build,
            },
            "history": self.history,
        }
        return out


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(f"not serializable: {type(o)}")


def write_json(obj, path):
    import os
    payload = json.dumps(obj, indent=1, default=_json_default, allow_nan=False)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload + "\n")


def build_model(cfg: RunConfig, verbose: bool = True) -> StagedModel:
    return StagedModel(cfg, verbose=verbose)


def run_case(cfg: RunConfig, out_json: Optional[str] = None, verbose: bool = True):
    model = build_model(cfg, verbose=verbose)
    res = model.run()
    if out_json:
        write_json(res, out_json)
    return res, model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_parser():
    p = argparse.ArgumentParser(
        prog="python -m cwfs_inv.horseshoe_fe",
        description="Staged plane-strain FE run of the horseshoe (or circular) opening.")
    p.add_argument("--config", help="JSON file with RunConfig keys (CLI values override it)")
    p.add_argument("--out", default="results/horseshoe_fe/run.json", help="output run JSON")
    p.add_argument("--label", default=None)
    g = p.add_argument_group("geometry")
    g.add_argument("--geometry", choices=["horseshoe", "circle"], default=None)
    g.add_argument("--circle-radius", type=float, default=None)
    for name, help_ in [("crown-radius", "crown arc radius (m)"),
                        ("crown-angle-deg", "crown arc central angle (deg)"),
                        ("bench-depth", "bench line below the crown (m)"),
                        ("base-depth", "wall base below the crown (m)"),
                        ("bottom-width", "width at the wall base (m)"),
                        ("invert-sagitta", "invert arch sagitta (m)"),
                        ("lining-thickness", "lining ring thickness (m)"),
                        ("reserved-deformation", "extra offset of the excavation line (m)")]:
        g.add_argument("--" + name, type=float, default=None, help=help_)
    m = p.add_argument_group("mesh")
    m.add_argument("--h-wall", type=float, default=None, help="element size at the excavation line")
    m.add_argument("--r-far", type=float, default=None, help="far boundary radius")
    m.add_argument("--h-far", type=float, default=None)
    m.add_argument("--dist-min", type=float, default=None)
    r = p.add_argument_group("rock and stress")
    r.add_argument("--material", choices=["elastic", "mc", "cwfs"], default=None)
    r.add_argument("--E", type=float, default=None)
    r.add_argument("--nu", type=float, default=None)
    r.add_argument("--c", type=float, default=None)
    r.add_argument("--phi", type=float, default=None, help="friction angle (deg)")
    r.add_argument("--psi", type=float, default=None, help="dilation angle (deg)")
    r.add_argument("--sigma-v", type=float, default=None)
    r.add_argument("--K", type=float, default=None)
    r.add_argument("--sigma-z", type=float, default=None)
    w = p.add_argument_group("cwfs strain-softening rock (--material cwfs)")
    w.add_argument("--c-peak", type=float, default=None)
    w.add_argument("--c-res", type=float, default=None)
    w.add_argument("--phi-peak", type=float, default=None, help="peak friction (deg)")
    w.add_argument("--phi-res", type=float, default=None, help="residual friction (deg)")
    w.add_argument("--gp-c", type=float, default=None, help="plastic shear strain at c_res")
    w.add_argument("--gp-phi", type=float, default=None, help="plastic shear strain at phi_res")
    w.add_argument("--eta", type=float, default=None,
                   help="Duvaut-Lions relaxation time in pseudo-time units (0 = off)")
    w.add_argument("--crack-band-h-ref", type=float, default=None,
                   help="crack-band reference band width in metres (0 = off); the "
                        "softening thresholds become w/h with w = gp * h_ref")
    w.add_argument("--crack-band-h-wall-ref", type=float, default=None,
                   help="same, given as the mesh size the thresholds are quoted at; "
                        "h_ref = fe_materials.crack_band_reference(h_wall_ref)")
    w.add_argument("--no-scale-phi", action="store_true",
                   help="scale only the cohesion threshold, not the friction one")
    w.add_argument("--time-scale", type=float, default=None,
                   help="pseudo-time per unit release factor (default 1)")
    l = p.add_argument_group("lining")
    l.add_argument("--no-lining", action="store_true")
    l.add_argument("--E-lining", type=float, default=None)
    l.add_argument("--nu-lining", type=float, default=None)
    l.add_argument("--lining-model", choices=["elastic", "tresca"], default=None,
                   help="'tresca' gives the ring a strength limit (needs --sigma-lim)")
    l.add_argument("--sigma-lim", type=float, default=None,
                   help="ring strength limit s1 - s3 (stress units of the run)")
    l.add_argument("--install-gap", type=float, default=None,
                   help="radial closing gap (m) the rock must take up before the ring "
                        "carries; applies to every stage")
    s = p.add_argument_group("stages")
    s.add_argument("--full-face", action="store_true", help="single stage removing all regions")
    s.add_argument("--lambda-install", type=float, nargs="+", default=None,
                   help="per stage (one value = all stages); use -1 for never")
    s.add_argument("--n-increments", type=int, default=None)
    s.add_argument("--lambda-end", type=float, default=None, help="final release factor (last stage)")
    s.add_argument("--tol-rel", type=float, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def config_from_args(args) -> RunConfig:
    base = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            base = json.load(f)
    cfg = RunConfig.from_dict(base)
    if args.geometry:
        cfg.geometry = args.geometry
    if args.circle_radius is not None:
        cfg.circle_radius = args.circle_radius
    for key in ["crown_radius", "crown_angle_deg", "bench_depth", "base_depth", "bottom_width",
                "invert_sagitta", "lining_thickness", "reserved_deformation"]:
        v = getattr(args, key)
        if v is not None:
            cfg.horseshoe[key] = v
    for key in ["h_wall", "r_far", "h_far", "dist_min"]:
        v = getattr(args, key)
        if v is not None:
            cfg.mesh[key] = v
    if args.material:
        cfg.rock["model"] = args.material
    for key in ["E", "nu", "c"]:
        v = getattr(args, key)
        if v is not None:
            cfg.rock[key] = v
    if args.phi is not None:
        cfg.rock["phi_deg"] = args.phi
    if args.psi is not None:
        cfg.rock["psi_deg"] = args.psi
    if cfg.rock["model"] == "mc":
        for key in ("c", "phi_deg", "psi_deg"):
            cfg.rock.setdefault(key, {"c": 0.2, "phi_deg": 25.6, "psi_deg": 5.0}[key])
    if cfg.rock["model"] == "cwfs":
        for name, key in (("c_peak", "c_peak"), ("c_res", "c_res"), ("phi_peak", "phi_peak_deg"),
                          ("phi_res", "phi_res_deg"), ("gp_c", "gp_c"), ("gp_phi", "gp_phi")):
            v = getattr(args, name)
            if v is not None:
                cfg.rock[key] = v
        if args.psi is not None:
            cfg.rock["psi_deg"] = args.psi
        for key, dflt in (("c_peak", 1.0), ("c_res", 0.2), ("phi_peak_deg", 20.0),
                          ("phi_res_deg", 30.0), ("gp_c", 0.01), ("gp_phi", 0.015),
                          ("psi_deg", 5.0)):
            cfg.rock.setdefault(key, dflt)
        for key in ("c", "phi_deg"):
            cfg.rock.pop(key, None)                 # inherited elastic/mc defaults
        if args.eta is not None:
            cfg.rock["viscoplastic"] = {"eta": args.eta} if args.eta > 0 else None
        h_ref = args.crack_band_h_ref
        if h_ref is None and args.crack_band_h_wall_ref is not None:
            from cwfs_inv.fe_materials import crack_band_reference
            h_ref = crack_band_reference(args.crack_band_h_wall_ref)
        if h_ref is not None:
            cfg.rock["crack_band"] = ({"h_ref": h_ref, "scale_phi": not args.no_scale_phi}
                                      if h_ref > 0 else None)
    if args.time_scale is not None:
        cfg.solver["time_scale"] = args.time_scale
    for key in ["sigma_v", "K", "sigma_z"]:
        v = getattr(args, key)
        if v is not None:
            cfg.stress[key] = v
    if args.no_lining:
        cfg.lining["enabled"] = False
    if args.E_lining is not None:
        cfg.lining["E"] = args.E_lining
    if args.nu_lining is not None:
        cfg.lining["nu"] = args.nu_lining
    if args.lining_model is not None:
        cfg.lining["model"] = args.lining_model
    if args.sigma_lim is not None:
        cfg.lining["sigma_lim"] = args.sigma_lim
        cfg.lining.setdefault("model", "tresca")
    if cfg.lining.get("model") != "tresca":
        cfg.lining.pop("sigma_lim", None)
    if cfg.geometry == "circle" and not base.get("stages"):
        cfg.stages = [asdict(StageSpec([1]))]
    if args.full_face:
        regions = sorted({r for s in cfg.stages for r in s["regions"]})
        cfg.stages = [asdict(StageSpec(regions, n_increments=cfg.stages[0]["n_increments"],
                                       lambda_install=cfg.stages[0]["lambda_install"]))]
    if args.lambda_install is not None:
        vals = args.lambda_install
        if len(vals) == 1:
            vals = vals * len(cfg.stages)
        for s, v in zip(cfg.stages, vals):
            s["lambda_install"] = None if v < 0 else float(v)
    if args.n_increments is not None:
        for s in cfg.stages:
            s["n_increments"] = args.n_increments
    if args.lambda_end is not None:
        cfg.stages[-1]["lambda_end"] = args.lambda_end
    if args.install_gap is not None:
        for s in cfg.stages:
            s["install_gap_m"] = None if args.install_gap <= 0 else float(args.install_gap)
    if args.tol_rel is not None:
        cfg.solver["tol_rel"] = args.tol_rel
    if args.label is not None:
        cfg.label = args.label
    return cfg


def main(argv=None):
    args = _cli_parser().parse_args(argv)
    cfg = config_from_args(args)
    res, model = run_case(cfg, out_json=args.out, verbose=not args.quiet)
    rd = res["readings"]
    print(f"crown settlement: total {rd['crown_settlement']['total']:.5e} m, "
          f"post-install {rd['crown_settlement']['post_install']}")
    for ch in rd["chord_shortening"]:
        print(f"chord {ch['depth_below_crown']:.2f} m: total {ch['total']:.5e} m, post-install {ch['post_install']}")
    for name, st in res["lining"]["stations"].items():
        print(f"lining {name}: pressure {st['pressure']}  hoop_mean {st['hoop_mean']}")
    print(f"plastic zone depth: {res['plastic_zone']['depth']}  max gamma_p {res['plastic_zone']['max_gamma_p']:.3e}")
    print(f"wall time solve {res['diagnostics']['wall_time_solve']:.1f} s; written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
