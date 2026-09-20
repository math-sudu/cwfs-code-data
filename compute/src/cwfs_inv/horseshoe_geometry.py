"""Parametric opening profile, offset curves and gmsh meshing for ``horseshoe_fe``.

Profile convention
------------------
A half profile (x >= 0) is a list of primitives (:class:`Arc`, :class:`Line`)
traversed from the top axis point (crown) clockwise to the bottom axis point
(invert centre); the opening interior is on the right-hand side of the
direction of travel, so the LEFT normal of the travel direction points outward
into the rock.  Coordinates: x horizontal, y vertical (up), origin at the crown
of the selected reference line, metres. Arcs are traversed monotonically in angle from
``th0`` to ``th1`` (decreasing angle = clockwise about the arc centre).

CAD reference
-------------
``cad_reference_profile`` reads the six exact tangent arcs in the recorded
similarity transform and returns the clockwise right half. Its reference is
the primary-support inner boundary; the excavation line is offset outward by
the independently specified primary-support thickness. Stage cuts and target
positions are supplied separately and never taken from figure display data.

Historical horseshoe net profile (drawing 11-34, read 2026-09-07)
------------------------------------------------------------------------
* crown arc: radius ``crown_radius`` (3.4 m), central angle ``crown_angle_deg``
  (120 deg 30 min), centre on the axis ``crown_radius`` below the crown;
* upper sidewall arcs: tangent to the crown arc, vertical (maximum width) at
  the bench line ``bench_depth`` (4.7 m) below the crown -- the radius and the
  maximum width follow from those two conditions (3.4 / 120.5 deg / 4.7 give
  R = 6.02 m and a maximum width of 7.49 m, the drawing's 7.5 m);
* lower sidewalls: straight from the bench line down to the wall base
  ``base_depth`` (7.4 m) below the crown, bottom width ``bottom_width``
  (7.15 m);
* invert: circular arc below the wall base with sagitta ``invert_sagitta``
  (0.6 m, ASSUMED -- the drawing does not dimension the invert);
* the drawing's nominal net height is 7.5 m; the model uses the dimensioned
  wall-base depth and the invert sagitta.

For that historical parameterization, excavation line = net line offset by ``lining_thickness +
reserved_deformation``; lining ring = band between the net line offset by
``reserved_deformation`` and the excavation line (thickness = lining
thickness).  Convex corners of the net line (wall foot) become fillets of the
offset radius; corners turning less than ``snap_deg`` (the bench kink between
the vertical arc end and the slightly inclined lower wall) are closed by
snapping the next line start onto the previous offset end.

Meshing
-------
gmsh (OpenCASCADE kernel) builds one conforming mesh of six-node triangles
(second order, curved on the arcs) for the half domain bounded by the axis and a
half-disc of radius ``r_far`` centred on the profile's mid-height: rock,
lining-ring segments per stage, core segments per stage.  Region membership is
known by construction (explicit curve loops), not by boolean classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

@dataclass
class Arc:
    cx: float
    cy: float
    r: float
    th0: float
    th1: float

    @property
    def start(self):
        return (self.cx + self.r * math.cos(self.th0), self.cy + self.r * math.sin(self.th0))

    @property
    def end(self):
        return (self.cx + self.r * math.cos(self.th1), self.cy + self.r * math.sin(self.th1))

    @property
    def clockwise(self):
        return self.th1 < self.th0

    def tangent(self, th):
        s = -1.0 if self.clockwise else 1.0
        return (-s * math.sin(th), s * math.cos(th))

    @property
    def start_tangent(self):
        return self.tangent(self.th0)

    @property
    def end_tangent(self):
        return self.tangent(self.th1)

    @property
    def length(self):
        return abs(self.th1 - self.th0) * self.r

    def point(self, t):
        th = self.th0 + t * (self.th1 - self.th0)
        return (self.cx + self.r * math.cos(th), self.cy + self.r * math.sin(th))

    def sample(self, n):
        th = np.linspace(self.th0, self.th1, n)
        return np.column_stack([self.cx + self.r * np.cos(th), self.cy + self.r * np.sin(th)])


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def start(self):
        return (self.x0, self.y0)

    @property
    def end(self):
        return (self.x1, self.y1)

    @property
    def length(self):
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def start_tangent(self):
        L = self.length
        return ((self.x1 - self.x0) / L, (self.y1 - self.y0) / L)

    end_tangent = start_tangent

    def point(self, t):
        return (self.x0 + t * (self.x1 - self.x0), self.y0 + t * (self.y1 - self.y0))

    def sample(self, n):
        t = np.linspace(0.0, 1.0, n)
        return np.column_stack([self.x0 + t * (self.x1 - self.x0), self.y0 + t * (self.y1 - self.y0)])


def _left_normal(t):
    return (-t[1], t[0])


# ---------------------------------------------------------------------------
# horseshoe / circle net profiles
# ---------------------------------------------------------------------------

@dataclass
class HorseshoeParams:
    crown_radius: float = 3.4
    crown_angle_deg: float = 120.5
    bench_depth: float = 4.7
    base_depth: float = 7.4
    bottom_width: float = 7.15
    invert_sagitta: float = 0.6
    lining_thickness: float = 0.25
    reserved_deformation: float = 0.10
    snap_deg: float = 5.0

    @property
    def offset(self):
        return self.lining_thickness + self.reserved_deformation

    def derived(self) -> dict:
        R1 = self.crown_radius
        tht = math.radians(90.0 - 0.5 * self.crown_angle_deg)
        R2 = R1 - (R1 - self.bench_depth) / math.sin(tht)
        c2x = (R1 - R2) * math.cos(tht)
        x_bench = c2x + R2
        x_base = 0.5 * self.bottom_width
        s = self.invert_sagitta
        R3 = (x_base ** 2 + s ** 2) / (2.0 * s)
        return {"crown_end_angle_deg": math.degrees(tht), "sidewall_radius": R2,
                "sidewall_centre": (c2x, -self.bench_depth), "max_half_width": x_bench,
                "invert_radius": R3, "invert_centre": (0.0, -self.base_depth - s + R3),
                "invert_bottom_depth": self.base_depth + s}


def horseshoe_net_profile(p: HorseshoeParams) -> list:
    d = p.derived()
    R1 = p.crown_radius
    tht = math.radians(d["crown_end_angle_deg"])
    R2 = d["sidewall_radius"]
    c2x, c2y = d["sidewall_centre"]
    if not (R2 >= R1 - 1e-12 and c2x + R2 > 0.0):
        raise ValueError("inconsistent crown/bench parameters (sidewall arc)")
    x_bench = d["max_half_width"]
    x_base = 0.5 * p.bottom_width
    R3 = d["invert_radius"]
    c3x, c3y = d["invert_centre"]
    prims = [
        Arc(0.0, -R1, R1, math.pi / 2.0, tht),                       # crown arc
        Arc(c2x, c2y, R2, tht, 0.0),                                 # upper sidewall
        Line(x_bench, -p.bench_depth, x_base, -p.base_depth),        # lower sidewall
        Arc(c3x, c3y, R3, math.atan2(-p.base_depth - c3y, x_base), -math.pi / 2.0),  # invert
    ]
    return prims


def circle_profile(a: float, cy: float = 0.0) -> list:
    return [Arc(0.0, cy, a, math.pi / 2.0, 0.0), Arc(0.0, cy, a, 0.0, -math.pi / 2.0)]


def cad_reference_profile(document: dict) -> list:
    """Exact clockwise right half of the author-selected six-arc CAD outline.

    This is the primary-support INNER boundary, with its crown at (0, 0).
    Stage cuts and targets are separate inputs; display_layout is never read.
    The source's semicircular crown and invert are split only at the axis.
    No replacement curve, corner snapping or change of curvature is applied.
    """
    transform = document["transform"]
    scale = float(transform["scale"])
    crown = float(transform["source_crown_y_m"])
    if document["source"]["boundary_role"] != "initial_support_inner":
        raise ValueError("CAD reference must identify the initial-support inner boundary")
    source = {a["arc_id"]: a for a in document["source_circular_arcs"]}
    required = {"crown", "left_wall", "left_knee", "invert", "right_knee", "right_wall"}
    if set(source) != required or len(document["source_circular_arcs"]) != 6:
        raise ValueError("Expected the six recorded tangent CAD arcs")
    # Validate the complete source, including the unused mirrored half.
    full = []
    for a in document["source_circular_arcs"]:
        cx, cy = a["center"]
        full.append(Arc(scale * cx, scale * (cy - crown), scale * a["radius"],
                        a["start_rad"], a["start_rad"] + a["sweep_rad"]))
    for before, after in zip(full, full[1:] + full[:1]):
        if not np.allclose(before.end, after.start, atol=1e-10, rtol=0):
            raise ValueError("CAD arc endpoints are disconnected")
        if not np.allclose(before.end_tangent, after.start_tangent, atol=1e-10, rtol=0):
            raise ValueError("CAD arc junction is not tangent")
    half = []
    for name in ("crown", "right_wall", "right_knee", "invert"):
        a = source[name]
        high = a["start_rad"] + a["sweep_rad"]
        low = a["start_rad"]
        if name == "crown":
            high = math.pi / 2
        elif name == "invert":
            low = 3 * math.pi / 2
        cx, cy = a["center"]
        half.append(Arc(scale * cx, scale * (cy - crown), scale * a["radius"], high, low))
    for before, after in zip(half, half[1:]):
        if not np.allclose(before.end, after.start, atol=1e-10, rtol=0):
            raise ValueError("CAD right-half extraction is disconnected")
    height = half[0].start[1] - half[-1].end[1]
    width = 2 * half[0].end[0]
    if not (np.isclose(height, transform["resulting_height_m"], atol=1e-11, rtol=0)
            and np.isclose(width, transform["target_width_m"], atol=1e-11, rtol=0)):
        raise ValueError("CAD dimensions disagree with the recorded similarity transform")
    return half


# ---------------------------------------------------------------------------
# offset and split
# ---------------------------------------------------------------------------

def offset_profile(prims: list, d: float, snap_deg: float = 5.0) -> list:
    """Offset a clockwise half profile outward by ``d`` (fillets at convex corners)."""
    if d == 0.0:
        return [Arc(**vars(p)) if isinstance(p, Arc) else Line(**vars(p)) for p in prims]
    off = []
    for p in prims:
        if isinstance(p, Arc):
            r = p.r + d if p.clockwise else p.r - d
            off.append(Arc(p.cx, p.cy, r, p.th0, p.th1))
        else:
            n = _left_normal(p.start_tangent)
            off.append(Line(p.x0 + d * n[0], p.y0 + d * n[1], p.x1 + d * n[0], p.y1 + d * n[1]))
    out = [off[0]]
    snap = math.radians(snap_deg)
    for i in range(1, len(prims)):
        t1, t2 = prims[i - 1].end_tangent, prims[i].start_tangent
        cross = t1[0] * t2[1] - t1[1] * t2[0]
        dot = t1[0] * t2[0] + t1[1] * t2[1]
        alpha = math.atan2(cross, dot)
        prev, nxt = out[-1], off[i]
        if abs(alpha) < snap:
            pe = prev.end
            if isinstance(nxt, Line):
                nxt = Line(pe[0], pe[1], nxt.x1, nxt.y1)
            elif isinstance(prev, Line):
                ns = nxt.start
                out[-1] = Line(prev.x0, prev.y0, ns[0], ns[1])
            else:
                gap = math.hypot(pe[0] - nxt.start[0], pe[1] - nxt.start[1])
                if gap > 1e-9:
                    raise ValueError(f"non-tangent arc/arc junction with gap {gap}")
            out.append(nxt)
        elif alpha < 0.0:
            cx, cy = prims[i - 1].end
            n1, n2 = _left_normal(t1), _left_normal(t2)
            a0 = math.atan2(n1[1], n1[0])
            a1 = math.atan2(n2[1], n2[0])
            while a1 > a0:
                a1 -= 2.0 * math.pi
            out.append(Arc(cx, cy, d, a0, a1))
            out.append(nxt)
        else:
            raise NotImplementedError("concave corner offset (trim) not implemented")
    return out


def split_at_y(prims: list, y0: float, tol: float = 1e-9) -> list:
    """Split primitives where they cross y = y0 (interior crossings only)."""
    out = []
    for p in prims:
        if isinstance(p, Line):
            ya, yb = p.y0, p.y1
            if (y0 - ya) * (y0 - yb) < 0.0 and abs(ya - yb) > tol:
                t = (y0 - ya) / (yb - ya)
                xm = p.x0 + t * (p.x1 - p.x0)
                out.append(Line(p.x0, p.y0, xm, y0))
                out.append(Line(xm, y0, p.x1, p.y1))
            else:
                out.append(p)
        else:
            v = (y0 - p.cy) / p.r
            cands = []
            if abs(v) <= 1.0:
                base = math.asin(max(-1.0, min(1.0, v)))
                for th in (base, math.pi - base):
                    lo, hi = min(p.th0, p.th1), max(p.th0, p.th1)
                    for k in (-2, -1, 0, 1, 2):
                        thk = th + 2.0 * math.pi * k
                        if lo + tol < thk < hi - tol:
                            cands.append(thk)
            cands = sorted(set(cands), reverse=p.clockwise)
            if not cands:
                out.append(p)
                continue
            th_prev = p.th0
            for thk in cands:
                out.append(Arc(p.cx, p.cy, p.r, th_prev, thk))
                th_prev = thk
            out.append(Arc(p.cx, p.cy, p.r, th_prev, p.th1))
    return out


def point_at_y(prims: list, y0: float) -> tuple:
    """Point (x > 0) where the profile crosses y = y0 (first crossing from the top)."""
    for p in split_at_y(prims, y0):
        e = p.end
        if abs(e[1] - y0) < 1e-7 and e[0] > 1e-9:
            return (e[0], y0)
        s = p.start
        if abs(s[1] - y0) < 1e-7 and s[0] > 1e-9:
            return (s[0], y0)
    raise ValueError(f"profile does not cross y = {y0}")


def profile_top(prims):
    return prims[0].start


def profile_bottom(prims):
    return prims[-1].end


def profile_length(prims):
    return sum(p.length for p in prims)


# ---------------------------------------------------------------------------
# dense polyline for projections
# ---------------------------------------------------------------------------

class ProfileCurve:
    """Dense polyline of a profile with arclength, outward normals and projection."""

    def __init__(self, prims: list, ds: float = 0.005):
        pts, tans = [], []
        for p in prims:
            n = max(int(math.ceil(p.length / ds)) + 1, 2)
            xy = p.sample(n)
            if isinstance(p, Arc):
                th = np.linspace(p.th0, p.th1, n)
                s = -1.0 if p.clockwise else 1.0
                tg = np.column_stack([-s * np.sin(th), s * np.cos(th)])
            else:
                t = np.array(p.start_tangent)
                tg = np.tile(t, (n, 1))
            if pts:
                xy, tg = xy[1:], tg[1:]
            pts.append(xy)
            tans.append(tg)
        self.pts = np.vstack(pts)
        self.tan = np.vstack(tans)
        self.nrm = np.column_stack([-self.tan[:, 1], self.tan[:, 0]])   # outward
        seg = np.diff(self.pts, axis=0)
        self.s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        from scipy.spatial import cKDTree
        self._tree = cKDTree(self.pts)

    @property
    def length(self):
        return float(self.s[-1])

    def project(self, xy: np.ndarray):
        """Arclength ``s`` of the closest profile point and signed outward distance
        ``dist`` (> 0 rock side, < 0 opening side) for points ``xy`` (n, 2)."""
        xy = np.atleast_2d(xy)
        _, k = self._tree.query(xy)
        n = self.pts.shape[0]
        best_s = self.s[k].copy()
        best_d2 = np.sum((xy - self.pts[k]) ** 2, axis=1)
        best_pt = self.pts[k].copy()
        best_n = self.nrm[k].copy()
        for k0, k1 in ((np.maximum(k - 1, 0), k), (k, np.minimum(k + 1, n - 1))):
            p0, p1 = self.pts[k0], self.pts[k1]
            d = p1 - p0
            L2 = np.sum(d * d, axis=1)
            t = np.clip(np.sum((xy - p0) * d, axis=1) / np.maximum(L2, 1e-30), 0.0, 1.0)
            q = p0 + t[:, None] * d
            d2 = np.sum((xy - q) ** 2, axis=1)
            better = (d2 < best_d2) & (L2 > 0)
            best_d2[better] = d2[better]
            best_pt[better] = q[better]
            best_s[better] = (self.s[k0] + t * (self.s[k1] - self.s[k0]))[better]
            nn = self.nrm[k0] * (1.0 - t)[:, None] + self.nrm[k1] * t[:, None]
            nn /= np.maximum(np.hypot(nn[:, 0], nn[:, 1]), 1e-30)[:, None]
            best_n[better] = nn[better]
        dist = np.sum((xy - best_pt) * best_n, axis=1)
        return best_s, dist, best_pt, best_n

    def point_at_s(self, s: float):
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = min(max(i, 0), len(self.s) - 2)
        t = (s - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-30)
        pt = self.pts[i] + t * (self.pts[i + 1] - self.pts[i])
        nn = self.nrm[i] * (1.0 - t) + self.nrm[i + 1] * t
        nn /= max(np.hypot(*nn), 1e-30)
        return pt, nn

    def ray_intersection(self, origin, direction):
        """Closest profile point on the ray origin + t * direction (t >= 0)."""
        o = np.asarray(origin, float)
        dvec = np.asarray(direction, float)
        dvec /= np.hypot(*dvec)
        rel = self.pts - o
        t = rel @ dvec
        perp = rel[:, 0] * dvec[1] - rel[:, 1] * dvec[0]
        sign = np.sign(perp)
        cross = np.nonzero((sign[:-1] * sign[1:] <= 0) & (t[:-1] > 0))[0]
        if len(cross) == 0:
            raise ValueError("ray does not hit the profile")
        i = cross[0]
        w = abs(perp[i]) / max(abs(perp[i]) + abs(perp[i + 1]), 1e-30)
        pt = self.pts[i] + w * (self.pts[i + 1] - self.pts[i])
        s = self.s[i] + w * (self.s[i + 1] - self.s[i])
        return pt, s


# ---------------------------------------------------------------------------
# gmsh mesh
# ---------------------------------------------------------------------------

@dataclass
class MeshData:
    nodes: np.ndarray                 # (n_nodes, 2)
    elems: np.ndarray                 # (n_el, 6) T6 connectivity (gmsh order)
    elem_kind: np.ndarray             # 0 rock, 1 core, 2 ring
    elem_stage: np.ndarray            # 0 rock, k>=1 stage of core/ring
    outer_nodes: np.ndarray
    axis_nodes: np.ndarray
    exc_edges: dict                   # stage -> (n_edge, 3) node triples on the excavation line
    inner_edges: dict                 # stage -> (n_edge, 3) on the ring inner line
    h_wall: float
    r_far: float
    n_stages: int
    info: dict = field(default_factory=dict)

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elems(self):
        return self.elems.shape[0]

    def exc_edges_all(self):
        return np.vstack([e for e in self.exc_edges.values()])


def build_mesh(exc_prims: list, inner_prims: Optional[list], cut_levels: list,
               r_far: float, h_wall: float, h_far: Optional[float] = None,
               h_ring: Optional[float] = None, dist_min: Optional[float] = None,
               growth_length: Optional[float] = None, centre_y: Optional[float] = None,
               verbose: bool = False, algorithm: int = 6) -> MeshData:
    """Mesh the half domain.  ``cut_levels``: y levels separating stages (top to
    bottom); ``inner_prims`` None = no lining ring.  Element size: ``h_wall`` within
    ``dist_min`` of the excavation line, then ``h_wall (1 + (d - dist_min)/growth_length)``
    capped at ``h_far``."""
    import gmsh

    cut_levels = sorted(cut_levels, reverse=True)
    n_stages = len(cut_levels) + 1
    y_top = profile_top(exc_prims)[1]
    y_bot = profile_bottom(exc_prims)[1]
    if centre_y is None:
        centre_y = 0.5 * (y_top + y_bot)
    if h_far is None:
        h_far = 0.1 * r_far
    if h_ring is None:
        h_ring = 0.5 * h_wall
    height = y_top - y_bot
    if dist_min is None:
        dist_min = 0.5 * height
    if growth_length is None:
        growth_length = 0.35 * height

    exc = exc_prims
    inn = inner_prims
    for y0 in cut_levels:
        exc = split_at_y(exc, y0)
        if inn is not None:
            inn = split_at_y(inn, y0)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("horseshoe")
    occ = gmsh.model.occ

    ptags = {}

    def P(x, y):
        key = (round(x, 9), round(y, 9))
        if key not in ptags:
            ptags[key] = occ.addPoint(x, y, 0.0)
        return ptags[key]

    def curve(p):
        if isinstance(p, Line):
            return occ.addLine(P(p.x0, p.y0), P(p.x1, p.y1))
        s, e = p.start, p.end
        return occ.addCircleArc(P(*s), P(p.cx, p.cy), P(*e))

    def stage_of(p):
        ym = p.point(0.5)[1]
        k = 1
        for y0 in cut_levels:
            if ym < y0:
                k += 1
        return k

    exc_curves = {k: [] for k in range(1, n_stages + 1)}
    for p in exc:
        exc_curves[stage_of(p)].append(curve(p))
    inn_curves = None
    if inn is not None:
        inn_curves = {k: [] for k in range(1, n_stages + 1)}
        for p in inn:
            inn_curves[stage_of(p)].append(curve(p))

    # axis points from top to bottom
    yi_top = profile_top(inn)[1] if inn is not None else None
    yi_bot = profile_bottom(inn)[1] if inn is not None else None
    x_exc_cut = [point_at_y(exc, y0)[0] for y0 in cut_levels]
    x_inn_cut = [point_at_y(inn, y0)[0] for y0 in cut_levels] if inn is not None else None

    # horizontal cut lines: [0, x_inn] and [x_inn, x_exc] (or [0, x_exc])
    cut_in = []    # per level: curve tag of [0, x_inner]
    cut_ring = []  # per level: curve tag of [x_inner, x_exc] (None if no ring)
    for j, y0 in enumerate(cut_levels):
        if inn is not None:
            cut_in.append(occ.addLine(P(0.0, y0), P(x_inn_cut[j], y0)))
            cut_ring.append(occ.addLine(P(x_inn_cut[j], y0), P(x_exc_cut[j], y0)))
        else:
            cut_in.append(occ.addLine(P(0.0, y0), P(x_exc_cut[j], y0)))
            cut_ring.append(None)

    # axis segments
    ax = {}
    T = (0.0, centre_y + r_far)
    B = (0.0, centre_y - r_far)
    ax["outer_top"] = occ.addLine(P(*T), P(0.0, y_top))
    ax["outer_bot"] = occ.addLine(P(0.0, y_bot), P(*B))
    if inn is not None:
        ax["ring_top"] = occ.addLine(P(0.0, y_top), P(0.0, yi_top))
        ax["ring_bot"] = occ.addLine(P(0.0, yi_bot), P(0.0, y_bot))
        y_core_top, y_core_bot = yi_top, yi_bot
    else:
        y_core_top, y_core_bot = y_top, y_bot
    levels = [y_core_top] + cut_levels + [y_core_bot]
    ax_core = [occ.addLine(P(0.0, levels[k]), P(0.0, levels[k + 1])) for k in range(n_stages)]

    # outer arc (two quarters)
    pc = P(0.0, centre_y)
    outer_arcs = [occ.addCircleArc(P(*B), pc, P(r_far, centre_y)),
                  occ.addCircleArc(P(r_far, centre_y), pc, P(*T))]

    surfaces = {}  # name -> tag

    def face(curves):
        loop = occ.addCurveLoop(curves)
        return occ.addPlaneSurface([loop])

    core_bnd = inn_curves if inn is not None else exc_curves
    for k in range(1, n_stages + 1):
        cs = list(core_bnd[k])
        if k < n_stages:
            cs.append(cut_in[k - 1])
        cs.append(ax_core[k - 1])
        if k > 1:
            cs.append(cut_in[k - 2])
        surfaces[("core", k)] = face(cs)
    if inn is not None:
        for k in range(1, n_stages + 1):
            cs = list(exc_curves[k])
            if k < n_stages:
                cs.append(cut_ring[k - 1])
            else:
                cs.append(ax["ring_bot"])
            cs += list(reversed(inn_curves[k]))
            if k == 1:
                cs.append(ax["ring_top"])
            else:
                cs.append(cut_ring[k - 2])
            surfaces[("ring", k)] = face(cs)
    rock_curves = [ax["outer_top"]]
    for k in range(1, n_stages + 1):
        rock_curves += exc_curves[k]
    rock_curves += [ax["outer_bot"]] + outer_arcs
    surfaces[("rock", 0)] = face(rock_curves)
    occ.synchronize()

    # physical groups
    phys_surf = {}
    for key, tag in surfaces.items():
        phys_surf[key] = gmsh.model.addPhysicalGroup(2, [tag])
    phys_curves = {}
    for k in range(1, n_stages + 1):
        phys_curves[("exc", k)] = gmsh.model.addPhysicalGroup(1, exc_curves[k])
        if inn is not None:
            phys_curves[("inner", k)] = gmsh.model.addPhysicalGroup(1, inn_curves[k])
    phys_curves[("outer", 0)] = gmsh.model.addPhysicalGroup(1, outer_arcs)

    # size fields
    all_exc = [c for k in exc_curves for c in exc_curves[k]]
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", all_exc)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
    # size h_wall within dist_min of the excavation line, then growing in proportion to
    # h_wall (so halving h_wall refines the whole mesh), capped at h_far
    f_thr = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(
        f_thr, "F", f"Min({h_wall}*(1 + Max(F{f_dist} - {dist_min}, 0)/{growth_length}), {h_far})")
    fields = [f_thr]
    if inn is not None:
        f_ring = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumbers(f_ring, "SurfacesList",
                                         [surfaces[("ring", k)] for k in range(1, n_stages + 1)])
        gmsh.model.mesh.field.setNumber(f_ring, "VIn", h_ring)
        gmsh.model.mesh.field.setNumber(f_ring, "VOut", 1e22)
        fields.append(f_ring)
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm", algorithm)
    # angular resolution on small arcs (fillets): at most 30 deg per element,
    # applied only where the size field alone would leave fewer elements
    for dim, tag in gmsh.model.getEntities(1):
        if gmsh.model.getType(dim, tag) != "Circle":
            continue
        lo, hi = gmsh.model.getParametrizationBounds(dim, tag)
        span = float(hi[0] - lo[0])
        n_min = int(math.ceil(span / math.radians(30.0))) + 1
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(dim, tag)
        chord = math.hypot(x1 - x0, y1 - y0)
        if n_min > 2 and chord < 4.0 * h_wall:
            gmsh.model.mesh.setTransfiniteCurve(tag, n_min)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.mesh.setOrder(2)

    # --- extraction
    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, dtype=np.int64)
    coords = np.asarray(ncoords, dtype=float).reshape(-1, 3)[:, :2]
    remap = np.full(ntags.max() + 1, -1, dtype=np.int64)
    remap[ntags] = np.arange(ntags.shape[0])
    elems, kinds, stages = [], [], []
    kind_id = {"rock": 0, "core": 1, "ring": 2}
    for (kind, k), ptag in phys_surf.items():
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, ptag):
            etypes, etags, enodes = gmsh.model.mesh.getElements(2, ent)
            for et, nd in zip(etypes, enodes):
                if et != 9:
                    raise RuntimeError(f"unexpected element type {et} (expected T6 = 9)")
                conn = remap[np.asarray(nd, dtype=np.int64).reshape(-1, 6)]
                elems.append(conn)
                kinds.append(np.full(conn.shape[0], kind_id[kind], dtype=np.int8))
                stages.append(np.full(conn.shape[0], k, dtype=np.int8))
    elems = np.vstack(elems)
    kinds = np.concatenate(kinds)
    stages = np.concatenate(stages)

    def curve_edges(ptag):
        out = []
        for ent in gmsh.model.getEntitiesForPhysicalGroup(1, ptag):
            etypes, etags, enodes = gmsh.model.mesh.getElements(1, ent)
            for et, nd in zip(etypes, enodes):
                if et != 8:
                    raise RuntimeError(f"unexpected boundary element type {et}")
                out.append(remap[np.asarray(nd, dtype=np.int64).reshape(-1, 3)])
        return np.vstack(out) if out else np.zeros((0, 3), dtype=np.int64)

    exc_edges = {k: curve_edges(phys_curves[("exc", k)]) for k in range(1, n_stages + 1)}
    inner_edges = {}
    if inn is not None:
        inner_edges = {k: curve_edges(phys_curves[("inner", k)]) for k in range(1, n_stages + 1)}
    outer_nodes = np.unique(curve_edges(phys_curves[("outer", 0)]))
    gmsh.finalize()

    used = np.unique(elems)
    if used.shape[0] != coords.shape[0]:
        # compress unused nodes (should not happen, but keep the arrays tight)
        newid = np.full(coords.shape[0], -1, dtype=np.int64)
        newid[used] = np.arange(used.shape[0])
        coords = coords[used]
        elems = newid[elems]
        exc_edges = {k: newid[v] for k, v in exc_edges.items()}
        inner_edges = {k: newid[v] for k, v in inner_edges.items()}
        outer_nodes = newid[outer_nodes]
        outer_nodes = outer_nodes[outer_nodes >= 0]
    axis_nodes = np.nonzero(np.abs(coords[:, 0]) < 1e-7)[0]
    coords[axis_nodes, 0] = 0.0
    # orient every triangle counter-clockwise (gmsh orients per surface normal)
    v0, v1, v2 = coords[elems[:, 0]], coords[elems[:, 1]], coords[elems[:, 2]]
    area2 = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    flip = area2 < 0
    if flip.any():
        elems[flip] = elems[flip][:, [0, 2, 1, 5, 4, 3]]
    md = MeshData(nodes=coords, elems=elems, elem_kind=kinds, elem_stage=stages,
                  outer_nodes=outer_nodes, axis_nodes=axis_nodes, exc_edges=exc_edges,
                  inner_edges=inner_edges, h_wall=h_wall, r_far=r_far, n_stages=n_stages,
                  info={"h_far": h_far, "h_ring": h_ring, "dist_min": dist_min,
                        "growth_length": growth_length, "centre_y": centre_y,
                        "n_rock": int((kinds == 0).sum()), "n_core": int((kinds == 1).sum()),
                        "n_ring": int((kinds == 2).sum())})
    return md
