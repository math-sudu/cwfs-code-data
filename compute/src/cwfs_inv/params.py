"""Material parameter bounds and sampling for the neural material."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import source_stack

THETA_FIELDS = ('c_peak_MPa', 'c_res_MPa', 'phi_peak_deg', 'phi_res_deg', 'gamma_p_c', 'gamma_p_phi')

@dataclass
class ThetaSoftening:
    c_peak_MPa: float
    c_res_MPa: float
    phi_peak_deg: float
    phi_res_deg: float
    gamma_p_c: float
    gamma_p_phi: float

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in THETA_FIELDS], dtype=float)

    @classmethod
    def from_array(cls, arr) -> 'ThetaSoftening':
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size != len(THETA_FIELDS):
            raise ValueError(f'theta array must have {len(THETA_FIELDS)} entries, got {arr.size}')
        return cls(**{f: float(v) for f, v in zip(THETA_FIELDS, arr)})

    def physical_ok(self) -> bool:
        """CWFS ordering constraints (config theta_softening_box.constraints)."""
        return self.c_res_MPa <= self.c_peak_MPa and self.phi_res_deg >= self.phi_peak_deg and (self.gamma_p_c > 0.0) and (self.gamma_p_phi > 0.0)

@dataclass
class PriorBox:
    """Axis-aligned bounds over the 6 softening parameters."""
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def from_config(cls, cfg: dict | None=None) -> 'PriorBox':
        cfg = cfg or source_stack.load_config()
        box = cfg['theta_softening_box']
        lo = np.array([box[f][0] for f in THETA_FIELDS], dtype=float)
        hi = np.array([box[f][1] for f in THETA_FIELDS], dtype=float)
        return cls(lo=lo, hi=hi)

    def normalize(self, theta_arr: np.ndarray) -> np.ndarray:
        """Physical -> [-1, 1] per axis (network conditioning space)."""
        return 2.0 * (np.asarray(theta_arr, dtype=float) - self.lo) / (self.hi - self.lo) - 1.0

    def denormalize(self, unit_arr: np.ndarray) -> np.ndarray:
        return (np.asarray(unit_arr, dtype=float) + 1.0) * (self.hi - self.lo) / 2.0 + self.lo

    def contains(self, theta_arr: np.ndarray, tol: float=1e-09) -> bool:
        a = np.asarray(theta_arr, dtype=float)
        return bool(np.all(a >= self.lo - tol) and np.all(a <= self.hi + tol))

    def corners(self) -> np.ndarray:
        """All 2^6 = 64 box corners, row-wise."""
        grids = np.meshgrid(*[np.array([l, h]) for l, h in zip(self.lo, self.hi)], indexing='ij')
        return np.stack([g.reshape(-1) for g in grids], axis=1)

    def sobol(self, n: int, seed: int) -> np.ndarray:
        """Scrambled-Sobol interior samples (physically admissible rows only).

        Rows violating the CWFS ordering constraints are resampled from the
        continuation of the sequence so the count stays exact.
        """
        from scipy.stats import qmc
        eng = qmc.Sobol(d=len(THETA_FIELDS), scramble=True, seed=seed)
        rows = []
        while len(rows) < n:
            batch = self.lo + eng.random(max(n, 16)) * (self.hi - self.lo)
            for r in batch:
                if ThetaSoftening.from_array(r).physical_ok():
                    rows.append(r)
                    if len(rows) == n:
                        break
        return np.array(rows)

    def as_dict(self) -> dict:
        return {f: [float(l), float(h)] for f, l, h in zip(THETA_FIELDS, self.lo, self.hi)}
