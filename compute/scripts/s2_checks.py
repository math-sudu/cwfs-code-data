"""Numerical result extraction and committed-state checks."""
import numpy as np

def quantities(result):
    rd = result["readings"]
    observations = [rd["crown_settlement"], *rd["chord_shortening"]]
    q = {}
    for obs in observations:
        for key in ("total", "post_install"):
            value = obs[key]
            q[f"{obs['channel']}_{key}_mm"] = None if value is None else 1000 * value
    pz = result["plastic_zone"]
    for key in ("gamma_p_integral_m2", "plastic_area_m2", "max_gamma_p",
                "max_plastic_slip_m", "depth_max_m"):
        q[key] = pz[key]
    for name, v in pz["depth"].items():
        q[f"plastic_depth_{name}_m"] = v
    for name, st in result["lining"]["stations"].items():
        q[f"{name}_pressure_MPa"] = st["pressure"]
    return q


def verify_result(result, model):
    """Check the committed states, installation zeros and independent area sums."""
    history = [e for e in result["history"] if e.get("event") != "excavate"]
    ratios = [e["equilibrium"]["residual_over_tolerance"] for e in history]
    zeros = {}
    for event in history:
        if event.get("event") == "install":
            stage = event["stage"]
            for name, point in model.points.items():
                if point["stage"] == stage:
                    zeros[point["channel"]] = bool(abs(event["readings_post_install"][name]) <= 1e-12)
    rock = model.active & (model.elem_mat != 1)
    gp = model.state[rock, :, 0]
    weights = model.kin.wdet[rock]
    pz = result["plastic_zone"]
    integral = float((gp * weights).sum())
    rock_state = model.state[rock, :, :model.rock.n_state].reshape(-1, model.rock.n_state)
    mask = model.rock.plastic_mask(rock_state)
    area = float(weights.ravel()[mask].sum())
    checks = dict(finite_committed_state=bool(all(np.isfinite(a).all()
                  for a in (model.u, model.sig, model.state))),
                  positive_quadrature=bool((model.kin.wdet > 0).all()),
                  committed_equilibrium=bool(all(v <= 1 + 1e-9 for v in ratios)),
                  max_committed_residual_over_tolerance=max(ratios, default=None),
                  installation_zeros=zeros,
                  plastic_integral_matches=bool(np.isclose(integral, pz["gamma_p_integral_m2"], rtol=1e-9)),
                  plastic_area_matches=bool(np.isclose(area, pz["plastic_area_m2"], rtol=1e-9)))
    required = ("finite_committed_state", "positive_quadrature", "committed_equilibrium",
                "plastic_integral_matches", "plastic_area_matches")
    checks["passed"] = all(checks[k] for k in required) and all(zeros.values())
    return checks
