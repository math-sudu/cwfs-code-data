"""Recover the reported metrics from the supplied numeric results."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/constitutive/principal_material"


def evolving_metrics(name):
    with np.load(RESULTS / f"{name}_paths.npz", allow_pickle=False) as data:
        reference = data["stress_reference"][3:]
        neural = data["stress_neural"][3:]
        gamma = data["gamma_reference"][3:]
        gamma_neural = data["gamma_neural"][3:]
        previous = np.concatenate([np.zeros_like(gamma[:, :1]), gamma[:, :-1]], axis=1)
        threshold = data["theta"][3:, 4:6].max(axis=1)[:, None, None]
        evolving = (previous < threshold) & (gamma > previous)
        stress_errors, gamma_errors = [], []
        for material in range(len(reference)):
            for path in range(reference.shape[2]):
                selected = evolving[material, :, path]
                if not selected.any():
                    continue
                ref = reference[material, :, path]
                stress_errors.append(np.sqrt(np.mean((neural[material, selected, path] - ref[selected]) ** 2)) /
                                     max(np.abs(ref).max(), 1.0))
                gamma_errors.append(np.sqrt(np.mean((gamma_neural[material, selected, path] - gamma[material, selected, path]) ** 2)))
    return {
        "heldout_paths": int(reference.shape[0] * reference.shape[2]),
        "paths_with_evolving_steps": len(stress_errors),
        "evolving_stress_p90_percent": float(np.quantile(stress_errors, 0.9) * 100),
        "evolving_plastic_shear_p90": float(np.quantile(gamma_errors, 0.9)),
    }


if __name__ == "__main__":
    models = ("ablation_axisymmetric", "ablation_general_matched", "production")
    summary = {name: evolving_metrics(name) for name in models}
    assessment = json.loads((RESULTS / "production_assessment.json").read_text())
    summary["production"]["heldout_single_step"] = assessment["summary"]["heldout_materials"]
    s2 = json.loads((ROOT / "results/horseshoe_fe/neural_s2_prescribed.json").read_text())
    summary["s2"] = {"completed": s2["completed"], "SL2_post_install_mm": s2["quantities"]["SL2_post_install_mm"]}
    print(json.dumps(summary, indent=2))
