# CWFS neural material: code and data

Code and numerical data for *Physics-constrained learning of rock strength
evolution for staged tunnel analysis*.

The package contains the neural constitutive material, its synthetic training
corpus, the three retained trained models, material-point assessments, recurrent
loading histories, and the prescribed staged-excavation example. It also includes
the digitized monitoring curves used in the study, expressed in relative time.

## Download

Download the complete code and data archive from the [v1.0.0 release](https://github.com/math-sudu/cwfs-code-data/releases/tag/v1.0.0).
Alternatively, clone this repository to obtain the same code, model weights and data.

## Installation

Use Python 3.11 or later in a virtual environment. From the extracted
`cwfs-code-data` directory:

```text
python -m pip install -r requirements.txt
python summarize_results.py
python -m pytest compute/tests/test_neural_material.py -q
```

The release was checked with Python 3.13, NumPy 2.4.4, SciPy 1.16.2,
PyTorch 2.8.0, Gmsh 4.15.2 and pytest 8.4.2. CPU execution is supported;
training uses CUDA when available. Install a PyTorch build appropriate for your
hardware if GPU training is required. On Linux, Gmsh may also require the system
OpenGL library, commonly supplied by `libGLU`.

All commands below run from the package root. Existing results are supplied;
the commands show how to regenerate them. Use `reproduced/` for new outputs.

## Contents and units

| Location | Contents |
| --- | --- |
| `compute/src/cwfs_inv/` | CWFS return mapping, neural material, mesh and staged FE solver |
| `compute/configs/material.json` | Six-parameter bounds and elastic/dilatancy constants |
| `compute/configs/s2_prescribed.json` | Complete numerical geometry, support, stress, stages and solver settings |
| `compute/runs/principal_material/corpus.npz` | 428032 training states and 65536 held-out states |
| `compute/runs/principal_material/*/best.pt` | Production, axisymmetric and general matched-sample checkpoints |
| `results/constitutive/principal_material/` | Training histories, assessments and recurrent paths for those models |
| `results/horseshoe_fe/neural_s2_prescribed.json` | Completed staged-solve history and reported quantities |
| `results/figure_fields/neural_s2_prescribed.*` | Mesh, five accepted field states, events and material metadata |
| `data/` | Digitized monitoring curves and final observations for the example section |

Stress and elastic modulus use MPa, length and displacement use m, friction and
dilatancy angles use degrees, and strains are dimensionless. Reported monitoring
and summary displacements use mm. Stress is compression positive.

The six material inputs are peak cohesion, residual cohesion, peak friction
angle, residual friction angle, cohesion strain threshold and friction strain
threshold, in that order. The shared constants are E = 560 MPa, nu = 0.30 and
psi = 5 degrees.

In `corpus.npz`, `trial` and `stress` are ordered principal stresses;
`gamma_old` and `gamma` are the preceding and accepted accumulated plastic shear;
`theta` contains the six inputs; `heldout` identifies the separate material test
set; and `region` identifies elastic/face/edge/apex returns. `elastic_params`
stores E, nu and psi. The arrays contain only numerical values and can be read
with `numpy.load(..., allow_pickle=False)`.

In each path NPZ, stress has shape `(35, 400, 9, 4)` and plastic histories have
shape `(35, 400, 9)`. The first three materials are anchors and the remaining 32
are held out. The nine paths combine pressures 2, 5 and 10 MPa with deviatoric
loading, unloading/reloading and rotating shear, in that order within each
pressure. Stress components are `(xx, yy, zz, xy)` and strain increments use
engineering shear. The supplied JSON summaries describe complete histories;
`summarize_results.py` separately computes the strength-evolution metrics used in
the manuscript, using the same step selection and normalization as its figure.

In the field NPZ, `u` stores nodal `(ux, uy)`; `sig` stores integration-point
`(xx, yy, zz, xy)`; and `state` stores accumulated plastic shear, return-region
index and plastic volumetric strain. `active` must be applied before interpreting
excavated elements. `elem_mat` is 0 for unreinforced rock, 1 for primary support
and 2 for bolted rock. `nodes`, `elems`, `ip_xy`, `ip_weight` and `exc_edges`
define the T6 mesh, quadrature locations/areas and excavation edges. The JSON
events identify heading support installation, before bench excavation, before
invert excavation, ring closure and full release.

## Reproduce material evaluation

```text
python compute/scripts/evaluate_principal_material.py --checkpoint compute/runs/principal_material/production/best.pt --out reproduced/production_assessment.json
python compute/scripts/evaluate_neural_paths.py --checkpoint compute/runs/principal_material/production/best.pt --out reproduced/production_paths.json
```

Replace `production` with `ablation_axisymmetric` or
`ablation_general_matched` to evaluate either comparison model. The recurrent
evaluation generates 315 paths, including 27 anchor and 288 held-out paths.
Only histories with selected evolving plastic steps contribute to the
strength-evolution percentile; this subset contains 278 held-out paths.

## Regenerate the corpus and train

The shipped corpus is synthetic, generated from the included CWFS material law.
The following commands regenerate it and train separate model directories,
preserving the supplied checkpoints:

```text
python compute/scripts/train_principal_material.py corpus
python compute/scripts/train_principal_material.py train --name reproduced_production
python compute/scripts/train_principal_material.py train --name reproduced_axisymmetric --pool axisymmetric
python compute/scripts/train_principal_material.py train --name reproduced_general_matched --pool matched
```

The defaults use seed 20260917, batch size 4096 and 300 epochs. The saved
checkpoints contain the best selected network, not necessarily the final epoch.
They include tensor weights, normalization, material constants and architectural
metadata. Optimizer state and machine-specific provenance have been removed.

## Run the staged finite-element example

```text
python compute/scripts/run_neural_s2.py --checkpoint compute/runs/principal_material/production/best.pt --out reproduced/neural_s2_prescribed.json --fields reproduced/neural_s2_prescribed.npz
```

The numerical geometry is provided directly in local model coordinates, with
the support-inner crown at `(0, 0)` and y positive upwards. The provided profile
produces the same arcs as the original numerical configuration. The solver uses
the right half-domain, three excavation stages, support installation at release
0.30 and stage-end release values 0.60, 0.85 and 1.00. A full staged nonlinear
solve is substantially more expensive than reading the supplied results.

`data/s2_monitoring_digitized.csv` contains 91 digitized plot anchors, not raw
instrument exports. `elapsed_days` is the retained relative time coordinate;
calendar dates, chainages and source-document identifiers have been removed.
The display points in the manuscript can be reconstructed by piecewise-linear
interpolation within each channel. GD denotes crown settlement; SL1–SL3 denote
the convergence chords. The observations document is in mm and days.

## Sharing scope

This archive contains the numerical inputs and outputs needed for the retained
constitutive and staged-analysis results. Site names, absolute locations,
calendar dates, internal source records, local filesystem paths and private
project dependencies are omitted. The geometric coordinates are model-relative,
not geographic coordinates. Raw engineering reports, original CAD files,
photographs, manuscript drafts, correspondence, credentials and Git history are
not part of this release. The numerical arrays and trained weights are preserved;
metadata and configuration loading have been adapted for a portable package.
