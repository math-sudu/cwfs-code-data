"""Generate general principal-state CWFS labels and train the FE material.

The old triaxial checkpoints are preserved. This corpus samples trial stress,
Lode position, plastic history and six strength parameters independently of
tunnel measurements. The S2 material is a known conditioning vector, not a
fit to monitored displacements. Held-out material vectors are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'compute/src'))
from cwfs_inv.fe_materials import CWFSMaterial
from cwfs_inv.params import PriorBox
from cwfs_inv.source_stack import load_config
from cwfs_inv.neural_material import PrincipalCWFS, strengths

RUN = ROOT/'compute/runs/principal_material'
RESULTS = ROOT/'results/constitutive/principal_material'
FROZEN = load_config()['frozen_element_params']
MATERIAL = dict(E=FROZEN['E_MPa'], nu=FROZEN['nu'], psi_deg=FROZEN['psi_deg'])
S2 = np.array([1., .2, 20., 30., .01, .015])


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def material(theta):
    return CWFSMaterial(**MATERIAL, **dict(zip(
        ('c_peak','c_res','phi_peak_deg','phi_res_deg','gp_c','gp_phi'), theta)))


def samples(theta, n, rng):
    mat = material(theta)
    gp = rng.uniform(0, 1.4, n)*max(theta[4:6])
    gp[:n//4] = 0
    N, sc, *_ = mat.strength_values(gp)
    # Resolve small load increments as well as FE line-search overshoots.
    minor = rng.uniform(-3, 32, n)
    over = 10**rng.uniform(-5, 1.5, n)
    major = N*minor+sc+over
    frac = rng.uniform(0, 1, n)
    frac[:n//10] = 0
    frac[n//10:n//5] = 1
    trial = np.sort(np.column_stack((major, minor+(major-minor)*frac, minor)), axis=1)[:, ::-1].copy()
    # Tension/apex and large mixed-component trials are included explicitly.
    count = n//8
    trial[-count:] = np.sort(rng.uniform(-12, 65, (count, 3)), axis=1)[:, ::-1]
    deps = np.column_stack((trial, np.zeros(n))) @ np.linalg.inv(mat.C).T
    state = mat.initial_state(n)
    state[:, 0] = gp
    out = mat.update(deps, np.zeros_like(deps), state)
    oracle, _, _ = mat._fixed_strength_map(trial,out.state[:,0])
    if not np.allclose(oracle,out.sig[:,:3],rtol=1e-8,atol=1e-8):
        raise RuntimeError('A CWFS label is not on its admissible evolved-strength return')
    return {'trial': trial, 'gamma_old': gp, 'theta': np.tile(theta, (n, 1)),
            'stress': out.sig[:, :3], 'gamma': out.state[:, 0],
            'region': out.state[:, 1]}


def corpus(args):
    rng = np.random.default_rng(args.seed)
    box = PriorBox.from_config()
    bolted = S2.copy()
    bolt_stress = 235*np.pi*.032**2/4/(1.2*.6)
    bolted[:2] += .5*np.sqrt((1+np.sin(np.deg2rad(S2[2:4])))/
                             (1-np.sin(np.deg2rad(S2[2:4]))))*bolt_stress
    training = np.vstack((box.corners(), box.sobol(args.materials, args.seed), S2, bolted,
                          [8.,1.,25.,35.,.008,.012]))
    heldout = box.sobol(32, args.seed+1)
    chunks = []
    counts = {}
    for i, theta in enumerate(np.vstack((training, heldout))):
        n = args.samples * (8 if i in (len(training)-3, len(training)-2) else 1)
        chunk = samples(theta, n, rng)
        chunk['heldout'] = np.full(n, i >= len(training))
        chunks.append(chunk)
        if (i+1) % 16 == 0:
            print(f'generated {i+1}/{len(training)+len(heldout)} material vectors', flush=True)
    data = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    RUN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUN/'corpus.npz', **data, elastic_params=np.array(list(MATERIAL.values())))
    for group in (False, True):
        keep = data['heldout']==group
        counts['heldout' if group else 'training'] = {
            'samples': int(keep.sum()),
            'regions': {str(int(r)):int((data['region'][keep]==r).sum()) for r in np.unique(data['region'][keep])}}
    write_json(RESULTS/'corpus.json', {'material':MATERIAL,'seed':args.seed,
               'training_vectors':training.tolist(),'heldout_vectors':heldout.tolist(),
               'sampling':'General sorted trial stress, variable intermediate principal stress, accumulated plastic history; no displacement observations',
               'counts':counts,'label_source_sha256':hashlib.sha256(
                   (ROOT/'compute/src/cwfs_inv/fe_materials.py').read_bytes()).hexdigest()})


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(RUN/'corpus.npz')
    if not np.array_equal(z['elastic_params'], list(MATERIAL.values())):
        raise ValueError('Corpus material differs')
    box = PriorBox.from_config()
    model = PrincipalCWFS(box.lo, box.hi, **MATERIAL).to(device)
    arrays = {k:torch.as_tensor(z[k],dtype=torch.float32,device=device) for k in
              ('trial','gamma_old','theta','stress','gamma')}
    # Fit to the log ratio of the accumulated-plastic increment to the
    # fixed-old-strength update, plus the actual returned state and yield loss.
    with torch.no_grad():
        _, _, _, _, _, base = model(arrays['trial'],arrays['gamma_old'],arrays['theta'])
        delta = arrays['gamma']-arrays['gamma_old']
        trainable = (base > 1e-9) & (delta > 1e-9) & (arrays['gamma_old']<arrays['theta'][:,4:6].amax(1))
        target = torch.log(delta.clamp_min(1e-12)/base.clamp_min(1e-12))
    pool = np.flatnonzero(trainable.cpu().numpy() & ~z['heldout'])
    equal_principal = np.isclose(z['trial'][:,0],z['trial'][:,1],rtol=1e-10,atol=1e-10) | np.isclose(
        z['trial'][:,1],z['trial'][:,2],rtol=1e-10,atol=1e-10)
    axisymmetric_pool = pool[equal_principal[pool]]
    if args.pool == 'axisymmetric':
        pool = axisymmetric_pool
    elif args.pool == 'matched':
        pool = np.random.default_rng(20260917).choice(pool,len(axisymmetric_pool),replace=False)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pool)
    nv = max(1,len(pool)//5)
    val = torch.as_tensor(pool[:nv],device=device)
    train_idx = torch.as_tensor(pool[nv:],device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-5)
    folder = RUN/args.name
    folder.mkdir(exist_ok=True)
    best = float('inf')
    history = []
    started = time.time()
    source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   [Path(__file__),ROOT/'compute/src/cwfs_inv/neural_material.py']}
    corpus_hash=hashlib.sha256((RUN/'corpus.npz').read_bytes()).hexdigest()

    def loss(idx):
        out,dp,gp,reg,ratio,_ = model(arrays['trial'][idx],arrays['gamma_old'][idx],arrays['theta'][idx])
        th = arrays['theta'][idx]
        c,phi = strengths(gp,th)
        f = out.amax(1)*(1-torch.sin(phi))-out.amin(1)*(1+torch.sin(phi))-2*c*torch.cos(phi)
        stress_scale = arrays['stress'][idx].abs().amax(1).clamp_min(1)
        log_loss = (ratio-target[idx]).square().mean()
        stress_loss = ((out-arrays['stress'][idx])/stress_scale[:,None]).square().mean()
        gamma_loss = ((gp-arrays['gamma'][idx])/th[:,4:6].amax(1)).square().mean()
        yield_loss = (torch.relu(f)/stress_scale).square().mean()
        return log_loss + stress_loss + .25*gamma_loss + args.yield_weight*yield_loss

    for epoch in range(1,args.epochs+1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx),device=device)]
        totals = torch.zeros((),device=device)
        for batch in perm.split(args.batch):
            optimizer.zero_grad(set_to_none=True)
            value = loss(batch)
            if not torch.isfinite(value):
                raise RuntimeError(f'Nonfinite training loss at epoch {epoch}')
            value.backward()
            optimizer.step()
            totals += value.detach()*len(batch)
        scheduler.step()
        if epoch==1 or epoch%10==0 or epoch==args.epochs:
            model.eval()
            with torch.no_grad():
                score = sum(loss(b)*len(b) for b in val.split(args.batch))/len(val)
            row = dict(epoch=epoch,train_loss=float(totals/len(perm)),val_loss=float(score),wall_s=time.time()-started)
            history.append(row)
            saved = {'format':'principal_cwfs_v1','material':MATERIAL,'lo':box.lo.tolist(),'hi':box.hi.tolist(),
                     'hidden':model.hidden,'state_dict':model.state_dict(),'epoch':epoch,'seed':args.seed,
                     'optimizer':optimizer.state_dict(),'history':history,'yield_weight':args.yield_weight,
                     'pool':args.pool,
                     'source_sha256':source_hashes,'corpus_sha256':corpus_hash,
                     'architecture':'learned plastic-shear increment ratio; differentiable fixed-strength face/edge/apex return',
                     'input_scaling':{'stress':'mean / 30 MPa, equivalent deviatoric stress / 30 MPa, normalized third invariant',
                                      'plastic_history':'current cohesion and friction thresholds',
                                      'increment':'log(fixed-strength plastic shear / maximum threshold) / 6',
                                      'parameters':'prior-box affine [-1,1]'}}
            torch.save(saved,folder/'last.tmp')
            (folder/'last.tmp').replace(folder/'last.pt')
            if float(score)<best:
                best = float(score)
                torch.save(saved,folder/'best.tmp')
                (folder/'best.tmp').replace(folder/'best.pt')
            write_json(RESULTS/(args.name+'_training.json'),{'model':args.name,'material':MATERIAL,
                       'train_samples':len(train_idx),'validation_samples':len(val),'seed':args.seed,
                       'epochs_completed':epoch,'best_val_loss':best,'history':history,
                       'source_sha256':source_hashes,'corpus_sha256':corpus_hash,'pool':args.pool})
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['corpus','train'])
    p.add_argument('--seed',type=int,default=20260917)
    p.add_argument('--materials',type=int,default=128)
    p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--batch',type=int,default=4096)
    p.add_argument('--name',default='production')
    p.add_argument('--yield-weight',type=float,default=1.)
    p.add_argument('--pool',choices=['general','axisymmetric','matched'],default='general')
    args=p.parse_args()
    globals()[args.mode](args)
