"""Recurrent tensor-state tests for the same neural material used by S2 FE."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compute/src'))
from cwfs_inv.fe_materials import CWFSMaterial
from cwfs_inv.neural_material import NeuralCWFSMaterial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--anchors-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    corpus=json.loads((ROOT/'results/constitutive/principal_material/corpus.json').read_text())
    vectors=np.vstack(([1,.2,20,30,.01,.015],[8,1,25,35,.008,.012],
                       [.5,.05,40,45,.002,.004],corpus['heldout_vectors']))
    if args.anchors_only:
        vectors=vectors[:3]
    histories=('deviatoric_loading','unload_reload','rotating_shear')
    pressures=(2.,5.,10.)
    nsteps=400
    increments=np.zeros((nsteps,9,4))
    for i in range(nsteps):
        for j in range(9):
            mode=j%3
            if mode==0:
                increments[i,j]=[.0002,-.0002,0,0]
            elif mode==1:
                sign=-1 if 200<=i<300 else 1
                increments[i,j]=np.array([.0002,-.0002,0,.0001])*sign
            else:
                angle=2*np.pi*i/nsteps
                increments[i,j]=[.0002*np.cos(angle),-.0002*np.cos(angle),0,.0004*np.sin(angle)]
    rows=[]
    collected={k:[] for k in ('stress_reference','stress_neural','gamma_reference','gamma_neural',
                               'volume_reference','volume_neural')}
    started=time.time()
    for it,theta in enumerate(vectors):
        spec=dict(E=560.,nu=.3,psi_deg=5.,**dict(zip(
            ('c_peak','c_res','phi_peak_deg','phi_res_deg','gp_c','gp_phi'),theta)))
        teacher=CWFSMaterial(**spec)
        neural=NeuralCWFSMaterial(**spec,checkpoint=args.checkpoint)
        stress=np.zeros((9,4));stress[:,:3]=np.repeat(pressures,3)[:,None]
        sn=stress.copy();state=teacher.initial_state(9);gn=neural.initial_state(9)
        data={k:[] for k in collected}
        for de in increments:
            truth=teacher.update(de,stress,state)
            pred=neural.update(de,sn,gn)
            stress,state=truth.sig,truth.state
            sn,gn=pred.sig,pred.state
            for key,value in zip(data,(stress,sn,state[:,0],gn[:,0],state[:,2],gn[:,2])):
                data[key].append(value.copy())
        data={k:np.stack(v) for k,v in data.items()}
        for key in collected:
            collected[key].append(data[key])
        def strength(g):
            return (theta[0]+(theta[1]-theta[0])*np.minimum(g/theta[4],1),
                    theta[2]+(theta[3]-theta[2])*np.minimum(g/theta[5],1))
        ct,pt=strength(data['gamma_reference']);cn,pn=strength(data['gamma_neural'])
        for j in range(9):
            diff=data['stress_neural'][:,j]-data['stress_reference'][:,j]
            scale=max(1.,abs(data['stress_reference'][:,j]).max())
            rows.append({'vector_index':it,'set':'anchor' if it<3 else 'heldout','theta':theta.tolist(),
                         'pressure_MPa':pressures[j//3],'history':histories[j%3],
                         'gamma_final_reference':float(data['gamma_reference'][-1,j]),
                         'gamma_final_neural':float(data['gamma_neural'][-1,j]),
                         'stress_rmse_MPa':float(np.sqrt(np.mean(diff**2))),
                         'stress_relative_rmse':float(np.sqrt(np.mean(diff**2))/scale),
                         'gamma_rmse':float(np.sqrt(np.mean((data['gamma_neural'][:,j]-data['gamma_reference'][:,j])**2))),
                         'plastic_volume_rmse':float(np.sqrt(np.mean((data['volume_neural'][:,j]-data['volume_reference'][:,j])**2))),
                         'cohesion_rmse_MPa':float(np.sqrt(np.mean((cn[:,j]-ct[:,j])**2))),
                         'friction_rmse_deg':float(np.sqrt(np.mean((pn[:,j]-pt[:,j])**2)))})
        print(f'Completed recurrent material {it+1}/{len(vectors)}',flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out.with_suffix('.npz'),**{k:np.stack(v) for k,v in collected.items()},
                        increments=increments,theta=vectors,pressures=pressures)
    summary={}
    for name in histories:
        subset=[r for r in rows if r['history']==name and (args.anchors_only or r['set']=='heldout')]
        summary[name]={'paths':len(subset)}
        for metric in ('stress_relative_rmse','gamma_rmse','cohesion_rmse_MPa','friction_rmse_deg'):
            vals=[r[metric] for r in subset]
            summary[name][metric]={'median':float(np.median(vals)),'p90':float(np.quantile(vals,.9)),
                                  'max':float(np.max(vals))}
    payload={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
             'paths':len(rows),'steps':nsteps,'summary':summary,'rows':rows,'wall_s':time.time()-started}
    args.out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
