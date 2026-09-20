"""Evaluate the general neural material against its constitutive training law."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compute/src'))
from cwfs_inv.neural_material import load_principal_checkpoint, strengths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,saved=load_principal_checkpoint(args.checkpoint,560.,.3,5.,device)
    z=np.load(ROOT/'compute/runs/principal_material/corpus.npz')
    errors={k:[] for k in ('stress','plastic_increment','gamma','cohesion','friction','yield')}
    with torch.no_grad():
        for start in range(0,len(z['trial']),8192):
            sl=slice(start,start+8192)
            tr,go,theta,target,gt=[torch.as_tensor(z[k][sl],dtype=torch.float64,device=device)
                                 for k in ('trial','gamma_old','theta','stress','gamma')]
            pred,dp,gp,reg,ratio,base=model(tr,go,theta)
            dpt=(tr-target)@model.Cinv.T
            cp,pp=strengths(gp,theta);ct,pt=strengths(gt,theta)
            f=pred.amax(1)*(1-torch.sin(pp))-pred.amin(1)*(1+torch.sin(pp))-2*cp*torch.cos(pp)
            for key,value in zip(errors,((pred-target).square().mean(1),(dp-dpt).square().mean(1),
                                        (gp-gt).square(),(cp-ct).square(),torch.rad2deg(pp-pt).square(),torch.relu(f))):
                errors[key].append(value.cpu().numpy())
    errors={k:np.concatenate(v) for k,v in errors.items()}
    groups={'all':np.ones(len(z['trial']),bool),'heldout_materials':z['heldout'],
            's2':np.all(np.isclose(z['theta'],[1.,.2,20.,30.,.01,.015]),axis=1)}
    summary={}
    for name,mask in groups.items():
        row={'samples':int(mask.sum())}
        for key,values in errors.items():
            row[key+'_rmse' if key!='yield' else 'yield_p90_MPa']=float(np.sqrt(values[mask].mean()) if key!='yield' else np.quantile(values[mask],.9))
        summary[name]=row
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'checkpoint':str(args.checkpoint),'epoch':saved['epoch'],
                                   'summary':summary},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
