"""Run the S2 boundary-value problem with the learned material at rock points."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compute/src'))
from cwfs_inv.horseshoe_fe import RunConfig, StagedModel, write_json
import s2_checks as h10


class NeuralFieldRecorder(StagedModel):
    def __init__(self,cfg,progress):
        self.frames=[]
        self.events=[]
        self.progress=progress
        super().__init__(cfg,verbose=True)

    def _record(self,stage_no,lam,extra):
        super()._record(stage_no,lam,extra)
        write_json({'history':self.history,'last_stage':stage_no,'last_lambda':lam},str(self.progress))
        if extra.get('event')=='excavate':
            return
        stages=self.cfg.stages
        selected={(1,stages[0]['lambda_install']):'heading_install',
                  (1,stages[0]['lambda_end']):'before_bench'}
        if len(stages)>1:
            selected[(2,stages[1]['lambda_end'])]='before_invert'
        if len(stages)>2:
            selected[(3,stages[2]['lambda_install'])]='ring_closure'
            selected[(3,stages[2]['lambda_end'])]='full_release'
        for (stage,value),name in selected.items():
            if stage==stage_no and abs(value-lam)<1e-10:
                self.frames.append({k:getattr(self,k).copy() for k in ('u','sig','state','active','elem_mat')})
                self.frames[-1]['u']=self.frames[-1]['u'].reshape(-1,2)
                self.events.append({'name':name,'stage':stage,'lambda':lam,'readings':self.history[-1]})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--schedule',choices=['prescribed'],default='prescribed')
    p.add_argument('--heading-only',action='store_true')
    p.add_argument('--fields',type=Path)
    p.add_argument('--line-search-steps',type=int,default=6)
    p.add_argument('--max-iter',type=int,default=25)
    p.add_argument('--resume',type=Path,help='Continue a stopped state saved by this script')
    args=p.parse_args()
    torch.set_num_threads(4)
    source=ROOT/'compute/configs/s2_prescribed.json'
    cfg=RunConfig.from_dict(json.loads(source.read_text(encoding='utf-8')))
    cfg.rock.update(model='neural_cwfs',checkpoint=str(args.checkpoint))
    cfg.solver['line_search_steps']=args.line_search_steps
    cfg.solver['max_iter']=args.max_iter
    cfg.label='neural_s2_'+args.schedule
    if args.heading_only:
        cfg.stages=cfg.stages[:1]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    progress=ROOT/'compute/runs/principal_material'/(args.out.stem+'_progress.json')
    started=time.time()
    if args.resume:
        with args.resume.open('rb') as stream:
            snapshot=pickle.load(stream)
        saved=snapshot['cfg'].to_dict();requested=cfg.to_dict()
        # Only numerical iteration settings may change during continuation.
        for value in (saved,requested):
            for key in ('max_iter','line_search_steps'):
                value['solver'].pop(key,None)
        if saved!=requested:
            raise ValueError('Saved and requested physical configurations differ')
        model=NeuralFieldRecorder.__new__(NeuralFieldRecorder)
        model.__dict__.update(snapshot)
        model.cfg=cfg
        model.progress=progress
    else:
        model=NeuralFieldRecorder(cfg,progress)
    result=model.run(resume=bool(args.resume))
    checks=h10.verify_result(result,model)
    payload={'config':cfg.to_dict(),'completed':result['completed'],'stopped':result['stopped'],
             'result':result,'quantities':h10.quantities(result),'verification':checks,
             'wall_time_s':model.wall_time,'configuration_source':str(source.relative_to(ROOT)),
             'constitutive_role':'Neural strength evolution drives stress and plastic state in the global equilibrium solve',
             'sources_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in
                 [Path(__file__),ROOT/'compute/src/cwfs_inv/neural_material.py',
                  ROOT/'compute/src/cwfs_inv/horseshoe_fe.py',ROOT/'compute/src/cwfs_inv/fe_materials.py']}}
    write_json(payload,str(args.out))
    if args.fields and model.frames:
        arrays={k:np.stack([f[k] for f in model.frames]) for k in model.frames[0]}
        arrays.update(nodes=model.mesh.nodes,elems=model.mesh.elems,elem_kind=model.mesh.elem_kind,
                      elem_stage=model.mesh.elem_stage,ip_xy=model.kin.ip_xy,ip_weight=model.kin.wdet,
                      exc_edges=model.mesh.exc_edges_all())
        args.fields.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.fields,**arrays)
        meta={'case':cfg.label,'engine':'Neural CWFS finite elements','config':cfg.to_dict(),
              'events':model.events,'history':model.history,'readings':result['readings'],
              'materials':result['materials'],'completed':result['completed'],
              'output_sha256':hashlib.sha256(args.fields.read_bytes()).hexdigest(),
              'sources_sha256':payload['sources_sha256']}
        write_json(meta,str(args.fields.with_suffix('.json')))
    print(json.dumps({'completed':result['completed'],'quantities':payload['quantities'],
                      'verification':checks}),flush=True)
    if not result['completed'] or not checks['passed']:
        # Preserve the actual committed state for diagnosis and continuation.
        snapshot=dict(model.__dict__)
        snapshot['_elastic_lu_cache']=None
        with (progress.parent/(args.out.stem+'_stopped.pkl')).open('wb') as stream:
            pickle.dump(snapshot,stream)
        raise RuntimeError('Neural S2 calculation did not complete with accepted equilibrium states')


if __name__=='__main__':main()
