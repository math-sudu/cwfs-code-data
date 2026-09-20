"""FE contract, objective tensor update, branch flow and actual-map tangent."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from cwfs_inv.fe_materials import MohrCoulombEPP, CWFSMaterial, make_material, _rotation_matrices
from cwfs_inv.neural_material import PrincipalCWFS, NeuralCWFSMaterial, fixed_strength_return
from cwfs_inv.params import PriorBox


@pytest.fixture
def material(tmp_path):
    box = PriorBox.from_config()
    torch.manual_seed(17)
    model = PrincipalCWFS(box.lo, box.hi)
    with torch.no_grad():
        model.net[-1].weight.fill_(.01)
        model.net[-1].bias.fill_(.15)
    path = tmp_path/'model.pt'
    torch.save({'format':'principal_cwfs_v1','material':dict(E=560.,nu=.3,psi_deg=5.),
                'hidden':model.hidden,'lo':box.lo,'hi':box.hi,'state_dict':model.state_dict()},path)
    return NeuralCWFSMaterial(E=560.,nu=.3,psi_deg=5.,c_peak=1.,c_res=.2,
                             phi_peak_deg=20.,phi_res_deg=30.,gp_c=.01,gp_phi=.015,checkpoint=path)


def test_all_fixed_strength_regions():
    rng = np.random.default_rng(19)
    teacher = MohrCoulombEPP(560.,.3,.5,30.,5.)
    tr = np.sort(rng.uniform(-20,40,(1000,3)),axis=1)[:,::-1].copy()
    trial4 = np.column_stack((tr,np.zeros(len(tr))))
    target = teacher.update(trial4@np.linalg.inv(teacher.C).T,np.zeros_like(trial4),teacher.initial_state(len(tr)))
    ts = torch.tensor(tr)
    out,reg = fixed_strength_return(ts,torch.full((len(tr),),.5,dtype=ts.dtype),
                                    torch.full((len(tr),),np.pi/6,dtype=ts.dtype),
                                    torch.tensor(teacher.C[:3,:3]),5.)
    np.testing.assert_allclose(out.numpy(),target.sig[:,:3],atol=1e-11)
    assert set(reg.tolist())=={0,1,2,3,4}


def test_strengthening_edge_does_not_take_negative_multipliers():
    mat=CWFSMaterial(E=560.,nu=.3,psi_deg=5.,c_peak=1.,c_res=.2,
                     phi_peak_deg=20.,phi_res_deg=30.,gp_c=.01,gp_phi=.015)
    # The old edge solve crossed through negative multipliers and returned
    # sigma1=76.075 MPa and gamma=0.03438 for this slightly overstressed trial.
    trial=np.array([[68.1211586557331,31.98525081580371,31.98525081580371,0.]])
    out=mat.update(trial@np.linalg.inv(mat.C).T,np.zeros_like(trial),mat.initial_state(1))
    dp=(trial-out.sig)@np.linalg.inv(mat.C).T
    assert dp[0,0]>0 and dp[0,1]<0 and dp[0,2]<0
    assert out.sig[0,0]<trial[0,0]
    assert 0<out.state[0,0]<1e-4
    assert abs(mat.yield_function(out.sig,out.state[:,0])[0])<1e-9


@pytest.mark.parametrize('theta',[[1,.2,20,30,.01,.015],[8,1,25,35,.008,.012],
                                  [.5,.05,40,45,.002,.004]])
def test_updated_strength_reproduces_the_same_admissible_return(theta):
    rng=np.random.default_rng(25)
    mat=CWFSMaterial(E=560.,nu=.3,psi_deg=5.,**dict(zip(
        ('c_peak','c_res','phi_peak_deg','phi_res_deg','gp_c','gp_phi'),theta)))
    trial=np.sort(rng.uniform(-12,65,(1000,3)),axis=1)[:,::-1].copy()
    trial4=np.column_stack((trial,np.zeros(1000)))
    state=mat.initial_state(1000);state[:,0]=rng.uniform(0,max(theta[4:6]),1000)
    out=mat.update(trial4@np.linalg.inv(mat.C).T,np.zeros_like(trial4),state)
    tr,gt,th=torch.tensor(trial),torch.tensor(out.state[:,0]),torch.tensor(np.tile(theta,(1000,1)))
    from cwfs_inv.neural_material import strengths
    c,phi=strengths(gt,th)
    stress,_=fixed_strength_return(tr,c,phi,torch.tensor(mat.Cn),5.)
    np.testing.assert_allclose(stress.numpy(),out.sig[:,:3],rtol=1e-8,atol=1e-8)


def test_neural_elastic_gate_and_history(material):
    old = np.tile([10.,10.,10.,0.],(2,1))
    state = material.initial_state(2)
    state[:,0]=[0.,.005]
    state[:,2]=[-.001,-.002]
    de = np.tile([1e-5,-1e-5,0.,1e-5],(2,1))
    out = material.update(de,old,state)
    np.testing.assert_allclose(out.sig,old+de@material.C.T,atol=1e-12)
    np.testing.assert_allclose(out.state[:,[0,2]],state[:,[0,2]],atol=1e-12)
    np.testing.assert_allclose(out.tangent,np.tile(material.C,(2,1,1)),atol=1e-10)


def test_consistent_tangent_and_rotation(material):
    old = np.array([[8.,3.,4.,.8],[15.,2.,6.,1.]])
    de = np.array([[.004,-.001,0.,.0005],[.002,-.0007,0.,-.001]])
    state = material.initial_state(2)
    state[:,0]=[.001,.003]
    out = material.update(de,old,state)
    for k in range(4):
        h = 1e-7
        plus,minus=de.copy(),de.copy()
        plus[:,k]+=h;minus[:,k]-=h
        fd=(material.update(plus,old,state).sig-material.update(minus,old,state).sig)/(2*h)
        np.testing.assert_allclose(fd,out.tangent[:,:,k],rtol=3e-5,atol=2e-5)
    Ts,Te=_rotation_matrices(np.full(2,.47))
    # Ts rotates stress to xy; inverse Te rotates engineering strain to xy.
    rold=np.einsum('nij,nj->ni',Ts,old)
    rde=np.einsum('nij,nj->ni',np.linalg.inv(Te),de)
    rotated=material.update(rde,rold,state)
    np.testing.assert_allclose(rotated.sig,np.einsum('nij,nj->ni',Ts,out.sig),atol=1e-10)
    np.testing.assert_allclose(rotated.state[:,[0,2]],out.state[:,[0,2]],atol=1e-10)
    np.testing.assert_allclose(rotated.tangent,Ts@out.tangent@Te,rtol=1e-8,atol=1e-8)


def test_network_changes_accepted_stress_and_state(material):
    old=np.array([[9.,3.,4.,.3]])
    de=np.array([[.003,-.001,0.,.0005]])
    state=material.initial_state(1)
    before=material.update(de,old,state)
    with torch.no_grad():
        material.model.net[-1].bias.add_(.5)
    after=material.update(de,old,state)
    assert np.max(abs(after.sig-before.sig))>1e-8
    assert abs(after.state[0,0]-before.state[0,0])>1e-10
    assert after.info['n_neural_transition']==1


def test_repeated_principal_stress_has_objective_tangent(material):
    old=np.array([[12.,12.,3.,0.]])
    de=np.array([[.002,.002,0.,0.]])
    state=material.initial_state(1);state[:,0]=.002
    out=material.update(de,old,state)
    Ts,Te=_rotation_matrices(np.array([.73]))
    np.testing.assert_allclose(Ts@out.tangent@Te,out.tangent,rtol=1e-8,atol=1e-8)


def test_stress_plastic_history_and_volume_share_one_increment(material):
    old=np.array([[9.,3.,4.,.3]])
    de=np.array([[.003,-.001,0.,.0005]])
    state=material.initial_state(1)
    out=material.update(de,old,state)
    dp=de-(out.sig-old)@np.linalg.inv(material.C).T
    tensor=np.array([[dp[0,0],dp[0,3]/2,0],[dp[0,3]/2,dp[0,1],0],[0,0,dp[0,2]]])
    principal=np.linalg.eigvalsh(tensor)
    assert out.state[0,0]==pytest.approx(np.ptp(principal),abs=1e-12)
    assert out.state[0,2]==pytest.approx(np.trace(tensor),abs=1e-12)


def test_residual_strength_matches_full_cwfs(material):
    rng=np.random.default_rng(2026)
    n=100
    teacher=CWFSMaterial(E=560.,nu=.3,psi_deg=5.,c_peak=1.,c_res=.2,
                         phi_peak_deg=20.,phi_res_deg=30.,gp_c=.01,gp_phi=.015)
    de=rng.normal(0,.01,(n,4));de[:,2]=0
    old=np.tile([6.,9.,11.,0.],(n,1))
    state=material.initial_state(n);state[:,0]=.1
    actual=material.update(de,old,state)
    target=teacher.update(de,old,state)
    np.testing.assert_allclose(actual.sig,target.sig,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(actual.state[:,[0,2]],target.state[:,[0,2]],rtol=1e-6,atol=1e-8)


def test_fe_factory_and_material_mismatch(material):
    spec=dict(model='neural_cwfs',E=560.,nu=.3,psi_deg=5.,c_peak=1.,c_res=.2,
              phi_peak_deg=20.,phi_res_deg=30.,gp_c=.01,gp_phi=.015,checkpoint=material.checkpoint)
    assert make_material(spec).name=='neural_cwfs'
    spec['E']=15000
    with pytest.raises(ValueError,match='parameters differ'):
        make_material(spec)


def test_equilibrium_backtracking_resolves_strong_nonlinearity():
    """A valid Newton direction can require a step below the old 1/32 cutoff."""
    from types import SimpleNamespace
    from scipy.sparse import csc_matrix
    from cwfs_inv.horseshoe_fe import StagedModel
    solver=StagedModel.__new__(StagedModel)
    solver.n_dof=1
    solver.free=np.array([0])
    solver.cfg=SimpleNamespace(solver=dict(max_iter=25,modified_max_iter=0,max_stalls=0))
    solver._acceptance_budget=lambda _:dict(active_residual_tolerance=1e-10,
                                          newly_active_load_norm=0.,full_residual_tolerance=1e-10)
    solver._material_update=lambda u:(u+1e6*u*u,np.zeros(1),1+2e6*u,{})
    solver._residual=lambda sig,force:force-sig
    solver._assemble=lambda D:csc_matrix(D.reshape(1,1))
    assert not solver._solve_increment(np.ones(1),1.)[0]
    solver.cfg.solver['line_search_steps']=12
    converged,du,*_=solver._solve_increment(np.ones(1),1.)
    assert converged
    np.testing.assert_allclose(du+1e6*du*du,1.,atol=1e-10)
