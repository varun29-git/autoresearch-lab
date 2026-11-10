import copy

import pytest

from production.amd.autotune import TILES, plan, shapes, split_candidates, unmeasured
from production.amd.recipe import model_for
from production.amd.tuning import resolve_splits, validate_measured_table, validate_tile
from terminus.integrity import fingerprint


def test_shape_plan_covers_actual_forward_dgrad_and_long_fp32_wgrad():
    cases=shapes(model_for(),16)
    assert len(cases)==12
    qkv={r['role']:r for r in cases if r['projection']=='qkv'}
    assert [qkv['forward'][k] for k in ('m','n','k')]==[65536,3840,2304]
    assert [qkv['wgrad'][k] for k in ('m','n','k','group')]==[3840,2304,65536,1]
    assert qkv['wgrad']['dtype']=='float32'
    for tile in TILES:validate_tile(tile)


def test_split_plan_lists_wgrad_candidates_and_selects_none():
    rows=plan(model_for(),16)
    assert len(rows)==12
    assert unmeasured(model_for(),16)['status']=='prepared_not_measured'
    assert 'tiles' not in unmeasured(model_for(),16)
    for row in rows:
        expected=(1,2,4,8) if row['role']=='wgrad' else (1,)
        assert row['split_candidates']==expected
    wide=plan(model_for(),32)
    qkv={(r['projection'],r['role']):r for r in wide}
    assert qkv[('qkv','forward')]['m']==131072 and qkv[('qkv','wgrad')]['k']==131072
    assert qkv[('down','wgrad')]['m']==2304 and qkv[('down','wgrad')]['n']==5760
    assert qkv[('qkv','wgrad')]['split_candidates']==(1,2,4,8)
    assert split_candidates({'group':1,'k':128})==(1,)
    assert split_candidates({'group':1,'k':256})==(1,2)
    assert split_candidates({'group':1,'k':512})==(1,2,4)
    assert split_candidates({'group':128,'k':65536})==(1,)
    assert split_candidates({'group':1,'k':384})==(1,2)
    with pytest.raises(ValueError,match='multiple of 128'):
        split_candidates({'group':1,'k':100})


def test_loaded_table_replaces_one_global_split_and_untuned_math_stays():
    assert resolve_splits(4,None,group=1,k=65536)==4
    assert resolve_splits(16,None,group=1,k=256)==2
    assert resolve_splits(4,8,group=1,k=65536)==8
    assert resolve_splits(4,1,group=1,k=65536)==1
    with pytest.raises(ValueError,match='forward or dgrad'):
        resolve_splits(1,4,group=128,k=65536)
    with pytest.raises(ValueError,match='fit'):
        resolve_splits(1,8,group=1,k=256)
    with pytest.raises(ValueError,match='1, 2, 4, 8 or 16'):
        resolve_splits(3,None,group=1,k=65536)
    with pytest.raises(ValueError,match='1, 2, 4, 8 or 16'):
        resolve_splits(4,3,group=1,k=65536)


def test_measured_table_binds_correctness_tile_and_runtime():
    identity={'test':'cpu-contract-only'};key='128,128,128,1';tile=TILES[0]
    payload={'schema':2,'arch':'gfx942','identity':identity,'tiles':{key:tile},
             'measurements':{key:{'tile':tile,'correctness_passed':True,'splits':1}}}
    payload['sha256']=fingerprint(payload)
    validate_measured_table(payload,identity=identity)
    with pytest.raises(ValueError,match='runtime'):validate_measured_table(payload,identity={'changed':True})
    bad=copy.deepcopy(payload);bad['measurements'][key]['correctness_passed']=False
    bad['sha256']=fingerprint({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='correctness'):validate_measured_table(bad,identity=identity)
    bad=copy.deepcopy(payload);bad['measurements'][key]['splits']=3
    bad['sha256']=fingerprint({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='split-K'):validate_measured_table(bad,identity=identity)
    forward='128,128,128,128'
    bad=copy.deepcopy(payload)
    bad['tiles'][forward]=tile
    bad['measurements'][forward]={'tile':tile,'correctness_passed':True,'splits':4}
    bad['sha256']=fingerprint({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='forward or dgrad'):validate_measured_table(bad,identity=identity)
