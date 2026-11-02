import copy

import pytest

from production.amd.autotune import TILES, shapes
from production.amd.recipe import model_for
from production.amd.tuning import validate_measured_table, validate_tile
from terminus.integrity import fingerprint


def test_shape_plan_covers_actual_forward_dgrad_and_long_fp32_wgrad():
    cases=shapes(model_for(),16)
    assert len(cases)==12
    qkv={r['role']:r for r in cases if r['projection']=='qkv'}
    assert [qkv['forward'][k] for k in ('m','n','k')]==[65536,3840,2304]
    assert [qkv['wgrad'][k] for k in ('m','n','k','group')]==[3840,2304,65536,1]
    assert qkv['wgrad']['dtype']=='float32'
    for tile in TILES:validate_tile(tile)


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
