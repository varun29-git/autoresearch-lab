"""Describe or explicitly measure gfx942 GEMM tiles for the real model shapes.

Default is CPU-only shape preparation. --run is a human-invoked native test,
not a paid launch performed by this tool's implementation or by documentation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

from production.artifacts import external_path
from terminus.integrity import fingerprint, write_json_atomic
from terminus.train import load_config
from .tuning import native_identity, validate_tile

TILES = [
    {'bm':32,'bn':64,'num_warps':4,'num_stages':2,'waves_per_eu':0},
    {'bm':32,'bn':128,'num_warps':4,'num_stages':2,'waves_per_eu':0},
    {'bm':64,'bn':64,'num_warps':4,'num_stages':2,'waves_per_eu':0},
    {'bm':64,'bn':128,'num_warps':8,'num_stages':2,'waves_per_eu':0},
    {'bm':32,'bn':64,'num_warps':4,'num_stages':1,'waves_per_eu':1},
    {'bm':32,'bn':64,'num_warps':4,'num_stages':3,'waves_per_eu':2},
]


def shapes(model, microbatch):
    if type(microbatch) is not int or microbatch<=0:raise ValueError('microbatch must be positive')
    m=microbatch*model.max_seq_len;d=model.dim;kvd=model.n_kv_heads*model.head_dim;f=model.ffn_dim
    out=[]
    for name,n,k in [('qkv',d+2*kvd,d),('o',d,d),('gate_up',2*f,d),('down',d,f)]:
        for role,a,b,c,group in [('forward',m,n,k,128),('dgrad',m,k,n,128),('wgrad',n,k,m,1)]:
            out.append({'projection':name,'role':role,'m':a,'n':b,'k':c,'group':group,
                        'dtype':'float32' if group==1 else 'bfloat16','key':f'{a},{b},{c},{group}'})
    return out


def measure(model,microbatch,*,iterations=10):
    import torch
    from unittest.mock import patch
    from . import fp8_kernels, reference
    from .preflight import _errors
    if type(iterations) is not int or iterations<3:raise ValueError('need at least three timing samples')
    identity=native_identity();tiles={};measurements={};failures=[]
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        torch.manual_seed(1337)
        for shape in shapes(model,microbatch):
            key=shape['key']
            if key in tiles:continue
            m,n,k,group=(shape[x] for x in ('m','n','k','group'))
            a=reference.quantize(torch.randn(m,k,device='cuda')*.02)
            b=reference.quantize(torch.randn(n,k,device='cuda')*.02,group_rows=group)
            a=(a[0],a[1].T.contiguous().T)
            dtype=torch.float32 if group==1 else torch.bfloat16
            expected=reference.gemm(a,b,rhs_group_size=group,out_dtype=dtype)
            winners=[]
            for tile in TILES:
                validate_tile(tile)
                try:
                    with patch('production.amd.tuning.tile_for',return_value=tile):
                        def call():
                            return fp8_kernels.gemm(a,b,rhs_group_size=group,out_dtype=dtype,
                                splits=(model.fp8_wgrad_splits or 1) if group==1 else 1)
                        error=_errors([call()],[expected],limit=.025,atol=0.)
                        for _ in range(3):call()
                        times=[]
                        for _ in range(iterations):
                            torch.cuda.synchronize();begin=time.perf_counter();call();torch.cuda.synchronize()
                            times.append(time.perf_counter()-begin)
                        winners.append({'tile':tile,'correctness_passed':True,'errors':error,
                                        'median_seconds':statistics.median(times),'samples_seconds':times,
                                        'splits':min(model.fp8_wgrad_splits or 1,k//128) if group==1 else 1})
                except (RuntimeError,AssertionError) as exc:
                    failures.append({'key':key,'tile':tile,'error':str(exc)})
                    if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
            if not winners:raise RuntimeError(f'no correct GEMM tile for {key}')
            winner=min(winners,key=lambda x:x['median_seconds'])
            tiles[key]=winner['tile'];measurements[key]=winner
            del a,b,expected
        result={'schema':2,'arch':'gfx942','identity':identity,'tiles':tiles,'measurements':measurements,
                'failed_candidates':failures,'scope':'synthetic GEMM only; validate full updates before promotion'}
        result['sha256']=fingerprint(result)
        return result
    finally:torch.backends.cuda.matmul.allow_tf32=old_tf32


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--microbatch',type=int,default=16);p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--run',action='store_true',help='Explicit native measurement on already available hardware')
    args=p.parse_args();model,_,_=load_config(external_path(args.config))
    output=external_path(args.output,fresh=True)
    result=measure(model,args.microbatch,iterations=args.iterations) if args.run else {
        'status':'prepared_not_measured','shapes':shapes(model,args.microbatch),'candidate_tiles':TILES,
        'model':asdict(model),'scope':'CPU shape plan only; no tuning table selected'}
    write_json_atomic(output,result)
    print(json.dumps({'output':str(output),'native_measured':args.run}))


if __name__=='__main__':main()
