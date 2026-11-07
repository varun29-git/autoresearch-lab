"""Finite, deterministic language/source/child mixing of packed sequences.

Allocation and physical writing order do not determine training order. This
CPU adapter preserves each sequence's descriptor and visits every sequence
exactly once. Native segment-aware trainer integration remains a separate gate.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from production.document_packing import decode_pack, file_sha

MASK64=(1<<64)-1


def seed64(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'little')


def mix64(value):
    value=(value^(value>>30))*0xbf58476d1ce4e5b9&MASK64
    value=(value^(value>>27))*0x94d049bb133111eb&MASK64
    return value^(value>>31)


def permute(index,size,key):
    """Cycle-walk a bijection; no random replacement or dropped tail rows."""
    if not 0<=index<size:raise IndexError('permutation index out of range')
    if size==1:return 0
    half=((size-1).bit_length()+1)//2;mask=(1<<half)-1;value=index
    while True:
        left,right=value>>half,value&mask
        for turn in range(6):
            left,right=right,left^(mix64(right^key^((turn+1)*0x9e3779b97f4a7c15&MASK64))&mask)
        value=(left<<half)|right
        if value<size:return value


class Node:
    def __init__(self,left=None,right=None,source=None,count=0,seed=''):
        self.left,self.right,self.source=left,right,source
        self.count=count if source is not None else left.count+right.count
        self.rotation=seed64(seed)%self.count

    def route(self,index):
        if self.source is not None:return self.source,index
        index=(index+self.rotation)%self.count
        before=index*self.left.count//self.count
        if (index+1)*self.left.count//self.count>before:return self.left.route(before)
        return self.right.route(index-before)


def join(nodes,seed):
    if len(nodes)==1:return nodes[0]
    middle=len(nodes)//2
    return Node(join(nodes[:middle],seed+'/L'),join(nodes[middle:],seed+'/R'),seed=seed)


class PackedMixture:
    """Hierarchical fair interleaving with bounded, shuffled read windows.

    Weights are exact finite sequence counts, without 0.001 rounding. Every
    allocated content token remains in its original sequence; actual content,
    padding and valid-target totals remain separate from the sampling slots.
    """
    def __init__(self,shards,*,seed,block_sequences=4096):
        if type(block_sequences)is not int or block_sequences<=0:raise ValueError('invalid mixing block')
        self.seed=seed;self.block_sequences=block_sequences;self.shards={};self.ends={}
        hierarchy=defaultdict(lambda:defaultdict(dict));seen=set();counts={}
        for shard in shards:
            if type(shard['sequences'])is not int or shard['sequences']<=0:raise ValueError('invalid packed sequence count')
            identity=shard['payload']['uri']
            if identity in seen:raise ValueError('same packed shard registered twice')
            seen.add(identity);source=(shard['leaf'],shard['stratum'])
            self.shards.setdefault(source,[]).append(shard);counts[source]=counts.get(source,0)+shard['sequences']
        if not counts:raise ValueError('empty packed mixture')
        for source,count in sorted(counts.items()):
            leaf,stratum=source;language=leaf.split('/')[0]
            hierarchy[language][leaf][stratum]=Node(source=source,count=count,seed=seed+'/'+leaf+'/'+stratum)
            items=self.shards[source]
            items.sort(key=lambda s:(hashlib.sha256((seed+'/'+s['payload']['sha256']).encode()).digest(),
                                     s['payload']['uri']))
            self.ends[source]=list(np.cumsum([s['sequences']for s in items],dtype=np.int64))
        languages=[]
        for language,leaves in sorted(hierarchy.items()):
            sources=[join(list(children.values()),seed+'/'+leaf)for leaf,children in sorted(leaves.items())]
            languages.append(join(sources,seed+'/'+language))
        self.tree=join(languages,seed+'/languages');self.sequence_counts=counts
        self.total_sequences=self.tree.count

    def locate(self,global_index):
        if not 0<=global_index<self.total_sequences:raise IndexError('global packed index out of range')
        block,offset=divmod(global_index,self.block_sequences)
        size=min(self.block_sequences,self.total_sequences-block*self.block_sequences)
        shuffled=block*self.block_sequences+permute(offset,size,seed64(self.seed+'/block/'+str(block)))
        source,ordinal=self.tree.route(shuffled)
        number=bisect_right(self.ends[source],ordinal)
        start=0 if number==0 else self.ends[source][number-1]
        shard=self.shards[source][number]
        row=permute(int(ordinal-start),shard['sequences'],seed64(self.seed+'/'+shard['payload']['sha256']))
        return shard,row


class PackedSequenceReader:
    """CPU descriptor-aware reader; caller supplies verified local cache paths."""
    def __init__(self,mixture,*,resolve_path,max_open_shards=128):
        self.mixture=mixture;self.resolve_path=resolve_path;self.max_open=max_open_shards
        self.open=OrderedDict();self.verified=set()

    def _load(self,shard):
        identity=shard['payload']['uri']
        if identity in self.open:
            self.open.move_to_end(identity);return self.open[identity]
        paths={}
        for name in ['payload','index','index_offsets']:
            spec=shard[name];path=Path(self.resolve_path(spec['uri']))
            if path.stat().st_size!=spec['bytes']:raise ValueError('packed cache size differs')
            binding=(str(path),spec['sha256'],path.stat().st_mtime_ns)
            if binding not in self.verified:
                if file_sha(path)!=spec['sha256']:raise ValueError('packed cache checksum differs')
                self.verified.add(binding)
            paths[name]=path
        offsets=np.memmap(paths['index_offsets'],mode='r',dtype='<u8')
        if (len(offsets)!=shard['sequences']+1 or offsets[0]!=0
                or offsets[-1]!=shard['index']['bytes'] or np.any(offsets[1:]<=offsets[:-1])):
            raise ValueError('descriptor offset table differs')
        length=shard['sequence_length']
        if shard['payload']['bytes']!=shard['sequences']*length*2:raise ValueError('packed geometry differs')
        payload=np.memmap(paths['payload'],mode='r',dtype='<u2').reshape(-1,length)
        metadata=np.memmap(paths['index'],mode='r',dtype=np.uint8)
        value=(payload,metadata,offsets);self.open[identity]=value
        while len(self.open)>self.max_open:self.open.popitem(last=False)
        return value

    def sample(self,global_index):
        shard,row=self.mixture.locate(global_index);payload,metadata,offsets=self._load(shard)
        descriptor=json.loads(metadata[int(offsets[row]):int(offsets[row+1])].tobytes())
        inputs=payload[row].copy();targets,positions,segments=decode_pack(inputs,descriptor)
        return {'inputs':inputs,'targets':targets,'positions':positions,'segment_ids':segments,
            'content_tokens':descriptor['content_tokens'],'valid_targets':descriptor['valid_targets'],
            'leaf':shard['leaf'],'stratum':shard['stratum'],
            'payload_uri':shard['payload']['uri'],'sequence_row':row}


def mixture_candidate(report,indexed_reports=()):
    """Finalize read-order metadata only from an exact completed physical phase."""
    if not report.get('physical_creation_finished') or not report.get('all_curriculum_quotas_satisfied'):
        raise ValueError('complete physical allocation required for a phase mixture')
    lookup={}
    for indexed in indexed_reports:
        if indexed.get('status')!='complete':raise ValueError('incomplete descriptor indexing')
        for shard in indexed['shards']:
            lookup[(shard['payload']['uri'],shard['index']['sha256'])]=shard
    shards=[];population={}
    for original in report['shards']:
        shard=original.copy()
        if 'index_offsets'not in shard:
            cached=lookup.get((shard['payload']['uri'],shard['index']['sha256']))
            if not cached:raise ValueError('packed shard needs a verified descriptor-offset index')
            for field in ['sequences','leaf','stratum','payload','index','plan_sha256']:
                if cached[field]!=original[field]:raise ValueError('indexed shard binding differs')
            shard['index_offsets']=cached['index_offsets']
            shard['index_offsets_bind_to_sha256']=cached['index_offsets_bind_to_sha256']
        if (shard['plan_sha256']!=report['plan_sha256']
                or shard['index_offsets_bind_to_sha256']!=shard['index']['sha256']
                or shard['index_offsets']['bytes']!=(shard['sequences']+1)*8):
            raise ValueError('phase or descriptor offsets differ')
        key=shard['leaf']+'::'+shard['stratum']
        row=population.setdefault(key,{'leaf':shard['leaf'],'stratum':shard['stratum'],
            'sequences':0,'content_tokens':0,'valid_targets':0,'padding_slots':0})
        for field in ['sequences','content_tokens','valid_targets','padding_slots']:row[field]+=shard[field]
        shards.append(shard)
    if sum(s['content_tokens']for s in shards)!=report['totals']['content_tokens']:
        raise ValueError('packed mixture content budget differs')
    seed='ci1-packed-mixture-v1/'+report['plan_sha256']
    router=PackedMixture(shards,seed=seed)
    return {'schema':'ci1_packed_mixture_candidate_v1','plan_sha256':report['plan_sha256'],
        'seed':seed,'block_sequences':4096,'hierarchy':['language','source_leaf','dataset_stratum'],
        'total_sequences':router.total_sequences,'content_tokens':report['totals']['content_tokens'],
        'population':population,'shards':shards,'repeats_sequences':False,
        'native_training_accepted':False,'requires_segment_aware_reader':True,
        'legacy_continuous_stream_loader_compatible':False}
