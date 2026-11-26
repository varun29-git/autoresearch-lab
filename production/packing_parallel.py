"""Bounded CPU preparation and encoded packing chunks for the fast worker."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import uuid

import numpy as np

from production.compact_ledger import key, row_key_hash
from production.document_packing import ShardWriter, best_fit
from production.document_packing import fragments
from production.packing_worker import Exclusions, fingerprint

EXCLUSIONS=None


@dataclass
class FragmentReference:
    path: str
    start: int
    count: int
    document: dict
    offset: int
    document_end: bool
    number: int


def expand_references(items,length):
    """Read verified local inputs in packing children, avoiding token IPC."""
    maps={};result=[];identity=None;iterator=None;number=-1
    try:
        for item in items:
            if not isinstance(item,FragmentReference):
                result.append(item);continue
            current=(item.path,item.start,item.count,id(item.document),item.offset,item.document_end)
            if current!=identity:
                if item.path not in maps:maps[item.path]=np.memmap(item.path,dtype='<u2',mode='r')
                values=maps[item.path]
                if item.start<0 or item.count<=0 or item.start+item.count>len(values):
                    raise ValueError('referenced fragment exceeds verified source payload')
                iterator=iter(fragments(values[item.start:item.start+item.count],item.document,
                    length=length,offset=item.offset,document_end=item.document_end))
                identity=current;number=-1
            if item.number<=number:raise ValueError('referenced fragments are out of order')
            try:
                while number<item.number:fragment=next(iterator);number+=1
            except StopIteration as exc:raise ValueError('referenced fragment exceeds selected span')from exc
            result.append(fragment)
        return result
    finally:
        # Fragment payloads are owned copies; input maps need not survive packing.
        iterator=None
        for value in maps.values():value._mmap.close()


def initialize(exclusions_path, exclusions_sha256):
    global EXCLUSIONS
    EXCLUSIONS=Exclusions(exclusions_path,exclusions_sha256)


def loads(raw):
    try:
        import orjson
        return orjson.loads(raw)
    except ImportError:
        import json
        return json.loads(raw)


def prepare_index(path, seed):
    rows=[];cursor=0;documents=0;heldout=0
    with Path(path).open('rb',buffering=8<<20)as stream:
        for line in stream:
            row=loads(line);documents+=1
            if (type(row['n_tokens'])is not int or row['n_tokens']<=0
                    or row['token_start']!=cursor or row['validation']is not False):
                raise ValueError('invalid or noncontiguous training index')
            cursor+=row['n_tokens']
            if EXCLUSIONS.excludes(row):heldout+=1;continue
            row['_sha']=key(row['source_text_sha256'])
            row['_row_sha']=row_key_hash(row['row_key'])
            row['_rank']=fingerprint([seed,row['row_key']])
            rows.append(row)
    rows.sort(key=lambda row:row['_rank'])
    return {'rows':rows,'tokens':cursor,'documents':documents,'heldout':heldout}


def prepare_records(raw, source_scope=None, skipped_shas=None):
    rows=[]
    for line in raw.splitlines():
        row=loads(line)
        if source_scope is not None and row['leaf']not in source_scope:continue
        if skipped_shas and row['source_text_sha256']in skipped_shas:continue
        row['_sha']=key(row['source_text_sha256'])
        row['_row_sha']=row_key_hash(row['row_key'])
        if EXCLUSIONS.excludes(row|{'validation':False}):
            raise ValueError('protected record overlaps frozen heldout')
        rows.append(row)
    return rows


def pack_chunk(items, length, seed, directory):
    if any(isinstance(item,FragmentReference)for item in items):
        items=expand_references(items,length)
    packs=best_fit(items,length=length,seed=seed)
    writer=ShardWriter(directory,'chunk-'+uuid.uuid4().hex,length=length,
        buffer_bytes=64<<20,fast_json=True)
    names=['content_tokens','valid_targets','used_slots','padding_slots','synthetic_tokens']
    prefixes=np.zeros((len(packs)+1,len(names)),dtype=np.int64)
    offsets=np.zeros(len(packs)+1,dtype=np.int64)
    source_prefixes={}
    for i,pack in enumerate(packs):
        writer.add(pack)
        prefixes[i+1]=[writer.counts[n]for n in names]
        offsets[i+1]=writer.meta_bytes
        for key,n in writer.source_counts.items():
            if key not in source_prefixes:source_prefixes[key]=np.zeros(len(packs)+1,dtype=np.int64)
            source_prefixes[key][i+1]=n
    result=writer.close()
    result.update(prefixes=prefixes,offsets=offsets,names=names,source_prefixes=source_prefixes)
    return result


def download_verified(client, spec, path, local_cache=None):
    """Hash while receiving large blocks; avoid a second complete file read."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if local_cache and spec['uri']in local_cache:
        cached=local_cache[spec['uri']];source=Path(cached['path'])
        from production.document_packing import file_sha
        if (cached['sha256']!=spec['sha256']or cached['bytes']!=spec['bytes']
                or source.stat().st_size!=spec['bytes']or file_sha(source)!=spec['sha256']):
            raise ValueError('shared immutable input differs from its pinned identity')
        os.link(source,path)
        return path
    bucket,name=spec['uri'][5:].split('/',1)
    body=client.get_object(Bucket=bucket,Key=name)['Body']
    digest=hashlib.sha256();size=0
    try:
        with path.open('wb',buffering=64<<20)as stream:
            while chunk:=body.read(8<<20):
                digest.update(chunk);size+=len(chunk);stream.write(chunk)
                if size>spec['bytes']:raise ValueError('download exceeds pinned size')
    finally:
        body.close()
    if size!=spec['bytes']or digest.hexdigest()!=spec['sha256']:
        path.unlink(missing_ok=True)
        raise ValueError('source payload/index identity differs: '+spec['uri'])
    return path
