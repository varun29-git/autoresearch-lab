import base64
import hashlib
import io
import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest
import torch

from production.amd.checkpoint_service import CheckpointService, marker_name, restore
from production.amd.checkpoint_store import LocalStore, S3Store
from terminus.checkpoint import load_checkpoint, rng_state, save_checkpoint
from terminus.integrity import fingerprint, write_json_atomic


def run_fixture(root):
    root.mkdir()
    contract = {'model':'test', 'world_size':1}
    write_json_atomic(root/'run.json', {'contract':contract,'contract_sha256':fingerprint(contract)})
    model = torch.nn.Linear(2, 2); optimizer = torch.optim.AdamW(model.parameters())
    def save(update):
        return save_checkpoint(root/'checkpoints', model, optimizer,
            {'updates':update,'tokens_seen':update*8,'next_micro_step':update}, contract, [rng_state()])
    return save, contract


def test_active_upload_retains_two_and_restores_deleted_checkpoint(tmp_path):
    run=tmp_path/'run'; save, contract=run_fixture(run)
    paths=[save(i) for i in range(1,5)]
    store=LocalStore(tmp_path/'durable')
    service=CheckpointService(run,store,keep_last=2,min_free_bytes=0)
    service.scan(); service.close()
    assert not paths[0].exists() and not paths[1].exists()
    assert paths[2].exists() and paths[3].exists()
    assert load_checkpoint(run/'checkpoints',contract)['state']['updates']==4
    restored=tmp_path/'restored'
    restore(store, paths[0].name, restored)
    assert load_checkpoint(restored/'checkpoints',contract)['state']['updates']==1


def test_interrupted_remote_marker_never_marks_local_durable_or_prunes(tmp_path):
    run=tmp_path/'run';save,_=run_fixture(run)
    paths=[save(i) for i in range(1,4)];store=LocalStore(tmp_path/'durable')
    original=store.put
    def fail_marker(source,name,digest):
        if name.endswith('.durable.json'): raise OSError('simulated connection loss')
        return original(source,name,digest)
    with patch.object(store,'put',side_effect=fail_marker):
        service=CheckpointService(run,store,keep_last=2,min_free_bytes=0)
        service.enqueue(paths[0])
        with pytest.raises(RuntimeError,match='backup failed'): service.close()
    assert all(p.exists() for p in paths)
    assert not (run/'checkpoints/.durable'/marker_name(paths[0].name)).exists()
    # A fresh process discovers and retries the incomplete upload idempotently.
    service=CheckpointService(run,store,keep_last=2,min_free_bytes=0)
    service.scan();service.close()
    assert not paths[0].exists()


def test_corrupted_durable_payload_prevents_retention(tmp_path):
    run=tmp_path/'run';save,_=run_fixture(run)
    paths=[save(i) for i in range(1,4)];store=LocalStore(tmp_path/'durable')
    service=CheckpointService(run,store,keep_last=3,min_free_bytes=0)
    service.scan();service.close()
    (store.root/paths[0].name).write_bytes(b'corrupt')
    service.keep_last=2
    with pytest.raises(RuntimeError,match='durable copy changed'): service.prune()
    assert paths[0].exists()


def test_byte_backpressure_does_not_drop_checkpoint(tmp_path):
    run=tmp_path/'run';save,_=run_fixture(run);path=save(1)
    service=CheckpointService(run,LocalStore(tmp_path/'durable'),max_pending_bytes=1,min_free_bytes=0)
    with pytest.raises(RuntimeError,match='byte limit'):service.enqueue(path)
    service.close();assert path.exists()


class Missing(Exception):
    response={'Error':{'Code':'404'}}


class FakeS3:
    def __init__(self):self.objects={};self.uploads={};self.aborted=[];self.fail_part=False
    def head_object(self,**kw):
        if kw['Key'] not in self.objects:raise Missing()
        o=self.objects[kw['Key']]
        return {'ContentLength':len(o['data']),'ChecksumSHA256':o['checksum'],'Metadata':o['meta']}
    def put_object(self,**kw):
        data=kw['Body'].read()
        assert base64.b64encode(hashlib.sha256(data).digest()).decode()==kw['ChecksumSHA256']
        self.objects[kw['Key']]={'data':data,'checksum':kw['ChecksumSHA256'],'meta':kw['Metadata']}
    def create_multipart_upload(self,**kw):
        self.uploads['u']={'key':kw['Key'],'meta':kw['Metadata'],'parts':[]};return {'UploadId':'u'}
    def upload_part(self,**kw):
        if self.fail_part:raise OSError('interrupted part')
        assert base64.b64encode(hashlib.sha256(kw['Body']).digest()).decode()==kw['ChecksumSHA256']
        self.uploads['u']['parts'].append(kw['Body'])
        return {'ETag':'opaque-not-a-sha256','ChecksumSHA256':kw['ChecksumSHA256']}
    def complete_multipart_upload(self,**kw):
        o=self.uploads['u'];parts=o['parts']
        composite=base64.b64encode(hashlib.sha256(b''.join(hashlib.sha256(p).digest() for p in parts)).digest()).decode()+f'-{len(parts)}'
        self.objects[o['key']]={'data':b''.join(parts),'checksum':composite,'meta':o['meta']}
    def abort_multipart_upload(self,**kw):self.aborted.append(kw['UploadId'])


def test_s3_multipart_verifies_composite_not_etag_and_aborts_interruption(tmp_path):
    source=tmp_path/'payload';source.write_bytes(bytes(range(100)))
    digest=hashlib.sha256(source.read_bytes()).hexdigest();client=FakeS3()
    store=S3Store('s3://bucket/run-id',client=client);store.threshold=20;store.part_bytes=16
    store.put(source,'step_00000001.pt',digest)
    assert store.verify('step_00000001.pt',digest,100)
    client.objects[store.key('step_00000001.pt')]['checksum']='bad'
    assert not store.verify('step_00000001.pt',digest,100)
    client.fail_part=True
    with pytest.raises(OSError):store.put(source,'step_00000002.pt',digest)
    assert client.aborted==['u']
    assert not store.exists('step_00000002.pt')
