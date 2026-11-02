import json

from production.amd.checkpoint_service import marker_name


def test_training_operations_survive_exact_cpu_resume_with_async_snapshots(tmp_path):
    from production.amd.allocation import create
    from production.amd.operations import OperationsConfig, TrainingOperations
    from terminus.train import prepare_training
    from tests.helpers import fixture
    import time
    data=tmp_path/'data';fixture(data,examples=12,dropout=.1)
    cfg=json.loads((data/'config.json').read_text());cfg['training']['async_checkpoint']=True
    (data/'config.json').write_text(json.dumps(cfg))
    allocation=tmp_path/'allocation.json'
    create(allocation,allocation_id='cpu-test',started_unix=time.time()-1,gpu_hours=1,gpus=1)
    ops_config=OperationsConfig(str(allocation),str(tmp_path/'durable'),min_free_bytes=0,health_interval_updates=1)
    run=tmp_path/'run'
    def execute(resume=None,stop=None):
        with prepare_training(data/'config.json',data/'manifest.json',data,run,resume=resume) as w:
            w.stop_after_updates=stop;ops=TrainingOperations(w,ops_config)
            while w.state['next_micro_step']<w.total_micro_steps:
                ops.poll();w.complete_update()
                if w.cadence():break
            ops.close()
    execute(stop=1);execute(resume=run/'checkpoints')
    receipt=json.loads((run/'checkpoints/latest.json').read_text())
    assert receipt['state']['tokens_seen']==96
    assert (tmp_path/'durable'/marker_name(receipt['file'])).exists()
