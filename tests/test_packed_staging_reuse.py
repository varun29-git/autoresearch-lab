import copy
from dataclasses import replace

import pytest

from production.amd.staging import create, rebind_microbatch, verify
from terminus.integrity import validate_manifest
from tests.test_document_training import packed_fixture


def readonly_fixture(root):
    model, manifest = packed_fixture(root)
    for path in manifest.provenance['packed_stream']['paths'].values():
        (root/path).chmod(0o444)
    return model, manifest


def test_reused_microbatch_receipt_equals_full_independent_acceptance(tmp_path):
    model, manifest = readonly_fixture(tmp_path)
    receipt = create(manifest, tmp_path, model, development_mode=True)
    derived = replace(manifest, config=replace(manifest.config, batch_size=2))
    rebound = rebind_microbatch(receipt, manifest, derived, tmp_path, model, development_mode=True)
    assert verify(rebound, derived, tmp_path, model, development_mode=True) == validate_manifest(
        derived, tmp_path, model, development_mode=True)
    assert rebound['validation_reused_from_receipt_sha256'] == receipt['receipt_sha256']
    assert receipt['manifest_sha256'] != rebound['manifest_sha256']


@pytest.mark.parametrize('change', ['seed', 'targets', 'horizon'])
def test_receipt_reuse_cannot_admit_changed_data_or_curriculum(tmp_path, change):
    model, manifest = readonly_fixture(tmp_path)
    receipt = create(manifest, tmp_path, model, development_mode=True)
    derived = copy.deepcopy(manifest)
    if change == 'seed':
        derived.provenance['packed_stream']['train'][0]['seed'] += '-changed'
    elif change == 'targets':
        derived.provenance['packed_stream']['train'][0]['shards'][0]['valid_targets'] += 1
    else:
        derived = replace(derived, config=replace(derived.config, total_train_tokens=48))
    with pytest.raises(ValueError, match='only a packed microbatch'):
        rebind_microbatch(receipt, manifest, derived, tmp_path, model, development_mode=True)


def test_receipt_reuse_rejects_changed_local_snapshot(tmp_path):
    model, manifest = readonly_fixture(tmp_path)
    receipt = create(manifest, tmp_path, model, development_mode=True)
    derived = replace(manifest, config=replace(manifest.config, batch_size=2))
    (tmp_path/manifest.shards[0].path).touch()
    with pytest.raises(ValueError, match='stale or files changed'):
        rebind_microbatch(receipt, manifest, derived, tmp_path, model, development_mode=True)
